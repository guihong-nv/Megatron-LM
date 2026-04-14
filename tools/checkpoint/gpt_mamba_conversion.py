# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""
GPT <-> Mamba Checkpoint Conversion Tool
=========================================

Directly converts checkpoints between GPTModel (homogeneous Transformer) and
MambaModel (hybrid Mamba+Transformer) without going through HuggingFace as an
intermediary.

Supported directions:
    gpt-to-mamba : Convert a GPT checkpoint to Mamba hybrid format.
    mamba-to-gpt : Convert a Mamba hybrid checkpoint to GPT format.

How the hybrid layer pattern maps GPT layers (gpt-to-mamba):
    - Each GPT layer contains both attention and MLP sub-layers.
    - The target Mamba model's hybrid_layer_pattern specifies per-layer types:
        M = Mamba SSM layer
        * = Attention-only layer
        - = MLP-only layer
        G = GDN layer
        E = MoE layer
    - GPT layer i's attention params map to the i-th '*' layer in the pattern.
    - GPT layer i's MLP params map to the i-th '-' layer in the pattern.
    - The number of '*' and '-' layers in the pattern must both equal the number
      of GPT layers.
    - Mamba SSM ('M') layers have no GPT equivalent and are initialized from
      scratch using standard Mamba initialization.

What happens to SSM parameters:
    gpt-to-mamba: SSM layers (M) are initialized from scratch:
        - A_log:          log(uniform(1, 16))
        - dt_bias:        inverse_softplus(log_uniform(dt_min, dt_max))
        - D:              ones
        - conv1d.weight:  kaiming_uniform(a=sqrt(5))
        - conv1d.bias:    zeros
        - in_proj.weight: kaiming_uniform(a=sqrt(5))
        - in_proj.layer_norm_weight: ones
        - out_proj.weight: kaiming_uniform(a=sqrt(5))
        - norm.weight:    ones
    mamba-to-gpt: SSM layers are discarded with a warning.

Example commands:
    # GPT -> Mamba
    python tools/checkpoint/gpt_mamba_conversion.py \\
        --direction gpt-to-mamba \\
        --load-dir /path/to/gpt-checkpoint \\
        --save-dir /path/to/mamba-checkpoint \\
        --hybrid-layer-pattern "M*-M*-M*-M*-" \\
        --target-tp-size 1 \\
        --target-pp-size 1 \\
        --d-model 4096 \\
        --mamba-d-state 128 \\
        --mamba2-n-groups 8 \\
        --mamba2-head-dim 64

    # Mamba -> GPT
    python tools/checkpoint/gpt_mamba_conversion.py \\
        --direction mamba-to-gpt \\
        --load-dir /path/to/mamba-checkpoint \\
        --save-dir /path/to/gpt-checkpoint \\
        --hybrid-layer-pattern "M*-M*-M*-M*-" \\
        --target-tp-size 1 \\
        --target-pp-size 1 \\
        --d-model 4096 \\
        --mamba-d-state 128 \\
        --mamba2-n-groups 8 \\
        --mamba2-head-dim 64
"""

import argparse
import copy
import math
import os
import re
from collections import OrderedDict

import torch


# ---------------------------------------------------------------------------
# TP split-dim mapping (reused from hybrid_conversion.py)
# ---------------------------------------------------------------------------

# Maps parameter-name substrings to the tensor dimension along which they are
# sharded across TP ranks.  -1 means "replicated" (not sharded).
TP_SPLIT_DIM = {
    # embeddings / output
    'word_embeddings.weight': 0,
    'output_layer.weight': 0,
    # norms (replicated)
    'norm.weight': -1,
    'final_norm.weight': -1,
    'final_layernorm.weight': -1,
    'final_layernorm.bias': -1,
    # mamba SSM params
    'A_log': 0,
    'D': 0,
    'dt_bias': 0,
    'in_proj.weight': 0,
    'conv1d.weight': 0,
    'conv1d.bias': 0,
    'x_proj.weight': 1,
    'dt_proj.weight': 0,
    'dt_proj.bias': 0,
    'out_proj.weight': 1,
    'mixer.norm.weight': 0,
    # MLP (transformer-style)
    'linear_fc1.layer_norm_weight': -1,
    'linear_fc1.weight': 0,
    'linear_fc2.weight': 1,
    # attention (transformer-style)
    'self_attention.linear_proj.weight': 1,
    'self_attention.linear_qkv.layer_norm_weight': -1,
    'self_attention.linear_qkv.weight': 0,
    # standalone layer norms (used in non-TE / "local" transformer impl)
    'input_layernorm.weight': -1,
    'input_layernorm.bias': -1,
    'pre_mlp_layernorm.weight': -1,
    'pre_mlp_layernorm.bias': -1,
    # TE-fused layer norms in Mamba in_proj
    'in_proj.layer_norm_weight': -1,
    'in_proj.layer_norm_bias': -1,
}


def get_split_dim(tensor_name):
    """Determine the TP-split dimension for a given parameter name."""
    # Disambiguate mixer.norm.weight vs generic norm.weight
    if 'norm.weight' in tensor_name:
        if 'mixer.norm.weight' in tensor_name:
            return TP_SPLIT_DIM['mixer.norm.weight']
        elif 'final_norm.weight' in tensor_name:
            return TP_SPLIT_DIM['final_norm.weight']
        elif 'final_layernorm.weight' in tensor_name:
            return TP_SPLIT_DIM['final_layernorm.weight']
        elif 'layer_norm_weight' in tensor_name:
            # TE-fused layer norm weights
            for key in TP_SPLIT_DIM:
                if key in tensor_name:
                    return TP_SPLIT_DIM[key]
            return -1
        else:
            return TP_SPLIT_DIM['norm.weight']

    for key in TP_SPLIT_DIM:
        if key in tensor_name:
            return TP_SPLIT_DIM[key]
    raise ValueError(f"Unknown tensor name for TP splitting: {tensor_name}")


# ---------------------------------------------------------------------------
# TP combine / split  (reused from hybrid_conversion.py)
# ---------------------------------------------------------------------------

def combine_tp_tensors(params, key, dim, tensors):
    """Combine TP-sharded tensors back into one full tensor.

    Handles special Mamba v2 in_proj and conv1d interleaved layouts.
    """
    tp_size = len(tensors)

    if 'mixer.in_proj.weight' in key and params.mamba_version == 1:
        xs, zs = [], []
        for tensor in tensors:
            x, z = torch.split(
                tensor,
                [params.mamba_d_inner // tp_size, params.mamba_d_inner // tp_size],
                dim=dim,
            )
            xs.append(x)
            zs.append(z)
        return torch.cat([torch.cat(xs, dim=dim), torch.cat(zs, dim=dim)], dim=dim)

    elif 'mixer.in_proj.weight' in key and params.mamba_version == 2:
        xs, zs, Bs, Cs, dts = [], [], [], [], []
        for tensor in tensors:
            x, z, B, C, dt = torch.split(
                tensor,
                [
                    params.mamba_d_inner // tp_size,
                    params.mamba_d_inner // tp_size,
                    (params.mamba2_n_groups // tp_size) * params.mamba_d_state,
                    (params.mamba2_n_groups // tp_size) * params.mamba_d_state,
                    params.mamba2_n_heads // tp_size,
                ],
                dim=dim,
            )
            xs.append(x)
            zs.append(z)
            Bs.append(B)
            Cs.append(C)
            dts.append(dt)

        for ii in range(len(Bs)):
            Bs[ii] = Bs[ii].reshape(-1, params.mamba_d_state, Bs[ii].shape[-1])
            Cs[ii] = Cs[ii].reshape(-1, params.mamba_d_state, Cs[ii].shape[-1])
        B = torch.cat(Bs, dim=dim)
        C = torch.cat(Cs, dim=dim)
        x = torch.cat(xs, dim=dim)
        z = torch.cat(zs, dim=dim)
        dt = torch.cat(dts, dim=dim)
        return torch.cat([x, z, B.flatten(0, 1), C.flatten(0, 1), dt], dim=dim)

    elif 'mixer.conv1d' in key and params.mamba_version == 2:
        xs, Bs, Cs = [], [], []
        for tensor in tensors:
            x, B, C = torch.split(
                tensor,
                [
                    params.mamba_d_inner // tp_size,
                    (params.mamba2_n_groups // tp_size) * params.mamba_d_state,
                    (params.mamba2_n_groups // tp_size) * params.mamba_d_state,
                ],
                dim=dim,
            )
            xs.append(x)
            Bs.append(B)
            Cs.append(C)

        for ii in range(len(Bs)):
            if 'weight' in key:
                Bs[ii] = Bs[ii].reshape(-1, params.mamba_d_state, Bs[ii].shape[-2], Bs[ii].shape[-1])
                Cs[ii] = Cs[ii].reshape(-1, params.mamba_d_state, Cs[ii].shape[-2], Cs[ii].shape[-1])
            elif 'bias' in key:
                Bs[ii] = Bs[ii].reshape(-1, params.mamba_d_state)
                Cs[ii] = Cs[ii].reshape(-1, params.mamba_d_state)
            else:
                raise ValueError(f"Unknown conv1d key: {key}")
        B = torch.cat(Bs, dim=dim)
        C = torch.cat(Cs, dim=dim)
        x = torch.cat(xs, dim=dim)
        return torch.cat([x, B.flatten(0, 1), C.flatten(0, 1)], dim=dim)

    else:
        return torch.cat(tensors, dim=dim)


def split_tensor_for_tp(params, key, dim, tensor):
    """Split a full tensor into TP shards.

    Handles special Mamba v2 in_proj and conv1d interleaved layouts.
    """
    tp_size = params.target_tp_size

    if 'mixer.in_proj.weight' in key and params.mamba_version == 1:
        x, z = torch.split(
            tensor, [params.mamba_d_inner, params.mamba_d_inner], dim=dim
        )
        x_sliced = torch.chunk(x, tp_size, dim=dim)
        z_sliced = torch.chunk(z, tp_size, dim=dim)
        return [torch.cat((xi, zi), dim=dim) for xi, zi in zip(x_sliced, z_sliced)]

    elif 'mixer.in_proj.weight' in key and params.mamba_version == 2:
        x, z, B, C, dt = torch.split(
            tensor,
            [
                params.mamba_d_inner,
                params.mamba_d_inner,
                params.mamba2_n_groups * params.mamba_d_state,
                params.mamba2_n_groups * params.mamba_d_state,
                params.mamba2_n_heads,
            ],
            dim=dim,
        )
        B = B.reshape(-1, params.mamba_d_state, B.shape[-1])
        C = C.reshape(-1, params.mamba_d_state, C.shape[-1])
        x_s = torch.chunk(x, tp_size, dim=dim)
        z_s = torch.chunk(z, tp_size, dim=dim)
        B_s = torch.chunk(B, tp_size, dim=dim)
        C_s = torch.chunk(C, tp_size, dim=dim)
        dt_s = torch.chunk(dt, tp_size, dim=dim)
        return [
            torch.cat((xi, zi, Bi.flatten(0, 1), Ci.flatten(0, 1), dti), dim=dim)
            for xi, zi, Bi, Ci, dti in zip(x_s, z_s, B_s, C_s, dt_s)
        ]

    elif 'mixer.conv1d' in key and params.mamba_version == 2:
        x, B, C = torch.split(
            tensor,
            [
                params.mamba_d_inner,
                params.mamba2_n_groups * params.mamba_d_state,
                params.mamba2_n_groups * params.mamba_d_state,
            ],
            dim=dim,
        )
        if 'weight' in key:
            B = B.reshape(-1, params.mamba_d_state, B.shape[-2], B.shape[-1])
            C = C.reshape(-1, params.mamba_d_state, C.shape[-2], C.shape[-1])
        elif 'bias' in key:
            B = B.reshape(-1, params.mamba_d_state)
            C = C.reshape(-1, params.mamba_d_state)
        else:
            raise ValueError(f"Unknown conv1d key: {key}")

        x_s = torch.chunk(x, tp_size, dim=dim)
        B_s = torch.chunk(B, tp_size, dim=dim)
        C_s = torch.chunk(C, tp_size, dim=dim)
        return [
            torch.cat((xi, Bi.flatten(0, 1), Ci.flatten(0, 1)), dim=dim)
            for xi, Bi, Ci in zip(x_s, B_s, C_s)
        ]

    else:
        return list(torch.chunk(tensor, tp_size, dim=dim))


# ---------------------------------------------------------------------------
# Hybrid layer pattern parsing (standalone, no Megatron imports needed)
# ---------------------------------------------------------------------------

VALID_LAYER_SYMBOLS = {'M', 'G', '*', '-', 'E'}


def parse_hybrid_layer_pattern(pattern):
    """Parse a hybrid layer pattern string into a list of layer types.

    Strips MTP separators (/) and pipeline stage separators (|), returning only
    the main decoder pattern as a list of single-character layer types.

    Returns:
        list[str]: e.g. ['M', '*', '-', 'M', '*', '-']
    """
    # Take only the main pattern (before first '/')
    main_pattern = pattern.split('/')[0]
    # Remove pipeline stage separators
    main_pattern = main_pattern.replace('|', '')
    layer_types = list(main_pattern)
    for ch in layer_types:
        if ch not in VALID_LAYER_SYMBOLS:
            raise ValueError(
                f"Invalid layer symbol '{ch}' in pattern. "
                f"Valid symbols: {VALID_LAYER_SYMBOLS}"
            )
    return layer_types


def build_layer_index_mapping(layer_types, direction):
    """Build mapping between GPT layer indices and Mamba layer indices.

    For gpt-to-mamba:
        Returns (attn_map, mlp_map) where:
        - attn_map[gpt_layer_i] = mamba_layer_j  (j is the index of the i-th '*')
        - mlp_map[gpt_layer_i]  = mamba_layer_k  (k is the index of the i-th '-')

    For mamba-to-gpt:
        Returns (attn_map, mlp_map) where:
        - attn_map[mamba_attn_idx] = gpt_layer_i
        - mlp_map[mamba_mlp_idx]   = gpt_layer_i
    """
    attn_indices = [i for i, t in enumerate(layer_types) if t == '*']
    mlp_indices = [i for i, t in enumerate(layer_types) if t == '-']
    ssm_indices = [i for i, t in enumerate(layer_types) if t == 'M']

    if direction == 'gpt-to-mamba':
        if len(attn_indices) != len(mlp_indices):
            raise ValueError(
                f"For gpt-to-mamba, the number of attention layers ({len(attn_indices)}) "
                f"must equal the number of MLP layers ({len(mlp_indices)}) in the pattern."
            )
        # attn_map: gpt_layer_i -> mamba_layer_j
        attn_map = {i: attn_indices[i] for i in range(len(attn_indices))}
        mlp_map = {i: mlp_indices[i] for i in range(len(mlp_indices))}
        return attn_map, mlp_map, ssm_indices

    elif direction == 'mamba-to-gpt':
        if len(attn_indices) != len(mlp_indices):
            raise ValueError(
                f"For mamba-to-gpt, the number of attention layers ({len(attn_indices)}) "
                f"must equal the number of MLP layers ({len(mlp_indices)}) in the pattern."
            )
        # attn_map: mamba_layer_idx -> gpt_layer_i
        attn_map = {attn_indices[i]: i for i in range(len(attn_indices))}
        mlp_map = {mlp_indices[i]: i for i in range(len(mlp_indices))}
        return attn_map, mlp_map, ssm_indices

    else:
        raise ValueError(f"Unknown direction: {direction}")


# ---------------------------------------------------------------------------
# SSM parameter initialization (for gpt-to-mamba)
# ---------------------------------------------------------------------------

def initialize_ssm_layer_params(
    layer_idx,
    d_model,
    mamba_d_inner,
    mamba_d_state,
    mamba2_n_groups,
    mamba2_n_heads,
    mamba_head_dim,
    d_conv=4,
    dt_min=0.001,
    dt_max=0.1,
    dt_init_floor=1e-4,
    A_init_range=(1, 16),
    init_method_std=0.02,
    dtype=torch.float32,
):
    """Initialize parameters for a single Mamba SSM layer from scratch.

    Follows the initialization logic from MambaMixer.__init__:
        - A_log:          log(uniform(A_init_range))
        - dt_bias:        inverse_softplus(log_uniform(dt_min, dt_max))
        - D:              ones(nheads)
        - conv1d.weight:  kaiming_uniform(a=sqrt(5))
        - conv1d.bias:    zeros
        - in_proj.weight: kaiming_uniform(a=sqrt(5))
        - in_proj.layer_norm_weight: ones(d_model)
        - out_proj.weight: kaiming_uniform(a=sqrt(5)) or normal(0, std)
        - norm.weight:    ones(d_inner)

    Returns:
        dict: {param_suffix: tensor} for one SSM layer
    """
    prefix = f'decoder.layers.{layer_idx}.mixer.'

    nheads = mamba2_n_heads
    conv_dim = mamba_d_inner + 2 * mamba2_n_groups * mamba_d_state
    in_proj_out_dim = 2 * mamba_d_inner + 2 * mamba2_n_groups * mamba_d_state + nheads

    params = OrderedDict()

    # in_proj (ColumnParallelLinear)
    in_proj_weight = torch.empty(in_proj_out_dim, d_model, dtype=dtype)
    torch.nn.init.kaiming_uniform_(in_proj_weight, a=math.sqrt(5))
    params[prefix + 'in_proj.weight'] = in_proj_weight

    # in_proj layer norm weight (fused into ColumnParallelLinear in TE)
    params[prefix + 'in_proj.layer_norm_weight'] = torch.ones(d_model, dtype=dtype)

    # conv1d
    conv_weight = torch.empty(conv_dim, 1, d_conv, dtype=dtype)
    torch.nn.init.kaiming_uniform_(conv_weight, a=math.sqrt(5))
    params[prefix + 'conv1d.weight'] = conv_weight
    params[prefix + 'conv1d.bias'] = torch.zeros(conv_dim, dtype=dtype)

    # A_log (kept in fp32)
    A = torch.empty(nheads, dtype=torch.float32)
    A.uniform_(*A_init_range)
    params[prefix + 'A_log'] = torch.log(A)

    # D (kept in fp32)
    params[prefix + 'D'] = torch.ones(nheads, dtype=torch.float32)

    # dt_bias
    dt = torch.exp(
        torch.rand(nheads, dtype=dtype)
        * (math.log(dt_max) - math.log(dt_min))
        + math.log(dt_min)
    ).clamp(min=dt_init_floor)
    inv_dt = dt + torch.log(-torch.expm1(-dt))
    params[prefix + 'dt_bias'] = inv_dt

    # norm (RMSNorm)
    params[prefix + 'norm.weight'] = torch.ones(mamba_d_inner, dtype=dtype)

    # out_proj (RowParallelLinear)
    out_proj_weight = torch.empty(d_model, mamba_d_inner, dtype=dtype)
    torch.nn.init.kaiming_uniform_(out_proj_weight, a=math.sqrt(5))
    params[prefix + 'out_proj.weight'] = out_proj_weight

    return params


# ---------------------------------------------------------------------------
# Checkpoint I/O helpers (patterns from hybrid_conversion.py)
# ---------------------------------------------------------------------------

def get_checkpoint_iteration(load_dir):
    """Read the latest iteration number from a checkpoint directory."""
    tracker_file = os.path.join(load_dir, 'latest_checkpointed_iteration.txt')
    with open(tracker_file, 'r') as f:
        metastring = f.read().strip()
        try:
            iteration = int(metastring)
        except ValueError:
            raise ValueError(
                f"Invalid iteration in {tracker_file}: '{metastring}'"
            )
    return iteration


def load_checkpoint_shards(load_dir, iteration, input_tp_size, input_pp_size):
    """Load all TP/PP shards of a checkpoint.

    Returns:
        list[list[dict]]: models[pp_rank][tp_rank] = checkpoint dict
        dict: sample_model (first shard, for metadata)
    """
    model_dir = os.path.join(load_dir, f'iter_{iteration:07d}')
    sample_model = None
    all_shards = []

    for pp in range(input_pp_size):
        tp_shards = []
        for tp in range(input_tp_size):
            dir_name = f"mp_rank_{tp:02d}"
            if input_pp_size > 1:
                dir_name += f"_{pp:03d}"
            model_file = os.path.join(model_dir, dir_name, "model_optim_rng.pt")
            checkpoint = torch.load(model_file, map_location='cpu', weights_only=False)
            tp_shards.append(checkpoint)
            if sample_model is None:
                sample_model = checkpoint
            print(f"  Loaded {model_file}")
        all_shards.append(tp_shards)

    return all_shards, sample_model


def combine_tp_shards(tp_models, params):
    """Combine TP-sharded models into a single state dict with full tensors."""
    input_tp_size = len(tp_models)
    if input_tp_size == 1:
        return OrderedDict(tp_models[0]['model'])

    combined = OrderedDict()
    for key, original_tensor in tp_models[0]['model'].items():
        if '_extra_state' in key:
            combined[key] = original_tensor
            continue

        split_dim = get_split_dim(key)
        if split_dim != -1:
            tensors = [tp_models[j]['model'][key].cpu() for j in range(input_tp_size)]
            combined[key] = combine_tp_tensors(params, key, split_dim, tensors)
        else:
            combined[key] = original_tensor

    return combined


def stitch_pp_shards(all_combined_shards, num_layers_per_pp_rank):
    """Stitch PP shards into one flat model with globally-indexed layers."""
    full_model = OrderedDict()

    for pp, combined_shard in enumerate(all_combined_shards):
        for key, tensor in combined_shard.items():
            try:
                layer_num = int(re.findall(r'\d+', key)[0])
                new_key = key.replace(
                    str(layer_num),
                    str(layer_num + pp * num_layers_per_pp_rank),
                    1,
                )
            except (IndexError, ValueError):
                new_key = key
            full_model[new_key] = tensor

    return full_model


def finalize_checkpoint(sample_model, model, params, verbose=False):
    """Finalize checkpoint metadata from a sample source checkpoint."""
    reset_iterations = params.reset_iterations

    model['args'] = copy.deepcopy(sample_model['args'])
    model['args'].tensor_model_parallel_size = params.target_tp_size
    model['args'].pipeline_model_parallel_size = params.target_pp_size
    if reset_iterations:
        model['args'].iteration = 0
        model['args'].consumed_valid_samples = 0
        model['args'].consumed_train_samples = 0
        model['args'].train_iters = 0
        model['args'].train_samples = 0

    model['checkpoint_version'] = copy.deepcopy(sample_model['checkpoint_version'])

    model['iteration'] = copy.deepcopy(sample_model['iteration'])
    if reset_iterations:
        model['iteration'] = 0

    if 'opt_param_scheduler' in sample_model:
        model['opt_param_scheduler'] = copy.deepcopy(sample_model['opt_param_scheduler'])

    model['rng_state'] = copy.deepcopy(sample_model['rng_state'])

    if verbose:
        original_args = sample_model['args'].__dict__
        final_args = model['args'].__dict__
        for key in original_args:
            if key in final_args:
                if final_args[key] != original_args[key]:
                    print(f"  ARG MISMATCH: {key}")
                    print(f"    original: {original_args[key]}")
                    print(f"    final:    {final_args[key]}")
            else:
                print(f"  ARG MISSING from final: {key} = {original_args[key]}")
        for key in final_args:
            if key not in original_args:
                print(f"  ARG ADDED to final: {key} = {final_args[key]}")

    return model


def save_checkpoint_shards(target_state_dicts, sample_model, params, save_dir, iteration):
    """Split and save checkpoint for target TP/PP configuration.

    Args:
        target_state_dicts: OrderedDict with globally-indexed layer keys (full tensors).
        sample_model: Source checkpoint dict for metadata.
        params: argparse namespace with target_tp_size, target_pp_size, etc.
        save_dir: Output directory.
        iteration: Iteration number to write.
    """
    total_layers = params.target_num_layers
    num_layers_per_pp_rank = total_layers // params.target_pp_size

    out_iteration = iteration if not params.reset_iterations else 0

    pp_offset = 0
    # Build a list of (key, tensor) for iteration
    all_items = list(target_state_dicts.items())

    for pp in range(params.target_pp_size):
        print(f"  Saving PP rank {pp}")
        tp_models = [{'model': OrderedDict()} for _ in range(params.target_tp_size)]

        for idx in range(pp_offset, len(all_items)):
            key, tensor = all_items[idx]

            # Determine if this key belongs to this PP rank
            try:
                layer_num = int(re.findall(r'\d+', key)[0])
                if layer_num >= num_layers_per_pp_rank * (pp + 1):
                    break
                new_key = key.replace(
                    str(layer_num),
                    str(layer_num - pp * num_layers_per_pp_rank),
                    1,
                )
            except (IndexError, ValueError):
                new_key = key

            pp_offset += 1

            if '_extra_state' in new_key:
                for j in range(params.target_tp_size):
                    tp_models[j]['model'][new_key] = tensor
                continue

            split_dim = get_split_dim(new_key)
            if split_dim != -1:
                slices = split_tensor_for_tp(params, new_key, split_dim, tensor)
                for j in range(params.target_tp_size):
                    tp_models[j]['model'][new_key] = slices[j]
            else:
                for j in range(params.target_tp_size):
                    tp_models[j]['model'][new_key] = tensor

        for tp in range(params.target_tp_size):
            dir_name = f"mp_rank_{tp:02d}"
            if params.target_pp_size > 1:
                dir_name += f"_{pp:03d}"

            model = finalize_checkpoint(sample_model, tp_models[tp], params, verbose=False)

            out_dir = os.path.join(save_dir, f'iter_{out_iteration:07d}', dir_name)
            os.makedirs(out_dir, exist_ok=True)
            model_file = os.path.join(out_dir, "model_optim_rng.pt")
            torch.save(model, model_file)
            print(f"    Saved {model_file}")

    # Write iteration tracker
    tracker_file = os.path.join(save_dir, 'latest_checkpointed_iteration.txt')
    with open(tracker_file, 'w') as f:
        f.write(str(out_iteration))


# ---------------------------------------------------------------------------
# Key name helpers
# ---------------------------------------------------------------------------

def get_layer_num_from_key(key):
    """Extract the layer number from a state dict key like 'decoder.layers.5.mlp...'"""
    match = re.search(r'decoder\.layers\.(\d+)\.', key)
    if match:
        return int(match.group(1))
    return None


def replace_layer_num(key, old_num, new_num):
    """Replace the layer number in a state dict key."""
    return key.replace(f'decoder.layers.{old_num}.', f'decoder.layers.{new_num}.', 1)


def is_attention_param(key):
    """Check if a key belongs to an attention sub-layer."""
    return 'self_attention.' in key or 'input_layernorm.' in key


def is_mlp_param(key):
    """Check if a key belongs to an MLP sub-layer."""
    return ('mlp.' in key or 'pre_mlp_layernorm.' in key) and 'self_attention' not in key


def is_ssm_param(key):
    """Check if a key belongs to a Mamba SSM mixer sub-layer."""
    ssm_markers = ['mixer.in_proj', 'mixer.conv1d', 'mixer.A_log', 'mixer.D',
                   'mixer.dt_bias', 'mixer.norm', 'mixer.out_proj',
                   'mixer.x_proj', 'mixer.dt_proj']
    return any(m in key for m in ssm_markers)


def is_layer_norm_for_ssm(key):
    """Check if a key is the input layer norm for an SSM layer.

    In hybrid models, SSM layers can have their own input_layernorm or the
    norm can be fused into in_proj.layer_norm_weight.
    """
    return 'in_proj.layer_norm_weight' in key


# ---------------------------------------------------------------------------
# Core conversion: GPT -> Mamba
# ---------------------------------------------------------------------------

def convert_gpt_to_mamba(full_model, layer_types, args):
    """Convert a GPT state dict to a Mamba hybrid state dict.

    Args:
        full_model: OrderedDict with globally-indexed GPT state dict keys.
        layer_types: list of layer type chars from hybrid_layer_pattern.
        args: Parsed CLI arguments.

    Returns:
        OrderedDict: Mamba state dict with globally-indexed keys.
    """
    attn_map, mlp_map, ssm_indices = build_layer_index_mapping(
        layer_types, 'gpt-to-mamba'
    )
    num_gpt_layers = len(attn_map)

    # Validate GPT layer count
    gpt_layer_nums = set()
    for key in full_model:
        lnum = get_layer_num_from_key(key)
        if lnum is not None:
            gpt_layer_nums.add(lnum)

    if len(gpt_layer_nums) != num_gpt_layers:
        raise ValueError(
            f"GPT checkpoint has {len(gpt_layer_nums)} layers, but the pattern "
            f"has {num_gpt_layers} attention ('*') and {num_gpt_layers} MLP ('-') "
            f"layers. These must match."
        )

    target = OrderedDict()
    dtype = None

    # Copy / rename non-layer params
    for key, tensor in full_model.items():
        if dtype is None and tensor.dtype in (torch.float16, torch.bfloat16, torch.float32):
            dtype = tensor.dtype

        if 'decoder.layers.' in key:
            continue

        # Rename final_layernorm -> final_norm
        if 'decoder.final_layernorm' in key:
            new_key = key.replace('decoder.final_layernorm', 'decoder.final_norm')
            target[new_key] = tensor
        else:
            target[key] = tensor

    if dtype is None:
        dtype = torch.float32

    # Map attention and MLP params
    for key, tensor in full_model.items():
        lnum = get_layer_num_from_key(key)
        if lnum is None:
            continue

        if is_attention_param(key):
            target_layer = attn_map[lnum]
            new_key = replace_layer_num(key, lnum, target_layer)
            target[new_key] = tensor

        elif is_mlp_param(key):
            target_layer = mlp_map[lnum]
            new_key = replace_layer_num(key, lnum, target_layer)
            target[new_key] = tensor

        # (any other layer params get copied as-is with their own mapping,
        #  but for pure GPT there should only be attention + MLP)

    # Initialize SSM layers from scratch
    print(f"  Initializing {len(ssm_indices)} SSM layers from scratch...")
    for layer_idx in ssm_indices:
        ssm_params = initialize_ssm_layer_params(
            layer_idx=layer_idx,
            d_model=args.d_model,
            mamba_d_inner=args.mamba_d_inner,
            mamba_d_state=args.mamba_d_state,
            mamba2_n_groups=args.mamba2_n_groups,
            mamba2_n_heads=args.mamba2_n_heads,
            mamba_head_dim=args.mamba2_head_dim,
            d_conv=getattr(args, 'd_conv', 4),
            init_method_std=getattr(args, 'init_method_std', 0.02),
            dtype=dtype,
        )
        target.update(ssm_params)

    # Sort by layer index for consistent ordering
    target = _sort_state_dict(target)

    return target


# ---------------------------------------------------------------------------
# Core conversion: Mamba -> GPT
# ---------------------------------------------------------------------------

def convert_mamba_to_gpt(full_model, layer_types, args):
    """Convert a Mamba hybrid state dict to a GPT state dict.

    Args:
        full_model: OrderedDict with globally-indexed Mamba state dict keys.
        layer_types: list of layer type chars from hybrid_layer_pattern.
        args: Parsed CLI arguments.

    Returns:
        OrderedDict: GPT state dict with globally-indexed keys.
    """
    attn_map, mlp_map, ssm_indices = build_layer_index_mapping(
        layer_types, 'mamba-to-gpt'
    )
    num_gpt_layers = len(attn_map)

    target = OrderedDict()
    discarded_ssm_keys = []

    # Copy / rename non-layer params
    for key, tensor in full_model.items():
        if 'decoder.layers.' in key:
            continue

        # Rename final_norm -> final_layernorm
        if 'decoder.final_norm' in key:
            new_key = key.replace('decoder.final_norm', 'decoder.final_layernorm')
            target[new_key] = tensor
        else:
            target[key] = tensor

    # Map attention and MLP params, discard SSM
    for key, tensor in full_model.items():
        lnum = get_layer_num_from_key(key)
        if lnum is None:
            continue

        if is_ssm_param(key) or is_layer_norm_for_ssm(key):
            # Discard SSM params
            if lnum in ssm_indices:
                discarded_ssm_keys.append(key)
                continue

        if is_attention_param(key) and lnum in attn_map:
            target_layer = attn_map[lnum]
            new_key = replace_layer_num(key, lnum, target_layer)
            target[new_key] = tensor

        elif is_mlp_param(key) and lnum in mlp_map:
            target_layer = mlp_map[lnum]
            new_key = replace_layer_num(key, lnum, target_layer)
            target[new_key] = tensor

        elif lnum in ssm_indices:
            # Any remaining SSM-layer param not caught above
            discarded_ssm_keys.append(key)

    if discarded_ssm_keys:
        print(f"\n  WARNING: Discarded {len(discarded_ssm_keys)} SSM parameter tensors "
              f"from {len(ssm_indices)} Mamba layers (no GPT equivalent).")
        print(f"  First few discarded keys: {discarded_ssm_keys[:5]}")

    target = _sort_state_dict(target)

    return target


# ---------------------------------------------------------------------------
# Sorting helper
# ---------------------------------------------------------------------------

def _sort_state_dict(state_dict):
    """Sort state dict keys so that layer-indexed keys are in order."""
    def sort_key(item):
        key = item[0]
        # Extract layer number if present
        match = re.search(r'decoder\.layers\.(\d+)\.', key)
        if match:
            return (1, int(match.group(1)), key)
        # Non-layer keys: embeddings first, output_layer last
        if 'embedding' in key:
            return (0, 0, key)
        if 'output_layer' in key:
            return (2, 0, key)
        if 'decoder.final' in key:
            return (1, 999999, key)
        return (0, 1, key)

    return OrderedDict(sorted(state_dict.items(), key=sort_key))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    print("\n====RUNNING GPT <-> MAMBA CHECKPOINT CONVERSION====\n")
    print(f"  Direction:            {args.direction}")
    print(f"  Source:               {args.load_dir}")
    print(f"  Target:               {args.save_dir}")
    print(f"  Hybrid layer pattern: {args.hybrid_layer_pattern}")
    print(f"  Target TP size:       {args.target_tp_size}")
    print(f"  Target PP size:       {args.target_pp_size}")

    # Compute derived Mamba dimensions
    args.mamba_d_inner = args.d_model * 2
    args.mamba2_n_heads = args.mamba_d_inner // args.mamba2_head_dim

    # Parse hybrid layer pattern
    layer_types = parse_hybrid_layer_pattern(args.hybrid_layer_pattern)
    total_mamba_layers = len(layer_types)
    attn_count = sum(1 for t in layer_types if t == '*')
    mlp_count = sum(1 for t in layer_types if t == '-')
    ssm_count = sum(1 for t in layer_types if t == 'M')
    print(f"\n  Pattern: {len(layer_types)} total layers "
          f"({attn_count} attn, {mlp_count} MLP, {ssm_count} SSM, "
          f"{len(layer_types) - attn_count - mlp_count - ssm_count} other)")

    # 1. Load source checkpoint
    print("\n[Step 1] Loading source checkpoint...")
    iteration = get_checkpoint_iteration(args.load_dir)
    print(f"  Iteration: {iteration}")

    model_dir = os.path.join(args.load_dir, f'iter_{iteration:07d}')
    sub_models = os.listdir(model_dir)
    sample_file = os.path.join(model_dir, sub_models[0], "model_optim_rng.pt")
    sample_model = torch.load(sample_file, map_location='cpu', weights_only=False)

    input_tp_size = sample_model['args'].tensor_model_parallel_size
    input_pp_size = sample_model['args'].pipeline_model_parallel_size
    input_num_layers = sample_model['args'].num_layers
    num_layers_per_pp_rank = input_num_layers // input_pp_size

    print(f"  Source: TP={input_tp_size}, PP={input_pp_size}, "
          f"num_layers={input_num_layers}")

    all_shards, sample_model = load_checkpoint_shards(
        args.load_dir, iteration, input_tp_size, input_pp_size
    )

    # 2. Combine TP shards
    print("\n[Step 2] Combining TP shards into full tensors...")
    combined_pp_shards = []
    for pp in range(input_pp_size):
        combined = combine_tp_shards(all_shards[pp], args)
        combined_pp_shards.append(combined)
        print(f"  PP rank {pp}: {len(combined)} parameters combined")

    # 3. Stitch PP shards
    print("\n[Step 3] Stitching PP shards into flat model...")
    full_model = stitch_pp_shards(combined_pp_shards, num_layers_per_pp_rank)
    print(f"  Full model: {len(full_model)} parameters")

    # 4. Convert
    print(f"\n[Step 4] Converting ({args.direction})...")
    if args.direction == 'gpt-to-mamba':
        target_state_dict = convert_gpt_to_mamba(full_model, layer_types, args)
        args.target_num_layers = total_mamba_layers
        # Update checkpoint args for Mamba model
        sample_model['args'].num_layers = total_mamba_layers
    elif args.direction == 'mamba-to-gpt':
        target_state_dict = convert_mamba_to_gpt(full_model, layer_types, args)
        args.target_num_layers = attn_count
        # Update checkpoint args for GPT model
        sample_model['args'].num_layers = attn_count
    else:
        raise ValueError(f"Unknown direction: {args.direction}")
    print(f"  Target model: {len(target_state_dict)} parameters")

    # 5. Save with target TP/PP
    print(f"\n[Step 5] Saving to {args.save_dir}...")
    save_checkpoint_shards(
        target_state_dict, sample_model, args, args.save_dir, iteration
    )

    print("\n====CONVERSION COMPLETE====\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert checkpoints between GPTModel and MambaModel formats.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        '--direction', type=str, required=True,
        choices=['gpt-to-mamba', 'mamba-to-gpt'],
        help='Conversion direction.',
    )
    parser.add_argument('--load-dir', type=str, required=True,
                        help='Path to source checkpoint directory.')
    parser.add_argument('--save-dir', type=str, required=True,
                        help='Path to target checkpoint directory.')
    parser.add_argument('--hybrid-layer-pattern', type=str, required=True,
                        help='Hybrid layer pattern string, e.g. "M*-M*-M*-M*-".')
    parser.add_argument('--target-tp-size', type=int, default=1,
                        help='Target tensor parallel size.')
    parser.add_argument('--target-pp-size', type=int, default=1,
                        help='Target pipeline parallel size.')

    # Model architecture params
    parser.add_argument('--d-model', type=int, default=4096,
                        help='Model hidden dimension.')
    parser.add_argument('--mamba-version', type=int, default=2,
                        choices=[1, 2], help='Mamba SSM version.')
    parser.add_argument('--mamba-d-state', type=int, default=128,
                        help='Mamba state dimension.')
    parser.add_argument('--mamba2-n-groups', type=int, default=8,
                        help='Number of groups (Mamba v2).')
    parser.add_argument('--mamba2-head-dim', type=int, default=64,
                        help='Head dimension (Mamba v2).')
    parser.add_argument('--d-conv', type=int, default=4,
                        help='Causal convolution kernel size.')

    # Initialization params
    parser.add_argument('--init-method-std', type=float, default=0.02,
                        help='Std for initializing new Mamba SSM params.')

    # Checkpoint control
    parser.add_argument('--reset-iterations', action='store_true',
                        help='Zero out the training iteration count.')

    args = parser.parse_args()
    main(args)
