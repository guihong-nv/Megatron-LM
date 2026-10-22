# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Ordered FP32 expert accumulation, including its final shared-expert rounding."""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from megatron.core.models.deepseek_v41.config import DeepSeekV41Config
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.moe.moe_utils import permute, unpermute
from megatron.core.transformer.moe.token_dispatcher import MoEAlltoAllTokenDispatcher
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils

DEVICES = [
    "cpu",
    pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA")),
]


def _config(**overrides):
    values = dict(
        num_layers=1,
        hidden_size=8,
        num_attention_heads=1,
        num_moe_experts=8,
        moe_router_topk=3,
        moe_token_dispatcher_type="alltoall",
        moe_combine_in_fp32=True,
    )
    values.update(overrides)
    return TransformerConfig(**values)


def _mapping(routing_map, hidden):
    tokens = torch.zeros((routing_map.size(0), hidden), device=routing_map.device)
    _, _, indices, inverse_map, _ = permute(
        tokens,
        routing_map,
        num_out_tokens=int(routing_map.sum()),
        return_batch_invariant_inverse_map=True,
    )
    return indices, inverse_map


def _combine(rows, routing_map, indices, inverse_map):
    return unpermute(
        rows,
        indices,
        restore_shape=(routing_map.size(0), rows.size(1)),
        routing_map=routing_map,
        batch_invariant_inverse_map=inverse_map,
        combine_in_fp32=True,
    )


def _expert_loop(rows, routing_map):
    """Independent reference: one ascending-expert update at a time, as in the model."""
    output = torch.zeros(
        (routing_map.size(0), rows.size(1)), dtype=torch.float32, device=rows.device
    )
    offset = 0
    for expert in range(routing_map.size(1)):
        tokens = torch.where(routing_map[:, expert])[0]
        count = tokens.numel()
        output[tokens] = output[tokens] + rows[offset : offset + count].float()
        offset += count
    return output


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("topk", [1, 3, 6])
def test_ordered_combine_forward_and_gradient(device, topk):
    """Match the independent expert loop and send each gradient to its owning row."""
    routing = torch.zeros((7, 8), dtype=torch.bool, device=device)
    for token in range(7):
        routing[token, (torch.arange(topk, device=device) + token) % 8] = True
    indices, inverse = _mapping(routing, 4)
    rows = (
        (torch.arange(indices.numel() * 4, device=device).reshape(-1, 4).remainder(17) - 8)
        .to(torch.bfloat16)
        .requires_grad_()
    )
    actual = _combine(rows, routing, indices, inverse)
    torch.testing.assert_close(actual, _expert_loop(rows, routing), rtol=0, atol=0)
    assert actual.dtype == torch.float32
    gradient = torch.arange(28, device=device).reshape(7, 4).float() / 4
    (actual * gradient).sum().backward()
    torch.testing.assert_close(rows.grad, gradient[indices].to(rows.dtype), rtol=0, atol=0)


@pytest.mark.parametrize("device", DEVICES)
def test_fp32_expert_order_is_not_fp64_or_reverse_order(device):
    """Cancellation distinguishes the required FP32 chain from other exact sums."""
    routing = torch.ones((1, 3), dtype=torch.bool, device=device)
    indices, inverse = _mapping(routing, 1)
    rows = torch.tensor([[2**24], [1], [-(2**24)]], device=device, dtype=torch.bfloat16)
    actual = _combine(rows, routing, indices, inverse)
    assert actual.item() == 0
    assert rows.double().sum().item() == 1
    reverse = torch.zeros(1, device=device)
    for row in rows.flip(0):
        reverse += row.float()
    assert reverse.item() == 1


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("overlap", [False, True])
def test_shared_expert_is_added_before_final_cast(device, overlap):
    """Exercise the actual MoELayer postprocess and both shared-addition call sites."""
    routing = torch.ones((1, 2), dtype=torch.bool, device=device)
    indices, inverse = _mapping(routing, 1)
    rows = torch.tensor([[256], [1]], dtype=torch.bfloat16, device=device, requires_grad=True)
    shared = torch.tensor([[-256]], dtype=torch.bfloat16, device=device, requires_grad=True)
    dispatcher = MoEAlltoAllTokenDispatcher.__new__(MoEAlltoAllTokenDispatcher)
    dispatcher.config = _config()
    dispatcher.hidden_shape = dispatcher.hidden_shape_before_permute = (1, 1)
    dispatcher.hidden_dtype = torch.bfloat16
    dispatcher.reversed_local_input_permutation_mapping = indices
    dispatcher.batch_invariant_inverse_permutation_mapping = inverse
    dispatcher.routing_map = routing
    dispatcher.drop_and_pad = False
    dispatcher.shared_experts = SimpleNamespace(get_output=lambda: shared) if overlap else None
    layer = SimpleNamespace(config=dispatcher.config, token_dispatcher=dispatcher)
    actual = MoELayer.postprocess(layer, rows, None if overlap else shared)
    assert actual.dtype == torch.bfloat16
    assert actual.item() == 1
    actual.float().sum().backward()
    torch.testing.assert_close(rows.grad, torch.ones_like(rows), rtol=0, atol=0)
    torch.testing.assert_close(shared.grad, torch.ones_like(shared), rtol=0, atol=0)


@pytest.mark.parametrize("device", DEVICES)
def test_ordered_combine_probability_gradient_and_empty_input(device):
    """Preserve upstream router gradients and the gradient of an empty expert tensor."""
    routing = torch.ones((2, 3), dtype=torch.bool, device=device)
    indices, inverse = _mapping(routing, 2)
    rows = torch.arange(12, device=device).reshape(6, 2).to(torch.bfloat16)
    probabilities = torch.full((6,), 0.25, device=device, requires_grad=True)
    weighted = (rows.float() * probabilities[:, None]).to(rows.dtype)
    _combine(weighted, routing, indices, inverse).sum().backward()
    torch.testing.assert_close(probabilities.grad, rows.float().sum(-1), rtol=0, atol=0)
    empty_routing = torch.zeros((0, 3), dtype=torch.bool, device=device)
    empty_indices, empty_inverse = _mapping(empty_routing, 2)
    empty = torch.empty((0, 2), device=device, dtype=torch.bfloat16, requires_grad=True)
    output = _combine(empty, empty_routing, empty_indices, empty_inverse)
    assert output.shape == (0, 2) and output.dtype == torch.float32
    output.sum().backward()
    assert empty.grad is not None and empty.grad.shape == empty.shape


def test_default_combine_policy_and_v41_defaults():
    """Only the V4.1 architecture changes its default dispatch and combine policy."""
    generic = TransformerConfig(num_layers=1, hidden_size=8, num_attention_heads=1)
    assert not generic.moe_combine_in_fp32
    assert generic.moe_token_dispatcher_type == "allgather"
    path = Path(__file__).resolve().parents[4] / "examples/deepseek_v41/config_flash.json"
    v41 = DeepSeekV41Config.from_hf(json.loads(path.read_text()))
    assert v41.moe_combine_in_fp32 and v41.moe_token_dispatcher_type == "alltoall"
    rows = torch.tensor([[1], [2]], dtype=torch.bfloat16)
    indices = torch.tensor([1, 0])
    actual = unpermute(rows, indices, (2, 1))
    assert actual.dtype == rows.dtype
    torch.testing.assert_close(actual, rows.flip(0), rtol=0, atol=0)


@pytest.mark.parametrize(
    "override",
    [
        {"moe_token_dispatcher_type": "allgather"},
        {"moe_token_dispatcher_type": "flex"},
        {"moe_permute_fusion": True},
        {"moe_permute_fusion_into_hybridep": True},
        {"moe_token_dropping": True},
        {"moe_expert_capacity_factor": 1.0},
        {"moe_pad_expert_input_to_capacity": True},
        {"moe_pad_experts_for_cuda_graph_inference": True},
        {"moe_router_padding_for_quantization": True},
        {"moe_virtual_expert_load_balance": True},
        {"moe_num_hash_layers": 1},
        {"batch_invariant_mode": True},
        {"moe_latent_size": 4},
        {"transformer_impl": "inference_optimized"},
    ],
)
def test_ordered_combine_rejects_incompatible_config(override):
    """Unsupported paths must fail at construction instead of silently changing sums."""
    with pytest.raises(ValueError, match="moe_combine_in_fp32"):
        _config(**override)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA")
@pytest.mark.parametrize("ep_size", [1, 2])
def test_alltoall_ordered_combine_distributed(ep_size):
    """AllToAll restores ascending global experts and preserves input/shared gradients."""
    if int(os.environ.get("WORLD_SIZE", "1")) % ep_size:
        pytest.skip(f"EP={ep_size} requires a divisible world size")
    Utils.initialize_model_parallel(expert_model_parallel_size=ep_size)
    try:
        groups = ProcessGroupCollection.use_mpu_process_groups()
        local_experts = 8 // ep_size
        first_expert = groups.ep.rank() * local_experts
        config = _config(expert_model_parallel_size=ep_size)
        dispatcher = MoEAlltoAllTokenDispatcher(
            local_experts,
            list(range(first_expert, first_expert + local_experts)),
            config=config,
            pg_collection=groups,
        )
        hidden = torch.ones((7, 8), device="cuda", dtype=torch.bfloat16, requires_grad=True)
        routing = torch.zeros((7, 8), device="cuda", dtype=torch.bool)
        for token in range(7):
            routing[token, (torch.tensor([0, 1, 3], device="cuda") + token) % 8] = True
        probs = routing.float() / 8
        dispatched, dispatched_probs = dispatcher.dispatch_preprocess(hidden, routing, probs)
        dispatched, dispatched_probs = dispatcher.token_dispatch(dispatched, dispatched_probs)
        dispatched, counts, permuted_probs = dispatcher.dispatch_postprocess(
            dispatched, dispatched_probs
        )
        coefficients = torch.repeat_interleave(
            torch.arange(first_expert + 1, first_expert + local_experts + 1, device="cuda"),
            counts.to(device="cuda"),
            output_size=dispatched.size(0),
        )
        expert_output = (dispatched.float() * coefficients[:, None] * permuted_probs[:, None]).to(
            hidden.dtype
        )
        combined = dispatcher.token_combine(dispatcher.combine_preprocess(expert_output))
        shared = torch.ones_like(hidden, requires_grad=True)
        layer = SimpleNamespace(config=config, token_dispatcher=dispatcher)
        output = MoELayer.postprocess(layer, combined, shared)
        expected_scale = (probs * torch.arange(1, 9, device="cuda")).sum(-1, keepdim=True)
        expected = (hidden.float() * expected_scale + shared.float()).to(hidden.dtype)
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        output.float().sum().backward()
        torch.testing.assert_close(
            hidden.grad, expected_scale.expand_as(hidden).to(hidden.dtype), rtol=0, atol=0
        )
        torch.testing.assert_close(shared.grad, torch.ones_like(shared), rtol=0, atol=0)
    finally:
        Utils.destroy_model_parallel()
