# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""V4.1 resumes must reconstruct the architecture before loading model weights."""

import json
from argparse import ArgumentParser
from dataclasses import asdict, fields
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from megatron.core.models.deepseek_v41.config import DeepSeekV41Config
from megatron.training.argument_utils import core_transformer_config_from_args
from megatron.training.arguments import add_megatron_arguments
from megatron.training.checkpointing import CheckpointType, load_args_from_checkpoint
from megatron.training.models.deepseek_v41 import configure_v41_args


def _args(*flags):
    parser = ArgumentParser()
    add_megatron_arguments(parser)
    args = parser.parse_args(list(flags))
    args.params_dtype = torch.bfloat16
    return args


def _restore(args, saved):
    with mock.patch(
        'megatron.training.checkpointing._load_base_checkpoint',
        return_value=({'args': saved}, 'checkpoint', False, CheckpointType.LEGACY),
    ):
        restored, _ = load_args_from_checkpoint(args)
    return restored


def test_v41_checkpoint_restores_flash_architecture_after_engram_json_moves(tmp_path):
    """Load architecture from saved args, preserving the new PP layout and map path."""
    engram = {
        'layer_ids': [1, 14],
        'num_embeddings': [384006168, 384016682],
        'max_ngram_size': 4,
        'vocab_size': 16000000,
        'n_heads': 8,
        'head_dim': 256,
        'pad_token_id': 2,
        'compressed_vocab_size': 99092,
    }
    source = tmp_path / 'engram.json'
    source.write_text(json.dumps(engram))
    ratios = [value for ratio in [0] * 2 + [2] * 18 + [1] * 20 for value in (ratio, 0)]
    saved = _args(
        *"""
        --dsv4-version v4.1 --num-layers 80 --hidden-size 5120 --num-attention-heads 64
        --multi-latent-attention --q-lora-rank 1280 --v-head-dim 512
        --qk-pos-emb-head-dim 64 --qk-layernorm --normalization RMSNorm
        --norm-epsilon 1e-20 --output-projection-groups 8
        --output-projection-lora-rank 1024 --rotary-base 10000
        --rotary-scaling-factor 16 --original-max-position-embeddings 65536
        --mscale 0 --mscale-all-dim 0 --csa-compress-rotary-base 160000
        --csa-window-size 128 --csa2-kv-source-layers 4 16 28 40
        --csa2-index-source-layers 4 16 28 40 48 56 64 72
        --csa2-candidate-source-layer 40 --csa2-candidate-topk-blocks 2048
        --csa2-candidate-block-size 8 --dsa-indexer-n-heads 32
        --dsa-indexer-head-dim 128 --dsa-indexer-topk 512
        --no-dsa-indexer-rotate-activation --dsa-kernel-backend none
        --dsa-indexer-loss-coeff 0.01 --enable-mhc-connections --mhc-single-pass
        --mhc-num-residual-streams 4 --mhc-sinkhorn-iterations 20
        --num-experts 384 --moe-router-topk 6 --moe-ffn-hidden-size 2304
        --moe-shared-expert-intermediate-size 2304 --moe-router-score-function sqrtsoftplus
        --moe-router-topk-scaling-factor 1.5 --moe-router-dtype fp32
        --moe-router-enable-expert-bias --moe-router-load-balancing-type none
        --moe-combine-in-fp32 --moe-token-dispatcher-type alltoall
        --moe-aux-loss-coeff 0
        --swiglu --activation-func-clamp-value 10 --disable-bias-linear
        --attention-dropout 0 --hidden-dropout 0 --bf16
        --no-create-attention-mask-in-dataloader --untie-embeddings-and-output-weights
        """.split(),
        '--hybrid-layer-pattern',
        'VE' * 40,
        '--csa-compress-ratios',
        json.dumps(ratios, separators=(',', ':')),
        '--engram-config',
        str(source),
        '--engram-token-map',
        '/old-cluster/map.pt',
    )
    original = core_transformer_config_from_args(saved)
    assert saved.engram_config_data == engram
    source.unlink()

    resumed = _args(
        '--load',
        'checkpoint',
        '--use-checkpoint-args',
        '--bf16',
        '--hybrid-layer-pattern',
        'VE' * 20 + '|' + 'VE' * 20,
        '--engram-token-map',
        '/new-cluster/map.pt',
        '--no-create-attention-mask-in-dataloader',
    )
    _restore(resumed, saved)
    rebuilt = core_transformer_config_from_args(resumed)
    assert isinstance(rebuilt, DeepSeekV41Config)
    assert resumed.hybrid_layer_pattern == 'VE' * 20 + '|' + 'VE' * 20
    assert resumed.engram_token_map == '/new-cluster/map.pt'
    assert asdict(rebuilt.engram_config) == engram
    # Compare the complete transformer config; generated initializer callables
    # are fresh objects, while architecture and objective settings must match.
    for field in fields(original):
        expected = getattr(original, field.name)
        if not callable(expected):
            assert getattr(rebuilt, field.name) == expected, field.name
    resumed.csa2_kv_source_layers.append(78)
    resumed.engram_config_data['layer_ids'].append(39)
    assert saved.csa2_kv_source_layers == [4, 16, 28, 40]
    assert saved.engram_config_data == engram


def test_v41_checkpoint_restores_disabled_candidates_and_runtime_backend():
    saved = SimpleNamespace(
        dsv4_version='v4.1',
        csa2_candidate_source_layer=None,
        csa2_candidate_topk_blocks=0,
        csa2_candidate_block_size=0,
        dsa_indexer_loss_coeff=None,
        dsa_kernel_backend='none',
    )
    args = _args(
        '--load',
        'checkpoint',
        '--csa2-candidate-source-layer',
        '4',
        '--csa2-candidate-topk-blocks',
        '16',
        '--csa2-candidate-block-size',
        '8',
        '--dsa-indexer-loss-coeff',
        '0.01',
        '--dsa-kernel-backend',
        'cudnn',
    )
    _restore(args, saved)
    assert args.csa2_candidate_source_layer is None
    assert args.csa2_candidate_topk_blocks == args.csa2_candidate_block_size == 0
    assert args.dsa_indexer_loss_coeff is None
    assert args.dsa_kernel_backend == 'cudnn'


@pytest.mark.parametrize('enabled', [False, True])
def test_v41_checkpoint_restores_expert_accumulation_policy(enabled):
    """Checkpoint-driven resumes retain the numerical policy and its dispatcher."""
    saved = SimpleNamespace(
        dsv4_version='v4.1', moe_combine_in_fp32=enabled, moe_token_dispatcher_type='alltoall'
    )
    args = _args('--load', 'checkpoint')
    args.moe_combine_in_fp32 = not enabled
    _restore(args, saved)
    assert args.moe_combine_in_fp32 is enabled
    if enabled:
        assert args.moe_token_dispatcher_type == 'alltoall'


@pytest.mark.parametrize('enabled', [False, True])
def test_legacy_v41_checkpoint_preserves_requested_accumulation_policy(enabled):
    """Older checkpoints have no policy metadata; keep the caller's choice."""
    args = _args('--load', 'checkpoint', '--moe-token-dispatcher-type', 'alltoall')
    args.moe_combine_in_fp32 = enabled
    _restore(args, SimpleNamespace(dsv4_version='v4.1'))
    assert args.moe_combine_in_fp32 is enabled


@pytest.mark.parametrize('version', [None, 'v4'])
def test_other_checkpoint_architectures_keep_v41_defaults(version):
    saved = SimpleNamespace(dsv4_version=version, q_lora_rank=1280, mhc_single_pass=True)
    args = _args('--load', 'checkpoint')
    defaults = args.q_lora_rank, args.mhc_single_pass
    _restore(args, saved)
    assert (args.q_lora_rank, args.mhc_single_pass) == defaults


def test_v41_checkpoint_supports_legacy_engram_path_override(tmp_path):
    source = tmp_path / 'engram.json'
    options = dict(
        layer_ids=[1],
        num_embeddings=[12],
        max_ngram_size=2,
        vocab_size=11,
        n_heads=1,
        head_dim=8,
        pad_token_id=2,
        compressed_vocab_size=16,
    )
    source.write_text(json.dumps(options))
    saved = SimpleNamespace(dsv4_version='v4.1', engram_config='/removed/engram.json')
    args = _args(
        '--load',
        'checkpoint',
        '--hybrid-layer-pattern',
        'VEVE',
        '--num-layers',
        '4',
        '--engram-config',
        str(source),
        '--engram-token-map',
        '/new/map.pt',
        '--no-create-attention-mask-in-dataloader',
    )
    _restore(args, saved)
    kwargs = {}
    configure_v41_args(args, kwargs)
    assert asdict(kwargs['engram_config']) == options
