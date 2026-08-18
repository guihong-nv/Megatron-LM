# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Construction-time selection for core attention implementations."""

from __future__ import annotations

from dataclasses import dataclass

from megatron.core.ops._dependencies import require_module, require_symbols
from megatron.core.ops.config import Backend, Operation, OpsBackendConfig


@dataclass(frozen=True)
class AttentionOps:
    """Resolved core-attention target."""

    core_attention: type


def _resolve_core_attention(config: OpsBackendConfig) -> type:
    backend = config.backend_for(Operation.CORE_ATTENTION)
    if backend is Backend.LOCAL:
        from megatron.core.transformer.dot_product_attention import DotProductAttention

        return DotProductAttention
    if backend in {Backend.TRANSFORMER_ENGINE, Backend.INFERENCE_OPTIMIZED}:
        require_module("transformer_engine.pytorch", purpose=Operation.CORE_ATTENTION.value)
        return require_symbols(
            "megatron.core.extensions.transformer_engine",
            ("TEDotProductAttention",),
            purpose=Operation.CORE_ATTENTION.value,
        )[
            "TEDotProductAttention"
        ]  # type: ignore[return-value]
    if backend is Backend.KITCHEN:
        require_module("nvidia_kitchen", purpose=Operation.CORE_ATTENTION.value)
        symbol = (
            "KitchenFlashAttention"
            if config.kitchen_attention_backend == "fa"
            else "KitchenDotProductAttention"
        )
        return require_symbols(
            "megatron.core.extensions.kitchen", (symbol,), purpose=Operation.CORE_ATTENTION.value
        )[
            symbol
        ]  # type: ignore[return-value]
    raise ValueError(
        f"Backend '{backend.value}' does not implement {Operation.CORE_ATTENTION.value}"
    )


def resolve_attention_ops(config: OpsBackendConfig) -> AttentionOps:
    """Resolve core attention once during provider construction."""
    return AttentionOps(core_attention=_resolve_core_attention(config))
