# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Construction-time selection for Transformer Engine fused MLP modules."""

from __future__ import annotations

from typing import cast

from megatron.core.ops import _dependencies
from megatron.core.ops.config import Backend, Operation


def resolve_fused_mlp(backend: Backend, grouped: bool = False) -> type | None:
    """Return the fused MLP module, or ``None`` for the existing unfused path."""

    if backend is not Backend.TRANSFORMER_ENGINE:
        if backend in (Backend.LOCAL, Backend.NONE, Backend.INFERENCE_OPTIMIZED, Backend.KITCHEN):
            return None
        raise ValueError(f"unsupported {Operation.FUSED_MLP.value} backend: {backend!r}")

    purpose = Operation.FUSED_MLP.value
    _dependencies.require_module("transformer_engine.pytorch", purpose=purpose)
    _dependencies.require_version("transformer-engine", "1.13.0", purpose=purpose)
    symbol = "TEFusedMLPWithGroupedLinear" if grouped else "TEFusedMLP"
    module_name = "megatron.core.extensions.transformer_engine"
    target = _dependencies.require_symbols(module_name, (symbol,), purpose=purpose)[symbol]
    if target is None:
        raise ImportError(f"{module_name}.{symbol} is required for {purpose}")
    return cast(type, target)
