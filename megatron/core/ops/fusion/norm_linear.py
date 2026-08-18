# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Construction-time selection for fused norm + column-parallel linear ops."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from megatron.core.ops import _dependencies
from megatron.core.ops.config import Backend, Operation


@dataclass(frozen=True)
class NormLinearSpec:
    """Resolved fused norm-linear target.

    ``linear`` is ``None`` when the caller must build separate norm and linear ops.
    """

    linear: type | None
    fuses_norm: bool

    def __post_init__(self) -> None:
        if self.fuses_norm != (self.linear is not None):
            raise ValueError("fuses_norm must match whether a fused linear target is present")


_UNFUSED = NormLinearSpec(linear=None, fuses_norm=False)


def _require_target(dependency: str, module_name: str, symbol: str) -> type:
    purpose = Operation.NORM_LINEAR.value
    _dependencies.require_module(dependency, purpose=purpose)
    target = _dependencies.require_symbols(module_name, (symbol,), purpose=purpose)[symbol]
    if target is None:
        raise ImportError(f"{module_name}.{symbol} is required for {Operation.NORM_LINEAR.value}")
    return cast(type, target)


def resolve_norm_linear(backend: Backend) -> NormLinearSpec:
    """Resolve the fused norm-linear target for ``backend``."""

    if backend in (Backend.LOCAL, Backend.NONE):
        return _UNFUSED
    if backend is Backend.TRANSFORMER_ENGINE:
        target = _require_target(
            "transformer_engine.pytorch",
            "megatron.core.extensions.transformer_engine",
            "TELayerNormColumnParallelLinear",
        )
    elif backend is Backend.INFERENCE_OPTIMIZED:
        target = _require_target(
            "transformer_engine.pytorch",
            "megatron.core.tensor_parallel.inference_layers",
            "InferenceLayerNormColumnParallelLinear",
        )
    elif backend is Backend.KITCHEN:
        target = _require_target(
            "nvidia_kitchen",
            "megatron.core.extensions.kitchen",
            "KitchenLayerNormColumnParallelLinear",
        )
    else:
        raise ValueError(f"unsupported {Operation.NORM_LINEAR.value} backend: {backend!r}")
    return NormLinearSpec(linear=target, fuses_norm=True)
