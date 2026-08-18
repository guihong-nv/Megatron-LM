# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Construction-time selection for normalization implementations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from packaging.version import Version

from megatron.core.ops._dependencies import (
    dependency_version,
    optional_module,
    require_module,
    require_symbols,
)
from megatron.core.ops.config import Backend, Operation, OpsBackendConfig

NormSelector = Callable[[bool, bool], type]


@dataclass(frozen=True)
class NormOps:
    """Resolved normal and Q/K normalization selectors."""

    norm: NormSelector
    qk_norm: NormSelector


def _apex_norm(*, required: bool, purpose: str) -> type | None:
    apex = optional_module("apex", purpose=purpose)
    if apex is None:
        if required:
            raise ImportError(f"Apex is required for {purpose}")
        return None
    return require_symbols(
        "megatron.core.fusions.fused_layer_norm", ("FusedLayerNorm",), purpose=purpose
    )[
        "FusedLayerNorm"
    ]  # type: ignore[return-value]


def _local_norm_selector() -> NormSelector:
    from megatron.core.transformer.torch_norm import WrappedTorchNorm

    apex_norm = _apex_norm(required=False, purpose=Operation.NORM.value)

    def select(rms_norm: bool, has_residual: bool) -> type:  # pylint: disable=unused-argument
        return WrappedTorchNorm if rms_norm or apex_norm is None else apex_norm

    return select


def _te_norm_selector(*, qk: bool, inference: bool = False) -> NormSelector:
    purpose = Operation.QK_NORM.value if qk else Operation.NORM.value
    require_module("transformer_engine.pytorch", purpose=purpose)
    te_norm = require_symbols(
        "megatron.core.extensions.transformer_engine", ("TENorm",), purpose=purpose
    )["TENorm"]

    if qk and dependency_version("transformer-engine") < Version("1.9.0"):
        apex_norm = _apex_norm(required=True, purpose="Q/K norm with Transformer Engine < 1.9")

        def select_apex(
            rms_norm: bool, has_residual: bool
        ) -> type:  # pylint: disable=unused-argument
            return apex_norm  # type: ignore[return-value]

        return select_apex

    class _TENormWithResidual:
        def __new__(cls, *args, **kwargs):
            return te_norm(*args, has_residual=True, **kwargs)  # type: ignore[operator]

    def select(rms_norm: bool, has_residual: bool) -> type:  # pylint: disable=unused-argument
        if has_residual and not inference:
            return _TENormWithResidual
        return te_norm  # type: ignore[return-value]

    return select


def _resolve_norm_selector(backend: Backend, *, qk: bool) -> NormSelector:
    if backend is Backend.LOCAL:
        return _local_norm_selector()
    if backend is Backend.TRANSFORMER_ENGINE:
        return _te_norm_selector(qk=qk)
    if backend is Backend.INFERENCE_OPTIMIZED:
        return _te_norm_selector(qk=qk, inference=True)
    operation = Operation.QK_NORM if qk else Operation.NORM
    raise ValueError(f"Backend '{backend.value}' does not implement {operation.value}")


def resolve_norm_ops(config: OpsBackendConfig) -> NormOps:
    """Resolve norm selectors once during provider construction."""
    return NormOps(
        norm=_resolve_norm_selector(config.backend_for(Operation.NORM), qk=False),
        qk_norm=_resolve_norm_selector(config.backend_for(Operation.QK_NORM), qk=True),
    )
