# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Construction-time selection for linear operation implementations."""

from __future__ import annotations

from dataclasses import dataclass

from megatron.core.ops._dependencies import require_module, require_symbols
from megatron.core.ops.config import Backend, Operation, OpsBackendConfig


@dataclass(frozen=True)
class LinearOps:
    """Resolved linear targets returned directly to ModuleSpec builders."""

    linear: type | None
    column_parallel_linear: type
    row_parallel_linear: type


def _transformer_engine_symbol(symbol: str, *, purpose: str) -> type:
    require_module("transformer_engine.pytorch", purpose=purpose)
    return require_symbols(
        "megatron.core.extensions.transformer_engine", (symbol,), purpose=purpose
    )[
        symbol
    ]  # type: ignore[return-value]


def _kitchen_symbol(symbol: str, *, purpose: str) -> type:
    require_module("nvidia_kitchen", purpose=purpose)
    return require_symbols("megatron.core.extensions.kitchen", (symbol,), purpose=purpose)[
        symbol
    ]  # type: ignore[return-value]


def _resolve_linear(backend: Backend) -> type | None:
    if backend in {Backend.NONE, Backend.LOCAL}:
        return None
    if backend in {Backend.TRANSFORMER_ENGINE, Backend.INFERENCE_OPTIMIZED}:
        return _transformer_engine_symbol("TELinear", purpose=Operation.LINEAR.value)
    raise ValueError(f"Backend '{backend.value}' does not implement {Operation.LINEAR.value}")


def _resolve_column_parallel_linear(backend: Backend) -> type:
    if backend is Backend.LOCAL:
        from megatron.core.tensor_parallel.layers import ColumnParallelLinear

        return ColumnParallelLinear
    if backend is Backend.TRANSFORMER_ENGINE:
        return _transformer_engine_symbol(
            "TEColumnParallelLinear", purpose=Operation.COLUMN_PARALLEL_LINEAR.value
        )
    if backend is Backend.INFERENCE_OPTIMIZED:
        require_module("transformer_engine.pytorch", purpose=Operation.COLUMN_PARALLEL_LINEAR.value)
        from megatron.core.tensor_parallel.inference_layers import InferenceColumnParallelLinear

        return InferenceColumnParallelLinear
    if backend is Backend.KITCHEN:
        return _kitchen_symbol(
            "KitchenColumnParallelLinear", purpose=Operation.COLUMN_PARALLEL_LINEAR.value
        )
    raise ValueError(
        f"Backend '{backend.value}' does not implement {Operation.COLUMN_PARALLEL_LINEAR.value}"
    )


def _resolve_row_parallel_linear(backend: Backend) -> type:
    if backend is Backend.LOCAL:
        from megatron.core.tensor_parallel.layers import RowParallelLinear

        return RowParallelLinear
    if backend is Backend.TRANSFORMER_ENGINE:
        return _transformer_engine_symbol(
            "TERowParallelLinear", purpose=Operation.ROW_PARALLEL_LINEAR.value
        )
    if backend is Backend.INFERENCE_OPTIMIZED:
        require_module("transformer_engine.pytorch", purpose=Operation.ROW_PARALLEL_LINEAR.value)
        from megatron.core.tensor_parallel.inference_layers import InferenceRowParallelLinear

        return InferenceRowParallelLinear
    if backend is Backend.KITCHEN:
        return _kitchen_symbol(
            "KitchenRowParallelLinear", purpose=Operation.ROW_PARALLEL_LINEAR.value
        )
    raise ValueError(
        f"Backend '{backend.value}' does not implement {Operation.ROW_PARALLEL_LINEAR.value}"
    )


def resolve_linear_ops(config: OpsBackendConfig) -> LinearOps:
    """Resolve all linear slots after config overrides have been applied."""
    return LinearOps(
        linear=_resolve_linear(config.backend_for(Operation.LINEAR)),
        column_parallel_linear=_resolve_column_parallel_linear(
            config.backend_for(Operation.COLUMN_PARALLEL_LINEAR)
        ),
        row_parallel_linear=_resolve_row_parallel_linear(
            config.backend_for(Operation.ROW_PARALLEL_LINEAR)
        ),
    )
