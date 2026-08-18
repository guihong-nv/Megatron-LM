# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Construction-time selection for activation, grouped MLP, and MoE routing."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from functools import partial
from typing import Callable

from packaging.version import Version

from megatron.core.ops._dependencies import dependency_version, require_module, require_symbols
from megatron.core.ops.config import Backend, Operation, OpsBackendConfig

GroupedMlpSelector = Callable[[bool], object]


@dataclass(frozen=True)
class MoeOps:
    """Resolved activation, expert, and router targets."""

    activation_func: object | None
    grouped_mlp_modules: GroupedMlpSelector
    router: type | None


def _resolve_activation(backend: Backend) -> object | None:
    if backend in {Backend.NONE, Backend.LOCAL}:
        return None
    if backend in {Backend.TRANSFORMER_ENGINE, Backend.INFERENCE_OPTIMIZED}:
        require_module("transformer_engine.pytorch", purpose=Operation.ACTIVATION.value)
        return require_symbols(
            "megatron.core.extensions.transformer_engine",
            ("TEActivationOp",),
            purpose=Operation.ACTIVATION.value,
        )["TEActivationOp"]
    raise ValueError(f"Backend '{backend.value}' does not implement {Operation.ACTIVATION.value}")


def _local_grouped_mlp(activation_func: object | None) -> GroupedMlpSelector:
    from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
    from megatron.core.transformer.mlp import MLPSubmodules
    from megatron.core.transformer.moe.experts import SequentialMLP

    def select(moe_use_grouped_gemm: bool) -> object:  # pylint: disable=unused-argument
        return partial(
            SequentialMLP,
            submodules=MLPSubmodules(
                linear_fc1=ColumnParallelLinear,
                linear_fc2=RowParallelLinear,
                activation_func=activation_func,
            ),
        )

    return select


def _te_grouped_mlp(activation_func: object | None) -> GroupedMlpSelector:
    require_module("transformer_engine.pytorch", purpose=Operation.GROUPED_MLP.value)
    te_extension = require_module(
        "megatron.core.extensions.transformer_engine", purpose=Operation.GROUPED_MLP.value
    )
    from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
    from megatron.core.transformer.mlp import MLPSubmodules
    from megatron.core.transformer.moe.experts import (
        GroupedMLPSubmodules,
        SequentialMLP,
        TEGroupedMLP,
    )

    te_column = getattr(te_extension, "TEColumnParallelLinear")
    te_row = getattr(te_extension, "TERowParallelLinear")
    te_grouped_column = getattr(te_extension, "TEColumnParallelGroupedLinear", None)
    te_grouped_row = getattr(te_extension, "TERowParallelGroupedLinear", None)
    version = dependency_version("transformer-engine")

    def select(moe_use_grouped_gemm: bool) -> object:
        if moe_use_grouped_gemm and te_grouped_column is not None and te_grouped_row is not None:
            return partial(
                TEGroupedMLP,
                submodules=GroupedMLPSubmodules(
                    linear_fc1=te_grouped_column,
                    linear_fc2=te_grouped_row,
                    activation_func=activation_func,
                ),
            )
        if version < Version("1.7.0.dev0"):
            warnings.warn(
                "Transformer Engine before 1.7 does not provide MoE experts; "
                "using local linear implementations instead."
            )
            linear_fc1, linear_fc2 = ColumnParallelLinear, RowParallelLinear
        else:
            linear_fc1, linear_fc2 = te_column, te_row
        return partial(
            SequentialMLP,
            submodules=MLPSubmodules(
                linear_fc1=linear_fc1, linear_fc2=linear_fc2, activation_func=activation_func
            ),
        )

    return select


def _inference_grouped_mlp(activation_func: object | None) -> GroupedMlpSelector:
    require_module("transformer_engine.pytorch", purpose=Operation.GROUPED_MLP.value)
    symbols = require_symbols(
        "megatron.core.extensions.transformer_engine",
        ("TEColumnParallelGroupedLinear", "TERowParallelGroupedLinear"),
        purpose=Operation.GROUPED_MLP.value,
    )
    from megatron.core.transformer.moe.experts import GroupedMLPSubmodules, InferenceGroupedMLP

    def select(moe_use_grouped_gemm: bool) -> object:  # pylint: disable=unused-argument
        return partial(
            InferenceGroupedMLP,
            submodules=GroupedMLPSubmodules(
                linear_fc1=symbols["TEColumnParallelGroupedLinear"],
                linear_fc2=symbols["TERowParallelGroupedLinear"],
                activation_func=activation_func,
            ),
        )

    return select


def _kitchen_grouped_mlp(activation_func: object | None) -> GroupedMlpSelector:
    require_module("nvidia_kitchen", purpose=Operation.GROUPED_MLP.value)
    symbols = require_symbols(
        "megatron.core.extensions.kitchen",
        (
            "KitchenColumnParallelGroupedLinear",
            "KitchenColumnParallelLinear",
            "KitchenRowParallelGroupedLinear",
            "KitchenRowParallelLinear",
        ),
        purpose=Operation.GROUPED_MLP.value,
    )
    from megatron.core.transformer.mlp import MLPSubmodules
    from megatron.core.transformer.moe.experts import (
        GroupedMLPSubmodules,
        SequentialMLP,
        TEGroupedMLP,
    )

    def select(moe_use_grouped_gemm: bool) -> object:
        if moe_use_grouped_gemm:
            return partial(
                TEGroupedMLP,
                submodules=GroupedMLPSubmodules(
                    linear_fc1=symbols["KitchenColumnParallelGroupedLinear"],
                    linear_fc2=symbols["KitchenRowParallelGroupedLinear"],
                    activation_func=activation_func,
                ),
            )
        return partial(
            SequentialMLP,
            submodules=MLPSubmodules(
                linear_fc1=symbols["KitchenColumnParallelLinear"],
                linear_fc2=symbols["KitchenRowParallelLinear"],
                activation_func=activation_func,
            ),
        )

    return select


def _resolve_grouped_mlp(backend: Backend, activation_func: object | None) -> GroupedMlpSelector:
    if backend is Backend.LOCAL:
        return _local_grouped_mlp(activation_func)
    if backend is Backend.TRANSFORMER_ENGINE:
        return _te_grouped_mlp(activation_func)
    if backend is Backend.INFERENCE_OPTIMIZED:
        return _inference_grouped_mlp(activation_func)
    if backend is Backend.KITCHEN:
        return _kitchen_grouped_mlp(activation_func)
    raise ValueError(f"Backend '{backend.value}' does not implement {Operation.GROUPED_MLP.value}")


def _resolve_router(backend: Backend) -> type | None:
    if backend in {Backend.NONE, Backend.LOCAL, Backend.TRANSFORMER_ENGINE}:
        return None
    if backend is Backend.INFERENCE_OPTIMIZED:
        from megatron.core.transformer.moe.router import InferenceTopKRouter

        return InferenceTopKRouter
    raise ValueError(f"Backend '{backend.value}' does not implement {Operation.MOE_ROUTER.value}")


def resolve_moe_ops(config: OpsBackendConfig) -> MoeOps:
    """Resolve MoE-related operation slots once during provider construction."""
    activation_func = _resolve_activation(config.backend_for(Operation.ACTIVATION))
    return MoeOps(
        activation_func=activation_func,
        grouped_mlp_modules=_resolve_grouped_mlp(
            config.backend_for(Operation.GROUPED_MLP), activation_func
        ),
        router=_resolve_router(config.backend_for(Operation.MOE_ROUTER)),
    )
