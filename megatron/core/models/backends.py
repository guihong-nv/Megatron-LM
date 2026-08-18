# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
from __future__ import annotations

import warnings
from abc import abstractmethod
from functools import partial
from typing import Literal, Optional, Protocol, cast, runtime_checkable

from megatron.core.extensions.transformer_engine import (
    HAVE_TE,
    TEColumnParallelGroupedLinear,
    TERowParallelGroupedLinear,
)
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer.dot_product_attention import DotProductAttention
from megatron.core.transformer.mlp import MLPSubmodules, TEActivationFunctionBuilder
from megatron.core.transformer.moe.experts import (
    GroupedMLPSubmodules,
    InferenceGroupedMLP,
    SequentialMLP,
)
from megatron.core.transformer.moe.moe_layer import ExpertsBuilder
from megatron.core.transformer.torch_norm import LayerNormBuilder, WrappedTorchNorm
from megatron.core.typed_torch import not_none
from megatron.core.utils import is_te_min_version

try:
    import apex  # pylint: disable=unused-import

    from megatron.core.fusions.fused_layer_norm import FusedLayerNorm

    HAVE_APEX = True
    LNImpl = FusedLayerNorm
except ImportError:
    warnings.warn("Apex is not installed. Falling back to Torch Norm")
    FusedLayerNorm = None
    HAVE_APEX = False
    LNImpl = WrappedTorchNorm

from megatron.core.extensions.transformer_engine import (
    TEActivationOp,
    TEDotProductAttention,
    TELinear,
    TENorm,
)
from megatron.core.tensor_parallel.inference_layers import (
    InferenceColumnParallelLinear,
    InferenceLayerNormColumnParallelLinear,
    InferenceRowParallelLinear,
)
from megatron.core.utils import is_te_min_version

BackendName = Literal["local", "transformer_engine", "inference_optimized"]


class _BackendSpecProviderConfig(Protocol):
    """Configuration fields needed to construct a backend provider."""

    transformer_impl: BackendName
    use_kitchen: bool
    use_kitchen_attention: bool
    kitchen_attention_backend: Literal["sdpa", "fa"]


@runtime_checkable
class BackendSpecProvider(Protocol):
    """A protocol for providing the submodules used in Spec building."""

    @abstractmethod
    def linear(self) -> type:
        """Which non-parallel linear module the backend uses."""
        ...

    @abstractmethod
    def column_parallel_linear(self) -> type:
        """Which column parallel linear module the backend uses"""
        ...

    @abstractmethod
    def row_parallel_linear(self) -> type:
        """Which row parallel linear module the backend uses"""
        ...

    @abstractmethod
    def fuse_layernorm_and_linear(self) -> bool:
        """Does the backend support a single module for layernorm and linear"""
        ...

    @abstractmethod
    def column_parallel_layer_norm_linear(self) -> Optional[type]:
        """Which module for sequential layernorm and linear"""
        ...

    @abstractmethod
    def layer_norm(
        self, rms_norm: bool = False, for_qk: bool = False, has_residual: bool = False
    ) -> LayerNormBuilder:
        """Which module for layernorm"""
        ...

    @abstractmethod
    def core_attention(self) -> type:
        """Which module to use for attention"""
        ...

    @abstractmethod
    def grouped_mlp_modules(self, moe_use_grouped_gemm: bool) -> ExpertsBuilder:
        """Which module and submodules to use for grouped mlp"""
        ...

    @abstractmethod
    def activation_func(self) -> TEActivationFunctionBuilder | None:
        """Which module to use for activation function"""
        ...


class LocalSpecProvider(BackendSpecProvider):
    """A protocol for providing Local submodules used in Spec building."""

    def linear(self) -> type:
        """Report that local duplicated-linear semantics are not defined yet.

        Do not silently substitute a tensor-parallel linear implementation with different
        parameter ownership and communication behavior.
        """
        raise NotImplementedError(f"{type(self).__name__} does not provide linear()")

    def column_parallel_linear(self) -> type:
        """Which column parallel linear module the backend uses"""
        return ColumnParallelLinear

    def row_parallel_linear(self) -> type[RowParallelLinear]:
        """Which row parallel linear module the backend uses"""
        return RowParallelLinear

    def fuse_layernorm_and_linear(self) -> bool:
        """Does the backend choose a single module for layernorm and linear"""
        return False

    def column_parallel_layer_norm_linear(self) -> Optional[type]:
        """Which module for sequential layernorm and linear"""
        return None

    def layer_norm(
        self, rms_norm: bool = False, for_qk: bool = False, has_residual: bool = False
    ) -> LayerNormBuilder:
        """Which module to use for layer norm"""
        return WrappedTorchNorm if rms_norm else LNImpl

    def core_attention(self) -> type:
        """Which module to use for attention"""
        return DotProductAttention

    def grouped_mlp_modules(self, moe_use_grouped_gemm: bool) -> ExpertsBuilder:
        """Which module and submodules to use for grouped mlp"""
        return partial(
            SequentialMLP,
            submodules=MLPSubmodules(
                linear_fc1=ColumnParallelLinear,
                linear_fc2=RowParallelLinear,
                activation_func=self.activation_func(),
            ),
        )

    def activation_func(self) -> TEActivationFunctionBuilder | None:
        """Which module to use for activation function"""
        return None


class InferenceSpecProvider(BackendSpecProvider):
    """A protocol for providing the submodules used in Spec building."""

    def linear(self) -> type:
        """Which linear module TE backend uses"""
        return TELinear

    def column_parallel_linear(self) -> type:
        """Which column parallel linear module TE backend uses"""
        return InferenceColumnParallelLinear

    def row_parallel_linear(self) -> type[InferenceRowParallelLinear]:
        """Which row parallel linear module Inference backend uses"""
        return InferenceRowParallelLinear

    def fuse_layernorm_and_linear(self) -> bool:
        """TE backend chooses a single module for layernorm and linear"""
        return True

    def column_parallel_layer_norm_linear(self) -> type[InferenceLayerNormColumnParallelLinear]:
        """Which module for sequential layernorm and linear"""
        return InferenceLayerNormColumnParallelLinear

    def layer_norm(
        self, rms_norm: bool = False, for_qk: bool = False, has_residual: bool = False
    ) -> LayerNormBuilder:
        """Which module to use for layer norm"""
        if for_qk and not is_te_min_version("1.9.0"):
            # TENorm significantly harms convergence when used
            # for QKLayerNorm if TE Version < 1.9;
            # we instead use the Apex implementation.
            return not_none(FusedLayerNorm)
        return TENorm

    def core_attention(self) -> type[TEDotProductAttention]:
        """Which module to use for attention"""
        return TEDotProductAttention

    def activation_func(self) -> TEActivationFunctionBuilder | None:
        """Which module to use for activation function"""
        # transformer_engine.BasicOperation.forward has an overly permissive return type, but by
        # design these classes always meet the interface.
        return cast(TEActivationFunctionBuilder, TEActivationOp)

    def grouped_mlp_modules(self, moe_use_grouped_gemm: bool) -> ExpertsBuilder:
        """Which module and submodules to use for grouped mlp"""
        return partial(
            InferenceGroupedMLP,
            submodules=GroupedMLPSubmodules(
                linear_fc1=TEColumnParallelGroupedLinear,
                linear_fc2=TERowParallelGroupedLinear,
                activation_func=self.activation_func(),
            ),
        )


def _validate_backend_spec_provider(provider: object) -> BackendSpecProvider:
    """Validate a structurally implemented provider at the construction boundary."""
    provider_methods = {
        name
        for name, value in BackendSpecProvider.__dict__.items()
        if not name.startswith("_") and callable(value)
    }
    missing_methods = sorted(
        name for name in provider_methods if not callable(getattr(provider, name, None))
    )
    if missing_methods:
        raise TypeError(
            f"{type(provider).__name__} does not implement BackendSpecProvider; "
            f"missing methods: {', '.join(missing_methods)}"
        )
    return cast(BackendSpecProvider, provider)


def get_backend_spec_provider(
    config_or_transformer_impl: _BackendSpecProviderConfig | BackendName,
    *,
    transformer_impl_override: BackendName | None = None,
    use_kitchen: bool | None = None,
    use_kitchen_attention: bool | None = None,
    kitchen_attention_backend: Literal["sdpa", "fa"] | None = None,
) -> BackendSpecProvider:
    """Build the backend provider selected by existing configuration values.

    This function is the construction-time boundary for built-in providers. It deliberately
    returns concrete classes and builders; it does not add a resolver or proxy to the forward
    path. Callers that have a transformer config should pass it directly. The string form keeps
    compatibility with static specs that currently defer optional-dependency failures until their
    selected modules are built.
    """
    config_based_selection = not isinstance(config_or_transformer_impl, str)
    if isinstance(config_or_transformer_impl, str):
        transformer_impl = transformer_impl_override or config_or_transformer_impl
        use_kitchen = False if use_kitchen is None else use_kitchen
        use_kitchen_attention = False if use_kitchen_attention is None else use_kitchen_attention
        kitchen_attention_backend = kitchen_attention_backend or "sdpa"
    else:
        config = config_or_transformer_impl
        transformer_impl = transformer_impl_override or config.transformer_impl
        use_kitchen = config.use_kitchen if use_kitchen is None else use_kitchen
        use_kitchen_attention = (
            config.use_kitchen_attention if use_kitchen_attention is None else use_kitchen_attention
        )
        kitchen_attention_backend = (
            config.kitchen_attention_backend
            if kitchen_attention_backend is None
            else kitchen_attention_backend
        )

    if transformer_impl == "transformer_engine":
        if config_based_selection and not HAVE_TE:
            raise ImportError(
                "Transformer Engine was requested but is not installed. "
                "Install it with `pip install transformer-engine`."
            )
        from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider

        provider: object = TESpecProvider()
    elif transformer_impl == "inference_optimized":
        provider = InferenceSpecProvider()
    elif transformer_impl == "local":
        provider = LocalSpecProvider()
    else:
        raise ValueError(f"unknown transformer_impl='{transformer_impl}'")

    if use_kitchen:
        if transformer_impl == "inference_optimized":
            raise ValueError("Kitchen is not supported with inference_optimized")

        from megatron.core.extensions.kitchen import HAVE_KITCHEN, KitchenSpecProvider

        if not HAVE_KITCHEN:
            raise ImportError("Kitchen was requested but nvidia-kitchen is not installed")
        provider = KitchenSpecProvider(
            fallback=provider,
            use_kitchen_attention=use_kitchen_attention,
            kitchen_attention_backend=kitchen_attention_backend,
        )

    return _validate_backend_spec_provider(provider)


def get_backend(transformer_impl: BackendName) -> BackendSpecProvider:
    """Compatibility wrapper for the former base-provider factory."""
    return get_backend_spec_provider(transformer_impl)
