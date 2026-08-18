# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""The single model-facing provider for operation implementations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, cast

from megatron.core.ops.attention import AttentionOps, resolve_attention_ops
from megatron.core.ops.config import (
    Backend,
    BackendLike,
    Operation,
    OperationLike,
    OpsBackendConfig,
)
from megatron.core.ops.fusion import (
    NormLinearSpec,
    resolve_bias_dropout_add,
    resolve_fused_mlp,
    resolve_norm_linear,
)
from megatron.core.ops.linear import LinearOps, resolve_linear_ops
from megatron.core.ops.moe import MoeOps, resolve_moe_ops
from megatron.core.ops.norm import NormOps, resolve_norm_ops


@dataclass(frozen=True)
class ResolvedBackendOps:
    """Concrete targets selected once while building a provider."""

    linear: LinearOps
    norm: NormOps
    attention: AttentionOps
    moe: MoeOps
    norm_linear: NormLinearSpec
    fused_mlp: Any | None
    grouped_fused_mlp: Any | None
    bias_dropout_add: object


def _resolve_backend_ops(config: OpsBackendConfig) -> ResolvedBackendOps:
    """Resolve the final operation table after all overrides are known."""
    fused_mlp_backend = config.backend_for(Operation.FUSED_MLP)
    return ResolvedBackendOps(
        linear=resolve_linear_ops(config),
        norm=resolve_norm_ops(config),
        attention=resolve_attention_ops(config),
        moe=resolve_moe_ops(config),
        norm_linear=resolve_norm_linear(config.backend_for(Operation.NORM_LINEAR)),
        fused_mlp=resolve_fused_mlp(fused_mlp_backend),
        grouped_fused_mlp=resolve_fused_mlp(fused_mlp_backend, grouped=True),
        bias_dropout_add=resolve_bias_dropout_add(config.backend_for(Operation.BIAS_DROPOUT_ADD)),
    )


class BackendSpecProvider:
    """Provide concrete operation targets selected by one immutable config.

    This is the only provider implementation. Backends are config values, not provider classes.
    All selection and optional-dependency checks happen while this object is constructed; its
    methods only return already resolved targets to ModuleSpec builders.
    """

    def __init__(self, config: OpsBackendConfig):
        self.config = config
        self._ops = _resolve_backend_ops(config)

    def backend_for(self, operation: OperationLike) -> Backend:
        """Return the configured backend for an operation slot."""
        return self.config.backend_for(operation)

    def linear(self) -> type:
        """Return the non-parallel linear target."""
        if self._ops.linear.linear is None:
            raise NotImplementedError(
                f"Backend '{self.backend_for(Operation.LINEAR).value}' does not provide linear()"
            )
        return self._ops.linear.linear

    def column_parallel_linear(self) -> type:
        """Return the column-parallel linear target."""
        return self._ops.linear.column_parallel_linear

    def row_parallel_linear(self) -> type:
        """Return the row-parallel linear target."""
        return self._ops.linear.row_parallel_linear

    def norm_linear(self) -> NormLinearSpec:
        """Return the fused norm + column-parallel-linear selection."""
        return self._ops.norm_linear

    def fuse_layernorm_and_linear(self) -> bool:
        """Compatibility query derived from the norm-linear fusion slot."""
        return self._ops.norm_linear.fuses_norm

    def column_parallel_layer_norm_linear(self) -> type | None:
        """Compatibility accessor for the fused norm-linear target."""
        return self._ops.norm_linear.linear

    def layer_norm(
        self, rms_norm: bool = False, for_qk: bool = False, has_residual: bool = False
    ) -> type:
        """Return the selected normalization builder."""
        selector = self._ops.norm.qk_norm if for_qk else self._ops.norm.norm
        return selector(rms_norm, has_residual)

    def core_attention(self) -> type:
        """Return the selected core-attention target."""
        return self._ops.attention.core_attention

    def grouped_mlp_modules(self, moe_use_grouped_gemm: bool) -> object:
        """Return selected expert modules."""
        return self._ops.moe.grouped_mlp_modules(moe_use_grouped_gemm)

    def activation_func(self) -> object | None:
        """Return the selected activation target."""
        return self._ops.moe.activation_func

    def moe_router(self) -> type | None:
        """Return an explicit router override, or ``None`` for the training default."""
        return self._ops.moe.router

    def fused_mlp(self, *, grouped: bool = False) -> Any | None:
        """Return the selected fused MLP target."""
        return self._ops.grouped_fused_mlp if grouped else self._ops.fused_mlp

    def bias_dropout_add(self):
        """Return the selected bias-dropout-add builder."""
        return self._ops.bias_dropout_add


def _config_from_value(
    config_or_preset: OpsBackendConfig | BackendLike | object,
    *,
    transformer_impl_override: BackendLike | None,
    overrides: Mapping[OperationLike, BackendLike] | None,
    use_kitchen: bool | None,
    use_kitchen_attention: bool | None,
    kitchen_attention_backend: str | None,
    use_te_op_fuser: bool | None,
    use_te_activation_func: bool | None,
) -> OpsBackendConfig:
    if isinstance(config_or_preset, OpsBackendConfig):
        if any(
            value is not None
            for value in (
                transformer_impl_override,
                use_kitchen,
                use_kitchen_attention,
                kitchen_attention_backend,
                use_te_op_fuser,
                use_te_activation_func,
            )
        ):
            raise ValueError("Legacy backend options cannot be combined with OpsBackendConfig")
        return config_or_preset.with_overrides(overrides)

    if isinstance(config_or_preset, (Backend, str)):
        return OpsBackendConfig.from_preset(
            transformer_impl_override or config_or_preset,
            overrides=overrides,
            use_kitchen=False if use_kitchen is None else use_kitchen,
            use_kitchen_attention=(
                False if use_kitchen_attention is None else use_kitchen_attention
            ),
            kitchen_attention_backend=kitchen_attention_backend or "sdpa",
            use_te_op_fuser=False if use_te_op_fuser is None else use_te_op_fuser,
            use_te_activation_func=(
                False if use_te_activation_func is None else use_te_activation_func
            ),
        )

    config = config_or_preset
    config_overrides = getattr(config, "op_backend_overrides", None)
    resolved = OpsBackendConfig.from_preset(
        transformer_impl_override or getattr(config, "transformer_impl"),
        overrides=cast(Mapping[OperationLike, BackendLike] | None, config_overrides),
        use_kitchen=getattr(config, "use_kitchen", False) if use_kitchen is None else use_kitchen,
        use_kitchen_attention=(
            getattr(config, "use_kitchen_attention", False)
            if use_kitchen_attention is None
            else use_kitchen_attention
        ),
        kitchen_attention_backend=(
            getattr(config, "kitchen_attention_backend", "sdpa")
            if kitchen_attention_backend is None
            else kitchen_attention_backend
        ),
        use_te_op_fuser=(
            getattr(config, "use_transformer_engine_op_fuser", False)
            if use_te_op_fuser is None
            else use_te_op_fuser
        ),
        use_te_activation_func=(
            getattr(config, "use_te_activation_func", False)
            if use_te_activation_func is None
            else use_te_activation_func
        ),
    )
    return resolved.with_overrides(overrides)


def get_backend_spec_provider(
    config_or_preset: OpsBackendConfig | BackendLike | object,
    *,
    transformer_impl_override: BackendLike | None = None,
    overrides: Mapping[OperationLike, BackendLike] | None = None,
    use_kitchen: bool | None = None,
    use_kitchen_attention: bool | None = None,
    kitchen_attention_backend: str | None = None,
    use_te_op_fuser: bool | None = None,
    use_te_activation_func: bool | None = None,
) -> BackendSpecProvider:
    """Build the sole provider after resolving config and dependencies once."""
    config = _config_from_value(
        config_or_preset,
        transformer_impl_override=transformer_impl_override,
        overrides=overrides,
        use_kitchen=use_kitchen,
        use_kitchen_attention=use_kitchen_attention,
        kitchen_attention_backend=kitchen_attention_backend,
        use_te_op_fuser=use_te_op_fuser,
        use_te_activation_func=use_te_activation_func,
    )
    return BackendSpecProvider(config)


def get_backend(
    preset: BackendLike, *, overrides: Mapping[OperationLike, BackendLike] | None = None
) -> BackendSpecProvider:
    """Build the sole provider from a named preset."""
    return get_backend_spec_provider(preset, overrides=overrides)
