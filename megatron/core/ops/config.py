# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Configuration for choosing one backend implementation per operation."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Mapping, cast


class Backend(str, Enum):
    """Backend choices understood by the first operation-management slice."""

    LOCAL = "local"
    TRANSFORMER_ENGINE = "transformer_engine"
    INFERENCE_OPTIMIZED = "inference_optimized"
    KITCHEN = "kitchen"
    NONE = "none"


class Operation(str, Enum):
    """Operation slots selected independently by :class:`OpsBackendConfig`."""

    LINEAR = "linear"
    COLUMN_PARALLEL_LINEAR = "column_parallel_linear"
    ROW_PARALLEL_LINEAR = "row_parallel_linear"
    NORM = "norm"
    QK_NORM = "qk_norm"
    CORE_ATTENTION = "core_attention"
    ACTIVATION = "activation"
    GROUPED_MLP = "grouped_mlp"
    MOE_ROUTER = "moe_router"
    NORM_LINEAR = "norm_linear"
    FUSED_MLP = "fused_mlp"
    BIAS_DROPOUT_ADD = "bias_dropout_add"


BackendLike = Backend | str
OperationLike = Operation | str


def _as_backend(value: BackendLike) -> Backend:
    try:
        return value if isinstance(value, Backend) else Backend(value)
    except ValueError as error:
        choices = ", ".join(backend.value for backend in Backend)
        raise ValueError(
            f"Unknown operation backend '{value}'. Valid choices: {choices}"
        ) from error


def _as_operation(value: OperationLike) -> Operation:
    try:
        return value if isinstance(value, Operation) else Operation(value)
    except ValueError as error:
        choices = ", ".join(operation.value for operation in Operation)
        raise ValueError(f"Unknown operation slot '{value}'. Valid choices: {choices}") from error


def _normalized_overrides(
    overrides: Mapping[OperationLike, BackendLike] | None,
) -> dict[Operation, Backend]:
    if not overrides:
        return {}
    return {
        _as_operation(operation): _as_backend(backend) for operation, backend in overrides.items()
    }


@dataclass(frozen=True)
class OpsBackendConfig:
    """A default backend plus explicit per-operation replacements."""

    default_backend: BackendLike
    overrides: Mapping[OperationLike, BackendLike] = field(default_factory=dict)
    kitchen_attention_backend: str = "sdpa"

    def __post_init__(self) -> None:
        backend = _as_backend(self.default_backend)
        overrides = MappingProxyType(_normalized_overrides(self.overrides))
        if self.kitchen_attention_backend not in {"sdpa", "fa"}:
            raise ValueError(
                "kitchen_attention_backend must be either 'sdpa' or 'fa', "
                f"got '{self.kitchen_attention_backend}'"
            )
        object.__setattr__(self, "default_backend", backend)
        object.__setattr__(self, "overrides", overrides)

    def backend_for(self, operation: OperationLike) -> Backend:
        """Return the final backend choice for one operation."""
        normalized = _as_operation(operation)
        return self.overrides.get(normalized, self.default_backend)  # type: ignore[return-value]

    def with_overrides(
        self, overrides: Mapping[OperationLike, BackendLike] | None
    ) -> "OpsBackendConfig":
        """Return a new config with the supplied operation choices applied last."""
        if not overrides:
            return self
        merged: dict[Operation, Backend] = _normalized_overrides(self.overrides)
        merged.update(_normalized_overrides(overrides))
        return OpsBackendConfig(
            default_backend=self.default_backend,
            overrides=cast(Mapping[OperationLike, BackendLike], merged),
            kitchen_attention_backend=self.kitchen_attention_backend,
        )

    @classmethod
    def from_preset(
        cls,
        preset: BackendLike,
        *,
        overrides: Mapping[OperationLike, BackendLike] | None = None,
        use_kitchen: bool = False,
        use_kitchen_attention: bool = False,
        kitchen_attention_backend: str = "sdpa",
        use_te_op_fuser: bool = False,
        use_te_activation_func: bool = False,
    ) -> "OpsBackendConfig":
        """Expand a legacy transformer implementation into a complete operation preset."""
        default = _as_backend(preset)
        if default not in {Backend.LOCAL, Backend.TRANSFORMER_ENGINE, Backend.INFERENCE_OPTIMIZED}:
            raise ValueError(f"'{default.value}' is not a complete backend preset")

        preset_overrides: dict[Operation, Backend] = {
            Operation.ACTIVATION: Backend.NONE,
            Operation.BIAS_DROPOUT_ADD: Backend.LOCAL,
            Operation.FUSED_MLP: Backend.NONE,
            Operation.MOE_ROUTER: Backend.NONE,
        }
        if default is Backend.LOCAL:
            preset_overrides.update(
                {
                    Operation.LINEAR: Backend.NONE,
                    Operation.ACTIVATION: Backend.NONE,
                    Operation.NORM_LINEAR: Backend.NONE,
                }
            )
        elif default is Backend.INFERENCE_OPTIMIZED:
            preset_overrides.update(
                {
                    Operation.ACTIVATION: Backend.INFERENCE_OPTIMIZED,
                    Operation.MOE_ROUTER: Backend.INFERENCE_OPTIMIZED,
                }
            )

        if use_kitchen_attention and not use_kitchen:
            raise ValueError("use_kitchen_attention requires use_kitchen")
        if use_kitchen:
            if default is Backend.INFERENCE_OPTIMIZED:
                raise ValueError("Kitchen is not supported with inference_optimized")
            preset_overrides.update(
                {
                    Operation.COLUMN_PARALLEL_LINEAR: Backend.KITCHEN,
                    Operation.ROW_PARALLEL_LINEAR: Backend.KITCHEN,
                    Operation.NORM_LINEAR: Backend.KITCHEN,
                    Operation.GROUPED_MLP: Backend.KITCHEN,
                }
            )
            if use_kitchen_attention:
                preset_overrides[Operation.CORE_ATTENTION] = Backend.KITCHEN

        if use_te_op_fuser:
            preset_overrides[Operation.FUSED_MLP] = Backend.TRANSFORMER_ENGINE
        if use_te_activation_func:
            preset_overrides[Operation.ACTIVATION] = Backend.TRANSFORMER_ENGINE

        preset_overrides.update(_normalized_overrides(overrides))
        return cls(
            default_backend=default,
            overrides=cast(Mapping[OperationLike, BackendLike], preset_overrides),
            kitchen_attention_backend=kitchen_attention_backend,
        )

    @classmethod
    def from_transformer_config(
        cls,
        config: object,
        *,
        transformer_impl_override: BackendLike | None = None,
        overrides: Mapping[OperationLike, BackendLike] | None = None,
    ) -> "OpsBackendConfig":
        """Translate existing TransformerConfig fields into operation choices."""
        preset = transformer_impl_override or getattr(config, "transformer_impl")
        config_overrides = getattr(config, "op_backend_overrides", None)
        result = cls.from_preset(
            preset,
            use_kitchen=getattr(config, "use_kitchen", False),
            use_kitchen_attention=getattr(config, "use_kitchen_attention", False),
            kitchen_attention_backend=getattr(config, "kitchen_attention_backend", "sdpa"),
            use_te_op_fuser=getattr(config, "use_transformer_engine_op_fuser", False),
            use_te_activation_func=getattr(config, "use_te_activation_func", False),
            overrides=config_overrides,
        )
        return result.with_overrides(overrides)
