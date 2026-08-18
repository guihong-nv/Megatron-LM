# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Selection and compatibility export for bias-dropout-add."""

from __future__ import annotations

from collections.abc import Callable

from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.ops.config import Backend, Operation


def resolve_bias_dropout_add(backend: Backend) -> Callable:
    """Return the existing Megatron bias-dropout-add resolver."""

    if backend is Backend.LOCAL:
        return get_bias_dropout_add
    raise ValueError(f"unsupported {Operation.BIAS_DROPOUT_ADD.value} backend: {backend!r}")


__all__ = ["get_bias_dropout_add", "resolve_bias_dropout_add"]
