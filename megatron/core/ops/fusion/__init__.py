# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Fused-operation contracts and construction-time resolvers."""

from .bias_dropout_add import get_bias_dropout_add, resolve_bias_dropout_add
from .fused_mlp import resolve_fused_mlp
from .norm_linear import NormLinearSpec, resolve_norm_linear

__all__ = [
    "NormLinearSpec",
    "get_bias_dropout_add",
    "resolve_bias_dropout_add",
    "resolve_fused_mlp",
    "resolve_norm_linear",
]
