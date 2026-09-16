# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Validate and load only the selected GDP training or context-parallel backend."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from megatron.core.ops._backends import require

if TYPE_CHECKING:
    from megatron.core.ops.ssm.context_parallel.chunkwise import LinearAttentionCPBackend


def select_gated_delta_product(use_cutedsl: bool = False) -> Callable:
    """Return the requested callable; availability never changes the selection."""
    if use_cutedsl:
        return require(
            "gdp_attn", "chunk_gated_delta_product", needed_by="GDP (gdp_cutedsl_kernel)"
        ).chunk_gated_delta_product
    return require(
        "fla.ops.gated_delta_product", "chunk_gated_delta_product", needed_by="GDP"
    ).chunk_gated_delta_product


def select_gdp_cp_backend(
    use_cutedsl: bool = False, *, recompute_chunk_num: int = 0
) -> LinearAttentionCPBackend:
    """Construct the selected chunkwise-CP adapter.

    Each adapter module imports the FLA or CuTeDSL internals it wraps at import time and
    raises a clear ``ImportError`` when they are missing, so importing it is the check.
    """
    if use_cutedsl:
        adapter = require(
            "megatron.core.ops.ssm.context_parallel.gdp_cutedsl",
            "CuTeDSLGatedDeltaProductCPBackend",
            needed_by="GDP chunkwise context parallelism (gdp_cutedsl_kernel)",
        )
        return adapter.CuTeDSLGatedDeltaProductCPBackend(recompute_chunk_num=recompute_chunk_num)
    adapter = require(
        "megatron.core.ops.ssm.context_parallel.gdp",
        "FLAGatedDeltaProductCPBackend",
        needed_by="GDP chunkwise context parallelism",
    )
    return adapter.FLAGatedDeltaProductCPBackend()
