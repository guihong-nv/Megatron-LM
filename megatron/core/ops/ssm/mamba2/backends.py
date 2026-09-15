# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Construction-time loading of Mamba scan kernels and their capabilities."""

import inspect
from dataclasses import dataclass
from typing import Callable

from megatron.core.ops._backends import is_available, require


@dataclass(frozen=True)
class MambaKernels:
    """Direct targets for ordinary scan and optional memory-efficient training."""

    scan: Callable
    split_scan: Callable | None
    has_state_dtype: bool
    causal_conv1d: Callable | None


def select_mamba_kernels(use_mem_eff_path: bool) -> MambaKernels:
    """Load the scan needed by prefill and the explicitly enabled fused training scan."""
    ssd = require(
        "mamba_ssm.ops.triton.ssd_combined", "mamba_chunk_scan_combined", needed_by="Mamba2"
    )
    targets = [ssd.mamba_chunk_scan_combined]
    split_scan = None
    if use_mem_eff_path:
        require(
            "mamba_ssm.ops.triton.ssd_combined",
            "mamba_split_conv1d_scan_combined",
            needed_by="Mamba2 (use_mamba_mem_eff_path)",
        )
        split_scan = ssd.mamba_split_conv1d_scan_combined
        targets.append(split_scan)

    # Mamba's ordinary path has always used Torch convolution when causal-conv1d is absent.
    # The fused path needs it, and an installed-but-broken package is an error in both cases,
    # not a reason to fall back to a different implementation.
    conv = None
    if use_mem_eff_path or is_available("causal_conv1d"):
        needed_by = "Mamba2 convolution" + (
            " (use_mamba_mem_eff_path)" if use_mem_eff_path else ""
        )
        conv = require("causal_conv1d", "causal_conv1d_fn", needed_by=needed_by).causal_conv1d_fn
    return MambaKernels(
        scan=ssd.mamba_chunk_scan_combined,
        split_scan=split_scan,
        has_state_dtype=all(
            "state_dtype" in inspect.signature(target).parameters for target in targets
        ),
        causal_conv1d=conv,
    )
