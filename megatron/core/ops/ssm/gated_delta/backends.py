# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Construction-time selection of the existing GDN and GDN2 recurrences."""

from dataclasses import dataclass
from typing import Callable, Literal, cast

from megatron.core.ops._backends import require
from megatron.core.ops.ssm.gated_delta import GatedDeltaRuleInterface


def select_gated_delta_rule(
    variant: Literal["gdn", "gdn2"], deterministic: bool = False
) -> GatedDeltaRuleInterface:
    """Return the original reference or FLA callable without wrapping its forward.

    Deterministic mode selects the Torch reference because the FLA kernels are not
    bit-reproducible. The mixers L2-normalize q/k themselves, so the reference kernels'
    optional in-kernel normalization (and its FLA dependency) is never selected here.
    """
    if variant == "gdn":
        if deterministic:
            from megatron.core.ops.ssm.gated_delta.reference import torch_chunk_gated_delta_rule

            # The shared protocol cannot express the variant's required gate keywords.
            return cast(GatedDeltaRuleInterface, torch_chunk_gated_delta_rule)
        return require(
            "fla.ops.gated_delta_rule", "chunk_gated_delta_rule", needed_by="GDN"
        ).chunk_gated_delta_rule
    if variant == "gdn2":
        if deterministic:
            from megatron.core.ops.ssm.gated_delta.reference_gdn2 import torch_chunk_gdn2

            return cast(GatedDeltaRuleInterface, torch_chunk_gdn2)
        return require(
            "fla.ops.gdn2.chunk",
            "chunk_gdn2",
            min_version="0.5.1",
            dist="fla-core",
            needed_by="GDN2",
        ).chunk_gdn2
    raise ValueError(f"Unknown gated delta variant: {variant!r}")


@dataclass(frozen=True)
class GatedDeltaInferenceKernels:
    """The FLA phase family GDN dynamic inference binds once: decode and packed prefill.

    These take the mixer's ``A_log``/``dt_bias`` and fuse the gates, so they are not
    interchangeable with the training recurrence and are bound separately from it.
    """

    chunk: Callable
    recurrent: Callable
    conv_update: Callable


def select_gated_delta_inference_kernels() -> GatedDeltaInferenceKernels:
    """Bind the FLA decode/prefill kernels GDN dynamic inference has always used."""
    rule = require(
        "fla.ops.gated_delta_rule",
        "chunk_gated_delta_rule",
        "fused_recurrent_gated_delta_rule",
        needed_by="GDN dynamic inference",
    )
    convolution = require(
        "fla.modules.convolution", "causal_conv1d_update", needed_by="GDN dynamic inference"
    )
    return GatedDeltaInferenceKernels(
        chunk=rule.chunk_gated_delta_rule,
        recurrent=rule.fused_recurrent_gated_delta_rule,
        conv_update=convolution.causal_conv1d_update,
    )
