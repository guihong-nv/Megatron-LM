# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Construction-time selection of the existing GDN and GDN2 recurrences."""

from dataclasses import dataclass
from typing import Callable, Literal, cast

from megatron.core.ops.kernel_metadata import DeterminismPolicy, validate_kernel
from megatron.core.ops.ssm.gated_delta import GatedDeltaRuleInterface
from megatron.core.ops.ssm.gated_delta.kernel_metadata import (
    FLA_CONV_UPDATE,
    GDN2_FLA,
    GDN2_TORCH,
    GDN_FLA,
    GDN_RECURRENT,
    GDN_TORCH,
)


def select_gated_delta_rule(
    variant: Literal["gdn", "gdn2"], deterministic: bool = False
) -> GatedDeltaRuleInterface:
    """Return the original reference or FLA callable without wrapping its forward.

    The mixers L2-normalize q/k themselves, so the reference kernels' optional in-kernel
    normalization (and its FLA dependency) is never selected here.
    """
    if variant == "gdn":
        if deterministic:
            validate_kernel(GDN_TORCH, determinism=DeterminismPolicy.WARN)
            from megatron.core.ops.ssm.gated_delta.reference import torch_chunk_gated_delta_rule

            # The shared protocol cannot express the variant's required gate keywords.
            return cast(GatedDeltaRuleInterface, torch_chunk_gated_delta_rule)
        validate_kernel(GDN_FLA)
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule

        return chunk_gated_delta_rule
    if variant == "gdn2":
        if deterministic:
            validate_kernel(GDN2_TORCH, determinism=DeterminismPolicy.WARN)
            from megatron.core.ops.ssm.gated_delta.reference_gdn2 import torch_chunk_gdn2

            return cast(GatedDeltaRuleInterface, torch_chunk_gdn2)
        validate_kernel(GDN2_FLA)
        from fla.ops.gdn2.chunk import chunk_gdn2

        return chunk_gdn2
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


def select_gated_delta_inference_kernels(
    deterministic: bool = False,
) -> GatedDeltaInferenceKernels:
    """Bind the FLA decode/prefill kernels GDN dynamic inference has always used."""
    policy = DeterminismPolicy.WARN if deterministic else DeterminismPolicy.IGNORE
    validate_kernel(GDN_FLA, determinism=policy)
    validate_kernel(GDN_RECURRENT, determinism=policy)
    validate_kernel(FLA_CONV_UPDATE, determinism=policy)
    from fla.modules.convolution import causal_conv1d_update
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule

    return GatedDeltaInferenceKernels(
        chunk=chunk_gated_delta_rule,
        recurrent=fused_recurrent_gated_delta_rule,
        conv_update=causal_conv1d_update,
    )
