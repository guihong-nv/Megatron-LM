# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Tests that the inference-optimized backend wires InferenceTopKRouter."""

from unittest.mock import MagicMock

import pytest

from megatron.core.models.gpt.moe_module_specs import get_moe_module_spec_for_backend
from megatron.core.ops import BackendSpecProvider, get_backend
from megatron.core.transformer.moe.moe_layer import MoESubmodules
from megatron.core.transformer.moe.router import InferenceTopKRouter, TopKRouter


def _router_of(spec):
    """Return the router builder from a get_moe_module_spec_for_backend() result."""
    submodules = spec.keywords["submodules"]
    assert isinstance(submodules, MoESubmodules)
    return submodules.router


class TestMoeModuleSpecRouter:
    @pytest.mark.parametrize("moe_grouped_gemm", [True, False])
    def test_inference_backend_wires_inference_router(self, moe_grouped_gemm):
        """The MoE consumer uses the router already selected by its provider."""
        backend = MagicMock(spec=BackendSpecProvider)
        backend.moe_router.return_value = InferenceTopKRouter
        spec = get_moe_module_spec_for_backend(
            backend, num_experts=8, moe_grouped_gemm=moe_grouped_gemm
        )
        assert _router_of(spec) is InferenceTopKRouter

    @pytest.mark.parametrize("moe_grouped_gemm", [True, False])
    def test_non_inference_backend_uses_default_router(self, moe_grouped_gemm):
        """Non-inference backends keep the MoESubmodules default (training router)."""
        spec = get_moe_module_spec_for_backend(
            get_backend("local"), num_experts=8, moe_grouped_gemm=moe_grouped_gemm
        )
        # No router override -> dataclass default.
        assert _router_of(spec) is TopKRouter
