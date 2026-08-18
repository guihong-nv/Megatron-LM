# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Tests for the construction-time BackendSpecProvider boundary."""

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest

import megatron.core.models.backends as backends
from megatron.core.models.backends import (
    BackendSpecProvider,
    LocalSpecProvider,
    get_backend,
    get_backend_spec_provider,
)
from megatron.core.transformer.torch_norm import WrappedTorchNorm


def test_local_norm_selection_does_not_change_later_calls(monkeypatch):
    """Selecting RMSNorm must not mutate the default local LayerNorm target."""
    layer_norm_target = object()
    monkeypatch.setattr(backends, "LNImpl", layer_norm_target)

    provider = LocalSpecProvider()

    assert provider.layer_norm(rms_norm=True) is WrappedTorchNorm
    assert provider.layer_norm(rms_norm=False) is layer_norm_target


def test_local_linear_fails_clearly_until_a_compatible_target_is_defined():
    """Do not silently map the non-parallel linear slot to a TP linear implementation."""
    with pytest.raises(NotImplementedError, match=r"LocalSpecProvider does not provide linear\(\)"):
        LocalSpecProvider().linear()


def test_local_factory_and_compatibility_wrapper_select_the_same_provider():
    """The old factory name remains a behavior-preserving compatibility wrapper."""
    provider = get_backend_spec_provider("local")

    assert isinstance(provider, LocalSpecProvider)
    assert isinstance(provider, BackendSpecProvider)
    assert isinstance(get_backend("local"), LocalSpecProvider)


def test_config_factory_reads_base_and_composition_settings():
    """The canonical form accepts the existing transformer configuration directly."""
    config = SimpleNamespace(
        transformer_impl="local",
        use_kitchen=False,
        use_kitchen_attention=False,
        kitchen_attention_backend="sdpa",
    )

    assert isinstance(get_backend_spec_provider(config), LocalSpecProvider)


def test_transformer_engine_provider_is_imported_lazily(monkeypatch):
    """TE provider construction stays behind the factory's lazy import boundary."""
    provider = MagicMock(spec=BackendSpecProvider)
    module = ModuleType("megatron.core.extensions.transformer_engine_spec_provider")
    module.TESpecProvider = MagicMock(return_value=provider)
    monkeypatch.setattr(backends, "HAVE_TE", False)
    monkeypatch.setitem(sys.modules, module.__name__, module)

    assert get_backend_spec_provider("transformer_engine") is provider
    module.TESpecProvider.assert_called_once_with()


def test_missing_transformer_engine_fails_at_construction(monkeypatch):
    """A named TE backend must fail before a spec can store placeholder targets."""
    monkeypatch.setattr(backends, "HAVE_TE", False)
    config = SimpleNamespace(
        transformer_impl="transformer_engine",
        use_kitchen=False,
        use_kitchen_attention=False,
        kitchen_attention_backend="sdpa",
    )

    with pytest.raises(ImportError, match="Transformer Engine was requested but is not installed"):
        get_backend_spec_provider(config)


def test_incomplete_provider_reports_all_missing_methods(monkeypatch):
    """The construction boundary reports structural contract failures together."""

    class IncompleteProvider:
        def __getattr__(self, name):
            if name in {"layer_norm", "linear"}:
                raise AttributeError(name)
            return lambda *args, **kwargs: None

    module = ModuleType("megatron.core.extensions.transformer_engine_spec_provider")
    module.TESpecProvider = IncompleteProvider
    monkeypatch.setattr(backends, "HAVE_TE", True)
    monkeypatch.setitem(sys.modules, module.__name__, module)

    with pytest.raises(TypeError, match="missing methods: layer_norm, linear"):
        get_backend_spec_provider("transformer_engine")


def test_kitchen_composes_over_the_selected_base_provider(monkeypatch):
    """Kitchen composition happens once and preserves the requested base provider."""
    from megatron.core.extensions import kitchen

    captured = {}

    class FakeKitchenSpecProvider:
        def __init__(self, *, fallback, use_kitchen_attention, kitchen_attention_backend):
            captured.update(
                fallback=fallback,
                use_kitchen_attention=use_kitchen_attention,
                kitchen_attention_backend=kitchen_attention_backend,
            )
            self.fallback = fallback

        def __getattr__(self, name):
            return getattr(self.fallback, name)

    monkeypatch.setattr(kitchen, "HAVE_KITCHEN", True)
    monkeypatch.setattr(kitchen, "KitchenSpecProvider", FakeKitchenSpecProvider)

    provider = get_backend_spec_provider(
        "local", use_kitchen=True, use_kitchen_attention=True, kitchen_attention_backend="fa"
    )

    assert isinstance(provider, FakeKitchenSpecProvider)
    assert isinstance(captured["fallback"], LocalSpecProvider)
    assert captured["use_kitchen_attention"] is True
    assert captured["kitchen_attention_backend"] == "fa"


def test_missing_kitchen_dependency_fails_at_construction(monkeypatch):
    """An explicitly requested optional provider must fail clearly when unavailable."""
    from megatron.core.extensions import kitchen

    monkeypatch.setattr(kitchen, "HAVE_KITCHEN", False)

    with pytest.raises(ImportError, match="nvidia-kitchen is not installed"):
        get_backend_spec_provider("local", use_kitchen=True)


def test_unknown_backend_fails_at_construction():
    """Unknown selectors must fail before model construction reaches a spec builder."""
    with pytest.raises(ValueError, match="unknown transformer_impl='unknown'"):
        get_backend_spec_provider("unknown")


@pytest.mark.parametrize(
    ("use_transformer_engine", "config_normalization", "normalization", "selector", "rms_norm"),
    (
        (True, "LayerNorm", None, "transformer_engine", False),
        (False, "RMSNorm", None, "local", True),
        (False, "LayerNorm", "RMSNorm", "local", True),
    ),
)
def test_gpt_final_norm_is_selected_by_provider(
    monkeypatch, use_transformer_engine, config_normalization, normalization, selector, rms_norm
):
    """GPT final norm stores the provider target during block-spec construction."""
    from megatron.core.models.gpt import gpt_layer_specs

    layer_spec = object()
    norm_target = object()
    provider = MagicMock(spec=BackendSpecProvider)
    provider.layer_norm.return_value = norm_target
    resolver = MagicMock(return_value=provider)
    config = MagicMock(
        pipeline_model_parallel_layout=None,
        transformer_impl=selector,
        normalization=config_normalization,
    )

    monkeypatch.setattr(gpt_layer_specs, "get_backend_spec_provider", resolver)
    layer_specs_builder = MagicMock(return_value=[layer_spec])
    monkeypatch.setattr(gpt_layer_specs, "get_gpt_decoder_layer_specs", layer_specs_builder)
    monkeypatch.setattr(gpt_layer_specs, "get_num_layers_to_build", MagicMock(return_value=1))
    monkeypatch.setattr(gpt_layer_specs, "get_transformer_layer_offset", MagicMock(return_value=0))

    block_spec = gpt_layer_specs.get_gpt_decoder_block_spec(
        config, use_transformer_engine=use_transformer_engine, normalization=normalization
    )

    effective_normalization = normalization or config_normalization
    resolver.assert_called_once_with(config, transformer_impl_override=selector)
    layer_specs_builder.assert_called_once_with(
        config, use_transformer_engine, effective_normalization, False, backend=provider
    )
    provider.layer_norm.assert_called_once_with(rms_norm=rms_norm)
    assert block_spec.layer_norm is norm_target


def test_gpt_decoder_layers_reuse_one_resolved_provider(monkeypatch):
    """Dense and MoE layer specs receive the same construction-time provider."""
    from megatron.core.models.gpt import gpt_layer_specs

    provider = MagicMock(spec=BackendSpecProvider)
    resolver = MagicMock(return_value=provider)
    dense_layer_spec = object()
    moe_layer_spec = object()
    local_spec_builder = MagicMock(side_effect=(dense_layer_spec, moe_layer_spec))
    config = SimpleNamespace(
        experimental_attention_variant=None,
        transformer_impl="local",
        normalization="RMSNorm",
        use_kitchen=False,
        use_kitchen_attention=False,
        kitchen_attention_backend="sdpa",
        qk_layernorm=False,
        multi_latent_attention=False,
        num_moe_experts=2,
        moe_grouped_gemm=False,
        moe_layer_freq=2,
        num_layers=2,
    )

    monkeypatch.setattr(gpt_layer_specs, "get_backend_spec_provider", resolver)
    monkeypatch.setattr(gpt_layer_specs, "get_gpt_layer_local_spec", local_spec_builder)

    layer_specs = gpt_layer_specs.get_gpt_decoder_layer_specs(config, use_transformer_engine=False)

    resolver.assert_called_once_with(config, transformer_impl_override="local")
    assert layer_specs == [moe_layer_spec, dense_layer_spec]
    assert local_spec_builder.call_count == 2
    assert all(call.kwargs["backend"] is provider for call in local_spec_builder.call_args_list)
    assert all(
        call.kwargs["normalization"] == "RMSNorm" for call in local_spec_builder.call_args_list
    )


def test_experimental_standard_attention_reuses_resolved_provider(monkeypatch):
    """A mixed-attention block must not rebuild its TE or Kitchen provider."""
    from megatron.core.models.gpt import (
        experimental_attention_variant_module_specs,
        gpt_layer_specs,
    )

    provider = MagicMock(spec=BackendSpecProvider)
    provider.fuse_layernorm_and_linear.return_value = True
    attention_spec = SimpleNamespace(metainfo={})
    layer_spec = SimpleNamespace(submodules=SimpleNamespace(self_attention=attention_spec))
    spec_builder = MagicMock(return_value=layer_spec)
    config = SimpleNamespace(
        num_moe_experts=2,
        moe_grouped_gemm=False,
        qk_layernorm=False,
        multi_latent_attention=False,
        qk_l2_norm=False,
        use_kitchen=True,
        use_te_activation_func=False,
        use_kitchen_attention=True,
        kitchen_attention_backend="fa",
        mla_down_proj_fusion=False,
    )
    monkeypatch.setattr(gpt_layer_specs, "get_gpt_layer_with_transformer_engine_spec", spec_builder)

    result = experimental_attention_variant_module_specs._get_self_attention_module_spec(
        config, backend=provider
    )

    assert result is attention_spec
    assert spec_builder.call_args.kwargs["backend"] is provider
    provider.fuse_layernorm_and_linear.assert_called_once_with()


def test_local_gpt_block_uses_one_real_provider_for_rms_norm(monkeypatch):
    """Local layer and final-norm construction share the effective RMSNorm choice."""
    from megatron.core.models.gpt import gpt_layer_specs

    layer_spec = object()
    layer_specs_builder = MagicMock(return_value=[layer_spec])
    config = SimpleNamespace(
        pipeline_model_parallel_layout=None,
        transformer_impl="local",
        normalization="RMSNorm",
        use_kitchen=False,
        use_kitchen_attention=False,
        kitchen_attention_backend="sdpa",
    )

    monkeypatch.setattr(gpt_layer_specs, "get_gpt_decoder_layer_specs", layer_specs_builder)
    monkeypatch.setattr(gpt_layer_specs, "get_num_layers_to_build", MagicMock(return_value=1))
    monkeypatch.setattr(gpt_layer_specs, "get_transformer_layer_offset", MagicMock(return_value=0))

    block_spec = gpt_layer_specs.get_gpt_decoder_block_spec(config, use_transformer_engine=False)

    assert block_spec.layer_norm is WrappedTorchNorm
    call = layer_specs_builder.call_args
    assert call.args[2] == "RMSNorm"
    assert isinstance(call.kwargs["backend"], LocalSpecProvider)
