# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Tests for the construction-time, single-class BackendSpecProvider boundary."""

import inspect
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest

import megatron.core.ops.provider as provider_module
from megatron.core.ops import (
    Backend,
    BackendSpecProvider,
    Operation,
    OpsBackendConfig,
    get_backend,
    get_backend_spec_provider,
)


@pytest.fixture
def resolved_ops(monkeypatch):
    """Replace optional backend resolution with a dependency-free sentinel."""
    value = MagicMock(name="resolved_backend_ops")
    resolver = MagicMock(return_value=value)
    monkeypatch.setattr(provider_module, "_resolve_backend_ops", resolver)
    return value, resolver


def _resolved_config(resolver: MagicMock) -> OpsBackendConfig:
    """Return the normalized config passed to the backend-op resolver."""
    resolver.assert_called_once()
    config = resolver.call_args.args[0]
    assert isinstance(config, OpsBackendConfig)
    return config


def test_backend_spec_provider_is_a_concrete_class():
    """The public provider is a concrete aggregate, not a Protocol or abstract base class."""
    assert inspect.isclass(BackendSpecProvider)
    assert not inspect.isabstract(BackendSpecProvider)
    assert not getattr(BackendSpecProvider, "_is_protocol", False)


@pytest.mark.parametrize(
    "preset", (Backend.LOCAL, Backend.TRANSFORMER_ENGINE, Backend.INFERENCE_OPTIMIZED)
)
def test_every_builtin_preset_returns_the_exact_provider_type(resolved_ops, preset):
    """Presets fill one provider class rather than selecting provider subclasses."""
    value, resolver = resolved_ops

    provider = get_backend(preset)

    assert type(provider) is BackendSpecProvider
    assert provider._ops is value
    _resolved_config(resolver)


@pytest.mark.parametrize("preset", (Backend.KITCHEN, Backend.NONE))
def test_composition_only_backends_are_not_whole_provider_presets(resolved_ops, preset):
    """Kitchen composition and the disabled marker are valid only as operation overrides."""
    _, resolver = resolved_ops

    with pytest.raises(ValueError, match=preset.value):
        get_backend(preset)

    resolver.assert_not_called()


def test_string_preset_is_normalized_before_resolution(resolved_ops):
    """Legacy string selectors normalize to the public Backend enum."""
    _, resolver = resolved_ops

    provider = get_backend("local")

    assert type(provider) is BackendSpecProvider
    config = _resolved_config(resolver)
    assert config.default_backend is Backend.LOCAL


def test_get_backend_spec_provider_accepts_an_ops_config(resolved_ops):
    """The canonical factory accepts an already-normalized per-operation config."""
    _, resolver = resolved_ops
    config = OpsBackendConfig(
        default_backend=Backend.LOCAL, overrides={Operation.NORM: Backend.TRANSFORMER_ENGINE}
    )

    provider = get_backend_spec_provider(config)

    assert type(provider) is BackendSpecProvider
    resolved_config = _resolved_config(resolver)
    assert resolved_config.backend_for(Operation.NORM) is Backend.TRANSFORMER_ENGINE
    assert resolved_config.backend_for(Operation.LINEAR) is Backend.LOCAL


def test_compatibility_facade_returns_the_exact_provider_type(resolved_ops):
    """The old import location delegates without reintroducing provider classes."""
    from megatron.core.models import backends as compatibility_facade

    assert compatibility_facade.BackendSpecProvider is BackendSpecProvider
    assert type(compatibility_facade.get_backend("local")) is BackendSpecProvider


def test_per_operation_override_is_applied_before_backend_resolution(monkeypatch):
    """Only the final merged selection reaches dependency and API resolution."""
    resolved = MagicMock(name="resolved_backend_ops")

    def assert_final_selection(config):
        assert config.default_backend is Backend.LOCAL
        assert config.backend_for(Operation.NORM) is Backend.TRANSFORMER_ENGINE
        assert config.backend_for(Operation.LINEAR) is Backend.NONE
        return resolved

    resolver = MagicMock(side_effect=assert_final_selection)
    monkeypatch.setattr(provider_module, "_resolve_backend_ops", resolver)

    provider = get_backend(Backend.LOCAL, overrides={Operation.NORM: Backend.TRANSFORMER_ENGINE})

    assert type(provider) is BackendSpecProvider
    assert provider._ops is resolved
    resolver.assert_called_once()


def test_string_operation_and_backend_overrides_are_normalized(resolved_ops):
    """CLI-friendly override strings have the same meaning as enum values."""
    _, resolver = resolved_ops

    get_backend("local", overrides={"norm": "transformer_engine"})

    config = _resolved_config(resolver)
    assert config.backend_for(Operation.NORM) is Backend.TRANSFORMER_ENGINE


def test_explicit_overrides_take_precedence_over_config_overrides(resolved_ops):
    """Call-site overrides replace the same operation selected by the input config."""
    _, resolver = resolved_ops
    config = OpsBackendConfig(
        default_backend=Backend.LOCAL, overrides={Operation.NORM: Backend.LOCAL}
    )

    get_backend_spec_provider(config, overrides={Operation.NORM: Backend.TRANSFORMER_ENGINE})

    resolved_config = _resolved_config(resolver)
    assert resolved_config.backend_for(Operation.NORM) is Backend.TRANSFORMER_ENGINE
    assert resolved_config.backend_for(Operation.LINEAR) is Backend.LOCAL


def test_caller_override_mapping_is_copied(resolved_ops):
    """Mutating a caller-owned dictionary cannot change a constructed provider."""
    _, resolver = resolved_ops
    overrides = {"norm": "transformer_engine"}

    get_backend("local", overrides=overrides)
    config = _resolved_config(resolver)
    overrides["norm"] = "local"

    assert config.backend_for(Operation.NORM) is Backend.TRANSFORMER_ENGINE


def test_unknown_operation_override_fails_before_resolution(resolved_ops):
    """Unknown operation names fail before any optional dependency is inspected."""
    _, resolver = resolved_ops

    with pytest.raises((KeyError, ValueError), match="not_an_operation"):
        get_backend("local", overrides={"not_an_operation": "local"})

    resolver.assert_not_called()


def test_unknown_backend_fails_before_resolution(resolved_ops):
    """Unknown backend names fail before any optional dependency is inspected."""
    _, resolver = resolved_ops

    with pytest.raises((KeyError, ValueError), match="unknown"):
        get_backend("unknown")

    resolver.assert_not_called()


def test_transformer_config_and_legacy_override_produce_one_provider(resolved_ops):
    """The compatibility form normalizes old selectors into the one provider model."""
    _, resolver = resolved_ops
    config = SimpleNamespace(
        transformer_impl="transformer_engine",
        use_kitchen=False,
        use_kitchen_attention=False,
        kitchen_attention_backend="sdpa",
    )

    provider = get_backend_spec_provider(config, transformer_impl_override="local")

    assert type(provider) is BackendSpecProvider
    resolved_config = _resolved_config(resolver)
    assert resolved_config.default_backend is Backend.LOCAL


def test_transformer_config_applies_canonical_operation_overrides(resolved_ops):
    """The public config field selects an operation before the provider is resolved."""
    _, resolver = resolved_ops
    config = SimpleNamespace(
        transformer_impl="local", op_backend_overrides={"norm": "transformer_engine"}
    )

    get_backend_spec_provider(config)

    resolved_config = _resolved_config(resolver)
    assert resolved_config.backend_for(Operation.NORM) is Backend.TRANSFORMER_ENGINE
    assert resolved_config.backend_for(Operation.CORE_ATTENTION) is Backend.LOCAL


def test_inference_preset_keeps_inference_operation_semantics(resolved_ops):
    """Inference is one full preset, including its norm and activation behavior."""
    _, resolver = resolved_ops

    get_backend("inference_optimized")

    config = _resolved_config(resolver)
    for operation in (
        Operation.LINEAR,
        Operation.NORM,
        Operation.QK_NORM,
        Operation.CORE_ATTENTION,
        Operation.ACTIVATION,
        Operation.GROUPED_MLP,
        Operation.NORM_LINEAR,
        Operation.MOE_ROUTER,
    ):
        assert config.backend_for(operation) is Backend.INFERENCE_OPTIMIZED


def test_legacy_kitchen_arguments_become_operation_overrides(resolved_ops):
    """Legacy Kitchen flags compose operation slots without wrapping the provider."""
    _, resolver = resolved_ops

    provider = get_backend_spec_provider(
        "local", use_kitchen=True, use_kitchen_attention=True, kitchen_attention_backend="fa"
    )

    assert type(provider) is BackendSpecProvider
    config = _resolved_config(resolver)
    assert config.kitchen_attention_backend == "fa"
    for operation in (
        Operation.COLUMN_PARALLEL_LINEAR,
        Operation.ROW_PARALLEL_LINEAR,
        Operation.NORM_LINEAR,
        Operation.GROUPED_MLP,
        Operation.CORE_ATTENTION,
    ):
        assert config.backend_for(operation) is Backend.KITCHEN


def test_selected_dependency_is_loaded_once_and_unselected_modules_are_untouched(monkeypatch):
    """The dependency cache imports only the module explicitly requested by a selected spec."""
    from megatron.core.ops import _dependencies

    selected = ModuleType("selected_backend")
    original_kernel = object()
    selected.kernel = original_kernel
    imported = []

    def fake_import_module(name):
        imported.append(name)
        if name == selected.__name__:
            return selected
        raise AssertionError(f"unselected dependency was imported: {name}")

    _dependencies._reset_dependency_cache()
    monkeypatch.setattr(_dependencies.importlib, "import_module", fake_import_module)

    try:
        first = _dependencies.require_symbols(selected.__name__, ("kernel",))
        selected.kernel = object()
        second = _dependencies.require_symbols(selected.__name__, ("kernel",))

        assert first["kernel"] is original_kernel
        assert second["kernel"] is original_kernel
        assert imported == [selected.__name__]
    finally:
        _dependencies._reset_dependency_cache()


def test_dependency_import_failure_is_cached(monkeypatch):
    """A broken selected package is imported once and reports the same early error thereafter."""
    from megatron.core.ops import _dependencies

    imported = []

    def fail_import(name):
        imported.append(name)
        raise RuntimeError("broken native library")

    _dependencies._reset_dependency_cache()
    monkeypatch.setattr(_dependencies.importlib, "import_module", fail_import)

    try:
        for _ in range(2):
            with pytest.raises(ImportError, match="failed to import"):
                _dependencies.require_module("broken_backend", purpose="norm")
        assert imported == ["broken_backend"]
    finally:
        _dependencies._reset_dependency_cache()


def test_mlp_uses_provider_fusion_selection_without_a_second_switch():
    """The operation table, not the old boolean argument, selects the fused MLP."""
    from megatron.core.models.gpt import gpt_layer_specs

    backend = MagicMock(spec=BackendSpecProvider)
    fused_target = SimpleNamespace(as_mlp_submodule=MagicMock(name="fused_mlp_builder"))
    backend.fused_mlp.return_value = fused_target
    backend.activation_func.return_value = None
    backend.norm_linear.return_value = SimpleNamespace(linear=None, fuses_norm=False)

    spec = gpt_layer_specs.get_mlp_module_spec_for_backend(backend)

    assert spec.func is fused_target.as_mlp_submodule
    backend.fused_mlp.assert_called_once_with(grouped=False)


def test_unfused_norm_linear_override_builds_explicit_gpt_norms():
    """Disabling norm-linear fusion produces a valid explicit norm + linear spec."""
    from megatron.core.models.gpt import gpt_layer_specs

    backend = MagicMock(spec=BackendSpecProvider)
    column_linear = MagicMock(name="column_linear")
    norm = MagicMock(name="norm")
    backend.column_parallel_linear.return_value = column_linear
    backend.layer_norm.return_value = norm
    backend.norm_linear.return_value = SimpleNamespace(linear=None, fuses_norm=False)
    backend.fused_mlp.return_value = None
    backend.activation_func.return_value = None

    submodules = gpt_layer_specs.get_gpt_layer_with_transformer_engine_submodules(backend=backend)

    assert submodules.input_layernorm is norm
    assert submodules.self_attention.submodules.linear_qkv is column_linear
    assert submodules.pre_mlp_layernorm is norm


@pytest.mark.parametrize(
    ("use_transformer_engine", "config_normalization", "normalization", "selector", "rms_norm"),
    (
        (True, "LayerNorm", None, "transformer_engine", False),
        (False, "RMSNorm", None, "local", True),
        (False, "LayerNorm", "RMSNorm", "local", True),
    ),
)
def test_gpt_final_norm_reuses_the_resolved_provider(
    monkeypatch, use_transformer_engine, config_normalization, normalization, selector, rms_norm
):
    """GPT final norm and decoder layers receive the same construction-time provider."""
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
    """Dense and MoE layer specs receive one identical provider object."""
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


def test_experimental_standard_attention_reuses_resolved_provider(monkeypatch):
    """A mixed-attention block passes its provider into the standard-attention builder."""
    from megatron.core.models.gpt import (
        experimental_attention_variant_module_specs,
        gpt_layer_specs,
    )

    provider = MagicMock(spec=BackendSpecProvider)
    provider.norm_linear.return_value = SimpleNamespace(fuses_norm=True)
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
    provider.norm_linear.assert_called_once_with()
