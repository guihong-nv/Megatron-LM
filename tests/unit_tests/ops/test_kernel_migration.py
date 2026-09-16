# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Ownership, compatibility and construction-time kernel selection contracts."""

import ast
import importlib
import subprocess
import sys
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest

from megatron.core.models.backends import (
    BackendSpecProvider,
    KernelSelection,
    LocalSpecProvider,
    backend_slot,
    resolve_kernel_backend,
)
from megatron.core.ops import _backends
from megatron.core.ops.attention.dsa import backends as dsa_backends
from megatron.core.ops.attention.dsa.backends import DSAKernels, select_dsa_kernels
from megatron.core.ops.ssm.gated_delta.backends import select_gated_delta_rule


@pytest.mark.parametrize(
    "module",
    [
        "ops.ssm.common.causal_conv1d_varlen",
        "ops.ssm.mamba2.ssd_combined",
        "ops.ssm.gdp.decode_prepare",
        "ops.ssm.triton_cache_manager",
        *[
            f"ops.attention.dsa.{name}"
            for name in ("dsa_kernels", "dsa_layout", "dsa_masking", "dsa_indexer_loss")
        ],
    ],
)
def test_kernel_module_has_canonical_source(module):
    path = "megatron.core." + module
    target = importlib.import_module(path)
    assert target.__name__ == path
    assert Path(target.__file__).as_posix().endswith(path.replace(".", "/") + ".py")


@pytest.mark.parametrize(
    ("owner", "kernel", "symbol"),
    [
        ("ops.attention.csa.modules", "ops.attention.csa.reference", "_pool_compressor_values"),
        (
            "ops.ssm.gated_delta.gdn",
            "ops.ssm.gated_delta.reference",
            "torch_chunk_gated_delta_rule",
        ),
        ("ops.ssm.gated_delta.gdn2", "ops.ssm.gated_delta.reference_gdn2", "torch_chunk_gdn2"),
        ("ops.attention.dsa.modules", "ops.attention.dsa.reference", "unfused_dsa_fn"),
        ("ops.attention.dsa.modules", "ops.attention.dsa.reference", "FusedDSAIndexerLoss"),
        ("ops.attention.csa.modules", "ops.attention.csa.reference", "get_window_topk_idxs"),
        (
            "ops.attention.csa.modules",
            "ops.attention.csa.reference",
            "unfused_compressed_sparse_attn",
        ),
    ],
)
def test_operation_uses_canonical_reference(owner, kernel, symbol):
    assert getattr(importlib.import_module("megatron.core." + owner), symbol) is getattr(
        importlib.import_module("megatron.core." + kernel), symbol
    )


def test_family_namespaces_do_not_load_optional_dependencies():
    code = """
import importlib
import sys
import megatron.core
before = set(sys.modules)
for family in ('', '.ssm', '.ssm.common', '.ssm.mamba2', '.ssm.gated_delta', '.ssm.gdp',
               '.attention', '.attention.dsa', '.attention.csa'):
    importlib.import_module('megatron.core.ops' + family)
optional = {'triton', 'tilelang', 'fla', 'mamba_ssm', 'causal_conv1d',
            'fast_hadamard_transform', 'cudnn', 'gdp_attn'}
assert not {name for name in set(sys.modules) - before if name.split('.')[0] in optional}
"""
    subprocess.run([sys.executable, "-c", code], check=True, timeout=120)


def _imports_model_assembly_or_legacy_path(name):
    if name == "megatron.core.models.backends" or name == "megatron.core.models.common.embeddings":
        return False
    if name.startswith("megatron.core.models.common.embeddings."):
        return False
    return name.startswith(
        (
            "megatron.core.ssm",
            "megatron.core.models",
            "megatron.core.transformer.experimental_attention_variant",
        )
    )


@pytest.mark.parametrize(
    ("name", "forbidden"),
    [
        ("megatron.core.models.backends", False),
        ("megatron.core.models.common.embeddings", False),
        ("megatron.core.models.common.embeddings.rope_utils", False),
        ("megatron.core.models.hybrid.hybrid_layer_specs", True),
        ("megatron.core.models.gpt", True),
        ("megatron.core.models.common.language_module", True),
        ("megatron.core.ssm.mamba_mixer", True),
        ("megatron.core.transformer.experimental_attention_variant.dsa", True),
    ],
)
def test_operation_import_boundary_distinguishes_shared_infrastructure(name, forbidden):
    assert _imports_model_assembly_or_legacy_path(name) is forbidden


def test_ops_do_not_import_model_assembly_or_legacy_paths():
    import megatron.core.ops

    root = Path(megatron.core.ops.__file__).parent
    for path in root.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            elif isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            else:
                continue
            assert not any(_imports_model_assembly_or_legacy_path(name) for name in names), path


_CUDNN_DSA_SYMBOLS = [
    symbol.split(".")[1]
    for module, symbols in dsa_backends._NATIVE_REQUIREMENTS["cudnn"]
    for symbol in symbols
    if module == "cudnn"
]


def test_dsa_binds_direct_hooks_once_and_keeps_instances_independent(monkeypatch):
    first_hook = lambda **kwargs: kwargs
    second_hook = lambda **kwargs: None
    modules = {
        "tilelang": SimpleNamespace(run_fused_qk_topk=first_hook),
        "cudnn": SimpleNamespace(run_fused_qk_topk=second_hook),
    }
    calls = []

    def load(module_name):
        if module_name in ("tilelang", "triton", "cudnn", "flash_mla"):
            return SimpleNamespace(
                DSA=SimpleNamespace(**{n: object() for n in _CUDNN_DSA_SYMBOLS}),
                flash_mla_sparse_fwd=object(),
            )
        name = "tilelang" if "tilelang" in module_name else "cudnn"
        calls.append(name)
        return modules[name]

    monkeypatch.setattr(_backends, "import_module", load)
    first = select_dsa_kernels("tilelang")
    second = select_dsa_kernels("cudnn")
    assert first.run_fused_qk_topk is first_hook
    assert second.run_fused_qk_topk is second_hook
    assert first.run_fused_qk_topk(q=1) == {"q": 1}
    assert second.run_fused_qk_topk(q=1) is None
    assert first.backend == "tilelang"
    assert first.run_fused_dsa_attention is None
    assert calls == ["tilelang", "cudnn"]


@pytest.mark.parametrize(("fused", "kernel"), [(False, "cudnn"), (True, "none")])
def test_disabled_dsa_does_not_import_backend(monkeypatch, fused, kernel):
    def unexpected(_name):
        pytest.fail("disabled fused DSA must not import a backend")

    monkeypatch.setattr(_backends, "import_module", unexpected)
    assert select_dsa_kernels(kernel, fused=fused) == DSAKernels()


def test_invalid_dsa_backend_is_rejected():
    with pytest.raises(ValueError, match="dsa_kernel_backend"):
        select_dsa_kernels("invalid")


@pytest.mark.parametrize("error", [ImportError, OSError])
def test_missing_selected_dsa_backend_fails_at_construction(monkeypatch, error):
    def fail_import(_name):
        raise error("missing extension")

    monkeypatch.setattr(_backends, "import_module", fail_import)
    with pytest.raises(ImportError, match="DSA kernel backend 'cudnn' requires"):
        select_dsa_kernels("cudnn")


@pytest.mark.parametrize("variant", ["gdn", "gdn2"])
def test_gated_delta_reference_and_missing_selected_kernel(monkeypatch, variant):
    from megatron.core.ops.ssm.gated_delta.reference import torch_chunk_gated_delta_rule
    from megatron.core.ops.ssm.gated_delta.reference_gdn2 import torch_chunk_gdn2

    reference = torch_chunk_gated_delta_rule if variant == "gdn" else torch_chunk_gdn2
    assert select_gated_delta_rule(variant, deterministic=True) is reference

    def missing(_name):
        raise ImportError("missing selected kernel")

    monkeypatch.setattr(_backends, "import_module", missing)
    requirement = "GDN requires fla.ops.gated_delta_rule" if variant == "gdn" else "GDN2 requires"
    with pytest.raises(ImportError, match=requirement):
        select_gated_delta_rule(variant)


_KERNEL_SLOTS = (
    "mamba_kernels",
    "gated_delta_rule",
    "gated_delta_product",
    "gated_delta_product_cp_backend",
    "dsa_kernels",
)


@pytest.mark.parametrize("slot", _KERNEL_SLOTS)
def test_older_providers_use_family_defaults(slot):
    # Protocol methods may be inherited as stubs or absent on structural providers.
    inherited = SimpleNamespace()
    setattr(inherited, slot, MethodType(getattr(BackendSpecProvider, slot), inherited))
    for provider in (SimpleNamespace(), inherited):
        sentinel = object()
        assert backend_slot(provider, slot, default=lambda: sentinel) is sentinel


@pytest.mark.parametrize("slot", _KERNEL_SLOTS)
def test_kernel_slots_take_no_implementation_arguments(slot):
    """A provider is configured once; slots describe the operation, never the backend."""
    import inspect

    parameters = [
        name
        for name in inspect.signature(getattr(BackendSpecProvider, slot)).parameters
        if name != "self"
    ]
    assert parameters == (["variant"] if slot == "gated_delta_rule" else [])


def test_kernel_selection_reads_the_existing_config_fields():
    from megatron.core.transformer.enums import AttnBackend

    config = SimpleNamespace(
        deterministic_mode=True,
        use_mamba_mem_eff_path=True,
        gdp_cutedsl_kernel=True,
        gdp_num_chunk_states_to_recompute=3,
        dsa_kernel_backend="cudnn",
        attention_backend=AttnBackend.unfused,
    )
    assert KernelSelection.from_config(config) == KernelSelection(
        deterministic=True,
        mamba_mem_eff_path=True,
        gdp_cutedsl=True,
        gdp_recompute_chunk_num=3,
        dsa_backend="cudnn",
        dsa_fused=False,
    )
    assert KernelSelection.from_config(SimpleNamespace()) == KernelSelection()


def test_local_and_te_share_one_kernel_selection(monkeypatch):
    """Both base providers answer the kernel slots identically from the same settings."""
    from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider
    from megatron.core.ops.ssm.gated_delta import backends as gdn_backends
    from megatron.core.ops.ssm.gdp import backends as gdp_backends
    from megatron.core.ops.ssm.mamba2 import backends as mamba_backends

    calls = []
    monkeypatch.setattr(
        gdn_backends, "select_gated_delta_rule", lambda *a, **k: calls.append(("gdn", a, k))
    )
    monkeypatch.setattr(
        gdp_backends, "select_gated_delta_product", lambda *a, **k: calls.append(("gdp", a, k))
    )
    monkeypatch.setattr(
        gdp_backends, "select_gdp_cp_backend", lambda *a, **k: calls.append(("gdp_cp", a, k))
    )
    monkeypatch.setattr(
        mamba_backends, "select_mamba_kernels", lambda *a, **k: calls.append(("mamba", a, k))
    )
    monkeypatch.setattr(
        dsa_backends, "select_dsa_kernels", lambda *a, **k: calls.append(("dsa", a, k))
    )
    selection = KernelSelection(
        deterministic=True, mamba_mem_eff_path=True, gdp_cutedsl=True, gdp_recompute_chunk_num=2
    )
    for provider in (LocalSpecProvider(kernels=selection), TESpecProvider(kernels=selection)):
        calls.clear()
        provider.gated_delta_rule("gdn2")
        provider.gated_delta_product()
        provider.gated_delta_product_cp_backend()
        provider.mamba_kernels()
        provider.dsa_kernels()
        assert calls == [
            ("gdn", ("gdn2", True), {}),
            ("gdp", (True,), {}),
            ("gdp_cp", (True,), {"recompute_chunk_num": 2}),
            ("mamba", (True,), {}),
            ("dsa", ("none",), {"fused": True}),
        ]


def test_local_and_te_preserve_gated_delta_defaults():
    from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider

    deterministic = KernelSelection(deterministic=True)
    for variant in ("gdn", "gdn2"):
        assert LocalSpecProvider(kernels=deterministic).gated_delta_rule(variant) is (
            TESpecProvider(kernels=deterministic).gated_delta_rule(variant)
        )


def test_resolve_kernel_backend_prefers_the_explicit_provider():
    config = SimpleNamespace(transformer_impl="local", deterministic_mode=True)

    custom = object()  # a wrapper/custom provider without the mixin is returned untouched
    assert resolve_kernel_backend(custom, config) is custom

    derived = resolve_kernel_backend(None, config)
    assert isinstance(derived, LocalSpecProvider)
    assert derived._kernels == KernelSelection(deterministic=True)

    configured = LocalSpecProvider(kernels=KernelSelection(deterministic=False))
    assert resolve_kernel_backend(configured, config) is configured  # explicit wins over config


def test_bare_provider_is_configured_from_the_model_config_without_mutation():
    """``TESpecProvider()`` handed to a spec must still honor the config's kernel settings."""
    from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider

    bare = TESpecProvider()
    config = SimpleNamespace(dsa_kernel_backend="cudnn", deterministic_mode=True)
    bound = resolve_kernel_backend(bare, config)
    assert bound is not bare
    assert bound._kernels == KernelSelection(deterministic=True, dsa_backend="cudnn")
    assert bare._kernel_selection is None
    with pytest.raises(RuntimeError, match="built without a KernelSelection"):
        bare.gated_delta_product()


def test_specs_preserve_the_explicit_kernel_provider():
    from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider
    from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
        get_dsa_module_spec_for_backend,
        get_gated_delta_net_module_spec,
    )
    from megatron.core.transformer.transformer_config import TransformerConfig

    config = TransformerConfig(
        num_layers=1, hidden_size=16, num_attention_heads=2, multi_latent_attention=True
    )
    provider = TESpecProvider()
    dsa = get_dsa_module_spec_for_backend(config, provider)
    assert dsa.submodules.core_attention.params["kernel_backend"] is provider
    gdn = get_gated_delta_net_module_spec(config, provider)
    assert gdn.params["kernel_backend"] is provider


def test_dsa_construction_uses_explicit_provider_without_rebuilding_it():
    from megatron.core.ops.attention.dsa.modules import DSAttention, DSAttentionSubmodules
    from megatron.core.transformer.enums import AttnMaskType

    kernels = DSAKernels(backend="custom")
    seen = []

    class CustomProvider:
        def dsa_kernels(self):
            seen.append("bound")
            return kernels

    config = SimpleNamespace(
        dsa_indexer_topk=8, dsa_indexer_topk_freq=4, dsa_indexer_skip_topk_offset=1, kv_channels=16
    )
    attention = DSAttention(
        config=config,
        submodules=DSAttentionSubmodules(indexer=object()),
        layer_number=2,
        attn_mask_type=AttnMaskType.causal,
        attention_type="self",
        softmax_scale=1.0,
        pg_collection=SimpleNamespace(),
        kernel_backend=CustomProvider(),
    )
    assert attention.dsa_kernels is kernels
    assert seen == ["bound"]
