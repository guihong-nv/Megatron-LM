# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Optional-dependency ownership: selectors check only what was selected, at construction."""

import ast
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from megatron.core.ops import _backends
from megatron.core.ops._backends import require
from megatron.core.ops.ssm.gated_delta.backends import select_gated_delta_rule
from megatron.core.ops.ssm.gdp.backends import select_gated_delta_product, select_gdp_cp_backend
from megatron.core.ops.ssm.mamba2.backends import select_mamba_kernels
from megatron.core.transformer.transformer_config import TransformerConfig


def _module(monkeypatch, name, **exports):
    module = ModuleType(name)
    module.__dict__.update(exports)
    monkeypatch.setitem(sys.modules, name, module)
    return module


# ---------------------------------------------------------------------------------------------
# require()
# ---------------------------------------------------------------------------------------------


def test_require_returns_the_module_and_checks_exports(monkeypatch):
    target = lambda: None
    _module(monkeypatch, "fake_kernels", entry=target, namespace=SimpleNamespace(deep=target))
    module = require("fake_kernels", "entry", "namespace.deep", needed_by="test op")
    assert module.entry is target
    with pytest.raises(ImportError, match="test op requires fake_kernels with fake_kernels.nope"):
        require("fake_kernels", "nope", needed_by="test op")


def test_require_treats_a_none_export_as_missing(monkeypatch):
    _module(monkeypatch, "fake_kernels", entry=None)
    with pytest.raises(ImportError, match="fake_kernels.entry, which is unavailable"):
        require("fake_kernels", "entry", needed_by="test op")


@pytest.mark.parametrize("error", [ImportError, OSError, RuntimeError])
def test_require_reports_broken_libraries_as_import_errors_with_the_cause(monkeypatch, error):
    cause = error("libcuda mismatch")

    def fail(_name):
        raise cause

    monkeypatch.setattr(_backends, "import_module", fail)
    with pytest.raises(ImportError, match="test op requires fake_kernels") as caught:
        require("fake_kernels", needed_by="test op")
    assert caught.value.__cause__ is cause


def test_require_prefers_dunder_version_over_distribution_metadata(monkeypatch):
    _module(monkeypatch, "fake_versioned", __version__="1.7.0", entry=object())
    monkeypatch.setattr(_backends.metadata, "version", lambda _name: pytest.fail("not needed"))
    assert require("fake_versioned", "entry", min_version="1.6.0", needed_by="op")
    with pytest.raises(ImportError, match=r"fake_versioned>=1\.8\.0; found 1\.7\.0"):
        require("fake_versioned", min_version="1.8.0", needed_by="op")


def test_require_falls_back_to_distribution_metadata(monkeypatch):
    _module(monkeypatch, "fake_dist", entry=object())
    monkeypatch.setattr(_backends.metadata, "version", lambda name: {"fake-dist": "2.0"}[name])
    assert require("fake_dist", min_version="1.0", dist="fake-dist", needed_by="op")

    def not_found(name):
        raise _backends.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(_backends.metadata, "version", not_found)
    with pytest.raises(ImportError, match="version cannot be determined"):
        require("fake_dist", min_version="1.0", dist="fake-dist", needed_by="op")


def test_is_available_never_imports(monkeypatch):
    monkeypatch.setattr(_backends, "import_module", lambda _n: pytest.fail("imported"))
    assert _backends.is_available("json")
    assert not _backends.is_available("definitely_not_installed_xyz")


# ---------------------------------------------------------------------------------------------
# Family selectors
# ---------------------------------------------------------------------------------------------


def test_selectors_do_not_import_unselected_optional_libraries():
    code = """
import importlib
import importlib.abc
import sys
import megatron.core
blocked = {'fla', 'mamba_ssm', 'gdp_attn', 'tilelang', 'cudnn', 'flash_mla'}
class BlockOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in blocked:
            raise AssertionError('unselected dependency imported: ' + fullname)
sys.meta_path.insert(0, BlockOptional())
before = set(sys.modules)
for family in ('ssm.gdp', 'ssm.gated_delta', 'ssm.mamba2', 'attention.dsa'):
    importlib.import_module('megatron.core.ops.' + family + '.backends')
assert not {name for name in set(sys.modules) - before if name.split('.')[0] in blocked}
"""
    subprocess.run([sys.executable, "-c", code], check=True, timeout=120)


@pytest.mark.parametrize(
    "select",
    [
        lambda: select_gated_delta_rule("gdn"),
        lambda: select_gated_delta_rule("gdn2"),
        lambda: select_gated_delta_product(False),
        lambda: select_gated_delta_product(True),
        lambda: select_gdp_cp_backend(False),
        lambda: select_gdp_cp_backend(True),
        lambda: select_mamba_kernels(False),
        lambda: select_mamba_kernels(True),
    ],
)
@pytest.mark.parametrize("error", [ImportError, OSError])
def test_selected_dependency_failure_preserves_cause_without_fallback(monkeypatch, select, error):
    cause = error("selected kernel is broken")
    calls = []

    def fail(name):
        calls.append(name)
        raise cause

    monkeypatch.setattr(_backends, "import_module", fail)
    with pytest.raises(ImportError, match="requires") as caught:
        select()
    assert caught.value.__cause__ is cause
    assert len(calls) == 1


def test_mamba_does_not_require_unselected_split_scan(monkeypatch):
    target = lambda x, state_dtype=None: None
    _module(monkeypatch, "mamba_ssm.ops.triton.ssd_combined", mamba_chunk_scan_combined=target)
    monkeypatch.setattr(_backends, "is_available", lambda _name: False)
    kernels = select_mamba_kernels(False)
    assert kernels.scan is target
    assert kernels.split_scan is None
    assert kernels.has_state_dtype
    assert kernels.causal_conv1d is None
    with pytest.raises(ImportError, match="mamba_split_conv1d_scan_combined, which is missing"):
        select_mamba_kernels(True)


@pytest.mark.parametrize("use_mem_eff_path", [False, True])
def test_mamba_convolution_fallback_only_handles_an_absent_package(monkeypatch, use_mem_eff_path):
    scan = lambda: None
    _module(
        monkeypatch,
        "mamba_ssm.ops.triton.ssd_combined",
        mamba_chunk_scan_combined=scan,
        mamba_split_conv1d_scan_combined=scan,
    )
    original_import = _backends.import_module

    # Absent package: the ordinary path keeps Torch convolution, the fused path needs it.
    def absent(name):
        if name == "causal_conv1d":
            raise ModuleNotFoundError("No module named 'causal_conv1d'", name=name)
        return original_import(name)

    monkeypatch.setattr(_backends, "is_available", lambda name: name != "causal_conv1d")
    monkeypatch.setattr(_backends, "import_module", absent)
    if use_mem_eff_path:
        with pytest.raises(ImportError, match="requires causal_conv1d"):
            select_mamba_kernels(True)
    else:
        assert select_mamba_kernels(False).causal_conv1d is None

    # Installed but broken: an error in both cases, never a silent fallback.
    monkeypatch.setattr(_backends, "is_available", lambda _name: True)
    cause = RuntimeError("causal_conv1d_cuda was built against another CUDA")

    def load(name):
        if name == "causal_conv1d":
            raise cause
        return original_import(name)

    monkeypatch.setattr(_backends, "import_module", load)
    with pytest.raises(ImportError, match="installed but failed to load") as caught:
        select_mamba_kernels(use_mem_eff_path)
    assert caught.value.__cause__ is cause


def test_packed_cp_convolution_decides_its_version_once_and_checks_a_bool_per_call(monkeypatch):
    from megatron.core.ops.ssm.common import causal_conv1d_cp as module

    _module(monkeypatch, "causal_conv1d", __version__="1.6.1")
    assert module.packed_cp_conv_supported() is False
    _module(monkeypatch, "causal_conv1d", __version__="1.7.0")
    assert module.packed_cp_conv_supported() is True

    monkeypatch.setattr(_backends, "import_module", lambda _n: pytest.fail("imported in forward"))
    with pytest.raises(ImportError, match="causal-conv1d >= 1.7.0"):
        module.causal_conv1d_cp(
            None,
            None,
            None,
            None,
            None,
            global_seq_idx=object(),
            conv_fn=None,
            packed_supported=False,
        )


def test_gdp_decode_prepare_checks_libdevice_exports(monkeypatch):
    from megatron.core.ops.ssm.gdp.mixer import GatedDeltaProductMixer

    _module(monkeypatch, "triton", __version__="3.1.0")
    _module(monkeypatch, "triton.language.extra.libdevice", exp=object(), log1p=object())
    with pytest.raises(ImportError, match=r"div_rn, which is missing"):
        GatedDeltaProductMixer.bind_dynamic_inference_kernels(SimpleNamespace())


@pytest.mark.parametrize("batch_invariant", [False, True])
def test_mamba_decode_binds_cuda_update_only_for_batch_invariant_mode(monkeypatch, batch_invariant):
    from megatron.core.ops.ssm.mamba2.mixer import MambaMixer

    _module(monkeypatch, "triton")
    update = lambda: None
    _module(monkeypatch, "causal_conv1d", causal_conv1d_update=update)
    owner = SimpleNamespace(
        config=SimpleNamespace(batch_invariant_mode=batch_invariant),
        _causal_conv1d_update_cuda=None,
        _inference_kernels_bound=False,
    )
    bound = MambaMixer.bind_dynamic_inference_kernels(owner)
    assert (bound is update) is batch_invariant
    assert (owner._causal_conv1d_update_cuda is update) is batch_invariant
    # Bound once: a second call (every decode step) must not touch the import system.
    monkeypatch.setattr(_backends, "import_module", lambda _n: pytest.fail("re-imported"))
    assert MambaMixer.bind_dynamic_inference_kernels(owner) is bound


@pytest.mark.parametrize("missing", ["tilelang", "triton"])
def test_dsa_selection_reports_missing_native_libraries_against_the_operation(monkeypatch, missing):
    from megatron.core.ops.attention.dsa.backends import select_dsa_kernels

    original_import = _backends.import_module

    def load(name):
        if name == missing:
            raise ModuleNotFoundError(f"No module named {name!r}", name=name)
        if name in ("tilelang", "triton"):
            return ModuleType(name)
        return original_import(name)

    monkeypatch.setattr(_backends, "import_module", load)
    with pytest.raises(ImportError, match=f"DSA kernel backend 'tilelang' requires {missing}"):
        select_dsa_kernels("tilelang")


def test_dsa_selection_checks_cudnn_namespace_members(monkeypatch):
    from megatron.core.ops.attention.dsa.backends import select_dsa_kernels

    _module(monkeypatch, "cudnn", DSA=SimpleNamespace(indexer_top_k_wrapper=object()))
    with pytest.raises(ImportError, match=r"cudnn\.DSA\.indexer_forward_wrapper, which is missing"):
        select_dsa_kernels("cudnn")


# ---------------------------------------------------------------------------------------------
# Operation constructors check their auxiliary kernels, independently of the recurrence.
# ---------------------------------------------------------------------------------------------


class _BeforeParameters(Exception):
    pass


def _stop_before_parameters(*args, **kwargs):
    raise _BeforeParameters


def _config(**kwargs):
    return TransformerConfig(
        num_layers=1,
        hidden_size=128,
        num_attention_heads=4,
        mamba_num_heads=4,
        mamba_num_groups=1,
        mamba_head_dim=32,
        mamba_state_dim=32,
        **kwargs,
    )


def _groups():
    return SimpleNamespace(tp=SimpleNamespace(size=lambda: 1), cp=SimpleNamespace(size=lambda: 1))


def _recording_require(*, fail_module=None):
    """A ``require`` stand-in that records module names and hands back stub exports."""
    seen = []

    def fake_require(module, *symbols, needed_by, **kwargs):
        seen.append(module)
        if module == fail_module:
            raise ImportError(f"{needed_by} requires {module}: missing")
        exports = {symbol.split(".")[0]: (lambda *a, **k: None) for symbol in symbols}
        return SimpleNamespace(**exports)

    return seen, fake_require


@pytest.mark.parametrize("use_cutedsl", [False, True])
@pytest.mark.parametrize("rmsnorm", [False, True])
def test_gdp_custom_provider_is_not_gated_by_default_recurrence(monkeypatch, use_cutedsl, rmsnorm):
    from megatron.core.ops.ssm.gdp import mixer

    target = lambda *args, **kwargs: None
    calls = []

    class Provider:
        def gated_delta_product(self):
            calls.append("bound")
            return target

    seen, fake_require = _recording_require(
        fail_module="megatron.core.ops.ssm.gdp.norm" if rmsnorm else None
    )
    monkeypatch.setattr(mixer, "require", fake_require)
    monkeypatch.setattr(mixer, "assert_causal_conv1d_deterministic", lambda _mode: None)
    monkeypatch.setattr(mixer, "packed_cp_conv_supported", lambda: True)
    monkeypatch.setattr(mixer, "build_module", _stop_before_parameters)
    model = mixer.GatedDeltaProductMixer.__new__(mixer.GatedDeltaProductMixer)
    expected = ImportError if rmsnorm else _BeforeParameters
    with pytest.raises(expected):
        model.__init__(
            _config(gdp_cutedsl_kernel=use_cutedsl),
            mixer.GatedDeltaProductMixerSubmodules(),
            128,
            pg_collection=_groups(),
            kernel_backend=Provider(),
            rmsnorm=rmsnorm,
        )
    assert model.gdp_kernel is target
    assert calls == ["bound"]
    assert not model._parameters and not model._modules
    assert "causal_conv1d" in seen
    assert ("fla.modules.l2norm" in seen) is (not use_cutedsl)
    # The recurrence itself is never required by the mixer: the provider owns that choice.
    assert not any(name.startswith(("fla.ops", "gdp_attn")) for name in seen)


@pytest.mark.parametrize("variant", ["gdn", "gdn2"])
@pytest.mark.parametrize("normalize", [False, True])
def test_gdn_custom_provider_does_not_require_default_recurrence(monkeypatch, variant, normalize):
    from megatron.core.ops.ssm.gated_delta import common, gdn, gdn2

    target = lambda *args, **kwargs: None
    calls = []

    class Provider:
        def gated_delta_rule(self, variant):
            calls.append(variant)
            return target

    seen, fake_require = _recording_require()
    monkeypatch.setattr(common, "require", fake_require)
    monkeypatch.setattr(common, "build_module", _stop_before_parameters)
    cls = gdn.GatedDeltaNet if variant == "gdn" else gdn2.GatedDeltaNet2
    model = cls.__new__(cls)
    with pytest.raises(_BeforeParameters):
        model.__init__(
            _config(),
            common.GatedDeltaNetSubmodules(),
            pg_collection=_groups(),
            kernel_backend=Provider(),
            use_qk_l2norm=normalize,
        )
    assert model.gated_delta_rule is target
    assert calls == [variant]
    assert seen[0] == "fla.modules.convolution"
    assert ("fla.modules.l2norm" in seen) is normalize
    assert not any(name.startswith("fla.ops") for name in seen)
    assert not model._parameters and not model._modules


@pytest.mark.parametrize("owner", ["mla", "dsv4", "compressor", "csa_indexer"])
def test_fused_rope_dependencies_are_checked_before_attention_allocation(monkeypatch, owner):
    from megatron.core.ops.attention import dsv4, mla
    from megatron.core.ops.attention.csa import modules as csa

    for module in (dsv4, mla, csa):
        _, fake_require = _recording_require(
            fail_module="megatron.core.fusions.fused_mla_yarn_rope_apply"
        )
        monkeypatch.setattr(module, "require", fake_require)
    config = SimpleNamespace(apply_rope_fusion=True, deterministic_mode=False)
    with pytest.raises(ImportError, match="fused_mla_yarn_rope_apply"):
        if owner == "mla":
            mla.AbsorbedMLASelfAttention(config, submodules=None, layer_number=1)
        elif owner == "dsv4":
            dsv4.DSv4HybridSelfAttention(
                config, submodules=None, layer_number=1, pg_collection=_groups()
            )
        elif owner == "compressor":
            csa.Compressor(config, None, compress_ratio=4, head_dim=16, pg_collection=_groups())
        else:
            csa.CSAIndexer(config, None, compress_ratio=4, pg_collection=_groups())


@pytest.mark.parametrize("owner", ["mla", "dsv4"])
def test_disabled_rope_fusion_does_not_import_or_check_its_kernel(monkeypatch, owner):
    from megatron.core.ops.attention import dsv4, mla
    from megatron.core.transformer.attention import Attention

    monkeypatch.setitem(sys.modules, "megatron.core.fusions.fused_mla_yarn_rope_apply", None)
    monkeypatch.setattr(Attention, "__init__", _stop_before_parameters)
    for module in (dsv4, mla):
        monkeypatch.setattr(
            module, "require", lambda *a, **k: pytest.fail(f"Disabled fusion checked {a}")
        )
    config = SimpleNamespace(apply_rope_fusion=False)
    with pytest.raises(_BeforeParameters):
        if owner == "mla":
            mla.AbsorbedMLASelfAttention(config, None, 1, pg_collection=_groups())
        else:
            dsv4.DSv4HybridSelfAttention(
                config, submodules=None, layer_number=1, pg_collection=_groups()
            )


# ---------------------------------------------------------------------------------------------
# Architecture checks
# ---------------------------------------------------------------------------------------------


def test_operation_constructors_do_not_gate_on_kernel_availability_flags():
    import megatron.core.ops

    root = Path(megatron.core.ops.__file__).parent
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if (
                not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                or node.name != "__init__"
            ):
                continue
            for child in ast.walk(node):
                if isinstance(child, ast.Name) and child.id.startswith("HAVE_"):
                    # These guard existing TE class checks/GTP integration, not kernel selection.
                    assert child.id in {"HAVE_TE", "HAVE_GTP"}, (path, child.lineno, child.id)


_CONSTRUCTION_TIME = ("__init__", "bind_dynamic_inference_kernels", "_setup_variant_attrs")


def test_require_is_only_called_at_construction_time():
    """``require`` binds kernels once; nothing may call it from a forward or helper method."""
    import megatron.core.ops

    root = Path(megatron.core.ops.__file__).parent
    offenders = []
    for path in root.rglob("*.py"):
        if path.name in ("backends.py", "_backends.py"):
            continue  # selectors are construction-time by contract
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if node.name in _CONSTRUCTION_TIME:
                continue
            for child in ast.walk(node):
                if (
                    isinstance(child, ast.Call)
                    and isinstance(child.func, ast.Name)
                    and child.func.id == "require"
                ):
                    offenders.append((path.relative_to(root).as_posix(), node.name, child.lineno))
    assert not offenders, offenders
