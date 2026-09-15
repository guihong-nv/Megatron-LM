# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Canonical operation ownership and the deprecated forwarders left at the old paths."""

import ast
import importlib
import importlib.util
import pickle
import subprocess
import sys
from pathlib import Path

import pytest

from tests.unit_tests.ops.deprecated_paths import FORWARDED, PACKAGE_MARKERS

_DEPRECATED_PACKAGES = (
    "megatron.core.ssm",
    "megatron.core.transformer.experimental_attention_variant",
)

_OPERATION_MODULES = (
    "ops.ssm.mamba2.mixer",
    "ops.ssm.mamba2.context_parallel",
    "ops.ssm.gdp.mixer",
    "ops.ssm.gdp.context_parallel",
    "ops.ssm.gated_delta.common",
    "ops.ssm.gated_delta.gdn",
    "ops.ssm.gated_delta.gdn2",
    "ops.ssm.context_parallel.chunkwise",
    "ops.ssm.context_parallel.gdp",
    "ops.ssm.context_parallel.gdp_common",
    "ops.ssm.context_parallel.gdp_cutedsl",
    "ops.ssm.common.causal_conv1d_cp",
    "ops.ssm.common.packed_seq",
    "ops.ssm.common.checkpointing",
    "ops.attention.dsa.modules",
    "ops.attention.csa.modules",
    "ops.attention.mla",
    "ops.attention.dsv4",
)


@pytest.mark.parametrize("module", _OPERATION_MODULES)
def test_operation_module_has_canonical_source(module):
    path = "megatron.core." + module
    if (
        module == "ops.ssm.context_parallel.gdp_cutedsl"
        and importlib.util.find_spec("gdp_attn") is None
    ):
        with pytest.raises(ImportError, match="CuTeDSL GDP chunkwise CP backend is unavailable"):
            importlib.import_module(path)
        return
    target = importlib.import_module(path)
    assert target.__name__ == path
    assert Path(target.__file__).as_posix().endswith(path.replace(".", "/") + ".py")


@pytest.mark.parametrize(
    ("module", "name"),
    [
        ("ops.ssm.mamba2.mixer", "MambaMixer"),
        ("ops.ssm.mamba2.mixer", "MambaMixerSubmodules"),
        ("ops.ssm.gdp.mixer", "GatedDeltaProductMixer"),
        ("ops.ssm.gated_delta.gdn", "GatedDeltaNet"),
        ("ops.ssm.gated_delta.gdn2", "GatedDeltaNet2"),
        ("ops.ssm.gated_delta.common", "GatedDeltaNetSubmodules"),
        ("ops.ssm.common.inference", "SSMDynamicInferenceMixin"),
        ("ops.ssm.context_parallel.chunkwise", "PackedSequenceCPMetadata"),
        ("ops.attention.dsa.modules", "DSAttention"),
        ("ops.attention.dsa.modules", "DSAIndexer"),
        ("ops.attention.csa.modules", "Compressor"),
        ("ops.attention.csa.modules", "CompressedSparseAttention"),
        ("ops.attention.mla", "AbsorbedMLASelfAttention"),
        ("ops.attention.dsv4", "DSv4HybridSelfAttention"),
        ("transformer.dsa_loss", "DSAIndexerLossLoggingHelper"),
        ("transformer.dsa_loss", "DSAIndexerLossAutoScaler"),
        ("transformer.mamba_layer", "MambaLayer"),
        ("transformer.mamba_layer", "MambaLayerSubmodules"),
        ("transformer.mlp_layer", "MLPLayer"),
        ("transformer.mamba_layer_config", "MambaLayerConfig"),
        ("transformer.gdn_layer_config", "GDNLayerConfig"),
        ("transformer.mlp_layer_config", "MLPLayerConfig"),
        ("transformer.dsa_layer_config", "DSALayerConfig"),
        ("inference.ssm_config", "SSMChunking"),
    ],
)
def test_class_pickle_records_its_canonical_owner(module, name):
    path = "megatron.core." + module
    target = getattr(importlib.import_module(path), name)
    assert target.__module__ == path
    assert pickle.loads(pickle.dumps(target)) is target


@pytest.mark.parametrize("reverse", [False, True])
def test_construction_import_order_does_not_load_deprecated_packages(reverse):
    modules = (
        "megatron.core.inference.config",
        "megatron.core.transformer.mamba_layer",
        "megatron.core.models.gpt.experimental_attention_variant_module_specs",
        "megatron.core.models.hybrid.hybrid_layer_specs",
    )
    code = f"""
import importlib
import sys
for module in {modules[::-1] if reverse else modules!r}:
    importlib.import_module(module)
for retired in {_DEPRECATED_PACKAGES!r}:
    assert not any(name == retired or name.startswith(retired + '.') for name in sys.modules)
"""
    subprocess.run([sys.executable, "-c", code], check=True, timeout=120)


def test_training_loss_state_does_not_import_attention_implementations():
    code = """
import sys
from megatron.core.transformer.dsa_loss import DSAIndexerLossLoggingHelper
assert DSAIndexerLossLoggingHelper.tracker == {}
assert 'megatron.core.ops.attention.dsa.modules' not in sys.modules
assert 'megatron.core.ops.attention.csa.modules' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True, timeout=120)


def test_gdn_construction_targets_are_canonical():
    from megatron.core.ops.ssm.gated_delta import common, gdn, gdn2, modules

    assert modules.GatedDeltaNet is gdn.GatedDeltaNet
    assert modules.GatedDeltaNet2 is gdn2.GatedDeltaNet2
    assert modules.GatedDeltaNetSubmodules is common.GatedDeltaNetSubmodules


@pytest.mark.parametrize("deprecated", _DEPRECATED_PACKAGES)
def test_in_tree_code_uses_canonical_paths(deprecated):
    """The forwarders exist for downstream callers; nothing in the tree may use them."""
    root = Path(__file__).resolve().parents[3]
    shim_root = root / deprecated.replace(".", "/")
    for directory in ("megatron", "examples", "tools", "tests/functional_tests"):
        for path in (root / directory).rglob("*"):
            if path.suffix == ".py":
                if shim_root in path.parents:
                    continue  # the forwarder modules themselves
                for node in ast.walk(ast.parse(path.read_text())):
                    if isinstance(node, ast.ImportFrom):
                        names = [node.module or ""]
                    elif isinstance(node, ast.Import):
                        names = [alias.name for alias in node.names]
                    else:
                        continue
                    assert not any(
                        name == deprecated or name.startswith(deprecated + ".") for name in names
                    ), path
            elif path.suffix in (".sh", ".yaml", ".yml"):
                assert deprecated not in path.read_text(), path


@pytest.mark.parametrize("marker", PACKAGE_MARKERS)
def test_deprecated_package_markers_import_silently(marker):
    """Importing the bare package is not deprecated by itself; its submodules warn."""
    import warnings

    sys.modules.pop(marker, None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        module = importlib.import_module(marker)
    assert module.__name__ == marker
    assert not [w for w in caught if marker in str(w.message)], [str(w.message) for w in caught]


@pytest.mark.parametrize("deprecated", sorted(FORWARDED))
def test_deprecated_path_forwards_to_canonical_module(deprecated):
    """Every pre-move module path still resolves, warns once, and hands out the same objects."""
    targets = FORWARDED[deprecated]
    sys.modules.pop(deprecated, None)
    absent_before = [target for target in targets if target not in sys.modules]
    with pytest.warns(DeprecationWarning, match=targets[0].replace(".", r"\.")):
        shim = importlib.import_module(deprecated)
    # Importing the forwarder must not import the implementation or its optional kernels.
    assert not any(target in sys.modules for target in absent_before)

    resolved: dict[str, str] = {}  # name -> first target that defines it
    for target in targets:
        if _optional_dependency_missing(target):
            continue
        canonical = importlib.import_module(target)
        public = getattr(canonical, "__all__", None) or [
            name for name in vars(canonical) if not name.startswith("_")
        ]
        for name in public:
            # A split module lists several targets; the first one defining a name wins,
            # so a later target's same-named object (``logger``) is not what the shim returns.
            owner = resolved.setdefault(name, target)
            if owner != target:
                continue
            if _is_forwarder_submodule(shim, name):
                # ``from old_pkg import child`` yields the old package's own (forwarding)
                # submodule, exactly as it did before the move.
                assert getattr(shim, name).__name__ == f"{deprecated}.{name}"
            else:
                assert getattr(shim, name) is getattr(canonical, name), (deprecated, name)
        assert set(public) <= set(shim.__all__)
    assert not hasattr(shim, "__wrapped__")  # dunder probes never import the target
    with pytest.raises(AttributeError, match=deprecated):
        shim.definitely_not_a_real_name


def test_pickle_recorded_under_deprecated_path_loads_canonical_class():
    """Whole-object checkpoints and cached specs written before the move keep loading."""
    from megatron.core.ops.ssm.mamba2.mixer import MambaMixer

    data = pickle.dumps(MambaMixer)
    old_owner = b"megatron.core.ssm.mamba_mixer"
    assert MambaMixer.__module__.encode() in data
    data = data.replace(MambaMixer.__module__.encode(), old_owner)
    sys.modules.pop(old_owner.decode(), None)  # the warning fires when the shim is (re)loaded
    with pytest.warns(DeprecationWarning):
        assert pickle.loads(data) is MambaMixer


def _is_forwarder_submodule(shim, name: str) -> bool:
    if not hasattr(shim, "__path__"):
        return False
    return importlib.util.find_spec(f"{shim.__name__}.{name}") is not None


def _optional_dependency_missing(target: str) -> bool:
    try:
        importlib.import_module(target)
    except ImportError as exc:
        # Kernel modules that hard-require an optional library (TileLang, CuTeDSL, ...) are
        # forwarded correctly even when that library is absent in the test environment.
        return getattr(exc, "name", None) not in (None, target) or "unavailable" in str(exc)
    return False
