# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Static enforcement tests for the one-class BackendSpecProvider architecture."""

import ast
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
CORE_ROOT = REPO_ROOT / "megatron/core"
OPS_ROOT = CORE_ROOT / "ops"
FUSION_ROOT = OPS_ROOT / "fusion"
COMPATIBILITY_FACADE = CORE_ROOT / "models/backends.py"
MANAGED_CONSUMERS = (
    CORE_ROOT / "models/gpt/gpt_layer_specs.py",
    CORE_ROOT / "models/gpt/moe_module_specs.py",
    CORE_ROOT / "models/gpt/experimental_attention_variant_module_specs.py",
    CORE_ROOT / "transformer/mla_qk_norm_config.py",
    CORE_ROOT / "transformer/multi_token_prediction.py",
)
FUSION_RESOLVERS = {"resolve_norm_linear", "resolve_fused_mlp", "resolve_bias_dropout_add"}
_HAVE_NAME = re.compile(r"^_?HAVE_[A-Z0-9_]+$")


def _parse(path: Path) -> ast.Module:
    """Parse one Python source file."""
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _managed_paths() -> tuple[Path, ...]:
    """Return production files governed by this migration milestone."""
    ops_paths = tuple(OPS_ROOT.rglob("*.py")) if OPS_ROOT.exists() else ()
    return tuple(dict.fromkeys((COMPATIBILITY_FACADE, *MANAGED_CONSUMERS, *ops_paths)))


def _dotted_name(node: ast.AST) -> str | None:
    """Return a dotted name represented by a Name or Attribute expression."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _dotted_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return None


def test_backend_spec_provider_is_the_only_provider_class():
    """Presets and optional backends must not introduce provider subclasses."""
    provider_classes = []

    for path in CORE_ROOT.rglob("*.py"):
        tree = _parse(path)
        provider_classes.extend(
            (str(path.relative_to(REPO_ROOT)), node.lineno, node.name)
            for node in ast.walk(tree)
            if isinstance(node, ast.ClassDef) and node.name.endswith("SpecProvider")
        )

    assert (
        len(provider_classes) == 1 and provider_classes[0][2] == "BackendSpecProvider"
    ), f"Expected exactly one BackendSpecProvider class, found: {provider_classes}"


def test_managed_provider_and_consumer_paths_do_not_use_have_flags():
    """Managed paths resolve selected dependencies instead of consulting HAVE_* globals."""
    violations = []

    for path in _managed_paths():
        tree = _parse(path)
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Name):
                names.append(node.id)
            elif isinstance(node, ast.Attribute):
                names.append(node.attr)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    names.extend((alias.name.rsplit(".", 1)[-1], alias.asname or ""))
            for name in names:
                if _HAVE_NAME.fullmatch(name):
                    violations.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno} ({name})")

    assert not violations, "HAVE_* usage remains in managed paths: " + ", ".join(violations)


def test_managed_consumers_do_not_branch_on_provider_type():
    """Operation slots, not provider concrete types, determine behavior."""
    violations = []

    for path in MANAGED_CONSUMERS:
        tree = _parse(path)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id == "isinstance" and len(node.args) >= 2:
                    type_names = {
                        name
                        for name in (
                            _dotted_name(candidate) for candidate in ast.walk(node.args[1])
                        )
                        if name is not None
                    }
                    if any(name.endswith("SpecProvider") for name in type_names):
                        violations.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")
                elif node.func.id == "type" and node.args:
                    subject = _dotted_name(node.args[0])
                    if subject and subject.rsplit(".", 1)[-1] in {"backend", "provider"}:
                        violations.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")
            elif isinstance(node, ast.Attribute) and node.attr == "__class__":
                subject = _dotted_name(node.value)
                if subject and subject.rsplit(".", 1)[-1] in {"backend", "provider"}:
                    violations.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")

    assert not violations, "Provider-type branching remains: " + ", ".join(violations)


def test_optional_vendor_imports_are_not_at_ops_module_scope():
    """Importing megatron.core.ops must not import an unselected optional package."""
    optional_roots = {
        "apex",
        "causal_conv1d",
        "flash_attn",
        "flashinfer",
        "fla",
        "liger_kernel",
        "mamba_ssm",
        "nvidia_kitchen",
        "transformer_engine",
        "triton",
    }
    violations = []

    for path in OPS_ROOT.rglob("*.py"):
        tree = _parse(path)
        functions = [
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Import, ast.ImportFrom)):
                continue
            if any(
                function.lineno <= node.lineno <= (function.end_lineno or function.lineno)
                for function in functions
            ):
                continue
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module or ""]
            else:
                continue
            for module in modules:
                if module.split(".", 1)[0] in optional_roots:
                    violations.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno} ({module})")

    assert not violations, "Optional vendor imports execute at module scope: " + ", ".join(
        violations
    )


def test_fusion_contracts_and_resolvers_live_under_core_ops_fusion():
    """Cross-operation kernels have one explicit home separate from single-op adapters."""
    assert FUSION_ROOT.is_dir(), "Missing megatron/core/ops/fusion package"
    assert not (OPS_ROOT / "fusions").exists(), "Use singular megatron/core/ops/fusion"

    expected_modules = {
        "resolve_norm_linear": FUSION_ROOT / "norm_linear.py",
        "resolve_fused_mlp": FUSION_ROOT / "fused_mlp.py",
        "resolve_bias_dropout_add": FUSION_ROOT / "bias_dropout_add.py",
    }
    definitions = {name: [] for name in FUSION_RESOLVERS}
    for path in OPS_ROOT.rglob("*.py"):
        tree = _parse(path)
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name in FUSION_RESOLVERS:
                    definitions[node.name].append(path)

    for name, expected_path in expected_modules.items():
        assert expected_path.is_file(), f"Missing {expected_path.relative_to(REPO_ROOT)}"
        assert definitions[name] == [expected_path], (
            f"{name} must be defined once in {expected_path.relative_to(REPO_ROOT)}; "
            f"found {[str(path.relative_to(REPO_ROOT)) for path in definitions[name]]}"
        )


def test_managed_consumers_do_not_import_legacy_fusion_modules():
    """Managed consumers obtain fusion slots from core.ops rather than core.fusions."""
    violations = []

    for path in (COMPATIBILITY_FACADE, *MANAGED_CONSUMERS):
        tree = _parse(path)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
                "megatron.core.fusions"
            ):
                violations.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno} ({node.module})")

    assert not violations, "Legacy fusion imports remain in managed paths: " + ", ".join(violations)
