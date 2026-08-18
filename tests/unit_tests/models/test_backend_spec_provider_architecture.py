# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Static enforcement tests for the BackendSpecProvider construction boundary."""

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
BACKENDS_PATH = REPO_ROOT / "megatron/core/models/backends.py"
PROVIDER_CONSUMERS = (
    REPO_ROOT / "megatron/core/models/gpt/gpt_layer_specs.py",
    REPO_ROOT / "megatron/core/models/gpt/moe_module_specs.py",
    REPO_ROOT / "megatron/core/models/gpt/experimental_attention_variant_module_specs.py",
    REPO_ROOT / "megatron/core/transformer/mla_qk_norm_config.py",
    REPO_ROOT / "megatron/core/transformer/multi_token_prediction.py",
)
INFERENCE_CONSTRUCTOR_ALLOWLIST = {
    ("megatron/core/models/gpt/gpt_layer_specs.py", "get_gpt_layer_with_inference_submodules"),
    ("megatron/core/models/gpt/moe_module_specs.py", "get_inference_optimized_moe_spec"),
}


def _called_name(node: ast.Call) -> str | None:
    """Return the simple name of a called function or class."""
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def _enclosing_function(tree: ast.AST, node: ast.Call) -> str | None:
    """Return the innermost function containing a node."""
    functions = [
        candidate
        for candidate in ast.walk(tree)
        if isinstance(candidate, (ast.FunctionDef, ast.AsyncFunctionDef))
        and candidate.lineno <= node.lineno <= (candidate.end_lineno or candidate.lineno)
    ]
    return max(functions, key=lambda function: function.lineno).name if functions else None


def test_provider_consumers_do_not_construct_training_providers_directly():
    """Concrete provider construction must stay in the central construction function."""
    violations = []

    for path in (REPO_ROOT / "megatron/core").rglob("*.py"):
        if path == BACKENDS_PATH:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imported_provider_names = {
            alias.asname or alias.name: alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            for alias in node.names
            if alias.name.endswith("SpecProvider")
        }
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            called_name = _called_name(node)
            provider_name = imported_provider_names.get(called_name, called_name)
            if not provider_name or not provider_name.endswith("SpecProvider"):
                continue
            if provider_name == "BackendSpecProvider":
                continue

            relative_path = str(path.relative_to(REPO_ROOT))
            location = (relative_path, _enclosing_function(tree, node))
            if (
                provider_name == "InferenceSpecProvider"
                and location in INFERENCE_CONSTRUCTOR_ALLOWLIST
            ):
                continue
            violations.append(f"{relative_path}:{node.lineno} ({provider_name})")

    assert not violations, "Concrete provider construction bypasses the factory: " + ", ".join(
        violations
    )


def test_provider_protocol_declares_every_slot_used_by_migrated_consumers():
    """A provider method cannot be used by migrated construction code without being declared."""
    tree = ast.parse(BACKENDS_PATH.read_text(encoding="utf-8"), filename=str(BACKENDS_PATH))
    protocol = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "BackendSpecProvider"
    )
    declared_methods = {
        node.name
        for node in protocol.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    called_methods = set()
    for path in PROVIDER_CONSUMERS:
        consumer = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        called_methods.update(
            node.func.attr
            for node in ast.walk(consumer)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "backend"
        )

    assert "linear" in declared_methods
    assert called_methods <= declared_methods
