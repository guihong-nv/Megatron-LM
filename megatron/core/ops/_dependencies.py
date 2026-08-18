# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""One-time dependency and API checks for operation backends."""

from __future__ import annotations

import importlib
import importlib.metadata
import threading
from dataclasses import dataclass
from types import MappingProxyType, ModuleType
from typing import Iterable, Mapping
from unittest.mock import Mock

from packaging.version import Version


@dataclass(frozen=True)
class DependencyOutcome:
    """Cached result of importing one optional dependency module."""

    module: ModuleType | None = None
    error: BaseException | None = None
    missing: bool = False


@dataclass(frozen=True)
class _SymbolOutcome:
    """Cached result of validating an exact module API."""

    values: Mapping[str, object] | None = None
    missing: tuple[str, ...] = ()
    placeholders: tuple[str, ...] = ()
    error: Exception | None = None


_MODULE_OUTCOMES: dict[str, DependencyOutcome] = {}
_SYMBOL_OUTCOMES: dict[tuple[str, tuple[str, ...]], _SymbolOutcome] = {}
_VERSION_OUTCOMES: dict[str, Version | BaseException] = {}
_LOCK = threading.RLock()


def _import_once(module_name: str) -> DependencyOutcome:
    """Import a module once and cache both successful and failed outcomes."""
    with _LOCK:
        cached = _MODULE_OUTCOMES.get(module_name)
        if cached is not None:
            return cached

        try:
            outcome = DependencyOutcome(module=importlib.import_module(module_name))
        except ModuleNotFoundError as error:
            requested_root = module_name.split(".", maxsplit=1)[0]
            missing = error.name in {requested_root, module_name}
            outcome = DependencyOutcome(error=error, missing=missing)
        except Exception as error:
            outcome = DependencyOutcome(error=error)

        _MODULE_OUTCOMES[module_name] = outcome
        return outcome


def optional_module(module_name: str, *, purpose: str | None = None) -> ModuleType | None:
    """Return an optional module, or ``None`` only when that module is not installed.

    A broken installation, missing transitive library, or native-loader failure is reported
    instead of being treated as an unavailable optional backend.
    """
    outcome = _import_once(module_name)
    if outcome.module is not None:
        return outcome.module
    if outcome.missing:
        return None
    context = f" required by {purpose}" if purpose else ""
    raise ImportError(
        f"Failed to import optional module '{module_name}'{context}"
    ) from outcome.error


def require_module(module_name: str, *, purpose: str | None = None) -> ModuleType:
    """Return a selected dependency module or raise an actionable construction-time error."""
    outcome = _import_once(module_name)
    if outcome.module is not None:
        return outcome.module
    context = f" for {purpose}" if purpose else ""
    if outcome.missing:
        raise ImportError(
            f"Backend dependency '{module_name}' is required{context}"
        ) from outcome.error
    raise ImportError(
        f"Backend dependency '{module_name}' failed to import{context}"
    ) from outcome.error


def _inspect_symbols_once(
    module_name: str, symbols: tuple[str, ...], module: ModuleType
) -> _SymbolOutcome:
    """Validate one exact symbol set and cache both successes and failures."""
    key = (module_name, symbols)
    with _LOCK:
        cached = _SYMBOL_OUTCOMES.get(key)
        if cached is not None:
            return cached

        values: dict[str, object] = {}
        missing: list[str] = []
        try:
            for symbol in symbols:
                try:
                    values[symbol] = getattr(module, symbol)
                except AttributeError:
                    missing.append(symbol)
        except Exception as error:
            outcome = _SymbolOutcome(error=error)
        else:
            placeholders = tuple(
                sorted(symbol for symbol, target in values.items() if isinstance(target, Mock))
            )
            outcome = _SymbolOutcome(
                values=MappingProxyType(values),
                missing=tuple(sorted(missing)),
                placeholders=placeholders,
            )
        _SYMBOL_OUTCOMES[key] = outcome
        return outcome


def require_symbols(
    module_name: str, symbols: Iterable[str], *, purpose: str | None = None
) -> dict[str, object]:
    """Import a selected module and return its required API symbols."""
    requested = tuple(dict.fromkeys(symbols))
    module = require_module(module_name, purpose=purpose)
    outcome = _inspect_symbols_once(module_name, requested, module)
    if outcome.error is not None:
        context = f" for {purpose}" if purpose else ""
        raise ImportError(
            f"Backend dependency '{module_name}' failed API validation{context}"
        ) from outcome.error
    if outcome.missing:
        context = f" for {purpose}" if purpose else ""
        raise ImportError(
            f"Backend dependency '{module_name}' is missing required API{context}: "
            f"{', '.join(outcome.missing)}"
        )
    if outcome.placeholders:
        context = f" for {purpose}" if purpose else ""
        raise ImportError(
            f"Backend dependency '{module_name}' exposes placeholder API{context}: "
            f"{', '.join(outcome.placeholders)}"
        )
    return dict(outcome.values or {})


def dependency_version(distribution_name: str) -> Version:
    """Read and cache an installed distribution version."""
    with _LOCK:
        cached = _VERSION_OUTCOMES.get(distribution_name)
        if cached is None:
            try:
                try:
                    version_text = importlib.metadata.version(distribution_name)
                except importlib.metadata.PackageNotFoundError as metadata_error:
                    module = optional_module(distribution_name.replace("-", "_"))
                    module_version = (
                        getattr(module, "__version__", None) if module is not None else None
                    )
                    if module_version is None:
                        raise metadata_error
                    version_text = str(module_version)
                cached = Version(version_text)
            except Exception as error:
                cached = error
            _VERSION_OUTCOMES[distribution_name] = cached

    if isinstance(cached, BaseException):
        raise ImportError(
            f"Backend distribution '{distribution_name}' is not installed"
        ) from cached
    return cached


def require_version(distribution_name: str, minimum: str, *, purpose: str | None = None) -> Version:
    """Require a minimum selected-backend version and return the installed version."""
    installed = dependency_version(distribution_name)
    required = Version(minimum)
    if installed < required:
        context = f" for {purpose}" if purpose else ""
        raise ImportError(
            f"Backend distribution '{distribution_name}>={required}' is required{context}; "
            f"found {installed}"
        )
    return installed


def _reset_dependency_cache() -> None:
    """Clear cached dependency outcomes. Intended for unit tests only."""
    with _LOCK:
        _MODULE_OUTCOMES.clear()
        _SYMBOL_OUTCOMES.clear()
        _VERSION_OUTCOMES.clear()
