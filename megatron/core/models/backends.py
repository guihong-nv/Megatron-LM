# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Compatibility imports for operation backend selection.

The canonical implementation lives in :mod:`megatron.core.ops`. This module intentionally
defines no provider implementations.
"""

from megatron.core.ops import (
    Backend,
    BackendSpecProvider,
    Operation,
    OpsBackendConfig,
    get_backend,
    get_backend_spec_provider,
)

BackendName = str

__all__ = [
    "Backend",
    "BackendName",
    "BackendSpecProvider",
    "Operation",
    "OpsBackendConfig",
    "get_backend",
    "get_backend_spec_provider",
]
