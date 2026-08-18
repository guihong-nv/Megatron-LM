# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Configuration-time operation backend selection."""

from .config import Backend, Operation, OpsBackendConfig
from .provider import BackendSpecProvider, get_backend, get_backend_spec_provider

__all__ = [
    "Backend",
    "BackendSpecProvider",
    "Operation",
    "OpsBackendConfig",
    "get_backend",
    "get_backend_spec_provider",
]
