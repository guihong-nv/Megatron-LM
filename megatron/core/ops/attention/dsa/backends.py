# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Construction-time binding for the existing optional DSA hook interfaces."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from importlib import import_module
from typing import TYPE_CHECKING, Callable

from megatron.core.ops.attention.dsa.kernel_metadata import (
    CUDNN_ATTENTION,
    CUDNN_FULL,
    CUDNN_LOSS,
    CUDNN_TOPK,
    DSA_INDEXER_REFERENCE,
    DSA_REFERENCE,
    TILELANG_ATTENTION,
    TILELANG_LOSS,
    TILELANG_TOPK,
)
from megatron.core.ops.kernel_metadata import DeterminismPolicy, KernelMetadata, validate_kernels

if TYPE_CHECKING:
    from torch import Tensor

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class DSAKernels:
    """Concrete backend hooks, with None for hooks the selected backend does not supply.

    Signatures are those in ``dsa_tilelang_kernels`` and ``dsa_cudnn_kernels``.
    A hook can also return None for unsupported runtime inputs. Callers then use
    the reference implementation. The object owns no tensors or model state.
    """

    backend: str = "none"
    run_fused_qk_topk: Callable[..., tuple[Tensor, Tensor | None] | None] | None = None
    run_fused_qk_topk_with_loss: (
        Callable[..., tuple[Tensor, Tensor | None, Tensor] | None] | None
    ) = None
    run_fused_absorbed_sparse_attention: Callable[..., Tensor | None] | None = None
    run_fused_dsa_attention: Callable[..., tuple[Tensor, Tensor] | None] | None = None
    metadata: tuple[KernelMetadata, ...] = ()

    def log_declined(self, hook_name: str) -> None:
        """Keep fallback diagnostics without wrapping or resolving a kernel call."""
        _LOGGER.debug(
            "DSA fused backend %s %s declined; falling back (backend returned None).",
            self.backend,
            hook_name,
        )


def select_dsa_kernels(
    backend: str, *, fused: bool = True, deterministic: bool = False
) -> DSAKernels:
    """Bind the named backend once; do not resolve it from a model forward.

    ``backend`` is ``config.dsa_kernel_backend``; ``fused`` is False when the attention
    backend is ``unfused``, which disables every optional fused hook.
    """
    from megatron.core.ops.attention.dsa.dsa_kernels import backend_module_name

    policy = DeterminismPolicy.WARN if deterministic else DeterminismPolicy.IGNORE
    module_name = backend_module_name(backend)  # validates the name even when unfused
    if not fused or module_name is None:
        validate_kernels((DSA_REFERENCE, DSA_INDEXER_REFERENCE), determinism=policy)
        return DSAKernels()
    try:
        declarations = (
            (TILELANG_TOPK, TILELANG_LOSS, TILELANG_ATTENTION)
            if backend == "tilelang"
            else (CUDNN_TOPK, CUDNN_LOSS, CUDNN_ATTENTION, CUDNN_FULL)
        )
        validate_kernels(declarations, determinism=policy)
        # Validate native requirements before loading the selected adapter.
        adapter = import_module(module_name)
    except (ImportError, OSError) as exc:
        raise RuntimeError(f"Failed to import DSA kernel backend {module_name}: {exc}") from exc
    return DSAKernels(
        backend=backend,
        run_fused_qk_topk=getattr(adapter, "run_fused_qk_topk", None),
        run_fused_qk_topk_with_loss=getattr(adapter, "run_fused_qk_topk_with_loss", None),
        run_fused_absorbed_sparse_attention=getattr(
            adapter, "run_fused_absorbed_sparse_attention", None
        ),
        run_fused_dsa_attention=getattr(adapter, "run_fused_dsa_attention", None),
        metadata=declarations,
    )
