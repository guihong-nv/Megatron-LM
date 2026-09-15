# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Construction-time binding for the existing optional DSA hook interfaces."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

from megatron.core.ops._backends import require

if TYPE_CHECKING:
    from torch import Tensor

_LOGGER = logging.getLogger(__name__)

# What each fused backend's adapter module needs before it can be imported. The adapter's
# own imports are the authoritative list; these name the *native* libraries so a missing
# one is reported against the operation rather than as a stack trace inside the adapter.
_NATIVE_REQUIREMENTS: dict[str, tuple[tuple[str, tuple[str, ...]], ...]] = {
    "tilelang": (("tilelang", ()), ("triton", ())),
    "cudnn": (
        (
            "cudnn",
            (
                "DSA.indexer_top_k_wrapper",
                "DSA.indexer_forward_wrapper",
                "DSA.indexer_backward_wrapper",
                "DSA.dense_indexer_backward_wrapper",
                "DSA.sparse_attn_score_recompute_wrapper",
                "DSA.dense_attn_score_recompute_wrapper",
                "DSA.sparse_attention_backward_wrapper",
            ),
        ),
        ("flash_mla", ("flash_mla_sparse_fwd",)),
    ),
}


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

    def log_declined(self, hook_name: str) -> None:
        """Keep fallback diagnostics without wrapping or resolving a kernel call."""
        _LOGGER.debug(
            "DSA fused backend %s %s declined; falling back (backend returned None).",
            self.backend,
            hook_name,
        )


def select_dsa_kernels(backend: str, *, fused: bool = True) -> DSAKernels:
    """Bind the named backend once; do not resolve it from a model forward.

    ``backend`` is ``config.dsa_kernel_backend``; ``fused`` is False when the attention
    backend is ``unfused``, which disables every optional fused hook. A selected backend
    whose libraries are missing is an ``ImportError``; nothing falls back silently.
    """
    from megatron.core.ops.attention.dsa.dsa_kernels import backend_module_name

    module_name = backend_module_name(backend)  # validates the name even when unfused
    if not fused or module_name is None:
        return DSAKernels()
    needed_by = f"DSA kernel backend {backend!r}"
    for module, symbols in _NATIVE_REQUIREMENTS[backend]:
        require(module, *symbols, needed_by=needed_by)
    adapter = require(module_name, needed_by=needed_by)
    return DSAKernels(
        backend=backend,
        run_fused_qk_topk=getattr(adapter, "run_fused_qk_topk", None),
        run_fused_qk_topk_with_loss=getattr(adapter, "run_fused_qk_topk_with_loss", None),
        run_fused_absorbed_sparse_attention=getattr(
            adapter, "run_fused_absorbed_sparse_attention", None
        ),
        run_fused_dsa_attention=getattr(adapter, "run_fused_dsa_attention", None),
    )
