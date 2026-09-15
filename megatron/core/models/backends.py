# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
from __future__ import annotations

import copy
from abc import abstractmethod
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Callable, Literal, Optional, Protocol, cast

import torch

from megatron.core.extensions.transformer_engine import (
    TEActivationOp,
    TEColumnParallelGroupedLinear,
    TEDotProductAttention,
    TELinear,
    TENorm,
    TERowParallelGroupedLinear,
    te_cross_entropy,
)
from megatron.core.fusions.fused_cross_entropy import (
    fused_vocab_parallel_cross_entropy as _fused_ce,
)
from megatron.core.tensor_parallel.cross_entropy import unfused_cross_entropy
from megatron.core.tensor_parallel.inference_layers import (
    InferenceColumnParallelLinear,
    InferenceLayerNormColumnParallelLinear,
    InferenceRowParallelLinear,
)
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer.dot_product_attention import DotProductAttention
from megatron.core.transformer.mlp import MLPSubmodules, TEActivationFunctionBuilder
from megatron.core.transformer.moe.experts import (
    GroupedMLPSubmodules,
    InferenceGroupedMLP,
    SequentialMLP,
)
from megatron.core.transformer.moe.moe_layer import ExpertsBuilder
from megatron.core.transformer.torch_norm import LayerNormBuilder, WrappedTorchNorm
from megatron.core.typed_torch import not_none
from megatron.core.utils import is_te_min_version

if TYPE_CHECKING:
    from megatron.core.ops.attention.dsa.backends import DSAKernels
    from megatron.core.ops.ssm.context_parallel.chunkwise import LinearAttentionCPBackend
    from megatron.core.ops.ssm.gated_delta import GatedDeltaRuleInterface
    from megatron.core.ops.ssm.mamba2.backends import MambaKernels

CrossEntropyTarget = Callable[
    [torch.Tensor, torch.Tensor, Optional[torch.distributed.ProcessGroup]], torch.Tensor
]
"""``loss = target(logits, labels, tp_group)``, with ``logits`` [s, b, vocab/tp] and the
target owning every reduction over ``tp_group``. Targets *consume* ``logits``: they subtract
the row max and exponentiate in place, so a caller that needs them afterwards passes a clone.
"""


def require(requirement: str, requested_by: str = "This backend", instead: str = "") -> None:
    """Refuse a backend whose optional package is missing or too old.

    Availability comes from the extension module's own ``HAVE_*`` flag, which is set by
    actually importing the package -- a package that is installed but unimportable (a broken
    CUDA build, a missing shared object) has to fail here, not later.

    The requirement is declared once, as ``REQUIRES`` on the provider, and checked wherever a
    caller wants an early, clear refusal. It is deliberately not called when a provider is
    constructed: a spec may be assembled without the optional package present, and it is
    instantiating the module that fails.
    """
    if requirement != "transformer_engine":
        raise ValueError(f"no availability check is known for {requirement!r}")
    from megatron.core.extensions.transformer_engine import HAVE_TE

    if not HAVE_TE:
        message = f"Transformer Engine is not installed, and {requested_by} needs it."
        raise ImportError(f"{message} {instead}".strip())


def select_cross_entropy(
    cross_entropy_loss_fusion: bool = False,
    cross_entropy_fusion_impl: str = "native",
    cuda_graph_impl: Optional[str] = None,
) -> CrossEntropyTarget:
    """Which cross entropy a config asks for, resolved once rather than on every forward.

    Shared by every provider: the choice depends on the config, not on which backend supplies
    the rest of the model, which is how it behaved before this was a provider decision.
    """
    if cross_entropy_fusion_impl not in ("native", "te"):
        raise ValueError(
            f"unknown cross_entropy_fusion_impl={cross_entropy_fusion_impl!r}. "
            "Valid choices: native, te"
        )
    if not cross_entropy_loss_fusion:
        return unfused_cross_entropy
    if cross_entropy_fusion_impl == "native":
        return _fused_ce

    # Full-iteration CUDA graphs capture the loss too, which TE only supports from 2.7.0.
    capturable = cuda_graph_impl == "full_iteration"
    if capturable and not is_te_min_version("2.7.0"):
        from megatron.core.utils import get_te_version

        raise AssertionError(
            "CUDA graph compatible cross entropy requires TransformerEngine >= 2.7.0, but "
            f"found version {get_te_version()}. Please upgrade TransformerEngine or set "
            "cuda_graph_impl to a value other than 'full_iteration'."
        )
    return partial(te_cross_entropy, cuda_graph_capturable=capturable)


class BackendSpecProvider(Protocol):
    """A protocol for providing the submodules used in Spec building."""

    def linear(self) -> type:
        """Which non-parallel linear module to use, if the backend supplies one."""
        ...

    @abstractmethod
    def column_parallel_linear(self) -> type:
        """Which column parallel linear module the backend uses"""
        ...

    @abstractmethod
    def row_parallel_linear(self) -> type:
        """Which row parallel linear module the backend uses"""
        ...

    @abstractmethod
    def fuse_layernorm_and_linear(self) -> bool:
        """Does the backend support a single module for layernorm and linear"""
        ...

    @abstractmethod
    def column_parallel_layer_norm_linear(self) -> Optional[type]:
        """Which module for sequential layernorm and linear"""
        ...

    @abstractmethod
    def layer_norm(
        self, rms_norm: bool = False, for_qk: bool = False, has_residual: bool = False
    ) -> LayerNormBuilder:
        """Which module for layernorm"""
        ...

    @abstractmethod
    def core_attention(self) -> type:
        """Which module to use for attention"""
        ...

    @abstractmethod
    def grouped_mlp_modules(self, moe_use_grouped_gemm: bool) -> ExpertsBuilder:
        """Which module and submodules to use for grouped mlp"""
        ...

    @abstractmethod
    def activation_func(self) -> TEActivationFunctionBuilder | None:
        """Which module to use for activation function"""
        ...

    # The slots below were added after this protocol was first published. They are not
    # abstract, so a provider written against the earlier contract stays instantiable, and
    # they have no body, so inheriting one is not mistaken for implementing it -- callers ask
    # through ``backend_slot``, which supplies the previous behaviour instead.

    def mlp_module(self, grouped: bool = False) -> type:
        """Which module to use for the dense MLP block."""
        ...

    def moe_router(self) -> Optional[type]:
        """Which MoE router to use, or None to keep the MoESubmodules default."""
        ...

    def vocab_parallel_cross_entropy(self) -> CrossEntropyTarget:
        """Which vocab-parallel cross entropy to use."""
        ...

    # The kernel slots below take no implementation-selection arguments. A provider is
    # configured once, from ``KernelSelection``, when it is built; ``variant`` on the GDN
    # slot names the recurrence being built (like ``rms_norm`` on ``layer_norm``), not an
    # implementation.

    def mamba_kernels(self) -> MambaKernels:
        """Which Mamba2 scan (and optional fused conv+scan) callables to bind."""
        ...

    def gated_delta_rule(self, variant: Literal["gdn", "gdn2"]) -> GatedDeltaRuleInterface:
        """Which GDN-family recurrence to bind during construction."""
        ...

    def gated_delta_product(self) -> Callable:
        """Which GDP training callable to bind during construction."""
        ...

    def gated_delta_product_cp_backend(self) -> LinearAttentionCPBackend:
        """Which chunkwise context-parallel adapter GDP binds when CP > 1."""
        ...

    def dsa_kernels(self) -> DSAKernels:
        """Which concrete optional sparse-attention hooks to bind during construction."""
        ...


@dataclass(frozen=True)
class KernelSelection:
    """The per-operation settings a provider's kernel slots are configured with.

    These are read from ``TransformerConfig`` once, when the provider is built, so model
    and operation code never passes implementation choices to a slot at call time.
    """

    deterministic: bool = False  # selects the Torch reference recurrence for GDN/GDN2
    mamba_mem_eff_path: bool = False
    gdp_cutedsl: bool = False
    gdp_recompute_chunk_num: int = 0
    dsa_backend: str = "none"
    dsa_fused: bool = True

    @classmethod
    def from_config(cls, config: object) -> "KernelSelection":
        """Read the selection-relevant fields; absent fields keep their defaults."""
        attention_backend = getattr(config, "attention_backend", None)
        return cls(
            deterministic=bool(getattr(config, "deterministic_mode", False)),
            mamba_mem_eff_path=bool(getattr(config, "use_mamba_mem_eff_path", False)),
            gdp_cutedsl=bool(getattr(config, "gdp_cutedsl_kernel", False)),
            gdp_recompute_chunk_num=int(
                getattr(config, "gdp_num_chunk_states_to_recompute", None) or 0
            ),
            dsa_backend=str(getattr(config, "dsa_kernel_backend", None) or "none"),
            dsa_fused=getattr(attention_backend, "name", attention_backend) != "unfused",
        )


class KernelSelectionMixin:
    """Implements the kernel slots of ``BackendSpecProvider`` from a ``KernelSelection``.

    Local and Transformer Engine providers pick the same SSM and sparse-attention kernels;
    TE does not supply its own. Sharing one implementation keeps the slot overridable by a
    provider that does (a partial provider forwards everything else to its fallback).
    """

    _kernel_selection: Optional[KernelSelection] = None

    @property
    def _kernels(self) -> KernelSelection:
        if self._kernel_selection is None:
            raise RuntimeError(
                f"{type(self).__name__} was built without a KernelSelection, so it cannot "
                "answer an SSM or sparse-attention slot. Build it with "
                "get_backend_from_config(config), pass kernels=KernelSelection.from_config("
                "config), or hand it to a module spec and let resolve_kernel_backend "
                "configure it from the model config."
            )
        return self._kernel_selection

    @_kernels.setter
    def _kernels(self, selection: Optional[KernelSelection]) -> None:
        self._kernel_selection = selection

    def with_kernel_selection(self, selection: KernelSelection) -> "KernelSelectionMixin":
        """This provider, configured with ``selection`` unless it already has one.

        A selection given at construction (``get_backend_from_config``) always wins. A bare
        provider handed to a spec builder -- ``TESpecProvider()`` -- is configured from the
        model config when a module binds its kernels, on a shallow copy, so a provider shared
        across a spec is never mutated and never silently selects defaults.
        """
        if self._kernel_selection is not None:
            return self
        configured = copy.copy(self)
        configured._kernel_selection = selection
        return configured

    def mamba_kernels(self) -> MambaKernels:
        """Mamba2 scan callables for the configured training path."""
        from megatron.core.ops.ssm.mamba2.backends import select_mamba_kernels

        return select_mamba_kernels(self._kernels.mamba_mem_eff_path)

    def gated_delta_rule(self, variant: Literal["gdn", "gdn2"]) -> GatedDeltaRuleInterface:
        """FLA recurrence, or the Torch reference in deterministic mode."""
        from megatron.core.ops.ssm.gated_delta.backends import select_gated_delta_rule

        return select_gated_delta_rule(variant, self._kernels.deterministic)

    def gated_delta_product(self) -> Callable:
        """FLA or CuTeDSL chunked gated delta product, as configured."""
        from megatron.core.ops.ssm.gdp.backends import select_gated_delta_product

        return select_gated_delta_product(self._kernels.gdp_cutedsl)

    def gated_delta_product_cp_backend(self) -> LinearAttentionCPBackend:
        """Chunkwise-CP adapter matching the configured GDP kernel."""
        from megatron.core.ops.ssm.gdp.backends import select_gdp_cp_backend

        return select_gdp_cp_backend(
            self._kernels.gdp_cutedsl, recompute_chunk_num=self._kernels.gdp_recompute_chunk_num
        )

    def dsa_kernels(self) -> DSAKernels:
        """Fused DSA hooks for the configured backend, or none."""
        from megatron.core.ops.attention.dsa.backends import select_dsa_kernels

        return select_dsa_kernels(self._kernels.dsa_backend, fused=self._kernels.dsa_fused)


class LocalSpecProvider(KernelSelectionMixin, BackendSpecProvider):
    """Every backend a Megatron-Core-only run uses."""

    # No optional package is required for the local backend.
    REQUIRES: Optional[str] = None

    def __init__(
        self,
        cross_entropy_loss_fusion: bool = False,
        cross_entropy_fusion_impl: str = "native",
        cuda_graph_impl: Optional[str] = None,
        kernels: Optional[KernelSelection] = None,
    ) -> None:
        self._cross_entropy_loss_fusion = cross_entropy_loss_fusion
        self._cross_entropy_fusion_impl = cross_entropy_fusion_impl
        self._cuda_graph_impl = cuda_graph_impl
        self._kernels = kernels  # None until configured; see KernelSelectionMixin

    def column_parallel_linear(self) -> type:
        """Which column parallel linear module the backend uses"""
        return ColumnParallelLinear

    def row_parallel_linear(self) -> type[RowParallelLinear]:
        """Which row parallel linear module the backend uses"""
        return RowParallelLinear

    def fuse_layernorm_and_linear(self) -> bool:
        """Does the backend choose a single module for layernorm and linear"""
        return False

    def column_parallel_layer_norm_linear(self) -> Optional[type]:
        """Which module for sequential layernorm and linear"""
        return None

    def layer_norm(
        self, rms_norm: bool = False, for_qk: bool = False, has_residual: bool = False
    ) -> LayerNormBuilder:
        """Which module to use for layer norm.

        Apex's fused LayerNorm when it is available, Torch's otherwise -- the same choice this
        made before, but decided here rather than by a module-level ``LNImpl`` that eight files
        each kept their own copy of and that this method used to mutate as a side effect.

        RMSNorm always comes from Torch: Apex's fused kernel implements LayerNorm only.
        """
        del for_qk, has_residual
        if rms_norm:
            return WrappedTorchNorm
        from megatron.core.fusions.fused_layer_norm import HAVE_FUSED_LAYER_NORM, FusedLayerNorm

        return FusedLayerNorm if HAVE_FUSED_LAYER_NORM else WrappedTorchNorm

    def core_attention(self) -> type:
        """Which module to use for attention"""
        return DotProductAttention

    def grouped_mlp_modules(self, moe_use_grouped_gemm: bool) -> ExpertsBuilder:
        """Which module and submodules to use for grouped mlp"""
        return partial(
            SequentialMLP,
            submodules=MLPSubmodules(
                linear_fc1=ColumnParallelLinear,
                linear_fc2=RowParallelLinear,
                activation_func=self.activation_func(),
            ),
        )

    def activation_func(self) -> TEActivationFunctionBuilder | None:
        """Which module to use for activation function"""
        return None

    def linear(self) -> type:
        """Megatron Core has no non-parallel linear; Transformer Engine is the one that does."""
        raise NotImplementedError(
            "This backend has no non-parallel linear. Use --transformer-impl transformer_engine."
        )

    def mlp_module(self, grouped: bool = False) -> type:
        """Megatron Core has one MLP block; grouped GEMM is a Transformer Engine feature."""
        del grouped
        from megatron.core.transformer.mlp import MLP

        return MLP

    def moe_router(self) -> Optional[type]:
        """Keep the MoESubmodules default."""
        return None

    def vocab_parallel_cross_entropy(self) -> CrossEntropyTarget:
        """Which vocab-parallel cross entropy to use."""
        return select_cross_entropy(
            self._cross_entropy_loss_fusion, self._cross_entropy_fusion_impl, self._cuda_graph_impl
        )



class InferenceSpecProvider(LocalSpecProvider):
    """Every backend an inference-optimized run uses.

    Attention and the grouped linears come from Transformer Engine; the inference-specific
    gains are in the linear and mixture-of-experts layers.
    """

    REQUIRES = "transformer_engine"

    def linear(self) -> type:
        """Which linear module TE backend uses"""
        return TELinear

    def column_parallel_linear(self) -> type:
        """Which column parallel linear module TE backend uses"""
        return InferenceColumnParallelLinear

    def row_parallel_linear(self) -> type[InferenceRowParallelLinear]:
        """Which row parallel linear module Inference backend uses"""
        return InferenceRowParallelLinear

    def fuse_layernorm_and_linear(self) -> bool:
        """TE backend chooses a single module for layernorm and linear"""
        return True

    def column_parallel_layer_norm_linear(self) -> type[InferenceLayerNormColumnParallelLinear]:
        """Which module for sequential layernorm and linear"""
        return InferenceLayerNormColumnParallelLinear

    def layer_norm(
        self, rms_norm: bool = False, for_qk: bool = False, has_residual: bool = False
    ) -> LayerNormBuilder:
        """Which module to use for layer norm"""
        if for_qk and not is_te_min_version("1.9.0"):
            # TENorm significantly harms convergence when used
            # for QKLayerNorm if TE Version < 1.9;
            # we instead use the Apex implementation.
            from megatron.core.fusions.fused_layer_norm import FusedLayerNorm

            return not_none(FusedLayerNorm)
        return TENorm

    def core_attention(self) -> type[TEDotProductAttention]:
        """Which module to use for attention"""
        return TEDotProductAttention

    def activation_func(self) -> TEActivationFunctionBuilder | None:
        """Which module to use for activation function"""
        # transformer_engine.BasicOperation.forward has an overly permissive return type, but by
        # design these classes always meet the interface.
        return cast(TEActivationFunctionBuilder, TEActivationOp)

    def grouped_mlp_modules(self, moe_use_grouped_gemm: bool) -> ExpertsBuilder:
        """Which module and submodules to use for grouped mlp"""
        return partial(
            InferenceGroupedMLP,
            submodules=GroupedMLPSubmodules(
                linear_fc1=TEColumnParallelGroupedLinear,
                linear_fc2=TERowParallelGroupedLinear,
                activation_func=self.activation_func(),
            ),
        )

    def moe_router(self) -> Optional[type]:
        """Inference needs compact [tokens, topk] index routing rather than a dense map."""
        from megatron.core.transformer.moe.router import InferenceTopKRouter

        return InferenceTopKRouter


def backend_slot(backend: BackendSpecProvider, name: str, default: Callable[[], object], **kwargs):
    """Ask a provider for a slot added after it may have been written.

    Two kinds of provider predate a newly added slot. One satisfies the protocol structurally
    and simply has no such attribute. The other subclasses the protocol and so *inherits* the
    default defined there -- including wrappers that delegate everything else to a fallback
    provider, which would quietly answer for the backend they wrap. Both are treated as "did
    not implement this", so ``default`` decides.

    ``default`` is a callable and is only evaluated when it is needed, so a provider that
    answers for itself never pays to build an answer it will not use.
    """
    method = getattr(backend, name, None)
    if method is None:
        return default()
    # An inherited protocol default is not an implementation.
    if getattr(method, "__func__", None) is getattr(BackendSpecProvider, name, None):
        return default()
    return method(**kwargs)


def get_backend(
    transformer_impl: Literal["local", "transformer_engine", "inference_optimized"],
    *,
    use_kitchen: bool = False,
    use_kitchen_attention: bool = False,
    kitchen_attention_backend: str = "sdpa",
    use_te_op_fuser: bool = False,
    cross_entropy_loss_fusion: bool = False,
    cross_entropy_fusion_impl: str = "native",
    cuda_graph_impl: str | None = None,
    kernels: KernelSelection | None = None,
) -> BackendSpecProvider:
    """Build the provider for a named backend.

    Kitchen is enabled independently of ``transformer_impl``. It overrides selected
    operations and delegates the rest to the chosen base provider through ``fallback``.
    ``use_te_op_fuser`` applies only to the Transformer Engine provider. ``kernels``
    configures the SSM and sparse-attention slots once; they take no arguments later.
    """
    if transformer_impl == "transformer_engine":
        from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider

        base: BackendSpecProvider = TESpecProvider(
            use_te_op_fuser=use_te_op_fuser,
            cross_entropy_loss_fusion=cross_entropy_loss_fusion,
            cross_entropy_fusion_impl=cross_entropy_fusion_impl,
            cuda_graph_impl=cuda_graph_impl,
            kernels=kernels,
        )
    elif transformer_impl == "inference_optimized":
        base = InferenceSpecProvider(
            cross_entropy_loss_fusion=cross_entropy_loss_fusion,
            cross_entropy_fusion_impl=cross_entropy_fusion_impl,
            cuda_graph_impl=cuda_graph_impl,
            kernels=kernels,
        )
    elif transformer_impl == "local":
        base = LocalSpecProvider(
            cross_entropy_loss_fusion=cross_entropy_loss_fusion,
            cross_entropy_fusion_impl=cross_entropy_fusion_impl,
            cuda_graph_impl=cuda_graph_impl,
            kernels=kernels,
        )
    else:
        raise ValueError(
            f"unknown transformer_impl='{transformer_impl}'. "
            "Valid choices: local, transformer_engine, inference_optimized"
        )

    if not use_kitchen:
        return base
    # Kitchen takes the operations it implements and forwards the rest to the provider that
    # would otherwise have owned them.
    from megatron.core.extensions.kitchen import HAVE_KITCHEN, KitchenSpecProvider

    if not HAVE_KITCHEN:
        raise ImportError(
            "Kitchen is not installed, and this backend needs it. The public stub would "
            "otherwise build a model out of mocks that fails somewhere unrelated."
        )
    return KitchenSpecProvider(
        fallback=base,
        use_kitchen_attention=use_kitchen_attention,
        kitchen_attention_backend=kitchen_attention_backend,
    )


def get_backend_from_config(
    config: object, *, transformer_impl: Optional[str] = None
) -> BackendSpecProvider:
    """Build the provider a TransformerConfig asks for.

    GPT spec builders also accept ``use_transformer_engine``, which can select local
    layers while ``config.transformer_impl`` retains its default of ``transformer_engine``.
    They pass the selected implementation here so the final norm and MTP use the same
    backend as the layers. Other callers use ``config.transformer_impl`` unchanged.
    """
    impl = transformer_impl or getattr(config, "transformer_impl", None)
    if impl is None:
        raise AttributeError(
            "config has no transformer_impl, and this needs to know which backend to build. "
            "Pass transformer_impl= explicitly."
        )
    return get_backend(
        impl,
        use_kitchen=getattr(config, "use_kitchen", False),
        use_kitchen_attention=getattr(config, "use_kitchen_attention", False),
        kitchen_attention_backend=getattr(config, "kitchen_attention_backend", "sdpa"),
        use_te_op_fuser=getattr(config, "use_transformer_engine_op_fuser", False),
        cross_entropy_loss_fusion=getattr(config, "cross_entropy_loss_fusion", False),
        cross_entropy_fusion_impl=getattr(config, "cross_entropy_fusion_impl", "native"),
        cuda_graph_impl=getattr(config, "cuda_graph_impl", None),
        kernels=KernelSelection.from_config(config),
    )


def resolve_kernel_backend(
    kernel_backend: Optional[BackendSpecProvider], config: object
) -> BackendSpecProvider:
    """The provider an operation module binds its kernels from.

    A provider handed down by the module spec (``params={"kernel_backend": ...}``) has the
    highest priority, so a user-supplied provider is never overridden. If that provider was
    built without a ``KernelSelection`` (a bare ``TESpecProvider()``), it is configured from
    ``config`` here so the existing per-operation settings still decide the kernels. Specs
    assembled without a config -- the module-level hybrid stack specs -- cannot inject a
    provider; those modules derive one from ``config`` through the same
    ``get_backend_from_config`` path the spec builders use, so both routes select identically.
    """
    if kernel_backend is None:
        return get_backend_from_config(config)
    if not isinstance(kernel_backend, KernelSelectionMixin):
        # A wrapper (which may delegate attribute lookups to its fallback) or a custom
        # provider: use it as given; backend_slot supplies the family default for slots it
        # does not implement. Duck-typing here could hand back the wrapper's fallback.
        return kernel_backend
    return kernel_backend.with_kernel_selection(KernelSelection.from_config(config))
