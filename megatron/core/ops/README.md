# Megatron Core Operations

`megatron/core/ops` is the implementation home for Megatron Core's operation
families, organized by operation. SSM and sparse attention moved here first;
further families follow the same layout.

This is an operation-implementation package, not just a kernel directory. It
owns concrete operation modules, kernels, backend adapters, operation-local
parameters and checkpoint mappings, state updates, and operation-specific
communication. Model assembly and global runtime management stay outside.

Import family namespaces for contracts and metadata without loading optional
kernel libraries. Import concrete implementation modules explicitly when
constructing an operation. Moving an implementation does not change its
parameter names, registered submodules, checkpoint layout or numerical behavior.

## Families

| Location | Contents |
| --- | --- |
| `ssm/common` | Convolution, packing, checkpoint helpers and per-operation inference execution |
| `ssm/mamba2` | `mixer.py`, context-parallel transforms, SSD training and inference kernels |
| `ssm/gated_delta` | GDN/GDN2 modules, reference recurrences and FLA adapters |
| `ssm/gdp` | `mixer.py`, context-parallel transforms, training adapters and inference kernels |
| `ssm/context_parallel` | Chunkwise SSM communication and GDP backend implementations |
| `attention/dsa` | `modules.py`, layout/masking, indexer loss, reference and fused adapters |
| `attention/dsa/kernels` | TileLang/Triton kernel implementations |
| `attention/csa` | `modules.py` with compressor/indexer/attention modules and reference kernels |
| `attention/mla.py` | Absorbed MLA operation and its projection/layout helpers |
| `attention/dsv4.py` | DeepSeek-v4 hybrid attention operation |

## Ownership Boundary

An implementation may own state when it belongs to that operation: parameters,
local buffers, checkpoint sharding, or communication using supplied process
groups. Merely subclassing `nn.Module` does not put an implementation outside
`ops`. Existing operation boundaries are preserved; relocation adds no wrapper.

The following remain outside:

- Model/layer assembly, hybrid allocation, configuration and spec builders.
- Inference contexts, global cache allocation, request scheduling and stack-wide
  recurrent-state configuration (`inference/ssm_config.py`).
- Training-wide indexer-loss tracking and gradient-scale management
  (`transformer/dsa_loss.py`).

Operations may use shared infrastructure such as embeddings, `MegatronModule`,
checkpoint utilities, inference contexts and explicit process groups. They must
not import concrete model assembly or the deprecated SSM/attention module paths.
The existing provider API is used only at construction; no new reverse
dependency on model spec builders is introduced.

The family docstrings and callable signatures describe tensor layouts, masks,
state ownership and distributed inputs. Moving a kernel does not certify its
determinism or change its supported dtypes, layouts or numerical tolerances.
Existing determinism guards and backend-specific restrictions still apply.

## Dependency Checks

Optional kernel libraries are checked once, at construction, with
`megatron.core.ops._backends.require`:

```python
from megatron.core.ops._backends import require

ssd = require("mamba_ssm.ops.triton.ssd_combined", "mamba_chunk_scan_combined", needed_by="Mamba2")
self.scan = ssd.mamba_chunk_scan_combined
```

`require(module, *symbols, min_version=None, dist=None, needed_by=...)` imports
the module, checks that each named export exists and is not `None` (dotted names
reach into lazy namespaces such as `cudnn.DSA`), optionally checks a minimum
version (`module.__version__` first, then distribution metadata, so source
checkouts work), and returns the module. Every failure is an `ImportError`
naming the operation that asked -- including a native extension that is
installed but fails to load -- and the original error is chained.

Rules:

- Each family's `backends.py` owns the selectors (`select_*`). They import only
  what was selected; an unavailable selection is an error, never a silent switch
  to another implementation. `test_selectors_do_not_import_unselected_optional_libraries`
  enforces this.
- Operation constructors `require` the auxiliary kernels they own (convolution,
  normalization, fused RoPE) separately from the provider-owned recurrence, before
  parameters are allocated.
- `require` is construction-time only. A capability that depends on execution-time
  input -- packed sequences under CP, say -- is decided once (`is_available`,
  `has_min_version`, `packed_cp_conv_supported`) and a bool is checked per call.
  `test_require_is_only_called_at_construction_time` enforces this.
- Inference-only kernels are bound by `bind_dynamic_inference_kernels`, which
  dynamic-inference setup calls on every pipeline-local mixer.
- Do not add `HAVE_*` flags, availability tables or kernel inventories. When a
  module's own imports already fail clearly (the GDP chunkwise-CP adapters do),
  `require(module, needed_by=...)` is the whole check.
- Determinism is not declared per kernel. The existing guards
  (`assert_causal_conv1d_deterministic`, the Torch reference recurrences selected
  by `deterministic_mode`, `CSA_OPERATION_DETERMINISM`) stay with their owners;
  the determinism developer docs describe what has been audited.

## Selection

`BackendSpecProvider` is the only construction API. A provider is configured once,
when it is built, from the existing config fields collected in
`megatron.core.models.backends.KernelSelection` (`deterministic_mode`,
`use_mamba_mem_eff_path`, `gdp_cutedsl_kernel`, `gdp_num_chunk_states_to_recompute`,
`dsa_kernel_backend`, `attention_backend`). The kernel slots then take no
implementation-selection arguments:

| Slot | Returns | Bound by |
| --- | --- | --- |
| `mamba_kernels()` | `MambaKernels` (scan, optional fused conv+scan, conv) | `MambaMixer` |
| `gated_delta_rule(variant)` | GDN or GDN2 recurrence; `variant` names the operation, not the backend | `GatedDeltaNet`, `GatedDeltaNet2` |
| `gated_delta_product()` | FLA or CuTeDSL chunked gated delta product | `GatedDeltaProductMixer` |
| `gated_delta_product_cp_backend()` | chunkwise-CP adapter matching the GDP kernel | `GatedDeltaProductMixer` when CP > 1 |
| `dsa_kernels()` | immutable `DSAKernels` hook set (or none) | `DSAttention` |

Local and TE providers share one implementation of these slots
(`KernelSelectionMixin`): TE has no SSM or sparse-attention kernels of its own, and
sharing the body keeps every slot overridable by a partial provider that does.
`backend_slot` supplies the family default for providers written before a slot
existed. There is no registry and no new CLI option.

Every operation module accepts a `kernel_backend` provider from its module spec
(`params={"kernel_backend": provider}`) and resolves it with
`resolve_kernel_backend(kernel_backend, config)`:

- A provider built with a selection (`get_backend_from_config`, or
  `kernels=KernelSelection(...)`) is used as is; the explicit selection wins even
  where it disagrees with `config`.
- A bare provider (`TESpecProvider()`) is configured from `config` at bind time, on
  a shallow copy, so the existing per-operation settings still decide the kernels
  and a provider shared across a spec is never mutated. Asking a bare provider for
  a kernel slot directly is an error, never a silent default.
- A wrapper or custom provider without the mixin is used untouched; `backend_slot`
  supplies the family default for slots it does not implement. Build wrappers
  through `get_backend`/`get_backend_from_config` so the fallback they delegate to
  is configured; a wrapper around a bare fallback fails loudly on a kernel slot.
- Specs assembled without a config -- the module-level hybrid stack specs -- cannot
  inject a provider, so those modules derive one from `config` through the same
  `get_backend_from_config` path the spec builders use; both routes select
  identically.

Kernels are bound once, in `__init__`, and called directly from `forward`. No
selection, availability check or optional import happens in the forward path.
DSAttention's hooks may still return `None` for unsupported runtime inputs, in
which case the caller runs the reference implementation. GDN dynamic inference uses
FLA's fused decode/prefill family (which takes `A_log`/`dt_bias` and fuses the
gates, so it is not interchangeable with the training recurrence); it is bound once
through `bind_dynamic_inference_kernels`, which inference setup calls on every
pipeline-local mixer so a missing library fails there rather than in the first
decode step.

## Import Migration

The former `megatron.core.ssm` and
`megatron.core.transformer.experimental_attention_variant` module paths are
deprecated, not removed. Every pre-move module still exists as a two-line
forwarder built on `megatron.core.ops._compat.deprecated_module`:

- Importing an old path emits one `DeprecationWarning` naming the replacement.
- Attributes resolve lazily through PEP 562 module `__getattr__`, so importing the
  old path does not import the implementation or its optional kernel libraries.
- `from old import *`, private names and pickles that recorded the old
  `__module__` keep working, and every object is the canonical one.
- The forwarders are scheduled for removal in the version recorded by
  `_compat.REMOVAL_VERSION`. In-tree code must use canonical paths; a unit test
  enforces this.

Ordinary state-dict keys and checkpoint tensor mappings do not depend on the
source directory and are unchanged.

The full old-to-new table is `tests/unit_tests/ops/deprecated_paths.py`. The main
entries, relative to `megatron.core`:

| Former owner | Canonical owner |
| --- | --- |
| `ssm.mamba_mixer`, `ssm.gated_delta_product`, `ssm.gated_delta_net` | `ops.ssm.mamba2.mixer`, `ops.ssm.gdp.mixer`, `ops.ssm.gated_delta.modules` |
| `ssm.ops.{common,mamba2,gdp}` | `ops.ssm.{common,mamba2,gdp}` |
| SSM CP, packing and checkpoint helpers | `ops.ssm` operation families and `ops.ssm.common` |
| Experimental DSA/CSA, absorbed MLA and DeepSeek-v4 attention | `ops.attention.{dsa,csa}.modules`, `ops.attention.mla`, `ops.attention.dsv4` |
| Experimental DSA kernel adapters and helpers | `ops.attention.dsa` and `ops.attention.dsa.kernels` |
| `ssm.mamba_layer`, `ssm.mlp_layer` and their layer-config classes | `transformer.mamba_layer`, `transformer.mlp_layer` and `transformer.*_layer_config` |
| Experimental `dsa_layer_config` | `transformer.dsa_layer_config` |
| Experimental `deepseek_v4_hybrid_attention_module_specs` | `models.gpt.deepseek_v4_hybrid_attention_module_specs` |
| `ssm.ssm_inference.SSMChunking` and `ssm_chunking` | `inference.ssm_config` |
| `ssm.ssm_inference.SSMDynamicInferenceMixin` | `ops.ssm.common.inference` |
| `ssm.mamba_block`, `ssm.mamba_hybrid_layer_allocation` | `models.hybrid.hybrid_block`, `models.hybrid.hybrid_layer_allocation` |

Update launch-script module strings as well as Python imports. In particular,
the cache-manager setting is now:

```bash
export TRITON_CACHE_MANAGER=megatron.core.ops.ssm.triton_cache_manager:ParallelFileCacheManager
```

Tests cover canonical module/class ownership and pickle round trips, construction
import order, the deprecated-path forwarders and the absence of deprecated imports
in the tree.

Vendored kernel files retain their original licenses and internal file structure.
Future changes should keep cohesive operation implementations here and model
assembly or global runtime lifecycle in its existing subsystem.
