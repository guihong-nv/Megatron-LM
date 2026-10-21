# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Packed index compilation must survive real mixed-layer training and evaluation."""

import importlib.util

import pytest
import torch

from megatron.core.transformer.experimental_attention_variant.csa_utils import packed_layout


def _load_with_eager_backend(monkeypatch):
    """Exercise production Dynamo policies without CUDA/Inductor compilation."""
    compiled_graphs = []

    def backend(graph, example_inputs):
        compiled_graphs.append(graph)
        return graph.forward

    original_compile = torch.compile

    def compile_with_eager_backend(*args, **kwargs):
        return original_compile(*args, **kwargs, backend=backend)

    # Load the production dispatch with only the backend substituted. In
    # particular, retain its real dynamic-shape and grad-mode policies so this
    # test fails if either policy regresses. An isolated module avoids changing
    # the compile cache used by other tests importing packed_layout.
    spec = importlib.util.spec_from_file_location('packed_layout_cpu_test', packed_layout.__file__)
    module = importlib.util.module_from_spec(spec)
    with monkeypatch.context() as patch:
        patch.setattr(torch, 'compile', compile_with_eager_backend)
        spec.loader.exec_module(module)
    return module, compiled_graphs


def test_attention_indices_reuse_graphs_across_packed_shapes_and_grad_modes(monkeypatch):
    """Integer metadata does not need separate training and evaluation graphs."""
    torch.compiler.reset()
    module, compiled_graphs = _load_with_eager_backend(monkeypatch)
    try:
        for training in (True, False, True):
            for documents in range(2, 8):
                cu = torch.arange(documents + 1, dtype=torch.int32) * (64 // documents)
                cu[-1] = 64
                for ratio in (0, 2, 1):
                    options = dict(cu_seqlens_unpadded=cu.clone(), output_alignment=8)
                    if ratio:
                        lengths = (cu[1:] - cu[:-1]) // ratio
                        compressed_cu = torch.cat((cu[:1], lengths.cumsum(0).int()))
                        options.update(
                            compressed_topk=torch.arange(16, dtype=torch.int32).expand(32, -1),
                            cu_seqlens_compressed=compressed_cu,
                            seq_to_rank_row=torch.arange(64 // ratio, dtype=torch.int64),
                            compressed_rows=64 // ratio,
                        )
                    inputs = (cu, 32, 32, 8, 8, ratio, 16 if ratio else 0)
                    expected = module._build_attention_indices(*inputs, **options)
                    with torch.set_grad_enabled(training):
                        actual = module._compiled_attention_indices(*inputs, **options)
                    for reference, result in zip(expected, actual):
                        if reference is None:
                            assert result is None
                        else:
                            assert torch.equal(result, reference)
            if training:
                training_graphs = len(compiled_graphs)
            else:
                assert len(compiled_graphs) == training_graphs, (
                    f'Evaluation recompiled integer metadata: '
                    f'{training_graphs} -> {len(compiled_graphs)} graphs'
                )
        assert 0 < len(compiled_graphs) <= 8
    finally:
        torch.compiler.reset()


def _check_compactor_parity(module, boundary_rows, device='cpu', document_counts=range(1, 8)):
    local_rows = 256 if device == 'cuda' else 32
    total_rows = 2 * local_rows
    dtype = torch.bfloat16 if device == 'cuda' else torch.float32
    for training in (True, False, True):
        for documents in document_counts:
            cu = torch.arange(documents + 1, dtype=torch.int32, device=device) * (
                total_rows // documents
            )
            cu[-1] = total_rows
            for ratio in (2, 1):
                hidden = torch.arange(local_rows * 4, dtype=torch.float32, device=device).reshape(
                    local_rows, 1, 4
                )
                boundary = torch.arange(
                    boundary_rows * 4, dtype=torch.float32, device=device
                ).reshape(boundary_rows, 1, 4)
                hidden = hidden.to(dtype).requires_grad_(training)
                boundary = boundary.to(dtype).requires_grad_(training)
                reference_hidden = hidden.detach().clone().requires_grad_(training)
                reference_boundary = boundary.detach().clone().requires_grad_(training)
                capacity = (
                    ((local_rows // ratio + 1) + (32 // ratio) - 1) // (32 // ratio) * (32 // ratio)
                )
                geometry = (cu, local_rows if boundary_rows else 0, ratio, ratio, capacity, 2)
                with torch.set_grad_enabled(training):
                    actual = module._compiled_compactor(hidden, boundary, *geometry)
                    expected = module._compact_compressor_input(
                        reference_hidden, reference_boundary, *geometry
                    )
                for reference, result in zip(expected, actual):
                    assert torch.equal(result, reference)
                assert actual[0].requires_grad == training
                if training:
                    weights = (torch.arange(actual[0].numel(), device=device) % 7).reshape_as(
                        actual[0]
                    )
                    actual_grads = torch.autograd.grad(
                        (actual[0] * weights).sum(), (hidden, boundary), allow_unused=True
                    )
                    expected_grads = torch.autograd.grad(
                        (expected[0] * weights).sum(),
                        (reference_hidden, reference_boundary),
                        allow_unused=True,
                    )
                    for reference, result in zip(expected_grads, actual_grads):
                        if reference is None:
                            assert result is None
                        else:
                            assert torch.equal(result, reference)


@pytest.mark.parametrize('boundary_rows', [0, 8])
def test_compactor_reuses_packed_shapes_with_forward_and_gradient_parity(
    monkeypatch, boundary_rows
):
    """Document counts vary within both grad modes; floating inputs keep gradients."""
    torch.compiler.reset()
    module, compiled_graphs = _load_with_eager_backend(monkeypatch)
    try:
        _check_compactor_parity(module, boundary_rows)
        assert 0 < len(compiled_graphs) <= 8
    finally:
        torch.compiler.reset()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA Inductor')
def test_compactor_cuda_packed_shape_and_gradient_parity():
    """Run real CUDA code generation over one-, two- and three-document batches."""
    torch.compiler.reset()
    try:
        _check_compactor_parity(packed_layout, 128, device='cuda', document_counts=(1, 2, 3))
    finally:
        torch.compiler.reset()


def test_topk_sanitizer_reuses_score_widths_across_train_and_eval(monkeypatch):
    """Real Top-K widths shrink with the longest packed document, before padding."""
    torch.compiler.reset()
    module, compiled_graphs = _load_with_eager_backend(monkeypatch)
    try:
        for training in (True, False, True):
            for rows in (32, 16):
                for score_width in (64, 32, 16, 12, 8, 4, 24):
                    width = min(16, score_width)
                    candidates = torch.arange(width, dtype=torch.int32).repeat(rows, 1)
                    candidates[:, -1] = -1
                    scores = torch.ones(rows, score_width, requires_grad=training)
                    with torch.no_grad():
                        scores[:, 0] = float('nan')
                    lengths = torch.arange(rows, dtype=torch.int32) % (score_width + 1)
                    inputs = (candidates, scores, lengths, 16)
                    expected = module._sanitize_topk(*inputs)
                    with torch.set_grad_enabled(training):
                        actual = module._compiled_sanitize_topk(*inputs)
                    for reference, result in zip(expected, actual):
                        assert torch.equal(result, reference)
        assert 0 < len(compiled_graphs) <= 8
    finally:
        torch.compiler.reset()
