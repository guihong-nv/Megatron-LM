# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Packed index compilation must survive real mixed-layer training and evaluation."""

import importlib.util

import torch

from megatron.core.transformer.experimental_attention_variant.csa_utils import packed_layout


def test_attention_indices_reuse_graphs_across_packed_shapes_and_grad_modes(monkeypatch):
    """Exercise Dynamo guards without paying for CUDA/Inductor kernel compilation."""
    torch.compiler.reset()
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
