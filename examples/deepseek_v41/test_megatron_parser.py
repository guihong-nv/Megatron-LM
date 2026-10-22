# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Check example commands against the real parser inside the Megatron dev environment."""

import argparse
import json
import tempfile
import unittest
from types import SimpleNamespace

import recipe
import torch

from megatron.core.models.deepseek_v41.config import DeepSeekV41Config
from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler
from megatron.training.arguments import add_megatron_arguments, core_transformer_config_from_args


class TestMegatronParser(unittest.TestCase):
    """Catch renamed/missing flags rather than validating only our own argument builder."""

    def test_generated_profiles(self):
        """Parse every generated profile with Megatron's real argument definitions."""
        parser = add_megatron_arguments(argparse.ArgumentParser(allow_abbrev=False))
        for profile in recipe.PROFILES:
            with self.subTest(profile=profile):
                options = recipe.parser().parse_args(
                    [
                        "--profile",
                        profile,
                        "--run-dir",
                        "/tmp/dsv41-parser",
                        "--pp",
                        "2",
                        "--cp",
                        "2",
                        "--ep",
                        "2",
                        "--full-recompute",
                    ]
                )
                model = recipe.build_profile(profile)
                parsed = parser.parse_args(recipe.training_args(model, options))
                self.assertEqual(parsed.hidden_size, model["text_config"]["hidden_size"])
                self.assertEqual(parsed.num_experts, model["text_config"]["n_routed_experts"])
                self.assertEqual(parsed.layernorm_epsilon, 1e-20)
                self.assertEqual(parsed.moe_router_score_function, "sqrtsoftplus")
                self.assertEqual(parsed.moe_router_topk, 6)
                self.assertEqual(
                    parsed.csa2_candidate_source_layer,
                    2 * model["text_config"]["candidate_source_layer_id"],
                )
                self.assertTrue(parsed.mhc_single_pass)
                self.assertTrue(parsed.use_distributed_optimizer)
                self.assertTrue(parsed.use_checkpoint_opt_param_scheduler)

    def test_generated_configs_match_released_architecture(self):
        """Compare constructed single-rank and distributed configs with the HF adapter."""
        parser = add_megatron_arguments(argparse.ArgumentParser(allow_abbrev=False))
        fields = (
            "num_layers",
            "hidden_size",
            "num_attention_heads",
            "multi_latent_attention",
            "experimental_attention_variant",
            "dsv4_version",
            "q_lora_rank",
            "v_head_dim",
            "qk_pos_emb_head_dim",
            "output_projection_groups",
            "output_projection_lora_rank",
            "qk_layernorm",
            "layernorm_epsilon",
            "normalization",
            "rotary_base",
            "csa_compress_rotary_base",
            "original_max_position_embeddings",
            "rotary_scaling_factor",
            "beta_fast",
            "beta_slow",
            "mscale",
            "mscale_all_dim",
            "csa_compress_ratios",
            "csa_window_size",
            "csa2_kv_source_layers",
            "csa2_index_source_layers",
            "csa2_candidate_source_layer",
            "csa2_candidate_topk_blocks",
            "csa2_candidate_block_size",
            "dsa_indexer_n_heads",
            "dsa_indexer_head_dim",
            "dsa_indexer_topk",
            "dsa_indexer_rotate_activation",
            "enable_mhc_connections",
            "mhc_single_pass",
            "mhc_epsilon",
            "mhc_num_residual_streams",
            "mhc_sinkhorn_iterations",
            "num_moe_experts",
            "moe_router_topk",
            "moe_ffn_hidden_size",
            "moe_shared_expert_intermediate_size",
            "moe_router_score_function",
            "moe_router_topk_scaling_factor",
            "moe_router_dtype",
            "moe_router_enable_expert_bias",
            "moe_token_dispatcher_type",
            "moe_combine_in_fp32",
            "activation_func",
            "gated_linear_unit",
            "activation_func_clamp_value",
            "add_bias_linear",
            "attention_dropout",
            "hidden_dropout",
            "engram_config",
        )
        for profile, distributed in ((p, d) for p in recipe.PROFILES for d in (False, True)):
            with (
                self.subTest(profile=profile, distributed=distributed),
                tempfile.TemporaryDirectory() as directory,
            ):
                runtime = (
                    ["--pp", "2", "--cp", "2", "--ep", "2", "--full-recompute"]
                    if distributed
                    else []
                )
                options = recipe.parser().parse_args(
                    ["--profile", profile, "--run-dir", directory, *runtime]
                )
                model = recipe.build_profile(profile)
                (options.run_dir / "engram.json").write_text(json.dumps(model["engram_config"]))
                parsed = parser.parse_args(recipe.training_args(model, options))
                parsed.params_dtype = torch.bfloat16
                actual = core_transformer_config_from_args(parsed)
                expected = DeepSeekV41Config.from_hf({"text_config": model["text_config"]})
                for field in fields:
                    with self.subTest(field=field):
                        self.assertEqual(getattr(actual, field), getattr(expected, field))

    def test_constant_schedule_resume_with_more_training_steps(self):
        """Restore constant learning-rate and weight-decay schedules beyond the old target."""

        def make_scheduler(total_steps):
            return OptimizerParamScheduler(
                optimizer=SimpleNamespace(param_groups=[{"lr": 0.0, "weight_decay": 0.0}]),
                init_lr=0.0,
                max_lr=1e-4,
                min_lr=1e-4,
                lr_warmup_steps=0,
                lr_decay_steps=total_steps,
                lr_decay_style="constant",
                start_wd=0.1,
                end_wd=0.1,
                wd_incr_steps=total_steps,
                wd_incr_style="constant",
                use_checkpoint_opt_param_scheduler=True,
            )

        original = make_scheduler(8)
        original.step(8)
        resumed = make_scheduler(16)
        resumed.load_state_dict(original.state_dict())
        resumed.step(1)
        self.assertEqual(resumed.num_steps, 9)
        self.assertEqual(resumed.optimizer.param_groups[0]["lr"], 1e-4)
        self.assertEqual(resumed.optimizer.param_groups[0]["weight_decay"], 0.1)


if __name__ == "__main__":
    unittest.main()
