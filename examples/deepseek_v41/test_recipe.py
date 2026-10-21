# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU/std-library recipe tests; run with unittest rather than CUDA pytest fixtures."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import recipe


class TestFlashRecipes(unittest.TestCase):
    """Protect faithful dimensions, complete attention roles and launch validation."""

    def test_full_schedule(self):
        """Retain the released 40-block schedule and omit only draft ratios."""
        config = recipe.build_profile("full")
        text = config["text_config"]
        self.assertEqual(text["num_hidden_layers"], 40)
        self.assertEqual(text["compress_ratios"], [0] * 2 + [2] * 18 + [1] * 20)
        self.assertEqual(text["kv_source_layer_ids"], [2, 8, 14, 20])
        self.assertEqual(text["index_source_layer_ids"], [2, 8, 14, 20, 24, 28, 32, 36])
        self.assertEqual(text["engram_layer_ids"], [1, 14])
        self.assertEqual(set(config["changes_from_released_text_config"]), {"compress_ratios"})

    def test_depth_proxy_changes_only_depth_and_schedule(self):
        """Keep all released widths, expert counts and Engram table capacities."""
        config = recipe.build_profile("depth-proxy")
        self.assertEqual(
            set(config["changes_from_released_text_config"]),
            {
                "num_hidden_layers",
                "compress_ratios",
                "kv_source_layer_ids",
                "index_source_layer_ids",
                "candidate_source_layer_id",
                "engram_layer_ids",
            },
        )
        self.assertEqual(config["text_config"]["hidden_size"], 5120)
        self.assertEqual(config["text_config"]["n_routed_experts"], 384)
        self.assertEqual(config["text_config"]["engram_num_embeddings"], [384006168, 384016682])

    def test_depth_proxy_retains_all_roles_and_both_engrams(self):
        """Exercise SWA, Full, Reindex, Reuse and both conditional-memory insertions."""
        config = recipe.build_profile("depth-proxy")
        self.assertEqual(config["original_logical_block_ids"], [0, 1, 2, 3, 14, 20, 24, 25])
        self.assertEqual(
            config["attention_roles"],
            ["SWA", "SWA", "Full", "Reuse", "Full", "Full", "Reindex", "Reuse"],
        )
        self.assertEqual(config["text_config"]["compress_ratios"], [0, 0, 2, 2, 2, 1, 1, 1])
        self.assertEqual(config["engram_config"]["layer_ids"], [1, 4])

    def test_mechanism_profile_is_explicit_and_retains_router_math(self):
        """Expose reduced geometry while retaining the model's routing mechanisms."""
        config = recipe.build_profile("mechanism-smoke")
        text = config["text_config"]
        self.assertEqual(text["n_routed_experts"], 8)
        self.assertEqual(text["num_experts_per_tok"], 6)
        self.assertEqual(text["n_shared_experts"], 1)
        self.assertEqual(text["scoring_func"], "sqrtsoftplus")
        self.assertEqual(text["hc_mult"], 4)
        self.assertEqual(text["head_dim"], 512)
        self.assertEqual(text["engram_head_dim"], 256)
        self.assertIn("engram_num_embeddings", config["changes_from_released_text_config"])

    def test_distinct_prime_layout(self):
        """Assign distinct primes across n-gram orders and layers."""
        self.assertEqual(recipe._prime_rows(2, 2, 2, 2), [17, 60])

    def test_memory_lower_bound_includes_full_tables(self):
        """Include both full-size Engram tables in the capacity estimate."""
        estimate = recipe.memory_estimate(recipe.build_profile("depth-proxy"))
        self.assertEqual(estimate["engram_table_parameters"], (384006168 + 384016682) * 256)
        self.assertGreater(estimate["adam_state_and_weights_lower_bound_gib"], 5000)
        self.assertGreater(estimate["minimum_gpus_at_75_percent_memory_lower_bound"], 8)

    def test_valid_parallel_layouts(self):
        """Accept supported TP1 data, expert, pipeline and context layouts."""
        config = recipe.build_profile("depth-proxy")
        for world, pp, cp, ep in [(8, 2, 2, 2), (64, 2, 1, 32), (8, 1, 1, 8)]:
            with self.subTest(world=world, pp=pp, cp=cp, ep=ep):
                recipe.validate_topology(config, world, pp, cp, ep)

    def test_invalid_parallel_layouts(self):
        """Reject impossible world-size and stage factorizations."""
        config = recipe.build_profile("depth-proxy")
        for world, pp, cp, ep in [(8, 3, 1, 1), (8, 1, 3, 1), (8, 1, 1, 3), (0, 1, 1, 1)]:
            with self.subTest(world=world, pp=pp, cp=cp, ep=ep), self.assertRaises(ValueError):
                recipe.validate_topology(config, world, pp, cp, ep)

    def test_invalid_expert_divisibility(self):
        """Require an integral number of experts per expert-parallel rank."""
        with self.assertRaises(ValueError):
            recipe.validate_topology(recipe.build_profile("mechanism-smoke"), 6, 1, 1, 3)

    def test_unknown_profile_rejected(self):
        """Reject profile names that could conceal an unreviewed architecture."""
        with self.assertRaises(ValueError):
            recipe.build_profile("tiny-but-faithful")

    def test_command_has_complete_architecture_and_physical_ids(self):
        """Translate the faithful architecture and logical owners into CLI arguments."""
        args = recipe.parser().parse_args(["--run-dir", "/tmp/dsv41", "--pp", "2", "--cp", "2"])
        command = recipe.training_args(recipe.build_profile("depth-proxy"), args)
        for flag, expected in {
            "--hidden-size": "5120",
            "--num-experts": "384",
            "--moe-router-topk": "6",
            "--moe-shared-expert-intermediate-size": "2304",
            "--moe-router-score-function": "sqrtsoftplus",
            "--norm-epsilon": "1e-20",
            "--csa2-candidate-source-layer": "10",
            "--num-layers": "16",
            "--hybrid-layer-pattern": "VEVEVEVE|VEVEVEVE",
            "--linear-cp-layout": "contiguous",
        }.items():
            with self.subTest(flag=flag):
                self.assertEqual(command[command.index(flag) + 1], expected)
        self.assertIn("--engram-config", command)
        self.assertIn("--moe-router-enable-expert-bias", command)
        self.assertIn("--dataloader-inter-document-masking", command)
        index = command.index("--csa2-kv-source-layers")
        self.assertEqual(command[index + 1 : index + 4], ["4", "8", "10"])

    def test_dry_run_is_stdlib_only_and_writes_manifest(self):
        """Generate auditable manifests without importing the training runtime."""
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            recipe.main(["--run-dir", directory, "--dry-run"])
            result = json.loads((Path(directory) / "recipe-rank0.json").read_text())
            self.assertEqual(result["profile"], "depth-proxy")
            self.assertEqual(result["data"], "synthetic mock data")
            self.assertTrue((Path(directory) / "engram.json").is_file())

    def test_faithful_execution_requires_real_map_before_torch_import(self):
        """Reject faithful launches without the released tokenizer identities."""
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                recipe.main(["--run-dir", directory, "--execute"])
            self.assertEqual(error.exception.code, 2)

    def test_architecture_overrides_rejected(self):
        """Prevent the launcher from silently shrinking a faithful profile."""
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            recipe.parser().parse_args(["--run-dir", "/tmp/dsv41", "--hidden-size", "256"])

    def test_full_recompute_and_resume_steps(self):
        """Preserve checkpoint scheduling when extending the training target."""
        args = recipe.parser().parse_args(
            ["--run-dir", "/tmp/dsv41", "--steps", "16", "--full-recompute"]
        )
        command = recipe.training_args(recipe.build_profile("mechanism-smoke"), args)
        self.assertEqual(command[command.index("--train-iters") + 1], "16")
        self.assertEqual(command[command.index("--load") + 1], "/tmp/dsv41/checkpoints")
        self.assertEqual(command[command.index("--save") + 1], "/tmp/dsv41/checkpoints")
        self.assertIn("--recompute-granularity", command)
        self.assertIn("--use-checkpoint-opt-param-scheduler", command)

    def test_mock_profile_cannot_accept_real_data_or_external_map(self):
        """Keep the synthetic test's identity map isolated from real-data recipes."""
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stderr(io.StringIO()):
            for option in [
                ["--engram-token-map", "/tmp/token_map.pt"],
                ["--data-path", "data", "--tokenizer-model", "tokenizer"],
            ]:
                with self.subTest(option=option), self.assertRaises(SystemExit):
                    recipe.main(["--run-dir", directory, "--profile", "mechanism-smoke", *option])


if __name__ == "__main__":
    unittest.main()
