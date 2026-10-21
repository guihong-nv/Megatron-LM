#!/usr/bin/env bash
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
exec uv run --no-sync python examples/deepseek_v41/recipe.py --execute "$@"
