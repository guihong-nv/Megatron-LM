#!/usr/bin/env bash
#SBATCH --job-name=dsv41
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --time=00:30:00
#SBATCH --output=dsv41-%j.log

# Submit from a shared filesystem after preparing the image and Python environment:
#   export MEGATRON_ROOT=/shared/Megatron-LM
#   export CONTAINER_IMAGE=/shared/images/mcore.sqsh
#   export RUN_ROOT=/shared/dsv41-runs/smoke
#   sbatch -A <account> -p batch -N 1 examples/deepseek_v41/run_oci.slurm \
#     --profile mechanism-smoke --ep 4 --steps 5
#
# The mechanism smoke reduces widths and table sizes as well as depth. For the
# faithful depth-only proxy, use --profile depth-proxy and provision enough nodes
# for its original-width experts and Engram tables. Full/depth-proxy execution
# also requires the official tokenizer and its derived Engram token map; pass
# --tokenizer-model and --engram-token-map explicitly.
#
# Account, partition, node count and optional rack placement (--segment) belong
# to the submitting site's Slurm policy. JHB and AGA GB300 nodes have four GPUs.
# Set GPUS_PER_NODE when overriding --gpus-per-node for another node type.
# EXTRA_CONTAINER_MOUNTS accepts additional Pyxis src:dst mounts for datasets or
# tokenizer files outside MEGATRON_ROOT and RUN_ROOT. Paths remain identical
# inside the container by default. No image or package installation runs here.
set -euo pipefail

: "${MEGATRON_ROOT:?Set MEGATRON_ROOT to the shared branch checkout}"
: "${CONTAINER_IMAGE:?Set CONTAINER_IMAGE to a prepared Pyxis image}"
: "${RUN_ROOT:?Set RUN_ROOT to a shared run/checkpoint directory}"
: "${SLURM_JOB_ID:?Submit this script with sbatch}"
: "${SLURM_JOB_NODELIST:?Slurm did not provide the allocation node list}"

for path in "$MEGATRON_ROOT" "$RUN_ROOT"; do
    if [[ "$path" != /* || "$path" == *","* || "$path" == *":"* ]]; then
        echo "Shared checkout and run paths must be absolute and contain no comma or colon: $path" >&2
        exit 2
    fi
done
if [[ ! -f "$MEGATRON_ROOT/examples/deepseek_v41/train.sh" ]]; then
    echo "The shared checkout does not contain examples/deepseek_v41/train.sh" >&2
    exit 2
fi

export NNODES="${SLURM_NNODES:?Slurm did not provide the node count}"
export GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
if [[ ! "$GPUS_PER_NODE" =~ ^[1-9][0-9]*$ ]]; then
    echo "GPUS_PER_NODE must be a positive integer" >&2
    exit 2
fi
export WORLD_SIZE=$((NNODES * GPUS_PER_NODE))
node_names=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
export MASTER_ADDR="${MASTER_ADDR:-${node_names%%$'\n'*}}"
export MASTER_PORT="${MASTER_PORT:-$((20000 + SLURM_JOB_ID % 20000))}"
export MEGATRON_ROOT RUN_ROOT

mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/cache"
cd "$MEGATRON_ROOT"
container_mounts="$MEGATRON_ROOT:$MEGATRON_ROOT,$RUN_ROOT:$RUN_ROOT"
if [[ -n "${EXTRA_CONTAINER_MOUNTS:-}" ]]; then
    container_mounts+=",$EXTRA_CONTAINER_MOUNTS"
fi

printf 'DSV4.1 job=%s nodes=%s GPUs=%s run=%s\n' \
    "$SLURM_JOB_ID" "$NNODES" "$WORLD_SIZE" "$RUN_ROOT"

# One worker group per node; train.sh uses uv's Python environment to start all
# ranks in one distributed job. Keep rank-dependent caches apart across nodes.
# Blackwell does not need CUDA_DEVICE_MAX_CONNECTIONS=1; no value is forced here.
srun --kill-on-bad-exit=1 --ntasks="$NNODES" --ntasks-per-node=1 \
    --gpus-per-task="$GPUS_PER_NODE" \
    --container-image="$CONTAINER_IMAGE" \
    --container-mounts="$container_mounts" \
    --container-workdir="$MEGATRON_ROOT" \
    --output="$RUN_ROOT/logs/job-${SLURM_JOB_ID}-node-%t.log" \
    bash -c '
        set -euo pipefail
        export NODE_RANK=${SLURM_NODEID:?Slurm did not provide a node rank}
        export PYTHONPATH="$MEGATRON_ROOT${PYTHONPATH:+:$PYTHONPATH}"
        export TRITON_CACHE_DIR="$RUN_ROOT/cache/node-$NODE_RANK/triton"
        export TORCHINDUCTOR_CACHE_DIR="$RUN_ROOT/cache/node-$NODE_RANK/inductor"
        export TORCH_EXTENSIONS_DIR="$RUN_ROOT/cache/node-$NODE_RANK/torch_extensions"
        mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$TORCH_EXTENSIONS_DIR"
        cd "$MEGATRON_ROOT"
        exec bash examples/deepseek_v41/train.sh "$@" \
            --run-dir "$RUN_ROOT" \
            --nnodes "$NNODES" \
            --nproc-per-node "$GPUS_PER_NODE" \
            --node-rank "$NODE_RANK" \
            --master-addr "$MASTER_ADDR" \
            --master-port "$MASTER_PORT"
    ' dsv41-node "$@"
