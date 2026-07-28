#!/usr/bin/env bash
set -euo pipefail

# Reproducible 8xA100 recipe used for the portfolio project. Large artifacts
# stay outside Git; override NANOCHAT_BASE_DIR to use attached storage.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

command -v uv >/dev/null 2>&1 || {
  echo "uv is required: https://docs.astral.sh/uv/" >&2
  exit 1
}

PROJECT_CACHE_ROOT="${XDG_CACHE_HOME:-${HOME}/.cache}"
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-${PROJECT_CACHE_ROOT}/allen-gpt}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export WANDB_RUN="${WANDB_RUN:-a100-d26}"

BASE_DEPTH="${BASE_DEPTH:-26}"
BASE_MODEL_TAG="${BASE_MODEL_TAG:-a100-d26}"
SFT_MODEL_TAG="${SFT_MODEL_TAG:-a100-d26-sft}"
SFT_NUM_ITERATIONS="${SFT_NUM_ITERATIONS:-1000}"
OPTIMIZER_KERNEL="${OPTIMIZER_KERNEL:-eager}"

uv sync --frozen --extra gpu
source .venv/bin/activate

python -m nanochat.report reset
python -m nanochat.dataset -n 8
python -m nanochat.dataset -n 370 &
DATASET_DOWNLOAD_PID=$!
python -m scripts.tok_train
python -m scripts.tok_eval
wait "${DATASET_DOWNLOAD_PID}"

python -m torch.distributed.run --standalone --nproc_per_node=8 \
  -m scripts.base_train -- \
  --depth="${BASE_DEPTH}" \
  --model-tag="${BASE_MODEL_TAG}" \
  --target-param-data-ratio=8.5 \
  --device-batch-size=16 \
  --optimizer-kernel="${OPTIMIZER_KERNEL}" \
  --run="${WANDB_RUN}-base"

python -m torch.distributed.run --standalone --nproc_per_node=8 \
  -m scripts.base_eval -- \
  --model-tag="${BASE_MODEL_TAG}" \
  --device-batch-size=16

curl --fail --location --retry 3 \
  --output "${NANOCHAT_BASE_DIR}/identity_conversations.jsonl" \
  https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

python -m torch.distributed.run --standalone --nproc_per_node=8 \
  -m scripts.chat_sft -- \
  --model-tag="${BASE_MODEL_TAG}" \
  --output-model-tag="${SFT_MODEL_TAG}" \
  --num-iterations="${SFT_NUM_ITERATIONS}" \
  --device-batch-size=2 \
  --packing-strategy=best_fit \
  --optimizer-kernel="${OPTIMIZER_KERNEL}" \
  --save-every=100 \
  --run="${WANDB_RUN}-sft"

python -m torch.distributed.run --standalone --nproc_per_node=8 \
  -m scripts.chat_eval -- \
  -i sft \
  --model-tag="${SFT_MODEL_TAG}"

python -m nanochat.report generate
