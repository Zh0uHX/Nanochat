#!/usr/bin/env bash
set -euo pipefail

# Fixed-budget SFT packing ablation. Run every strategy from the same base
# checkpoint and compare packing efficiency, tok/s, MFU, validation BPB, and
# downstream chat evaluation.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

BASE_MODEL_TAG="${BASE_MODEL_TAG:?set BASE_MODEL_TAG to an existing base checkpoint}"
ABLATION_STEPS="${ABLATION_STEPS:-200}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-2}"
OPTIMIZER_KERNEL="${OPTIMIZER_KERNEL:-compiled}"

for STRATEGY in sequential first_fit length_bucket best_fit; do
  OUTPUT_TAG="${BASE_MODEL_TAG}-packing-${STRATEGY}"
  python -m torch.distributed.run \
    --standalone \
    --nproc_per_node="${NPROC_PER_NODE}" \
    -m scripts.chat_sft -- \
    --model-tag="${BASE_MODEL_TAG}" \
    --output-model-tag="${OUTPUT_TAG}" \
    --num-iterations="${ABLATION_STEPS}" \
    --device-batch-size="${DEVICE_BATCH_SIZE}" \
    --packing-strategy="${STRATEGY}" \
    --optimizer-kernel="${OPTIMIZER_KERNEL}" \
    --save-every=-1 \
    --run="packing-${STRATEGY}"
done
