#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

python -m unittest tests.test_sft_packer tests.test_state_io tests.test_provenance -v
python -m compileall -q nanochat scripts tasks tests benchmarks

if python -c "import torch, pytest" >/dev/null 2>&1; then
  python -m pytest -m "not slow" -q
else
  echo "PyTorch/pytest unavailable; framework-dependent tests skipped."
fi
