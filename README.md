# Nanochat

An LLM training-systems portfolio built as an explicitly attributed fork of
[karpathy/nanochat](https://github.com/karpathy/nanochat).

The upstream project supplies the end-to-end baseline: BPE tokenization,
Decoder-only pretraining, SFT/RL, distributed Muon/AdamW, evaluation, KV-cache
inference, CLI, and Web UI. This fork focuses its original work on
**deterministic distributed SFT packing, exact interruption/resume, checkpoint
compatibility, provenance, and reproducible benchmarking**.

## Current status

- Code, tests, and experiment recipes are versioned.
- Large datasets, optimizer states, and model weights are excluded from Git.
- A historical 1.68B run is documented, but it predates the exact-resume
  refactor and is not presented as a validation of the new implementation.
- New GPU performance results are intentionally left unclaimed until the
  protocols in [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) are executed.

## Verified historical run

| Item | Recorded value |
|---|---:|
| Parameters | 1,681,790,292 |
| Pretraining tokens | 7,806,124,032 |
| Pretraining steps | 14,889 |
| Hardware | 8× A100-SXM4-80GB |
| Precision | BF16 |
| Wall time | 701.81 min |
| MFU | 45.86% |
| SFT steps | 747 |
| SFT validation BPB | 0.3245 |

The archived evaluation report references a different checkpoint step
(169150 versus the available 14889), so its CORE/BPB values are excluded.
See [the legacy manifest](docs/results/legacy_2026_02.json) and
[MODEL_CARD.md](MODEL_CARD.md).

## Original work in this fork

### Exact-resume distributed SFT packer

`nanochat/sft_packer.py` provides:

- deterministic rank ownership (`dataset_index % world_size == rank`);
- full rank-local state including the prefetched token buffer;
- configuration fingerprints and fail-closed resume validation;
- explicit overlength policies;
- preservation of tokenizer supervision masks, so prompt and tool-output tokens
  do not leak into assistant-only SFT loss;
- sequential, first-fit, coarse length-bucket, and exact best-fit strategies;
- padding, truncation, and utilization metrics;
- framework-independent CPU tests.

The SFT checkpoint stores the packer state before the currently pending batch.
After a restart, the same batch is reconstructed instead of approximately
resuming from a source cursor.

### Checkpoint and runtime engineering

- Each distributed SFT rank saves its own optimizer shard and packer state.
- Tensor files are atomically replaced, and a checkpoint is resumable only
  after a completion marker verifies every rank-local optimizer/packer shard.
- Legacy checkpoints missing Value Embedding or residual-scalar parameters are
  patched on the target device and checkpoint dtype with a neutral
  initialization.
- Optimizer kernels expose a compiled fast path and an eager compatibility
  fallback, with parity tests and a CUDA microbenchmark.
- Checkpoints and reports include the Git commit, dirty flag, diff hash,
  canonical configuration hash, and a working-tree content hash that also
  covers non-ignored untracked source files.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for ownership boundaries and
the resume data flow.

## Setup

Python 3.10+ and [uv](https://docs.astral.sh/uv/) are required.

CPU development:

```bash
uv sync --frozen --extra cpu
source .venv/bin/activate
bash runs/smoke.sh
```

CUDA 12.8:

```bash
uv sync --frozen --extra gpu
source .venv/bin/activate
```

By default nanochat stores data and checkpoints under `~/.cache/nanochat`.
For attached storage:

```bash
export NANOCHAT_BASE_DIR=/path/to/large/storage/allen-gpt
```

Do not place `.pt`, `.safetensors`, parquet/Arrow caches, or optimizer shards in
Git. Publish model checkpoints separately with SHA-256 checksums and the
metadata required by [MODEL_CARD.md](MODEL_CARD.md).

## Tests

Framework-independent exact-resume tests:

```bash
python -m unittest tests.test_sft_packer tests.test_state_io -v
```

Full CPU suite:

```bash
uv run pytest -m "not slow" -q
```

GPU/compiled-kernel parity:

```bash
uv run pytest tests/test_optimizer_kernels.py -m slow -v
```

GitHub Actions installs the locked CPU environment, performs a syntax check,
and runs the non-slow test suite.

## Training

### Small smoke model

The upstream CPU/MPS recipe remains available:

```bash
bash runs/runcpu.sh
```

### Reproducible 8×A100 pipeline

```bash
NANOCHAT_BASE_DIR=/attached/storage/allen-gpt \
WANDB_RUN=a100-d26 \
bash runs/a100_8x.sh
```

The A100 recipe uses explicit model tags, a fixed SFT step budget, periodic
checkpoints, the exact-resume packer, and the eager optimizer compatibility
path used by the historical environment. Set `OPTIMIZER_KERNEL=compiled` only
after running the parity test and benchmark on the target software stack.

### Resume SFT

```bash
python -m torch.distributed.run --standalone --nproc_per_node=8 \
  -m scripts.chat_sft -- \
  --output-model-tag=a100-d26-sft \
  --resume-from-step=500 \
  --num-iterations=1000 \
  --device-batch-size=2 \
  --packing-strategy=best_fit \
  --optimizer-kernel=eager
```

Resume-critical options must match the checkpoint. A mismatch fails with a
field-by-field error instead of silently changing the data or optimization
trajectory.

## Benchmarks and ablations

CPU packing-policy benchmark:

```bash
python -m benchmarks.benchmark_sft_packing \
  --output=benchmark_results/packing_cpu.json
```

Fixed-budget end-to-end packing ablation:

```bash
BASE_MODEL_TAG=<base-tag> ABLATION_STEPS=200 NPROC_PER_NODE=8 \
  bash runs/sft_packing_ablation.sh
```

KV-cache and optimizer benchmarks:

```bash
python -m benchmarks.benchmark_kv_cache \
  --source=sft --model-tag=<sft-tag> \
  --output=benchmark_results/kv_cache.json

python -m benchmarks.benchmark_optimizer_kernels \
  --output=benchmark_results/optimizer_kernels.json
```

Required controls and reporting fields are defined in
[docs/EXPERIMENTS.md](docs/EXPERIMENTS.md).
Calibrated Chinese résumé bullets and interview boundaries are available in
[docs/RESUME_BULLETS_ZH.md](docs/RESUME_BULLETS_ZH.md).

## Repository map

```text
nanochat/
  sft_packer.py          stateful rank-partitioned packing
  checkpoint_manager.py legacy loading and rank-local state
  provenance.py          config/Git fingerprints
  optim.py               compiled/eager optimizer paths
scripts/
  base_train.py          pretraining with provenance
  chat_sft.py            exact-resume SFT entry point
benchmarks/              packing, KV-cache, optimizer measurements
tests/                   CPU, checkpoint, engine, attention, GPU parity tests
runs/
  a100_8x.sh             reproducible portfolio training recipe
  sft_packing_ablation.sh
docs/                    architecture, experiment protocol, reviewed results
```

## Data, safety, and limitations

See [DATA_CARD.md](DATA_CARD.md) for dataset composition and known risks.
This is a research system, not a production assistant. Trained models may
hallucinate, reproduce harmful or biased content, leak memorized text, and fail
outside the evaluated distribution.

## Attribution and license

The baseline is Copyright © 2025 Andrej Karpathy and distributed under the MIT
License in [LICENSE](LICENSE). Upstream architecture and system capabilities are
not claimed as original work in this fork. The `upstream` remote should point to
`https://github.com/karpathy/nanochat.git`; configure `origin` to the owner’s
portfolio repository before publishing.
