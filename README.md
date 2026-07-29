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
- The CPU packing, two-rank exact-resume, optimizer-kernel, checkpoint
  reevaluation, and KV-cache crossover benchmarks are reviewed and published.
- End-to-end GPU SFT packing throughput and quality remain intentionally
  unclaimed until the controlled ablation in
  [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) is executed.

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
| Re-evaluated CORE (step 14,889) | 0.2680 |
| Re-evaluated validation BPB | 0.7453 |
| SFT steps | 747 |
| SFT validation BPB | 0.3245 |

The archived evaluation report references a different checkpoint step
(169150 versus the available 14889), so its CORE/BPB values are excluded. The
available checkpoint was reevaluated on 2×A800 using at most 500 examples per
CORE task; see
[the reviewed result](docs/results/base_model_014889_a800.json),
[the legacy manifest](docs/results/legacy_2026_02.json), and
[MODEL_CARD.md](MODEL_CARD.md).

## Reviewed packing result

On a 10,000-sample synthetic long-tailed conversation distribution, with 500
batches per seed, batch size 16, sequence length 2,048, and seeds 42/1337/2026:

| Strategy | Padding ratio (mean ± SD) | Median batch construction |
|---|---:|---:|
| Sequential | 14.965% ± 0.423% | 1.021 ± 0.015 ms |
| First fit | 0.422% ± 0.004% | 1.722 ± 0.010 ms |
| Length bucket | 0.256% ± 0.006% | 2.307 ± 0.017 ms |
| Best fit | 0.185% ± 0.008% | 1.934 ± 0.009 ms |

Best fit reduced padding by 98.76% relative to sequential packing while keeping
CPU batch construction below 2 ms median on the measured machine. This isolates
packer behavior; it is not evidence of end-to-end GPU training speedup or
unchanged model quality. The reviewed raw result includes the clean Git commit,
working-tree hash, configuration hash, and per-seed measurements:
[packing_cpu_2026_07.json](docs/results/packing_cpu_2026_07.json).

## Reviewed A800 systems results

The exact-resume acceptance test interrupted a deterministic two-rank CUDA run
at step 3 of 8, saved the production model/optimizer/packer checkpoint, and
resumed it. On both A800 ranks, the replayed batches, final packer states,
losses, model parameters, and optimizer tensors matched the uninterrupted run
exactly (all maximum absolute differences `0.0`). See
[exact_resume_2x_a800.json](docs/results/exact_resume_2x_a800.json).

The AdamW microbenchmark used five AB/BA-ordered rounds of 20 CUDA Event samples
on a `32768 × 1664` BF16 tensor. The compiled path had 0.454 ms median latency
versus 1.877 ms for eager, a 4.14× median kernel speedup. Five-step BF16 parity
passed at `atol=0.03125`, `rtol=0.01`; parameter relative L2 difference was
`1.55e-6`. This is a kernel microbenchmark, not an end-to-end training speedup.
See [optimizer_kernels_a800.json](docs/results/optimizer_kernels_a800.json).

The existing step-747 SFT checkpoint was also used to characterize the
**upstream** KV-cache implementation against naive full-prefix decoding:

| Context | Naive TPOT | Cached TPOT | TPOT speedup |
|---:|---:|---:|---:|
| 128 | 14.97 ms | 23.10 ms | 0.65× |
| 512 | 25.66 ms | 23.08 ms | 1.11× |
| 1,024 | 45.34 ms | 23.13 ms | 1.96× |
| 1,536 | 60.70 ms | 23.24 ms | 2.61× |
| 1,984 | 42.96 ms | 15.43 ms | 2.78× |

All greedy output-token hashes matched across modes and repeats. The result
also shows the boundary: cache management regresses short-context TPOT and
usually increases TTFT, so no context-independent acceleration is claimed. See
[kv_cache_sft_d26_a800.json](docs/results/kv_cache_sft_d26_a800.json).

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
export NANOCHAT_BASE_DIR=/path/to/large/storage/nanochat
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
uv run --frozen --extra cpu python -m pytest -m "not slow" -q
```

GPU/compiled-kernel parity:

```bash
uv run --frozen --extra gpu python -m pytest tests/test_optimizer_kernels.py -m slow -v
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
NANOCHAT_BASE_DIR=/attached/storage/nanochat \
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
  --seeds=42,1337,2026 \
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

Two-rank exact-resume acceptance test:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
torchrun --standalone --nproc_per_node=2 \
  -m benchmarks.benchmark_exact_resume \
  --output=benchmark_results/exact_resume_2x.json
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
