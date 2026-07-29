# Experiment protocol

No performance conclusion belongs in the README or résumé until the
corresponding result file is produced by these protocols.

## 1. SFT packing ablation

Goal: determine whether reduced padding offsets additional CPU selection cost
and whether data ordering changes model quality.

Status: the three-seed CPU microbenchmark is complete and published in
[`docs/results/packing_cpu_2026_07.json`](results/packing_cpu_2026_07.json).
It shows the packing-efficiency/CPU-latency tradeoff only. GPU throughput,
validation BPB, and downstream quality remain pending.

Controls:

- identical base checkpoint and tokenizer;
- identical optimizer, token batch size, sequence length, number of steps, and
  evaluation set;
- strategies: sequential, first-fit, length-bucket, best-fit;
- at least three seeds on a 50–150M proxy model;
- one confirmation run at the target scale after selecting a policy.

Report:

- packing efficiency and padding ratio;
- packer batches/s and p50/p95 CPU batch construction latency;
- training tok/s, MFU, peak device memory;
- validation BPB;
- MMLU, GSM8K, ARC, and spelling/counting task accuracy;
- truncation count and truncated-token ratio.

Commands:

```bash
python -m benchmarks.benchmark_sft_packing \
  --samples=10000 --batches=500 --seeds=42,1337,2026 \
  --output=benchmark_results/packing_cpu.json

BASE_MODEL_TAG=<tag> ABLATION_STEPS=200 NPROC_PER_NODE=8 \
  bash runs/sft_packing_ablation.sh
```

## 2. Optimizer compatibility path

Goal: determine the cost of the eager workaround and verify numerical parity.

```bash
python -m benchmarks.benchmark_optimizer_kernels \
  --parity-steps=5 \
  --output=benchmark_results/optimizer_kernels.json
pytest tests/test_optimizer_kernels.py -m slow -v
```

Report compiled/eager median kernel latency, PyTorch/CUDA/GPU versions, and the
maximum parameter difference after a fixed number of updates. The benchmark
JSON records these fields and clean-tree provenance directly. Do not describe
the eager path as “fused”.

## 3. KV-cache inference

Goal: quantify an upstream capability on the project’s trained checkpoint
without presenting it as an original implementation.

```bash
python -m benchmarks.benchmark_kv_cache \
  --source=sft --model-tag=<tag> \
  --context-lengths=128,512,1024 --new-tokens=64 \
  --output=benchmark_results/kv_cache.json
```

Report TTFT, TPOT, peak memory, context length, dtype, checkpoint step, and
hardware. Compare cached decoding with the model’s naive full-prefix generation.

## 4. Exact-resume acceptance test

For a tiny model:

1. run `N` uninterrupted steps;
2. run `K` steps, save, resume, then run `N-K` steps;
3. compare input token batches, losses, model parameters, optimizer state, and
   packer state;
4. repeat with two ranks.

The current CPU unit tests prove exact packer-batch replay. Full model and
multi-rank bitwise equivalence remains a GPU integration test and must be
reported separately.
