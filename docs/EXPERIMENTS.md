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

Status: complete on one NVIDIA A800-SXM4-80GB. Across five AB/BA-ordered rounds
of 20 CUDA Event samples, compiled/eager median latency was 0.454/1.877 ms
(4.14×). Five-step BF16 parity passed at the recorded tolerance. See
[`docs/results/optimizer_kernels_a800.json`](results/optimizer_kernels_a800.json).

```bash
python -m benchmarks.benchmark_optimizer_kernels \
  --parity-steps=5 \
  --output=benchmark_results/optimizer_kernels.json
pytest tests/test_optimizer_kernels.py -m slow -v
```

Report compiled/eager median kernel latency, PyTorch/CUDA/GPU versions, and the
maximum parameter difference after a fixed number of updates. The benchmark
JSON records these fields, BF16 `allclose` status at `atol=0.03125`,
`rtol=0.01`, and clean-tree provenance directly. A large pointwise relative
error near zero is not used as the acceptance criterion. Do not describe the
eager path as “fused”.

## 3. KV-cache inference

Goal: quantify an upstream capability on the project’s trained checkpoint
without presenting it as an original implementation.

Status: complete on the verified step-747 SFT checkpoint. Greedy output hashes
matched for every mode/repeat. Cached TPOT was 0.65× at context 128 and crossed
over to 1.96×/2.61×/2.78× at contexts 1,024/1,536/1,984. See
[`docs/results/kv_cache_sft_d26_a800.json`](results/kv_cache_sft_d26_a800.json).

```bash
python -m benchmarks.benchmark_kv_cache \
  --source=sft --model-tag=<tag> \
  --context-lengths=128,512,1024,1536,1984 \
  --new-tokens=64 --warmup=1 --repeats=3 \
  --output=benchmark_results/kv_cache.json
```

Report TTFT, TPOT, peak memory, context length, dtype, checkpoint step, and
hardware. Compare cached decoding with the model’s naive full-prefix generation.

## 4. Exact-resume acceptance test

The executable acceptance test uses a tiny deterministic language model while
exercising the production SFT packer, atomic model/optimizer checkpoint helpers,
rank-local packer state, completion marker, and reload path:

Status: accepted on 2×A800. Both ranks reproduced batches and packer state
exactly, with zero maximum absolute difference in losses, model parameters, and
optimizer state. See
[`docs/results/exact_resume_2x_a800.json`](results/exact_resume_2x_a800.json).

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
torchrun --standalone --nproc_per_node=2 \
  -m benchmarks.benchmark_exact_resume \
  --steps=8 --split-step=3 \
  --output=benchmark_results/exact_resume_2x.json
```

It performs the following comparison:

1. run `N` uninterrupted steps;
2. run `K` steps, save, resume, then run `N-K` steps;
3. compare input token batches, losses, model parameters, optimizer state, and
   packer state;
4. require exact equality independently on both ranks.

This is a checkpoint/data-path acceptance test, not a model-quality benchmark.
The separate CPU unit tests continue to cover individual packer invariants.
CUDA runs fail fast unless deterministic cuBLAS workspace configuration is set
before process launch.
