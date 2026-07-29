# Results policy

`legacy_2026_02.json` records only values traceable to the archived checkpoint
metadata and training report. The mismatched step-169150 evaluation is retained
as an excluded result so the discrepancy is visible rather than silently
removed.

New benchmark outputs belong in the ignored `benchmark_results/` directory
during development. Publish reviewed JSON files here only after adding the Git
commit, config hash, checkpoint checksum, hardware, command, and raw metrics.

## Reviewed results

- `packing_cpu_2026_07.json`: three-seed CPU microbenchmark of sequential,
  first-fit, length-bucket, and best-fit SFT packing at batch size 16 and
  sequence length 2,048. The embedded provenance points to clean commit
  `680413c9c402e745429aed012a5990273d3bfb52`. It does not measure GPU training
  throughput or model quality.
- `base_model_014889_a800.json` and `base_model_014889_a800.csv`: reviewed
  2×A800 reevaluation of the available historical base checkpoint. The JSON
  records the checkpoint/tokenizer hashes, exact command, clean evaluation
  commit, BPB, and aggregate CORE score; the CSV contains every task result.
- `exact_resume_2x_a800.json`: two-rank A800 acceptance test of the production
  checkpoint/data path. Continuous and interrupted/resumed execution match
  exactly for batches, losses, parameters, optimizer tensors, and packer state
  on both ranks.
- `optimizer_kernels_a800.json`: one-A800 AdamW compiled/eager microbenchmark
  using five AB/BA-ordered rounds and CUDA Event timing, plus five-step BF16
  parity. It does not measure end-to-end training throughput.
- `kv_cache_sft_d26_a800.json`: one-A800 naive/KV-cache crossover benchmark on
  the verified step-747 SFT checkpoint. It records per-repeat output hashes,
  TTFT, TPOT, peak memory, checkpoint SHA-256, and clean-tree provenance. KV
  Cache is an upstream capability, not original work in this fork.
