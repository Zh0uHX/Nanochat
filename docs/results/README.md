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
