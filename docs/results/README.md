# Results policy

`legacy_2026_02.json` records only values traceable to the archived checkpoint
metadata and training report. The mismatched step-169150 evaluation is retained
as an excluded result so the discrepancy is visible rather than silently
removed.

New benchmark outputs belong in the ignored `benchmark_results/` directory
during development. Publish reviewed JSON files here only after adding the Git
commit, config hash, checkpoint checksum, hardware, command, and raw metrics.
