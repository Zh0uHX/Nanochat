# Model card: Nanochat research checkpoints

## Status

This repository does not contain model weights. Checkpoints are intentionally
stored outside Git and should be published separately with a checksum and a
matching run manifest.

The historical February 2026 run predates the reproducibility changes in this
fork. It demonstrates training execution, but it is not sufficient evidence for
new architectural or performance claims.

## Historical configuration

- Architecture: upstream nanochat Decoder-only Transformer
- Parameters: 1,681,790,292
- Context length: 2,048
- Vocabulary: 32,768
- Pretraining: 7,806,124,032 tokens, 14,889 optimizer steps
- Hardware: 8× NVIDIA A100-SXM4-80GB
- Precision: BF16 (`fp8=False`)
- Recorded wall time: 701.81 minutes
- Recorded MFU: 45.86%
- SFT: 747 optimizer steps, recorded validation BPB 0.3245

These values come from the archived checkpoint metadata and local reports. The
archived base evaluation names step 169150 while the available base checkpoint
is step 14889; therefore its CORE/BPB values are excluded from verified model
quality claims.

## Verified artifact identities

The local archive and A800 copies were compared byte-for-byte with SHA-256:

| Artifact | Step | Bytes | SHA-256 |
|---|---:|---:|---|
| Base model | 14,889 | 5,200,509,259 | `e9bb6f3bbe483e8bc603df389ac96e7c4dac97ac3ef7862a1c7f76759d8e3f7a` |
| SFT model | 747 | 5,200,509,196 | `29243bd0d986a21ed9725fdcb2c0b6ebedf3cfc14af40b6acfbdab70e3acd56f` |

The runtime tokenizer has vocabulary size 32,768 and SHA-256
`c3945190ac5c38a6ac941480e21895763775619e756cde4d108cafb5a081e185`
for `tokenizer.pkl`. The SFT checkpoint was loaded on commit `c626a2a` and
completed deterministic generation. See
[`docs/results/legacy_2026_02.json`](docs/results/legacy_2026_02.json) for the
machine-readable manifest.

## Intended use

- education and LLM systems research;
- data-pipeline, optimizer, and inference benchmarking;
- small-scale controlled experiments.

## Out-of-scope use

- production decision making;
- high-stakes medical, legal, financial, or safety applications;
- claims of factual reliability or alignment;
- public deployment without abuse controls and an independent safety review.

## Limitations

Training data are predominantly English web text and synthetic/general
instruction data. The model can hallucinate, reproduce biases or memorized
content, generate unsafe text, and perform poorly outside the evaluated task
distribution. No comprehensive safety, privacy, contamination, or memorization
audit has been completed.

## Required release metadata

Every published checkpoint should include:

- SHA-256 checksum;
- Git commit and dirty flag;
- configuration SHA-256;
- parent/base checkpoint identity;
- tokenizer checksum;
- dataset versions and mixture weights;
- hardware, precision, step count, and random seeds;
- evaluation commands and raw result JSON.
