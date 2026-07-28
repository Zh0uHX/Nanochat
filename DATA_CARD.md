# Data card

## Pretraining

The upstream pipeline downloads shuffled FineWeb-Edu parquet shards from
`karpathy/fineweb-edu-100b-shuffle`. The historical run used 370 downloaded
shards and recorded 7.806B training tokens.

## Supervised fine-tuning

The default mixture is inherited from nanochat:

- Hugging Face SmolTalk;
- MMLU auxiliary training examples;
- GSM8K, repeated twice;
- synthetic identity conversations, repeated twice;
- generated spelling and letter-counting tasks.

Validation uses SmolTalk test, an MMLU test subset, and a GSM8K test subset.
Dataset loading uses deterministic seed 42 where supported by the upstream task
wrappers.

## Processing

Conversations are rendered by the trained 32K BPE tokenizer and packed into
fixed-length rows. This fork exposes four packing strategies and records
padding, truncation, and source-consumption metrics. Overlength conversations
are truncated or rejected according to an explicit CLI policy.
The tokenizer's per-token supervision mask is preserved during packing:
assistant response/tool-call tokens are supervised, while user prompts and
tool outputs remain excluded from the loss.

## Known risks

- Web and synthetic corpora may contain bias, unsafe content, personal data,
  copyrighted material, duplication, and benchmark contamination.
- Dataset licenses and terms must be reviewed at the source before redistribution.
- The historical run does not include a documented deduplication, PII, toxicity,
  or contamination audit.
- Synthetic identity data intentionally modifies model persona and can introduce
  narrow or repetitive behavior.

## Reproducibility requirements

A new experiment must record dataset repository/revision, split, local file
checksums where practical, mixture weights, tokenizer checksum, packer
fingerprint, truncation count, and random seed. Raw datasets and cached Arrow or
parquet files must not be committed to Git.
