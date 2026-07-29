"""Compare SFT packing policies on an identical synthetic length distribution.

This benchmark isolates CPU packing behavior. End-to-end tok/s, MFU, and
validation BPB must still be measured with ``runs/sft_packing_ablation.sh``.
"""

import argparse
import json
import math
import random
import statistics
import sys
import time

from nanochat.provenance import collect_run_provenance
from nanochat.sft_packer import StatefulDistributedSFTPacker


STRATEGIES = ("sequential", "first_fit", "length_bucket", "best_fit")
AGGREGATE_METRICS = (
    "packing_efficiency",
    "padding_ratio",
    "batches_per_second",
    "median_batch_ms",
    "p95_batch_ms",
    "truncated_conversations",
    "truncated_tokens",
)


class LengthDataset:
    def __init__(self, lengths):
        self.lengths = lengths

    def __len__(self):
        return len(self.lengths)

    def __getitem__(self, index):
        return {"length": self.lengths[index], "index": index}


class SyntheticTokenizer:
    def get_bos_token_id(self):
        return 0

    def render_conversation(self, conversation, max_tokens=None):
        length = conversation["length"]
        return [0] + [conversation["index"] + 1] * (length - 1), None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--batches", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--sequence-len", type=int, default=2048)
    parser.add_argument("--buffer-size", type=int, default=100)
    parser.add_argument("--bucket-width", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--seeds",
        default="",
        help="Comma-separated seeds. Overrides --seed when provided.",
    )
    parser.add_argument("--output", default="")
    return parser.parse_args()


def resolve_seeds(seed, seeds):
    if not seeds.strip():
        return [seed]
    resolved = [int(value.strip()) for value in seeds.split(",") if value.strip()]
    if not resolved:
        raise ValueError("--seeds must contain at least one integer")
    # Preserve the requested order while avoiding accidental duplicate work.
    return list(dict.fromkeys(resolved))


def synthetic_lengths(sample_count, sequence_len, seed):
    rng = random.Random(seed)
    # A long-tailed conversational distribution with a small explicit oversize
    # tail, useful for exercising both packing and truncation accounting.
    lengths = []
    for _ in range(sample_count):
        length = round(math.exp(rng.normalvariate(math.log(180), 1.0)))
        lengths.append(max(2, min(length, sequence_len * 2)))
    return lengths


def benchmark_strategy(strategy, dataset, args):
    packer = StatefulDistributedSFTPacker(
        dataset,
        SyntheticTokenizer(),
        batch_size=args.batch_size,
        sequence_len=args.sequence_len,
        buffer_size=args.buffer_size,
        strategy=strategy,
        bucket_width=args.bucket_width,
    )
    batch_latencies = []
    start = time.perf_counter()
    for _ in range(args.batches):
        batch_start = time.perf_counter()
        packer.next_batch()
        batch_latencies.append(time.perf_counter() - batch_start)
    elapsed = time.perf_counter() - start
    summary = packer.metrics.summary()
    summary.update(
        {
            "strategy": strategy,
            "wall_seconds": elapsed,
            "batches_per_second": args.batches / elapsed,
            "median_batch_ms": 1000 * statistics.median(batch_latencies),
            "p95_batch_ms": 1000
            * sorted(batch_latencies)[round(0.95 * (len(batch_latencies) - 1))],
        }
    )
    return summary


def summarize(values):
    return {
        "mean": statistics.fmean(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def aggregate_runs(runs):
    grouped = {strategy: [] for strategy in STRATEGIES}
    for run in runs:
        for result in run["results"]:
            grouped[result["strategy"]].append(result)
    aggregate = [
        {
            "strategy": strategy,
            "num_seeds": len(grouped[strategy]),
            "metrics": {
                metric: summarize([row[metric] for row in grouped[strategy]])
                for metric in AGGREGATE_METRICS
            },
        }
        for strategy in STRATEGIES
    ]
    baseline = aggregate[0]["metrics"]
    for row in aggregate:
        metrics = row["metrics"]
        row["vs_sequential"] = {
            "packing_efficiency_absolute_gain": (
                metrics["packing_efficiency"]["mean"]
                - baseline["packing_efficiency"]["mean"]
            ),
            "padding_reduction_fraction": (
                1.0
                - metrics["padding_ratio"]["mean"]
                / baseline["padding_ratio"]["mean"]
            ),
            "median_batch_latency_multiplier": (
                metrics["median_batch_ms"]["mean"]
                / baseline["median_batch_ms"]["mean"]
            ),
        }
    return aggregate


def main():
    args = parse_args()
    seeds = resolve_seeds(args.seed, args.seeds)
    config = {
        "samples": args.samples,
        "batches": args.batches,
        "batch_size": args.batch_size,
        "sequence_len": args.sequence_len,
        "buffer_size": args.buffer_size,
        "bucket_width": args.bucket_width,
        "seeds": seeds,
    }
    runs = []
    for seed in seeds:
        lengths = synthetic_lengths(args.samples, args.sequence_len, seed)
        dataset = LengthDataset(lengths)
        runs.append(
            {
                "seed": seed,
                "length_summary": {
                    "min": min(lengths),
                    "median": statistics.median(lengths),
                    "max": max(lengths),
                },
                "results": [
                    benchmark_strategy(strategy, dataset, args)
                    for strategy in STRATEGIES
                ],
            }
        )
    payload = {
        "schema_version": 2,
        "benchmark": "sft_packing_cpu",
        "config": config,
        "command": [sys.executable, *sys.argv],
        "provenance": collect_run_provenance(config),
        "runs": runs,
        "aggregate": aggregate_runs(runs),
    }
    # Keep the original single-seed fields for downstream compatibility.
    if len(runs) == 1:
        payload["length_summary"] = runs[0]["length_summary"]
        payload["results"] = runs[0]["results"]
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(rendered + "\n")


if __name__ == "__main__":
    main()
