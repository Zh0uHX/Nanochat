"""Validate and summarize fixed-budget SFT packing benchmark JSON files."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys


STRATEGIES = ("sequential", "first_fit", "length_bucket", "best_fit")
COMMON_CONFIG_KEYS = (
    "model_tag",
    "model_step",
    "num_iterations",
    "device_batch_size",
    "total_batch_size",
    "max_seq_len",
    "dtype",
    "optimizer_kernel",
    "validation_packing_strategy",
    "packing_buffer_size",
    "packing_bucket_width",
    "oversize_policy",
    "eval_every",
    "eval_tokens",
    "dry_run",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint-sha256", default="")
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_runs(input_dir):
    runs = {}
    for strategy in STRATEGIES:
        path = input_dir / f"{strategy}.json"
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        actual_strategy = payload["config"]["packing_strategy"]
        if actual_strategy != strategy:
            raise ValueError(
                f"{path} declares packing strategy {actual_strategy!r}"
            )
        if payload["benchmark"] != "sft_packing_fixed_budget":
            raise ValueError(f"{path} is not an SFT packing benchmark")
        runs[strategy] = (path, payload)
    return runs


def common_fields(payload):
    return {
        key: payload["config"][key]
        for key in COMMON_CONFIG_KEYS
    }


def validate_runs(runs):
    reference = runs["sequential"][1]
    expected = {
        "config": common_fields(reference),
        "environment": reference["environment"],
        "parent_checkpoint": reference["parent_checkpoint"],
        "model_parameters": reference["model_parameters"],
        "git_commit": reference["provenance"]["git"]["commit"],
    }
    for strategy, (_, payload) in runs.items():
        actual = {
            "config": common_fields(payload),
            "environment": payload["environment"],
            "parent_checkpoint": payload["parent_checkpoint"],
            "model_parameters": payload["model_parameters"],
            "git_commit": payload["provenance"]["git"]["commit"],
        }
        if actual != expected:
            raise ValueError(f"{strategy} differs from the common controls")
        if payload["provenance"]["git"]["dirty"]:
            raise ValueError(f"{strategy} was executed from a dirty worktree")
        if payload["result"]["checkpoint_written"]:
            raise ValueError(f"{strategy} unexpectedly wrote a checkpoint")
    return expected


def summarize_run(path, payload, sequential_effective_tps):
    result = payload["result"]
    packer = result["global_packer"]
    timing = result["timing"]
    effective_tps = (
        timing["tokens_per_second_median"] * packer["packing_efficiency"]
    )
    return {
        "strategy": payload["config"]["packing_strategy"],
        "raw_result": path.name,
        "raw_result_sha256": sha256_file(path),
        "packing_efficiency": packer["packing_efficiency"],
        "padding_ratio": packer["padding_ratio"],
        "content_tokens_emitted": packer["content_tokens_emitted"],
        "padding_tokens_emitted": packer["padding_tokens_emitted"],
        "target_tokens_emitted": packer["target_tokens_emitted"],
        "median_padded_tokens_per_second": timing["tokens_per_second_median"],
        "median_effective_content_tokens_per_second": effective_tps,
        "effective_content_speedup_vs_sequential": (
            effective_tps / sequential_effective_tps
        ),
        "median_mfu_percent": timing["mfu_percent_median"],
        "measured_steps_after_warmup": timing["measured_steps"],
        "validation_bpb": result["validation_bpb"],
        "final_train_loss": result["final_train_loss"],
        "peak_memory_bytes_rank0": result["peak_memory_bytes_rank0"],
    }


def build_summary(input_dir, checkpoint_sha256=""):
    runs = load_runs(input_dir)
    common = validate_runs(runs)
    sequential_payload = runs["sequential"][1]
    sequential_effective_tps = (
        sequential_payload["result"]["timing"]["tokens_per_second_median"]
        * sequential_payload["result"]["global_packer"]["packing_efficiency"]
    )
    rows = [
        summarize_run(path, payload, sequential_effective_tps)
        for path, payload in runs.values()
    ]
    return {
        "schema_version": 1,
        "benchmark": "sft_packing_fixed_budget_summary",
        "status": "reviewed_single_run_target_scale_pilot",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "summary_command": [sys.executable, *sys.argv],
        "training_git_commit": common["git_commit"],
        "common_config": common["config"],
        "environment": common["environment"],
        "parent_checkpoint": {
            **common["parent_checkpoint"],
            "sha256": checkpoint_sha256 or None,
        },
        "model_parameters": common["model_parameters"],
        "strategies": rows,
        "limitations": [
            "One 30-step target-scale run per strategy; no multi-seed uncertainty estimate.",
            "The budget fixes padded tokens and optimizer steps, so better packing exposes more content tokens to the model.",
            "Validation uses 262144 padded tokens and is suitable for a pilot comparison, not a definitive quality claim.",
            "Reported throughput excludes the first 10 optimizer steps and does not include dataset/model startup or validation.",
        ],
    }


def main():
    args = parse_args()
    summary = build_summary(
        Path(args.input_dir),
        checkpoint_sha256=args.checkpoint_sha256,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
