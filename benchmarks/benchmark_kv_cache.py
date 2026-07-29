"""Measure TTFT, TPOT, and peak memory with and without KV cache."""

import argparse
from contextlib import nullcontext
import hashlib
import json
import os
import statistics
import sys
import time

import torch

from nanochat.checkpoint_manager import load_model
from nanochat.common import (
    autodetect_device_type,
    compute_cleanup,
    compute_init,
    get_base_dir,
)
from nanochat.engine import Engine
from nanochat.provenance import collect_run_provenance


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["base", "sft"], default="sft")
    parser.add_argument("--model-tag", default=None)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--device-type", choices=["", "cuda", "cpu", "mps"], default="")
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16")
    parser.add_argument("--context-lengths", default="128,512,1024")
    parser.add_argument("--new-tokens", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", default="")
    return parser.parse_args()


def synchronize(device_type):
    if device_type == "cuda":
        torch.cuda.synchronize()


def timed_generation(generator, device_type):
    start = time.perf_counter()
    timestamps = []
    for _ in generator:
        synchronize(device_type)
        timestamps.append(time.perf_counter())
    if not timestamps:
        raise RuntimeError("generator emitted no tokens")
    time_to_first_token = timestamps[0] - start
    inter_token = [
        current - previous for previous, current in zip(timestamps, timestamps[1:])
    ]
    return {
        "ttft_ms": 1000 * time_to_first_token,
        "tpot_ms": 1000 * statistics.mean(inter_token) if inter_token else 0.0,
        "generated_tokens": len(timestamps),
    }


def make_prompt(tokenizer, context_length):
    seed_tokens = tokenizer(
        "A reproducible language-model systems benchmark.",
        prepend="<|bos|>",
    )
    repeats = (context_length + len(seed_tokens) - 1) // len(seed_tokens)
    prompt = (seed_tokens * repeats)[:context_length]
    prompt[0] = tokenizer.get_bos_token_id()
    return prompt


def checkpoint_path(source, model_tag, step):
    source_dir = {
        "base": "base_checkpoints",
        "sft": "chatsft_checkpoints",
    }[source]
    return os.path.join(
        get_base_dir(),
        source_dir,
        model_tag,
        f"model_{step:06d}.pt",
    )


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize(values):
    return {
        "mean": statistics.fmean(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def aggregate_results(results):
    aggregate = []
    keys = sorted({(row["context_length"], row["mode"]) for row in results})
    for context_length, mode in keys:
        rows = [
            row
            for row in results
            if row["context_length"] == context_length and row["mode"] == mode
        ]
        aggregate.append(
            {
                "context_length": context_length,
                "mode": mode,
                "repeats": len(rows),
                "ttft_ms": summarize([row["ttft_ms"] for row in rows]),
                "tpot_ms": summarize([row["tpot_ms"] for row in rows]),
                "peak_memory_bytes": summarize(
                    [row["peak_memory_bytes"] for row in rows]
                ),
            }
        )
    return aggregate


def main():
    args = parse_args()
    config = {
        "source": args.source,
        "model_tag": args.model_tag,
        "step": args.step,
        "device_type": args.device_type,
        "dtype": args.dtype,
        "context_lengths": args.context_lengths,
        "new_tokens": args.new_tokens,
        "warmup": args.warmup,
        "repeats": args.repeats,
    }
    device_type = (
        autodetect_device_type() if args.device_type == "" else args.device_type
    )
    _, _, _, _, device = compute_init(device_type)
    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    autocast = (
        torch.amp.autocast(device_type=device_type, dtype=dtype)
        if device_type == "cuda"
        else nullcontext()
    )
    model, tokenizer, metadata = load_model(
        args.source,
        device,
        phase="eval",
        model_tag=args.model_tag,
        step=args.step,
    )
    engine = Engine(model, tokenizer)
    context_lengths = [int(value) for value in args.context_lengths.split(",")]
    results = []

    for context_length in context_lengths:
        if context_length + args.new_tokens > model.config.sequence_len:
            raise ValueError(
                f"context {context_length} + generation {args.new_tokens} exceeds "
                f"model sequence length {model.config.sequence_len}"
            )
        prompt = make_prompt(tokenizer, context_length)
        for mode in ("naive", "kv_cache"):
            for _ in range(args.warmup):
                with autocast:
                    if mode == "naive":
                        warmup_generator = model.generate(
                            prompt,
                            max_tokens=args.new_tokens,
                            temperature=0,
                        )
                    else:
                        warmup_generator = engine.generate(
                            prompt,
                            num_samples=1,
                            max_tokens=args.new_tokens,
                            temperature=0,
                        )
                    for _ in warmup_generator:
                        pass
                synchronize(device_type)
            for repeat in range(args.repeats):
                if device_type == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                with autocast:
                    if mode == "naive":
                        generator = model.generate(
                            prompt,
                            max_tokens=args.new_tokens,
                            temperature=0,
                        )
                    else:
                        generator = engine.generate(
                            prompt,
                            num_samples=1,
                            max_tokens=args.new_tokens,
                            temperature=0,
                        )
                    timing = timed_generation(generator, device_type)
                peak_memory = (
                    torch.cuda.max_memory_allocated()
                    if device_type == "cuda"
                    else 0
                )
                results.append(
                    {
                        "mode": mode,
                        "context_length": context_length,
                        "repeat": repeat,
                        "peak_memory_bytes": peak_memory,
                        **timing,
                    }
                )

    checkpoint_step = int(metadata["step"])
    resolved_model_tag = args.model_tag
    if resolved_model_tag is None:
        raise ValueError(
            "--model-tag is required for a provenance-complete benchmark"
        )
    model_path = checkpoint_path(args.source, resolved_model_tag, checkpoint_step)
    properties = (
        torch.cuda.get_device_properties(0) if device_type == "cuda" else None
    )
    payload = {
        "schema_version": 2,
        "benchmark": "kv_cache_inference",
        "config": config,
        "command": [sys.executable, *sys.argv],
        "provenance": collect_run_provenance(config),
        "checkpoint": {
            "source": args.source,
            "model_tag": resolved_model_tag,
            "step": checkpoint_step,
            "path": model_path,
            "bytes": os.path.getsize(model_path),
            "sha256": sha256_file(model_path),
        },
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "device": str(device),
        "device_name": (
            torch.cuda.get_device_name(0) if device_type == "cuda" else None
        ),
        "device_capability": (
            list(torch.cuda.get_device_capability(0))
            if device_type == "cuda"
            else None
        ),
        "device_memory_bytes": (
            properties.total_memory if properties is not None else None
        ),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "dtype": args.dtype,
        "aggregate": aggregate_results(results),
        "results": results,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(rendered + "\n")
    compute_cleanup()


if __name__ == "__main__":
    main()
