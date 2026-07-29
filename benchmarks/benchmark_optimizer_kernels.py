"""CUDA microbenchmark for compiled and eager AdamW optimizer kernels."""

import argparse
import json
import statistics
import sys
import time

import torch

from nanochat.optim import _adamw_step_compiled, _adamw_step_eager
from nanochat.provenance import collect_run_provenance


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=32768)
    parser.add_argument("--columns", type=int, default=1664)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--parity-steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", default="")
    return parser.parse_args()


def make_inputs(rows, columns, seed):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    parameter = torch.randn(
        rows,
        columns,
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )
    gradient = torch.randn(
        rows,
        columns,
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )
    first_moment = torch.zeros_like(parameter)
    second_moment = torch.zeros_like(parameter)
    scalars = [
        torch.tensor(value, dtype=torch.float32)
        for value in (3.0, 1e-3, 0.9, 0.95, 1e-8, 0.01)
    ]
    return parameter, gradient, first_moment, second_moment, *scalars


def clone_inputs(inputs):
    return tuple(value.clone() for value in inputs)


def benchmark(function, args, seed):
    inputs = make_inputs(args.rows, args.columns, seed)
    for _ in range(args.warmup):
        function(*inputs)
    torch.cuda.synchronize()
    timings = []
    for _ in range(args.repeats):
        start = time.perf_counter()
        function(*inputs)
        torch.cuda.synchronize()
        timings.append(time.perf_counter() - start)
    return {
        "median_ms": 1000 * statistics.median(timings),
        "min_ms": 1000 * min(timings),
        "max_ms": 1000 * max(timings),
    }


def parity_check(args):
    base_inputs = make_inputs(args.rows, args.columns, args.seed)
    eager_inputs = clone_inputs(base_inputs)
    compiled_inputs = clone_inputs(base_inputs)
    for step in range(1, args.parity_steps + 1):
        eager_inputs[4].fill_(step)
        compiled_inputs[4].fill_(step)
        _adamw_step_eager(*eager_inputs)
        _adamw_step_compiled(*compiled_inputs)
    torch.cuda.synchronize()
    tensor_names = ("parameter", "gradient", "first_moment", "second_moment")
    differences = {}
    for name, eager, compiled in zip(
        tensor_names,
        eager_inputs[:4],
        compiled_inputs[:4],
    ):
        absolute = (eager.float() - compiled.float()).abs()
        denominator = eager.float().abs().clamp_min(torch.finfo(torch.float32).eps)
        differences[name] = {
            "max_abs": absolute.max().item(),
            "max_rel": (absolute / denominator).max().item(),
        }
    return {
        "steps": args.parity_steps,
        "tensors": differences,
        "max_abs": max(row["max_abs"] for row in differences.values()),
        "max_rel": max(row["max_rel"] for row in differences.values()),
    }


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    config = {
        "rows": args.rows,
        "columns": args.columns,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "parity_steps": args.parity_steps,
        "seed": args.seed,
    }
    properties = torch.cuda.get_device_properties(0)
    results = {
        "schema_version": 2,
        "benchmark": "adamw_optimizer_kernel",
        "config": config,
        "command": [sys.executable, *sys.argv],
        "provenance": collect_run_provenance(config),
        "device": torch.cuda.get_device_name(),
        "device_capability": list(torch.cuda.get_device_capability()),
        "device_memory_bytes": properties.total_memory,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "shape": [args.rows, args.columns],
        "parity": parity_check(args),
        "compiled": benchmark(_adamw_step_compiled, args, args.seed),
        "eager": benchmark(_adamw_step_eager, args, args.seed),
    }
    results["compiled_speedup"] = (
        results["eager"]["median_ms"] / results["compiled"]["median_ms"]
    )
    rendered = json.dumps(results, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(rendered + "\n")


if __name__ == "__main__":
    main()
