"""CUDA microbenchmark for compiled and eager AdamW optimizer kernels."""

import argparse
import json
import statistics
import time

import torch

from nanochat.optim import _adamw_step_compiled, _adamw_step_eager


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=32768)
    parser.add_argument("--columns", type=int, default=1664)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--output", default="")
    return parser.parse_args()


def make_inputs(rows, columns):
    parameter = torch.randn(rows, columns, device="cuda", dtype=torch.bfloat16)
    gradient = torch.randn_like(parameter)
    first_moment = torch.zeros_like(parameter)
    second_moment = torch.zeros_like(parameter)
    scalars = [
        torch.tensor(value, dtype=torch.float32)
        for value in (3.0, 1e-3, 0.9, 0.95, 1e-8, 0.01)
    ]
    return parameter, gradient, first_moment, second_moment, *scalars


def benchmark(function, args):
    inputs = make_inputs(args.rows, args.columns)
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


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    results = {
        "device": torch.cuda.get_device_name(),
        "shape": [args.rows, args.columns],
        "compiled": benchmark(_adamw_step_compiled, args),
        "eager": benchmark(_adamw_step_eager, args),
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
