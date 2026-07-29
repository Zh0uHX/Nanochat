from benchmarks.benchmark_kv_cache import (
    aggregate_results,
    compare_modes,
    timed_generation,
)


def test_timed_generation_normalizes_engine_output():
    naive = timed_generation(iter([7, 8]), "cpu")
    cached = timed_generation(iter([([7], [1]), ([8], [1])]), "cpu")

    assert naive["generated_tokens"] == 2
    assert naive["output_token_sha256"] == cached["output_token_sha256"]


def test_compare_modes_reports_speedup_and_output_parity():
    results = [
        {
            "context_length": 128,
            "mode": "naive",
            "repeat": 0,
            "ttft_ms": 4.0,
            "tpot_ms": 8.0,
            "peak_memory_bytes": 100,
            "output_token_sha256": "same",
        },
        {
            "context_length": 128,
            "mode": "kv_cache",
            "repeat": 0,
            "ttft_ms": 2.0,
            "tpot_ms": 4.0,
            "peak_memory_bytes": 80,
            "output_token_sha256": "same",
        },
    ]

    comparison = compare_modes(aggregate_results(results), results)[0]

    assert comparison["outputs_match"] is True
    assert comparison["ttft_speedup"] == 2.0
    assert comparison["tpot_speedup"] == 2.0
    assert comparison["peak_memory_ratio"] == 0.8
