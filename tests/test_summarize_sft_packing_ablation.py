from benchmarks.summarize_sft_packing_ablation import summarize_run


def test_summarize_run_computes_effective_content_throughput(tmp_path):
    path = tmp_path / "best_fit.json"
    path.write_text("{}", encoding="utf-8")
    payload = {
        "config": {"packing_strategy": "best_fit"},
        "result": {
            "global_packer": {
                "packing_efficiency": 0.9,
                "padding_ratio": 0.1,
                "content_tokens_emitted": 90,
                "padding_tokens_emitted": 10,
                "target_tokens_emitted": 50,
            },
            "timing": {
                "tokens_per_second_median": 1000,
                "mfu_percent_median": 20,
                "measured_steps": 5,
            },
            "validation_bpb": 0.5,
            "final_train_loss": 1.0,
            "peak_memory_bytes_rank0": 123,
        },
    }

    result = summarize_run(
        path,
        payload,
        sequential_effective_tps=600,
        sequential_efficiency=0.6,
    )

    assert result["median_effective_content_tokens_per_second"] == 900
    assert result["content_per_padded_token_ratio_vs_sequential"] == 1.5
    assert (
        result["observed_effective_content_throughput_ratio_vs_sequential"]
        == 1.5
    )
