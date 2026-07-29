from nanochat.sft_packer import PackerMetrics
from scripts.chat_sft import (
    aggregate_packer_summaries,
    summarize_step_measurements,
)


def test_aggregate_packer_summaries_recomputes_ratios():
    left = PackerMetrics(
        batches_emitted=1,
        rows_emitted=2,
        content_tokens_emitted=90,
        padding_tokens_emitted=10,
    ).summary()
    right = PackerMetrics(
        batches_emitted=1,
        rows_emitted=2,
        content_tokens_emitted=50,
        padding_tokens_emitted=50,
    ).summary()

    result = aggregate_packer_summaries([left, right])

    assert result["batches_emitted"] == 2
    assert result["content_tokens_emitted"] == 140
    assert result["padding_tokens_emitted"] == 60
    assert result["packing_efficiency"] == 0.7
    assert result["padding_ratio"] == 0.3


def test_summarize_step_measurements_uses_post_warmup_samples():
    result = summarize_step_measurements(
        [
            {
                "elapsed_ms": 4.0,
                "tokens_per_second": 100.0,
                "mfu_percent": 20.0,
            },
            {
                "elapsed_ms": 6.0,
                "tokens_per_second": 80.0,
                "mfu_percent": 16.0,
            },
        ]
    )

    assert result["measured_steps"] == 2
    assert result["elapsed_ms_median"] == 5.0
    assert result["tokens_per_second_mean"] == 90.0
    assert result["mfu_percent_median"] == 18.0
