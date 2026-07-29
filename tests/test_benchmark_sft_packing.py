import unittest

from benchmarks.benchmark_sft_packing import aggregate_runs, resolve_seeds


class PackingBenchmarkTests(unittest.TestCase):
    def test_resolve_seeds_preserves_order_and_deduplicates(self):
        self.assertEqual(resolve_seeds(1337, ""), [1337])
        self.assertEqual(resolve_seeds(1337, "42, 2026,42"), [42, 2026])

    def test_aggregate_runs_reports_cross_seed_statistics(self):
        runs = [
            {
                "results": [
                    self._result("sequential", 0.8, 0.2),
                    self._result("first_fit", 0.9, 0.1),
                    self._result("length_bucket", 0.95, 0.05),
                    self._result("best_fit", 0.99, 0.01),
                ]
            },
            {
                "results": [
                    self._result("sequential", 0.7, 0.3),
                    self._result("first_fit", 0.8, 0.2),
                    self._result("length_bucket", 0.9, 0.1),
                    self._result("best_fit", 0.97, 0.03),
                ]
            },
        ]

        aggregate = {row["strategy"]: row for row in aggregate_runs(runs)}

        self.assertEqual(aggregate["best_fit"]["num_seeds"], 2)
        self.assertAlmostEqual(
            aggregate["best_fit"]["metrics"]["packing_efficiency"]["mean"],
            0.98,
        )
        self.assertAlmostEqual(
            aggregate["sequential"]["metrics"]["padding_ratio"]["mean"],
            0.25,
        )
        self.assertAlmostEqual(
            aggregate["best_fit"]["vs_sequential"][
                "padding_reduction_fraction"
            ],
            0.92,
        )
        self.assertAlmostEqual(
            aggregate["best_fit"]["vs_sequential"][
                "median_batch_latency_multiplier"
            ],
            1.0,
        )

    @staticmethod
    def _result(strategy, efficiency, padding):
        return {
            "strategy": strategy,
            "packing_efficiency": efficiency,
            "padding_ratio": padding,
            "batches_per_second": 100.0,
            "median_batch_ms": 1.0,
            "p95_batch_ms": 2.0,
            "truncated_conversations": 0,
            "truncated_tokens": 0,
        }


if __name__ == "__main__":
    unittest.main()
