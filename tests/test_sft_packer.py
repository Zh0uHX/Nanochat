import copy
import unittest

from nanochat.sft_packer import StatefulDistributedSFTPacker


class FakeTokenizer:
    def __init__(self, bos_token=0):
        self.bos_token = bos_token

    def get_bos_token_id(self):
        return self.bos_token

    def render_conversation(self, conversation, max_tokens=None):
        tokens = list(conversation["tokens"])
        loss_mask = conversation.get("loss_mask")
        return tokens, None if loss_mask is None else list(loss_mask)


def make_dataset(lengths):
    dataset = []
    next_token = 10
    for length in lengths:
        tokens = [0] + list(range(next_token, next_token + max(length - 1, 0)))
        dataset.append({"tokens": tokens})
        next_token += max(length - 1, 0) + 10
    return dataset


class TestStatefulDistributedSFTPacker(unittest.TestCase):
    def test_shapes_and_loss_mask(self):
        packer = StatefulDistributedSFTPacker(
            make_dataset([3, 2, 4]),
            FakeTokenizer(),
            batch_size=2,
            sequence_len=5,
            buffer_size=3,
            strategy="best_fit",
        )

        batch = packer.next_batch()

        self.assertEqual(batch.batch_size, 2)
        self.assertEqual(batch.row_capacity, 6)
        self.assertTrue(all(len(row) == 6 for row in batch.rows))
        for content_length, mask in zip(
            batch.content_lengths, batch.target_loss_mask()
        ):
            self.assertEqual(len(mask), 5)
            self.assertEqual(sum(mask), max(content_length - 1, 0))

    def test_assistant_only_mask_survives_packing(self):
        dataset = [
            {
                "tokens": [0, 11, 12, 13, 14],
                "loss_mask": [0, 0, 0, 1, 1],
            }
        ]
        packer = StatefulDistributedSFTPacker(
            dataset,
            FakeTokenizer(),
            batch_size=1,
            sequence_len=5,
            buffer_size=1,
            strategy="sequential",
        )

        batch = packer.next_batch()

        self.assertEqual(
            batch.target_loss_mask(),
            [[False, False, True, True, False]],
        )
        self.assertEqual(packer.metrics.target_tokens_emitted, 2)

    def test_exact_resume_reproduces_next_batches(self):
        kwargs = {
            "batch_size": 2,
            "sequence_len": 8,
            "buffer_size": 4,
            "strategy": "best_fit",
        }
        dataset = make_dataset([2, 3, 4, 5, 6, 7, 8])
        original = StatefulDistributedSFTPacker(
            dataset, FakeTokenizer(), **kwargs
        )
        original.next_batch()
        checkpoint = copy.deepcopy(original.state_dict())
        expected_second = original.next_batch()
        expected_third = original.next_batch()

        restored = StatefulDistributedSFTPacker(
            dataset, FakeTokenizer(), **kwargs
        )
        restored.load_state_dict(checkpoint)

        self.assertEqual(restored.next_batch(), expected_second)
        self.assertEqual(restored.next_batch(), expected_third)
        self.assertEqual(restored.state_dict(), original.state_dict())

    def test_rank_partitions_are_disjoint_within_epoch(self):
        dataset = make_dataset([5] * 10)
        common = {
            "batch_size": 1,
            "sequence_len": 4,
            "world_size": 2,
            "buffer_size": 1,
            "strategy": "sequential",
        }
        rank0 = StatefulDistributedSFTPacker(
            dataset, FakeTokenizer(), rank=0, **common
        )
        rank1 = StatefulDistributedSFTPacker(
            dataset, FakeTokenizer(), rank=1, **common
        )

        indices0 = {
            rank0.next_batch().sample_indices[0][0][1] for _ in range(5)
        }
        indices1 = {
            rank1.next_batch().sample_indices[0][0][1] for _ in range(5)
        }

        self.assertFalse(indices0 & indices1)
        self.assertEqual(indices0 | indices1, set(range(10)))

    def test_oversize_conversation_is_explicitly_truncated(self):
        packer = StatefulDistributedSFTPacker(
            make_dataset([20]),
            FakeTokenizer(),
            batch_size=1,
            sequence_len=7,
            buffer_size=1,
            strategy="sequential",
            oversize_policy="truncate",
        )

        batch = packer.next_batch()

        self.assertEqual(batch.content_lengths, [8])
        self.assertEqual(packer.metrics.truncated_conversations, 1)
        self.assertEqual(packer.metrics.truncated_tokens, 12)
        self.assertEqual(packer.metrics.padding_tokens_emitted, 0)

    def test_oversize_error_policy_fails_closed(self):
        packer = StatefulDistributedSFTPacker(
            make_dataset([20]),
            FakeTokenizer(),
            batch_size=1,
            sequence_len=7,
            buffer_size=1,
            oversize_policy="error",
        )
        with self.assertRaisesRegex(ValueError, "exceeding row capacity"):
            packer.next_batch()

    def test_incompatible_state_is_rejected(self):
        dataset = make_dataset([2, 3, 4])
        source = StatefulDistributedSFTPacker(
            dataset,
            FakeTokenizer(),
            batch_size=1,
            sequence_len=8,
            strategy="best_fit",
        )
        state = source.state_dict()
        target = StatefulDistributedSFTPacker(
            dataset,
            FakeTokenizer(),
            batch_size=1,
            sequence_len=9,
            strategy="best_fit",
        )

        with self.assertRaisesRegex(ValueError, "incompatible"):
            target.load_state_dict(state)

    def test_dataset_identity_participates_in_fingerprint(self):
        dataset = make_dataset([2, 3, 4])
        source = StatefulDistributedSFTPacker(
            dataset,
            FakeTokenizer(),
            batch_size=1,
            sequence_len=8,
            dataset_id="mixture-revision-a",
        )
        target = StatefulDistributedSFTPacker(
            dataset,
            FakeTokenizer(),
            batch_size=1,
            sequence_len=8,
            dataset_id="mixture-revision-b",
        )
        with self.assertRaisesRegex(ValueError, "incompatible"):
            target.load_state_dict(source.state_dict())

    def test_all_strategies_report_bounded_efficiency(self):
        dataset = make_dataset([2, 3, 4, 5, 6, 7])
        for strategy in ("sequential", "first_fit", "best_fit", "length_bucket"):
            with self.subTest(strategy=strategy):
                packer = StatefulDistributedSFTPacker(
                    dataset,
                    FakeTokenizer(),
                    batch_size=2,
                    sequence_len=8,
                    buffer_size=4,
                    strategy=strategy,
                    bucket_width=2,
                )
                packer.next_batch()
                summary = packer.metrics.summary()
                self.assertGreaterEqual(summary["packing_efficiency"], 0.0)
                self.assertLessEqual(summary["packing_efficiency"], 1.0)
                self.assertAlmostEqual(
                    summary["packing_efficiency"] + summary["padding_ratio"], 1.0
                )


if __name__ == "__main__":
    unittest.main()
