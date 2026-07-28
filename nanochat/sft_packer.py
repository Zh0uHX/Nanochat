"""Stateful, distributed-aware conversation packing for supervised fine-tuning.

The packer intentionally has no PyTorch dependency. It turns indexable
conversation datasets into fixed-length Python token rows; the training script
owns tensor allocation and device transfer. Keeping these responsibilities
separate makes packing invariants and exact checkpoint/resume behavior testable
on a CPU-only machine.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any, Literal


PackingStrategy = Literal["sequential", "first_fit", "best_fit", "length_bucket"]
OversizePolicy = Literal["truncate", "error"]


@dataclass
class BufferedConversation:
    dataset_index: int
    epoch: int
    tokens: list[int]
    loss_mask: list[bool]
    original_length: int

    def state_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "BufferedConversation":
        return cls(
            dataset_index=int(state["dataset_index"]),
            epoch=int(state["epoch"]),
            tokens=[int(token) for token in state["tokens"]],
            loss_mask=[bool(value) for value in state["loss_mask"]],
            original_length=int(state["original_length"]),
        )


@dataclass
class PackerMetrics:
    batches_emitted: int = 0
    rows_emitted: int = 0
    conversations_fetched: int = 0
    conversations_packed: int = 0
    source_tokens_fetched: int = 0
    source_tokens_packed: int = 0
    content_tokens_emitted: int = 0
    target_tokens_emitted: int = 0
    padding_tokens_emitted: int = 0
    truncated_conversations: int = 0
    truncated_tokens: int = 0

    def summary(self) -> dict[str, int | float]:
        target_capacity = self.content_tokens_emitted + self.padding_tokens_emitted
        return {
            **asdict(self),
            "packing_efficiency": (
                self.content_tokens_emitted / target_capacity
                if target_capacity
                else 0.0
            ),
            "padding_ratio": (
                self.padding_tokens_emitted / target_capacity
                if target_capacity
                else 0.0
            ),
        }


@dataclass
class PackedBatch:
    """One fixed-shape batch before conversion to framework tensors."""

    rows: list[list[int]]
    # Per-token supervision flags aligned with ``rows``. Padding is always false.
    loss_masks: list[list[bool]]
    content_lengths: list[int]
    # Each row contains the (epoch, dataset_index) pairs packed into that row.
    sample_indices: list[list[tuple[int, int]]]

    @property
    def batch_size(self) -> int:
        return len(self.rows)

    @property
    def row_capacity(self) -> int:
        return len(self.rows[0]) if self.rows else 0

    def target_loss_mask(self) -> list[list[bool]]:
        """Return supervision flags aligned with next-token targets ``row[1:]``."""
        return [row_mask[1:] for row_mask in self.loss_masks]


class StatefulDistributedSFTPacker:
    """Pack tokenized conversations with deterministic, exact resume semantics.

    Dataset ownership is determined by ``dataset_index % world_size == rank``.
    This property holds independently in every epoch, so ranks never consume the
    same source example in the same epoch. Checkpoints include the prefetched
    token buffer, not just a cursor; restoring a state therefore reproduces the
    exact next batch instead of an approximate location.
    """

    STATE_VERSION = 2
    VALID_STRATEGIES = {"sequential", "first_fit", "best_fit", "length_bucket"}
    VALID_OVERSIZE_POLICIES = {"truncate", "error"}

    def __init__(
        self,
        dataset: Any,
        tokenizer: Any,
        *,
        batch_size: int,
        sequence_len: int,
        rank: int = 0,
        world_size: int = 1,
        buffer_size: int = 100,
        strategy: PackingStrategy = "best_fit",
        bucket_width: int = 64,
        oversize_policy: OversizePolicy = "truncate",
        dataset_id: str | None = None,
    ) -> None:
        if len(dataset) <= 0:
            raise ValueError("dataset must contain at least one conversation")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if sequence_len <= 0:
            raise ValueError("sequence_len must be positive")
        if world_size <= 0:
            raise ValueError("world_size must be positive")
        if not 0 <= rank < world_size:
            raise ValueError(f"rank must be in [0, {world_size}), got {rank}")
        if rank >= len(dataset):
            raise ValueError(
                f"rank {rank} owns no samples because dataset size is {len(dataset)}"
            )
        if buffer_size <= 0:
            raise ValueError("buffer_size must be positive")
        if strategy not in self.VALID_STRATEGIES:
            raise ValueError(
                f"unknown strategy {strategy!r}; expected one of "
                f"{sorted(self.VALID_STRATEGIES)}"
            )
        if bucket_width <= 0:
            raise ValueError("bucket_width must be positive")
        if oversize_policy not in self.VALID_OVERSIZE_POLICIES:
            raise ValueError(
                f"unknown oversize policy {oversize_policy!r}; expected one of "
                f"{sorted(self.VALID_OVERSIZE_POLICIES)}"
            )

        self.dataset = dataset
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.sequence_len = sequence_len
        self.row_capacity = sequence_len + 1
        self.rank = rank
        self.world_size = world_size
        self.buffer_size = buffer_size
        self.strategy = strategy
        self.bucket_width = bucket_width
        self.oversize_policy = oversize_policy
        self.bos_token = int(tokenizer.get_bos_token_id())
        self.dataset_id = dataset_id or (
            f"{type(dataset).__module__}.{type(dataset).__qualname__}:{len(dataset)}"
        )

        self.epoch = 1
        self.local_position = 0
        self.buffer: list[BufferedConversation] = []
        self.metrics = PackerMetrics()

        self._config = {
            "dataset_size": len(dataset),
            "dataset_id": self.dataset_id,
            "batch_size": batch_size,
            "sequence_len": sequence_len,
            "rank": rank,
            "world_size": world_size,
            "buffer_size": buffer_size,
            "strategy": strategy,
            "bucket_width": bucket_width,
            "oversize_policy": oversize_policy,
            "bos_token": self.bos_token,
        }
        encoded_config = json.dumps(
            self._config, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        self.config_fingerprint = hashlib.sha256(encoded_config).hexdigest()

    def __iter__(self) -> "StatefulDistributedSFTPacker":
        return self

    def __next__(self) -> PackedBatch:
        return self.next_batch()

    def _next_source_location(self) -> tuple[int, int]:
        """Return ``(epoch, dataset_index)`` for the next sample owned by rank."""
        dataset_size = len(self.dataset)
        dataset_index = self.rank + self.local_position * self.world_size
        if dataset_index >= dataset_size:
            self.epoch += 1
            self.local_position = 0
            dataset_index = self.rank
        source_epoch = self.epoch
        self.local_position += 1
        return source_epoch, dataset_index

    def _fetch_one(self) -> BufferedConversation:
        source_epoch, dataset_index = self._next_source_location()
        conversation = self.dataset[dataset_index]
        tokens, loss_mask = self.tokenizer.render_conversation(
            conversation,
            max_tokens=None,
        )
        tokens = [int(token) for token in tokens]
        if loss_mask is None:
            # Generic/testing tokenizers may not distinguish prompt and response.
            loss_mask = [True] * len(tokens)
        else:
            loss_mask = [bool(value) for value in loss_mask]
        if len(loss_mask) != len(tokens):
            raise ValueError(
                f"conversation {dataset_index} returned {len(tokens)} tokens but "
                f"{len(loss_mask)} loss-mask entries"
            )
        if not tokens:
            tokens = [self.bos_token]
            loss_mask = [False]

        original_length = len(tokens)
        self.metrics.conversations_fetched += 1
        self.metrics.source_tokens_fetched += original_length
        if original_length > self.row_capacity:
            if self.oversize_policy == "error":
                raise ValueError(
                    f"conversation {dataset_index} has {original_length} tokens, "
                    f"exceeding row capacity {self.row_capacity}"
                )
            tokens = tokens[: self.row_capacity]
            loss_mask = loss_mask[: self.row_capacity]
            self.metrics.truncated_conversations += 1
            self.metrics.truncated_tokens += original_length - self.row_capacity

        return BufferedConversation(
            dataset_index=dataset_index,
            epoch=source_epoch,
            tokens=tokens,
            loss_mask=loss_mask,
            original_length=original_length,
        )

    def _refill_buffer(self) -> None:
        while len(self.buffer) < self.buffer_size:
            self.buffer.append(self._fetch_one())

    def _select_buffer_index(self, remaining: int) -> int:
        if self.strategy == "sequential":
            return 0 if len(self.buffer[0].tokens) <= remaining else -1

        fitting = [
            (index, len(item.tokens))
            for index, item in enumerate(self.buffer)
            if len(item.tokens) <= remaining
        ]
        if not fitting:
            return -1
        if self.strategy == "first_fit":
            return fitting[0][0]
        if self.strategy == "best_fit":
            return max(fitting, key=lambda pair: pair[1])[0]

        # Approximate best-fit with fixed-width length buckets. Within a bucket,
        # preserve FIFO order; this exposes a lower-resolution policy that can be
        # benchmarked against exact best-fit.
        return max(
            fitting,
            key=lambda pair: (pair[1] // self.bucket_width, -pair[0]),
        )[0]

    def _build_row(
        self,
    ) -> tuple[list[int], list[bool], int, list[tuple[int, int]]]:
        row: list[int] = []
        row_loss_mask: list[bool] = []
        locations: list[tuple[int, int]] = []
        while len(row) < self.row_capacity:
            self._refill_buffer()
            remaining = self.row_capacity - len(row)
            selected_index = self._select_buffer_index(remaining)
            if selected_index < 0:
                break

            item = self.buffer.pop(selected_index)
            row.extend(item.tokens)
            row_loss_mask.extend(item.loss_mask)
            locations.append((item.epoch, item.dataset_index))
            self.metrics.conversations_packed += 1
            self.metrics.source_tokens_packed += item.original_length

        content_length = len(row)
        padding_length = self.row_capacity - content_length
        row.extend([self.bos_token] * padding_length)
        row_loss_mask.extend([False] * padding_length)

        self.metrics.rows_emitted += 1
        self.metrics.content_tokens_emitted += content_length
        self.metrics.target_tokens_emitted += sum(row_loss_mask[1:])
        self.metrics.padding_tokens_emitted += padding_length
        return row, row_loss_mask, content_length, locations

    def next_batch(self) -> PackedBatch:
        rows: list[list[int]] = []
        loss_masks: list[list[bool]] = []
        content_lengths: list[int] = []
        sample_indices: list[list[tuple[int, int]]] = []
        for _ in range(self.batch_size):
            row, row_loss_mask, content_length, locations = self._build_row()
            rows.append(row)
            loss_masks.append(row_loss_mask)
            content_lengths.append(content_length)
            sample_indices.append(locations)
        self.metrics.batches_emitted += 1
        return PackedBatch(
            rows=rows,
            loss_masks=loss_masks,
            content_lengths=content_lengths,
            sample_indices=sample_indices,
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": self.STATE_VERSION,
            "config": dict(self._config),
            "config_fingerprint": self.config_fingerprint,
            "epoch": self.epoch,
            "local_position": self.local_position,
            "buffer": [item.state_dict() for item in self.buffer],
            "metrics": asdict(self.metrics),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        version = int(state.get("version", -1))
        if version != self.STATE_VERSION:
            raise ValueError(
                f"unsupported packer state version {version}; "
                f"expected {self.STATE_VERSION}"
            )
        fingerprint = state.get("config_fingerprint")
        if fingerprint != self.config_fingerprint:
            raise ValueError(
                "packer checkpoint is incompatible with the current dataset or "
                f"configuration: expected {self.config_fingerprint}, got {fingerprint}"
            )

        self.epoch = int(state["epoch"])
        self.local_position = int(state["local_position"])
        self.buffer = [
            BufferedConversation.from_state_dict(item) for item in state["buffer"]
        ]
        metric_state = state["metrics"]
        self.metrics = PackerMetrics(
            **{
                field_name: int(metric_state[field_name])
                for field_name in PackerMetrics.__dataclass_fields__
            }
        )
