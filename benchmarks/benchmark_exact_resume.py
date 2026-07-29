"""Two-rank exact-resume acceptance test for the SFT checkpoint path.

This intentionally uses a tiny deterministic language model so the benchmark
isolates data/checkpoint semantics instead of spending compute on model quality.
It exercises the production packer and checkpoint helpers under torchrun.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import sys
import uuid

import torch
import torch.distributed as dist
import torch.nn.functional as F

from nanochat.checkpoint_manager import (
    load_checkpoint,
    load_rank_state,
    mark_checkpoint_complete,
    save_checkpoint,
    save_rank_state,
    validate_checkpoint_complete,
)
from nanochat.provenance import collect_run_provenance
from nanochat.sft_packer import StatefulDistributedSFTPacker


class SyntheticConversationDataset:
    def __init__(self, size: int, vocab_size: int):
        self.examples = []
        for index in range(size):
            length = 4 + (index * 7) % 13
            tokens = [0] + [
                1 + (index * 17 + offset) % (vocab_size - 1)
                for offset in range(length - 1)
            ]
            self.examples.append(
                {
                    "tokens": tokens,
                    "loss_mask": [False] + [True] * (length - 1),
                }
            )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


class SyntheticTokenizer:
    def get_bos_token_id(self):
        return 0

    def render_conversation(self, conversation, max_tokens=None):
        return list(conversation["tokens"]), list(conversation["loss_mask"])


class TinyLanguageModel(torch.nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, hidden_size)
        self.projection = torch.nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, token_ids):
        return self.projection(self.embedding(token_ids))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--split-step", type=int, default=3)
    parser.add_argument("--dataset-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--sequence-len", type=int, default=31)
    parser.add_argument("--buffer-size", type=int, default=8)
    parser.add_argument("--vocab-size", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--device",
        choices=("cuda", "cpu"),
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--output", default="")
    return parser.parse_args()


def setup_distributed(device_type):
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if device_type == "cuda":
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"
    if world_size > 1:
        dist.init_process_group(backend=backend)
    return rank, local_rank, world_size, device


def validate_deterministic_environment(device_type):
    """Fail before CUDA setup when exact cuBLAS reproducibility is unavailable."""
    if device_type != "cuda":
        return
    workspace_config = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if workspace_config not in {":4096:8", ":16:8"}:
        raise RuntimeError(
            "CUDA exact-resume validation requires "
            "CUBLAS_WORKSPACE_CONFIG=:4096:8 (or :16:8) to be set before launch"
        )


def barrier(world_size):
    if world_size > 1:
        dist.barrier()


def unwrap(model):
    return model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model


def make_training_objects(args, rank, world_size, device):
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    model = TinyLanguageModel(args.vocab_size, args.hidden_size).to(
        device=device,
        dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
    )
    if world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
        )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.0,
    )
    dataset = SyntheticConversationDataset(args.dataset_size, args.vocab_size)
    packer = StatefulDistributedSFTPacker(
        dataset,
        SyntheticTokenizer(),
        batch_size=args.batch_size,
        sequence_len=args.sequence_len,
        rank=rank,
        world_size=world_size,
        buffer_size=args.buffer_size,
        strategy="best_fit",
        dataset_id="exact-resume-synthetic-v1",
    )
    return model, optimizer, packer


def batch_digest(batch):
    payload = {
        "rows": batch.rows,
        "loss_masks": batch.loss_masks,
        "sample_indices": batch.sample_indices,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def train_steps(model, optimizer, packer, steps, device):
    losses = []
    batch_digests = []
    for _ in range(steps):
        batch = packer.next_batch()
        token_rows = torch.tensor(batch.rows, dtype=torch.long, device=device)
        targets = token_rows[:, 1:].clone()
        target_mask = torch.tensor(
            batch.target_loss_mask(),
            dtype=torch.bool,
            device=device,
        )
        targets.masked_fill_(~target_mask, -1)
        optimizer.zero_grad(set_to_none=True)
        logits = model(token_rows[:, :-1])
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=-1,
        )
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().float().cpu()))
        batch_digests.append(batch_digest(batch))
    return losses, batch_digests


def clone_model_state(model):
    return {
        name: tensor.detach().clone()
        for name, tensor in unwrap(model).state_dict().items()
    }


def max_model_difference(left, right):
    return max(
        (left[name].float() - right[name].float()).abs().max().item()
        for name in left
    )


def max_nested_tensor_difference(left, right):
    if isinstance(left, torch.Tensor):
        return (left.float() - right.float()).abs().max().item()
    if isinstance(left, dict):
        return max(
            (
                max_nested_tensor_difference(left[key], right[key])
                for key in left
            ),
            default=0.0,
        )
    if isinstance(left, (list, tuple)):
        return max(
            (
                max_nested_tensor_difference(a, b)
                for a, b in zip(left, right)
            ),
            default=0.0,
        )
    return 0.0 if left == right else float("inf")


def checkpoint_directory(rank, world_size):
    identifier = uuid.uuid4().hex if rank == 0 else None
    if world_size > 1:
        values = [identifier]
        dist.broadcast_object_list(values, src=0)
        identifier = values[0]
    return os.path.join("/tmp", f"nanochat-exact-resume-{identifier}")


def main():
    args = parse_args()
    if not 0 < args.split_step < args.steps:
        raise ValueError("--split-step must be between zero and --steps")
    validate_deterministic_environment(args.device)
    rank, local_rank, world_size, device = setup_distributed(args.device)
    torch.use_deterministic_algorithms(True)
    config = {
        "steps": args.steps,
        "split_step": args.split_step,
        "dataset_size": args.dataset_size,
        "batch_size": args.batch_size,
        "sequence_len": args.sequence_len,
        "buffer_size": args.buffer_size,
        "vocab_size": args.vocab_size,
        "hidden_size": args.hidden_size,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "device": args.device,
        "world_size": world_size,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }

    baseline_model, baseline_optimizer, baseline_packer = make_training_objects(
        args,
        rank,
        world_size,
        device,
    )
    baseline_losses, baseline_batches = train_steps(
        baseline_model,
        baseline_optimizer,
        baseline_packer,
        args.steps,
        device,
    )
    baseline_model_state = clone_model_state(baseline_model)
    baseline_optimizer_state = copy.deepcopy(baseline_optimizer.state_dict())
    baseline_packer_state = copy.deepcopy(baseline_packer.state_dict())
    del baseline_model, baseline_optimizer

    split_model, split_optimizer, split_packer = make_training_objects(
        args,
        rank,
        world_size,
        device,
    )
    before_losses, before_batches = train_steps(
        split_model,
        split_optimizer,
        split_packer,
        args.split_step,
        device,
    )
    checkpoint_dir = checkpoint_directory(rank, world_size)
    save_checkpoint(
        checkpoint_dir,
        args.split_step,
        unwrap(split_model).state_dict(),
        split_optimizer.state_dict(),
        {"step": args.split_step, "config": config},
        rank=rank,
    )
    save_rank_state(
        checkpoint_dir,
        args.split_step,
        rank,
        {"packer": split_packer.state_dict()},
    )
    barrier(world_size)
    if rank == 0:
        mark_checkpoint_complete(checkpoint_dir, args.split_step, world_size)
    barrier(world_size)
    validate_checkpoint_complete(checkpoint_dir, args.split_step, world_size)
    del split_model, split_optimizer, split_packer

    resumed_model, resumed_optimizer, resumed_packer = make_training_objects(
        args,
        rank,
        world_size,
        device,
    )
    model_state, optimizer_state, _ = load_checkpoint(
        checkpoint_dir,
        args.split_step,
        device,
        load_optimizer=True,
        rank=rank,
    )
    unwrap(resumed_model).load_state_dict(model_state)
    resumed_optimizer.load_state_dict(optimizer_state)
    resumed_packer.load_state_dict(
        load_rank_state(checkpoint_dir, args.split_step, rank)["packer"]
    )
    after_losses, after_batches = train_steps(
        resumed_model,
        resumed_optimizer,
        resumed_packer,
        args.steps - args.split_step,
        device,
    )
    resumed_model_state = clone_model_state(resumed_model)
    resumed_optimizer_state = resumed_optimizer.state_dict()
    resumed_packer_state = resumed_packer.state_dict()

    resumed_losses = before_losses + after_losses
    resumed_batches = before_batches + after_batches
    rank_result = {
        "rank": rank,
        "local_rank": local_rank,
        "batch_replay_exact": baseline_batches == resumed_batches,
        "max_loss_abs": max(
            abs(left - right)
            for left, right in zip(baseline_losses, resumed_losses)
        ),
        "max_parameter_abs": max_model_difference(
            baseline_model_state,
            resumed_model_state,
        ),
        "max_optimizer_state_abs": max_nested_tensor_difference(
            baseline_optimizer_state,
            resumed_optimizer_state,
        ),
        "packer_state_exact": baseline_packer_state == resumed_packer_state,
    }
    rank_result["accepted"] = (
        rank_result["batch_replay_exact"]
        and rank_result["max_loss_abs"] == 0.0
        and rank_result["max_parameter_abs"] == 0.0
        and rank_result["max_optimizer_state_abs"] == 0.0
        and rank_result["packer_state_exact"]
    )

    if world_size > 1:
        rank_results = [None] * world_size
        dist.all_gather_object(rank_results, rank_result)
    else:
        rank_results = [rank_result]
    barrier(world_size)

    if rank == 0:
        payload = {
            "schema_version": 1,
            "benchmark": "exact_resume_integration",
            "config": config,
            "command": [sys.executable, *sys.argv],
            "provenance": collect_run_provenance(config),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "device_names": [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ]
            if device.type == "cuda"
            else [],
            "accepted": all(row["accepted"] for row in rank_results),
            "rank_results": rank_results,
        }
        rendered = json.dumps(payload, indent=2, sort_keys=True)
        print(rendered)
        if args.output:
            with open(args.output, "w", encoding="utf-8") as handle:
                handle.write(rendered + "\n")
        shutil.rmtree(checkpoint_dir)
    barrier(world_size)
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
