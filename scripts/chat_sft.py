"""Supervised fine-tuning with deterministic packing and exact rank-local resume.

Examples:

    python -m scripts.chat_sft --num-iterations=20 --device-batch-size=1

    torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft -- \
        --model-tag=d26 --num-iterations=1000 --device-batch-size=2
"""

import argparse
from contextlib import nullcontext
import os
import statistics
import sys
import time

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import torch
import wandb

from nanochat.checkpoint_manager import (
    load_model,
    load_optimizer_state,
    load_rank_state,
    mark_checkpoint_complete,
    save_checkpoint,
    save_json_atomic,
    save_rank_state,
    validate_checkpoint_complete,
)
from nanochat.common import (
    DummyWandb,
    autodetect_device_type,
    compute_cleanup,
    compute_init,
    get_base_dir,
    get_peak_flops,
    print0,
)
from nanochat.loss_eval import evaluate_bpb
from nanochat.optim import set_optimizer_kernel_mode
from nanochat.provenance import collect_run_provenance
from nanochat.sft_packer import (
    PackedBatch,
    PackerMetrics,
    StatefulDistributedSFTPacker,
)
from nanochat.tokenizer import get_token_bytes
from tasks.common import TaskMixture
from tasks.customjson import CustomJSON
from tasks.gsm8k import GSM8K
from tasks.mmlu import MMLU
from tasks.smoltalk import SmolTalk
from tasks.spellingbee import SimpleSpelling, SpellingBee


RESUME_CRITICAL_KEYS = (
    "dtype",
    "num_iterations",
    "max_seq_len",
    "device_batch_size",
    "total_batch_size",
    "packing_strategy",
    "validation_packing_strategy",
    "packing_buffer_size",
    "packing_bucket_width",
    "oversize_policy",
    "optimizer_kernel",
)
TRAIN_DATASET_ID = (
    "sft-train-v1:smoltalk+mmlu+2gsm8k+2identity+simple-spelling+"
    "spellingbee:shuffle42"
)
VALIDATION_DATASET_ID = "sft-val-v1:smoltalk+mmlu+gsm8k:shuffle42"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Supervised fine-tuning with exact data-pipeline resume"
    )
    parser.add_argument("--run", default="dummy")
    parser.add_argument(
        "--device-type", default="", choices=["", "cuda", "cpu", "mps"]
    )
    parser.add_argument(
        "--dtype", default="bfloat16", choices=["float32", "bfloat16"]
    )
    parser.add_argument(
        "--compile-model",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="compile the SFT model; disabled by default for broad A100 compatibility",
    )
    parser.add_argument(
        "--optimizer-kernel",
        choices=["compiled", "eager"],
        default="compiled",
        help="compiled fast path or eager compatibility fallback",
    )

    parser.add_argument("--model-tag", default=None, help="base checkpoint tag")
    parser.add_argument("--model-step", type=int, default=None)
    parser.add_argument(
        "--output-model-tag",
        default=None,
        help="SFT checkpoint tag; defaults to --model-tag or d<depth>",
    )
    parser.add_argument("--resume-from-step", type=int, default=-1)

    parser.add_argument(
        "--num-iterations",
        type=int,
        default=-1,
        help="optimization steps; required for DDP, -1 means one approximate epoch on one rank",
    )
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--device-batch-size", type=int, default=32)
    parser.add_argument("--total-batch-size", type=int, default=524288)

    parser.add_argument("--embedding-lr", type=float, default=0.3)
    parser.add_argument("--unembedding-lr", type=float, default=0.004)
    parser.add_argument("--matrix-lr", type=float, default=0.02)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--init-lr-frac", type=float, default=1.0)

    parser.add_argument("--eval-every", type=int, default=150)
    parser.add_argument("--eval-tokens", type=int, default=20 * 524288)
    parser.add_argument("--save-every", type=int, default=-1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--result-output",
        default="",
        help="optional rank-0 JSON summary for fixed-budget benchmark runs",
    )

    parser.add_argument(
        "--packing-strategy",
        choices=["sequential", "first_fit", "best_fit", "length_bucket"],
        default="best_fit",
    )
    parser.add_argument(
        "--validation-packing-strategy",
        choices=["sequential", "first_fit", "best_fit", "length_bucket"],
        default="best_fit",
        help="fixed validation policy; keep constant across training ablations",
    )
    parser.add_argument("--packing-buffer-size", type=int, default=100)
    parser.add_argument("--packing-bucket-width", type=int, default=64)
    parser.add_argument(
        "--oversize-policy", choices=["truncate", "error"], default="truncate"
    )
    return parser.parse_args()


def build_datasets(base_dir):
    identity_path = os.path.join(base_dir, "identity_conversations.jsonl")
    train_dataset = TaskMixture(
        [
            SmolTalk(split="train"),
            MMLU(subset="auxiliary_train", split="train"),
            GSM8K(subset="main", split="train"),
            GSM8K(subset="main", split="train"),
            CustomJSON(filepath=identity_path),
            CustomJSON(filepath=identity_path),
            SimpleSpelling(size=200000, split="train"),
            SpellingBee(size=80000, split="train"),
        ]
    )
    validation_dataset = TaskMixture(
        [
            SmolTalk(split="test"),
            MMLU(subset="all", split="test", stop=5200),
            GSM8K(subset="main", split="test", stop=420),
        ]
    )
    return train_dataset, validation_dataset


def build_packer(
    dataset,
    tokenizer,
    args,
    rank,
    world_size,
    dataset_id,
    *,
    strategy=None,
):
    return StatefulDistributedSFTPacker(
        dataset,
        tokenizer,
        batch_size=args.device_batch_size,
        sequence_len=args.max_seq_len,
        rank=rank,
        world_size=world_size,
        buffer_size=args.packing_buffer_size,
        strategy=args.packing_strategy if strategy is None else strategy,
        bucket_width=args.packing_bucket_width,
        oversize_policy=args.oversize_policy,
        dataset_id=dataset_id,
    )


def aggregate_packer_summaries(summaries):
    aggregate = PackerMetrics()
    for field_name in PackerMetrics.__dataclass_fields__:
        setattr(
            aggregate,
            field_name,
            sum(int(summary[field_name]) for summary in summaries),
        )
    return aggregate.summary()


def summarize_step_measurements(measurements):
    if not measurements:
        return {
            "warmup_steps_excluded": 10,
            "measured_steps": 0,
        }
    elapsed_ms = [row["elapsed_ms"] for row in measurements]
    tokens_per_second = [row["tokens_per_second"] for row in measurements]
    mfu = [row["mfu_percent"] for row in measurements]
    return {
        "warmup_steps_excluded": 10,
        "measured_steps": len(measurements),
        "elapsed_ms_mean": statistics.fmean(elapsed_ms),
        "elapsed_ms_median": statistics.median(elapsed_ms),
        "tokens_per_second_mean": statistics.fmean(tokens_per_second),
        "tokens_per_second_median": statistics.median(tokens_per_second),
        "mfu_percent_mean": statistics.fmean(mfu),
        "mfu_percent_median": statistics.median(mfu),
    }


def batch_to_tensors(batch: PackedBatch, device, device_type):
    use_cuda = device_type == "cuda"
    batch_tensor = torch.tensor(
        batch.rows,
        dtype=torch.long,
        pin_memory=use_cuda,
    )
    inputs = batch_tensor[:, :-1].to(
        device=device,
        dtype=torch.int32,
        non_blocking=use_cuda,
    )
    targets = batch_tensor[:, 1:].to(
        device=device,
        dtype=torch.int64,
        non_blocking=use_cuda,
    )
    target_loss_mask = torch.tensor(
        batch.target_loss_mask(),
        dtype=torch.bool,
        pin_memory=use_cuda,
    ).to(device=device, non_blocking=use_cuda)
    targets.masked_fill_(~target_loss_mask, -1)
    return inputs, targets


def tensor_batch_iterator(packer, device, device_type):
    while True:
        yield batch_to_tensors(packer.next_batch(), device, device_type)


def validate_resume_config(saved_config, current_config):
    mismatches = []
    for key in RESUME_CRITICAL_KEYS:
        if saved_config.get(key) != current_config.get(key):
            mismatches.append(
                f"{key}: checkpoint={saved_config.get(key)!r}, "
                f"current={current_config.get(key)!r}"
            )
    if mismatches:
        raise ValueError(
            "resume configuration differs from the checkpoint:\n  "
            + "\n  ".join(mismatches)
        )


def get_lr_multiplier(progress):
    return 1.0 if progress < 0.8 else max(1.0 - (progress - 0.8) / 0.2, 0.0)


def get_muon_momentum(step):
    fraction = min(step / 300, 1.0)
    return (1.0 - fraction) * 0.85 + fraction * 0.95


def main():
    args = parse_args()
    user_config = vars(args).copy()
    set_optimizer_kernel_mode(args.optimizer_kernel)
    run_provenance = collect_run_provenance(user_config)

    device_type = (
        autodetect_device_type() if args.device_type == "" else args.device_type
    )
    ddp, rank, local_rank, world_size, device = compute_init(device_type)
    if ddp and args.num_iterations <= 0:
        raise ValueError(
            "--num-iterations must be positive for DDP so every rank executes "
            "the same number of collective operations"
        )
    master_process = rank == 0
    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    autocast_ctx = (
        torch.amp.autocast(device_type=device_type, dtype=dtype)
        if device_type == "cuda"
        else nullcontext()
    )
    synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None
    get_max_memory = (
        torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0
    )
    if device_type == "cuda":
        gpu_peak_flops = get_peak_flops(torch.cuda.get_device_name(local_rank))
    else:
        gpu_peak_flops = float("inf")

    use_dummy_wandb = args.run == "dummy" or not master_process
    wandb_run = (
        DummyWandb()
        if use_dummy_wandb
        else wandb.init(project="nanochat-sft", name=args.run, config=user_config)
    )

    base_dir = get_base_dir()
    resuming = args.resume_from_step >= 0
    if resuming and args.output_model_tag is None:
        if args.model_tag is None:
            raise ValueError(
                "--output-model-tag or --model-tag is required when resuming SFT"
            )
        output_model_tag = args.model_tag
    else:
        output_model_tag = args.output_model_tag

    if resuming:
        checkpoint_dir = os.path.join(
            base_dir,
            "chatsft_checkpoints",
            output_model_tag,
        )
        validate_checkpoint_complete(
            checkpoint_dir,
            args.resume_from_step,
            world_size,
        )
        model, tokenizer, meta = load_model(
            "sft",
            device,
            phase="train",
            model_tag=output_model_tag,
            step=args.resume_from_step,
        )
        validate_resume_config(meta["user_config"], user_config)
    else:
        model, tokenizer, meta = load_model(
            "base",
            device,
            phase="train",
            model_tag=args.model_tag,
            step=args.model_step,
        )

    pretrain_batch_size = meta.get("device_batch_size")
    if pretrain_batch_size and args.device_batch_size > pretrain_batch_size:
        print0(
            "WARNING: SFT device batch size exceeds the pretraining device "
            f"batch size ({args.device_batch_size} > {pretrain_batch_size})"
        )

    original_model = model
    if args.compile_model:
        model = torch.compile(model, dynamic=False)
    depth = original_model.config.n_layer
    if output_model_tag is None:
        output_model_tag = args.model_tag or f"d{depth}"
    checkpoint_dir = os.path.join(base_dir, "chatsft_checkpoints", output_model_tag)

    tokens_per_micro_batch = args.device_batch_size * args.max_seq_len
    world_tokens_per_micro_batch = tokens_per_micro_batch * world_size
    if args.total_batch_size % world_tokens_per_micro_batch != 0:
        raise ValueError(
            "total batch size must be divisible by per-rank micro-batch tokens "
            f"times world size ({args.total_batch_size} % "
            f"{world_tokens_per_micro_batch} != 0)"
        )
    gradient_accumulation_steps = (
        args.total_batch_size // world_tokens_per_micro_batch
    )
    print0(
        f"Tokens/micro-batch/rank: {tokens_per_micro_batch:,}; "
        f"world: {world_tokens_per_micro_batch:,}; "
        f"gradient accumulation: {gradient_accumulation_steps}"
    )

    optimizer = model.setup_optimizer(
        unembedding_lr=args.unembedding_lr,
        embedding_lr=args.embedding_lr,
        matrix_lr=args.matrix_lr,
        weight_decay=args.weight_decay,
    )
    for group in optimizer.param_groups:
        group["lr"] *= args.init_lr_frac
        group["initial_lr"] = group["lr"]
    if resuming:
        optimizer.load_state_dict(
            load_optimizer_state(
                checkpoint_dir,
                args.resume_from_step,
                device=device,
                rank=rank,
            )
        )

    train_dataset, validation_dataset = build_datasets(base_dir)
    train_packer = build_packer(
        train_dataset,
        tokenizer,
        args,
        rank=rank,
        world_size=world_size,
        dataset_id=TRAIN_DATASET_ID,
    )
    if resuming:
        rank_state = load_rank_state(
            checkpoint_dir, args.resume_from_step, rank=rank
        )
        if int(rank_state["step"]) != args.resume_from_step:
            raise ValueError("rank-local state step does not match checkpoint step")
        train_packer.load_state_dict(rank_state["pending_packer_state"])

    def next_training_batch():
        pending_state = train_packer.state_dict()
        tensors = batch_to_tensors(
            train_packer.next_batch(), device=device, device_type=device_type
        )
        return tensors[0], tensors[1], pending_state

    inputs, targets, pending_packer_state = next_training_batch()
    token_bytes = get_token_bytes(device=device)
    num_flops_per_token = original_model.estimate_flops()

    if resuming:
        step = int(meta["step"])
        loop_state = meta["loop_state"]
        validation_bpb = meta.get("val_bpb")
        min_validation_bpb = float(loop_state["min_val_bpb"])
        smooth_train_loss = float(loop_state["smooth_train_loss"])
        total_training_time = float(loop_state["total_training_time"])
    else:
        step = 0
        validation_bpb = None
        min_validation_bpb = float("inf")
        smooth_train_loss = 0.0
        total_training_time = 0.0

    ema_beta = 0.9
    last_mfu = 0.0
    debiased_loss = None
    step_measurements = []
    while True:
        if args.num_iterations > 0:
            last_step = step >= args.num_iterations
            progress = min(step / args.num_iterations, 1.0)
        else:
            packed = train_packer.metrics.conversations_packed
            last_step = packed >= len(train_dataset)
            progress = min(packed / len(train_dataset), 1.0)

        flops_so_far = num_flops_per_token * args.total_batch_size * step

        should_evaluate = last_step or (
            args.eval_every > 0 and step % args.eval_every == 0
        )
        if should_evaluate:
            model.eval()
            validation_packer = build_packer(
                validation_dataset,
                tokenizer,
                args,
                rank=rank,
                world_size=world_size,
                dataset_id=VALIDATION_DATASET_ID,
                strategy=args.validation_packing_strategy,
            )
            validation_loader = tensor_batch_iterator(
                validation_packer, device=device, device_type=device_type
            )
            evaluation_steps = args.eval_tokens // world_tokens_per_micro_batch
            if evaluation_steps <= 0:
                raise ValueError("--eval-tokens is too small for one evaluation step")
            with autocast_ctx:
                validation_bpb = evaluate_bpb(
                    model, validation_loader, evaluation_steps, token_bytes
                )
            print0(f"Step {step:05d} | validation bpb: {validation_bpb:.4f}")
            min_validation_bpb = min(min_validation_bpb, validation_bpb)
            wandb_run.log(
                {
                    "step": step,
                    "total_training_flops": flops_so_far,
                    "total_training_time": total_training_time,
                    "val/bpb": validation_bpb,
                }
            )
            model.train()

        should_save = (
            not args.dry_run
            and (
                last_step
                or (
                    args.save_every > 0
                    and step > 0
                    and step != args.resume_from_step
                    and step % args.save_every == 0
                )
            )
        )
        if should_save:
            save_rank_state(
                checkpoint_dir,
                step,
                rank,
                {
                    "step": step,
                    "pending_packer_state": pending_packer_state,
                },
            )
            save_checkpoint(
                checkpoint_dir,
                step,
                original_model.state_dict(),
                optimizer.state_dict(),
                {
                    "step": step,
                    "val_bpb": validation_bpb,
                    "model_config": {
                        "sequence_len": args.max_seq_len,
                        "vocab_size": tokenizer.get_vocab_size(),
                        "n_layer": depth,
                        "n_head": original_model.config.n_head,
                        "n_kv_head": original_model.config.n_kv_head,
                        "n_embd": original_model.config.n_embd,
                        "window_pattern": original_model.config.window_pattern,
                    },
                    "user_config": user_config,
                    "provenance": run_provenance,
                    "device_batch_size": args.device_batch_size,
                    "max_seq_len": args.max_seq_len,
                    "packer_config": train_packer.state_dict()["config"],
                    "loop_state": {
                        "min_val_bpb": min_validation_bpb,
                        "smooth_train_loss": smooth_train_loss,
                        "total_training_time": total_training_time,
                    },
                },
                rank=rank,
            )
            if ddp:
                torch.distributed.barrier()
            if master_process:
                mark_checkpoint_complete(checkpoint_dir, step, world_size)
            if ddp:
                torch.distributed.barrier()

        if last_step:
            break

        synchronize()
        start_time = time.time()
        accumulated_train_loss = torch.zeros((), device=device)
        for _ in range(gradient_accumulation_steps):
            with autocast_ctx:
                loss = model(inputs, targets)
            accumulated_train_loss += loss.detach()
            (loss / gradient_accumulation_steps).backward()
            inputs, targets, pending_packer_state = next_training_batch()

        learning_rate_multiplier = get_lr_multiplier(progress)
        muon_momentum = get_muon_momentum(step)
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] * learning_rate_multiplier
            if group["kind"] == "muon":
                group["momentum"] = muon_momentum
        optimizer.step()
        model.zero_grad(set_to_none=True)
        accumulated_train_loss /= gradient_accumulation_steps
        if ddp:
            torch.distributed.all_reduce(
                accumulated_train_loss,
                op=torch.distributed.ReduceOp.AVG,
            )
        train_loss_value = accumulated_train_loss.item()
        synchronize()
        elapsed = time.time() - start_time
        step += 1

        smooth_train_loss = (
            ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_value
        )
        debiased_loss = smooth_train_loss / (1 - ema_beta**step)
        flops_so_far = num_flops_per_token * args.total_batch_size * step
        tokens_per_second = int(args.total_batch_size / elapsed)
        flops_per_second = (
            num_flops_per_token * args.total_batch_size / elapsed
        )
        last_mfu = 100 * flops_per_second / (gpu_peak_flops * world_size)
        if step > 10:
            total_training_time += elapsed
            step_measurements.append(
                {
                    "step": step,
                    "elapsed_ms": 1000 * elapsed,
                    "tokens_per_second": tokens_per_second,
                    "mfu_percent": last_mfu,
                }
            )

        packer_summary = train_packer.metrics.summary()
        print0(
            f"step {step:05d} ({100 * progress:.2f}%) | "
            f"loss {debiased_loss:.6f} | lr {learning_rate_multiplier:.2f} | "
            f"{elapsed * 1000:.1f}ms | tok/s {tokens_per_second:,} | "
            f"mfu {last_mfu:.2f}% | packing "
            f"{100 * packer_summary['packing_efficiency']:.2f}%"
        )
        if step % 10 == 0:
            wandb_run.log(
                {
                    "step": step,
                    "total_training_flops": flops_so_far,
                    "total_training_time": total_training_time,
                    "train/loss": debiased_loss,
                    "train/lrm": learning_rate_multiplier,
                    "train/dt": elapsed,
                    "train/tok_per_sec": tokens_per_second,
                    "train/mfu": last_mfu,
                    "data/packing_efficiency": packer_summary[
                        "packing_efficiency"
                    ],
                    "data/padding_ratio": packer_summary["padding_ratio"],
                    "data/truncated_tokens": packer_summary["truncated_tokens"],
                }
            )

    local_packer_summary = train_packer.metrics.summary()
    if ddp:
        rank_packer_summaries = [None] * world_size
        torch.distributed.all_gather_object(
            rank_packer_summaries,
            local_packer_summary,
        )
    else:
        rank_packer_summaries = [local_packer_summary]
    global_packer_summary = aggregate_packer_summaries(rank_packer_summaries)

    peak_memory_bytes = get_max_memory()
    print0(f"Peak memory: {peak_memory_bytes / 1024 / 1024:.2f} MiB")
    print0(f"Training time: {total_training_time / 60:.2f} min")
    if validation_bpb is not None:
        print0(f"Minimum validation bpb: {min_validation_bpb:.4f}")

    if master_process and args.result_output:
        result_payload = {
            "schema_version": 1,
            "benchmark": "sft_packing_fixed_budget",
            "command": [sys.executable, *sys.argv],
            "config": user_config,
            "provenance": run_provenance,
            "environment": {
                "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda,
                "device_type": device_type,
                "device_names": (
                    [
                        torch.cuda.get_device_name(index)
                        for index in range(torch.cuda.device_count())
                    ]
                    if device_type == "cuda"
                    else []
                ),
                "world_size": world_size,
                "dtype": args.dtype,
            },
            "parent_checkpoint": {
                "source": "sft" if resuming else "base",
                "model_tag": (
                    output_model_tag if resuming else args.model_tag
                ),
                "step": int(meta["step"]),
            },
            "model_parameters": sum(
                parameter.numel() for parameter in original_model.parameters()
            ),
            "result": {
                "completed_steps": step,
                "final_train_loss": debiased_loss,
                "validation_bpb": validation_bpb,
                "minimum_validation_bpb": (
                    min_validation_bpb
                    if validation_bpb is not None
                    else None
                ),
                "peak_memory_bytes_rank0": peak_memory_bytes,
                "global_packer": global_packer_summary,
                "timing": summarize_step_measurements(step_measurements),
                "rank0_step_measurements": step_measurements,
                "checkpoint_written": not args.dry_run,
            },
        }
        save_json_atomic(args.result_output, result_payload)
        print0(f"Saved benchmark result: {args.result_output}")

    if not args.dry_run:
        from nanochat.report import get_report

        get_report().log(
            section="SFT",
            data=[
                user_config,
                {
                    "Number of iterations": step,
                    "DDP world size": world_size,
                    "Packer fingerprint": train_packer.config_fingerprint,
                    "Config SHA256": run_provenance["config_sha256"],
                    "Git commit": run_provenance["git"]["commit"],
                    "Git dirty": run_provenance["git"]["dirty"],
                },
                {
                    "Minimum validation bpb": (
                        min_validation_bpb
                        if validation_bpb is not None
                        else None
                    ),
                    "Final MFU %": last_mfu,
                    **global_packer_summary,
                },
            ],
        )

    wandb_run.finish()
    compute_cleanup()


if __name__ == "__main__":
    main()
