"""
Utilities for saving and loading model/optim/state checkpoints.
"""
import os
import re
import glob
import logging
import tempfile
import torch

from nanochat.common import get_base_dir
from nanochat.gpt import GPT, GPTConfig, has_ve
from nanochat.state_io import load_json, save_json_atomic
from nanochat.tokenizer import get_tokenizer
from nanochat.common import setup_default_logging

# Set up logging
setup_default_logging()
logger = logging.getLogger(__name__)


def log0(message):
    if int(os.environ.get('RANK', 0)) == 0:
        logger.info(message)

def _patch_missing_config_keys(model_config_kwargs):
    """Add default values for new config keys missing in old checkpoints."""
    # Old models were trained with full context (no sliding window)
    if "window_pattern" not in model_config_kwargs:
        model_config_kwargs["window_pattern"] = "L"
        log0("Patching missing window_pattern in model config to 'L'")

def _patch_missing_keys(model_data, model_config, device=None):
    """Add default values for new parameters that may be missing in old checkpoints."""
    n_layer = model_config.n_layer

    # VE (value embedding) shape constants - needed for both reporting and patching
    pad_vocab_size_to = 64
    padded_vocab_size = ((model_config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
    head_dim = model_config.n_embd // model_config.n_head
    kv_dim = model_config.n_kv_head * head_dim
    ve_gate_channels = 32

    # Collect missing params: (name, shape) for reporting
    missing = []
    if "resid_lambdas" not in model_data:
        missing.append(("resid_lambdas", (n_layer,)))
    if "x0_lambdas" not in model_data:
        missing.append(("x0_lambdas", (n_layer,)))
    for i in range(n_layer):
        if not has_ve(i, n_layer):
            continue
        ve_gate_key = f"transformer.h.{i}.attn.ve_gate.weight"
        if ve_gate_key not in model_data:
            missing.append((ve_gate_key, (model_config.n_kv_head, ve_gate_channels)))
        value_embed_key = f"value_embeds.{i}.weight"
        if value_embed_key not in model_data:
            missing.append((value_embed_key, (padded_vocab_size, kv_dim)))

    if missing:
        log0(f"Checkpoint missing {len(missing)} parameters (model requires them):")
        for name, shape in missing:
            log0(f"  - {name}: shape={shape}")

    tensor_values = [
        value for value in model_data.values() if isinstance(value, torch.Tensor)
    ]
    if not tensor_values:
        raise ValueError("checkpoint contains no tensor parameters")

    # Use the requested target device when available and preserve checkpoint dtype.
    if device is None:
        cuda_tensors = [
            value for value in tensor_values if value.device.type == "cuda"
        ]
        if cuda_tensors:
            device = cuda_tensors[0].device
        else:
            device = tensor_values[0].device
    dtype = next(
        (value.dtype for value in tensor_values if value.is_floating_point()),
        torch.get_default_dtype(),
    )

    # resid_lambdas defaults to 1.0 (identity scaling)
    if "resid_lambdas" not in model_data:
        model_data["resid_lambdas"] = torch.ones(n_layer, device=device, dtype=dtype)
        log0("Patching missing resid_lambdas in model data to 1.0")
    # x0_lambdas defaults to 0.0 (disabled)
    if "x0_lambdas" not in model_data:
        model_data["x0_lambdas"] = torch.zeros(n_layer, device=device, dtype=dtype)
        log0("Patching missing x0_lambdas in model data to 0.0")

    # VE (value embedding) modules: missing in old checkpoints (e.g. HF nanochat-d34). Init to identity:
    # - ve_gate zeros => 2*sigmoid(0)=1 (neutral gate); value_embeds zeros => add 0 (no change to v).
    for i in range(n_layer):
        if not has_ve(i, n_layer):
            continue
        ve_gate_key = f"transformer.h.{i}.attn.ve_gate.weight"
        if ve_gate_key not in model_data:
            model_data[ve_gate_key] = torch.zeros(model_config.n_kv_head, ve_gate_channels, device=device, dtype=dtype)
            log0(f"Patching missing {ve_gate_key} with zeros (identity gate)")
        value_embed_key = f"value_embeds.{i}.weight"
        if value_embed_key not in model_data:
            model_data[value_embed_key] = torch.zeros(padded_vocab_size, kv_dim, device=device, dtype=dtype)
            log0(f"Patching missing {value_embed_key} with zeros (identity / no residual)")


def _save_torch_atomic(path, data):
    """Prevent interrupted model/optimizer writes from becoming valid files."""
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    file_descriptor, temporary_path = tempfile.mkstemp(
        dir=directory,
        prefix=f".{os.path.basename(path)}.",
        suffix=".tmp",
    )
    os.close(file_descriptor)
    try:
        torch.save(data, temporary_path)
        with open(temporary_path, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)


def save_checkpoint(checkpoint_dir, step, model_data, optimizer_data, meta_data, rank=0):
    if rank == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)
        # Save the model state parameters
        model_path = os.path.join(checkpoint_dir, f"model_{step:06d}.pt")
        _save_torch_atomic(model_path, model_data)
        logger.info(f"Saved model parameters to: {model_path}")
        # Save the metadata dict as json
        meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
        save_json_atomic(meta_path, meta_data)
        logger.info(f"Saved metadata to: {meta_path}")
    # Note that optimizer state is sharded across ranks, so each rank must save its own.
    if optimizer_data is not None:
        os.makedirs(checkpoint_dir, exist_ok=True)
        optimizer_path = os.path.join(checkpoint_dir, f"optim_{step:06d}_rank{rank:d}.pt")
        _save_torch_atomic(optimizer_path, optimizer_data)
        logger.info(f"Saved optimizer state to: {optimizer_path}")


def get_rank_state_path(checkpoint_dir, step, rank):
    return os.path.join(checkpoint_dir, f"rank_state_{step:06d}_rank{rank:d}.json")


def save_rank_state(checkpoint_dir, step, rank, state_data):
    """Save small JSON-serializable state that differs for every training rank."""
    path = get_rank_state_path(checkpoint_dir, step, rank)
    save_json_atomic(path, state_data)
    logger.info(f"Saved rank-local training state to: {path}")


def load_rank_state(checkpoint_dir, step, rank):
    path = get_rank_state_path(checkpoint_dir, step, rank)
    return load_json(path)


def load_optimizer_state(checkpoint_dir, step, device, rank=0):
    """Load one optimizer shard without loading the model checkpoint a second time."""
    optimizer_path = os.path.join(
        checkpoint_dir, f"optim_{step:06d}_rank{rank:d}.pt"
    )
    return torch.load(optimizer_path, map_location=device)


def get_checkpoint_complete_path(checkpoint_dir, step):
    return os.path.join(checkpoint_dir, f"complete_{step:06d}.json")


def mark_checkpoint_complete(checkpoint_dir, step, world_size):
    """Publish a completion marker only after every rank-local file exists."""
    expected_paths = [
        os.path.join(checkpoint_dir, f"model_{step:06d}.pt"),
        os.path.join(checkpoint_dir, f"meta_{step:06d}.json"),
    ]
    for rank in range(world_size):
        expected_paths.extend(
            [
                os.path.join(
                    checkpoint_dir,
                    f"optim_{step:06d}_rank{rank:d}.pt",
                ),
                get_rank_state_path(checkpoint_dir, step, rank),
            ]
        )
    missing = [path for path in expected_paths if not os.path.isfile(path)]
    if missing:
        raise FileNotFoundError(
            "cannot mark an incomplete checkpoint complete; missing: "
            + ", ".join(missing)
        )
    save_json_atomic(
        get_checkpoint_complete_path(checkpoint_dir, step),
        {
            "schema_version": 1,
            "step": step,
            "world_size": world_size,
        },
    )


def validate_checkpoint_complete(checkpoint_dir, step, world_size):
    marker = load_json(get_checkpoint_complete_path(checkpoint_dir, step))
    if int(marker.get("step", -1)) != step:
        raise ValueError("checkpoint completion marker has the wrong step")
    saved_world_size = int(marker.get("world_size", -1))
    if saved_world_size != world_size:
        raise ValueError(
            "checkpoint world size differs from the current run: "
            f"{saved_world_size} != {world_size}"
        )
    return marker


def load_checkpoint(checkpoint_dir, step, device, load_optimizer=False, rank=0):
    # Load the model state
    model_path = os.path.join(checkpoint_dir, f"model_{step:06d}.pt")
    model_data = torch.load(model_path, map_location=device)
    # Load the optimizer state if requested
    optimizer_data = None
    if load_optimizer:
        optimizer_data = load_optimizer_state(
            checkpoint_dir, step, device=device, rank=rank
        )
    # Load the metadata
    meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
    meta_data = load_json(meta_path)
    return model_data, optimizer_data, meta_data


def build_model(checkpoint_dir, step, device, phase):
    """
    A bunch of repetitive code to build a model from a given checkpoint.
    Returns:
    - base model - uncompiled, not wrapped in DDP
    - tokenizer
    - meta data saved during base model training
    """
    assert phase in ["train", "eval"], f"Invalid phase: {phase}"
    model_data, optimizer_data, meta_data = load_checkpoint(checkpoint_dir, step, device, load_optimizer=False)
    if device.type in {"cpu", "mps"}:
        # Convert bfloat16 tensors to float for CPU inference
        model_data = {
            k: v.float() if v.dtype == torch.bfloat16 else v
            for k, v in model_data.items()
        }
    # Hack: fix torch compile issue, which prepends all keys with _orig_mod.
    model_data = {k.removeprefix("_orig_mod."): v for k, v in model_data.items()}
    model_config_kwargs = meta_data["model_config"]
    _patch_missing_config_keys(model_config_kwargs)
    log0(f"Building model with config: {model_config_kwargs}")
    model_config = GPTConfig(**model_config_kwargs)
    _patch_missing_keys(model_data, model_config, device=device)
    with torch.device("meta"):
        model = GPT(model_config)
    # Load the model state
    model.to_empty(device=device)
    model.init_weights() # note: this is dumb, but we need to init the rotary embeddings. TODO: fix model re-init
    model.load_state_dict(model_data, strict=True, assign=True)
    # Put the model in the right training phase / mode
    if phase == "eval":
        model.eval()
    else:
        model.train()
    # Load the Tokenizer
    tokenizer = get_tokenizer()
    # Sanity check: compatibility between model and tokenizer
    assert tokenizer.get_vocab_size() == model_config_kwargs["vocab_size"], f"Tokenizer vocab size {tokenizer.get_vocab_size()} does not match model config vocab size {model_config_kwargs['vocab_size']}"
    return model, tokenizer, meta_data


def find_largest_model(checkpoints_dir):
    # attempt to guess the model tag: take the biggest model available
    model_tags = [f for f in os.listdir(checkpoints_dir) if os.path.isdir(os.path.join(checkpoints_dir, f))]
    if not model_tags:
        raise FileNotFoundError(f"No checkpoints found in {checkpoints_dir}")
    # 1) normally all model tags are of the form d<number>, try that first:
    candidates = []
    for model_tag in model_tags:
        match = re.match(r"d(\d+)", model_tag)
        if match:
            model_depth = int(match.group(1))
            candidates.append((model_depth, model_tag))
    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]
    # 2) if that failed, take the most recently updated model:
    model_tags.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoints_dir, x)), reverse=True)
    return model_tags[0]


def find_last_step(checkpoint_dir):
    # Look into checkpoint_dir and find model_<step>.pt with the highest step
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "model_*.pt"))
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    last_step = int(max(os.path.basename(f).split("_")[-1].split(".")[0] for f in checkpoint_files))
    return last_step

# -----------------------------------------------------------------------------
# convenience functions that take into account nanochat's directory structure

def load_model_from_dir(checkpoints_dir, device, phase, model_tag=None, step=None):
    if model_tag is None:
        # guess the model tag by defaulting to the largest model
        model_tag = find_largest_model(checkpoints_dir)
        log0(f"No model tag provided, guessing model tag: {model_tag}")
    checkpoint_dir = os.path.join(checkpoints_dir, model_tag)
    if step is None:
        # guess the step by defaulting to the last step
        step = find_last_step(checkpoint_dir)
    assert step is not None, f"No checkpoints found in {checkpoint_dir}"
    # build the model
    log0(f"Loading model from {checkpoint_dir} with step {step}")
    model, tokenizer, meta_data = build_model(checkpoint_dir, step, device, phase)
    return model, tokenizer, meta_data

def load_model(source, *args, **kwargs):
    model_dir = {
        "base": "base_checkpoints",
        "sft": "chatsft_checkpoints",
        "rl": "chatrl_checkpoints",
    }[source]
    base_dir = get_base_dir()
    checkpoints_dir = os.path.join(base_dir, model_dir)
    return load_model_from_dir(checkpoints_dir, *args, **kwargs)
