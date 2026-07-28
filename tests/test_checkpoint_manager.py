from types import SimpleNamespace

import torch

from nanochat.checkpoint_manager import (
    _patch_missing_keys,
    load_optimizer_state,
    load_rank_state,
    mark_checkpoint_complete,
    save_checkpoint,
    save_rank_state,
    validate_checkpoint_complete,
)


def test_value_embedding_compatibility_patch_is_neutral():
    config = SimpleNamespace(
        n_layer=2,
        n_head=2,
        n_kv_head=1,
        n_embd=4,
        vocab_size=65,
    )
    model_data = {
        "transformer.wte.weight": torch.ones(128, 4, dtype=torch.float32)
    }

    _patch_missing_keys(model_data, config, device=torch.device("cpu"))

    assert torch.equal(model_data["resid_lambdas"], torch.ones(2))
    assert torch.equal(model_data["x0_lambdas"], torch.zeros(2))
    assert torch.equal(
        model_data["transformer.h.1.attn.ve_gate.weight"],
        torch.zeros(1, 32),
    )
    assert torch.equal(
        model_data["value_embeds.1.weight"],
        torch.zeros(128, 2),
    )


def test_checkpoint_writes_every_optimizer_shard_and_rank_state(tmp_path):
    checkpoint_dir = str(tmp_path)
    metadata = {
        "step": 4,
        "model_config": {"vocab_size": 8},
    }
    save_checkpoint(
        checkpoint_dir,
        4,
        {"weight": torch.arange(4)},
        {"rank": 0},
        metadata,
        rank=0,
    )
    save_checkpoint(
        checkpoint_dir,
        4,
        {"unused_on_nonzero_rank": torch.tensor(0)},
        {"rank": 1},
        metadata,
        rank=1,
    )
    save_rank_state(checkpoint_dir, 4, 0, {"step": 4, "cursor": 10})
    save_rank_state(checkpoint_dir, 4, 1, {"step": 4, "cursor": 11})
    mark_checkpoint_complete(checkpoint_dir, 4, world_size=2)

    assert (tmp_path / "model_000004.pt").exists()
    assert (tmp_path / "meta_000004.json").exists()
    assert (tmp_path / "optim_000004_rank0.pt").exists()
    assert (tmp_path / "optim_000004_rank1.pt").exists()
    assert (tmp_path / "complete_000004.json").exists()
    assert load_optimizer_state(checkpoint_dir, 4, "cpu", rank=1) == {
        "rank": 1
    }
    assert load_rank_state(checkpoint_dir, 4, rank=0)["cursor"] == 10
    assert load_rank_state(checkpoint_dir, 4, rank=1)["cursor"] == 11
    assert validate_checkpoint_complete(checkpoint_dir, 4, world_size=2)[
        "schema_version"
    ] == 1
