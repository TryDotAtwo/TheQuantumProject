from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


CHECKPOINT_VERSION = 2


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> Any:
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    # Checkpoints are trusted local training artifacts and include optimizer/RNG state.
    return torch.load(checkpoint_path, map_location=map_location, weights_only=False)


def model_state_dict(payload: Any) -> Mapping[str, torch.Tensor]:
    state = payload.get("model") if isinstance(payload, Mapping) and "model" in payload else payload
    if not isinstance(state, Mapping) or not state or not all(isinstance(key, str) for key in state):
        raise ValueError("Checkpoint does not contain a model state dictionary")
    return state


def load_model_weights(
    model: torch.nn.Module,
    path: str | Path,
    map_location: str | torch.device = "cpu",
) -> Any:
    payload = load_checkpoint(path, map_location)
    model.load_state_dict(model_state_dict(payload), strict=True)
    return payload


def _rng_state(loader_generator: torch.Generator) -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "loader_generator": loader_generator.get_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def save_checkpoint_payload(path: str | Path, payload: Mapping[str, Any]) -> None:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = checkpoint_path.with_name(checkpoint_path.name + ".tmp")
    try:
        torch.save(dict(payload), temporary_path)
        os.replace(temporary_path, checkpoint_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def restore_training_state(
    payload: Any,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    loader_generator: torch.Generator,
) -> int:
    if not isinstance(payload, Mapping) or payload.get("format_version", 0) < CHECKPOINT_VERSION:
        raise ValueError("This checkpoint is weights-only or too old for full resume; use --init-ckpt for a warm start.")
    required = {"optimizer", "scaler", "epoch", "rng"}
    missing = sorted(required - payload.keys())
    if missing:
        raise ValueError(f"Checkpoint cannot resume training; missing state: {', '.join(missing)}")

    optimizer.load_state_dict(payload["optimizer"])
    scaler.load_state_dict(payload["scaler"])
    state = payload["rng"]
    for key in ("python", "numpy", "torch", "loader_generator"):
        if key not in state:
            raise ValueError(f"Checkpoint cannot resume training; missing RNG state {key!r}")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    loader_generator.set_state(state["loader_generator"].cpu())

    saved_cuda = state.get("cuda")
    if saved_cuda is not None:
        if not torch.cuda.is_available() or len(saved_cuda) != torch.cuda.device_count():
            raise RuntimeError("Exact CUDA resume requires the same number of visible CUDA devices as the saved run.")
        torch.cuda.set_rng_state_all([cuda_state.cpu() for cuda_state in saved_cuda])
    return int(payload["epoch"])


def validate_resume_compatibility(
    payload: Any,
    *,
    cfg: Any,
    device: torch.device,
    manifest_sha256: str,
) -> None:
    if not isinstance(payload, Mapping) or not isinstance(payload.get("cfg"), Mapping):
        raise ValueError("Checkpoint does not record the configuration needed for a full resume")
    saved_cfg = payload["cfg"]
    exact_fields = (
        "n",
        "d_model",
        "d_pair",
        "n_heads",
        "n_blocks",
        "recycle_steps",
        "mlp_mult",
        "head_mlp_mult",
        "dropout",
        "batch_size",
        "micro_batch_size",
        "lr",
        "weight_decay",
        "clip_grad_norm",
        "amp",
        "num_workers",
    )
    mismatches = [
        f"{field}: saved={saved_cfg.get(field)!r}, current={cfg.to_dict().get(field)!r}"
        for field in exact_fields
        if saved_cfg.get(field) != cfg.to_dict().get(field)
    ]
    runtime = payload.get("runtime")
    if not isinstance(runtime, Mapping):
        mismatches.append("checkpoint has no runtime metadata")
    else:
        if runtime.get("device") != str(device):
            mismatches.append(f"device: saved={runtime.get('device')!r}, current={str(device)!r}")
        if runtime.get("manifest_sha256") != manifest_sha256:
            mismatches.append("training manifest content changed")
    if mismatches:
        raise ValueError("Full resume requires the same training setup: " + "; ".join(mismatches))


def save_training_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    loader_generator: torch.Generator,
    cfg: Any,
    epoch: int,
    summary: dict[str, Any],
    runtime: Mapping[str, str],
) -> None:
    payload = {
        "format_version": CHECKPOINT_VERSION,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "cfg": cfg.to_dict(),
        "epoch": int(epoch),
        "summary": summary,
        "rng": _rng_state(loader_generator),
        "runtime": dict(runtime),
    }
    save_checkpoint_payload(path, payload)
