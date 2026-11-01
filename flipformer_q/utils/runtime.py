from __future__ import annotations

import hashlib
import random

import numpy as np
import torch


def resolve_device(requested: str) -> torch.device:
    """Resolve an explicit device without silently moving a GPU run to CPU."""
    value = requested.strip().lower()
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        device = torch.device(value)
    except (RuntimeError, ValueError) as exc:
        raise ValueError(f"Invalid device {requested!r}; use 'auto', 'cpu', 'cuda', or 'cuda:N'.") from exc

    if device.type not in {"cpu", "cuda"}:
        raise ValueError(f"Unsupported device {requested!r}; only CPU and CUDA are supported.")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device {requested!r} was requested, but CUDA is unavailable.")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise RuntimeError(
                f"CUDA device index {device.index} was requested, but only "
                f"{torch.cuda.device_count()} device(s) are visible."
            )
    return device


def seed_everything(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def derive_seed(base_seed: int, *parts: object) -> int:
    """Derive stable independent seeds without Python's process-randomized hash()."""
    payload = "\0".join((str(int(base_seed)), *(str(part) for part in parts))).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "little") & 0x7FFFFFFF
