from __future__ import annotations

import bisect
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import Config
from .domain import State, all_actions, apply_flip, gap_heuristic, random_scramble_depth, random_state
from .features import all_action_features, state_pair_features, state_single_features
from .search import beam_teacher_targets
from .utils.io_utils import save_json
from .utils.logging_utils import ProgressPrinter


@dataclass
class Example:
    single: np.ndarray
    pair: np.ndarray
    action_feats: np.ndarray
    value: float
    policy: np.ndarray
    q_values: np.ndarray
    far: np.ndarray
    scramble: int


SHARD_ARRAY_KEYS = ("single", "pair", "action_feats", "value", "policy", "q_values", "far", "scramble")
REQUIRED_SHARD_ARRAY_KEYS = SHARD_ARRAY_KEYS[:-1]
FLOAT_SHARD_ARRAY_KEYS = set(REQUIRED_SHARD_ARRAY_KEYS)
INT_SHARD_ARRAY_KEYS = {"scramble"}


def bootstrap_targets(state: State):
    actions = all_actions(len(state))
    q = np.asarray([gap_heuristic(apply_flip(state, action)) for action in actions], dtype=np.float32)
    best = float(q.min())
    opt = (q <= best + 1e-6).astype(np.float32)
    policy = opt / max(float(opt.sum()), 1.0)
    far = (q >= q.max() - 1e-6).astype(np.float32)
    far = far / max(float(far.sum()), 1.0)
    return policy.astype(np.float32), q.astype(np.float32), far.astype(np.float32), best


def generate_example(
    cfg: Config,
    rng: random.Random,
    model,
    device: torch.device,
    dataset_mode: str = "bootstrap",
) -> Example:
    if model is None and dataset_mode != "bootstrap":
        raise ValueError(f"dataset_mode={dataset_mode!r} requires a teacher model")
    scramble = random_scramble_depth(rng, cfg.n, cfg.scramble_min, cfg.scramble_max)
    state = random_state(cfg.n, rng, scramble)

    single = state_single_features(state).astype(np.float32)
    pair = state_pair_features(state).astype(np.float32)
    acts = all_action_features(state).astype(np.float32)

    if dataset_mode == "bootstrap":
        policy, q_values, far, value = bootstrap_targets(state)
    else:
        policy, q_values, far, value = beam_teacher_targets(
            model=model,
            state=state,
            device=device,
            beam_width=cfg.beam_width,
            beam_depth=cfg.beam_depth,
        )

    return Example(
        single=single,
        pair=pair,
        action_feats=acts,
        value=float(value),
        policy=policy.astype(np.float32),
        q_values=q_values.astype(np.float32),
        far=far.astype(np.float32),
        scramble=int(scramble),
    )


def write_sharded_dataset(
    cfg: Config,
    out_dir: Path,
    total_samples: int,
    shard_size: int,
    model,
    device: torch.device,
    dataset_mode: str = "bootstrap",
) -> Path:
    cfg.validate()
    if total_samples <= 0 or shard_size <= 0:
        raise ValueError("total_samples and shard_size must be positive")
    if dataset_mode not in {"bootstrap", "selftrain", "distill", "stage3"}:
        raise ValueError(f"Unsupported dataset_mode: {dataset_mode}")
    if model is None and dataset_mode != "bootstrap":
        raise ValueError(f"dataset_mode={dataset_mode!r} requires a teacher model")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(cfg.seed)
    n_shards = (total_samples + shard_size - 1) // shard_size
    manifest = {
        "schema_version": 2,
        "n": int(cfg.n),
        "total_samples": int(total_samples),
        "shard_size": int(shard_size),
        "dataset_mode": str(dataset_mode),
        "seed": int(cfg.seed),
        "config": cfg.to_dict(),
        "shards": [],
    }

    prog = ProgressPrinter("dataset_generation", total_samples, every_sec=2.0)
    written = 0
    for shard_idx in range(n_shards):
        take = min(shard_size, total_samples - written)
        singles: List[np.ndarray] = []
        pairs: List[np.ndarray] = []
        acts: List[np.ndarray] = []
        values: List[float] = []
        policies: List[np.ndarray] = []
        qvals: List[np.ndarray] = []
        fars: List[np.ndarray] = []
        scrambles: List[int] = []
        for _ in range(take):
            ex = generate_example(cfg, rng, model, device, dataset_mode=dataset_mode)
            singles.append(ex.single)
            pairs.append(ex.pair)
            acts.append(ex.action_feats)
            values.append(ex.value)
            policies.append(ex.policy)
            qvals.append(ex.q_values)
            fars.append(ex.far)
            scrambles.append(ex.scramble)
            written += 1
            prog.update(written, extra=f"scramble={ex.scramble} | mode={dataset_mode}")

        shard_path = out_dir / f"shard_{shard_idx:05d}"
        shard_path.mkdir(parents=True, exist_ok=True)
        arrays = {
            "single": np.stack(singles).astype(np.float32),
            "pair": np.stack(pairs).astype(np.float32),
            "action_feats": np.stack(acts).astype(np.float32),
            "value": np.asarray(values, dtype=np.float32),
            "policy": np.stack(policies).astype(np.float32),
            "q_values": np.stack(qvals).astype(np.float32),
            "far": np.stack(fars).astype(np.float32),
            "scramble": np.asarray(scrambles, dtype=np.int32),
        }
        for key, value in arrays.items():
            np.save(shard_path / f"{key}.npy", value)
        manifest["shards"].append({"path": shard_path.name, "size": int(take), "format": "npy_dir"})

    manifest_path = out_dir / "manifest.json"
    manifest["storage"] = "npy_dir"
    save_json(manifest, manifest_path)
    return manifest_path


class ShardedDataset(Dataset):
    def __init__(self, manifest_path: Path):
        super().__init__()
        self.manifest_path = Path(manifest_path)
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.n = int(self.manifest["n"]) if "n" in self.manifest else None
        if self.n is not None and self.n < 2:
            raise ValueError(f"Manifest n must be at least 2, got {self.n}")
        raw_shards = self.manifest.get("shards", [])
        if not raw_shards:
            raise RuntimeError(f"No shards found in manifest: {self.manifest_path}")

        self.shards: List[Dict[str, object]] = []
        self.cum_sizes: List[int] = []
        total = 0
        for shard_entry in raw_shards:
            if isinstance(shard_entry, str):
                raw_path = shard_entry
                size = None
            elif isinstance(shard_entry, dict):
                raw_path = shard_entry["path"]
                size = shard_entry.get("size")
            else:
                raise TypeError(f"Unsupported shard entry type: {type(shard_entry)}")

            shard_path = self._resolve_shard_path(raw_path)
            if not shard_path.exists():
                raise FileNotFoundError(f"Shard not found: {shard_path}")
            if size is None:
                if shard_path.is_dir():
                    size = int(np.load(shard_path / "single.npy", mmap_mode="r").shape[0])
                else:
                    with np.load(shard_path) as data:
                        size = int(data["single"].shape[0])
            size = int(size)
            if size <= 0:
                raise ValueError(f"Shard size must be positive, got {size}: {shard_path}")
            self.shards.append({"path": shard_path, "size": size})
            total += size
            self.cum_sizes.append(total)

        self.total_size = total
        declared_total = self.manifest.get("total_samples")
        if declared_total is not None and int(declared_total) != total:
            raise ValueError(
                f"Manifest total_samples={declared_total} but shard sizes sum to {total}: {self.manifest_path}"
            )
        uses_npy_dirs = any(Path(entry["path"]).is_dir() for entry in self.shards)
        self._cache_limit = min(len(self.shards), 8) if uses_npy_dirs else 1
        self._cache_data_by_shard: Dict[int, Dict[str, np.ndarray]] = {}
        self._cache_order: List[int] = []

    def _resolve_shard_path(self, raw_path: str | Path) -> Path:
        raw_path = Path(raw_path)
        if raw_path.is_absolute():
            if raw_path.exists():
                return raw_path.resolve()
            portable_candidate = self.manifest_path.parent / raw_path.name
            if portable_candidate.exists():
                return portable_candidate.resolve()
            return raw_path.resolve()

        manifest_candidate = (self.manifest_path.parent / raw_path).resolve()
        if manifest_candidate.exists():
            return manifest_candidate
        # Compatibility for legacy manifests whose relative paths were rooted at cwd.
        return raw_path.resolve()

    def __len__(self) -> int:
        return self.total_size

    def _locate_index(self, idx: int):
        if idx < 0:
            idx += self.total_size
        if idx < 0 or idx >= self.total_size:
            raise IndexError(f"Index out of range: {idx}")
        shard_idx = bisect.bisect_right(self.cum_sizes, idx)
        shard_start = 0 if shard_idx == 0 else self.cum_sizes[shard_idx - 1]
        local_idx = idx - shard_start
        return shard_idx, local_idx

    def _load_shard(self, shard_idx: int) -> Dict[str, np.ndarray]:
        if shard_idx in self._cache_data_by_shard:
            if shard_idx in self._cache_order:
                self._cache_order.remove(shard_idx)
            self._cache_order.append(shard_idx)
            return self._cache_data_by_shard[shard_idx]

        shard_path = self.shards[shard_idx]["path"]
        if Path(shard_path).is_dir():
            data = self._load_npy_shard(Path(shard_path))
        else:
            data = self._load_npz_shard(Path(shard_path))
        expected_rows = int(self.shards[shard_idx]["size"])
        actual_rows = int(data["single"].shape[0])
        if actual_rows != expected_rows:
            raise ValueError(f"Shard {shard_path} has {actual_rows} rows, manifest declares {expected_rows}")

        self._cache_data_by_shard[shard_idx] = data
        self._cache_order.append(shard_idx)
        while len(self._cache_order) > self._cache_limit:
            old_idx = self._cache_order.pop(0)
            self._cache_data_by_shard.pop(old_idx, None)
        return data

    def _load_npy_shard(self, shard_path: Path) -> Dict[str, np.ndarray]:
        data: Dict[str, np.ndarray] = {}
        for key in SHARD_ARRAY_KEYS:
            array_path = shard_path / f"{key}.npy"
            if array_path.exists():
                data[key] = np.load(array_path, mmap_mode="r")
        self._validate_shard_keys(shard_path, data)
        return data

    def _load_npz_shard(self, shard_path: Path) -> Dict[str, np.ndarray]:
        with np.load(shard_path, allow_pickle=False) as archive:
            data = {key: archive[key] for key in archive.files}
        self._validate_shard_keys(shard_path, data)
        return data

    def _validate_shard_keys(self, shard_path: Path, data: Dict[str, np.ndarray]) -> None:
        missing = [key for key in REQUIRED_SHARD_ARRAY_KEYS if key not in data]
        if missing:
            raise RuntimeError(f"Shard {shard_path} missing arrays: {missing}")
        if self.n is None:
            return

        expected = {
            "single": (self.n, 8),
            "pair": (self.n, self.n, 8),
            "action_feats": (self.n - 1, 10),
            "value": (),
            "policy": (self.n - 1,),
            "q_values": (self.n - 1,),
            "far": (self.n - 1,),
        }
        row_count = None
        for key, trailing_shape in expected.items():
            array = data[key]
            shape = tuple(array.shape)
            if not shape:
                raise ValueError(f"Shard array {key!r} has no sample dimension: {shard_path}")
            if row_count is None:
                row_count = shape[0]
            elif shape[0] != row_count:
                raise ValueError(f"Shard arrays have inconsistent row counts in {shard_path}")
            if shape[1:] != trailing_shape:
                expected_shape = (row_count, *trailing_shape)
                raise ValueError(f"Shard array {key!r} has shape {shape}; expected {expected_shape}: {shard_path}")
            if not np.issubdtype(array.dtype, np.number):
                raise TypeError(f"Shard array {key!r} must be numeric, got {array.dtype}: {shard_path}")
        if "scramble" in data:
            scramble = data["scramble"]
            if scramble.ndim != 1 or scramble.shape[0] != row_count:
                raise ValueError(f"Shard array 'scramble' must have shape [{row_count}], got {scramble.shape}")

    def _to_tensor(self, key: str, value) -> torch.Tensor:
        tensor = torch.as_tensor(value)
        if key in FLOAT_SHARD_ARRAY_KEYS and tensor.dtype != torch.float32:
            tensor = tensor.to(torch.float32)
        if key in INT_SHARD_ARRAY_KEYS and tensor.dtype != torch.int64:
            tensor = tensor.to(torch.int64)
        return tensor

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        shard_idx, local_idx = self._locate_index(idx)
        data = self._load_shard(shard_idx)
        item = {
            "single": self._to_tensor("single", data["single"][local_idx]),
            "pair": self._to_tensor("pair", data["pair"][local_idx]),
            "action_feats": self._to_tensor("action_feats", data["action_feats"][local_idx]),
            "value": self._to_tensor("value", data["value"][local_idx]),
            "policy": self._to_tensor("policy", data["policy"][local_idx]),
            "q_values": self._to_tensor("q_values", data["q_values"][local_idx]),
            "far": self._to_tensor("far", data["far"][local_idx]),
        }
        if "scramble" in data:
            item["scramble"] = self._to_tensor("scramble", data["scramble"][local_idx])
        return item

    def __getitems__(self, indices) -> Dict[str, torch.Tensor]:
        if isinstance(indices, torch.Tensor):
            indices = indices.tolist()
        if isinstance(indices, np.ndarray):
            indices = indices.tolist()
        if not isinstance(indices, list):
            return self.__getitem__(int(indices))
        if len(indices) == 0:
            return {}

        groups: Dict[int, Dict[str, List[int]]] = {}
        for out_idx, raw_idx in enumerate(indices):
            shard_idx, local_idx = self._locate_index(int(raw_idx))
            group = groups.setdefault(shard_idx, {"positions": [], "local_indices": []})
            group["positions"].append(out_idx)
            group["local_indices"].append(local_idx)

        batch_arrays: Dict[str, np.ndarray] = {}
        for shard_idx, group in groups.items():
            data = self._load_shard(shard_idx)
            positions = np.asarray(group["positions"], dtype=np.int64)
            local_indices = np.asarray(group["local_indices"], dtype=np.int64)
            for key in SHARD_ARRAY_KEYS:
                if key not in data:
                    continue
                source = data[key]
                if key not in batch_arrays:
                    batch_arrays[key] = np.empty((len(indices),) + source.shape[1:], dtype=source.dtype)
                batch_arrays[key][positions] = source[local_indices]

        return {key: self._to_tensor(key, value) for key, value in batch_arrays.items()}


# Keep the old import name available to existing local scripts.
ShardedNpzDataset = ShardedDataset
