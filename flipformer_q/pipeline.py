from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

from .config import Config
from .checkpoint import load_model_weights
from .data import write_sharded_dataset
from .evolution import build_model, evaluate_individual, score_from_metrics
from .train import train_model
from .utils.io_utils import save_json
from .utils.runtime import derive_seed, resolve_device, seed_everything


def run_stage(cfg: Config, stage: str, root_dir: Path, teacher_ckpt: Optional[Path] = None, round_idx: int = 0):
    cfg.validate()
    if stage not in {"seed", "selftrain", "distill"}:
        raise ValueError(f"Unsupported training stage: {stage}")
    if stage in {"selftrain", "distill"} and teacher_ckpt is None:
        raise ValueError(f"stage={stage!r} requires --teacher-ckpt")
    root_dir = Path(root_dir)
    root_dir.mkdir(parents=True, exist_ok=True)
    scfg = Config(**{**cfg.to_dict(), "seed": derive_seed(cfg.seed, "stage", stage, round_idx)}).validate()
    dataset_mode = {"seed": "bootstrap", "selftrain": "selftrain", "distill": "distill"}[stage]
    dataset_dir = root_dir / f"dataset_{stage}_r{round_idx}"
    run_dir = root_dir / f"run_{stage}_r{round_idx}"
    seed_everything(scfg.seed)
    device = resolve_device(scfg.device)

    teacher_model = None
    if teacher_ckpt is not None and stage in {"selftrain", "distill"}:
        teacher_model = build_model(scfg).to(device)
        load_model_weights(teacher_model, teacher_ckpt, map_location=device)
        teacher_model.eval()

    manifest = write_sharded_dataset(
        cfg=scfg,
        out_dir=dataset_dir,
        total_samples=scfg.train_samples,
        shard_size=scfg.shard_size,
        model=teacher_model,
        device=device,
        dataset_mode=dataset_mode,
    )
    if teacher_model is not None:
        del teacher_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    model = build_model(scfg)
    final_ckpt = train_model(scfg, model, manifest, run_dir, init_ckpt=teacher_ckpt)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    metrics = evaluate_individual(scfg, final_ckpt, root_dir / f"eval_{stage}_r{round_idx}", mode="classical")
    result = {
        "stage": stage,
        "round": round_idx,
        "manifest": str(manifest),
        "ckpt": str(final_ckpt),
        "metrics": metrics,
        "score": score_from_metrics(metrics),
    }
    save_json(result, root_dir / f"stage_{stage}_r{round_idx}.json")
    return result


def iterative_self_training(cfg: Config, root_dir: Path):
    cfg.validate()
    root_dir = Path(root_dir)
    root_dir.mkdir(parents=True, exist_ok=True)
    history = []

    seed_result = run_stage(cfg, stage="seed", root_dir=root_dir, teacher_ckpt=None, round_idx=0)
    history.append(seed_result)
    best = seed_result
    best_ckpt = Path(seed_result["ckpt"])

    stale = 0
    for r in range(1, cfg.selftrain_rounds + 1):
        stage = "selftrain" if r < cfg.selftrain_rounds else "distill"
        result = run_stage(cfg, stage=stage, root_dir=root_dir, teacher_ckpt=best_ckpt, round_idx=r)
        history.append(result)
        if result["score"] > best["score"] + cfg.accept_margin:
            best = result
            best_ckpt = Path(result["ckpt"])
            stale = 0
        else:
            stale += 1
        if stale >= cfg.patience:
            break

    summary = {"best": best, "history": history}
    save_json(summary, root_dir / "iterative_summary.json")
    return summary
