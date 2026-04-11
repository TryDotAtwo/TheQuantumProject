from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Optional

import torch

from .config import Config
from .data import write_sharded_dataset
from .evolution import build_model, evaluate_individual
from .train import train_model
from .utils.io_utils import save_json


def score_from_metrics(metrics: dict) -> float:
    return 5.0 * float(metrics["solved_rate"]) - 0.02 * float(metrics["mean_length"]) - 0.0002 * float(metrics["mean_nodes"])


def run_stage(cfg: Config, stage: str, root_dir: Path, teacher_ckpt: Optional[Path] = None, round_idx: int = 0):
    root_dir = Path(root_dir)
    root_dir.mkdir(parents=True, exist_ok=True)
    scfg = cfg.with_stage(stage)
    dataset_mode = "bootstrap" if stage == "seed" else ("selftrain" if stage == "selftrain" else ("distill" if stage == "distill" else "stage3"))
    dataset_dir = root_dir / f"dataset_{stage}_r{round_idx}"
    run_dir = root_dir / f"run_{stage}_r{round_idx}"
    device = torch.device(scfg.device if torch.cuda.is_available() and scfg.device == "cuda" else "cpu")

    teacher_model = None
    if teacher_ckpt is not None and Path(teacher_ckpt).exists():
        teacher_model = build_model(scfg).to(device)
        teacher_state = torch.load(teacher_ckpt, map_location=device)
        teacher_model.load_state_dict(teacher_state["model"], strict=False)
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

    final_quantum = run_stage(cfg, stage="quantum", root_dir=root_dir, teacher_ckpt=best_ckpt, round_idx=len(history))
    history.append(final_quantum)
    if final_quantum["score"] > best["score"] + cfg.accept_margin:
        best = final_quantum

    summary = {"best": best, "history": history}
    save_json(summary, root_dir / "iterative_summary.json")
    return summary
