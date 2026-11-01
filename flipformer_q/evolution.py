from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from .config import Config
from .checkpoint import load_checkpoint, load_model_weights, model_state_dict, save_checkpoint_payload
from .data import write_sharded_dataset
from .domain import gap_heuristic
from .model import FlipformerModel
from .quantum.aer_emulator import QuantumFeatureEmulator
from .reporting import write_summary_csv
from .search import beam_search
from .train import train_model
from .utils.io_utils import save_json
from .utils.logging_utils import JsonlLogger, ProgressPrinter
from .utils.runtime import derive_seed, resolve_device, seed_everything


@dataclass
class Individual:
    name: str
    generation: int
    parent_a: Optional[str]
    parent_b: Optional[str]
    ckpt: Optional[Path]
    score: float = -1e18
    solved_rate: float = 0.0
    mean_length: Optional[float] = None
    mean_unsolved_gap: Optional[float] = None
    mean_nodes: float = 0.0


def build_model(cfg: Config) -> FlipformerModel:
    cfg.validate()
    return FlipformerModel(
        n=cfg.n,
        single_in=8,
        pair_in=8,
        action_in=10,
        d_model=cfg.d_model,
        d_pair=cfg.d_pair,
        n_heads=cfg.n_heads,
        n_blocks=cfg.n_blocks,
        recycle_steps=cfg.recycle_steps,
        mlp_mult=cfg.mlp_mult,
        head_mlp_mult=cfg.head_mlp_mult,
        dropout=cfg.dropout,
    )


def mutate_state_dict(state_dict: Dict[str, torch.Tensor], scale: float = 0.02) -> Dict[str, torch.Tensor]:
    if scale < 0.0:
        raise ValueError(f"mutation scale must be nonnegative, got {scale}")
    out = {}
    for k, v in state_dict.items():
        out[k] = v.clone() if not torch.is_floating_point(v) else v + scale * torch.randn_like(v)
    return out


def crossover_state_dict(a: Dict[str, torch.Tensor], b: Dict[str, torch.Tensor], alpha: float = 0.5) -> Dict[str, torch.Tensor]:
    if a.keys() != b.keys():
        raise ValueError("Cannot cross over models with different state-dictionary keys")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    out = {}
    for k in a.keys():
        va, vb = a[k], b[k]
        if va.shape != vb.shape or va.dtype != vb.dtype:
            raise ValueError(f"Cannot cross over incompatible tensors at {k!r}: {va.shape}/{vb.shape} vs {va.dtype}/{vb.dtype}")
        out[k] = alpha * va + (1.0 - alpha) * vb if torch.is_floating_point(va) and torch.is_floating_point(vb) else va.clone()
    return out


def score_from_metrics(metrics: Dict[str, object]) -> float:
    mean_length = metrics["mean_length"]
    length_penalty = 0.02 * float(mean_length) if mean_length is not None else 0.0
    solved_rate = float(metrics["solved_rate"])
    unsolved_gap = metrics.get("mean_unsolved_gap")
    residual_penalty = 0.02 * (1.0 - solved_rate) * float(unsolved_gap) if unsolved_gap is not None else 0.0
    return 5.0 * solved_rate - length_penalty - residual_penalty - 0.0002 * float(metrics["mean_nodes"])


def evaluate_individual(cfg: Config, ckpt: Path, out_dir: Path, mode: str = "classical") -> Dict[str, object]:
    if mode not in {"classical", "hybrid"}:
        raise ValueError(f"Unsupported evaluation mode: {mode}")
    cfg.validate()
    device = resolve_device(cfg.device)
    model = build_model(cfg).to(device)
    load_model_weights(model, ckpt, map_location=device)
    model.eval()

    rows = []
    rng = random.Random(cfg.seed + 999)
    quantum = None
    if mode == "hybrid":
        if cfg.quantum_mode == "none":
            raise ValueError("Hybrid evaluation requires --quantum-mode aer_cpu or aer_gpu")
        quantum = QuantumFeatureEmulator(
            n_qubits=cfg.quantum_qubits,
            n_layers=cfg.quantum_layers,
            mode=cfg.quantum_mode,
            cache_size=cfg.quantum_cache_size,
        )
        if not quantum.is_available():
            raise RuntimeError(f"Requested quantum backend {cfg.quantum_mode!r} is unavailable")

    from .domain import random_state
    prog = ProgressPrinter(f"eval {mode}", cfg.benchmark_samples, every_sec=2.0)
    for t in range(cfg.benchmark_samples):
        scramble = rng.randint(cfg.scramble_min, cfg.scramble_max)
        s = random_state(cfg.n, rng, scramble)
        res = beam_search(
            model=model,
            start=s,
            beam_width=cfg.beam_width,
            beam_depth=cfg.beam_depth,
            expand_topk=cfg.expand_topk,
            device=device,
            rerank_topk=cfg.rerank_topk if quantum is not None and quantum.is_available() else 0,
            quantum_emulator=quantum if quantum is not None and quantum.is_available() else None,
            quantum_weight=cfg.quantum_weight,
            uncertainty_margin=cfg.uncertainty_margin,
        )
        rows.append({
            "trial": t,
            "mode": mode,
            "solved": 1 if res.solved else 0,
            "solution_length": len(res.path) if res.solved else "",
            "partial_depth": len(res.path) if not res.solved else "",
            "final_gap": gap_heuristic(res.final_state),
            "nodes_expanded": res.nodes_expanded,
            "q_calls": res.q_calls,
        })
        prog.update(t + 1, extra=f"solved={int(res.solved)} | path_len={len(res.path)} | nodes={res.nodes_expanded}")
    write_summary_csv(rows, out_dir / f"trials_{mode}.csv")
    solved_rate = float(np.mean([r["solved"] for r in rows]))
    solved_lengths = [int(r["solution_length"]) for r in rows if r["solution_length"] != ""]
    mean_length = float(np.mean(solved_lengths)) if solved_lengths else None
    partial_depths = [int(r["partial_depth"]) for r in rows if r["partial_depth"] != ""]
    unsolved_gaps = [int(r["final_gap"]) for r in rows if r["solved"] == 0]
    mean_partial_depth = float(np.mean(partial_depths)) if partial_depths else None
    mean_unsolved_gap = float(np.mean(unsolved_gaps)) if unsolved_gaps else None
    mean_nodes = float(np.mean([r["nodes_expanded"] for r in rows]))
    metrics = {
        "score": 0.0,
        "solved_rate": solved_rate,
        "mean_length": mean_length,
        "mean_partial_depth": mean_partial_depth,
        "mean_unsolved_gap": mean_unsolved_gap,
        "mean_nodes": mean_nodes,
        "quantum_backend": quantum.mode if quantum is not None else None,
        "requested_device": cfg.device,
        "device": str(device),
    }
    metrics["score"] = score_from_metrics(metrics)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return metrics


def evolutionary_self_training(cfg: Config, root_out_dir: Path):
    cfg.validate()
    root_out_dir = Path(root_out_dir)
    root_out_dir.mkdir(parents=True, exist_ok=True)
    lineage_logger = JsonlLogger(root_out_dir / "lineage.jsonl")
    lineage = {"nodes": [], "edges": []}
    device = resolve_device(cfg.device)
    evolution_rng = random.Random(derive_seed(cfg.seed, "evolution"))

    population: List[Individual] = [Individual(name=f"g0_m{i}", generation=0, parent_a=None, parent_b=None, ckpt=None) for i in range(cfg.population_size)]
    best_ind: Optional[Individual] = None
    best_score = -1e18
    stale = 0

    for gen in range(cfg.generations):
        evaluated: List[Individual] = []
        for ind in population:
            stage_cfg = Config(**{**cfg.to_dict(), "seed": derive_seed(cfg.seed, "evolution", gen, ind.name)}).validate()
            seed_everything(stage_cfg.seed)
            model = build_model(stage_cfg)
            if ind.ckpt is not None:
                load_model_weights(model, ind.ckpt, map_location="cpu")
            dataset_dir = root_out_dir / "datasets" / ind.name
            manifest = write_sharded_dataset(
                cfg=stage_cfg,
                out_dir=dataset_dir,
                total_samples=stage_cfg.train_samples,
                shard_size=stage_cfg.shard_size,
                model=model.to(device).eval() if ind.ckpt is not None else None,
                device=device,
                dataset_mode="selftrain" if ind.ckpt is not None else "bootstrap",
            )
            run_dir = root_out_dir / "runs" / ind.name
            ind.ckpt = train_model(stage_cfg, model, manifest, run_dir, init_ckpt=ind.ckpt)
            metrics = evaluate_individual(stage_cfg, ind.ckpt, root_out_dir / "evals" / ind.name, mode="classical")
            ind.score = metrics["score"]
            ind.solved_rate = metrics["solved_rate"]
            ind.mean_length = metrics["mean_length"]
            ind.mean_unsolved_gap = metrics["mean_unsolved_gap"]
            ind.mean_nodes = metrics["mean_nodes"]
            evaluated.append(ind)
            lineage["nodes"].append({
                "name": ind.name,
                "generation": gen,
                "score": ind.score,
                "solved_rate": ind.solved_rate,
                "mean_length": ind.mean_length,
                "mean_unsolved_gap": ind.mean_unsolved_gap,
                "mean_nodes": ind.mean_nodes,
            })
            if ind.parent_a is not None:
                lineage["edges"].append({"src": ind.parent_a, "dst": ind.name})
            if ind.parent_b is not None:
                lineage["edges"].append({"src": ind.parent_b, "dst": ind.name})
            lineage_logger.log(lineage["nodes"][-1])

        evaluated.sort(key=lambda x: x.score, reverse=True)
        save_json(lineage, root_out_dir / "lineage_graph.json")
        if evaluated[0].score > best_score:
            best_score = evaluated[0].score
            best_ind = evaluated[0]
            stale = 0
        else:
            stale += 1
        if stale >= cfg.patience:
            break

        elites = evaluated[: cfg.elite_size]
        next_population: List[Individual] = [Individual(name=f"g{gen+1}_{e.name}_elite", generation=gen + 1, parent_a=e.name, parent_b=None, ckpt=e.ckpt) for e in elites]
        while len(next_population) < cfg.population_size:
            parents = evaluated[: max(2, len(evaluated) // 2)]
            pa, pb = evolution_rng.sample(parents, 2)
            child = Individual(name=f"g{gen+1}_child{len(next_population)}", generation=gen + 1, parent_a=pa.name, parent_b=pb.name, ckpt=None)
            child_seed = derive_seed(cfg.seed, "evolution-child", gen + 1, child.name)
            seed_everything(child_seed)
            ma = build_model(cfg)
            sa = model_state_dict(load_checkpoint(pa.ckpt, map_location="cpu"))
            sb = model_state_dict(load_checkpoint(pb.ckpt, map_location="cpu"))
            state = mutate_state_dict(
                crossover_state_dict(sa, sb, alpha=evolution_rng.uniform(0.35, 0.65)),
                scale=0.02,
            )
            ma.load_state_dict(state, strict=True)
            child_dir = root_out_dir / "init_children"
            child_dir.mkdir(parents=True, exist_ok=True)
            ckpt = child_dir / f"{child.name}.pt"
            save_checkpoint_payload(ckpt, {"model": ma.state_dict(), "cfg": cfg.to_dict(), "epoch": 0})
            child.ckpt = ckpt
            next_population.append(child)
        population = next_population

    return best_ind if best_ind is not None else max(population, key=lambda x: x.score)
