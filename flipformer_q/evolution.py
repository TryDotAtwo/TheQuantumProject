from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from .config import Config
from .data import write_sharded_dataset
from .model import FlipformerModel
from .quantum.aer_emulator import QuantumFeatureEmulator
from .reporting import make_compare_plots, write_summary_csv
from .search import beam_search
from .train import train_model
from .utils.io_utils import save_json
from .utils.logging_utils import JsonlLogger, ProgressPrinter


@dataclass
class Individual:
    name: str
    generation: int
    parent_a: Optional[str]
    parent_b: Optional[str]
    ckpt: Optional[Path]
    score: float = -1e18
    solved_rate: float = 0.0
    mean_length: float = 0.0
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
    out = {}
    for k, v in state_dict.items():
        out[k] = v.clone() if not torch.is_floating_point(v) else v + scale * torch.randn_like(v)
    return out


def crossover_state_dict(a: Dict[str, torch.Tensor], b: Dict[str, torch.Tensor], alpha: float = 0.5) -> Dict[str, torch.Tensor]:
    out = {}
    for k in a.keys():
        va, vb = a[k], b[k]
        out[k] = alpha * va + (1.0 - alpha) * vb if torch.is_floating_point(va) and torch.is_floating_point(vb) else va.clone()
    return out


def evaluate_individual(cfg: Config, ckpt: Path, out_dir: Path, mode: str = "classical") -> Dict[str, float]:
    device = torch.device(cfg.device if torch.cuda.is_available() and cfg.device == "cuda" else "cpu")
    model = build_model(cfg).to(device)
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state["model"], strict=False)
    model.eval()

    rows = []
    rng = random.Random(cfg.seed + 999)
    quantum = None
    if mode == "hybrid":
        quantum = QuantumFeatureEmulator(
            n_qubits=cfg.quantum_qubits,
            n_layers=cfg.quantum_layers,
            mode=cfg.quantum_mode,
            cache_size=cfg.quantum_cache_size,
        )

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
        rows.append({"trial": t, "mode": mode, "solved": 1 if res.solved else 0, "path_len": len(res.path), "nodes_expanded": res.nodes_expanded, "q_calls": res.q_calls})
        prog.update(t + 1, extra=f"solved={int(res.solved)} | path_len={len(res.path)} | nodes={res.nodes_expanded}")
    write_summary_csv(rows, out_dir / f"trials_{mode}.csv")
    solved_rate = float(np.mean([r["solved"] for r in rows]))
    mean_length = float(np.mean([r["path_len"] for r in rows]))
    mean_nodes = float(np.mean([r["nodes_expanded"] for r in rows]))
    score = 5.0 * solved_rate - 0.02 * mean_length - 0.0002 * mean_nodes
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {"score": score, "solved_rate": solved_rate, "mean_length": mean_length, "mean_nodes": mean_nodes}


def evolutionary_self_training(cfg: Config, root_out_dir: Path):
    root_out_dir = Path(root_out_dir)
    root_out_dir.mkdir(parents=True, exist_ok=True)
    lineage_logger = JsonlLogger(root_out_dir / "lineage.jsonl")
    lineage = {"nodes": [], "edges": []}
    device = torch.device(cfg.device if torch.cuda.is_available() and cfg.device == "cuda" else "cpu")

    population: List[Individual] = [Individual(name=f"g0_m{i}", generation=0, parent_a=None, parent_b=None, ckpt=None) for i in range(cfg.population_size)]
    best_ind: Optional[Individual] = None
    best_score = -1e18
    stale = 0

    for gen in range(cfg.generations):
        evaluated: List[Individual] = []
        for ind in population:
            stage_cfg = cfg.with_stage("seed" if gen == 0 and ind.ckpt is None else "selftrain")
            model = build_model(stage_cfg)
            if ind.ckpt is not None and Path(ind.ckpt).exists():
                state = torch.load(ind.ckpt, map_location="cpu")
                model.load_state_dict(state["model"], strict=False)
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
            ind.mean_nodes = metrics["mean_nodes"]
            evaluated.append(ind)
            lineage["nodes"].append({"name": ind.name, "generation": gen, "score": ind.score, "solved_rate": ind.solved_rate, "mean_length": ind.mean_length, "mean_nodes": ind.mean_nodes})
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
            pa, pb = random.sample(evaluated[: max(2, len(evaluated) // 2)], 2)
            child = Individual(name=f"g{gen+1}_child{len(next_population)}", generation=gen + 1, parent_a=pa.name, parent_b=pb.name, ckpt=None)
            ma = build_model(cfg)
            sa = torch.load(pa.ckpt, map_location="cpu")["model"]
            sb = torch.load(pb.ckpt, map_location="cpu")["model"]
            state = mutate_state_dict(crossover_state_dict(sa, sb, alpha=random.uniform(0.35, 0.65)), scale=0.02)
            ma.load_state_dict(state, strict=False)
            child_dir = root_out_dir / "init_children"
            child_dir.mkdir(parents=True, exist_ok=True)
            ckpt = child_dir / f"{child.name}.pt"
            torch.save({"model": ma.state_dict(), "cfg": cfg.to_dict(), "epoch": 0}, ckpt)
            child.ckpt = ckpt
            next_population.append(child)
        population = next_population

    return best_ind if best_ind is not None else max(population, key=lambda x: x.score)
