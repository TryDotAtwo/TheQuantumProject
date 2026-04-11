from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from flipformer_q.config import Config
from flipformer_q.data import write_sharded_dataset
from flipformer_q.evolution import build_model, evolutionary_self_training, evaluate_individual
from flipformer_q.pipeline import iterative_self_training, run_stage
from flipformer_q.reporting import make_compare_plots
from flipformer_q.train import train_model


def add_common_config(parser):
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--train-samples", type=int, default=100000)
    parser.add_argument("--val-samples", type=int, default=5000)
    parser.add_argument("--benchmark-samples", type=int, default=200)
    parser.add_argument("--shard-size", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=123456)

    parser.add_argument("--scramble-min", type=int, default=10)
    parser.add_argument("--scramble-max", type=int, default=80)

    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--d-pair", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--n-blocks", type=int, default=3)
    parser.add_argument("--recycle-steps", type=int, default=2)
    parser.add_argument("--mlp-mult", type=int, default=8)
    parser.add_argument("--head-mlp-mult", type=int, default=6)

    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--micro-batch-size", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num-workers", type=int, default=4)

    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--beam-width", type=int, default=48)
    parser.add_argument("--beam-depth", type=int, default=96)
    parser.add_argument("--expand-topk", type=int, default=12)
    parser.add_argument("--rerank-topk", type=int, default=4)

    parser.add_argument("--quantum-weight", type=float, default=0.25)
    parser.add_argument("--uncertainty-margin", type=float, default=0.10)
    parser.add_argument("--quantum-mode", type=str, choices=["none", "aer_cpu", "aer_gpu"], default="none")
    parser.add_argument("--quantum-qubits", type=int, default=4)
    parser.add_argument("--quantum-layers", type=int, default=2)

    parser.add_argument("--out-dir", type=str, default="outputs")

    parser.add_argument("--selftrain-rounds", type=int, default=3)
    parser.add_argument("--patience", type=int, default=2)

    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--no-amp", action="store_true")


def args_to_config(args) -> Config:
    return Config(
        n=args.n,
        train_samples=args.train_samples,
        val_samples=args.val_samples,
        benchmark_samples=args.benchmark_samples,
        shard_size=args.shard_size,
        seed=args.seed,
        scramble_min=args.scramble_min,
        scramble_max=args.scramble_max,
        d_model=args.d_model,
        d_pair=args.d_pair,
        n_heads=args.n_heads,
        n_blocks=args.n_blocks,
        recycle_steps=args.recycle_steps,
        mlp_mult=args.mlp_mult,
        head_mlp_mult=args.head_mlp_mult,
        batch_size=args.batch_size,
        micro_batch_size=args.micro_batch_size,
        epochs=args.epochs,
        lr=args.lr,
        device=args.device,
        beam_width=args.beam_width,
        beam_depth=args.beam_depth,
        expand_topk=args.expand_topk,
        rerank_topk=args.rerank_topk,
        quantum_weight=args.quantum_weight,
        uncertainty_margin=args.uncertainty_margin,
        quantum_mode=args.quantum_mode,
        quantum_qubits=args.quantum_qubits,
        quantum_layers=args.quantum_layers,
        out_dir=args.out_dir,
        selftrain_rounds=args.selftrain_rounds,
        patience=args.patience,
        log_every=args.log_every,
        amp=(not args.no_amp),
        num_workers=args.num_workers,
    ).validate()


def pick_device(cfg: Config) -> torch.device:
    if cfg.device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_ckpt_into_model(cfg: Config, ckpt_path: str | Path):
    device = pick_device(cfg)
    model = build_model(cfg).to(device)
    payload = torch.load(ckpt_path, map_location=device)
    state_dict = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model, device


def command_generate(args):
    cfg = args_to_config(args)

    model = None
    device = pick_device(cfg)

    if args.teacher_ckpt:
        model, device = load_ckpt_into_model(cfg, args.teacher_ckpt)

    dataset_mode = args.dataset_mode
    if dataset_mode is None:
        dataset_mode = "selftrain" if model is not None else "bootstrap"

    manifest = write_sharded_dataset(
        cfg=cfg,
        out_dir=Path(args.dataset_dir),
        total_samples=cfg.train_samples,
        shard_size=cfg.shard_size,
        model=model,
        device=device,
        dataset_mode=dataset_mode,
    )

    print(
        json.dumps(
            {
                "manifest": str(manifest),
                "dataset_mode": dataset_mode,
            }
        ),
        flush=True,
    )


def command_train(args):
    cfg = args_to_config(args)
    model = build_model(cfg)
    final_ckpt = train_model(
        cfg,
        model,
        Path(args.manifest),
        Path(args.run_dir),
        init_ckpt=Path(args.init_ckpt) if args.init_ckpt else None,
    )
    print(json.dumps({"final_ckpt": str(final_ckpt)}), flush=True)


def command_eval(args):
    cfg = args_to_config(args)
    out_dir = Path(args.run_dir) / "eval"
    metrics = evaluate_individual(cfg, Path(args.ckpt), out_dir, mode=args.mode)
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


def command_compare(args):
    cfg = args_to_config(args)
    out_dir = Path(args.run_dir) / "compare"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    import csv

    for mode, ckpt in [
        ("classical", args.cls_ckpt),
        ("hybrid", args.hybrid_ckpt or args.cls_ckpt),
    ]:
        metrics = evaluate_individual(cfg, Path(ckpt), out_dir, mode=mode)
        rows.append({"mode": mode, **metrics})

    all_rows = []
    for mode in ["classical", "hybrid"]:
        path = out_dir / f"trials_{mode}.csv"
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                all_rows.extend(list(csv.DictReader(f)))

    if all_rows:
        make_compare_plots(all_rows, out_dir)

    print(json.dumps(rows, ensure_ascii=False, indent=2), flush=True)


def command_evolve(args):
    cfg = args_to_config(args)
    cfg.population_size = args.population_size
    cfg.elite_size = args.elite_size
    cfg.generations = args.generations
    cfg.patience = args.patience

    best = evolutionary_self_training(cfg, Path(args.run_dir))
    print(
        json.dumps(
            {
                "best_model": str(best.ckpt),
                "score": best.score,
                "name": best.name,
            }
        ),
        flush=True,
    )


def command_stage_run(args):
    cfg = args_to_config(args)
    result = run_stage(
        cfg,
        stage=args.stage,
        root_dir=Path(args.run_dir),
        teacher_ckpt=Path(args.teacher_ckpt) if args.teacher_ckpt else None,
        round_idx=args.round_idx,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


def command_iterative(args):
    cfg = args_to_config(args)
    summary = iterative_self_training(cfg, Path(args.run_dir))
    print(json.dumps(summary["best"], ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(conflict_handler="resolve")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("generate")
    add_common_config(p)
    p.add_argument("--dataset-dir", type=str, required=True)
    p.add_argument("--teacher-ckpt", type=str, default=None)
    p.add_argument(
        "--dataset-mode",
        type=str,
        choices=["bootstrap", "selftrain", "distill", "stage3"],
        default=None,
    )
    p.set_defaults(func=command_generate)

    p = sub.add_parser("train")
    add_common_config(p)
    p.add_argument("--manifest", type=str, required=True)
    p.add_argument("--run-dir", type=str, required=True)
    p.add_argument("--init-ckpt", type=str, default=None)
    p.set_defaults(func=command_train)

    p = sub.add_parser("eval")
    add_common_config(p)
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--run-dir", type=str, required=True)
    p.add_argument("--mode", type=str, choices=["classical", "hybrid"], default="classical")
    p.set_defaults(func=command_eval)

    p = sub.add_parser("compare")
    add_common_config(p)
    p.add_argument("--run-dir", type=str, required=True)
    p.add_argument("--cls-ckpt", type=str, required=True)
    p.add_argument("--hybrid-ckpt", type=str, default=None)
    p.set_defaults(func=command_compare)

    p = sub.add_parser("evolve")
    add_common_config(p)
    p.add_argument("--run-dir", type=str, required=True)
    p.add_argument("--population-size", type=int, default=8)
    p.add_argument("--elite-size", type=int, default=2)
    p.add_argument("--generations", type=int, default=5)
    # НЕ добавлять --patience повторно: уже есть в add_common_config
    p.set_defaults(func=command_evolve)

    p = sub.add_parser("stage-run")
    add_common_config(p)
    p.add_argument("--run-dir", type=str, required=True)
    p.add_argument("--stage", type=str, choices=["seed", "selftrain", "distill", "quantum"], required=True)
    p.add_argument("--teacher-ckpt", type=str, default=None)
    p.add_argument("--round-idx", type=int, default=0)
    p.set_defaults(func=command_stage_run)

    p = sub.add_parser("iterative")
    add_common_config(p)
    p.add_argument("--run-dir", type=str, required=True)
    p.set_defaults(func=command_iterative)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
