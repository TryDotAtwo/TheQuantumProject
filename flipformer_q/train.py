from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import Config
from .data import ShardedNpzDataset
from .utils.logging_utils import ProgressPrinter, CsvLogger


LOSS_KEYS = [
    "loss",
    "value_loss",
    "policy_loss",
    "q_loss",
    "far_loss",
    "rank_loss",
    "policy_entropy",
    "far_entropy",
]


def collate_fn(batch):
    if isinstance(batch, dict):
        return batch
    out = {}
    for key in batch[0].keys():
        out[key] = torch.stack([x[key] for x in batch], dim=0)
    return out


def move_batch(batch, device: torch.device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def dataset_mode_loss_weights(cfg: Config, dataset_mode: str) -> tuple[float, float, float, float, float]:
    if dataset_mode == "bootstrap":
        return 1.0, 0.0, 1.0, 0.0, 0.0
    if dataset_mode == "selftrain":
        return 1.0, 0.25, 0.75, 0.0, 0.10
    if dataset_mode == "distill":
        return 1.0, 0.75, 1.0, 0.35, 0.20
    if dataset_mode == "stage3":
        return 1.0, 1.0, 1.0, 0.50, 0.25
    return cfg.lambda_value, cfg.lambda_policy, cfg.lambda_q, cfg.lambda_far, cfg.lambda_rank


def compute_losses(outputs, batch, weights):
    w_value, w_policy, w_q, w_far, w_rank = weights
    zero = torch.zeros((), device=outputs["value"].device, dtype=outputs["value"].dtype)
    value_loss = F.huber_loss(outputs["value"], batch["value"])

    policy_target = batch["policy"]
    policy_loss = zero
    rank_loss = zero
    if w_policy != 0.0 or w_rank != 0.0:
        policy_log_probs = torch.log_softmax(outputs["policy_logits"], dim=-1)
        if w_policy != 0.0:
            policy_loss = -(policy_target * policy_log_probs).sum(dim=-1).mean()

    q_loss = F.huber_loss(outputs["q_values"], batch["q_values"]) if w_q != 0.0 else zero

    far_target = batch["far"]
    far_loss = zero
    if w_far != 0.0:
        far_log_probs = torch.log_softmax(outputs["far_logits"], dim=-1)
        far_loss = -(far_target * far_log_probs).sum(dim=-1).mean()

    if w_rank != 0.0:
        with torch.no_grad():
            pos_mask = policy_target > 0.0
            neg_mask = policy_target <= 0.0
        scores = outputs["policy_logits"]
        ranking_terms = []
        for b in range(scores.shape[0]):
            pos = scores[b][pos_mask[b]]
            neg = scores[b][neg_mask[b]]
            if pos.numel() == 0 or neg.numel() == 0:
                continue
            pos = pos[: min(4, pos.numel())]
            neg = neg[: min(8, neg.numel())]
            ranking_terms.append(F.relu(1.0 - pos[:, None] + neg[None, :]).mean())
        rank_loss = torch.stack(ranking_terms).mean() if ranking_terms else zero

    total = w_value * value_loss + w_policy * policy_loss + w_q * q_loss + w_far * far_loss + w_rank * rank_loss

    with torch.no_grad():
        policy_entropy = -(policy_target * torch.log(policy_target.clamp_min(1e-8))).sum(dim=-1).mean()
        far_entropy = -(far_target * torch.log(far_target.clamp_min(1e-8))).sum(dim=-1).mean()

    return {
        "total": total,
        "value_loss": value_loss,
        "policy_loss": policy_loss,
        "q_loss": q_loss,
        "far_loss": far_loss,
        "rank_loss": rank_loss,
        "policy_entropy": policy_entropy,
        "far_entropy": far_entropy,
    }


def heads_for_loss_weights(weights) -> set[str]:
    _, w_policy, w_q, w_far, w_rank = weights
    heads = {"value"}
    if w_policy != 0.0 or w_rank != 0.0:
        heads.add("policy")
    if w_q != 0.0:
        heads.add("q")
    if w_far != 0.0:
        heads.add("far")
    return heads


def batch_len(batch) -> int:
    for value in batch.values():
        return int(value.shape[0])
    return 0


def resolve_micro_batch_size(cfg: Config, device: torch.device) -> int:
    if cfg.micro_batch_size > 0:
        return min(cfg.batch_size, cfg.micro_batch_size)
    if device.type != "cuda":
        return cfg.batch_size

    cap = 128
    if cfg.d_model >= 128 or cfg.d_pair >= 64 or cfg.n_blocks >= 3 or cfg.mlp_mult >= 8:
        cap = 64
    if cfg.n >= 50:
        cap = min(cap, 32)
    return min(cfg.batch_size, cap)


def iter_micro_batches(batch, micro_batch_size: int):
    size = batch_len(batch)
    if micro_batch_size >= size:
        yield batch, size
        return
    for start in range(0, size, micro_batch_size):
        end = min(size, start + micro_batch_size)
        yield {key: value[start:end] for key, value in batch.items()}, end - start


def train_model(cfg: Config, model, manifest_path: Path, out_dir: Path, init_ckpt: Optional[Path] = None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(manifest_path)
    device = torch.device(cfg.device if torch.cuda.is_available() and cfg.device == "cuda" else "cpu")
    model = model.to(device)
    if init_ckpt is not None and Path(init_ckpt).exists():
        payload = torch.load(init_ckpt, map_location=device)
        model.load_state_dict(payload["model"], strict=False)

    ds = ShardedNpzDataset(manifest_path)
    manifest_n = ds.manifest.get("n")
    if manifest_n is not None and int(manifest_n) != cfg.n:
        raise ValueError(f"Manifest n={manifest_n} does not match config n={cfg.n}: {manifest_path}")
    dataset_mode = ds.manifest.get("dataset_mode", "bootstrap")
    weights = dataset_mode_loss_weights(cfg, dataset_mode)
    required_heads = heads_for_loss_weights(weights)
    micro_batch_size = resolve_micro_batch_size(cfg, device)
    grad_accum_steps = max(1, (cfg.batch_size + micro_batch_size - 1) // micro_batch_size)

    dl = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(cfg.num_workers > 0),
        prefetch_factor=2 if cfg.num_workers > 0 else None,
        drop_last=False,
    )

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=(cfg.amp and device.type == "cuda"))
    csv = CsvLogger(
        out_dir / "train_metrics.csv",
        [
            "epoch", "step", "loss", "value_loss", "policy_loss", "q_loss", "far_loss", "rank_loss",
            "policy_entropy", "far_entropy",
        ],
    )

    print(json.dumps({
        "stage": "train_start",
        "manifest": str(manifest_path),
        "out_dir": str(out_dir),
        "dataset_size": len(ds),
        "batch_size": cfg.batch_size,
        "micro_batch_size": micro_batch_size,
        "micro_batch_auto": cfg.micro_batch_size <= 0,
        "grad_accum_steps_estimate": grad_accum_steps,
        "total_batches": len(dl),
        "epochs": cfg.epochs,
        "device": str(device),
        "num_workers": cfg.num_workers,
        "lr": cfg.lr,
        "weight_decay": cfg.weight_decay,
        "dataset_mode": dataset_mode,
        "loss_weights": {
            "value": weights[0], "policy": weights[1], "q": weights[2], "far": weights[3], "rank": weights[4]
        },
        "required_heads": sorted(required_heads),
    }), flush=True)

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        epoch_start = time.time()
        prog = ProgressPrinter(f"train epoch {epoch}/{cfg.epochs}", total=max(1, len(dl)), every_sec=2.0)
        running = {k: 0.0 for k in LOSS_KEYS}
        steps = 0

        for step, batch in enumerate(dl, start=1):
            full_batch_size = batch_len(batch)
            if full_batch_size <= 0:
                continue
            row_sums = {k: 0.0 for k in LOSS_KEYS}
            opt.zero_grad(set_to_none=True)
            for micro_batch, current_micro_batch_size in iter_micro_batches(batch, micro_batch_size):
                micro_batch = move_batch(micro_batch, device)
                loss_scale = current_micro_batch_size / full_batch_size
                with torch.amp.autocast(device_type=device.type, enabled=(cfg.amp and device.type == "cuda")):
                    out = model(micro_batch["single"], micro_batch["pair"], micro_batch["action_feats"], heads=required_heads)
                    losses = compute_losses(out, micro_batch, weights)
                scaler.scale(losses["total"] * loss_scale).backward()
                for key in LOSS_KEYS:
                    source_key = "total" if key == "loss" else key
                    row_sums[key] += float(losses[source_key].detach().item()) * loss_scale
                del micro_batch, out, losses
            if cfg.clip_grad_norm is not None and cfg.clip_grad_norm > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad_norm)
            scaler.step(opt)
            scaler.update()

            row = {
                "epoch": epoch,
                "step": step,
                **row_sums,
            }
            csv.log(row)
            for k, v in row.items():
                if k in running:
                    running[k] += float(v)
            steps += 1

            if step == 1 or step % max(1, cfg.log_every) == 0 or step == len(dl):
                prog.update(
                    step,
                    extra=(
                        f"batch_loss={row['loss']:.5f} | value={row['value_loss']:.5f} | policy={row['policy_loss']:.5f} | "
                        f"q={row['q_loss']:.5f} | far={row['far_loss']:.5f} | rank={row['rank_loss']:.5f} | "
                        f"Hpi={row['policy_entropy']:.5f} | Hfar={row['far_entropy']:.5f}"
                    ),
                )

        summary = {k: v / max(1, steps) for k, v in running.items()}
        summary["epoch"] = epoch
        summary["seconds"] = time.time() - epoch_start
        print(json.dumps(summary), flush=True)
        payload = {"model": model.state_dict(), "cfg": cfg.to_dict(), "epoch": epoch, "summary": summary}
        torch.save(payload, out_dir / f"epoch_{epoch:03d}.pt")
        torch.save(payload, out_dir / "model_last.pt")

    return out_dir / "model_last.pt"
