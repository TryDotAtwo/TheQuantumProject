from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np


def write_summary_csv(rows: List[Dict], path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def make_compare_plots(rows: List[Dict], out_dir: Path):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    modes = sorted({r["mode"] for r in rows})
    for metric in ["path_len", "nodes_expanded", "q_calls", "solved"]:
        plt.figure(figsize=(8, 5))
        data = [np.asarray([float(r[metric]) for r in rows if r["mode"] == m], dtype=np.float32) for m in modes]
        plt.boxplot(data, labels=modes, showmeans=True)
        plt.title(metric)
        plt.tight_layout()
        plt.savefig(out_dir / f"{metric}_box.png", dpi=160)
        plt.close()

    plt.figure(figsize=(8, 5))
    summary = []
    for m in modes:
        vals = np.asarray([float(r["solved"]) for r in rows if r["mode"] == m], dtype=np.float32)
        summary.append(vals.mean() if len(vals) else 0.0)
    plt.bar(modes, summary)
    plt.title("Solved rate")
    plt.tight_layout()
    plt.savefig(out_dir / "solved_rate_bar.png", dpi=160)
    plt.close()
