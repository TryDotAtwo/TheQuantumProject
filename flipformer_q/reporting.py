from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List


def write_summary_csv(rows: List[Dict], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        if keys:
            writer.writeheader()
            writer.writerows(rows)


def make_compare_plots(rows: List[Dict], out_dir: Path) -> None:
    # Plotting is only needed by `compare`; keep matplotlib out of train/eval imports.
    import matplotlib.pyplot as plt
    import numpy as np

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    modes = sorted({str(row["mode"]) for row in rows})
    metrics = ("solution_length", "partial_depth", "nodes_expanded", "q_calls", "solved")

    for metric in metrics:
        labels = []
        data = []
        for mode in modes:
            values = []
            for row in rows:
                if str(row["mode"]) != mode:
                    continue
                raw_value = row.get(metric, "")
                if raw_value in (None, ""):
                    continue
                value = float(raw_value)
                if np.isfinite(value):
                    values.append(value)
            if values:
                labels.append(mode)
                data.append(np.asarray(values, dtype=np.float32))
        if not data:
            continue
        plt.figure(figsize=(8, 5))
        plt.boxplot(data, labels=labels, showmeans=True)
        plt.title(metric)
        plt.tight_layout()
        plt.savefig(out_dir / f"{metric}_box.png", dpi=160)
        plt.close()

    rates = []
    for mode in modes:
        values = [float(row["solved"]) for row in rows if str(row["mode"]) == mode and row.get("solved") not in (None, "")]
        rates.append(float(np.mean(values)) if values else 0.0)
    plt.figure(figsize=(8, 5))
    plt.bar(modes, rates)
    plt.title("Solved rate")
    plt.tight_layout()
    plt.savefig(out_dir / "solved_rate_bar.png", dpi=160)
    plt.close()
