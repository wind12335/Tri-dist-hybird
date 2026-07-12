#!/usr/bin/env python3
"""Plot search-cost vs search-quality comparison for RS-GEMM."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    return parser.parse_args()


def load_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def shape_label(row: dict[str, str]) -> str:
    return f"{row['M']}x{row['N']}x{row['K']}"


def main() -> None:
    args = parse_args()
    csv_path = Path(args.input_csv)
    rows = load_rows(csv_path)
    if not rows:
        raise RuntimeError("input CSV is empty")

    output_dir = Path(args.output_dir) if args.output_dir else csv_path.parent / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = [shape_label(row) for row in rows]
    x = np.arange(len(labels), dtype=float)
    width = 0.28

    exhaustive_runs = np.array([float(row["exhaustive_total_search_runs"]) for row in rows], dtype=float)
    two_stage_runs = np.array([float(row["two_stage_total_search_runs"]) for row in rows], dtype=float)
    exhaustive_best = np.ones(len(rows), dtype=float)
    two_stage_best = np.array([float(row["two_stage_latency_ratio_vs_exhaustive"]) for row in rows], dtype=float)
    two_stage_time_ratio = np.array([float(row["two_stage_time_ratio_vs_exhaustive"]) for row in rows], dtype=float)

    plt.rcParams.update({
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    colors = {
        "exhaustive": "#F18F01",
        "two_stage": "#2E86AB",
    }

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12.2, 4.8), constrained_layout=True)

    ax0.bar(x - width / 2, exhaustive_runs, width=width, color=colors["exhaustive"], label="Exhaustive")
    ax0.bar(x + width / 2, two_stage_runs, width=width, color=colors["two_stage"], label="Constraint-aware")
    ax0.set_ylabel("Evaluated runs")
    ax0.set_xticks(x)
    ax0.set_xticklabels(labels, rotation=0)
    ax0.legend(frameon=False, loc="upper left")
    ax0.set_ylim(0.0, max(np.max(exhaustive_runs), np.max(two_stage_runs)) * 1.22)

    for idx, val in enumerate(exhaustive_runs):
        ax0.text(x[idx] - width / 2,
                 val + max(exhaustive_runs[idx], two_stage_runs[idx]) * 0.03,
                 f"{int(val)}",
                 ha="center",
                 va="bottom",
                 fontsize=9,
                 color="#8a4d00")
    for idx, val in enumerate(two_stage_runs):
        ax0.text(x[idx] + width / 2,
                 val + max(exhaustive_runs[idx], two_stage_runs[idx]) * 0.03,
                 f"{int(val)}\n({val / max(exhaustive_runs[idx], 1e-12):.2f}x)",
                 ha="center",
                 va="bottom",
                 fontsize=9,
                 color="#1f4f63")
        ax0.text(x[idx],
                 max(exhaustive_runs[idx], two_stage_runs[idx]) * 0.86,
                 f"time {two_stage_time_ratio[idx]:.2f}x",
                 ha="center",
                 va="bottom",
                 fontsize=8.5,
                 color="#444444")

    ax1.bar(x - width / 2, exhaustive_best, width=width, color=colors["exhaustive"], label="Exhaustive")
    ax1.bar(x + width / 2, two_stage_best, width=width, color=colors["two_stage"], label="Constraint-aware")
    ax1.axhline(1.0, color="#444444", linewidth=1.0, linestyle="--")
    ax1.set_ylabel("Best verified latency (normalized)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=0)
    ax1.legend(frameon=False, loc="upper left")
    ax1.set_ylim(0.0, max(np.max(two_stage_best), 1.0) * 1.16)

    for idx, val in enumerate(two_stage_best):
        ax1.text(x[idx] + width / 2,
                 val * 1.02,
                 f"{val:.4f}x",
                 ha="center",
                 va="bottom",
                 fontsize=9,
                 color="#1f4f63")

    png_path = output_dir / "gemmrs_search_compare.png"
    svg_path = output_dir / "gemmrs_search_compare.svg"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)

    print(f"[png] {png_path}")
    print(f"[svg] {svg_path}")


if __name__ == "__main__":
    main()
