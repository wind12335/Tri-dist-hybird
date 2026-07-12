#!/usr/bin/env python3
"""Plot the active-window ablation summary for RS-GEMM."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="csv")
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

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = [shape_label(row) for row in rows]
    x = np.arange(len(labels), dtype=float)
    width = 0.28

    baseline_latency = np.ones(len(rows), dtype=float)
    with_window_latency = np.array([float(row["window_latency_ratio_vs_unbounded"]) for row in rows], dtype=float)
    baseline_symm = np.array([float(row["unbounded_symmetric_staging_gib"]) for row in rows], dtype=float)
    with_window_symm = np.array([float(row["windowed_symmetric_staging_gib"]) for row in rows], dtype=float)
    with_window_symm_ratio = np.array([float(row["window_symmetric_ratio_vs_unbounded"]) for row in rows], dtype=float)

    plt.rcParams.update({
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12.2, 4.8), constrained_layout=True)
    colors = {
        "windowed": "#2E86AB",
        "baseline": "#F18F01",
    }

    ax0.bar(x - width / 2, baseline_latency, width=width, color=colors["baseline"], label="Without active window")
    ax0.bar(x + width / 2, with_window_latency, width=width, color=colors["windowed"], label="With active window")
    ax0.axhline(1.0, color="#444444", linewidth=1.0, linestyle="--")
    ax0.set_ylabel("Normalized latency")
    ax0.set_xticks(x)
    ax0.set_xticklabels(labels, rotation=0)
    ax0.legend(frameon=False, loc="upper left")
    ax0.set_ylim(0.0, max(np.max(with_window_latency), 1.0) * 1.18)

    ax1.bar(x - width / 2, baseline_symm, width=width, color=colors["baseline"], label="Without active window")
    ax1.bar(x + width / 2, with_window_symm, width=width, color=colors["windowed"], label="With active window")
    ax1.set_ylabel("Symmetric memory (GiB)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=0)
    ax1.legend(frameon=False, loc="upper left")
    ax1.set_ylim(0.0, max(np.max(baseline_symm), np.max(with_window_symm)) * 1.28)

    for idx, val in enumerate(with_window_latency):
        ax0.text(
            x[idx] + width / 2,
            val * 1.02,
            f"{val:.3f}x",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#1f4f63",
        )

    for idx, (abs_val, ratio_val) in enumerate(zip(with_window_symm, with_window_symm_ratio)):
        ax1.text(
            x[idx] + width / 2,
            abs_val + max(baseline_symm[idx], with_window_symm[idx]) * 0.03,
            f"{abs_val:.3f} GiB\n({ratio_val:.2f}x)",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#1f4f63",
        )

    for idx, abs_val in enumerate(baseline_symm):
        ax1.text(
            x[idx] - width / 2,
            abs_val + max(baseline_symm[idx], with_window_symm[idx]) * 0.03,
            f"{abs_val:.3f} GiB",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#8a4d00",
        )

    png_path = output_dir / "active_window_ablation.png"
    svg_path = output_dir / "active_window_ablation.svg"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)

    print(f"[png] {png_path}")
    print(f"[svg] {svg_path}")


if __name__ == "__main__":
    main()
