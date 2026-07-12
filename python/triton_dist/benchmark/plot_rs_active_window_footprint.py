#!/usr/bin/env python3
"""Plot RS active-window symmetric-memory footprint summaries."""

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
    parser.add_argument("--log_scale", action="store_true", default=False)
    return parser.parse_args()


def load_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def shape_label(row: dict[str, str]) -> str:
    return f"{row['M']}x{row['N']}x{row['K']}"


def main() -> None:
    args = parse_args()
    csv_path = Path(args.input_csv)
    output_dir = Path(args.output_dir) if args.output_dir else csv_path.parent / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(csv_path)
    if not rows:
        raise RuntimeError("input CSV is empty")

    labels = [shape_label(row) for row in rows]
    windowed = np.array([float(row["windowed_total_gib"]) for row in rows], dtype=float)
    no_reuse = np.array([float(row["logical_no_reuse_total_gib"]) for row in rows], dtype=float)
    legacy = np.array([float(row["legacy_total_gib"]) for row in rows], dtype=float)
    reuse_factor = np.array([float(row["logical_no_reuse_factor_vs_windowed"]) for row in rows], dtype=float)
    legacy_factor = np.array([float(row["legacy_factor_vs_windowed"]) for row in rows], dtype=float)
    num_chunks = [int(row["num_chunks"]) for row in rows]
    active_window = [int(row["effective_active_chunk_window"]) for row in rows]

    plt.rcParams.update({
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    fig, (ax0, ax1) = plt.subplots(
        1,
        2,
        figsize=(12.8, 4.8),
        gridspec_kw={"width_ratios": [1.45, 1.0]},
        constrained_layout=True,
    )

    x = np.arange(len(labels), dtype=float)
    width = 0.24
    colors = {
        "windowed": "#2E86AB",
        "no_reuse": "#F18F01",
        "legacy": "#7A7A7A",
    }

    ax0.bar(x - width, no_reuse, width=width, color=colors["no_reuse"], label="Without slot reuse")
    ax0.bar(x, windowed, width=width, color=colors["windowed"], label="With active window")
    ax0.bar(x + width, legacy, width=width, color=colors["legacy"], label="Legacy old path")
    if args.log_scale:
        ax0.set_yscale("log")
    else:
        ax0.set_ylim(0.0, max(np.max(no_reuse), np.max(windowed), np.max(legacy)) * 1.28)
    ax0.set_ylabel("Total symmetric allocation (GiB)")
    ax0.set_xticks(x)
    ax0.set_xticklabels(labels, rotation=0)
    ax0.legend(frameon=False, loc="upper left")

    for idx, row in enumerate(rows):
        note = f"chunks={num_chunks[idx]}, window={active_window[idx]}"
        ax0.text(
            x[idx],
            max(no_reuse[idx], windowed[idx], legacy[idx]) * (1.06 if not args.log_scale else 1.18),
            note,
            ha="center",
            va="bottom",
            fontsize=9,
            color="#444444",
        )

    ratio_width = 0.32
    ax1.bar(
        x - ratio_width / 2,
        reuse_factor,
        width=ratio_width,
        color=colors["no_reuse"],
        label="Without slot reuse / with active window",
    )
    ax1.bar(
        x + ratio_width / 2,
        legacy_factor,
        width=ratio_width,
        color=colors["legacy"],
        label="Legacy old path / with active window",
    )
    ax1.axhline(1.0, color="#444444", linewidth=1.0, linestyle="--")
    ax1.set_ylabel("Footprint ratio vs windowed")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=0)
    ax1.legend(frameon=False, loc="upper left")
    ax1.set_ylim(0.0, max(np.max(reuse_factor), np.max(legacy_factor)) * 1.18)

    for idx, factor in enumerate(reuse_factor):
        ax1.text(
            x[idx] - ratio_width / 2,
            factor * 1.02,
            f"{factor:.2f}x",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#7A4B00",
        )
    for idx, factor in enumerate(legacy_factor):
        ax1.text(
            x[idx] + ratio_width / 2,
            factor * 1.02,
            f"{factor:.2f}x",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#444444",
        )

    png_path = output_dir / "rs_active_window_footprint_summary.png"
    svg_path = output_dir / "rs_active_window_footprint_summary.svg"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)

    print(f"[png] {png_path}")
    print(f"[svg] {svg_path}")


if __name__ == "__main__":
    main()
