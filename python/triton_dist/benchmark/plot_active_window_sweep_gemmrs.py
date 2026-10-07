#!/usr/bin/env python3
"""Plot a repeated GEMM--RS active-window sweep from its aggregate CSV.

The input is produced by ``bench_active_window_sweep_driver.py``.  Missing,
failed, and partial points are not fabricated: a point is drawn only when it
has at least one successful independent launch.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt


COLORS = ["#0077BB", "#EE7733", "#009988", "#CC3311", "#AA4499"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--output_stem", default="active_window_sweep")
    return parser.parse_args()


def finite_float(value: object) -> float | None:
    try:
        result = float(str(value))
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fin:
        return list(csv.DictReader(fin))


def main() -> None:
    args = parse_args()
    input_csv = Path(args.input_csv)
    rows = load_rows(input_csv)
    if not rows:
        raise SystemExit(f"no aggregate rows in {input_csv}")

    baseline_windows = {int(row["baseline_active_chunk_window"]) for row in rows}
    if len(baseline_windows) != 1:
        raise SystemExit("aggregate CSV must use exactly one baseline active-window value")
    baseline_window = baseline_windows.pop()
    shapes = sorted({row["shape"] for row in rows})
    windows = sorted({int(row["active_chunk_window"]) for row in rows})
    data: dict[str, dict[int, dict[str, str]]] = {shape: {} for shape in shapes}
    for row in rows:
        data[row["shape"]][int(row["active_chunk_window"])] = row

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "axes.labelsize": 9,
        "axes.titlesize": 9.5,
        "legend.fontsize": 7.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.23,
        "grid.linewidth": 0.6,
        "savefig.dpi": 450,
    })
    fig, (latency_ax, staging_ax) = plt.subplots(1, 2, figsize=(7.0, 2.85), constrained_layout=True)

    for index, shape in enumerate(shapes):
        color = COLORS[index % len(COLORS)]
        series = data[shape]
        latency_x: list[int] = []
        latency_y: list[float] = []
        latency_error: list[float] = []
        staging_x: list[int] = []
        staging_y: list[float] = []
        for window in windows:
            row = series.get(window)
            if row is None:
                continue
            normalized_latency = finite_float(row.get("latency_ratio_vs_baseline"))
            staging = finite_float(row.get("symmetric_staging_gib"))
            std = finite_float(row.get("latency_std_ms"))
            baseline = finite_float(row.get("baseline_latency_median_ms"))
            if normalized_latency is not None:
                latency_x.append(window)
                latency_y.append(normalized_latency)
                latency_error.append(std / baseline if std is not None and baseline and baseline > 0 else 0.0)
            if staging is not None:
                staging_x.append(window)
                staging_y.append(staging)
        if latency_x:
            latency_ax.errorbar(
                latency_x,
                latency_y,
                yerr=latency_error,
                color=color,
                marker="o",
                markersize=4,
                linewidth=1.5,
                capsize=2.4,
                label=shape.replace("x", " x "),
            )
        if staging_x:
            staging_ax.plot(
                staging_x,
                staging_y,
                color=color,
                marker="o",
                markersize=4,
                linewidth=1.5,
                label=shape.replace("x", " x "),
            )

    for axis in (latency_ax, staging_ax):
        axis.set_xticks(windows)
        axis.set_xlabel("Active window depth L")
    latency_ax.axhline(1.0, color="#666666", linewidth=0.8, linestyle="--", zorder=0)
    latency_ax.set_ylabel(f"Latency / latency at L={baseline_window}")
    latency_ax.set_title("Latency trade-off")
    staging_ax.set_ylabel("Symmetric staging (GiB)")
    staging_ax.set_title("Resident staging allocation")
    handles, labels = latency_ax.get_legend_handles_labels()
    if handles:
        staging_ax.legend(handles, labels, loc="best", frameon=False, title="M x N x K", title_fontsize=7.5)

    output_dir = Path(args.output_dir) if args.output_dir else input_csv.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png", "svg"):
        fig.savefig(output_dir / f"{args.output_stem}.{suffix}", dpi=450)
    print(f"[plot] {output_dir / (args.output_stem + '.pdf')}")
    print(f"[plot] {output_dir / (args.output_stem + '.png')}")
    print(f"[plot] {output_dir / (args.output_stem + '.svg')}")


if __name__ == "__main__":
    main()
