#!/usr/bin/env python3
"""Plot multi-shape frontier scheduling ablation results.

Expected input is produced by merge_frontier_schedule_ablation_csvs.py.
This plot intentionally uses measured values only. Do not adjust speedups unless
the CSV itself contains separately measured results.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


UNIFORM_COLOR = "#F18F01"
FRONTIER_COLOR = "#2E86AB"
SPEEDUP_COLOR = "#238B45"
ANNOTATION_COLOR = "#4E5B6A"
GRID_COLOR = "#D8DEE9"


REQUIRED_COLUMNS = [
    "shape",
    "uniform_total_ms",
    "frontier_total_ms",
    "uniform_internal_overlap_ratio",
    "frontier_internal_overlap_ratio",
    "frontier_speedup_vs_uniform",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot multi-shape frontier scheduling ablation results.")
    parser.add_argument("--input_csv", required=True, type=str)
    parser.add_argument("--output_dir", default="", type=str)
    parser.add_argument("--formats", default="png,svg", type=str)
    parser.add_argument("--title", default="Uniform vs. Frontier-First Across Shapes", type=str)
    parser.add_argument("--subtitle", default="", type=str)
    parser.add_argument("--dpi", type=int, default=450)
    return parser.parse_args()


def safe_float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    try:
        return float(value)
    except Exception as exc:
        raise RuntimeError(f"invalid numeric value for {key}: {value!r}") from exc


def load_rows(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fin:
        rows = list(csv.DictReader(fin))
    if not rows:
        raise RuntimeError(f"no rows found in {path}")
    missing = [col for col in REQUIRED_COLUMNS if col not in rows[0]]
    if missing:
        raise RuntimeError(f"{path} is missing columns: {', '.join(missing)}")

    parsed: list[dict[str, object]] = []
    for row in rows:
        parsed.append(
            {
                "shape": row["shape"],
                "M": int(row.get("M", 0) or 0),
                "N": int(row.get("N", 0) or 0),
                "K": int(row.get("K", 0) or 0),
                "uniform_total_ms": safe_float(row, "uniform_total_ms"),
                "frontier_total_ms": safe_float(row, "frontier_total_ms"),
                "uniform_overlap_pct": 100.0 * safe_float(row, "uniform_internal_overlap_ratio"),
                "frontier_overlap_pct": 100.0 * safe_float(row, "frontier_internal_overlap_ratio"),
                "speedup": safe_float(row, "frontier_speedup_vs_uniform"),
            }
        )
    parsed.sort(key=lambda r: (int(r["M"]), int(r["N"]), int(r["K"])))
    return parsed


def shape_label(row: dict[str, object]) -> str:
    return f"{int(row['M'])}x{int(row['N'])}\nx{int(row['K'])}"


def setup_style(plt) -> None:
    plt.rcParams.update(
        {
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.2,
            "ytick.labelsize": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.alpha": 0.42,
            "grid.linewidth": 0.7,
            "grid.color": GRID_COLOR,
            "legend.frameon": False,
            "savefig.bbox": "tight",
        }
    )


def plot(rows: list[dict[str, object]], output_dir: Path, formats: list[str], title: str, subtitle: str, dpi: int) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as exc:
        raise RuntimeError(f"matplotlib and numpy are required for plotting: {exc}") from exc

    setup_style(plt)

    labels = [shape_label(row) for row in rows]
    x = np.arange(len(rows))
    width = 0.34
    uniform_latency = [float(row["uniform_total_ms"]) for row in rows]
    frontier_latency = [float(row["frontier_total_ms"]) for row in rows]
    speedups = [float(row["speedup"]) for row in rows]
    fig, (ax_latency, ax_speedup) = plt.subplots(
        1,
        2,
        figsize=(7.8, 3.55),
        gridspec_kw={"width_ratios": [1.28, 1.0], "wspace": 0.34},
    )

    bars_u = ax_latency.bar(
        x - width / 2,
        uniform_latency,
        width,
        color=UNIFORM_COLOR,
        edgecolor="white",
        linewidth=0.8,
        label="Uniform schedule",
    )
    bars_f = ax_latency.bar(
        x + width / 2,
        frontier_latency,
        width,
        color=FRONTIER_COLOR,
        edgecolor="white",
        linewidth=0.8,
        label="Frontier-first schedule",
    )
    ax_latency.set_title("GEMM-RS Chain Latency")
    ax_latency.set_ylabel("Total latency (ms)")
    ax_latency.set_xticks(x)
    ax_latency.set_xticklabels(labels)
    ax_latency.legend(loc="upper left", ncols=1)

    latency_top = max(max(uniform_latency), max(frontier_latency))
    ax_latency.set_ylim(0.0, latency_top * 1.20)
    for bars in (bars_u, bars_f):
        for bar in bars:
            val = bar.get_height()
            ax_latency.text(
                bar.get_x() + bar.get_width() / 2.0,
                val + latency_top * 0.018,
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=7.8,
                color=ANNOTATION_COLOR,
            )

    bars_s = ax_speedup.bar(
        x,
        speedups,
        width=0.52,
        color=SPEEDUP_COLOR,
        edgecolor="white",
        linewidth=0.8,
    )
    ax_speedup.axhline(1.0, color="#4b5563", linewidth=1.0, linestyle="--")
    ax_speedup.set_title("Frontier-First Schedule Speedup")
    ax_speedup.set_ylabel("Speedup vs. uniform schedule (x)")
    ax_speedup.set_xticks(x)
    ax_speedup.set_xticklabels(labels)
    ax_speedup.set_ylim(0.0, max(speedups) * 1.22)

    for bar, speedup in zip(bars_s, speedups):
        ax_speedup.text(
            bar.get_x() + bar.get_width() / 2.0,
            speedup + max(speedups) * 0.025,
            f"{speedup:.2f}x",
            ha="center",
            va="bottom",
            fontsize=8.2,
            color=ANNOTATION_COLOR,
        )
    fig.suptitle(title, y=0.985, fontsize=12.2)
    if subtitle.strip():
        fig.text(0.5, 0.925, subtitle, ha="center", va="center", fontsize=8.5, color=ANNOTATION_COLOR)
    fig.subplots_adjust(top=0.84 if subtitle.strip() else 0.86, bottom=0.18, left=0.08, right=0.98)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = "frontier_schedule_ablation_multi_shape"
    for fmt in formats:
        fmt = fmt.strip().lower()
        if not fmt:
            continue
        fig.savefig(output_dir / f"{stem}.{fmt}", dpi=dpi if fmt == "png" else None)
    plt.close(fig)


def default_output_dir(input_csv: Path) -> Path:
    return input_csv.resolve().parent


def main() -> None:
    args = parse_args()
    input_csv = Path(args.input_csv)
    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir(input_csv)
    rows = load_rows(input_csv)
    plot(
        rows=rows,
        output_dir=output_dir,
        formats=[fmt.strip() for fmt in args.formats.split(",")],
        title=args.title,
        subtitle=args.subtitle,
        dpi=args.dpi,
    )
    print(f"[frontier-schedule-ablation-multi] input csv: {input_csv}")
    print(f"[frontier-schedule-ablation-multi] figure dir: {output_dir}")


if __name__ == "__main__":
    main()
