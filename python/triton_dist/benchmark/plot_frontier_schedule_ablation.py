################################################################################
#
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
################################################################################

"""
Plot the frontier scheduling ablation result from
perf_frontier_schedule_ablation_gemm_rs_4_ranks.csv.

The CSV schema is the one emitted by
bench_frontier_schedule_ablation_gemmrs.py.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


UNIFORM_COLOR = "#F18F01"
FRONTIER_COLOR = "#2E86AB"
ANNOTATION_COLOR = "#4E5B6A"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Plot the frontier scheduling ablation benchmark. "
            "The input CSV should be produced by bench_frontier_schedule_ablation_gemmrs.py."
        )
    )
    parser.add_argument("--input_csv", required=True, type=str)
    parser.add_argument("--output_dir", default="", type=str)
    parser.add_argument("--formats", default="png,svg", type=str)
    parser.add_argument("--title", default="Frontier Scheduling Ablation", type=str)
    parser.add_argument(
        "--subtitle",
        default="Uniform vs frontier-first within the panelized/windowed RS-GEMM family",
        type=str,
    )
    parser.add_argument("--dpi", type=int, default=450)
    return parser.parse_args()


def safe_float(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    try:
        return float(text)
    except Exception:
        return None


def load_single_row(input_csv: Path) -> dict[str, object]:
    with open(input_csv, "r", encoding="utf-8-sig", newline="") as fin:
        rows = list(csv.DictReader(fin))
    if not rows:
        raise RuntimeError("no rows found in input CSV")
    row = rows[0]
    parsed = dict(row)
    numeric_fields = [
        "torch_total_ms",
        "uniform_total_ms",
        "uniform_internal_overlap_ratio",
        "frontier_total_ms",
        "frontier_internal_overlap_ratio",
        "frontier_speedup_vs_uniform",
        "frontier_speedup_vs_torch",
        "uniform_speedup_vs_torch",
        "M",
        "N",
        "K",
        "chunk_rows",
        "active_chunk_window",
        "stage_slots",
        "comm_lanes",
        "n_bands",
        "frontier_chunks",
    ]
    for field in numeric_fields:
        parsed[field] = safe_float(row.get(field))
    return parsed


def validate_row(row: dict[str, object]) -> None:
    required = [
        "uniform_total_ms",
        "uniform_internal_overlap_ratio",
        "frontier_total_ms",
        "frontier_internal_overlap_ratio",
        "frontier_speedup_vs_uniform",
    ]
    missing = [field for field in required if row.get(field) is None]
    if missing:
        raise RuntimeError("missing required columns in ablation CSV: " + ", ".join(missing))


def setup_style(plt):
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.7,
            "legend.frameon": False,
            "savefig.bbox": "tight",
        }
    )


def plot(row: dict[str, object], output_dir: Path, formats: list[str], title: str, subtitle: str, dpi: int):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError(f"matplotlib is required for plotting: {exc}") from exc

    setup_style(plt)
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.8, 3.8),
        gridspec_kw={"width_ratios": [1.0, 1.0], "wspace": 0.34},
    )

    labels = ["Uniform", "Frontier-first"]
    latency_vals = [float(row["uniform_total_ms"]), float(row["frontier_total_ms"])]
    overlap_vals = [
        100.0 * float(row["uniform_internal_overlap_ratio"]),
        100.0 * float(row["frontier_internal_overlap_ratio"]),
    ]
    colors = [UNIFORM_COLOR, FRONTIER_COLOR]
    bar_width = 0.52

    ax_latency, ax_overlap = axes

    bars_latency = ax_latency.bar(labels, latency_vals, color=colors, width=bar_width, edgecolor="white")
    ax_latency.set_ylabel("Total latency (ms)")
    ax_latency.set_title("End-to-End Latency")
    for bar, val in zip(bars_latency, latency_vals):
        ax_latency.text(
            bar.get_x() + bar.get_width() / 2.0,
            val + max(latency_vals) * 0.018,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color=ANNOTATION_COLOR,
        )

    bars_overlap = ax_overlap.bar(labels, overlap_vals, color=colors, width=bar_width, edgecolor="white")
    ax_overlap.set_ylabel("Internal overlap ratio (%)")
    ax_overlap.set_title("Overlap Quality")
    for bar, val in zip(bars_overlap, overlap_vals):
        ax_overlap.text(
            bar.get_x() + bar.get_width() / 2.0,
            val + max(overlap_vals) * 0.018,
            f"{val:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color=ANNOTATION_COLOR,
        )

    fig.suptitle(title, y=0.98, fontsize=12)
    shape_text = (
        f"M={int(row['M'])}, N={int(row['N'])}, K={int(row['K'])}; "
        f"chunk={int(row['chunk_rows'])}, window={int(row['active_chunk_window'])}, "
        f"slots={int(row['stage_slots'])}, bands={int(row['n_bands'])}, frontier={int(row['frontier_chunks'])}"
    )
    fig.text(0.5, 0.90, subtitle, ha="center", va="center", fontsize=8.5, color=ANNOTATION_COLOR)
    fig.text(0.5, 0.85, shape_text, ha="center", va="center", fontsize=8.0, color=ANNOTATION_COLOR)

    speedup_text = (
        f"Frontier speedup vs uniform: {float(row['frontier_speedup_vs_uniform']):.2f}x | "
        f"Uniform vs torch: {float(row['uniform_speedup_vs_torch']):.2f}x | "
        f"Frontier vs torch: {float(row['frontier_speedup_vs_torch']):.2f}x"
    )
    fig.text(0.5, 0.06, speedup_text, ha="center", va="center", fontsize=8.2, color=ANNOTATION_COLOR)
    fig.subplots_adjust(top=0.74, bottom=0.20, left=0.10, right=0.97)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = "frontier_schedule_ablation"
    for fmt in formats:
        fmt = fmt.strip().lower()
        if not fmt:
            continue
        fig.savefig(output_dir / f"{stem}.{fmt}", dpi=dpi if fmt == "png" else None)
    plt.close(fig)


def default_output_dir(input_csv: Path) -> Path:
    return input_csv.resolve().parent


def main():
    args = parse_args()
    input_csv = Path(args.input_csv)
    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir(input_csv)
    row = load_single_row(input_csv)
    validate_row(row)
    plot(
        row=row,
        output_dir=output_dir,
        formats=[fmt.strip() for fmt in args.formats.split(",")],
        title=args.title,
        subtitle=args.subtitle,
        dpi=args.dpi,
    )
    print(f"[frontier-schedule-ablation] input csv: {input_csv}")
    print(f"[frontier-schedule-ablation] figure dir: {output_dir}")


if __name__ == "__main__":
    main()
