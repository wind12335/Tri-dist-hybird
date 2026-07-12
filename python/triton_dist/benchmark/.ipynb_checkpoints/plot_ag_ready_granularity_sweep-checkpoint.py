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

import argparse
import csv
from pathlib import Path


LATENCY_COLOR = "#2E86AB"
READY_COLOR = "#A23B72"
TAIL_COLOR = "#F18F01"
HEURISTIC_COLOR = "#C73E1D"
BASELINE_COLOR = "#6C757D"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Plot the AG ready-granularity sweep summary. "
            "The main curve shows total latency over coarse-to-fine tile granularity; "
            "the auxiliary panel shows timing proxies that help explain the tradeoff."
        )
    )
    parser.add_argument("--input_csv", required=True, type=str)
    parser.add_argument("--output_dir", default="", type=str)
    parser.add_argument("--formats", default="png,svg", type=str)
    parser.add_argument("--title", default="AG ready-granularity sweep", type=str)
    parser.add_argument("--shape_label", default="", type=str)
    parser.add_argument("--dpi", type=int, default=450)
    parser.add_argument("--hide_heuristic", action="store_true", default=False)
    return parser.parse_args()


def safe_float(value):
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    try:
        return float(text)
    except Exception:
        return None


def safe_int(value):
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(float(text))
    except Exception:
        return None


def load_rows(input_csv: Path) -> list[dict]:
    rows = []
    with open(input_csv, "r", encoding="utf-8-sig", newline="") as fin:
        reader = csv.DictReader(fin)
        for row in reader:
            parsed = dict(row)
            parsed["mode_tag"] = (row.get("mode_tag") or "").strip()
            parsed["driver_status"] = (row.get("driver_status") or "").strip()
            parsed["requested_tile_rows_per_chunk"] = safe_int(row.get("requested_tile_rows_per_chunk"))
            parsed["tile_rows_per_chunk"] = safe_int(row.get("tile_rows_per_chunk"))
            parsed["enable_tile_ready"] = str(row.get("enable_tile_ready") or "").strip().lower() in {
                "1",
                "true",
                "yes",
                "y",
                "on",
            }
            parsed["new_triton_total_ms"] = safe_float(
                row.get("new_triton_total_ms") or row.get("new dist-triton ag gemm latency (ms)")
            )
            parsed["base_triton_total_ms"] = safe_float(
                row.get("base_triton_total_ms") or row.get("dist-triton ag gemm latency (ms)")
            )
            parsed["torch_total_ms"] = safe_float(
                row.get("torch_total_ms") or row.get("torch ag gemm latency (ms)")
            )
            parsed["first_ready_ts_ms"] = safe_float(row.get("first_ready_ts_ms"))
            parsed["last_completion_ts_ms"] = safe_float(row.get("last_completion_ts_ms"))
            parsed["ready_to_completion_tail_ms"] = safe_float(row.get("ready_to_completion_tail_ms"))
            parsed["M"] = safe_int(row.get("M"))
            parsed["N"] = safe_int(row.get("N"))
            parsed["K"] = safe_int(row.get("K"))
            rows.append(parsed)
    return rows


def fixed_sweep_rows(rows: list[dict]) -> list[dict]:
    kept = []
    for row in rows:
        if row.get("driver_status") not in {"ok", "dry_run"}:
            continue
        mode = row.get("mode_tag")
        if mode == "rank_ready":
            kept.append(row)
            continue
        requested = row.get("requested_tile_rows_per_chunk")
        if requested is not None and requested > 0:
            kept.append(row)
    def sort_key(row: dict):
        mode = row.get("mode_tag")
        if mode == "rank_ready":
            return (0, 10**12)
        requested = row.get("requested_tile_rows_per_chunk") or 0
        return (1, -requested)
    return sorted(kept, key=sort_key)


def heuristic_row(rows: list[dict]) -> dict | None:
    for row in rows:
        if row.get("mode_tag") == "tile_ready_heuristic" and row.get("driver_status") in {"ok", "dry_run"}:
            return row
    return None


def setup_style(plt):
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.7,
            "legend.frameon": False,
            "savefig.bbox": "tight",
        }
    )


def x_label_for_row(row: dict) -> str:
    if row.get("mode_tag") == "rank_ready":
        return "rank-ready"
    requested = row.get("requested_tile_rows_per_chunk")
    return str(requested) if requested is not None else row.get("mode_tag", "")


def annotate_best(ax, rows: list[dict], xs: list[int]) -> None:
    best_idx = None
    best_value = None
    for idx, row in enumerate(rows):
        value = row.get("new_triton_total_ms")
        if value is None:
            continue
        if best_value is None or value < best_value:
            best_value = value
            best_idx = idx
    if best_idx is None:
        return
    ax.scatter([xs[best_idx]], [best_value], s=54, color=LATENCY_COLOR, edgecolor="white", zorder=5)
    ax.annotate(
        f"best={best_value:.2f} ms",
        xy=(xs[best_idx], best_value),
        xytext=(8, -14),
        textcoords="offset points",
        fontsize=9,
        color=LATENCY_COLOR,
        weight="bold",
    )


def plot(input_csv: Path, output_dir: Path, formats: list[str], title: str, shape_label: str, dpi: int, hide_heuristic: bool):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError(f"matplotlib is required for plotting: {exc}") from exc

    rows = load_rows(input_csv)
    sweep_rows = fixed_sweep_rows(rows)
    if not sweep_rows:
        raise RuntimeError("no fixed sweep rows found in input CSV")

    heuristic = None if hide_heuristic else heuristic_row(rows)
    setup_style(plt)

    x_positions = list(range(len(sweep_rows)))
    x_labels = [x_label_for_row(row) for row in sweep_rows]

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(7.2, 6.2),
        sharex=True,
        gridspec_kw={"height_ratios": [1.35, 1.0], "hspace": 0.16},
    )
    ax_latency, ax_proxy = axes

    y_latency = [row.get("new_triton_total_ms") for row in sweep_rows]
    ax_latency.plot(
        x_positions,
        y_latency,
        color=LATENCY_COLOR,
        linewidth=1.8,
        marker="o",
        markersize=5,
        label="new kernel total latency",
    )

    if sweep_rows and sweep_rows[0].get("base_triton_total_ms") is not None:
        base_latency = sweep_rows[0].get("base_triton_total_ms")
        ax_latency.axhline(
            base_latency,
            color=BASELINE_COLOR,
            linewidth=1.1,
            linestyle="--",
            label=f"base-triton rank-ready baseline ({base_latency:.2f} ms)",
        )

    if heuristic is not None and heuristic.get("new_triton_total_ms") is not None:
        ax_latency.scatter(
            [x_positions[-1] + 0.45],
            [heuristic["new_triton_total_ms"]],
            marker="*",
            s=120,
            color=HEURISTIC_COLOR,
            edgecolor="white",
            linewidth=0.8,
            label=(
                "heuristic point"
                if heuristic.get("tile_rows_per_chunk") is None
                else f"heuristic point ({heuristic.get('tile_rows_per_chunk')} rows)"
            ),
            zorder=6,
        )
        ax_latency.annotate(
            "heuristic",
            xy=(x_positions[-1] + 0.45, heuristic["new_triton_total_ms"]),
            xytext=(6, 8),
            textcoords="offset points",
            fontsize=8.5,
            color=HEURISTIC_COLOR,
        )

    annotate_best(ax_latency, sweep_rows, x_positions)
    ax_latency.set_ylabel("Total latency (ms)")
    subtitle = shape_label.strip()
    if not subtitle:
        m = sweep_rows[0].get("M")
        n = sweep_rows[0].get("N")
        k = sweep_rows[0].get("K")
        if m is not None and n is not None and k is not None:
            subtitle = f"M={m}, N={n}, K={k}"
    ax_latency.set_title(title if not subtitle else f"{title}\n{subtitle}")
    ax_latency.legend(loc="best")

    y_ready = [row.get("first_ready_ts_ms") for row in sweep_rows]
    y_tail = [
        row.get("ready_to_completion_tail_ms")
        if row.get("ready_to_completion_tail_ms") is not None
        else (
            (row.get("last_completion_ts_ms") - row.get("first_ready_ts_ms"))
            if row.get("last_completion_ts_ms") is not None and row.get("first_ready_ts_ms") is not None
            else None
        )
        for row in sweep_rows
    ]
    ax_proxy.plot(
        x_positions,
        y_ready,
        color=READY_COLOR,
        linewidth=1.6,
        marker="o",
        markersize=4.5,
        label="first ready timestamp",
    )
    ax_proxy.plot(
        x_positions,
        y_tail,
        color=TAIL_COLOR,
        linewidth=1.6,
        marker="s",
        markersize=4.3,
        label="ready-to-completion tail",
    )
    ax_proxy.set_ylabel("Timing proxy (ms)")
    ax_proxy.set_xlabel("Ready granularity (tile_rows_per_chunk, coarse → fine)")
    ax_proxy.set_xticks(x_positions)
    ax_proxy.set_xticklabels(x_labels)
    ax_proxy.legend(loc="best")

    fig.align_ylabels(axes)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = input_csv.stem.replace("_summary", "")
    for fmt in formats:
        fmt = fmt.strip().lower()
        if not fmt:
            continue
        save_path = output_dir / f"{stem}_plot.{fmt}"
        if fmt == "png":
            fig.savefig(save_path, dpi=dpi)
        else:
            fig.savefig(save_path)
        print(f"[saved] {save_path}", flush=True)
    plt.close(fig)


def main():
    args = parse_args()
    input_csv = Path(args.input_csv)
    if not input_csv.exists():
        raise FileNotFoundError(f"input CSV not found: {input_csv}")
    output_dir = Path(args.output_dir) if args.output_dir else input_csv.parent / "plots"
    formats = [part for part in args.formats.split(",") if part.strip()]
    plot(
        input_csv=input_csv,
        output_dir=output_dir,
        formats=formats,
        title=args.title,
        shape_label=args.shape_label,
        dpi=args.dpi,
        hide_heuristic=args.hide_heuristic,
    )


if __name__ == "__main__":
    main()
