#!/usr/bin/env python3
"""Plot the AG-GEMM ready-granularity ablation from aggregated sweep CSVs.

Input: ``ag_ready_aggregated.csv`` produced by
``bench_ag_ready_granularity_sweep_eval.py`` (one row per shape x granularity
cell, median / min--max rank-max latencies).

For a single shape the figure has two panels:
  (a) total step latency (rank-max median) over coarse -> fine ready
      granularity, with the min--max band and the triton-dist base reference, and
  (b) speedup of the new path over the triton-dist base (base_median / new_median).
      The baseline is the framework's own base path, not torch (torch is a
      separate NCCL+cuBLAS stack and is not the paper's comparison).

For several shapes it becomes a 2-row x N-col grid, one column per shape: the top
row is each shape's latency panel (own y-scale, so shapes with very different
latency magnitudes are not flattened onto a shared axis), and the bottom row is
each shape's speedup-vs-base panel.

Note on the first-ready proxy: a benchmark-level poll of the tile barrier cannot
reliably observe the barrier in flight, because the consumer runs on a stream
that is serialized behind the AG chain (the probe only ever sees the barrier
after completion). A reliable first-ready timestamp requires kernel-instrumented
timestamps; until then the figure shows latency + speedup only.

Data-integrity guards (intentional):
  - ``effective_enable_row_tile_barrier`` must be 1 for tile points. When a tile
    cell silently fell back to rank-ready (M_per_rank below
    ``min_m_per_rank_for_tile_ready``), the script prints a warning and marks the
    point instead of plotting a phantom granularity trend.
  - Failed / partial cells (``failure_class`` != ok) are shown distinctly and are
    not silently dropped.

Usage::

  python benchmark/plot_ag_ready_ablation.py \
      --input_csv benchmark/ag_ready_granularity_eval_results/8192x28672x8192/20260812_160028/ag_ready_aggregated.csv \
      --output_dir overleaf_methodology_zh/figures
"""

from __future__ import annotations

import argparse
import glob
import csv
import re
import statistics
import sys
from pathlib import Path

LATENCY_COLOR = "#2E86AB"
SPEEDUP_COLOR = "#238B45"
BASELINE_COLOR = "#6C757D"
FAIL_COLOR = "#C73E1D"
FALLBACK_COLOR = "#8E44AD"
GRID_COLOR = "#D8DEE9"

SHAPE_COLORS = ["#2E86AB", "#D35400", "#238B45", "#8E44AD", "#C0392B", "#16A085"]
SHAPE_MARKERS = ["o", "s", "^", "D", "v", "P"]

GRANULARITY_RANK_READY = 10**9
GRANULARITY_HEURISTIC = -1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input_csv", required=True, nargs="+", type=str,
                        help="One or more ag_ready_aggregated.csv paths (glob patterns allowed).")
    parser.add_argument("--output_dir", default="", type=str,
                        help="Figure output dir (default: parent of the first input CSV).")
    parser.add_argument("--formats", default="png,svg,pdf", type=str)
    parser.add_argument("--title", default="AG-GEMM ready-granularity ablation", type=str)
    parser.add_argument("--dpi", type=int, default=450)
    return parser.parse_args()


def safe_float(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in ("nan", "none", ""):
        return None
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def safe_int(value: object) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in ("nan", "none"):
        return None
    try:
        return int(float(text))
    except (TypeError, ValueError):
        return None


def parse_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def resolve_inputs(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    seen: set[str] = set()
    for pattern in patterns:
        if any(ch in pattern for ch in "*?["):
            matches = sorted(glob.glob(pattern))
        else:
            matches = [pattern] if Path(pattern).exists() else []
        for match in matches:
            resolved = str(Path(match).resolve())
            if resolved not in seen:
                seen.add(resolved)
                paths.append(Path(match))
    if not paths:
        raise SystemExit(f"no input CSV matched: {patterns}")
    return paths


def load_aggregated(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as fin:
        return list(csv.DictReader(fin))


def derive_shape_identity(row: dict, csv_path: Path) -> dict:
    """Fill missing shape_tag/M/N/K from the CSV path.

    The aggregated CSVs do not carry shape identity; it lives only in the result
    dir name (``<root>/<shape_tag>/<run_id>/ag_ready_aggregated.csv``). Merged
    CSVs already have the columns injected, so this only fills blanks.
    """
    shape_tag = (row.get("shape_tag") or "").strip()
    m = (row.get("M") or "").strip()
    n = (row.get("N") or "").strip()
    k = (row.get("K") or "").strip()
    if shape_tag and m and n and k:
        return row
    dir_name = csv_path.resolve().parent.parent.name
    mt = re.fullmatch(r"(\d+)x(\d+)x(\d+)", dir_name)
    if not shape_tag:
        row["shape_tag"] = dir_name
    if mt:
        if not m:
            row["M"] = mt.group(1)
        if not n:
            row["N"] = mt.group(2)
        if not k:
            row["K"] = mt.group(3)
    return row


def cell_speedup(row: dict) -> float | None:
    """Per-cell speedup of the new path over the triton-dist base (base / new).

    Reads the pre-computed ``speedup`` column (written by
    ``merge_ag_ready_shapes.py``); falls back to base_median / new_median for
    raw aggregated CSVs that lack the column.
    """
    sp = safe_float(row.get("speedup"))
    if sp is not None:
        return sp
    median = safe_float(row.get("new_median_ms"))
    base_med = safe_float(row.get("base_median_ms"))
    if median is None or base_med is None or median == 0:
        return None
    return base_med / median


def group_by_shape(rows: list[dict]) -> list[tuple[str, list[dict]]]:
    groups: dict[tuple, dict] = {}
    for row in rows:
        key = (row.get("shape_tag"), row.get("M"), row.get("N"), row.get("K"))
        groups.setdefault(key, []).append(row)
    return list(groups.items())


def granularity_sort_key(row: dict) -> tuple[int, int]:
    """Order: rank_ready (left) -> fixed tiles coarse->fine -> heuristic (right)."""
    mode = (row.get("mode_tag") or "").strip()
    granularity = safe_float(row.get("granularity"))
    if mode == "rank_ready" or (granularity is not None and granularity >= GRANULARITY_RANK_READY / 2):
        return (0, 0)
    if mode == "tile_ready_heuristic" or (granularity is not None and granularity == GRANULARITY_HEURISTIC):
        return (2, 0)
    requested = safe_int(row.get("requested_tile_rows_per_chunk")) or 0
    return (1, -requested)


def x_positions(rows: list[dict]) -> tuple[dict[str, int], list[dict], dict[str, int]]:
    """Assign categorical x positions to the ordered cells."""
    ordered = sorted(rows, key=granularity_sort_key)
    pos: dict[str, int] = {}
    position_of_mode: dict[str, int] = {}
    for i, row in enumerate(ordered):
        tag = (row.get("mode_tag") or "").strip()
        pos[id(row)] = i
        position_of_mode[tag] = i
    return pos, ordered, position_of_mode


def x_label(row: dict) -> str:
    mode = (row.get("mode_tag") or "").strip()
    if mode == "rank_ready":
        return "rank-ready"
    requested = safe_int(row.get("requested_tile_rows_per_chunk"))
    if requested is not None:
        return str(requested)
    if mode == "tile_ready_heuristic":
        return "heuristic"
    return mode


def setup_style(plt) -> None:
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID_COLOR,
        "grid.alpha": 0.4,
        "grid.linewidth": 0.6,
        "legend.frameon": False,
        "savefig.bbox": "tight",
    })


def shape_title(row: dict) -> str:
    m = row.get("M")
    n = row.get("N")
    k = row.get("K")
    tag = row.get("shape_tag") or ""
    if m is not None and n is not None and k is not None:
        return f"{tag}  M={m}, N={n}, K={k}"
    return tag or "shape"


def draw_latency_panel(ax, rows: list[dict], pos: dict[str, int], show_ylabel: bool,
                       show_legend: bool = True) -> None:
    plotted_ys: list[float] = []
    fallback_points = []
    failed_points = []

    x_list = []
    y_list = []
    ymin_list = []
    ymax_list = []
    for row in rows:
        tag = (row.get("mode_tag") or "").strip()
        x = pos[id(row)]
        n_launches = safe_int(row.get("n_launches")) or 0
        n_success = safe_int(row.get("n_success")) or 0
        failure_class = (row.get("failure_class") or "").strip()

        engaged = parse_bool(row.get("effective_enable_row_tile_barrier"))
        if tag.startswith("tile_") and tag != "tile_ready_heuristic" and not engaged:
            fallback_points.append((x, row))

        median = safe_float(row.get("new_median_ms"))
        if median is None or n_success == 0:
            failed_points.append((x, row, failure_class))
            continue

        x_list.append(x)
        y_list.append(median)
        ymin_list.append(safe_float(row.get("new_min_ms")) or median)
        ymax_list.append(safe_float(row.get("new_max_ms")) or median)
        plotted_ys.append(median)

    if x_list:
        ax.plot(x_list, y_list, color=LATENCY_COLOR, linewidth=1.8, marker="o", markersize=5,
                label="ready path (rank-max)")
        ax.fill_between(x_list, ymin_list, ymax_list, color=LATENCY_COLOR, alpha=0.18, linewidth=0,
                        label="min-max range")

    # triton-dist base reference (granularity-independent): median of the old-path
    # latency across this shape's cells. We compare against the framework's own
    # base path, not torch (torch is a separate NCCL+cuBLAS stack and is not the
    # paper's baseline). Drawn before the "best" annotation so the annotation's
    # vertical placement can account for the final y-limits.
    base_values = [v for v in (safe_float(row.get("base_median_ms")) for row in rows) if v is not None]
    if base_values:
        base_ref = statistics.median(base_values)
        ax.axhline(base_ref, color=BASELINE_COLOR, linewidth=1.1, linestyle="--",
                   label="Triton-distributed")

    # Best-point highlight + annotation, kept inside the panel: anchored away
    # from the left/right plot edges and flipped below the point when there is no
    # room above (the base line can push the y-axis up).
    if x_list:
        best_idx = min(range(len(y_list)), key=lambda i: y_list[i])
        bx, by = x_list[best_idx], y_list[best_idx]
        ax.scatter([bx], [by], s=54, color=LATENCY_COLOR, edgecolor="white", zorder=5)
        ha = "center"
        if bx <= min(x_list):
            ha = "left"
        elif bx >= max(x_list):
            ha = "right"
        ylim_bottom, ylim_top = ax.get_ylim()
        above = (ylim_top - by) > (ylim_top - ylim_bottom) * 0.40
        ax.annotate(f"best {by:.2f} ms", xy=(bx, by),
                    xytext=(0, 12 if above else -14),
                    ha=ha, va="bottom" if above else "top",
                    textcoords="offset points",
                    fontsize=8.5, color=LATENCY_COLOR, weight="bold")

    # Failed cells: keep them visible, never silently drop.
    for x, row, failure_class in failed_points:
        top = max(plotted_ys) * 1.06 if plotted_ys else 1.0
        ax.scatter([x], [top], marker="x", s=70, color=FAIL_COLOR, zorder=6)
        ax.annotate(failure_class, xy=(x, top), xytext=(0, -12), textcoords="offset points",
                    fontsize=7.5, color=FAIL_COLOR, ha="center")

    # Silent rank-ready fallback: annotate, do not plot as a real tile trend.
    for x, row in fallback_points:
        if x_list:
            bottom = min(plotted_ys) * 0.97
        else:
            bottom = 1.0
        ax.annotate("fell back to\nrank-ready", xy=(x, bottom), xytext=(0, 10),
                    textcoords="offset points", fontsize=7.5, color=FALLBACK_COLOR, ha="center", va="bottom")

    if fallback_points:
        print(f"[warn] {len(fallback_points)} tile cell(s) did NOT engage row-tile barrier "
              f"(M_per_rank < min_m_per_rank_for_tile_ready); they measured rank-ready. "
              f"Fix by raising --min_m_per_rank_for_tile_ready (or the shape M).", file=sys.stderr)

    ax.set_ylabel("Total step latency (ms)" if show_ylabel else "")
    if show_legend:
        ax.legend(loc="upper right", fontsize=7.2)


def draw_speedup_panel(ax, rows: list[dict], pos: dict[str, int], show_ylabel: bool,
                       show_legend: bool = True) -> None:
    """Second panel: speedup of the new path over the triton-dist base path
    (base / new), not over torch — torch is a separate NCCL+cuBLAS stack and is
    not the paper's baseline. (A raw first-ready proxy is not plotted because a
    benchmark-level poll cannot observe the tile barrier in flight.)"""
    x_list = []
    y_list = []
    for row in rows:
        sp = cell_speedup(row)
        if sp is None:
            continue
        x_list.append(pos[id(row)])
        y_list.append(sp)

    if x_list:
        ax.plot(x_list, y_list, color=SPEEDUP_COLOR, linewidth=1.6, marker="o", markersize=4.5,
                label="reference speedup")
        ax.axhline(1.0, color=BASELINE_COLOR, linewidth=1.0, linestyle=":", alpha=0.8)
        ax.set_ylim(bottom=max(0.85, min(y_list) - 0.02))
    # xlabel is set figure-wide via fig.supxlabel() in main(), so it centers
    # under both panels instead of sitting under the right panel only.
    ax.set_ylabel("Speedup vs. reference (x)" if show_ylabel else "")
    if show_legend:
        ax.legend(loc="best", fontsize=7.2)


def shape_legend_tag(shape_rows: list[dict]) -> str:
    tag = (shape_rows[0].get("shape_tag") or "").strip()
    if tag:
        return tag
    m, n, k = shape_rows[0].get("M"), shape_rows[0].get("N"), shape_rows[0].get("K")
    if m is not None and n is not None and k is not None:
        return f"{m}x{n}x{k}"
    return "shape"


def draw_multi_curve_figure(plt, groups: list[tuple[tuple, list[dict]]], args):
    """Multiple shapes: a 2-row x N-col grid with one column per shape. Top row
    is that shape's latency panel (its own y-scale, so the granularity trend is
    not flattened by shapes with very different latency magnitudes, e.g. ~4.7 ms
    vs ~18 ms); bottom row is that shape's speedup-vs-base panel. Each column is
    self-contained (every panel gets its own y-label and legend)."""
    n = len(groups)
    fig, axes = plt.subplots(2, n, figsize=(8.8, 5.2), squeeze=False)
    fig.subplots_adjust(left=0.085, right=0.99, top=0.90, bottom=0.14,
                        hspace=0.50, wspace=0.32)

    tick_pos = None
    tick_labels = None
    for idx, (_key, shape_rows) in enumerate(groups):
        pos, ordered, _ = x_positions(shape_rows)
        if tick_pos is None:
            tick_pos = [pos[id(r)] for r in ordered]
            tick_labels = [x_label(r) for r in ordered]
        tag = shape_legend_tag(shape_rows)

        ax_lat = axes[0][idx]
        # Every panel keeps its y-label; the identical 3-entry legend is shown
        # once (on the first column) to avoid redundancy and overlap in the
        # narrow per-shape panels.
        draw_latency_panel(ax_lat, ordered, pos, show_ylabel=True, show_legend=(idx == 0))
        ax_lat.set_xticks(tick_pos)
        ax_lat.set_xticklabels(tick_labels, fontsize=8)
        ax_lat.set_title(f"({chr(ord('a') + idx)}) {tag}", fontsize=10)

        ax_spd = axes[1][idx]
        draw_speedup_panel(ax_spd, ordered, pos, show_ylabel=True, show_legend=(idx == 0))
        ax_spd.set_xticks(tick_pos)
        ax_spd.set_xticklabels(tick_labels, fontsize=8)
        ax_spd.set_title(f"({chr(ord('a') + n + idx)})", fontsize=10)

    fig.suptitle(args.title, fontsize=12)
    fig.supxlabel("Ready granularity (rows per chunk, coarse → fine)", fontsize=10)
    return fig


def main() -> None:
    args = parse_args()
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        raise SystemExit(f"matplotlib is required for plotting: {exc}") from exc

    inputs = resolve_inputs(args.input_csv)
    rows: list[dict] = []
    for path in inputs:
        for row in load_aggregated(path):
            rows.append(derive_shape_identity(row, path))
    if not rows:
        raise SystemExit("input CSV(s) contained no rows")

    groups = group_by_shape(rows)
    if not groups:
        raise SystemExit("no shape groups found in input CSV(s)")

    setup_style(plt)
    n_shapes = len(groups)

    if n_shapes == 1:
        # Single shape: detailed layout (min--max band, best-point annotation,
        # per-shape triton-dist base reference).
        fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.3), squeeze=False)
        # Manual layout: top=0.82 reserves explicit room for the 2-line panel
        # titles and the figure suptitle so they never collide.
        fig.subplots_adjust(left=0.10, right=0.985, top=0.82, bottom=0.17,
                            wspace=0.30, hspace=0.55)

        _key, shape_rows = groups[0]
        pos, ordered, _ = x_positions(shape_rows)
        ax_latency = axes[0][0]
        ax_proxy = axes[0][1]

        draw_latency_panel(ax_latency, ordered, pos, show_ylabel=True)
        draw_speedup_panel(ax_proxy, ordered, pos, show_ylabel=True)

        labels = [x_label(row) for row in ordered]
        for ax in (ax_latency, ax_proxy):
            ax.set_xticks([pos[id(row)] for row in ordered])
            ax.set_xticklabels(labels, fontsize=8)

        title = shape_title(shape_rows[0])
        ax_latency.set_title(f"{title}\n(best total latency)", fontsize=10)
        ax_proxy.set_title(f"{title}\n(speedup vs triton-dist base)", fontsize=10)

        fig.suptitle(args.title, fontsize=12, y=0.97)
        fig.supxlabel("Ready granularity (tile_rows_per_chunk, coarse → fine)", fontsize=10)
        fig.align_ylabels(axes)
    else:
        # Multiple shapes: one curve per shape in shared latency + speedup panels.
        fig = draw_multi_curve_figure(plt, groups, args)

    output_dir = Path(args.output_dir) if args.output_dir else inputs[0].resolve().parent
    output_dir.mkdir(parents=True, exist_ok=True)
    for fmt in (f.strip().lower() for f in args.formats.split(",")):
        if not fmt:
            continue
        save_path = output_dir / f"ag_ready_ablation.{fmt}"
        fig.savefig(save_path, dpi=args.dpi)
        print(f"[saved] {save_path}", flush=True)
    plt.close(fig)


if __name__ == "__main__":
    main()
