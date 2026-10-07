#!/usr/bin/env python3
"""Plot multi-shape RS frontier timeline events.

The figure is a multi-shape version of rs_frontier_timeline_bar_clean: for each
shape it draws one stacked horizontal bar for uniform scheduling and one for
frontier-first scheduling.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


READY_COLOR = "#2E86AB"
GAP_COLOR = "#F18F01"
COMMIT_COLOR = "#A23B72"
ANNOTATION_COLOR = "#4E5B6A"
GRID_COLOR = "#D8DEE9"


REQUIRED_COLUMNS = [
    "shape",
    "policy",
    "M",
    "N",
    "K",
    "first_panel_ready_ts_ms",
    "first_consumer_start_ts_ms",
    "first_output_commit_ts_ms",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot multi-shape RS frontier timeline bars.")
    parser.add_argument("--input_csv", required=True, type=str)
    parser.add_argument("--output_dir", default="", type=str)
    parser.add_argument("--formats", default="png,svg", type=str)
    parser.add_argument("--title", default="RS Frontier Timeline Across Shapes", type=str)
    parser.add_argument("--subtitle", default="", type=str)
    parser.add_argument("--dpi", type=int, default=450)
    return parser.parse_args()


def safe_float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    try:
        return float(value)
    except Exception as exc:
        raise RuntimeError(f"invalid numeric value for {key}: {value!r}") from exc


def normalize_policy(policy: str) -> str:
    text = policy.strip().lower().replace("-", "_").replace(" ", "_")
    if text in {"uniform", "frontier_first"}:
        return text
    raise RuntimeError(f"unsupported policy label: {policy!r}")


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
        ready = safe_float(row, "first_panel_ready_ts_ms")
        consumer = safe_float(row, "first_consumer_start_ts_ms")
        commit = safe_float(row, "first_output_commit_ts_ms")
        if not (0.0 <= ready <= consumer <= commit):
            raise RuntimeError(f"non-monotonic timeline values for {row.get('shape')} {row.get('policy')}")
        parsed.append(
            {
                "shape": row["shape"],
                "policy": normalize_policy(row["policy"]),
                "M": int(row["M"]),
                "N": int(row["N"]),
                "K": int(row["K"]),
                "ready": ready,
                "gap": consumer - ready,
                "commit_segment": commit - consumer,
                "consumer": consumer,
                "commit": commit,
            }
        )
    parsed.sort(key=lambda r: (int(r["M"]), int(r["N"]), int(r["K"]), 0 if r["policy"] == "uniform" else 1))
    return parsed


def policy_label(policy: str) -> str:
    return "Uniform" if policy == "uniform" else "Frontier-first"


def shape_label(row: dict[str, object]) -> str:
    return f"{int(row['M'])}x{int(row['N'])}x{int(row['K'])}"


def setup_style(plt) -> None:
    plt.rcParams.update(
        {
            "font.size": 9.5,
            "axes.titlesize": 11,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.1,
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
        from matplotlib.patches import Patch
    except Exception as exc:
        raise RuntimeError(f"matplotlib is required for plotting: {exc}") from exc

    setup_style(plt)

    y_positions: list[float] = []
    y_labels: list[str] = []
    ready_vals: list[float] = []
    gap_vals: list[float] = []
    commit_vals: list[float] = []
    commit_end_vals: list[float] = []
    row_positions: list[tuple[float, dict[str, object]]] = []

    current_y = 0.0
    current_shape = None
    for row in rows:
        if current_shape is not None and row["shape"] != current_shape:
            current_y += 0.50
        current_shape = row["shape"]
        y_positions.append(current_y)
        y_labels.append(f"{shape_label(row)} / {policy_label(str(row['policy']))}")
        ready_vals.append(float(row["ready"]))
        gap_vals.append(float(row["gap"]))
        commit_vals.append(float(row["commit_segment"]))
        commit_end_vals.append(float(row["commit"]))
        row_positions.append((current_y, row))
        current_y += 1.0

    fig_height = max(3.8, 1.0 + 0.52 * len(rows))
    fig, ax = plt.subplots(figsize=(7.8, fig_height))
    bar_height = 0.36

    ax.barh(y_positions, ready_vals, color=READY_COLOR, height=bar_height)
    ax.barh(y_positions, gap_vals, left=ready_vals, color=GAP_COLOR, height=bar_height)
    ax.barh(
        y_positions,
        commit_vals,
        left=[ready_vals[i] + gap_vals[i] for i in range(len(rows))],
        color=COMMIT_COLOR,
        height=bar_height,
    )

    for y, row in row_positions:
        ax.text(
            float(row["commit"]) + max(commit_end_vals) * 0.015,
            y,
            f"{float(row['commit']):.2f} ms",
            va="center",
            ha="left",
            fontsize=7.8,
            color=ANNOTATION_COLOR,
        )

    shape_groups: dict[str, list[float]] = {}
    for y, row in row_positions:
        shape_groups.setdefault(str(row["shape"]), []).append(y)
    for ys in shape_groups.values():
        if len(ys) == 2:
            ax.axhline(max(ys) + 0.50, color="#E5E7EB", linewidth=0.8)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels)
    ax.invert_yaxis()
    ax.set_xlabel("Time since producer launch (ms)")
    ax.set_title(title)
    if subtitle.strip():
        ax.text(0.5, 1.015, subtitle, transform=ax.transAxes, ha="center", va="bottom", fontsize=8.5, color=ANNOTATION_COLOR)
    ax.set_xlim(0.0, max(commit_end_vals) * 1.18)

    legend_handles = [
        Patch(facecolor=READY_COLOR, label="Launch to ready"),
        Patch(facecolor=GAP_COLOR, label="Ready to consumer"),
        Patch(facecolor=COMMIT_COLOR, label="Consumer to commit"),
    ]
    ax.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=3, fontsize=8.8)
    fig.subplots_adjust(top=0.90, bottom=0.18, left=0.33, right=0.94)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = "rs_frontier_timeline_multi_shape"
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
    print(f"[rs-frontier-timeline-multi] input csv: {input_csv}")
    print(f"[rs-frontier-timeline-multi] figure dir: {output_dir}")


if __name__ == "__main__":
    main()
