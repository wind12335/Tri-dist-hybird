#!/usr/bin/env python3
"""Vertical clean multi-shape RS frontier timeline bars."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
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
    parser = argparse.ArgumentParser(description="Plot vertical clean multi-shape RS frontier timeline bars.")
    parser.add_argument("--input_csv", required=True, type=str)
    parser.add_argument("--output_dir", default="", type=str)
    parser.add_argument("--formats", default="png,svg", type=str)
    parser.add_argument("--title", default="Frontier Timeline Across Shapes", type=str)
    parser.add_argument("--dpi", type=int, default=450)
    return parser.parse_args()


def normalize_policy(policy: str) -> str:
    text = policy.strip().lower().replace("-", "_").replace(" ", "_")
    if text in {"uniform", "frontier_first"}:
        return text
    raise RuntimeError(f"unsupported policy label: {policy!r}")


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
                "commit": commit,
            }
        )
    parsed.sort(key=lambda r: (int(r["M"]), int(r["N"]), int(r["K"]), 0 if r["policy"] == "uniform" else 1))
    return parsed


def group_rows(rows: list[dict[str, object]]) -> list[tuple[str, list[dict[str, object]]]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["shape"])].append(row)

    groups: list[tuple[str, list[dict[str, object]]]] = []
    for shape, shape_rows in grouped.items():
        policies = {str(row["policy"]) for row in shape_rows}
        if policies != {"uniform", "frontier_first"}:
            raise RuntimeError(f"{shape} should contain uniform and frontier_first rows, got {sorted(policies)}")
        shape_rows.sort(key=lambda r: 0 if r["policy"] == "uniform" else 1)
        groups.append((shape, shape_rows))
    groups.sort(key=lambda item: (int(item[1][0]["M"]), int(item[1][0]["N"]), int(item[1][0]["K"])))
    return groups


def shape_label(rows: list[dict[str, object]]) -> str:
    row = rows[0]
    return f"{int(row['M'])}x{int(row['N'])}\nx{int(row['K'])}"


def setup_style(plt) -> None:
    plt.rcParams.update(
        {
            "font.size": 9.8,
            "axes.titlesize": 11,
            "axes.labelsize": 9.8,
            "xtick.labelsize": 8.4,
            "ytick.labelsize": 8.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.alpha": 0.34,
            "grid.linewidth": 0.7,
            "grid.color": GRID_COLOR,
            "legend.frameon": False,
            "savefig.bbox": "tight",
        }
    )


def plot(groups: list[tuple[str, list[dict[str, object]]]], output_dir: Path, formats: list[str], title: str, dpi: int) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.patches import Patch
    except Exception as exc:
        raise RuntimeError(f"matplotlib and numpy are required for plotting: {exc}") from exc

    setup_style(plt)
    group_spacing = 0.70
    x = np.arange(len(groups)) * group_spacing
    width = 0.14
    offsets = {"uniform": -0.12, "frontier_first": 0.12}
    max_commit = max(float(row["commit"]) for _, rows in groups for row in rows)

    fig, ax = plt.subplots(figsize=(5.6, 3.55))
    for group_idx, (_, rows) in enumerate(groups):
        for row in rows:
            policy = str(row["policy"])
            xpos = x[group_idx] + offsets[policy]
            ready = float(row["ready"])
            gap = float(row["gap"])
            commit_segment = float(row["commit_segment"])
            commit = float(row["commit"])
            ax.bar(xpos, ready, width, color=READY_COLOR, edgecolor="white", linewidth=0.8)
            ax.bar(xpos, gap, width, bottom=ready, color=GAP_COLOR, edgecolor="white", linewidth=0.8)
            ax.bar(
                xpos,
                commit_segment,
                width,
                bottom=ready + gap,
                color=COMMIT_COLOR,
                edgecolor="white",
                linewidth=0.8,
            )
            ax.text(
                xpos,
                commit + max_commit * 0.018,
                f"{commit:.2f}",
                ha="center",
                va="bottom",
                fontsize=7.8,
                color=ANNOTATION_COLOR,
            )

    ax.set_xticks(x, [shape_label(rows) for _, rows in groups])
    ax.set_ylabel("Time since producer launch (ms)")
    ax.text(
        0.045,
        -0.070,
        r"$M\times N\times K$",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.4,
        color=ANNOTATION_COLOR,
    )
    ax.set_title(title)
    ax.set_ylim(0.0, max_commit * 1.26)

    for group_idx in range(len(groups)):
        y = max_commit * 1.12
        ax.text(
            x[group_idx] + offsets["uniform"],
            y,
            "Uniform",
            ha="center",
            va="bottom",
            rotation=25,
            fontsize=7.6,
            color=ANNOTATION_COLOR,
        )
        ax.text(
            x[group_idx] + offsets["frontier_first"],
            y,
            "Frontier",
            ha="center",
            va="bottom",
            rotation=25,
            fontsize=7.6,
            color=ANNOTATION_COLOR,
        )

    legend_handles = [
        Patch(facecolor=READY_COLOR, label="Launch to ready"),
        Patch(facecolor=GAP_COLOR, label="Ready to consumer"),
        Patch(facecolor=COMMIT_COLOR, label="Consumer to commit"),
    ]
    ax.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=3, fontsize=8.8)
    fig.subplots_adjust(top=0.86, bottom=0.30, left=0.10, right=0.98)

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
    groups = group_rows(rows)
    plot(groups, output_dir, [fmt.strip() for fmt in args.formats.split(",")], args.title, args.dpi)
    print(f"[rs-frontier-timeline-multi-vertical] input csv: {input_csv}")
    print(f"[rs-frontier-timeline-multi-vertical] figure dir: {output_dir}")


if __name__ == "__main__":
    main()
