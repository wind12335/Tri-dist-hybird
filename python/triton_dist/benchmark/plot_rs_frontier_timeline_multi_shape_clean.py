#!/usr/bin/env python3
"""Clean multi-shape RS frontier timeline stacked bars."""

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
    parser = argparse.ArgumentParser(description="Plot clean multi-shape RS frontier timeline bars.")
    parser.add_argument("--input_csv", required=True, type=str)
    parser.add_argument("--output_dir", default="", type=str)
    parser.add_argument("--formats", default="png,svg", type=str)
    parser.add_argument("--title", default="RS Frontier Timeline Across Shapes", type=str)
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
                "consumer": consumer,
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


def policy_label(policy: str) -> str:
    return "Uniform" if policy == "uniform" else "Frontier-first"


def setup_style(plt) -> None:
    plt.rcParams.update(
        {
            "font.size": 9.6,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.alpha": 0.35,
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
        from matplotlib.patches import Patch
    except Exception as exc:
        raise RuntimeError(f"matplotlib is required for plotting: {exc}") from exc

    setup_style(plt)
    max_commit = max(float(row["commit"]) for _, rows in groups for row in rows)
    fig_height = 1.25 + 1.28 * len(groups)
    fig, axes = plt.subplots(len(groups), 1, figsize=(7.2, fig_height), sharex=True)
    if len(groups) == 1:
        axes = [axes]

    bar_height = 0.34
    for ax, (shape, rows) in zip(axes, groups):
        y_positions = [0, 1]
        ready_vals = [float(row["ready"]) for row in rows]
        gap_vals = [float(row["gap"]) for row in rows]
        commit_vals = [float(row["commit_segment"]) for row in rows]
        commit_end_vals = [float(row["commit"]) for row in rows]

        ax.barh(y_positions, ready_vals, color=READY_COLOR, height=bar_height)
        ax.barh(y_positions, gap_vals, left=ready_vals, color=GAP_COLOR, height=bar_height)
        ax.barh(
            y_positions,
            commit_vals,
            left=[ready_vals[i] + gap_vals[i] for i in range(len(rows))],
            color=COMMIT_COLOR,
            height=bar_height,
        )

        for y, row in zip(y_positions, rows):
            ax.text(
                float(row["commit"]) + max_commit * 0.014,
                y,
                f"{float(row['commit']):.2f} ms",
                va="center",
                ha="left",
                fontsize=7.8,
                color=ANNOTATION_COLOR,
            )

        ax.set_yticks(y_positions, [policy_label(str(row["policy"])) for row in rows])
        ax.invert_yaxis()
        ax.set_title(shape, loc="left", pad=3)
        ax.set_xlim(0.0, max_commit * 1.14)

    axes[-1].set_xlabel("Time since producer launch (ms)")
    fig.suptitle(title, y=0.985, fontsize=12.2)
    legend_handles = [
        Patch(facecolor=READY_COLOR, label="Launch to ready"),
        Patch(facecolor=GAP_COLOR, label="Ready to consumer"),
        Patch(facecolor=COMMIT_COLOR, label="Consumer to commit"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=3, fontsize=8.8)
    fig.subplots_adjust(top=0.90, bottom=0.16, left=0.16, right=0.94, hspace=0.52)

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
    print(f"[rs-frontier-timeline-multi-clean] input csv: {input_csv}")
    print(f"[rs-frontier-timeline-multi-clean] figure dir: {output_dir}")


if __name__ == "__main__":
    main()
