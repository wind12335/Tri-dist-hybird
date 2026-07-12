################################################################################
#
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software,
# subject to the following conditions:
#
################################################################################

from __future__ import annotations

import argparse
import csv
from pathlib import Path


READY_COLOR = "#2E86AB"
CONSUMER_COLOR = "#F18F01"
COMMIT_COLOR = "#A23B72"
LINE_COLOR = "#97A3AF"


def parse_args():
    parser = argparse.ArgumentParser(description="Plot the RS frontier event timeline.")
    parser.add_argument("--input_csv", required=True, type=str)
    parser.add_argument("--output_dir", default="", type=str)
    parser.add_argument("--formats", default="png,svg", type=str)
    parser.add_argument("--title", default="", type=str)
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


def normalize_policy(value: str) -> str:
    text = value.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "uniform": "uniform",
        "average": "uniform",
        "avg": "uniform",
        "frontier_first": "frontier_first",
        "frontier": "frontier_first",
    }
    if text not in aliases:
        raise ValueError(f"unsupported policy label: {value}")
    return aliases[text]


def label_for_policy(policy: str) -> str:
    return {"uniform": "Uniform", "frontier_first": "Frontier-first"}[policy]


def load_rows(input_csv: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with open(input_csv, "r", encoding="utf-8-sig", newline="") as fin:
        reader = csv.DictReader(fin)
        for raw in reader:
            policy_raw = raw.get("policy") or raw.get("schedule_policy") or raw.get("mode") or ""
            if not str(policy_raw).strip():
                continue
            row = {
                "policy": normalize_policy(str(policy_raw)),
                "ready": safe_float(raw.get("first_panel_ready_ts_ms")),
                "consumer": safe_float(raw.get("first_consumer_start_ts_ms")),
                "commit": safe_float(raw.get("first_output_commit_ts_ms")),
            }
            if row["ready"] is None or row["consumer"] is None or row["commit"] is None:
                raise RuntimeError("missing timeline columns in input CSV")
            rows.append(row)
    if not rows:
        raise RuntimeError("no valid rows found in input CSV")
    rows.sort(key=lambda row: 0 if row["policy"] == "uniform" else 1)
    return rows


def setup_style(plt):
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "grid.linewidth": 0.7,
            "legend.frameon": False,
            "savefig.bbox": "tight",
        }
    )


def plot(rows: list[dict[str, object]], output_dir: Path, formats: list[str], title: str, dpi: int):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except Exception as exc:
        raise RuntimeError(f"matplotlib is required for plotting: {exc}") from exc

    setup_style(plt)
    fig, ax = plt.subplots(figsize=(7.2, 2.9))

    y_positions = [1, 0]
    labels = [label_for_policy(str(row["policy"])) for row in rows]

    for y, row in zip(y_positions, rows):
        xs = [float(row["ready"]), float(row["consumer"]), float(row["commit"])]
        ax.plot(xs, [y, y, y], color=LINE_COLOR, linewidth=1.5, zorder=1)
        ax.scatter(xs[0], y, s=70, color=READY_COLOR, marker="o", zorder=3)
        ax.scatter(xs[1], y, s=78, color=CONSUMER_COLOR, marker="s", zorder=3)
        ax.scatter(xs[2], y, s=82, color=COMMIT_COLOR, marker="D", zorder=3)

    ax.set_yticks(y_positions, labels)
    ax.set_xlabel("Time since producer launch (ms)")
    if title.strip():
        ax.set_title(title)
    xmax = max(float(row["commit"]) for row in rows) * 1.08
    ax.set_xlim(0.0, max(1.0, xmax))
    ax.set_ylim(-0.6, 1.6)

    legend_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=READY_COLOR, markersize=7, label="Ready"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor=CONSUMER_COLOR, markersize=7, label="Consumer"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor=COMMIT_COLOR, markersize=7, label="Commit"),
    ]
    ax.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, fontsize=9)
    fig.subplots_adjust(top=0.92 if title.strip() else 0.88, bottom=0.28, left=0.15, right=0.98)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = "rs_frontier_event_timeline"
    for fmt in formats:
        fmt = fmt.strip().lower()
        if not fmt:
            continue
        fig.savefig(output_dir / f"{stem}.{fmt}", dpi=dpi if fmt == "png" else None)


def default_output_dir(input_csv: Path) -> Path:
    return input_csv.resolve().parent


def main():
    args = parse_args()
    input_csv = Path(args.input_csv)
    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir(input_csv)
    rows = load_rows(input_csv)
    plot(
        rows=rows,
        output_dir=output_dir,
        formats=[fmt.strip() for fmt in args.formats.split(",")],
        title=args.title,
        dpi=args.dpi,
    )
    print(f"[rs-frontier-event-timeline] input csv: {input_csv}")
    print(f"[rs-frontier-event-timeline] figure dir: {output_dir}")


if __name__ == "__main__":
    main()
