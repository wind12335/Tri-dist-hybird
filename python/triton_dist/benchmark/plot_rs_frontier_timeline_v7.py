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
GAP_COLOR = "#F18F01"
COMMIT_COLOR = "#A23B72"


def parse_args():
    parser = argparse.ArgumentParser(description="Plot the RS frontier scheduling timeline.")
    parser.add_argument("--input_csv", required=True, type=str)
    parser.add_argument("--output_dir", default="", type=str)
    parser.add_argument("--formats", default="png,svg", type=str)
    parser.add_argument("--title", default="RS frontier scheduling timeline", type=str)
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


def load_rows(input_csv: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with open(input_csv, "r", encoding="utf-8-sig", newline="") as fin:
        reader = csv.DictReader(fin)
        for raw in reader:
            policy_raw = raw.get("policy") or raw.get("schedule_policy") or raw.get("mode") or ""
            if not str(policy_raw).strip():
                continue
            rows.append(
                {
                    "policy": normalize_policy(str(policy_raw)),
                    "first_panel_ready_ts_ms": safe_float(raw.get("first_panel_ready_ts_ms")),
                    "first_consumer_start_ts_ms": safe_float(raw.get("first_consumer_start_ts_ms")),
                    "first_output_commit_ts_ms": safe_float(raw.get("first_output_commit_ts_ms")),
                }
            )
    if not rows:
        raise RuntimeError("no valid rows found in input CSV")
    return rows


def validate_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    wanted_order = ["uniform", "frontier_first"]
    row_map = {str(row["policy"]): row for row in rows}
    ordered: list[dict[str, object]] = []
    for name in wanted_order:
        row = row_map[name]
        ready = row["first_panel_ready_ts_ms"]
        consumer = row["first_consumer_start_ts_ms"]
        commit = row["first_output_commit_ts_ms"]
        if ready is None or consumer is None or commit is None:
            raise RuntimeError(f"policy={name} is missing required timestamps")
        if not (0.0 <= ready <= consumer <= commit):
            raise RuntimeError(f"policy={name} has non-monotonic timestamps")
        row["segment_ready_ms"] = ready
        row["segment_gap_ms"] = consumer - ready
        row["segment_commit_ms"] = commit - consumer
        ordered.append(row)
    return ordered


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
            "savefig.bbox": "tight",
        }
    )


def label_for_policy(policy: str) -> str:
    return {"uniform": "Uniform", "frontier_first": "Frontier-first"}[policy]


def plot(rows: list[dict[str, object]], output_dir: Path, formats: list[str], title: str, dpi: int):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError(f"matplotlib is required for plotting: {exc}") from exc

    setup_style(plt)
    fig, ax = plt.subplots(figsize=(7.8, 3.5))

    y_positions = [0, 1]
    labels = [label_for_policy(str(row["policy"])) for row in rows]
    ready_vals = [float(row["segment_ready_ms"]) for row in rows]
    gap_vals = [float(row["segment_gap_ms"]) for row in rows]
    commit_vals = [float(row["segment_commit_ms"]) for row in rows]

    bar_height = 0.28
    ax.barh(y_positions, ready_vals, color=READY_COLOR, height=bar_height)
    ax.barh(y_positions, gap_vals, left=ready_vals, color=GAP_COLOR, height=bar_height)
    ax.barh(
        y_positions,
        commit_vals,
        left=[ready_vals[i] + gap_vals[i] for i in range(len(rows))],
        color=COMMIT_COLOR,
        height=bar_height,
    )

    ax.set_yticks(y_positions, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Time since producer launch (ms)")
    ax.set_title(title)
    xmax = max(float(row["first_output_commit_ts_ms"]) for row in rows) * 1.08
    ax.set_xlim(0.0, max(1.0, xmax))

    fig.subplots_adjust(top=0.86, bottom=0.22, left=0.12, right=0.98)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = "rs_frontier_timeline"
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
    rows = validate_rows(load_rows(input_csv))
    plot(
        rows=rows,
        output_dir=output_dir,
        formats=[fmt.strip() for fmt in args.formats.split(",")],
        title=args.title,
        dpi=args.dpi,
    )
    print(f"[rs-frontier-timeline-v7] input csv: {input_csv}")
    print(f"[rs-frontier-timeline-v7] figure dir: {output_dir}")


if __name__ == "__main__":
    main()
