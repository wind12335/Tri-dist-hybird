#!/usr/bin/env python3
"""Merge per-shape frontier scheduling ablation CSV files.

The benchmark script writes a fixed file name:
csv/perf_frontier_schedule_ablation_gemm_rs_<world>_ranks.csv.
Save each run under a shape-specific name first, then use this helper to build a
multi-shape summary for plotting or tables.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


REQUIRED_COLUMNS = [
    "Model",
    "M",
    "N",
    "K",
    "schedule_policy",
    "torch_total_ms",
    "uniform_total_ms",
    "uniform_internal_overlap_ratio",
    "frontier_total_ms",
    "frontier_internal_overlap_ratio",
    "frontier_speedup_vs_uniform",
    "chunk_rows",
    "active_chunk_window",
    "stage_slots",
    "comm_lanes",
    "n_bands",
    "frontier_chunks",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge per-shape frontier schedule ablation CSVs.")
    parser.add_argument("--inputs", nargs="+", required=True, help="Shape-specific CSV files to merge.")
    parser.add_argument("--output_csv", required=True, help="Merged output CSV.")
    parser.add_argument("--hardware", default="", help="Optional hardware label, e.g. A800-PCIe.")
    parser.add_argument("--autotune", default="", choices=["", "yes", "no"], help="Optional autotune label.")
    parser.add_argument("--run_tag", default="", help="Optional run tag added to every row.")
    return parser.parse_args()


def read_one(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8-sig", newline="") as fin:
        rows = list(csv.DictReader(fin))
    if len(rows) != 1:
        raise RuntimeError(f"{path} should contain exactly one data row, got {len(rows)}")
    row = dict(rows[0])
    missing = [col for col in REQUIRED_COLUMNS if col not in row]
    if missing:
        raise RuntimeError(f"{path} is missing columns: {', '.join(missing)}")
    row["shape"] = f"{row['M']}x{row['N']}x{row['K']}"
    row["source_csv"] = str(path)
    return row


def shape_key(row: dict[str, str]) -> tuple[int, int, int]:
    return (int(row["M"]), int(row["N"]), int(row["K"]))


def main() -> None:
    args = parse_args()
    rows = [read_one(Path(p)) for p in args.inputs]
    rows.sort(key=shape_key)

    seen: set[str] = set()
    for row in rows:
        if row["shape"] in seen:
            raise RuntimeError(f"duplicate shape in inputs: {row['shape']}")
        seen.add(row["shape"])
        row["hardware"] = args.hardware
        row["autotune"] = args.autotune
        row["run_tag"] = args.run_tag

    output = Path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)

    leading_cols = ["shape", "hardware", "autotune", "run_tag", "source_csv"]
    fieldnames = leading_cols + [col for col in rows[0].keys() if col not in leading_cols]
    with output.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[merge-frontier-schedule] wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
