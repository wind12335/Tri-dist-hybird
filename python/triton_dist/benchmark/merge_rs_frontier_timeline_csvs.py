#!/usr/bin/env python3
"""Merge per-shape RS frontier timeline CSV files.

Each collector output contains two rows: ``uniform`` and ``frontier_first``.
This helper adds a shape label and optional provenance columns, then concatenates
all rows into one CSV for multi-shape plotting.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


REQUIRED_COLUMNS = [
    "policy",
    "M",
    "N",
    "K",
    "dtype",
    "iters",
    "warmup_iters",
    "first_panel_ready_ts_ms",
    "first_consumer_start_ts_ms",
    "first_output_commit_ts_ms",
    "chunk_rows",
    "n_bands",
    "active_chunk_window",
    "stage_slots",
    "comm_lanes",
    "frontier_chunks",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge RS frontier timeline CSVs across shapes.")
    parser.add_argument("--inputs", nargs="+", required=True, help="Per-shape rs_frontier_timeline.csv files.")
    parser.add_argument("--output_csv", required=True, help="Merged output CSV.")
    parser.add_argument("--hardware", default="", help="Optional hardware label, e.g. A800-PCIe.")
    parser.add_argument("--autotune", default="", choices=["", "yes", "no"], help="Optional autotune label.")
    parser.add_argument("--run_tag", default="", help="Optional run tag added to every row.")
    return parser.parse_args()


def normalize_policy(policy: str) -> str:
    text = policy.strip().lower().replace("-", "_").replace(" ", "_")
    if text in {"uniform", "frontier_first"}:
        return text
    raise RuntimeError(f"unsupported policy label: {policy!r}")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fin:
        rows = list(csv.DictReader(fin))
    if not rows:
        raise RuntimeError(f"no rows found in {path}")
    missing = [col for col in REQUIRED_COLUMNS if col not in rows[0]]
    if missing:
        raise RuntimeError(f"{path} is missing columns: {', '.join(missing)}")

    policies = {normalize_policy(row["policy"]) for row in rows}
    if policies != {"uniform", "frontier_first"}:
        raise RuntimeError(f"{path} should contain uniform and frontier_first rows, got {sorted(policies)}")

    parsed: list[dict[str, str]] = []
    for row in rows:
        out = dict(row)
        out["policy"] = normalize_policy(row["policy"])
        out["shape"] = f"{row['M']}x{row['N']}x{row['K']}"
        out["source_csv"] = str(path)
        parsed.append(out)
    return parsed


def shape_key(row: dict[str, str]) -> tuple[int, int, int, int]:
    policy_order = 0 if row["policy"] == "uniform" else 1
    return (int(row["M"]), int(row["N"]), int(row["K"]), policy_order)


def main() -> None:
    args = parse_args()
    rows: list[dict[str, str]] = []
    for input_path in args.inputs:
        rows.extend(read_rows(Path(input_path)))
    rows.sort(key=shape_key)

    shape_to_policies: dict[str, set[str]] = {}
    for row in rows:
        shape_to_policies.setdefault(row["shape"], set()).add(row["policy"])
        row["hardware"] = args.hardware
        row["autotune"] = args.autotune
        row["run_tag"] = args.run_tag

    bad_shapes = [shape for shape, policies in shape_to_policies.items() if policies != {"uniform", "frontier_first"}]
    if bad_shapes:
        raise RuntimeError("missing policy pair for shapes: " + ", ".join(bad_shapes))

    output = Path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)

    leading_cols = ["shape", "policy", "hardware", "autotune", "run_tag", "source_csv"]
    fieldnames = leading_cols + [col for col in rows[0].keys() if col not in leading_cols]
    with output.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[merge-rs-frontier-timeline] wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
