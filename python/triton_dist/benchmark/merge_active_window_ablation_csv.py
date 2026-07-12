#!/usr/bin/env python3
"""Merge per-policy active-window ablation CSV files into one summary CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", type=str, required=True, help="Directory containing per-shape child CSV files")
    parser.add_argument("--output_csv", type=str, required=True, help="Path to the merged summary CSV")
    return parser.parse_args()


def read_single_row(csv_path: Path) -> dict[str, str]:
    with csv_path.open("r", newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 1:
        raise RuntimeError(f"expected exactly one row in {csv_path}, got {len(rows)}")
    return rows[0]


def policy_and_shape_from_name(path: Path) -> tuple[str, str]:
    name = path.stem
    prefix = "active_window_"
    if not name.startswith(prefix):
        raise ValueError(f"unexpected filename format: {path.name}")
    rest = name[len(prefix):]
    for policy in ("with_active_window", "unbounded_window"):
        policy_prefix = policy + "_"
        if rest.startswith(policy_prefix):
            shape_key = rest[len(policy_prefix):]
            return policy, shape_key
    raise ValueError(f"could not infer policy from filename: {path.name}")


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    output_csv = Path(args.output_csv)

    if not raw_dir.exists():
        raise FileNotFoundError(f"raw_dir does not exist: {raw_dir}")

    grouped: dict[str, dict[str, Path]] = {}
    for csv_path in sorted(raw_dir.glob("active_window_*.csv")):
        policy, shape_key = policy_and_shape_from_name(csv_path)
        grouped.setdefault(shape_key, {})[policy] = csv_path

    if not grouped:
        raise RuntimeError(f"no active_window_*.csv files found under {raw_dir}")

    merged_rows: list[dict[str, str]] = []
    for shape_key in sorted(grouped):
        per_shape = grouped[shape_key]
        if "with_active_window" not in per_shape or "unbounded_window" not in per_shape:
            missing = {"with_active_window", "unbounded_window"} - set(per_shape)
            raise RuntimeError(f"shape {shape_key} is missing policy files: {sorted(missing)}")

        with_row = read_single_row(per_shape["with_active_window"])
        without_row = read_single_row(per_shape["unbounded_window"])

        merged_rows.append({
            "Model": f"{with_row['M']}x{with_row['N']}x{with_row['K']}",
            "M": with_row["M"],
            "N": with_row["N"],
            "K": with_row["K"],
            "torch_total_ms": with_row["torch_total_ms"],
            "torch_gemm_only_ms": with_row["torch_gemm_only_ms"],
            "torch_rs_only_ms": with_row["torch_rs_only_ms"],
            "windowed_total_ms": with_row["windowed_total_ms"],
            "windowed_gemm_only_ms": with_row["windowed_gemm_only_ms"],
            "windowed_rs_only_ms": with_row["windowed_rs_only_ms"],
            "windowed_internal_overlap_ratio": with_row["windowed_internal_overlap_ratio"],
            "windowed_symmetric_staging_gib": with_row["windowed_symmetric_staging_gib"],
            "windowed_num_chunks": with_row["windowed_num_chunks"],
            "windowed_active_chunk_window": with_row["windowed_active_chunk_window"],
            "unbounded_total_ms": without_row["unbounded_total_ms"],
            "unbounded_gemm_only_ms": without_row["unbounded_gemm_only_ms"],
            "unbounded_rs_only_ms": without_row["unbounded_rs_only_ms"],
            "unbounded_internal_overlap_ratio": without_row["unbounded_internal_overlap_ratio"],
            "unbounded_symmetric_staging_gib": without_row["unbounded_symmetric_staging_gib"],
            "unbounded_num_chunks": without_row["unbounded_num_chunks"],
            "unbounded_active_chunk_window": without_row["unbounded_active_chunk_window"],
            "window_speedup_vs_unbounded": f"{float(without_row['unbounded_total_ms']) / float(with_row['windowed_total_ms']):.4f}",
            "window_latency_ratio_vs_unbounded": f"{float(with_row['windowed_total_ms']) / float(without_row['unbounded_total_ms']):.4f}",
            "window_symmetric_ratio_vs_unbounded": f"{float(with_row['windowed_symmetric_staging_gib']) / max(float(without_row['unbounded_symmetric_staging_gib']), 1e-12):.4f}",
        })

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(merged_rows[0].keys())
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in merged_rows:
            writer.writerow(row)

    print(f"[summary] {output_csv}")
    print(f"[shapes] {len(merged_rows)}")


if __name__ == "__main__":
    main()
