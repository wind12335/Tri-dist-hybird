#!/usr/bin/env python3
"""Analyze RS symmetric-memory envelopes for active-window staging.

This script is intentionally analytic. It derives the symmetric-memory
allocation envelope from the final RS implementations instead of trying to
force both variants to execute. That makes it suitable for large shapes where
the non-windowed baseline may be infeasible.

It produces three views for each shape:

1. `windowed_v2_*`
   The actual bounded symmetric-memory envelope implied by the final
   frontier-windowed RS implementation.
2. `logical_no_reuse_*`
   A counterfactual envelope under the same panelized design, but without
   slot reuse. This isolates the active-window mechanism itself.
3. `legacy_old_*`
   The symmetric-memory envelope implied by the legacy `new_3rd` RS path.
   This reflects the real implementation-level baseline used in benchmarking.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import math
from pathlib import Path


DTYPE_BYTES = {
    "float16": 2,
    "bfloat16": 2,
    "float32": 4,
}

SIGNAL_BYTES = 8  # utils.NVSHMEM_SIGNAL_DTYPE == torch.int64


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shapes",
        type=str,
        required=True,
        help="Comma-separated shapes like 8192x29568x8192,16384x53248x16384",
    )
    parser.add_argument("--world_size", type=int, default=4)
    parser.add_argument("--local_world_size", type=int, default=4)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=sorted(DTYPE_BYTES))
    parser.add_argument("--chunk_rows", type=int, default=512)
    parser.add_argument("--target_chunks_per_rank", type=int, default=2)
    parser.add_argument("--min_chunk_rows", type=int, default=512)
    parser.add_argument("--active_chunk_window", type=int, default=4)
    parser.add_argument("--stage_slots", type=int, default=4)
    parser.add_argument("--n_bands", type=int, default=2)
    parser.add_argument("--frontier_chunks", type=int, default=2)
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Optional explicit output directory. Default: benchmark/rs_active_window_results/<timestamp>",
    )
    return parser.parse_args()


def parse_shape_list(text: str) -> list[tuple[int, int, int]]:
    shapes: list[tuple[int, int, int]] = []
    for raw in text.split(","):
        raw = raw.strip().lower().replace(" ", "")
        if not raw:
            continue
        parts = raw.split("x")
        if len(parts) != 3:
            raise ValueError(f"invalid shape '{raw}', expected MxNxK")
        m, n, k = map(int, parts)
        shapes.append((m, n, k))
    if not shapes:
        raise ValueError("no valid shapes provided")
    return shapes


def round_up(x: int, align: int) -> int:
    return ((x + align - 1) // align) * align


def auto_chunk_rows(max_m_per_rank: int, target_chunks_per_rank: int, min_chunk_rows: int, align: int = 256) -> int:
    rows = math.ceil(max_m_per_rank / target_chunks_per_rank)
    rows = max(rows, min_chunk_rows)
    rows = min(rows, max_m_per_rank)
    rows = round_up(rows, align)
    return min(rows, max_m_per_rank)


def bytes_to_gib(nbytes: int) -> float:
    return nbytes / float(1024**3)


def compute_row(
    *,
    M: int,
    N: int,
    K: int,
    world_size: int,
    local_world_size: int,
    dtype: str,
    chunk_rows: int,
    target_chunks_per_rank: int,
    min_chunk_rows: int,
    active_chunk_window: int,
    stage_slots: int,
    n_bands: int,
    frontier_chunks: int,
) -> dict[str, object]:
    if world_size != local_world_size:
        raise ValueError("this analysis currently assumes single-node RS, so world_size must equal local_world_size")
    if M % world_size != 0:
        raise ValueError(f"M={M} must be divisible by world_size={world_size}")

    dtype_bytes = DTYPE_BYTES[dtype]
    m_per_rank = M // world_size
    effective_chunk_rows = chunk_rows if chunk_rows > 0 else auto_chunk_rows(
        m_per_rank,
        target_chunks_per_rank=target_chunks_per_rank,
        min_chunk_rows=min_chunk_rows,
    )
    num_chunks = math.ceil(m_per_rank / effective_chunk_rows)
    effective_active_window = max(1, min(active_chunk_window, num_chunks))
    effective_n_bands = max(1, min(n_bands, N))
    effective_stage_slots = max(1, min(stage_slots, num_chunks * effective_n_bands))
    max_band_cols = math.ceil(N / effective_n_bands)

    # Final v5 windowed RS: symmetric footprint is bounded by active_chunk_window.
    windowed_scatter_rows = effective_active_window * effective_n_bands * local_world_size * effective_chunk_rows
    windowed_scatter_bytes = windowed_scatter_rows * max_band_cols * dtype_bytes
    windowed_arrival_flag_bytes = local_world_size * effective_active_window * effective_n_bands * SIGNAL_BYTES
    windowed_free_flag_bytes = effective_active_window * effective_n_bands * SIGNAL_BYTES
    windowed_total_bytes = (
        windowed_scatter_bytes
        + windowed_arrival_flag_bytes
        + windowed_free_flag_bytes
    )

    # Counterfactual no-reuse version under the same panelized design.
    # This is equivalent to giving every logical chunk its own physical slot.
    no_reuse_slots = num_chunks
    logical_no_reuse_scatter_rows = no_reuse_slots * effective_n_bands * local_world_size * effective_chunk_rows
    logical_no_reuse_scatter_bytes = logical_no_reuse_scatter_rows * max_band_cols * dtype_bytes
    logical_no_reuse_arrival_flag_bytes = local_world_size * no_reuse_slots * effective_n_bands * SIGNAL_BYTES
    logical_no_reuse_free_flag_bytes = no_reuse_slots * effective_n_bands * SIGNAL_BYTES
    logical_no_reuse_total_bytes = (
        logical_no_reuse_scatter_bytes
        + logical_no_reuse_arrival_flag_bytes
        + logical_no_reuse_free_flag_bytes
    )

    # Legacy old path: actual implementation-level symmetric footprint.
    legacy_scatter_bytes = M * N * dtype_bytes
    legacy_rs_per_node_bytes = (M // local_world_size) * N * dtype_bytes
    legacy_p2p_bytes = (M // local_world_size) * N * dtype_bytes
    legacy_signal_bytes = (2 * world_size) * SIGNAL_BYTES
    legacy_arrival_flag_bytes = (local_world_size * num_chunks) * SIGNAL_BYTES
    legacy_gemm_out_bytes = M * N * dtype_bytes
    legacy_total_bytes = (
        legacy_scatter_bytes
        + legacy_rs_per_node_bytes
        + legacy_p2p_bytes
        + legacy_signal_bytes
        + legacy_arrival_flag_bytes
        + legacy_gemm_out_bytes
    )

    reuse_factor_vs_windowed = logical_no_reuse_total_bytes / max(windowed_total_bytes, 1)
    legacy_factor_vs_windowed = legacy_total_bytes / max(windowed_total_bytes, 1)
    window_is_active = num_chunks > effective_active_window

    notes: list[str] = []
    if not window_is_active:
        notes.append(
            "num_chunks<=active_window, so this shape does not exhibit slot reuse; it only shows absolute footprint"
        )
    if effective_chunk_rows != chunk_rows and chunk_rows > 0:
        notes.append(f"stage chunk_rows clipped to {effective_chunk_rows}")

    return {
        "shape_label": f"{M}x{N}x{K}",
        "M": M,
        "N": N,
        "K": K,
        "world_size": world_size,
        "local_world_size": local_world_size,
        "dtype": dtype,
        "dtype_bytes": dtype_bytes,
        "signal_bytes": SIGNAL_BYTES,
        "m_per_rank": m_per_rank,
        "chunk_rows": chunk_rows,
        "effective_chunk_rows": effective_chunk_rows,
        "num_chunks": num_chunks,
        "active_chunk_window": active_chunk_window,
        "effective_active_chunk_window": effective_active_window,
        "stage_slots": stage_slots,
        "effective_stage_slots": effective_stage_slots,
        "n_bands": n_bands,
        "effective_n_bands": effective_n_bands,
        "max_band_cols": max_band_cols,
        "frontier_chunks": frontier_chunks,
        "window_is_active": int(window_is_active),
        "windowed_scatter_bytes": windowed_scatter_bytes,
        "windowed_arrival_flag_bytes": windowed_arrival_flag_bytes,
        "windowed_free_flag_bytes": windowed_free_flag_bytes,
        "windowed_total_bytes": windowed_total_bytes,
        "windowed_total_gib": f"{bytes_to_gib(windowed_total_bytes):.6f}",
        "logical_no_reuse_scatter_bytes": logical_no_reuse_scatter_bytes,
        "logical_no_reuse_arrival_flag_bytes": logical_no_reuse_arrival_flag_bytes,
        "logical_no_reuse_free_flag_bytes": logical_no_reuse_free_flag_bytes,
        "logical_no_reuse_total_bytes": logical_no_reuse_total_bytes,
        "logical_no_reuse_total_gib": f"{bytes_to_gib(logical_no_reuse_total_bytes):.6f}",
        "legacy_scatter_bytes": legacy_scatter_bytes,
        "legacy_rs_per_node_bytes": legacy_rs_per_node_bytes,
        "legacy_p2p_bytes": legacy_p2p_bytes,
        "legacy_signal_bytes": legacy_signal_bytes,
        "legacy_arrival_flag_bytes": legacy_arrival_flag_bytes,
        "legacy_gemm_out_bytes": legacy_gemm_out_bytes,
        "legacy_total_bytes": legacy_total_bytes,
        "legacy_total_gib": f"{bytes_to_gib(legacy_total_bytes):.6f}",
        "logical_no_reuse_factor_vs_windowed": f"{reuse_factor_vs_windowed:.4f}",
        "legacy_factor_vs_windowed": f"{legacy_factor_vs_windowed:.4f}",
        "notes": " | ".join(notes),
    }


def write_csv(rows: list[dict[str, object]], output_csv: Path) -> None:
    if not rows:
        raise ValueError("no rows to write")
    fieldnames = list(rows[0].keys())
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_readme(rows: list[dict[str, object]], output_txt: Path, args: argparse.Namespace) -> None:
    lines = [
        "RS Active-Window Footprint Analysis",
        "",
        f"Generated: {dt.datetime.now().isoformat(timespec='seconds')}",
        f"Shapes: {args.shapes}",
        f"world_size/local_world_size: {args.world_size}/{args.local_world_size}",
        f"dtype: {args.dtype}",
        f"chunk_rows: {args.chunk_rows}",
        f"target_chunks_per_rank: {args.target_chunks_per_rank}",
        f"min_chunk_rows: {args.min_chunk_rows}",
        f"active_chunk_window: {args.active_chunk_window}",
        f"stage_slots: {args.stage_slots}",
        f"n_bands: {args.n_bands}",
        f"frontier_chunks: {args.frontier_chunks}",
        "",
        "Interpretation:",
        "- windowed_v2_*: actual bounded symmetric-memory envelope of the final frontier-windowed implementation.",
        "- logical_no_reuse_*: same panelized design without slot reuse; isolates the active-window mechanism itself.",
        "- legacy_old_*: actual symmetric-memory envelope of the older implementation-level baseline.",
        "",
        "Per-shape notes:",
    ]
    for row in rows:
        note = row["notes"] or "none"
        lines.append(f"- {row['shape_label']}: {note}")
    output_txt.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    shapes = parse_shape_list(args.shapes)
    run_id = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else Path("benchmark/rs_active_window_results") / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        compute_row(
            M=M,
            N=N,
            K=K,
            world_size=args.world_size,
            local_world_size=args.local_world_size,
            dtype=args.dtype,
            chunk_rows=args.chunk_rows,
            target_chunks_per_rank=args.target_chunks_per_rank,
            min_chunk_rows=args.min_chunk_rows,
            active_chunk_window=args.active_chunk_window,
            stage_slots=args.stage_slots,
            n_bands=args.n_bands,
            frontier_chunks=args.frontier_chunks,
        )
        for M, N, K in shapes
    ]

    csv_path = output_dir / "rs_active_window_footprint.csv"
    readme_path = output_dir / "README.txt"
    write_csv(rows, csv_path)
    write_readme(rows, readme_path, args)

    for row in rows:
        print(
            "[footprint] "
            f"{row['shape_label']}: "
            f"windowed={float(row['windowed_total_gib']):.3f} GiB, "
            f"no_reuse={float(row['logical_no_reuse_total_gib']):.3f} GiB, "
            f"legacy_old={float(row['legacy_total_gib']):.3f} GiB, "
            f"reuse_factor={float(row['logical_no_reuse_factor_vs_windowed']):.2f}x, "
            f"legacy_factor={float(row['legacy_factor_vs_windowed']):.2f}x"
        )
        if row["notes"]:
            print(f"  note: {row['notes']}")

    print(f"[csv] {csv_path}")
    print(f"[readme] {readme_path}")


if __name__ == "__main__":
    main()
