#!/usr/bin/env python3
"""Driver for multi-shape active-window ablation runs.

This wrapper avoids running `with_active_window` and `unbounded_window`
back-to-back inside the same NVSHMEM process, because that pattern can leave
stale peer tensor views when the symmetric allocation size changes.

Instead, it launches independent `torchrun` jobs per (shape, policy), then
merges their CSV outputs into one summary table suitable for plotting.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--nproc_per_node', type=int, required=True)
    parser.add_argument('--shapes', type=str, required=True)
    parser.add_argument('--dtype', type=str, default='bfloat16')
    parser.add_argument('--iters', type=int, default=10)
    parser.add_argument('--warmup_iters', type=int, default=5)
    parser.add_argument('--chunk_rows', type=int, default=256)
    parser.add_argument('--target_chunks_per_rank', type=int, default=2)
    parser.add_argument('--min_chunk_rows', type=int, default=256)
    parser.add_argument('--active_chunk_window', type=int, default=4)
    parser.add_argument('--stage_slots', type=int, default=4)
    parser.add_argument('--steady_sms', type=int, default=8)
    parser.add_argument('--tail_sms', type=int, default=20)
    parser.add_argument('--tail_chunk_window', type=int, default=1)
    parser.add_argument('--comm_lanes', type=int, default=2)
    parser.add_argument('--n_bands', type=int, default=2)
    parser.add_argument('--frontier_chunks', type=int, default=2)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--autotune', action='store_true', default=False)
    parser.add_argument('--local_seed_direct', action='store_true', default=False)
    return parser.parse_args()


def parse_shape_list(text: str) -> list[tuple[int, int, int]]:
    shapes = []
    for raw in text.split(','):
        raw = raw.strip().lower().replace(' ', '')
        if not raw:
            continue
        parts = raw.split('x')
        if len(parts) != 3:
            raise ValueError(f"invalid shape '{raw}', expected MxNxK")
        shapes.append(tuple(map(int, parts)))
    if not shapes:
        raise ValueError('no valid shapes provided')
    return shapes


def read_single_row(csv_path: Path) -> dict[str, str]:
    with csv_path.open('r', newline='') as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 1:
        raise RuntimeError(f'expected exactly one row in {csv_path}, got {len(rows)}')
    return rows[0]


def build_cmd(args: argparse.Namespace, shape: tuple[int, int, int], policy: str, output_csv: Path) -> list[str]:
    M, N, K = shape
    cmd = [
        'torchrun',
        '--nproc_per_node', str(args.nproc_per_node),
        'benchmark/bench_active_window_ablation_gemmrs.py',
        '--M', str(M),
        '--N', str(N),
        '--K', str(K),
        '--dtype', args.dtype,
        '--iters', str(args.iters),
        '--warmup_iters', str(args.warmup_iters),
        '--chunk_rows', str(args.chunk_rows),
        '--target_chunks_per_rank', str(args.target_chunks_per_rank),
        '--min_chunk_rows', str(args.min_chunk_rows),
        '--active_chunk_window', str(args.active_chunk_window),
        '--stage_slots', str(args.stage_slots),
        '--steady_sms', str(args.steady_sms),
        '--tail_sms', str(args.tail_sms),
        '--tail_chunk_window', str(args.tail_chunk_window),
        '--comm_lanes', str(args.comm_lanes),
        '--n_bands', str(args.n_bands),
        '--frontier_chunks', str(args.frontier_chunks),
        '--window_policy', policy,
        '--dump_csv',
        '--output_csv', str(output_csv),
    ]
    if args.autotune:
        cmd.append('--autotune')
    if args.local_seed_direct:
        cmd.append('--local_seed_direct')
    return cmd


def main() -> None:
    args = parse_args()
    shapes = parse_shape_list(args.shapes)
    run_id = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(args.output_dir) if args.output_dir else Path('benchmark/active_window_ablation_runs') / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / 'raw_child_csv'
    raw_dir.mkdir(parents=True, exist_ok=True)

    merged_rows: list[dict[str, str]] = []

    for shape in shapes:
        M, N, K = shape
        shape_label = f'{M}x{N}x{K}'
        print(f'[driver] shape={shape_label}', flush=True)
        policy_rows: dict[str, dict[str, str]] = {}
        for policy in ['with_active_window', 'unbounded_window']:
            policy_csv = raw_dir / f'active_window_{policy}_{M}_{N}_{K}.csv'
            cmd = build_cmd(args, shape, policy, policy_csv)
            print('[driver] run:', ' '.join(cmd), flush=True)
            subprocess.run(cmd, check=True)
            policy_rows[policy] = read_single_row(policy_csv)

        with_row = policy_rows['with_active_window']
        without_row = policy_rows['unbounded_window']
        merged_rows.append({
            'Model': shape_label,
            'M': with_row['M'],
            'N': with_row['N'],
            'K': with_row['K'],
            'torch_total_ms': with_row['torch_total_ms'],
            'torch_gemm_only_ms': with_row['torch_gemm_only_ms'],
            'torch_rs_only_ms': with_row['torch_rs_only_ms'],
            'windowed_total_ms': with_row['windowed_total_ms'],
            'windowed_gemm_only_ms': with_row['windowed_gemm_only_ms'],
            'windowed_rs_only_ms': with_row['windowed_rs_only_ms'],
            'windowed_internal_overlap_ratio': with_row['windowed_internal_overlap_ratio'],
            'windowed_symmetric_staging_gib': with_row['windowed_symmetric_staging_gib'],
            'windowed_num_chunks': with_row['windowed_num_chunks'],
            'windowed_active_chunk_window': with_row['windowed_active_chunk_window'],
            'unbounded_total_ms': without_row['unbounded_total_ms'],
            'unbounded_gemm_only_ms': without_row['unbounded_gemm_only_ms'],
            'unbounded_rs_only_ms': without_row['unbounded_rs_only_ms'],
            'unbounded_internal_overlap_ratio': without_row['unbounded_internal_overlap_ratio'],
            'unbounded_symmetric_staging_gib': without_row['unbounded_symmetric_staging_gib'],
            'unbounded_num_chunks': without_row['unbounded_num_chunks'],
            'unbounded_active_chunk_window': without_row['unbounded_active_chunk_window'],
            'window_speedup_vs_unbounded': f"{float(without_row['unbounded_total_ms']) / float(with_row['windowed_total_ms']):.4f}",
            'window_latency_ratio_vs_unbounded': f"{float(with_row['windowed_total_ms']) / float(without_row['unbounded_total_ms']):.4f}",
            'window_symmetric_ratio_vs_unbounded': f"{float(with_row['windowed_symmetric_staging_gib']) / max(float(without_row['unbounded_symmetric_staging_gib']), 1e-12):.4f}",
        })

    summary_csv = output_dir / 'active_window_ablation_summary.csv'
    fieldnames = list(merged_rows[0].keys())
    with summary_csv.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in merged_rows:
            writer.writerow(row)

    print(f'[summary] {summary_csv}', flush=True)
    print(f'[raw_child_csv] {raw_dir}', flush=True)


if __name__ == '__main__':
    main()
