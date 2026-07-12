#!/usr/bin/env python3
"""Compare exhaustive vs two-stage RS-GEMM search strategies."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SEARCH_SCRIPT = ROOT / "python" / "triton_dist" / "benchmark" / "bench_3rdv5_frontier_windowed_panel_gemmrs_search.py"

EXHAUSTIVE_TOTAL_RE = re.compile(r"\[search\] exhaustive search: (\d+) candidates")
TWO_STAGE_TOTAL_RE = re.compile(
    r"\[search\] structural candidates total=(\d+), buckets=(\d+), .*stage1 structural candidates=(\d+), "
    r"stage2 sms_pairs=(\d+), stage2 refine_topk=(\d+), .*verify_topk=(\d+)"
)
CANDIDATE_LINE_RE = re.compile(r"\[search\]\[(coarse|stage1|stage2)\] candidate ")
VERIFY_LINE_RE = re.compile(r"\[search\] verifying top candidate #")
TOP_ENTRY_RE = re.compile(
    r"#(?P<rank>\d+): max_total=(?P<max_total>[0-9.]+) ms, "
    r"mean_total=(?P<mean_total>[0-9.]+) ms, "
    r"speedup_mean=(?P<speedup_mean>[0-9.]+), "
    r"speedup_min=(?P<speedup_min>[0-9.]+), "
    r"overlap_mean=(?P<overlap_mean>[0-9.]+)%, "
    r"lead_ratio=(?P<lead_ratio>[0-9.]+) \| "
    r"chunk=(?P<chunk_rows>\d+), "
    r"window=(?P<active_chunk_window>\d+), "
    r"stage=(?P<stage_slots>\d+), "
    r"steady_sms=(?P<steady_sms>\d+), "
    r"tail_sms=(?P<tail_sms>\d+), "
    r"lanes=(?P<comm_lanes>\d+), "
    r"bands=(?P<n_bands>\d+), "
    r"frontier=(?P<frontier_chunks>\d+)"
)

BEST_CONFIG_FIELDS = [
    "chunk_rows",
    "active_chunk_window",
    "stage_slots",
    "steady_sms",
    "tail_sms",
    "comm_lanes",
    "n_bands",
    "frontier_chunks",
]
STRUCTURAL_FIELDS = [
    "chunk_rows",
    "active_chunk_window",
    "stage_slots",
    "comm_lanes",
    "n_bands",
    "frontier_chunks",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two-stage and exhaustive search for RS-GEMM.")
    parser.add_argument("--nproc_per_node", type=int, required=True)
    parser.add_argument("--shapes", type=str, required=True, help="Comma-separated MxNxK shapes")
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16"])
    parser.add_argument("--mode", default="v2", choices=["v2", "all"])
    parser.add_argument("--autotune", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--stage1_no_autotune", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--trans_b", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--local_seed_direct", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--target_chunks_per_rank", type=int, default=2)
    parser.add_argument("--min_chunk_rows", type=int, default=512)
    parser.add_argument("--tail_chunk_window", type=int, default=1)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--warmup_iters", type=int, default=5)
    parser.add_argument("--fast_iters", type=int, default=4)
    parser.add_argument("--fast_warmup_iters", type=int, default=2)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--verify_topk", type=int, default=3)
    parser.add_argument("--structural_topk", type=int, default=12)
    parser.add_argument("--structural_bucket_topk", type=int, default=2)
    parser.add_argument("--fast_budget", type=int, default=50)
    parser.add_argument("--sms_refine_strategy", default="focused", choices=["focused", "full_grid"])
    parser.add_argument("--candidate_timeout_sec", type=float, default=0.0)
    parser.add_argument("--torchrun_bin", default="torchrun")
    parser.add_argument("--shape_preset", default="none", choices=["none", "gpt3_175b_nbands2"])
    parser.add_argument("--quiet_subprocess", action="store_true", default=False)
    parser.add_argument("--search_chunk_rows_list", type=str, default="")
    parser.add_argument("--search_active_chunk_window_list", type=str, default="")
    parser.add_argument("--search_stage_slots_list", type=str, default="")
    parser.add_argument("--search_steady_sms_list", type=str, default="")
    parser.add_argument("--search_tail_sms_list", type=str, default="")
    parser.add_argument("--search_comm_lanes_list", type=str, default="")
    parser.add_argument("--search_n_bands_list", type=str, default="")
    parser.add_argument("--search_frontier_chunks_list", type=str, default="")
    parser.add_argument("--output_dir", type=str, default=None)
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
        shapes.append(tuple(map(int, parts)))
    if not shapes:
        raise ValueError("no valid shapes provided")
    return shapes


def append_bool_flag(cmd: list[str], flag: str, value: bool) -> None:
    cmd.append(flag if value else f"--no-{flag[2:]}")


def build_search_cmd(args: argparse.Namespace, shape: tuple[int, int, int], strategy: str) -> list[str]:
    M, N, K = shape
    cmd = [
        sys.executable,
        str(SEARCH_SCRIPT),
        "--nproc_per_node",
        str(args.nproc_per_node),
        "--M",
        str(M),
        "--N",
        str(N),
        "--K",
        str(K),
        "--dtype",
        args.dtype,
        "--mode",
        args.mode,
        "--iters",
        str(args.iters),
        "--warmup_iters",
        str(args.warmup_iters),
        "--fast_iters",
        str(args.fast_iters),
        "--fast_warmup_iters",
        str(args.fast_warmup_iters),
        "--topk",
        str(args.topk),
        "--verify_topk",
        str(args.verify_topk),
        "--search_strategy",
        strategy,
        "--structural_topk",
        str(args.structural_topk),
        "--structural_bucket_topk",
        str(args.structural_bucket_topk),
        "--fast_budget",
        str(args.fast_budget),
        "--sms_refine_strategy",
        args.sms_refine_strategy,
        "--target_chunks_per_rank",
        str(args.target_chunks_per_rank),
        "--min_chunk_rows",
        str(args.min_chunk_rows),
        "--tail_chunk_window",
        str(args.tail_chunk_window),
        "--candidate_timeout_sec",
        str(args.candidate_timeout_sec),
        "--torchrun_bin",
        args.torchrun_bin,
        "--shape_preset",
        args.shape_preset,
        "--search_chunk_rows_list",
        args.search_chunk_rows_list,
        "--search_active_chunk_window_list",
        args.search_active_chunk_window_list,
        "--search_stage_slots_list",
        args.search_stage_slots_list,
        "--search_steady_sms_list",
        args.search_steady_sms_list,
        "--search_tail_sms_list",
        args.search_tail_sms_list,
        "--search_comm_lanes_list",
        args.search_comm_lanes_list,
        "--search_n_bands_list",
        args.search_n_bands_list,
        "--search_frontier_chunks_list",
        args.search_frontier_chunks_list,
    ]
    append_bool_flag(cmd, "--autotune", args.autotune)
    append_bool_flag(cmd, "--stage1_no_autotune", args.stage1_no_autotune)
    append_bool_flag(cmd, "--trans_b", args.trans_b)
    append_bool_flag(cmd, "--local_seed_direct", args.local_seed_direct)
    if args.quiet_subprocess:
        cmd.append("--quiet_subprocess")
    return cmd


def run_and_capture(cmd: list[str], quiet: bool) -> tuple[int, str, float]:
    start = time.perf_counter()
    process = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
        bufsize=1,
    )
    captured_lines: list[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        captured_lines.append(line)
        if not quiet:
            print(line, end="")
    process.wait()
    elapsed = time.perf_counter() - start
    merged = "".join(captured_lines)
    return process.returncode, merged, elapsed


def parse_best_block(lines: list[str], header: str) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    capture = False
    for line in lines:
        stripped = line.strip()
        if stripped == header:
            capture = True
            continue
        if capture:
            if not stripped.startswith("#"):
                if stripped.startswith("[search]"):
                    break
                continue
            match = TOP_ENTRY_RE.search(stripped)
            if not match:
                continue
            item: dict[str, object] = {
                "rank": int(match.group("rank")),
                "max_total_ms": float(match.group("max_total")),
                "mean_total_ms": float(match.group("mean_total")),
                "speedup_mean": float(match.group("speedup_mean")),
                "speedup_min": float(match.group("speedup_min")),
                "overlap_mean": float(match.group("overlap_mean")) / 100.0,
                "lead_ratio": float(match.group("lead_ratio")),
            }
            for field in BEST_CONFIG_FIELDS:
                item[field] = int(match.group(field))
            entries.append(item)
    return entries


def parse_search_output(text: str, strategy: str) -> dict[str, object]:
    lines = text.splitlines()
    evaluated_candidate_runs = len(CANDIDATE_LINE_RE.findall(text))
    verify_runs = len(VERIFY_LINE_RE.findall(text))

    best_verified = parse_best_block(lines, "[search] Verified Top Candidates")
    best_coarse = parse_best_block(lines, "[search] Top Candidates")
    best_entries = best_verified or best_coarse
    best_source = "verified" if best_verified else "coarse"
    if not best_entries:
        raise RuntimeError("failed to parse best candidate block from search output")
    best = best_entries[0]

    summary: dict[str, object] = {
        "strategy": strategy,
        "evaluated_candidate_runs": evaluated_candidate_runs,
        "verify_runs": verify_runs,
        "total_search_runs": evaluated_candidate_runs + verify_runs,
        "best_source": best_source,
        "best_max_total_ms": best["max_total_ms"],
        "best_mean_total_ms": best["mean_total_ms"],
        "best_speedup_mean": best["speedup_mean"],
        "best_speedup_min": best["speedup_min"],
        "best_overlap_mean": best["overlap_mean"],
        "best_lead_ratio": best["lead_ratio"],
    }
    for field in BEST_CONFIG_FIELDS:
        summary[f"best_{field}"] = best[field]

    if strategy == "exhaustive":
        match = EXHAUSTIVE_TOTAL_RE.search(text)
        if match:
            summary["candidate_space_total"] = int(match.group(1))
    else:
        match = TWO_STAGE_TOTAL_RE.search(text)
        if match:
            summary["structural_candidates_total"] = int(match.group(1))
            summary["bucket_count"] = int(match.group(2))
            summary["stage1_structural_candidates"] = int(match.group(3))
            summary["stage2_sms_pairs"] = int(match.group(4))
            summary["stage2_refine_topk"] = int(match.group(5))
            summary["verify_topk_requested"] = int(match.group(6))

    return summary


def exact_match(row: dict[str, object], lhs_prefix: str, rhs_prefix: str) -> int:
    return int(all(int(row[f"{lhs_prefix}_{field}"]) == int(row[f"{rhs_prefix}_{field}"]) for field in BEST_CONFIG_FIELDS))


def structural_match(row: dict[str, object], lhs_prefix: str, rhs_prefix: str) -> int:
    return int(all(int(row[f"{lhs_prefix}_{field}"]) == int(row[f"{rhs_prefix}_{field}"]) for field in STRUCTURAL_FIELDS))


def main() -> None:
    args = parse_args()
    shapes = parse_shape_list(args.shapes)
    run_id = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else Path("benchmark/search_compare_runs") / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    merged_rows: list[dict[str, object]] = []
    for shape in shapes:
        M, N, K = shape
        shape_label = f"{M}x{N}x{K}"
        print(f"[compare] shape={shape_label}", flush=True)
        per_strategy: dict[str, dict[str, object]] = {}
        for strategy in ["exhaustive", "two_stage"]:
            cmd = build_search_cmd(args, shape, strategy)
            print(f"[compare] run({strategy}): {' '.join(cmd)}", flush=True)
            returncode, output, elapsed = run_and_capture(cmd, quiet=args.quiet_subprocess)
            log_path = logs_dir / f"{shape_label}_{strategy}.log"
            log_path.write_text(output, encoding="utf-8")
            if returncode != 0:
                raise SystemExit(f"[compare] {strategy} search failed for {shape_label}, see {log_path}")
            parsed = parse_search_output(output, strategy)
            parsed["search_wall_time_sec"] = elapsed
            per_strategy[strategy] = parsed

        ex = per_strategy["exhaustive"]
        ts = per_strategy["two_stage"]
        row: dict[str, object] = {
            "Model": shape_label,
            "M": M,
            "N": N,
            "K": K,
            "exhaustive_search_wall_time_sec": f"{float(ex['search_wall_time_sec']):.4f}",
            "exhaustive_evaluated_candidate_runs": ex["evaluated_candidate_runs"],
            "exhaustive_verify_runs": ex["verify_runs"],
            "exhaustive_total_search_runs": ex["total_search_runs"],
            "exhaustive_best_source": ex["best_source"],
            "exhaustive_best_max_total_ms": f"{float(ex['best_max_total_ms']):.4f}",
            "exhaustive_best_mean_total_ms": f"{float(ex['best_mean_total_ms']):.4f}",
            "exhaustive_best_speedup_mean": f"{float(ex['best_speedup_mean']):.4f}",
            "exhaustive_best_speedup_min": f"{float(ex['best_speedup_min']):.4f}",
            "exhaustive_best_overlap_mean": f"{float(ex['best_overlap_mean']):.4f}",
            "exhaustive_best_lead_ratio": f"{float(ex['best_lead_ratio']):.4f}",
            "two_stage_search_wall_time_sec": f"{float(ts['search_wall_time_sec']):.4f}",
            "two_stage_evaluated_candidate_runs": ts["evaluated_candidate_runs"],
            "two_stage_verify_runs": ts["verify_runs"],
            "two_stage_total_search_runs": ts["total_search_runs"],
            "two_stage_best_source": ts["best_source"],
            "two_stage_best_max_total_ms": f"{float(ts['best_max_total_ms']):.4f}",
            "two_stage_best_mean_total_ms": f"{float(ts['best_mean_total_ms']):.4f}",
            "two_stage_best_speedup_mean": f"{float(ts['best_speedup_mean']):.4f}",
            "two_stage_best_speedup_min": f"{float(ts['best_speedup_min']):.4f}",
            "two_stage_best_overlap_mean": f"{float(ts['best_overlap_mean']):.4f}",
            "two_stage_best_lead_ratio": f"{float(ts['best_lead_ratio']):.4f}",
            "two_stage_latency_ratio_vs_exhaustive": f"{float(ts['best_max_total_ms']) / max(float(ex['best_max_total_ms']), 1e-12):.4f}",
            "two_stage_eval_ratio_vs_exhaustive": f"{float(ts['total_search_runs']) / max(float(ex['total_search_runs']), 1e-12):.4f}",
            "two_stage_time_ratio_vs_exhaustive": f"{float(ts['search_wall_time_sec']) / max(float(ex['search_wall_time_sec']), 1e-12):.4f}",
        }
        for field in BEST_CONFIG_FIELDS:
            row[f"exhaustive_{field}"] = ex[f"best_{field}"]
            row[f"two_stage_{field}"] = ts[f"best_{field}"]

        row["exact_match"] = exact_match(row, "exhaustive", "two_stage")
        row["structural_match"] = structural_match(row, "exhaustive", "two_stage")
        merged_rows.append(row)

    summary_csv = output_dir / "search_compare_summary.csv"
    fieldnames = list(merged_rows[0].keys())
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in merged_rows:
            writer.writerow(row)

    print(f"[summary] {summary_csv}", flush=True)
    print(f"[logs] {logs_dir}", flush=True)


if __name__ == "__main__":
    main()
