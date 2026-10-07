#!/usr/bin/env python3
"""Compare exhaustive vs two-stage RS-GEMM search strategies."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
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
    parser.add_argument(
        "--candidate_plan_csv",
        type=str,
        default="",
        help="Frozen candidate-plan CSV that each strategy must verify before launching benchmarks.",
    )
    parser.add_argument(
        "--expected_candidate_count",
        type=int,
        default=0,
        help="Optional fail-closed candidate-space count assertion forwarded to each strategy.",
    )
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument(
        "--allow_partial_results",
        action="store_true",
        default=False,
        help="Write an incomplete comparison summary with exit code 0 when a strategy is interrupted or fails.",
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
        shapes.append(tuple(map(int, parts)))
    if not shapes:
        raise ValueError("no valid shapes provided")
    return shapes


def append_bool_flag(cmd: list[str], flag: str, value: bool) -> None:
    cmd.append(flag if value else f"--no-{flag[2:]}")


def build_search_cmd(args: argparse.Namespace,
                     shape: tuple[int, int, int],
                     strategy: str,
                     progress_csv: Path,
                     status_json: Path) -> list[str]:
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
        "--candidate_plan_csv",
        args.candidate_plan_csv,
        "--expected_candidate_count",
        str(args.expected_candidate_count),
        "--progress_csv",
        str(progress_csv),
        "--status_json",
        str(status_json),
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


def read_status_json(path: Path) -> dict[str, object] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[compare] cannot parse status record {path}: {exc}", file=sys.stderr, flush=True)
        return None
    return payload if isinstance(payload, dict) else None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fin:
        for chunk in iter(lambda: fin.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_search_output(text: str,
                        strategy: str,
                        status_record: dict[str, object] | None = None) -> dict[str, object]:
    lines = text.splitlines()
    evaluated_candidate_runs = len(CANDIDATE_LINE_RE.findall(text))
    verify_runs = len(VERIFY_LINE_RE.findall(text))

    best_verified = parse_best_block(lines, "[search] Verified Top Candidates")
    best_coarse = parse_best_block(lines, "[search] Top Candidates")
    best_entries = best_verified or best_coarse
    best_source = "verified" if best_verified else "coarse"
    best = best_entries[0] if best_entries else None
    if best is None and status_record is not None:
        candidate = status_record.get("best_completed_coarse")
        if isinstance(candidate, dict):
            best = candidate
            best_source = "partial_coarse"

    summary: dict[str, object] = {
        "strategy": strategy,
        "evaluated_candidate_runs": evaluated_candidate_runs,
        "verify_runs": verify_runs,
        "total_search_runs": evaluated_candidate_runs + verify_runs,
        "best_source": best_source if best is not None else "",
        "best_max_total_ms": None,
        "best_mean_total_ms": None,
        "best_speedup_mean": None,
        "best_speedup_min": None,
        "best_overlap_mean": None,
        "best_lead_ratio": None,
    }
    for field in BEST_CONFIG_FIELDS:
        summary[f"best_{field}"] = None
    if best is not None:
        metric_keys = {
            "best_max_total_ms": ("max_total_ms", "v2_total_ms_max"),
            "best_mean_total_ms": ("mean_total_ms", "v2_total_ms_mean"),
            "best_speedup_mean": ("speedup_mean", "v2_speedup_vs_torch_mean"),
            "best_speedup_min": ("speedup_min", "v2_speedup_vs_torch_min"),
            "best_overlap_mean": ("overlap_mean", "v2_internal_overlap_mean"),
            "best_lead_ratio": ("lead_ratio",),
        }
        for output_key, source_keys in metric_keys.items():
            for source_key in source_keys:
                if source_key in best:
                    summary[output_key] = best[source_key]
                    break
        for field in BEST_CONFIG_FIELDS:
            summary[f"best_{field}"] = best.get(field)

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

    if status_record is not None:
        for key in [
            "candidate_space_total",
            "attempted_candidate_runs",
            "successful_candidate_runs",
            "failed_candidate_runs",
            "timeout_candidate_runs",
            "correctness_failure_markers",
            "verification_runs",
            "successful_verification_runs",
            "failed_verification_runs",
            "planned_coarse_runs",
        ]:
            if key in status_record:
                summary[key] = status_record[key]
        summary["evaluated_candidate_runs"] = status_record.get(
            "attempted_candidate_runs", summary["evaluated_candidate_runs"]
        )
        summary["verify_runs"] = status_record.get("verification_runs", summary["verify_runs"])
        summary["total_search_runs"] = int(summary["evaluated_candidate_runs"]) + int(summary["verify_runs"])
        summary["completion_status"] = status_record.get("completion_status", "unknown")
        summary["failure_reason"] = status_record.get("failure_reason", "")
    else:
        summary["completion_status"] = "unknown"
        summary["failure_reason"] = ""

    return summary


def exact_match(row: dict[str, object], lhs_prefix: str, rhs_prefix: str) -> int | str:
    fields = [f"{prefix}_{field}" for prefix in [lhs_prefix, rhs_prefix] for field in BEST_CONFIG_FIELDS]
    if any(row.get(field) in (None, "") for field in fields):
        return ""
    return int(all(int(row[f"{lhs_prefix}_{field}"]) == int(row[f"{rhs_prefix}_{field}"]) for field in BEST_CONFIG_FIELDS))


def structural_match(row: dict[str, object], lhs_prefix: str, rhs_prefix: str) -> int | str:
    fields = [f"{prefix}_{field}" for prefix in [lhs_prefix, rhs_prefix] for field in STRUCTURAL_FIELDS]
    if any(row.get(field) in (None, "") for field in fields):
        return ""
    return int(all(int(row[f"{lhs_prefix}_{field}"]) == int(row[f"{rhs_prefix}_{field}"]) for field in STRUCTURAL_FIELDS))


def format_number(value: object, digits: int = 4) -> str:
    if value is None or value == "":
        return ""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return ""
    return "" if not math.isfinite(numeric) else f"{numeric:.{digits}f}"


def ratio_or_blank(numerator: object, denominator: object) -> str:
    try:
        lhs = float(numerator)
        rhs = float(denominator)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(lhs) or not math.isfinite(rhs) or rhs <= 0:
        return ""
    return f"{lhs / rhs:.4f}"


def main() -> None:
    args = parse_args()
    shapes = parse_shape_list(args.shapes)
    run_id = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else Path("benchmark/search_compare_runs") / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_plan_path = Path(args.candidate_plan_csv) if args.candidate_plan_csv else None
    if candidate_plan_path is not None and not candidate_plan_path.is_file():
        raise SystemExit(f"[compare] candidate plan does not exist: {candidate_plan_path}")
    candidate_plan_sha256 = sha256_file(candidate_plan_path) if candidate_plan_path is not None else ""
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    state_dir = output_dir / "strategy_state"
    state_dir.mkdir(parents=True, exist_ok=True)

    merged_rows: list[dict[str, object]] = []
    has_incomplete_strategy = False
    for shape in shapes:
        M, N, K = shape
        shape_label = f"{M}x{N}x{K}"
        print(f"[compare] shape={shape_label}", flush=True)
        per_strategy: dict[str, dict[str, object]] = {}
        for strategy in ["exhaustive", "two_stage"]:
            progress_csv = state_dir / f"{shape_label}_{strategy}_candidate_attempts.csv"
            status_json = state_dir / f"{shape_label}_{strategy}_status.json"
            cmd = build_search_cmd(args, shape, strategy, progress_csv, status_json)
            print(f"[compare] run({strategy}): {' '.join(cmd)}", flush=True)
            returncode, output, elapsed = run_and_capture(cmd, quiet=args.quiet_subprocess)
            log_path = logs_dir / f"{shape_label}_{strategy}.log"
            log_path.write_text(output, encoding="utf-8")
            status_record = read_status_json(status_json)
            parsed = parse_search_output(output, strategy, status_record)
            parsed["search_wall_time_sec"] = elapsed
            parsed["child_returncode"] = returncode
            parsed["log_path"] = str(log_path.relative_to(output_dir))
            parsed["log_sha256"] = sha256_file(log_path)
            parsed["progress_csv_path"] = str(progress_csv.relative_to(output_dir))
            parsed["status_json_path"] = str(status_json.relative_to(output_dir))
            if status_record is None:
                parsed["completion_status"] = "completed" if returncode == 0 else "failed"
                parsed["failure_reason"] = "missing_status_record" if returncode != 0 else ""
            elif returncode != 0 and parsed["completion_status"] == "running":
                parsed["completion_status"] = "interrupted"
                parsed["failure_reason"] = "child_process_ended_before_status_finalization"
            if parsed["completion_status"] != "completed":
                has_incomplete_strategy = True
            per_strategy[strategy] = parsed

        ex = per_strategy["exhaustive"]
        ts = per_strategy["two_stage"]
        row: dict[str, object] = {
            "Model": shape_label,
            "M": M,
            "N": N,
            "K": K,
            "candidate_plan_csv": str(candidate_plan_path) if candidate_plan_path is not None else "",
            "candidate_plan_sha256": candidate_plan_sha256,
        }

        for prefix, result in [("exhaustive", ex), ("two_stage", ts)]:
            row[f"{prefix}_completion_status"] = result["completion_status"]
            row[f"{prefix}_failure_reason"] = result["failure_reason"]
            row[f"{prefix}_child_returncode"] = result["child_returncode"]
            row[f"{prefix}_log_path"] = result["log_path"]
            row[f"{prefix}_log_sha256"] = result["log_sha256"]
            row[f"{prefix}_progress_csv_path"] = result["progress_csv_path"]
            row[f"{prefix}_status_json_path"] = result["status_json_path"]
            row[f"{prefix}_search_wall_time_sec"] = format_number(result["search_wall_time_sec"])
            for field in [
                "candidate_space_total",
                "planned_coarse_runs",
                "evaluated_candidate_runs",
                "successful_candidate_runs",
                "failed_candidate_runs",
                "timeout_candidate_runs",
                "correctness_failure_markers",
                "verify_runs",
                "successful_verification_runs",
                "failed_verification_runs",
                "total_search_runs",
            ]:
                row[f"{prefix}_{field}"] = result.get(field, "")
            row[f"{prefix}_best_source"] = result["best_source"]
            for field in [
                "best_max_total_ms",
                "best_mean_total_ms",
                "best_speedup_mean",
                "best_speedup_min",
                "best_overlap_mean",
                "best_lead_ratio",
            ]:
                row[f"{prefix}_{field}"] = format_number(result[field])
            for field in BEST_CONFIG_FIELDS:
                row[f"{prefix}_{field}"] = result[f"best_{field}"]

        both_complete = ex["completion_status"] == "completed" and ts["completion_status"] == "completed"
        row["comparison_status"] = "complete" if both_complete else "incomplete"
        row["two_stage_latency_ratio_vs_exhaustive"] = (
            ratio_or_blank(ts["best_max_total_ms"], ex["best_max_total_ms"]) if both_complete else ""
        )
        row["two_stage_eval_ratio_vs_exhaustive"] = (
            ratio_or_blank(ts["total_search_runs"], ex["total_search_runs"]) if both_complete else ""
        )
        row["two_stage_time_ratio_vs_exhaustive"] = (
            ratio_or_blank(ts["search_wall_time_sec"], ex["search_wall_time_sec"]) if both_complete else ""
        )

        row["exact_match"] = exact_match(row, "exhaustive", "two_stage")
        row["structural_match"] = structural_match(row, "exhaustive", "two_stage")
        merged_rows.append(row)

    summary_csv = output_dir / "search_compare_summary.csv"
    fieldnames: list[str] = []
    for row in merged_rows:
        for field in row:
            if field not in fieldnames:
                fieldnames.append(field)
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in merged_rows:
            writer.writerow(row)

    print(f"[summary] {summary_csv}", flush=True)
    print(f"[logs] {logs_dir}", flush=True)
    print(f"[strategy_state] {state_dir}", flush=True)
    if has_incomplete_strategy and not args.allow_partial_results:
        raise SystemExit(
            "[compare] one or more strategies are incomplete; the CSV, per-candidate ledgers, status records, and logs were preserved. "
            "Use --allow_partial_results only when downstream automation must continue with an explicitly incomplete record."
        )


if __name__ == "__main__":
    main()
