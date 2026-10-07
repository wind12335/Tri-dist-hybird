#!/usr/bin/env python3
"""Run a reproducible multi-point GEMM--RS active-window sweep.

The existing pairwise ablation driver compares ``L=4`` against ``L=8``.  This
driver deliberately keeps that historical protocol untouched and launches a
single-policy active-window benchmark for every ``(shape, L, repetition)``.
Each child is a separate process because changing symmetric allocations inside
one NVSHMEM process can retain stale peer views.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import platform
import shlex
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_RELATIVE_PATH = "benchmark/bench_active_window_ablation_gemmrs.py"
RUN_FIELDS = [
    "run_key",
    "attempt_index",
    "shape",
    "M",
    "N",
    "K",
    "num_chunks",
    "requested_active_chunk_window",
    "effective_active_chunk_window",
    "repetition",
    "latency_ms",
    "torch_latency_ms",
    "symmetric_staging_gib",
    "stage_slots",
    "steady_sms",
    "tail_sms",
    "comm_lanes",
    "n_bands",
    "frontier_chunks",
    "completion_status",
    "returncode",
    "failure_reason",
    "started_at_utc",
    "finished_at_utc",
    "elapsed_sec",
    "raw_csv_path",
    "raw_csv_sha256",
    "log_path",
    "log_sha256",
    "command",
]
AGGREGATE_FIELDS = [
    "shape",
    "M",
    "N",
    "K",
    "num_chunks",
    "active_chunk_window",
    "baseline_active_chunk_window",
    "planned_repetitions",
    "successful_repetitions",
    "failed_repetitions",
    "completion_status",
    "latency_median_ms",
    "latency_mean_ms",
    "latency_std_ms",
    "latency_min_ms",
    "latency_max_ms",
    "baseline_latency_median_ms",
    "latency_ratio_vs_baseline",
    "symmetric_staging_gib",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch independent repeated GEMM--RS active-window measurements and aggregate them."
    )
    parser.add_argument("--nproc_per_node", type=int, required=True)
    parser.add_argument("--shapes", type=str, required=True, help="Comma-separated MxNxK shapes.")
    parser.add_argument("--active_window_values", default="1,2,4,8")
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16"])
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--warmup_iters", type=int, default=5)
    parser.add_argument("--chunk_rows", type=int, default=256)
    parser.add_argument("--target_chunks_per_rank", type=int, default=2)
    parser.add_argument("--min_chunk_rows", type=int, default=256)
    parser.add_argument("--stage_slots", type=int, default=4)
    parser.add_argument("--steady_sms", type=int, default=8)
    parser.add_argument("--tail_sms", type=int, default=20)
    parser.add_argument("--tail_chunk_window", type=int, default=1)
    parser.add_argument("--comm_lanes", type=int, default=2)
    parser.add_argument("--n_bands", type=int, default=2)
    parser.add_argument("--frontier_chunks", type=int, default=2)
    parser.add_argument("--autotune", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--local_seed_direct", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--trans_b", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--candidate_timeout_sec", type=float, default=0.0)
    parser.add_argument("--torchrun_bin", default="torchrun")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument(
        "--resume",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Reuse terminal ledger entries from the same output directory.",
    )
    parser.add_argument(
        "--retry_failed",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Explicitly rerun failed/timeout entries; disabled by default so a crash is never silently retried.",
    )
    parser.add_argument("--dry_run", action="store_true", default=False)
    return parser.parse_args()


def parse_shape_list(text: str) -> list[tuple[int, int, int]]:
    shapes: list[tuple[int, int, int]] = []
    for raw in text.split(","):
        normalized = raw.strip().lower().replace(" ", "")
        if not normalized:
            continue
        parts = normalized.split("x")
        if len(parts) != 3:
            raise ValueError(f"invalid shape '{raw}', expected MxNxK")
        shapes.append(tuple(map(int, parts)))
    if not shapes:
        raise ValueError("no valid shapes provided")
    return shapes


def parse_positive_int_list(text: str, name: str) -> list[int]:
    values: list[int] = []
    for raw in text.split(","):
        raw = raw.strip()
        if not raw:
            continue
        value = int(raw)
        if value <= 0:
            raise ValueError(f"{name} values must be positive, got {value}")
        if value not in values:
            values.append(value)
    if not values:
        raise ValueError(f"no {name} values supplied")
    return values


def iso_now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fin:
        for chunk in iter(lambda: fin.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite_float(value: object) -> float | None:
    try:
        result = float(str(value))
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def format_float(value: float | None, digits: int = 6) -> str:
    return "" if value is None or not math.isfinite(value) else f"{value:.{digits}f}"


def effective_chunk_rows(args: argparse.Namespace, M: int) -> int:
    m_per_rank = M // args.nproc_per_node
    if args.chunk_rows > 0:
        return min(args.chunk_rows, m_per_rank)
    rows = max(math.ceil(m_per_rank / args.target_chunks_per_rank), args.min_chunk_rows)
    rows = min(rows, m_per_rank)
    return min(((rows + 255) // 256) * 256, m_per_rank)


def logical_chunk_count(args: argparse.Namespace, M: int) -> int:
    return math.ceil((M // args.nproc_per_node) / effective_chunk_rows(args, M))


def append_bool_flag(cmd: list[str], flag: str, value: bool) -> None:
    cmd.append(flag if value else f"--no-{flag[2:]}")


def build_child_cmd(args: argparse.Namespace,
                    shape: tuple[int, int, int],
                    active_window: int,
                    output_csv: Path) -> list[str]:
    M, N, K = shape
    cmd = [
        *shlex.split(args.torchrun_bin),
        "--nproc_per_node",
        str(args.nproc_per_node),
        BENCHMARK_RELATIVE_PATH,
        "--M",
        str(M),
        "--N",
        str(N),
        "--K",
        str(K),
        "--dtype",
        args.dtype,
        "--iters",
        str(args.iters),
        "--warmup_iters",
        str(args.warmup_iters),
        "--chunk_rows",
        str(args.chunk_rows),
        "--target_chunks_per_rank",
        str(args.target_chunks_per_rank),
        "--min_chunk_rows",
        str(args.min_chunk_rows),
        "--active_chunk_window",
        str(active_window),
        "--stage_slots",
        str(args.stage_slots),
        "--steady_sms",
        str(args.steady_sms),
        "--tail_sms",
        str(args.tail_sms),
        "--tail_chunk_window",
        str(args.tail_chunk_window),
        "--comm_lanes",
        str(args.comm_lanes),
        "--n_bands",
        str(args.n_bands),
        "--frontier_chunks",
        str(args.frontier_chunks),
        "--window_policy",
        "with_active_window",
        "--dump_csv",
        "--output_csv",
        str(output_csv),
    ]
    append_bool_flag(cmd, "--autotune", args.autotune)
    append_bool_flag(cmd, "--local_seed_direct", args.local_seed_direct)
    append_bool_flag(cmd, "--trans_b", args.trans_b)
    return cmd


def read_single_row(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as fin:
        rows = list(csv.DictReader(fin))
    if len(rows) != 1:
        raise RuntimeError(f"expected exactly one row in {path}, got {len(rows)}")
    return rows[0]


def write_csv_atomic(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def read_ledger(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        return {}
    with path.open(newline="", encoding="utf-8") as fin:
        return {row["run_key"]: row for row in csv.DictReader(fin) if row.get("run_key")}


def relative_to_root(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def manifest_payload(args: argparse.Namespace,
                     shapes: list[tuple[int, int, int]],
                     windows: list[int]) -> dict[str, Any]:
    script_paths = [Path(__file__), ROOT / BENCHMARK_RELATIVE_PATH]
    return {
        "schema_version": 1,
        "created_at_utc": iso_now(),
        "purpose": "Repeated fixed-implementation GEMM--RS active-window sweep",
        "invocation": sys.argv,
        "cwd": str(Path.cwd()),
        "python": sys.version,
        "platform": platform.platform(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "shapes": [f"{M}x{N}x{K}" for M, N, K in shapes],
        "active_window_values": windows,
        "repetitions": args.repetitions,
        "fixed_parameters": {
            "nproc_per_node": args.nproc_per_node,
            "dtype": args.dtype,
            "iters": args.iters,
            "warmup_iters": args.warmup_iters,
            "chunk_rows": args.chunk_rows,
            "target_chunks_per_rank": args.target_chunks_per_rank,
            "min_chunk_rows": args.min_chunk_rows,
            "stage_slots": args.stage_slots,
            "steady_sms": args.steady_sms,
            "tail_sms": args.tail_sms,
            "tail_chunk_window": args.tail_chunk_window,
            "comm_lanes": args.comm_lanes,
            "n_bands": args.n_bands,
            "frontier_chunks": args.frontier_chunks,
            "autotune": args.autotune,
            "local_seed_direct": args.local_seed_direct,
            "trans_b": args.trans_b,
            "window_policy": "with_active_window",
        },
        "script_sha256": {relative_to_root(path): sha256_file(path) for path in script_paths},
        "claim_boundary": (
            "The sweep measures a resident symmetric-staging/latency trade-off inside one GEMM--RS "
            "implementation. It is not a total-GPU-memory measurement or an external-system baseline comparison."
        ),
    }


def completed_row_is_reusable(row: dict[str, str]) -> bool:
    raw_path = Path(row.get("raw_csv_path", ""))
    return row.get("completion_status") == "completed" and raw_path.is_file()


def make_failed_row(*,
                    run_key: str,
                    attempt_index: int,
                    shape: tuple[int, int, int],
                    num_chunks: int,
                    active_window: int,
                    repetition: int,
                    started_at: str,
                    finished_at: str,
                    elapsed_sec: float,
                    raw_csv: Path,
                    log_path: Path,
                    command: list[str],
                    status: str,
                    returncode: int,
                    reason: str) -> dict[str, object]:
    M, N, K = shape
    return {
        "run_key": run_key,
        "attempt_index": attempt_index,
        "shape": f"{M}x{N}x{K}",
        "M": M,
        "N": N,
        "K": K,
        "num_chunks": num_chunks,
        "requested_active_chunk_window": active_window,
        "effective_active_chunk_window": "",
        "repetition": repetition,
        "latency_ms": "",
        "torch_latency_ms": "",
        "symmetric_staging_gib": "",
        "stage_slots": "",
        "steady_sms": "",
        "tail_sms": "",
        "comm_lanes": "",
        "n_bands": "",
        "frontier_chunks": "",
        "completion_status": status,
        "returncode": returncode,
        "failure_reason": reason,
        "started_at_utc": started_at,
        "finished_at_utc": finished_at,
        "elapsed_sec": format_float(elapsed_sec),
        "raw_csv_path": str(raw_csv),
        "raw_csv_sha256": sha256_file(raw_csv) if raw_csv.is_file() else "",
        "log_path": str(log_path),
        "log_sha256": sha256_file(log_path) if log_path.is_file() else "",
        "command": shlex.join(command),
    }


def make_completed_row(*,
                       run_key: str,
                       attempt_index: int,
                       shape: tuple[int, int, int],
                       num_chunks: int,
                       active_window: int,
                       repetition: int,
                       started_at: str,
                       finished_at: str,
                       elapsed_sec: float,
                       raw_csv: Path,
                       log_path: Path,
                       command: list[str],
                       raw: dict[str, str]) -> dict[str, object]:
    M, N, K = shape
    actual_window = finite_float(raw.get("windowed_active_chunk_window"))
    latency = finite_float(raw.get("windowed_rank_max_total_ms"))
    staging = finite_float(raw.get("windowed_symmetric_staging_gib"))
    if actual_window is None or int(actual_window) != active_window:
        raise RuntimeError(
            f"child returned active window {raw.get('windowed_active_chunk_window')!r}, expected {active_window}"
        )
    if latency is None or staging is None:
        raise RuntimeError(
            "child CSV lacks finite windowed_rank_max_total_ms or windowed_symmetric_staging_gib; "
            "this sweep only accepts the distributed rank-maximum timing field"
        )
    return {
        "run_key": run_key,
        "attempt_index": attempt_index,
        "shape": f"{M}x{N}x{K}",
        "M": M,
        "N": N,
        "K": K,
        "num_chunks": num_chunks,
        "requested_active_chunk_window": active_window,
        "effective_active_chunk_window": int(actual_window),
        "repetition": repetition,
        "latency_ms": format_float(latency),
        "torch_latency_ms": format_float(finite_float(raw.get("torch_rank_max_total_ms"))),
        "symmetric_staging_gib": format_float(staging),
        "stage_slots": raw.get("windowed_stage_slots", ""),
        "steady_sms": argsafe(raw.get("windowed_steady_sms", "")),
        "tail_sms": argsafe(raw.get("windowed_tail_sms", "")),
        "comm_lanes": raw.get("windowed_comm_lanes", ""),
        "n_bands": raw.get("windowed_n_bands", ""),
        "frontier_chunks": raw.get("windowed_frontier_chunks", ""),
        "completion_status": "completed",
        "returncode": 0,
        "failure_reason": "",
        "started_at_utc": started_at,
        "finished_at_utc": finished_at,
        "elapsed_sec": format_float(elapsed_sec),
        "raw_csv_path": str(raw_csv),
        "raw_csv_sha256": sha256_file(raw_csv),
        "log_path": str(log_path),
        "log_sha256": sha256_file(log_path),
        "command": shlex.join(command),
    }


def argsafe(value: object) -> object:
    """Retain blank fields when an older benchmark CSV lacks a metadata column."""
    return value


def next_attempt_paths(raw_dir: Path, log_dir: Path, run_key: str) -> tuple[int, Path, Path]:
    """Allocate unique child artifacts so an explicit retry preserves prior evidence."""
    for attempt_index in range(1, 10_000):
        stem = f"{run_key}_attempt{attempt_index:02d}"
        raw_csv = raw_dir / f"{stem}.csv"
        log_path = log_dir / f"{stem}.log"
        if not raw_csv.exists() and not log_path.exists():
            return attempt_index, raw_csv, log_path
    raise RuntimeError(f"too many retained attempts for {run_key}")


def aggregate_rows(rows: list[dict[str, str]], baseline_window: int, repetitions: int) -> list[dict[str, object]]:
    grouped: dict[tuple[str, int], list[dict[str, str]]] = {}
    for row in rows:
        key = (row["shape"], int(row["requested_active_chunk_window"]))
        grouped.setdefault(key, []).append(row)

    baseline_medians: dict[str, float] = {}
    for (shape, active_window), group_rows in grouped.items():
        if active_window != baseline_window:
            continue
        successful = [finite_float(row.get("latency_ms")) for row in group_rows if row.get("completion_status") == "completed"]
        values = [value for value in successful if value is not None]
        if values:
            baseline_medians[shape] = statistics.median(values)

    aggregates: list[dict[str, object]] = []
    for (shape, active_window), group_rows in sorted(grouped.items()):
        completed = [row for row in group_rows if row.get("completion_status") == "completed"]
        latency_values = [finite_float(row.get("latency_ms")) for row in completed]
        latency_values = [value for value in latency_values if value is not None]
        staging_values = [finite_float(row.get("symmetric_staging_gib")) for row in completed]
        staging_values = [value for value in staging_values if value is not None]
        first = group_rows[0]
        latency_median = statistics.median(latency_values) if latency_values else None
        baseline = baseline_medians.get(shape)
        if not latency_values:
            status = "failed"
        elif len(latency_values) == repetitions:
            status = "completed"
        else:
            status = "partial"
        aggregates.append({
            "shape": shape,
            "M": first["M"],
            "N": first["N"],
            "K": first["K"],
            "num_chunks": first["num_chunks"],
            "active_chunk_window": active_window,
            "baseline_active_chunk_window": baseline_window,
            "planned_repetitions": repetitions,
            "successful_repetitions": len(latency_values),
            "failed_repetitions": max(0, repetitions - len(latency_values)),
            "completion_status": status,
            "latency_median_ms": format_float(latency_median),
            "latency_mean_ms": format_float(statistics.mean(latency_values) if latency_values else None),
            "latency_std_ms": format_float(statistics.stdev(latency_values) if len(latency_values) >= 2 else None),
            "latency_min_ms": format_float(min(latency_values) if latency_values else None),
            "latency_max_ms": format_float(max(latency_values) if latency_values else None),
            "baseline_latency_median_ms": format_float(baseline),
            "latency_ratio_vs_baseline": format_float(latency_median / baseline if latency_median and baseline else None),
            "symmetric_staging_gib": format_float(statistics.median(staging_values) if staging_values else None),
        })
    return aggregates


def main() -> None:
    args = parse_args()
    shapes = parse_shape_list(args.shapes)
    windows = parse_positive_int_list(args.active_window_values, "active-window")
    if args.repetitions <= 0:
        raise SystemExit("--repetitions must be positive")
    for M, _, K in shapes:
        if M % args.nproc_per_node != 0 or K % args.nproc_per_node != 0:
            raise SystemExit("each shape must have M and K divisible by --nproc_per_node")
        num_chunks = logical_chunk_count(args, M)
        invalid = [window for window in windows if window > num_chunks]
        if invalid:
            raise SystemExit(
                f"shape {M} has {num_chunks} logical chunks with the fixed chunk configuration; "
                f"requested windows exceed it: {invalid}"
            )

    run_id = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else ROOT / "benchmark/active_window_sweep_runs" / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "raw_child_csv"
    log_dir = output_dir / "logs"
    raw_dir.mkdir(exist_ok=True)
    log_dir.mkdir(exist_ok=True)
    manifest_path = output_dir / "run_manifest.json"
    if not manifest_path.exists():
        manifest_path.write_text(json.dumps(manifest_payload(args, shapes, windows), indent=2, sort_keys=True) + "\n",
                                 encoding="utf-8")

    plan_rows: list[dict[str, object]] = []
    for shape in shapes:
        M, N, K = shape
        num_chunks = logical_chunk_count(args, M)
        for window in windows:
            for repetition in range(1, args.repetitions + 1):
                plan_rows.append({
                    "shape": f"{M}x{N}x{K}",
                    "M": M,
                    "N": N,
                    "K": K,
                    "num_chunks": num_chunks,
                    "active_chunk_window": window,
                    "repetition": repetition,
                })
    write_csv_atomic(output_dir / "sweep_plan.csv", list(plan_rows[0]), plan_rows)
    if args.dry_run:
        print(f"[dry-run] {len(plan_rows)} planned child launches in {output_dir}", flush=True)
        return

    ledger_path = output_dir / "run_ledger.csv"
    ledger_by_key = read_ledger(ledger_path)
    for plan in plan_rows:
        shape = (int(plan["M"]), int(plan["N"]), int(plan["K"]))
        shape_label = str(plan["shape"])
        window = int(plan["active_chunk_window"])
        repetition = int(plan["repetition"])
        run_key = f"{shape_label}_L{window}_rep{repetition}"
        existing = ledger_by_key.get(run_key)
        if args.resume and existing is not None:
            if completed_row_is_reusable(existing):
                print(f"[sweep] resume: retain completed {run_key}", flush=True)
                continue
            if existing.get("completion_status") in {"failed", "timeout"} and not args.retry_failed:
                print(f"[sweep] resume: retain failed {run_key}; use --retry_failed to rerun it", flush=True)
                continue

        attempt_index, raw_csv, log_path = next_attempt_paths(raw_dir, log_dir, run_key)
        command = build_child_cmd(args, shape, window, raw_csv)
        print(f"[sweep] attempt {attempt_index} run {run_key}: {shlex.join(command)}", flush=True)
        started_at = iso_now()
        started = time.perf_counter()
        status = "completed"
        failure_reason = ""
        returncode = 0
        stdout = ""
        stderr = ""
        try:
            process = subprocess.run(
                command,
                cwd=str(ROOT),
                text=True,
                capture_output=True,
                timeout=args.candidate_timeout_sec if args.candidate_timeout_sec > 0 else None,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
            returncode = process.returncode
            stdout, stderr = process.stdout, process.stderr
            if returncode != 0:
                status = "failed"
                failure_reason = "nonzero_returncode"
        except subprocess.TimeoutExpired as exc:
            returncode = 124
            status = "timeout"
            failure_reason = "timeout"
            stdout = exc.stdout or ""
            stderr = exc.stderr or ""
        elapsed = time.perf_counter() - started
        finished_at = iso_now()
        log_path.write_text(stdout + ("\n" if stdout and stderr else "") + stderr, encoding="utf-8")

        if status == "completed":
            try:
                raw = read_single_row(raw_csv)
                row = make_completed_row(
                    run_key=run_key,
                    attempt_index=attempt_index,
                    shape=shape,
                    num_chunks=int(plan["num_chunks"]),
                    active_window=window,
                    repetition=repetition,
                    started_at=started_at,
                    finished_at=finished_at,
                    elapsed_sec=elapsed,
                    raw_csv=raw_csv,
                    log_path=log_path,
                    command=command,
                    raw=raw,
                )
            except (OSError, RuntimeError, ValueError) as exc:
                row = make_failed_row(
                    run_key=run_key,
                    attempt_index=attempt_index,
                    shape=shape,
                    num_chunks=int(plan["num_chunks"]),
                    active_window=window,
                    repetition=repetition,
                    started_at=started_at,
                    finished_at=finished_at,
                    elapsed_sec=elapsed,
                    raw_csv=raw_csv,
                    log_path=log_path,
                    command=command,
                    status="failed",
                    returncode=returncode,
                    reason=f"invalid_child_output: {exc}",
                )
        else:
            row = make_failed_row(
                run_key=run_key,
                attempt_index=attempt_index,
                shape=shape,
                num_chunks=int(plan["num_chunks"]),
                active_window=window,
                repetition=repetition,
                started_at=started_at,
                finished_at=finished_at,
                elapsed_sec=elapsed,
                raw_csv=raw_csv,
                log_path=log_path,
                command=command,
                status=status,
                returncode=returncode,
                reason=failure_reason,
            )
        ledger_by_key[run_key] = {field: str(row.get(field, "")) for field in RUN_FIELDS}
        ordered_rows = [ledger_by_key[key] for key in sorted(ledger_by_key)]
        write_csv_atomic(ledger_path, RUN_FIELDS, ordered_rows)

    ledger_rows = [ledger_by_key[key] for key in sorted(ledger_by_key)]
    aggregates = aggregate_rows(ledger_rows, baseline_window=max(windows), repetitions=args.repetitions)
    write_csv_atomic(output_dir / "active_window_sweep_aggregate.csv", AGGREGATE_FIELDS, aggregates)
    print(f"[ledger] {ledger_path}", flush=True)
    print(f"[aggregate] {output_dir / 'active_window_sweep_aggregate.csv'}", flush=True)
    print(f"[manifest] {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
