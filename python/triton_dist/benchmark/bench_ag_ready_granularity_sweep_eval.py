################################################################################
#
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
################################################################################
"""
cd /root/Triton-distributed && source ./scripts/setenv.sh && cd python/triton_dist && python benchmark/bench_ag_ready_granularity_sweep_eval.py \
    --M 8192 --N 28672 --K 8192 \
    --repeats 2 --iters 3 --warmup_iters 2 \
    --granularity_values 1024,512 \
    --min_m_per_rank_for_tile_ready 2048 \
    --include_rank_ready \
    --timeout_sec 600


AG-GEMM ready-granularity sweep driver (repeat-capable, rank-max aware).

Drives ``bench_ag_gemm_eval_8rank.py`` across:
  - one or more shapes (``--shape_list "M,N,K;M,N,K"``),
  - a coarse-to-fine set of ready granularities (rank-ready / fixed tile rows /
    heuristic),
  - ``--repeats`` independent torchrun launches per cell.

Each launch is a fresh process and writes its own per-rank JSON + rank-0 summary
under ``raw_launches/<point>/``, so there is no shared-CSV race. The driver then
classifies failures and aggregates the per-launch rank-max latencies into a
median / min--max table (one row per shape x granularity cell) that the plotting
script consumes directly.

Output layout::

  <output_root>/<shape_tag>/<run_id>/
    logs/<point>__rep<k>.log
    raw_launches/<point>/ag_gemm_<M>_<N>_<K>_repeat_<k>.json
    ag_ready_granularity_sweep_summary.csv   # one row per launch
    ag_ready_aggregated.csv                  # one row per (shape, point) cell
    README.txt

Usage::

  python benchmark/bench_ag_ready_granularity_sweep_eval.py \
      --shape_list "32768,28672,8192;8192,28672,8192;16384,49152,8192" \
      --repeats 5 --iters 10 --warmup_iters 5 --dtype bfloat16 \
      --granularity_values 8192,4096,2048,1024,512,256 \
      --include_rank_ready
"""
import argparse
import csv
import json
import os
import statistics
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
PKG_ROOT = Path(__file__).resolve().parents[1]
BENCH_DIR = Path(__file__).resolve().parent
AG_SCRIPT = BENCH_DIR / "bench_ag_gemm_eval_8rank.py"
OUTPUT_ROOT = BENCH_DIR / "ag_ready_granularity_eval_results"

# Sentinels for plotting the coarse-to-fine x-axis.
GRANULARITY_RANK_READY = 10**9
GRANULARITY_HEURISTIC = -1


@dataclass(frozen=True)
class SweepPoint:
    tag: str
    enable_tile_ready: bool
    tile_rows_per_chunk: int


def default_sweep_points() -> list[SweepPoint]:
    # The fixed sweep is intentionally coarse-to-fine so the resulting CSV can
    # be plotted directly as a granularity curve. The heuristic point is kept as
    # an optional reference because it is not part of the monotonic sweep.
    return [
        SweepPoint("rank_ready", False, 0),
        SweepPoint("tile_8192", True, 8192),
        SweepPoint("tile_4096", True, 4096),
        SweepPoint("tile_2048", True, 2048),
        SweepPoint("tile_1024", True, 1024),
        SweepPoint("tile_512", True, 512),
        SweepPoint("tile_256", True, 256),
    ]


def parse_granularity_values(value: str | None) -> list[int]:
    if value is None:
        return [8192, 4096, 2048, 1024, 512, 256]
    value = value.strip()
    if not value:
        return [8192, 4096, 2048, 1024, 512, 256]
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_shape_list(value: str) -> list[tuple[int, int, int]]:
    """Parse 'M,N,K;M,N,K' into a list of (M, N, K) tuples."""
    shapes: list[tuple[int, int, int]] = []
    for part in value.split(";"):
        part = part.strip()
        if not part:
            continue
        dims = [int(x.strip()) for x in part.split(",")]
        if len(dims) != 3:
            raise ValueError(f"malformed shape '{part}'; expected M,N,K")
        shapes.append(tuple(dims))
    return shapes


def parse_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        text = str(value).strip()
        if not text or text.lower() in ("nan", "none"):
            return None
        return float(text)
    except (TypeError, ValueError):
        return None


def child_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    python_root = str(REPO_ROOT / "python")
    old_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = python_root if not old_pythonpath else f"{python_root}:{old_pythonpath}"
    return env


def run_child(cmd: list[str], log_path: Path, timeout_sec: float) -> tuple[int, str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(PKG_ROOT),
            env=child_env(),
            text=True,
            capture_output=True,
            timeout=timeout_sec if timeout_sec > 0 else None,
        )
        merged = proc.stdout + ("\n" + proc.stderr if proc.stderr else "")
        log_path.write_text(merged, encoding="utf-8")
        return proc.returncode, merged
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        merged = stdout + ("\n" + stderr if stderr else "")
        merged += f"\n[sweep-driver] timeout after {timeout_sec:.1f}s\n"
        log_path.write_text(merged, encoding="utf-8")
        return 124, merged


def classify_failure(returncode: int, log_text: str) -> str:
    """Classify a failed launch into a paper-friendly failure category."""
    if returncode == 0:
        return "ok"
    text = log_text.lower()
    if "timeout after" in text:
        return "timeout"
    if "out of memory" in text:
        return "oom"
    if "cuda error" in text:
        return "cuda_error"
    if "output mismatch" in text or "assertionerror" in text or "mismatch" in text:
        return "correctness_failed"
    return "child_failed"


def build_cmd(args, shape, point: SweepPoint, repeat_id: int, result_dir: Path) -> list[str]:
    M, N, K = shape
    cmd = [
        args.torchrun_bin,
        "--nproc_per_node",
        str(args.nproc_per_node),
        str(AG_SCRIPT),
        "--M",
        str(M),
        "--N",
        str(N),
        "--K",
        str(K),
        "--iters",
        str(args.iters),
        "--warmup_iters",
        str(args.warmup_iters),
        "--dtype",
        args.dtype,
        "--target_chunks_per_rank",
        str(args.target_chunks_per_rank),
        "--min_tile_rows_per_chunk",
        str(args.min_tile_rows_per_chunk),
        "--min_m_per_rank_for_tile_ready",
        str(args.min_m_per_rank_for_tile_ready),
        "--copy_sms",
        str(args.copy_sms),
        "--tile_rows_per_chunk",
        str(point.tile_rows_per_chunk),
        "--repeat_id",
        str(repeat_id),
        "--result_dir",
        str(result_dir),
    ]
    cmd.append("--autotune" if args.autotune else "--no-autotune")
    cmd.append("--trans_b" if args.trans_b else "--no-trans_b")
    cmd.append("--enable_tile_ready" if point.enable_tile_ready else "--no-enable_tile_ready")
    cmd.append("--cooperative_copy" if args.cooperative_copy else "--no-cooperative_copy")
    if args.skip_torch:
        cmd.append("--skip_torch")
    if args.profile:
        cmd.append("--profile")
    return cmd


def read_child_summary(json_path: Path) -> dict | None:
    if not json_path.exists():
        return None
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def extract_summary_fields(summary: dict, row: dict) -> dict:
    """Pull the effective granularity and rank-max latencies from the child's
    rank-0 summary JSON into the per-launch row."""
    rank_max = summary.get("rank_max") or {}
    shape = summary.get("shape") or {}
    row["effective_tile_rows_per_chunk"] = summary.get("tile_rows_per_chunk")
    # The *requested* flag is enable_tile_ready; the *actually engaged* value is
    # enable_row_tile_barrier (the kernel may silently fall back to rank-ready
    # when M_per_rank is below min_m_per_rank_for_tile_ready). Prefer the real
    # value; fall back to the request flag for legacy runs that lack the field.
    row["effective_enable_row_tile_barrier"] = summary.get(
        "enable_row_tile_barrier", summary.get("enable_tile_ready"))
    row["effective_num_tile_chunks"] = summary.get("num_tile_chunks")
    row["tile_barrier_present"] = summary.get("tile_barrier_present")
    row["new_triton_rankmax_ms"] = parse_float(rank_max.get("new_triton_ms"))
    row["torch_rankmax_ms"] = parse_float(rank_max.get("torch_ms"))
    row["base_triton_rankmax_ms"] = parse_float(rank_max.get("base_triton_ms"))
    row["speedup_vs_torch_rankmax"] = parse_float(rank_max.get("speedup_vs_torch"))
    row["new_vs_base_rankmax"] = parse_float(rank_max.get("new_vs_base"))
    row["first_ready_ms_proxy"] = parse_float(summary.get("first_ready_ms"))
    row["consumer_ts_is_proxy"] = summary.get("consumer_ts_is_proxy")
    row["world_size"] = summary.get("world_size")
    if shape:
        row["M"] = shape.get("M")
        row["N"] = shape.get("N")
        row["K"] = shape.get("K")
    return row


def granularity_rank(point: SweepPoint) -> int:
    if not point.enable_tile_ready:
        return GRANULARITY_RANK_READY
    if point.tag == "tile_ready_heuristic":
        return GRANULARITY_HEURISTIC
    return int(point.tile_rows_per_chunk)


def _median(values: list[float]) -> float:
    return statistics.median(values) if values else float("nan")


def _flag_int(value: object) -> int:
    """Normalize a JSON/CSV flag (0/1, bool, 'true'/'false') to int; None -> 0."""
    if value is None:
        return 0
    text = str(value).strip().lower()
    if text in ("1", "true", "yes", "on"):
        return 1
    if text in ("0", "false", "no", "off"):
        return 0
    return 1 if int(float(text)) else 0


def aggregate_point(point: SweepPoint, launch_rows: list[dict]) -> dict:
    """Aggregate one (shape, point) cell: median/min/max of rank-max latencies
    across successful launches, plus failure accounting."""
    ok_rows = [r for r in launch_rows if r["driver_status"] == "ok"]
    fail_rows = [r for r in launch_rows if r["driver_status"] != "ok"]
    agg = {
        "mode_tag": point.tag,
        "granularity": granularity_rank(point),
        "requested_tile_rows_per_chunk": point.tile_rows_per_chunk,
        "enable_tile_ready": int(point.enable_tile_ready),
        "n_launches": len(launch_rows),
        "n_success": len(ok_rows),
        "n_oom": sum(1 for r in fail_rows if r["failure_class"] == "oom"),
        "n_timeout": sum(1 for r in fail_rows if r["failure_class"] == "timeout"),
        "n_correctness_failed": sum(1 for r in fail_rows if r["failure_class"] == "correctness_failed"),
        "n_child_failed": sum(1 for r in fail_rows if r["failure_class"] not in ("oom", "timeout", "correctness_failed")),
        "failure_class": "ok" if not fail_rows else ("mixed" if ok_rows else fail_rows[0]["failure_class"]),
        # The *actually engaged* tile barrier (not the request flag): the kernel
        # silently falls back to rank-ready when M_per_rank is below
        # min_m_per_rank_for_tile_ready, so the request and the outcome can differ.
        "effective_enable_row_tile_barrier": int(
            any(_flag_int(r.get("effective_enable_row_tile_barrier")) for r in ok_rows)
        ) if ok_rows else 0,
        "effective_num_tile_chunks": max(
            (int(v) for r in ok_rows if (v := r.get("effective_num_tile_chunks")) is not None),
            default=0,
        ),
        "tile_barrier_present": int(
            any(_flag_int(r.get("tile_barrier_present")) for r in ok_rows)
        ) if ok_rows else 0,
    }
    if ok_rows:
        new_ms = [r["new_triton_rankmax_ms"] for r in ok_rows if r["new_triton_rankmax_ms"] is not None]
        torch_ms = [r["torch_rankmax_ms"] for r in ok_rows if r["torch_rankmax_ms"] is not None]
        base_ms = [r["base_triton_rankmax_ms"] for r in ok_rows if r["base_triton_rankmax_ms"] is not None]
        first_ready = [r["first_ready_ms_proxy"] for r in ok_rows if r["first_ready_ms_proxy"] is not None]
        agg.update(
            {
                "new_median_ms": _median(new_ms),
                "new_min_ms": min(new_ms) if new_ms else float("nan"),
                "new_max_ms": max(new_ms) if new_ms else float("nan"),
                "torch_median_ms": _median(torch_ms),
                "torch_min_ms": min(torch_ms) if torch_ms else float("nan"),
                "torch_max_ms": max(torch_ms) if torch_ms else float("nan"),
                "base_median_ms": _median(base_ms),
                "base_min_ms": min(base_ms) if base_ms else float("nan"),
                "base_max_ms": max(base_ms) if base_ms else float("nan"),
                "speedup_median": (
                    _median(torch_ms) / _median(new_ms) if new_ms and torch_ms else float("nan")
                ),
                "first_ready_median_ms_proxy": _median(first_ready),
                "correctness": "pass",
            }
        )
    else:
        agg.update(
            {
                "new_median_ms": float("nan"),
                "new_min_ms": float("nan"),
                "new_max_ms": float("nan"),
                "torch_median_ms": float("nan"),
                "torch_min_ms": float("nan"),
                "torch_max_ms": float("nan"),
                "base_median_ms": float("nan"),
                "base_min_ms": float("nan"),
                "base_max_ms": float("nan"),
                "speedup_median": float("nan"),
                "first_ready_median_ms_proxy": float("nan"),
                "correctness": "no_successful_launch",
            }
        )
    return agg


def write_summary_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    preferred = [
        "run_id",
        "shape_tag",
        "M",
        "N",
        "K",
        "mode_tag",
        "requested_tile_rows_per_chunk",
        "enable_tile_ready",
        "repeat_id",
        "driver_status",
        "returncode",
        "failure_class",
        "effective_tile_rows_per_chunk",
        "effective_enable_row_tile_barrier",
        "effective_num_tile_chunks",
        "tile_barrier_present",
        "new_triton_rankmax_ms",
        "torch_rankmax_ms",
        "base_triton_rankmax_ms",
        "speedup_vs_torch_rankmax",
        "new_vs_base_rankmax",
        "first_ready_ms_proxy",
        "consumer_ts_is_proxy",
        "world_size",
        "log_path",
        "summary_json_path",
        "command",
    ]
    fieldnames: list[str] = []
    seen = set()
    for name in preferred:
        if any(name in row for row in rows):
            fieldnames.append(name)
            seen.add(name)
    for row in rows:
        for key in row:
            if key in seen:
                continue
            fieldnames.append(key)
            seen.add(key)
    with open(path, "w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_aggregated_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    preferred = [
        "shape_tag",
        "M",
        "N",
        "K",
        "mode_tag",
        "granularity",
        "requested_tile_rows_per_chunk",
        "enable_tile_ready",
        "n_launches",
        "n_success",
        "n_oom",
        "n_timeout",
        "n_correctness_failed",
        "n_child_failed",
        "failure_class",
        "effective_enable_row_tile_barrier",
        "effective_num_tile_chunks",
        "tile_barrier_present",
        "new_median_ms",
        "new_min_ms",
        "new_max_ms",
        "torch_median_ms",
        "torch_min_ms",
        "torch_max_ms",
        "base_median_ms",
        "base_min_ms",
        "base_max_ms",
        "speedup_median",
        "first_ready_median_ms_proxy",
        "correctness",
    ]
    fieldnames: list[str] = []
    seen = set()
    for name in preferred:
        if any(name in row for row in rows):
            fieldnames.append(name)
            seen.add(name)
    for row in rows:
        for key in row:
            if key in seen:
                continue
            fieldnames.append(key)
            seen.add(key)
    with open(path, "w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Sweep AG row-chunk granularity (repeat-capable) for the ready-unit "
            "ablation figure. Produces per-launch and per-cell aggregated CSV."
        )
    )
    parser.add_argument("--M", type=int, default=8192)
    parser.add_argument("--N", type=int, default=28672)
    parser.add_argument("--K", type=int, default=8192)
    parser.add_argument("--shape_list", default="",
                        help="Semicolon-separated 'M,N,K;M,N,K'; overrides --M/--N/--K when set.")
    parser.add_argument("--shape_tag", default="",
                        help="Directory label; defaults to '<M>x<N>x<K>'.")
    parser.add_argument("--nproc_per_node", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=5,
                        help="Independent launches per (shape, granularity) cell.")
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--warmup_iters", type=int, default=5)
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16"])
    parser.add_argument("--autotune", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--profile", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--skip_torch", default=True, action=argparse.BooleanOptionalAction,
                        help="Skip the torch/NCCL serial allgather+GEMM baseline (default on; "
                             "the comparison is triton-dist base vs the new kernel only).")
    parser.add_argument("--trans_b", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--cooperative_copy", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--copy_sms", type=int, default=0)
    parser.add_argument("--target_chunks_per_rank", type=int, default=2)
    parser.add_argument("--min_tile_rows_per_chunk", type=int, default=1024)
    parser.add_argument("--min_m_per_rank_for_tile_ready", type=int, default=4096)
    parser.add_argument("--granularity_values", default="")
    parser.add_argument("--include_rank_ready", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--include_heuristic", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--timeout_sec", type=float, default=0.0)
    parser.add_argument("--fail_fast", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--dry_run", "--dry-run", dest="dry_run", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--torchrun_bin", default="torchrun")
    parser.add_argument("--output_root", default=str(OUTPUT_ROOT))
    return parser.parse_args()


def main():
    args = parse_args()
    if args.shape_list:
        shapes = parse_shape_list(args.shape_list)
    else:
        shapes = [(args.M, args.N, args.K)]

    sweep_points: list[SweepPoint] = []
    if args.include_rank_ready:
        sweep_points.append(SweepPoint("rank_ready", False, 0))
    if args.include_heuristic:
        sweep_points.append(SweepPoint("tile_ready_heuristic", True, 0))
    for value in parse_granularity_values(args.granularity_values or None):
        sweep_points.append(SweepPoint(f"tile_{value}", True, value))

    for shape in shapes:
        M, N, K = shape
        shape_tag = args.shape_tag if args.shape_tag else f"{M}x{N}x{K}"
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(args.output_root) / shape_tag / run_id
        logs_dir = output_dir / "logs"
        output_dir.mkdir(parents=True, exist_ok=True)

        launch_rows: list[dict] = []
        agg_rows: list[dict] = []
        print(f"\n=== shape {shape_tag} ({M}x{N}x{K}) run_id={run_id} ===", flush=True)

        for point in sweep_points:
            point_result_dir = output_dir / "raw_launches" / point.tag
            point_rows: list[dict] = []
            for rep in range(args.repeats):
                cmd = build_cmd(args, shape, point, rep, point_result_dir)
                log_path = logs_dir / f"{point.tag}__rep{rep}.log"
                summary_json = point_result_dir / f"ag_gemm_{M}_{N}_{K}_repeat_{rep}.json"

                if args.dry_run:
                    row = {
                        "run_id": run_id,
                        "shape_tag": shape_tag,
                        "M": M,
                        "N": N,
                        "K": K,
                        "mode_tag": point.tag,
                        "requested_tile_rows_per_chunk": point.tile_rows_per_chunk,
                        "enable_tile_ready": int(point.enable_tile_ready),
                        "repeat_id": rep,
                        "driver_status": "dry_run",
                        "returncode": 0,
                        "failure_class": "dry_run",
                        "log_path": str(log_path),
                        "summary_json_path": str(summary_json),
                        "command": " ".join(cmd),
                    }
                    launch_rows.append(row)
                    point_rows.append(row)
                    print(f"[dry-run][ag-sweep] {' '.join(cmd)}", flush=True)
                    continue

                print(
                    f"[run][ag-sweep] shape={shape_tag} point={point.tag} rep={rep}/{args.repeats}",
                    flush=True,
                )
                returncode, output = run_child(cmd, log_path=log_path, timeout_sec=args.timeout_sec)
                summary = read_child_summary(summary_json)
                if summary is not None:
                    driver_status = "ok"
                    failure_class = "ok"
                else:
                    driver_status = "child_failed" if returncode != 0 else "missing_json"
                    failure_class = classify_failure(returncode, output)

                row: dict = {
                    "run_id": run_id,
                    "shape_tag": shape_tag,
                    "M": M,
                    "N": N,
                    "K": K,
                    "mode_tag": point.tag,
                    "requested_tile_rows_per_chunk": point.tile_rows_per_chunk,
                    "enable_tile_ready": int(point.enable_tile_ready),
                    "repeat_id": rep,
                    "driver_status": driver_status,
                    "returncode": returncode,
                    "failure_class": failure_class,
                    "log_path": str(log_path),
                    "summary_json_path": str(summary_json),
                    "command": " ".join(cmd),
                }
                if summary is not None:
                    row = extract_summary_fields(summary, row)
                launch_rows.append(row)
                point_rows.append(row)
                if returncode != 0 and args.fail_fast:
                    raise RuntimeError(f"AG granularity sweep child benchmark failed: {' '.join(cmd)}")

            agg_rows.append(aggregate_point(point, point_rows))

        summary_path = output_dir / "ag_ready_granularity_sweep_summary.csv"
        write_summary_csv(summary_path, launch_rows)
        print(f"[summary-per-launch] {summary_path}", flush=True)

        agg_path = output_dir / "ag_ready_aggregated.csv"
        write_aggregated_csv(agg_path, agg_rows)
        print(f"[summary-aggregated] {agg_path}", flush=True)

        manifest = output_dir / "README.txt"
        manifest.write_text(
            "\n".join(
                [
                    f"run_id={run_id}",
                    f"shape_tag={shape_tag}",
                    f"shapes={';'.join(f'{s[0]},{s[1]},{s[2]}' for s in shapes)}",
                    f"nproc_per_node={args.nproc_per_node}",
                    f"repeats={args.repeats}",
                    f"dtype={args.dtype}",
                    f"iters={args.iters}",
                    f"warmup_iters={args.warmup_iters}",
                    f"autotune={args.autotune}",
                    f"granularity_values={args.granularity_values or '8192,4096,2048,1024,512,256'}",
                    f"include_rank_ready={args.include_rank_ready}",
                    f"include_heuristic={args.include_heuristic}",
                    "latency_metric=rank-max over ranks; per-launch values in _rankmax_ columns",
                    "aggregation=median / min--max over successful independent launches (>=5 recommended)",
                    "first_ready_ms_proxy=host-observed proxy; first_consumer_ts_is_proxy=1; do not claim device first-consumer",
                    "expected_pattern=coarse_to_fine tradeoff; look for a lowest-latency region rather than assuming a perfect symmetric U-shape",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"[manifest] {manifest}", flush=True)


if __name__ == "__main__":
    main()
