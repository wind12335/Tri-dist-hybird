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

import argparse
import csv
import os
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
# 先跑 sweep，生成 CSV：
#   python benchmark/bench_ag_ready_granularity_sweep.py \
#     --nproc_per_node 4 \
#     --M 32768 \
#     --N 28672 \
#     --K 8192 \
#     --dtype bfloat16 \
#     --iters 10 \
#     --warmup_iters 5 \
#     --granularity_values 8192,4096,2048,1024,512,256 \
#     --include_rank_ready \
#     --include_heuristic

#   运行完成后，会在：

#   benchmark/ag_ready_granularity_results/<run_id>/

#   下面生成最关键的文件：

#   - ag_ready_granularity_sweep_summary.csv

REPO_ROOT = Path(__file__).resolve().parents[3]
PKG_ROOT = Path(__file__).resolve().parents[1]
BENCH_DIR = Path(__file__).resolve().parent
AG_SCRIPT = BENCH_DIR / "bench_new_allgather_gemm_8rank.py"
CHILD_CSV_DIR = PKG_ROOT / "csv"
CHILD_CSV = "perf_new_ag_gemm_{nproc}_ranks.csv"
OUTPUT_ROOT = BENCH_DIR / "ag_ready_granularity_results"


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


def parse_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        text = str(value).strip()
        if not text or text.lower() == "nan":
            return None
        return float(text)
    except (TypeError, ValueError):
        return None


def child_csv_path(nproc: int) -> Path:
    return CHILD_CSV_DIR / CHILD_CSV.format(nproc=nproc)


def prepare_child_csv(csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if csv_path.exists():
        csv_path.unlink()


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


def read_single_row_csv(csv_path: Path) -> dict[str, str] | None:
    if not csv_path.exists():
        return None
    with open(csv_path, "r", encoding="utf-8") as fin:
        rows = list(csv.DictReader(fin))
    if not rows:
        return None
    return rows[0]


def archive_child_csv(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    return str(dst)


def build_cmd(args, point: SweepPoint) -> list[str]:
    cmd = [
        args.torchrun_bin,
        "--nproc_per_node",
        str(args.nproc_per_node),
        str(AG_SCRIPT),
        "--M",
        str(args.M),
        "--N",
        str(args.N),
        "--K",
        str(args.K),
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
        "--dump_csv",
    ]
    cmd.append("--autotune" if args.autotune else "--no-autotune")
    cmd.append("--trans_b" if args.trans_b else "--no-trans_b")
    cmd.append("--enable_tile_ready" if point.enable_tile_ready else "--no-enable_tile_ready")
    cmd.append("--cooperative_copy" if args.cooperative_copy else "--no-cooperative_copy")
    if args.profile:
        cmd.append("--profile")
    return cmd


def derive_fields(row: dict[str, object]) -> dict[str, object]:
    enriched = dict(row)
    new_total = parse_float(enriched.get("new dist-triton ag gemm latency (ms)"))
    base_total = parse_float(enriched.get("dist-triton ag gemm latency (ms)"))
    torch_total = parse_float(enriched.get("torch ag gemm latency (ms)"))
    first_ready = parse_float(enriched.get("first_ready_ts_ms"))
    completion = parse_float(enriched.get("last_completion_ts_ms"))
    if new_total is not None:
        enriched["new_triton_total_ms"] = new_total
    if base_total is not None:
        enriched["base_triton_total_ms"] = base_total
    if torch_total is not None:
        enriched["torch_total_ms"] = torch_total
    if first_ready is not None and completion is not None:
        enriched["ready_to_completion_tail_ms"] = completion - first_ready
    if first_ready is not None and new_total is not None:
        enriched["ready_fraction_of_total"] = first_ready / max(new_total, 1e-6)
    observed_rows = parse_float(enriched.get("tile_rows_per_chunk"))
    if observed_rows is not None and observed_rows > 0:
        enriched["granularity_rank"] = int(observed_rows)
    elif str(enriched.get("mode_tag")) == "rank_ready":
        enriched["granularity_rank"] = 10**9
    return enriched


def write_summary_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    preferred = [
        "run_id",
        "shape_tag",
        "mode_tag",
        "M",
        "N",
        "K",
        "requested_tile_rows_per_chunk",
        "enable_tile_ready",
        "driver_status",
        "returncode",
        "log_path",
        "raw_child_csv_path",
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


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Sweep AG row-chunk granularity for Figure 3 ready-unit evidence. "
            "This driver is intended to reveal a coarse-to-fine tradeoff rather than a single speedup point."
        )
    )
    parser.add_argument("--M", type=int, default=32768)
    parser.add_argument("--N", type=int, default=28672)
    parser.add_argument("--K", type=int, default=8192)
    parser.add_argument("--shape_tag", default="granularity_sweep")
    parser.add_argument("--nproc_per_node", type=int, default=4)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--warmup_iters", type=int, default=5)
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16"])
    parser.add_argument("--autotune", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--profile", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--trans_b", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--cooperative_copy", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--copy_sms", type=int, default=0)
    parser.add_argument("--target_chunks_per_rank", type=int, default=2)
    parser.add_argument("--min_tile_rows_per_chunk", type=int, default=256)
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
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    sweep_points: list[SweepPoint] = []
    if args.include_rank_ready:
        sweep_points.append(SweepPoint("rank_ready", False, 0))
    if args.include_heuristic:
        sweep_points.append(SweepPoint("tile_ready_heuristic", True, 0))
    for value in parse_granularity_values(args.granularity_values or None):
        sweep_points.append(SweepPoint(f"tile_{value}", True, value))

    rows: list[dict[str, object]] = []
    for point in sweep_points:
        cmd = build_cmd(args, point)
        run_tag = point.tag
        log_path = output_dir / "logs" / f"{run_tag}.log"
        raw_csv_path = output_dir / "raw_child_csv" / f"{run_tag}.csv"
        child_csv = child_csv_path(args.nproc_per_node)
        prepare_child_csv(child_csv)

        if args.dry_run:
            row = {
                "run_id": run_id,
                "shape_tag": args.shape_tag,
                "mode_tag": point.tag,
                "M": args.M,
                "N": args.N,
                "K": args.K,
                "requested_tile_rows_per_chunk": point.tile_rows_per_chunk,
                "enable_tile_ready": point.enable_tile_ready,
                "driver_status": "dry_run",
                "returncode": 0,
                "log_path": str(log_path),
                "raw_child_csv_path": str(raw_csv_path),
                "command": " ".join(cmd),
            }
            row = derive_fields(row)
            rows.append(row)
            print(f"[dry-run][ag-sweep] {' '.join(cmd)}", flush=True)
            continue

        print(f"[run][ag-sweep] mode={point.tag}", flush=True)
        returncode, output = run_child(cmd, log_path=log_path, timeout_sec=args.timeout_sec)
        csv_row = read_single_row_csv(child_csv)
        row: dict[str, object] = {
            "run_id": run_id,
            "shape_tag": args.shape_tag,
            "mode_tag": point.tag,
            "M": args.M,
            "N": args.N,
            "K": args.K,
            "requested_tile_rows_per_chunk": point.tile_rows_per_chunk,
            "enable_tile_ready": point.enable_tile_ready,
            "driver_status": "ok" if returncode == 0 else "child_failed",
            "returncode": returncode,
            "log_path": str(log_path),
            "raw_child_csv_path": "",
            "command": " ".join(cmd),
            "stdout_excerpt": output[:2000],
        }
        if csv_row is not None:
            row.update(csv_row)
            row["raw_child_csv_path"] = archive_child_csv(child_csv, raw_csv_path)
        else:
            row["driver_status"] = "missing_child_csv" if returncode == 0 else row["driver_status"]

        row = derive_fields(row)
        rows.append(row)
        if returncode != 0 and args.fail_fast:
            raise RuntimeError(f"AG granularity sweep child benchmark failed: {' '.join(cmd)}")

    summary_path = output_dir / "ag_ready_granularity_sweep_summary.csv"
    write_summary_csv(summary_path, rows)
    print(f"[summary] {summary_path}", flush=True)

    manifest_path = output_dir / "README.txt"
    manifest_path.write_text(
        "\n".join(
            [
                f"run_id={run_id}",
                f"shape_tag={args.shape_tag}",
                f"M={args.M}",
                f"N={args.N}",
                f"K={args.K}",
                f"nproc_per_node={args.nproc_per_node}",
                f"dtype={args.dtype}",
                f"iters={args.iters}",
                f"warmup_iters={args.warmup_iters}",
                f"autotune={args.autotune}",
                f"granularity_values={args.granularity_values or '8192,4096,2048,1024,512,256'}",
                f"include_rank_ready={args.include_rank_ready}",
                f"include_heuristic={args.include_heuristic}",
                "expected_pattern=coarse_to_fine tradeoff; look for a lowest-latency region rather than assuming a perfect symmetric U-shape",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[manifest] {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
