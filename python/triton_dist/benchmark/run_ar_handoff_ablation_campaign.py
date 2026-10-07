"""Run and aggregate the five-replicate, three-shape AR handoff campaign."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


SHAPES = (
    (8192, 29568, 8192),
    (8192, 49152, 12288),
    (8192, 53248, 16384),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--timeout_seconds", type=int, default=1200)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def median(values: list[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    return ordered[n // 2] if n % 2 else (ordered[n // 2 - 1] + ordered[n // 2]) / 2


def main() -> None:
    args = parse_args()
    args.output_root = args.output_root.resolve()
    repo = Path(__file__).resolve().parents[1]
    bench = repo / "benchmark/bench_ar_handoff_ablation.py"
    kernel = repo / "kernels/nvidia/new_windowed_panel_gemm_allreduce_v23.py"
    torchrun = shutil.which("torchrun")
    if torchrun is None:
        raise RuntimeError("torchrun not found")
    args.output_root.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    env.setdefault("NVSHMEM_SYMMETRIC_SIZE", "8589934592")
    manifest = {
        "schema": "ar-handoff-campaign-v1",
        "created_unix": time.time(),
        "experimental_unit": "one independent four-rank torchrun launch",
        "shapes": [{"M": m, "N": n, "K": k} for m, n, k in SHAPES],
        "repetitions": args.repetitions,
        "kernel_sha256": sha256(kernel),
        "benchmark_sha256": sha256(bench),
        "environment": {
            "CUDA_VISIBLE_DEVICES": env.get("CUDA_VISIBLE_DEVICES"),
            "NVSHMEM_SYMMETRIC_SIZE": env.get("NVSHMEM_SYMMETRIC_SIZE"),
            "LD_PRELOAD": env.get("LD_PRELOAD"),
        },
    }
    (args.output_root / "campaign_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    ledger = []
    for shape_index, (m, n, k) in enumerate(SHAPES):
        shape_name = f"{m}x{n}x{k}"
        for repetition in range(1, args.repetitions + 1):
            run_dir = args.output_root / shape_name / f"rep_{repetition:02d}"
            summary_file = run_dir / "summary.json"
            first_mode = "frontier" if (shape_index + repetition) % 2 else "bulk"
            if args.resume and summary_file.exists():
                ledger.append({"shape": shape_name, "repetition": repetition, "status": "existing",
                               "first_mode": first_mode, "returncode": 0, "elapsed_seconds": 0.0})
                continue
            run_dir.mkdir(parents=True, exist_ok=True)
            command = [
                torchrun, "--standalone", "--nnodes=1", "--nproc_per_node=4", str(bench),
                "--M", str(m), "--N", str(n), "--K", str(k), "--dtype", "bfloat16",
                "--chunk_rows", "1024", "--active_chunk_window", "4", "--n_bands", "1",
                "--stage_slots", "4", "--comm_lanes", "2", "--num_comm_sms", "64",
                "--warmup_pairs", "5", "--measured_pairs", "20", "--first_mode", first_mode,
                "--repetition", str(repetition), "--output_dir", str(run_dir),
            ]
            (run_dir / "command.json").write_text(json.dumps(command, indent=2) + "\n", encoding="utf-8")
            started = time.time()
            status = "completed"
            returncode = 0
            try:
                completed = subprocess.run(
                    command,
                    cwd=repo,
                    env=env,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    timeout=args.timeout_seconds,
                    check=False,
                )
                returncode = completed.returncode
                output = completed.stdout
                if returncode != 0 or not summary_file.exists():
                    status = "failed"
            except subprocess.TimeoutExpired as exc:
                status = "timeout"
                returncode = 124
                output = (exc.stdout or "") + "\n[TIMEOUT]\n"
            (run_dir / "console.log").write_text(output, encoding="utf-8")
            elapsed = time.time() - started
            row = {"shape": shape_name, "repetition": repetition, "status": status,
                   "first_mode": first_mode, "returncode": returncode, "elapsed_seconds": elapsed}
            ledger.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)

    ledger_fields = ("shape", "repetition", "status", "first_mode", "returncode", "elapsed_seconds")
    with (args.output_root / "run_ledger.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ledger_fields)
        writer.writeheader()
        writer.writerows(ledger)

    launch_rows = []
    for m, n, k in SHAPES:
        shape_name = f"{m}x{n}x{k}"
        for repetition in range(1, args.repetitions + 1):
            summary_file = args.output_root / shape_name / f"rep_{repetition:02d}" / "summary.json"
            if not summary_file.exists():
                continue
            summary = json.loads(summary_file.read_text(encoding="utf-8"))
            launch_rows.append({
                "M": m, "N": n, "K": k, "shape": shape_name, "repetition": repetition,
                "first_mode": summary["first_mode"],
                "bulk_median_rank_max_ms": summary["median_rank_max_ms"]["bulk"],
                "frontier_median_rank_max_ms": summary["median_rank_max_ms"]["frontier"],
                "frontier_speedup_over_bulk": summary["frontier_speedup_over_bulk"],
                "scatter_bytes_per_pe": summary["symmetric_staging"]["scatter_bytes_per_pe"],
            })
    launch_fields = tuple(launch_rows[0].keys()) if launch_rows else (
        "M", "N", "K", "shape", "repetition", "first_mode", "bulk_median_rank_max_ms",
        "frontier_median_rank_max_ms", "frontier_speedup_over_bulk", "scatter_bytes_per_pe"
    )
    with (args.output_root / "per_launch.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=launch_fields)
        writer.writeheader()
        writer.writerows(launch_rows)

    aggregate_rows = []
    for m, n, k in SHAPES:
        rows = [row for row in launch_rows if row["M"] == m and row["N"] == n and row["K"] == k]
        if not rows:
            continue
        speeds = [float(row["frontier_speedup_over_bulk"]) for row in rows]
        bulks = [float(row["bulk_median_rank_max_ms"]) for row in rows]
        frontiers = [float(row["frontier_median_rank_max_ms"]) for row in rows]
        aggregate_rows.append({
            "M": m, "N": n, "K": k, "completed_launches": len(rows),
            "bulk_median_of_launch_medians_ms": median(bulks),
            "frontier_median_of_launch_medians_ms": median(frontiers),
            "frontier_speedup_median": median(speeds),
            "frontier_speedup_min": min(speeds),
            "frontier_speedup_max": max(speeds),
            "scatter_bytes_per_pe": rows[0]["scatter_bytes_per_pe"],
        })
    aggregate_fields = tuple(aggregate_rows[0].keys()) if aggregate_rows else (
        "M", "N", "K", "completed_launches", "bulk_median_of_launch_medians_ms",
        "frontier_median_of_launch_medians_ms", "frontier_speedup_median", "frontier_speedup_min",
        "frontier_speedup_max", "scatter_bytes_per_pe"
    )
    with (args.output_root / "aggregate.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=aggregate_fields)
        writer.writeheader()
        writer.writerows(aggregate_rows)
    print(json.dumps({"completed_launches": len(launch_rows), "aggregate_rows": aggregate_rows}, sort_keys=True),
          flush=True)
    if len(launch_rows) != len(SHAPES) * args.repetitions:
        raise SystemExit("campaign incomplete; inspect run_ledger.csv and child console.log files")


if __name__ == "__main__":
    main()
