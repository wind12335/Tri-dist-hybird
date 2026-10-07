#!/usr/bin/env python3
"""Run the AG ready-granularity sweep across multiple shapes and plot them together.

``plot_ag_ready_ablation.py`` already renders one row of panels per shape when
given several aggregated CSVs; this orchestrator just drives the sweep for the
shapes that still need data and feeds every shape's ``ag_ready_aggregated.csv``
to the plot.

Shapes come from two sources (both optional):
  --shape_list "M,N,K;M,N,K"   shapes to sweep fresh (one driver invocation each,
                               so a failure in one shape does not lose the others)
  --reuse   dir1,dir2,...      existing shape result dirs whose newest
                               ag_ready_aggregated.csv should be included as-is

Run this after ``source ./scripts/setenv.sh`` (the driver child needs the env).

Usage::

  python benchmark/bench_ag_ready_granularity_multi.py \\
      --shape_list "16384x49152x12288;28672x53248x16384" \\
      --reuse benchmark/ag_ready_granularity_eval_results/16384x29568x8192,\\
              benchmark/ag_ready_granularity_eval_results/28672x28672x8192 \\
      --granularity_values 2048,1024,512,256 --repeats 2

Output: ``ag_ready_ablation.png/.svg/.pdf`` in ``overleaf_methodology_zh/figures``
(one row of panels per shape).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCH_DIR.parents[2]
DRIVER = BENCH_DIR / "bench_ag_ready_granularity_sweep_eval.py"
PLOT = BENCH_DIR / "plot_ag_ready_ablation.py"
DEFAULT_OUTPUT_ROOT = BENCH_DIR / "ag_ready_granularity_eval_results"
DEFAULT_FIG_DIR = REPO_ROOT / "overleaf_methodology_zh" / "figures"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--shape_list", default="", help="'M,N,K;M,N,K' shapes to sweep fresh")
    p.add_argument("--reuse", default="", help="comma-separated result dirs to reuse their newest aggregated CSV")
    p.add_argument("--granularity_values", default="2048,1024,512,256")
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--profile", default=False, action=argparse.BooleanOptionalAction)
    p.add_argument("--timeout_sec", type=float, default=300.0)
    p.add_argument("--min_m_per_rank_for_tile_ready", type=int, default=4096)
    p.add_argument("--output_root", default=str(DEFAULT_OUTPUT_ROOT))
    p.add_argument("--output_dir", default=str(DEFAULT_FIG_DIR))
    p.add_argument("--formats", default="png,svg,pdf")
    return p.parse_args()


def newest_aggregated(shape_dir: Path) -> Path | None:
    matches = sorted(shape_dir.rglob("ag_ready_aggregated.csv"),
                     key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)

    csvs: list[Path] = []
    for raw in args.reuse.split(","):
        raw = raw.strip()
        if not raw:
            continue
        shape_dir = Path(raw)
        if not shape_dir.is_absolute() and not shape_dir.exists():
            shape_dir = output_root / raw
        found = newest_aggregated(shape_dir)
        if found is None:
            print(f"[warn] no aggregated CSV under {shape_dir}; skipping", file=sys.stderr)
            continue
        csvs.append(found)
        print(f"[reuse] {found}", flush=True)

    for part in args.shape_list.split(";"):
        part = part.strip()
        if not part:
            continue
        m, n, k = (int(x.strip()) for x in part.split(","))
        cmd = [
            sys.executable, str(DRIVER),
            "--M", str(m), "--N", str(n), "--K", str(k),
            "--granularity_values", args.granularity_values,
            "--repeats", str(args.repeats),
            "--min_m_per_rank_for_tile_ready", str(args.min_m_per_rank_for_tile_ready),
            "--timeout_sec", str(args.timeout_sec),
            "--output_root", str(output_root),
        ]
        if args.profile:
            cmd.append("--profile")
        print(f"\n[driver] shape={m}x{n}x{k}:\n  {' '.join(cmd)}", flush=True)
        ret = subprocess.run(cmd, cwd=str(BENCH_DIR)).returncode
        if ret != 0:
            print(f"[error] driver failed for {m}x{n}x{k} (rc={ret}); continuing", file=sys.stderr)
        shape_dir = output_root / f"{m}x{n}x{k}"
        found = newest_aggregated(shape_dir)
        if found is None:
            print(f"[error] no aggregated CSV produced for {m}x{n}x{k}; skipping", file=sys.stderr)
            continue
        csvs.append(found)
        print(f"[fresh] {found}", flush=True)

    if not csvs:
        raise SystemExit("no aggregated CSVs collected (nothing to plot)")

    plot_cmd = [sys.executable, str(PLOT)]
    for c in csvs:
        plot_cmd += ["--input_csv", str(c)]
    plot_cmd += ["--output_dir", args.output_dir, "--formats", args.formats]
    print(f"\n[plot] {' '.join(plot_cmd)}", flush=True)
    ret = subprocess.run(plot_cmd, cwd=str(BENCH_DIR)).returncode
    if ret != 0:
        raise SystemExit(f"plot failed (rc={ret})")
    print(f"\n[done] plotted {len(csvs)} shape(s) -> {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
