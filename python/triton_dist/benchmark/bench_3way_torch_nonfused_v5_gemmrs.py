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
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
BENCH_DIR = ROOT / "python" / "triton_dist" / "benchmark"
NONFUSED_SCRIPT = BENCH_DIR / "benchmark_gemm_reducescatter.py"
V5_SCRIPT = BENCH_DIR / "bench_3rdv5_frontier_windowed_panel_gemmrs.py"
KV_RE = re.compile(r"([A-Za-z0-9_]+)=([^\s,]+)")
RANK_RE = re.compile(r"^Rank\s+(\d+)\s+\[(.*?)\]\s+latency\s+\(ms\):\s+(.*)$")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Driver benchmark: run torch-only, Triton-distributed nonfused, and V5 in separate torchrun jobs, then merge outputs."
    )
    parser.add_argument("--nproc_per_node", type=int, required=True)
    parser.add_argument("--M", type=int, default=8192)
    parser.add_argument("--N", type=int, required=True)
    parser.add_argument("--K", type=int, required=True)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--warmup_iters", type=int, default=5)
    parser.add_argument("--autotune", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--profile", action="store_true", default=False)
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    parser.add_argument("--trans_b", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--chunk_rows", type=int, default=0)
    parser.add_argument("--target_chunks_per_rank", type=int, default=2)
    parser.add_argument("--min_chunk_rows", type=int, default=512)
    parser.add_argument("--active_chunk_window", type=int, default=4)
    parser.add_argument("--stage_slots", type=int, default=4)
    parser.add_argument("--steady_sms", type=int, default=6)
    parser.add_argument("--tail_sms", type=int, default=12)
    parser.add_argument("--tail_chunk_window", type=int, default=1)
    parser.add_argument("--comm_lanes", type=int, default=2)
    parser.add_argument("--n_bands", type=int, default=1)
    parser.add_argument("--frontier_chunks", type=int, default=1)
    parser.add_argument("--local_seed_direct", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--dump_csv", action="store_true", default=False)
    parser.add_argument("--torchrun_bin", default="torchrun")
    parser.add_argument("--quiet_subprocess", action="store_true", default=False)
    return parser.parse_args()


def append_bool_flag(cmd: list[str], flag: str, value: bool) -> None:
    cmd.append(flag if value else f"--no-{flag[2:]}")


def common_v5_args(args) -> list[str]:
    cmd = [
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
        "--chunk_rows",
        str(args.chunk_rows),
        "--target_chunks_per_rank",
        str(args.target_chunks_per_rank),
        "--min_chunk_rows",
        str(args.min_chunk_rows),
        "--active_chunk_window",
        str(args.active_chunk_window),
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
    ]
    append_bool_flag(cmd, "--autotune", args.autotune)
    append_bool_flag(cmd, "--trans_b", args.trans_b)
    append_bool_flag(cmd, "--local_seed_direct", args.local_seed_direct)
    if args.profile:
        cmd.append("--profile")
    return cmd


def build_torch_cmd(args) -> list[str]:
    return [
        args.torchrun_bin,
        "--nproc_per_node",
        str(args.nproc_per_node),
        str(V5_SCRIPT),
        "--mode",
        "torch",
        *common_v5_args(args),
    ]


def build_nonfused_cmd(args) -> list[str]:
    cmd = [
        args.torchrun_bin,
        "--nproc_per_node",
        str(args.nproc_per_node),
        str(NONFUSED_SCRIPT),
        "--M",
        str(args.M),
        "--N",
        str(args.N),
        "--K",
        str(args.K),
        "--mode",
        "nonfused",
        "--iters",
        str(args.iters),
        "--warmup_iters",
        str(args.warmup_iters),
        "--dtype",
        args.dtype,
    ]
    append_bool_flag(cmd, "--autotune", args.autotune)
    append_bool_flag(cmd, "--trans_b", args.trans_b)
    if args.profile:
        cmd.append("--profile")
    return cmd


def build_v5_cmd(args) -> list[str]:
    return [
        args.torchrun_bin,
        "--nproc_per_node",
        str(args.nproc_per_node),
        str(V5_SCRIPT),
        "--mode",
        "v2",
        *common_v5_args(args),
    ]


def run_and_capture(cmd: list[str], quiet: bool) -> str:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    process = subprocess.run(
        cmd,
        cwd=str(ROOT),
        text=True,
        capture_output=True,
        env=env,
    )
    if not quiet:
        if process.stdout:
            print(process.stdout, end="")
        if process.stderr:
            print(process.stderr, end="", file=sys.stderr)
    if process.returncode != 0:
        raise RuntimeError(f"command failed with exit code {process.returncode}: {' '.join(cmd)}")
    return process.stdout + ("\n" + process.stderr if process.stderr else "")


def maybe_float(value: str):
    if value.endswith("%"):
        return float(value[:-1]) / 100.0
    if value.lower() == "nan":
        return float("nan")
    try:
        return float(value)
    except ValueError:
        return value


def parse_rank_metrics(text: str) -> dict[int, dict[str, object]]:
    per_rank: dict[int, dict[str, object]] = {}
    for line in text.splitlines():
        match = RANK_RE.match(line.strip())
        if not match:
            continue
        rank = int(match.group(1))
        kv_text = match.group(3)
        metrics: dict[str, object] = {}
        for key, value in KV_RE.findall(kv_text):
            metrics[key] = maybe_float(value)
        per_rank[rank] = metrics
    return per_rank


def fmt_num(value) -> str:
    if isinstance(value, float):
        if value != value:
            return "nan"
        return f"{value:.2f}"
    return str(value)


def fmt_pct(value) -> str:
    if isinstance(value, float):
        if value != value:
            return "nan"
        return f"{value:.2%}"
    return str(value)


def merge_rank_metrics(
    torch_metrics: dict[str, object],
    nonfused: dict[str, object],
    v5: dict[str, object],
) -> dict[str, object]:
    merged = {
        "torch_total": torch_metrics.get("torch_total", float("nan")),
        "torch_gemm_only": torch_metrics.get("torch_gemm_only", float("nan")),
        "torch_rs_only": torch_metrics.get("torch_rs_only", float("nan")),
        "triton_nonfused_total": nonfused.get("triton_nonfused_total", float("nan")),
        "triton_nonfused_gemm_only": nonfused.get("triton_nonfused_gemm_only", float("nan")),
        "triton_nonfused_rs_only": nonfused.get("triton_nonfused_rs_only", float("nan")),
        "triton_nonfused_internal_overlap": nonfused.get("nonfused_overlap_ratio", float("nan")),
        "v5_total": v5.get("v2_total", float("nan")),
        "v5_gemm_only": v5.get("v2_gemm_only", float("nan")),
        "v5_rs_only": v5.get("v2_rs_only", float("nan")),
        "v5_internal_overlap": v5.get("v2_internal_overlap", float("nan")),
        "chunk_rows": v5.get("chunk_rows", float("nan")),
        "num_chunks": v5.get("num_chunks", float("nan")),
        "active_chunk_window": v5.get("active_chunk_window", float("nan")),
        "stage_slots": v5.get("stage_slots", float("nan")),
        "comm_lanes": v5.get("comm_lanes", float("nan")),
        "n_bands": v5.get("n_bands", float("nan")),
        "frontier_chunks": v5.get("frontier_chunks", float("nan")),
    }

    torch_total = merged["torch_total"]
    nonfused_total = merged["triton_nonfused_total"]
    v5_total = merged["v5_total"]
    merged["triton_nonfused_speedup_vs_torch"] = (
        torch_total / nonfused_total
        if isinstance(torch_total, float) and isinstance(nonfused_total, float) and torch_total == torch_total and nonfused_total == nonfused_total and nonfused_total != 0
        else float("nan")
    )
    merged["v5_speedup_vs_torch"] = (
        torch_total / v5_total
        if isinstance(torch_total, float) and isinstance(v5_total, float) and torch_total == torch_total and v5_total == v5_total and v5_total != 0
        else float("nan")
    )
    merged["v5_speedup_vs_tridist_nonfused"] = (
        nonfused_total / v5_total
        if isinstance(nonfused_total, float) and isinstance(v5_total, float) and nonfused_total == nonfused_total and v5_total == v5_total and v5_total != 0
        else float("nan")
    )
    return merged


def print_summary(merged_per_rank: dict[int, dict[str, object]]) -> None:
    print("[merged] combined latency summary:")
    for rank in sorted(merged_per_rank):
        m = merged_per_rank[rank]
        line = (
            f"Rank {rank} [custom] latency (ms): "
            f"torch_total={fmt_num(m['torch_total'])}, "
            f"torch_gemm_only={fmt_num(m['torch_gemm_only'])}, "
            f"torch_rs_only={fmt_num(m['torch_rs_only'])}, "
            f"triton_nonfused_total={fmt_num(m['triton_nonfused_total'])}, "
            f"triton_nonfused_gemm_only={fmt_num(m['triton_nonfused_gemm_only'])}, "
            f"triton_nonfused_rs_only={fmt_num(m['triton_nonfused_rs_only'])}, "
            f"triton_nonfused_internal_overlap={fmt_pct(m['triton_nonfused_internal_overlap'])}, "
            f"triton_nonfused_speedup_vs_torch={fmt_num(m['triton_nonfused_speedup_vs_torch'])}, "
            f"v5_total={fmt_num(m['v5_total'])}, "
            f"v5_gemm_only={fmt_num(m['v5_gemm_only'])}, "
            f"v5_rs_only={fmt_num(m['v5_rs_only'])}, "
            f"v5_internal_overlap={fmt_pct(m['v5_internal_overlap'])}, "
            f"v5_speedup_vs_torch={fmt_num(m['v5_speedup_vs_torch'])}, "
            f"v5_speedup_vs_tridist_nonfused={fmt_num(m['v5_speedup_vs_tridist_nonfused'])}, "
            f"chunk_rows={fmt_num(m['chunk_rows'])}, "
            f"num_chunks={fmt_num(m['num_chunks'])}, "
            f"active_chunk_window={fmt_num(m['active_chunk_window'])}, "
            f"stage_slots={fmt_num(m['stage_slots'])}, "
            f"comm_lanes={fmt_num(m['comm_lanes'])}, "
            f"n_bands={fmt_num(m['n_bands'])}, "
            f"frontier_chunks={fmt_num(m['frontier_chunks'])}"
        )
        print(line)


def dump_csv(args, merged_per_rank: dict[int, dict[str, object]]) -> None:
    csv_dir = ROOT / "csv"
    csv_dir.mkdir(exist_ok=True)
    csv_path = csv_dir / f"perf_3way_torch_nonfused_v5_gemm_rs_{args.nproc_per_node}_ranks.csv"
    fields = [
        "rank",
        "torch_total",
        "torch_gemm_only",
        "torch_rs_only",
        "triton_nonfused_total",
        "triton_nonfused_gemm_only",
        "triton_nonfused_rs_only",
        "triton_nonfused_internal_overlap",
        "triton_nonfused_speedup_vs_torch",
        "v5_total",
        "v5_gemm_only",
        "v5_rs_only",
        "v5_internal_overlap",
        "v5_speedup_vs_torch",
        "v5_speedup_vs_tridist_nonfused",
        "chunk_rows",
        "num_chunks",
        "active_chunk_window",
        "stage_slots",
        "comm_lanes",
        "n_bands",
        "frontier_chunks",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for rank in sorted(merged_per_rank):
            row = {"rank": rank}
            row.update(merged_per_rank[rank])
            writer.writerow(row)
    print(f"csv file is dumped into {csv_path}")


def main():
    args = parse_args()

    if "LOCAL_RANK" in os.environ or "RANK" in os.environ:
        raise SystemExit(
            "This script is a driver wrapper. Please run it with `python`, not with `torchrun`.\n"
            "Example:\n"
            "python python/triton_dist/benchmark/bench_3way_torch_nonfused_v5_gemmrs.py --nproc_per_node=4 ..."
        )

    print("[driver] running torch-only baseline benchmark...")
    torch_text = run_and_capture(build_torch_cmd(args), quiet=args.quiet_subprocess)

    print("[driver] running Triton-distributed nonfused benchmark...")
    nonfused_text = run_and_capture(build_nonfused_cmd(args), quiet=args.quiet_subprocess)

    print("[driver] running V5 benchmark...")
    v5_text = run_and_capture(build_v5_cmd(args), quiet=args.quiet_subprocess)

    torch_ranks = parse_rank_metrics(torch_text)
    nonfused_ranks = parse_rank_metrics(nonfused_text)
    v5_ranks = parse_rank_metrics(v5_text)

    if not torch_ranks:
        raise RuntimeError("failed to parse rank metrics from torch-only benchmark output")
    if not nonfused_ranks:
        raise RuntimeError("failed to parse rank metrics from Triton-distributed nonfused benchmark output")
    if not v5_ranks:
        raise RuntimeError("failed to parse rank metrics from V5 benchmark output")

    merged_per_rank: dict[int, dict[str, object]] = {}
    common_ranks = sorted(set(torch_ranks) & set(nonfused_ranks) & set(v5_ranks))
    for rank in common_ranks:
        merged_per_rank[rank] = merge_rank_metrics(
            torch_ranks[rank],
            nonfused_ranks[rank],
            v5_ranks[rank],
        )

    if not merged_per_rank:
        raise RuntimeError("no overlapping rank metrics found across torch/nonfused/v5 outputs")

    print_summary(merged_per_rank)

    if args.dump_csv:
        dump_csv(args, merged_per_rank)


if __name__ == "__main__":
    main()
