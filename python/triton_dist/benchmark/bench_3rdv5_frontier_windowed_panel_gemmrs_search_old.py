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
import math
import os
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
BENCH_SCRIPT = ROOT / "python" / "triton_dist" / "benchmark" / "bench_3rdv5_frontier_windowed_panel_gemmrs.py"
KV_RE = re.compile(r"([A-Za-z0-9_]+)=([^\s,]+)")
RANK_RE = re.compile(r"^Rank\s+(\d+)\s+\[(.*?)\]\s+latency\s+\(ms\):\s+(.*)$")

# python python/triton_dist/benchmark/bench_3rdv5_frontier_windowed_panel_gemmrs_search.py \
#   --nproc_per_node 8 \
#   --M 16384 --N 29568 --K 8192 \
#   --autotune \
#   --fast_iters 4 --fast_warmup_iters 2 \
#   --iters 10 --warmup_iters 5 \
#   --search_chunk_rows_list 512,1024,2048 \
#   --search_active_chunk_window_list 1,2,4 \
#   --search_stage_slots_list 2,4,8 \
#   --search_steady_sms_list 8,12,16 \
#   --search_tail_sms_list 16,20,24 \
#   --search_comm_lanes_list 1,2,4 \
#   --search_n_bands_list 1,2 \
#   --search_frontier_chunks_list 1,2 \
#   --topk 8 \
#   --verify_topk 3 \
#   --quiet_subprocess


def parse_int_list_arg(value: str | None) -> list[int] | None:
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def unique_preserve_order(values):
    seen = set()
    result = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def append_bool_flag(cmd: list[str], flag: str, value: bool) -> None:
    cmd.append(flag if value else f"--no-{flag[2:]}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Search best params for bench_3rdv5_frontier_windowed_panel_gemmrs.py with separate torchrun jobs."
    )
    parser.add_argument("--nproc_per_node", type=int, required=True)
    parser.add_argument("--M", type=int, required=True)
    parser.add_argument("--N", type=int, required=True)
    parser.add_argument("--K", type=int, required=True)
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    parser.add_argument("--mode", default="v2", choices=["v2", "all"])
    parser.add_argument("--autotune", default=False, action=argparse.BooleanOptionalAction)
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
    parser.add_argument("--dump_csv", action="store_true", default=False)
    parser.add_argument("--quiet_subprocess", action="store_true", default=False)
    parser.add_argument("--torchrun_bin", default="torchrun")

    parser.add_argument("--run_profile_best", action="store_true", default=False)
    parser.add_argument("--profile_target", default="v2", choices=["all", "v2", "new_3rd", "torch"])
    parser.add_argument("--profile_merge_group", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--profile_with_stack", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--profile_barrier_after_merge", default=False, action=argparse.BooleanOptionalAction)

    parser.add_argument("--search_chunk_rows_list", type=str, default="")
    parser.add_argument("--search_active_chunk_window_list", type=str, default="")
    parser.add_argument("--search_stage_slots_list", type=str, default="")
    parser.add_argument("--search_steady_sms_list", type=str, default="")
    parser.add_argument("--search_tail_sms_list", type=str, default="")
    parser.add_argument("--search_comm_lanes_list", type=str, default="")
    parser.add_argument("--search_n_bands_list", type=str, default="")
    parser.add_argument("--search_frontier_chunks_list", type=str, default="")
    return parser.parse_args()


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


def finite_floats(per_rank: dict[int, dict[str, object]], key: str) -> list[float]:
    values = []
    for metrics in per_rank.values():
        value = metrics.get(key)
        if isinstance(value, (float, int)):
            value = float(value)
            if not math.isnan(value):
                values.append(value)
    return values


def summarize_metrics(per_rank: dict[int, dict[str, object]], candidate: dict[str, int]) -> dict[str, object]:
    v2_total = finite_floats(per_rank, "v2_total")
    torch_total = finite_floats(per_rank, "torch_total")
    speedup = finite_floats(per_rank, "v2_speedup_vs_torch")
    overlap = finite_floats(per_rank, "v2_internal_overlap")
    v2_gemm = finite_floats(per_rank, "v2_gemm_only")
    v2_rs = finite_floats(per_rank, "v2_rs_only")

    return {
        **candidate,
        "rank_count": len(per_rank),
        "torch_total_ms_mean": sum(torch_total) / len(torch_total) if torch_total else float("nan"),
        "v2_total_ms_mean": sum(v2_total) / len(v2_total) if v2_total else float("nan"),
        "v2_total_ms_max": max(v2_total) if v2_total else float("nan"),
        "v2_gemm_only_ms_mean": sum(v2_gemm) / len(v2_gemm) if v2_gemm else float("nan"),
        "v2_rs_only_ms_mean": sum(v2_rs) / len(v2_rs) if v2_rs else float("nan"),
        "v2_speedup_vs_torch_mean": sum(speedup) / len(speedup) if speedup else float("nan"),
        "v2_speedup_vs_torch_min": min(speedup) if speedup else float("nan"),
        "v2_internal_overlap_mean": sum(overlap) / len(overlap) if overlap else float("nan"),
        "lead_ratio": candidate["active_chunk_window"] * candidate["n_bands"] / max(candidate["stage_slots"], 1),
    }


def default_search_lists(args, m_per_rank: int):
    chunk_rows_list = parse_int_list_arg(args.search_chunk_rows_list)
    active_window_list = parse_int_list_arg(args.search_active_chunk_window_list)
    stage_slots_list = parse_int_list_arg(args.search_stage_slots_list)
    steady_sms_list = parse_int_list_arg(args.search_steady_sms_list)
    tail_sms_list = parse_int_list_arg(args.search_tail_sms_list)
    comm_lanes_list = parse_int_list_arg(args.search_comm_lanes_list)
    n_bands_list = parse_int_list_arg(args.search_n_bands_list)
    frontier_chunks_list = parse_int_list_arg(args.search_frontier_chunks_list)

    if chunk_rows_list is None:
        chunk_rows_list = [512, 1024, 2048]
    if active_window_list is None:
        active_window_list = [1, 2, 4]
    if stage_slots_list is None:
        stage_slots_list = [2, 4, 8]
    if steady_sms_list is None:
        steady_sms_list = [8, 12, 16]
    if tail_sms_list is None:
        tail_sms_list = [16, 20, 24]
    if comm_lanes_list is None:
        comm_lanes_list = [1, 2, 4]
    if n_bands_list is None:
        n_bands_list = [1, 2] if args.N <= 32768 else [1, 2, 4]
    if frontier_chunks_list is None:
        frontier_chunks_list = [1, 2]

    return {
        "chunk_rows_list": unique_preserve_order([x for x in chunk_rows_list if 0 < x <= m_per_rank]),
        "active_window_list": unique_preserve_order([x for x in active_window_list if x > 0]),
        "stage_slots_list": unique_preserve_order([x for x in stage_slots_list if x > 0]),
        "steady_sms_list": unique_preserve_order([x for x in steady_sms_list if x > 0]),
        "tail_sms_list": unique_preserve_order([x for x in tail_sms_list if x > 0]),
        "comm_lanes_list": unique_preserve_order([x for x in comm_lanes_list if x > 0]),
        "n_bands_list": unique_preserve_order([x for x in n_bands_list if x > 0]),
        "frontier_chunks_list": unique_preserve_order([x for x in frontier_chunks_list if x > 0]),
    }


def generate_candidates(args) -> list[dict[str, int]]:
    m_per_rank = args.M // args.nproc_per_node
    search_lists = default_search_lists(args, m_per_rank)
    candidates = []
    seen = set()
    for chunk_rows in search_lists["chunk_rows_list"]:
        num_chunks = max(1, math.ceil(m_per_rank / max(1, chunk_rows)))
        for active_chunk_window in search_lists["active_window_list"]:
            active_chunk_window = max(1, min(active_chunk_window, num_chunks))
            for n_bands in search_lists["n_bands_list"]:
                effective_n_bands = max(1, min(n_bands, args.N))
                for frontier_chunks in search_lists["frontier_chunks_list"]:
                    frontier_chunks = max(1, min(frontier_chunks, active_chunk_window, num_chunks))
                    for stage_slots in search_lists["stage_slots_list"]:
                        stage_slots = max(1, min(stage_slots, num_chunks * effective_n_bands))
                        for steady_sms in search_lists["steady_sms_list"]:
                            for tail_sms in search_lists["tail_sms_list"]:
                                for comm_lanes in search_lists["comm_lanes_list"]:
                                    comm_lanes = max(1, min(comm_lanes, args.nproc_per_node))
                                    candidate = {
                                        "chunk_rows": chunk_rows,
                                        "active_chunk_window": active_chunk_window,
                                        "stage_slots": stage_slots,
                                        "steady_sms": steady_sms,
                                        "tail_sms": tail_sms,
                                        "comm_lanes": comm_lanes,
                                        "n_bands": effective_n_bands,
                                        "frontier_chunks": frontier_chunks,
                                    }
                                    key = tuple(candidate.values())
                                    if key in seen:
                                        continue
                                    seen.add(key)
                                    candidates.append(candidate)
    return candidates


def build_bench_cmd(args, candidate: dict[str, int], *, iters: int, warmup_iters: int, profile: bool) -> list[str]:
    cmd = [
        args.torchrun_bin,
        "--nproc_per_node",
        str(args.nproc_per_node),
        str(BENCH_SCRIPT),
        "--M",
        str(args.M),
        "--N",
        str(args.N),
        "--K",
        str(args.K),
        "--mode",
        args.mode,
        "--iters",
        str(iters),
        "--warmup_iters",
        str(warmup_iters),
        "--dtype",
        args.dtype,
        "--target_chunks_per_rank",
        str(args.target_chunks_per_rank),
        "--min_chunk_rows",
        str(args.min_chunk_rows),
        "--tail_chunk_window",
        str(args.tail_chunk_window),
        "--chunk_rows",
        str(candidate["chunk_rows"]),
        "--active_chunk_window",
        str(candidate["active_chunk_window"]),
        "--stage_slots",
        str(candidate["stage_slots"]),
        "--steady_sms",
        str(candidate["steady_sms"]),
        "--tail_sms",
        str(candidate["tail_sms"]),
        "--comm_lanes",
        str(candidate["comm_lanes"]),
        "--n_bands",
        str(candidate["n_bands"]),
        "--frontier_chunks",
        str(candidate["frontier_chunks"]),
        "--profile_target",
        args.profile_target,
    ]
    append_bool_flag(cmd, "--autotune", args.autotune)
    append_bool_flag(cmd, "--trans_b", args.trans_b)
    append_bool_flag(cmd, "--local_seed_direct", args.local_seed_direct)
    if profile:
        cmd.append("--profile")
        append_bool_flag(cmd, "--profile_merge_group", args.profile_merge_group)
        append_bool_flag(cmd, "--profile_with_stack", args.profile_with_stack)
        append_bool_flag(cmd, "--profile_barrier_after_merge", args.profile_barrier_after_merge)
    return cmd


def run_and_capture(cmd: list[str], quiet: bool) -> tuple[int, str]:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    process = subprocess.run(
        cmd,
        cwd=str(ROOT),
        text=True,
        capture_output=True,
        env=env,
    )
    merged = process.stdout + ("\n" + process.stderr if process.stderr else "")
    if not quiet:
        if process.stdout:
            print(process.stdout, end="")
        if process.stderr:
            print(process.stderr, end="", file=sys.stderr)
    return process.returncode, merged


def format_candidate(candidate: dict[str, int]) -> str:
    return (
        f"chunk={candidate['chunk_rows']}, window={candidate['active_chunk_window']}, "
        f"stage={candidate['stage_slots']}, steady_sms={candidate['steady_sms']}, tail_sms={candidate['tail_sms']}, "
        f"lanes={candidate['comm_lanes']}, bands={candidate['n_bands']}, frontier={candidate['frontier_chunks']}"
    )


def format_result(result: dict[str, object]) -> str:
    return (
        f"max_total={result['v2_total_ms_max']:.4f} ms, mean_total={result['v2_total_ms_mean']:.4f} ms, "
        f"speedup_mean={result['v2_speedup_vs_torch_mean']:.4f}, speedup_min={result['v2_speedup_vs_torch_min']:.4f}, "
        f"overlap_mean={result['v2_internal_overlap_mean']:.2%}, lead_ratio={result['lead_ratio']:.3f}"
    )


def result_sort_key(result: dict[str, object]):
    return (
        float(result["v2_total_ms_max"]),
        float(result["v2_total_ms_mean"]),
        -float(result["v2_speedup_vs_torch_mean"]),
        -float(result["v2_speedup_vs_torch_min"]),
    )


def write_csv(csv_file: Path, results: list[dict[str, object]]) -> None:
    fields = [
        "chunk_rows",
        "active_chunk_window",
        "stage_slots",
        "steady_sms",
        "tail_sms",
        "comm_lanes",
        "n_bands",
        "frontier_chunks",
        "lead_ratio",
        "rank_count",
        "torch_total_ms_mean",
        "v2_total_ms_mean",
        "v2_total_ms_max",
        "v2_gemm_only_ms_mean",
        "v2_rs_only_ms_mean",
        "v2_speedup_vs_torch_mean",
        "v2_speedup_vs_torch_min",
        "v2_internal_overlap_mean",
    ]
    with open(csv_file, "w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=fields)
        writer.writeheader()
        for item in results:
            writer.writerow({name: item.get(name) for name in fields})


def print_topk(title: str, results: list[dict[str, object]], topk: int) -> None:
    print(f"[search] {title}", flush=True)
    for idx, item in enumerate(results[:topk], start=1):
        print(f"  #{idx}: {format_result(item)} | {format_candidate(item)}", flush=True)


def print_best_command(args, best: dict[str, object]) -> None:
    cmd = build_bench_cmd(args, best, iters=args.iters, warmup_iters=args.warmup_iters, profile=False)
    print("[search] best command:", flush=True)
    print(" ".join(cmd), flush=True)


def main():
    args = parse_args()

    if "LOCAL_RANK" in os.environ or "RANK" in os.environ:
        raise SystemExit(
            "This script is a driver wrapper. Please run it with `python`, not with `torchrun`.\n"
            "Example:\n"
            "python python/triton_dist/benchmark/bench_3rdv5_frontier_windowed_panel_gemmrs_search.py --nproc_per_node=8 ..."
        )

    if args.M % args.nproc_per_node != 0:
        raise SystemExit("--M must be divisible by --nproc_per_node for this search driver.")
    if args.K % args.nproc_per_node != 0:
        raise SystemExit("--K must be divisible by --nproc_per_node for this search driver.")

    candidates = generate_candidates(args)
    print(
        f"[search] evaluating {len(candidates)} candidates for M={args.M}, N={args.N}, K={args.K}, nproc={args.nproc_per_node}",
        flush=True,
    )

    coarse_results = []
    failures = []
    total_candidates = len(candidates)
    for idx, candidate in enumerate(candidates, start=1):
        cmd = build_bench_cmd(
            args,
            candidate,
            iters=args.fast_iters,
            warmup_iters=args.fast_warmup_iters,
            profile=False,
        )
        returncode, output = run_and_capture(cmd, quiet=args.quiet_subprocess)
        if returncode != 0:
            failures.append((candidate, returncode))
            print(
                f"[search] candidate {idx}/{total_candidates} failed(rc={returncode}): {format_candidate(candidate)}",
                flush=True,
            )
            continue

        per_rank = parse_rank_metrics(output)
        if not per_rank:
            failures.append((candidate, -1))
            print(
                f"[search] candidate {idx}/{total_candidates} failed(parse): {format_candidate(candidate)}",
                flush=True,
            )
            continue

        summary = summarize_metrics(per_rank, candidate)
        coarse_results.append(summary)
        print(
            f"[search] candidate {idx}/{total_candidates}: {format_result(summary)} | {format_candidate(candidate)}",
            flush=True,
        )

    if not coarse_results:
        raise SystemExit(f"[search] all candidates failed, failures={len(failures)}")

    coarse_results.sort(key=result_sort_key)
    print_topk("Top Candidates", coarse_results, args.topk)
    print_best_command(args, coarse_results[0])

    verify_count = min(args.verify_topk, len(coarse_results))
    verified_results = []
    for idx in range(verify_count):
        candidate = {name: int(coarse_results[idx][name]) for name in [
            "chunk_rows",
            "active_chunk_window",
            "stage_slots",
            "steady_sms",
            "tail_sms",
            "comm_lanes",
            "n_bands",
            "frontier_chunks",
        ]}
        print(f"[search] verifying top candidate #{idx + 1}: {format_candidate(candidate)}", flush=True)
        cmd = build_bench_cmd(args, candidate, iters=args.iters, warmup_iters=args.warmup_iters, profile=False)
        returncode, output = run_and_capture(cmd, quiet=args.quiet_subprocess)
        if returncode != 0:
            print(f"[search] verify failed(rc={returncode}): {format_candidate(candidate)}", flush=True)
            continue
        per_rank = parse_rank_metrics(output)
        if not per_rank:
            print(f"[search] verify failed(parse): {format_candidate(candidate)}", flush=True)
            continue
        verified_results.append(summarize_metrics(per_rank, candidate))

    if verified_results:
        verified_results.sort(key=result_sort_key)
        print_topk("Verified Top Candidates", verified_results, min(args.topk, len(verified_results)))

    best_for_profile = verified_results[0] if verified_results else coarse_results[0]

    if args.dump_csv:
        csv_dir = ROOT / "csv"
        csv_dir.mkdir(exist_ok=True)
        csv_file = csv_dir / f"perf_3rdv5_frontier_windowed_panel_gemmrs_search_{args.nproc_per_node}_ranks.csv"
        write_csv(csv_file, coarse_results)
        print(f"[search] csv file is dumped into {csv_file}", flush=True)

    if args.run_profile_best:
        candidate = {name: int(best_for_profile[name]) for name in [
            "chunk_rows",
            "active_chunk_window",
            "stage_slots",
            "steady_sms",
            "tail_sms",
            "comm_lanes",
            "n_bands",
            "frontier_chunks",
        ]}
        print(f"[search] running profile for best candidate: {format_candidate(candidate)}", flush=True)
        cmd = build_bench_cmd(args, candidate, iters=args.iters, warmup_iters=args.warmup_iters, profile=True)
        returncode, _ = run_and_capture(cmd, quiet=args.quiet_subprocess)
        if returncode != 0:
            raise SystemExit(f"[search] best-candidate profile run failed with rc={returncode}")


if __name__ == "__main__":
    main()
