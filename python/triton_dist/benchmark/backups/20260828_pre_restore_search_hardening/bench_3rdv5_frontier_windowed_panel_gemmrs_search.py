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
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

# python python/triton_dist/benchmark/bench_3rdv5_frontier_windowed_panel_gemmrs_search.py \
#   --nproc_per_node 2 \
#   --M 16384 --N 29568 --K 8192 \
#   --autotune \
#   --fast_iters 4 --fast_warmup_iters 2 \
#   --iters 10 --warmup_iters 5 \
#   --search_strategy two_stage \
#   --fast_budget 50 \
#   --structural_bucket_topk 2 \
#   --sms_refine_strategy focused \
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


ROOT = Path(__file__).resolve().parents[3]
BENCH_SCRIPT = ROOT / "python" / "triton_dist" / "benchmark" / "bench_3rdv5_frontier_windowed_panel_gemmrs.py"
KV_RE = re.compile(r"([A-Za-z0-9_]+)=([^\s,]+)")
RANK_RE = re.compile(r"^Rank\s+(\d+)\s+\[(.*?)\]\s+latency\s+\(ms\):\s+(.*)$")
CANDIDATE_FIELDS = [
    "chunk_rows",
    "active_chunk_window",
    "stage_slots",
    "steady_sms",
    "tail_sms",
    "comm_lanes",
    "n_bands",
    "frontier_chunks",
]


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
    parser.add_argument("--stage1_no_autotune", default=False, action=argparse.BooleanOptionalAction,
                        help="In fast search passes, disable autotune and use the benchmark's default GEMM config.")
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
    parser.add_argument("--search_strategy", default="two_stage", choices=["two_stage", "exhaustive"])
    parser.add_argument("--structural_topk", type=int, default=12,
                        help="Upper bound on how many stage1 structures enter stage2 SMS refinement.")
    parser.add_argument("--structural_bucket_topk", type=int, default=2,
                        help="For each (chunk_rows, active_chunk_window, n_bands) bucket, keep only the top-k micro variants.")
    parser.add_argument("--fast_budget", type=int, default=50,
                        help="Approximate max number of fast-pass runs in two-stage search. <=0 disables the cap.")
    parser.add_argument("--sms_refine_strategy", default="focused", choices=["focused", "full_grid"],
                        help="How many steady_sms/tail_sms points to explore for each selected structure.")
    parser.add_argument("--dump_csv", action="store_true", default=False)
    parser.add_argument(
        "--progress_csv",
        type=str,
        default="",
        help="Optional incremental candidate-attempt ledger. Rewritten after every completed child launch.",
    )
    parser.add_argument(
        "--status_json",
        type=str,
        default="",
        help="Optional compact search-status record. Rewritten after every completed child launch.",
    )
    parser.add_argument("--quiet_subprocess", action="store_true", default=False)
    parser.add_argument("--candidate_timeout_sec", type=float, default=0.0,
                        help="Per-subprocess timeout in seconds. <=0 disables timeout.")
    parser.add_argument("--torchrun_bin", default="torchrun")
    parser.add_argument("--shape_preset",
                        default="none",
                        choices=["none", "gpt3_175b_nbands2"],
                        help="Apply a shape-specific stable search preset unless the corresponding explicit lists are set.")

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
    parser.add_argument(
        "--candidate_plan_csv",
        type=str,
        default="",
        help=(
            "CSV defining the ordered full candidate space. With --plan_only it is written; "
            "during a search it must exactly match the regenerated space."
        ),
    )
    parser.add_argument(
        "--expected_candidate_count",
        type=int,
        default=0,
        help="Optional fail-closed count assertion for the feasible candidate space.",
    )
    parser.add_argument(
        "--plan_only",
        action="store_true",
        default=False,
        help="Write/validate the candidate plan and exit before any benchmark or torchrun launch.",
    )
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

    if args.shape_preset == "gpt3_175b_nbands2":
        if chunk_rows_list is None:
            chunk_rows_list = [512, 1024]
        if active_window_list is None:
            active_window_list = [1, 2, 4]
        if stage_slots_list is None:
            stage_slots_list = [2, 4, 8]
        if steady_sms_list is None:
            steady_sms_list = [4, 6, 8]
        if tail_sms_list is None:
            tail_sms_list = [8, 12, 16]
        if comm_lanes_list is None:
            comm_lanes_list = [1, 2, 4]
        if n_bands_list is None:
            n_bands_list = [2]
        if frontier_chunks_list is None:
            frontier_chunks_list = [1, 2]

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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fin:
        for chunk in iter(lambda: fin.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def candidate_plan_rows(candidates: list[dict[str, int]]) -> list[dict[str, int]]:
    return [
        {"candidate_index": index, **{field: int(candidate[field]) for field in CANDIDATE_FIELDS}}
        for index, candidate in enumerate(candidates, start=1)
    ]


def write_candidate_plan(path: Path, args, candidates: list[dict[str, int]]) -> tuple[Path, Path]:
    """Persist the exact post-clamp candidate list without touching CUDA or torchrun."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = candidate_plan_rows(candidates)
    with path.open("w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=["candidate_index", *CANDIDATE_FIELDS])
        writer.writeheader()
        writer.writerows(rows)

    metadata_path = path.with_suffix(path.suffix + ".json")
    metadata = {
        "schema_version": 1,
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "shape": {"M": args.M, "N": args.N, "K": args.K, "nproc_per_node": args.nproc_per_node},
        "candidate_count": len(candidates),
        "candidate_fields": CANDIDATE_FIELDS,
        "candidate_plan_csv": str(path),
        "candidate_plan_sha256": sha256_file(path),
        "search_list_overrides": {
            "chunk_rows": args.search_chunk_rows_list,
            "active_chunk_window": args.search_active_chunk_window_list,
            "stage_slots": args.search_stage_slots_list,
            "steady_sms": args.search_steady_sms_list,
            "tail_sms": args.search_tail_sms_list,
            "comm_lanes": args.search_comm_lanes_list,
            "n_bands": args.search_n_bands_list,
            "frontier_chunks": args.search_frontier_chunks_list,
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path, metadata_path


def load_candidate_plan(path: Path) -> list[dict[str, int]]:
    with path.open(newline="", encoding="utf-8") as fin:
        reader = csv.DictReader(fin)
        missing_fields = [field for field in CANDIDATE_FIELDS if field not in (reader.fieldnames or [])]
        if missing_fields:
            raise ValueError(f"candidate plan {path} is missing columns: {', '.join(missing_fields)}")
        loaded: list[dict[str, int]] = []
        for row_index, row in enumerate(reader, start=2):
            try:
                loaded.append({field: int(row[field]) for field in CANDIDATE_FIELDS})
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid candidate plan row {row_index} in {path}: {exc}") from exc
    return loaded


def prepare_candidate_space(args) -> list[dict[str, int]]:
    """Generate and optionally freeze/verify the candidate universe before a search."""
    candidates = generate_candidates(args)
    if args.expected_candidate_count > 0 and len(candidates) != args.expected_candidate_count:
        raise SystemExit(
            f"[search] expected {args.expected_candidate_count} feasible candidates, generated {len(candidates)}"
        )

    args.candidate_plan_sha256 = ""
    if not args.candidate_plan_csv:
        if args.plan_only:
            raise SystemExit("[search] --plan_only requires --candidate_plan_csv")
        return candidates

    plan_path = Path(args.candidate_plan_csv)
    if args.plan_only:
        csv_path, metadata_path = write_candidate_plan(plan_path, args, candidates)
        args.candidate_plan_sha256 = sha256_file(csv_path)
        print(
            f"[search] candidate plan: {len(candidates)} candidates -> {csv_path} "
            f"(sha256={args.candidate_plan_sha256})",
            flush=True,
        )
        print(f"[search] candidate plan metadata: {metadata_path}", flush=True)
        return candidates

    if not plan_path.is_file():
        raise SystemExit(f"[search] candidate plan does not exist: {plan_path}")
    expected_plan = load_candidate_plan(plan_path)
    if expected_plan != candidates:
        raise SystemExit(
            "[search] regenerated candidate space does not match --candidate_plan_csv; "
            "rerun --plan_only and inspect the list overrides before benchmarking."
        )
    args.candidate_plan_sha256 = sha256_file(plan_path)
    print(
        f"[search] candidate plan verified: {len(candidates)} candidates <- {plan_path} "
        f"(sha256={args.candidate_plan_sha256})",
        flush=True,
    )
    return candidates


def candidate_key(candidate: dict[str, int]) -> tuple[int, ...]:
    return tuple(int(candidate[name]) for name in CANDIDATE_FIELDS)


def structural_key(candidate: dict[str, int]) -> tuple[int, ...]:
    return tuple(int(candidate[name]) for name in [
        "chunk_rows",
        "active_chunk_window",
        "stage_slots",
        "comm_lanes",
        "n_bands",
        "frontier_chunks",
    ])


def choose_mid_value(values: list[int]) -> int:
    assert values, "values should not be empty"
    return values[len(values) // 2]


def choose_focus_values(values: list[int]) -> list[int]:
    assert values, "values should not be empty"
    if len(values) <= 3:
        return list(values)
    return unique_preserve_order([values[0], values[len(values) // 2], values[-1]])


def structural_bucket_key(candidate: dict[str, int]) -> tuple[int, int, int]:
    return (
        int(candidate["chunk_rows"]),
        int(candidate["active_chunk_window"]),
        int(candidate["n_bands"]),
    )


def structural_prescore(args, candidate: dict[str, int]) -> tuple[float, float, int, int, tuple[int, ...]]:
    m_per_rank = args.M // args.nproc_per_node
    num_chunks = max(1, math.ceil(m_per_rank / max(1, candidate["chunk_rows"])))
    pipeline_width = max(1, candidate["active_chunk_window"] * candidate["n_bands"])
    stage_slots = max(1, candidate["stage_slots"])
    lead_ratio = pipeline_width / stage_slots

    # We prefer a balanced queue depth: too shallow starves RS, too deep adds buffering/launch overhead.
    lead_penalty = abs(math.log2(max(lead_ratio, 1e-6) / 0.75))

    # Lanes above useful concurrency often add overhead without improving progress.
    useful_lanes = max(1, min(args.nproc_per_node, max(candidate["n_bands"], min(2, pipeline_width))))
    lane_penalty = abs(candidate["comm_lanes"] - useful_lanes) / useful_lanes

    # Frontier should be present, but normally does not need to exceed a small fraction of the active window.
    frontier_ratio = candidate["frontier_chunks"] / max(1, candidate["active_chunk_window"])
    frontier_penalty = abs(frontier_ratio - 0.5)

    # Reject shapes that clearly underfill the pipeline or expose almost no slack beyond the active window.
    chunk_slack_penalty = 0.0
    if num_chunks < candidate["active_chunk_window"]:
        chunk_slack_penalty += 4.0
    elif num_chunks < candidate["active_chunk_window"] + candidate["frontier_chunks"]:
        chunk_slack_penalty += 1.0

    stage_balance_penalty = 0.0
    if stage_slots < candidate["active_chunk_window"]:
        stage_balance_penalty += 0.5
    if stage_slots > pipeline_width * 2:
        stage_balance_penalty += 0.5

    # For large N, n_bands=1 is still allowed, but we lightly prefer keeping some band parallelism.
    band_penalty = 0.15 if args.N >= 16384 and candidate["n_bands"] == 1 else 0.0

    score = (
        2.5 * lead_penalty
        + 0.75 * lane_penalty
        + 0.5 * frontier_penalty
        + chunk_slack_penalty
        + stage_balance_penalty
        + band_penalty
    )
    return (
        score,
        abs(lead_ratio - 0.75),
        abs(candidate["comm_lanes"] - useful_lanes),
        abs(candidate["frontier_chunks"] - min(2, candidate["active_chunk_window"])),
        candidate_key(candidate),
    )


def build_sms_refine_pairs(search_lists: dict[str, list[int]], strategy: str) -> list[tuple[int, int]]:
    steady_values = list(search_lists["steady_sms_list"])
    tail_values = list(search_lists["tail_sms_list"])
    if strategy == "full_grid":
        return [(steady_sms, tail_sms) for steady_sms in steady_values for tail_sms in tail_values]

    steady_focus = choose_focus_values(steady_values)
    tail_focus = choose_focus_values(tail_values)
    steady_mid = choose_mid_value(steady_focus)
    tail_mid = choose_mid_value(tail_focus)
    pairs = unique_preserve_order([
        (steady_mid, tail_mid),
        (steady_focus[0], tail_mid),
        (steady_focus[-1], tail_mid),
        (steady_mid, tail_focus[0]),
        (steady_mid, tail_focus[-1]),
    ])
    return pairs


def generate_structural_candidates(args) -> tuple[list[dict[str, int]], dict[str, list[int]]]:
    m_per_rank = args.M // args.nproc_per_node
    search_lists = default_search_lists(args, m_per_rank)
    base_steady_sms = choose_mid_value(search_lists["steady_sms_list"])
    base_tail_sms = choose_mid_value(search_lists["tail_sms_list"])

    candidates = []
    seen = set()
    for candidate in generate_candidates(args):
        struct_key = structural_key(candidate)
        if struct_key in seen:
            continue
        seen.add(struct_key)
        new_candidate = dict(candidate)
        new_candidate["steady_sms"] = base_steady_sms
        new_candidate["tail_sms"] = base_tail_sms
        candidates.append(new_candidate)
    return candidates, search_lists


def reduce_structural_candidates(args,
                                 structural_candidates: list[dict[str, int]],
                                 search_lists: dict[str, list[int]]) -> tuple[list[dict[str, int]], list[tuple[int, int]], int]:
    buckets: dict[tuple[int, int, int], list[dict[str, int]]] = {}
    for candidate in structural_candidates:
        buckets.setdefault(structural_bucket_key(candidate), []).append(candidate)

    reduced_candidates = []
    bucket_topk = max(1, args.structural_bucket_topk)
    for bucket_key in sorted(buckets.keys()):
        bucket_candidates = buckets[bucket_key]
        bucket_candidates.sort(key=lambda candidate: structural_prescore(args, candidate))
        reduced_candidates.extend(bucket_candidates[:bucket_topk])

    reduced_candidates.sort(key=lambda candidate: structural_prescore(args, candidate))
    sms_pairs = build_sms_refine_pairs(search_lists, args.sms_refine_strategy)

    if args.fast_budget > 0 and reduced_candidates:
        reserved_refine_topk = min(max(1, args.structural_topk), 3)
        reserved_stage2_runs = reserved_refine_topk * max(1, len(sms_pairs))
        stage1_budget = max(1, args.fast_budget - reserved_stage2_runs)
        reduced_candidates = reduced_candidates[:min(len(reduced_candidates), stage1_budget)]

    return reduced_candidates, sms_pairs, len(buckets)


def expand_sms_candidates(structural_candidate: dict[str, int], sms_pairs: list[tuple[int, int]]) -> list[dict[str, int]]:
    candidates = []
    for steady_sms, tail_sms in sms_pairs:
        candidate = dict(structural_candidate)
        candidate["steady_sms"] = steady_sms
        candidate["tail_sms"] = tail_sms
        candidates.append(candidate)
    return candidates


def build_bench_cmd(args,
                    candidate: dict[str, int],
                    *,
                    iters: int,
                    warmup_iters: int,
                    profile: bool,
                    autotune_override: bool | None = None) -> list[str]:
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
    append_bool_flag(cmd, "--autotune", args.autotune if autotune_override is None else autotune_override)
    append_bool_flag(cmd, "--trans_b", args.trans_b)
    append_bool_flag(cmd, "--local_seed_direct", args.local_seed_direct)
    if profile:
        cmd.append("--profile")
        append_bool_flag(cmd, "--profile_merge_group", args.profile_merge_group)
        append_bool_flag(cmd, "--profile_with_stack", args.profile_with_stack)
        append_bool_flag(cmd, "--profile_barrier_after_merge", args.profile_barrier_after_merge)
    return cmd


def run_and_capture(cmd: list[str], quiet: bool, timeout_sec: float = 0.0) -> tuple[int, str]:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    try:
        process = subprocess.run(
            cmd,
            cwd=str(ROOT),
            text=True,
            capture_output=True,
            env=env,
            timeout=timeout_sec if timeout_sec > 0 else None,
        )
        merged = process.stdout + ("\n" + process.stderr if process.stderr else "")
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        merged = stdout + ("\n" + stderr if stderr else "")
        merged += f"\n[search] timeout after {timeout_sec:.1f}s"
        if not quiet:
            if stdout:
                print(stdout, end="")
            if stderr:
                print(stderr, end="", file=sys.stderr)
            print(f"[search] candidate timed out after {timeout_sec:.1f}s", flush=True)
        return 124, merged
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


PROGRESS_FIELDS = [
    "phase",
    "attempt_index",
    "planned_in_phase",
    "status",
    "failure_kind",
    "returncode",
    "elapsed_sec",
    "correctness_failure_marker",
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
CORRECTNESS_FAILURE_RE = re.compile(
    r"(?:\bcorrectness(?:\s+check)?\s+failed\b|\b(?:correctness|allclose)\s*(?:=|:)\s*(?:false|0|failed?))",
    re.IGNORECASE,
)


def json_safe(value: object) -> object:
    """Convert the numeric summaries to strict JSON without serializing NaN."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


class SearchLedger:
    """Persist partial search evidence without changing candidate selection semantics."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.progress_csv = Path(args.progress_csv) if args.progress_csv else None
        self.status_json = Path(args.status_json) if args.status_json else None
        self.attempts: list[dict[str, object]] = []
        self.candidate_space_total: int | None = None
        self.planned_coarse_runs: int | None = None
        self.completion_status = "running"
        self.failure_reason = ""

    def set_plan(self, *, candidate_space_total: int, planned_coarse_runs: int) -> None:
        self.candidate_space_total = candidate_space_total
        self.planned_coarse_runs = planned_coarse_runs
        self.flush()

    def record(self,
               *,
               phase: str,
               attempt_index: int,
               planned_in_phase: int,
               candidate: dict[str, int],
               returncode: int,
               elapsed_sec: float,
               summary: dict[str, object] | None,
               failure_kind: str = "",
               correctness_failure_marker: bool = False) -> None:
        item: dict[str, object] = {
            "phase": phase,
            "attempt_index": attempt_index,
            "planned_in_phase": planned_in_phase,
            "status": "completed" if summary is not None else "failed",
            "failure_kind": failure_kind,
            "returncode": returncode,
            "elapsed_sec": elapsed_sec,
            "correctness_failure_marker": int(correctness_failure_marker),
            **candidate,
        }
        if summary is not None:
            item.update(summary)
        self.attempts.append(item)
        self.flush()

    def finish(self, completion_status: str, failure_reason: str = "") -> None:
        self.completion_status = completion_status
        self.failure_reason = failure_reason
        self.flush()

    def _best_completed_coarse(self) -> dict[str, object] | None:
        completed = [
            item for item in self.attempts
            if item["phase"] != "verify" and item["status"] == "completed" and "v2_total_ms_max" in item
        ]
        if not completed:
            return None
        return min(completed, key=result_sort_key)

    def _status_payload(self) -> dict[str, object]:
        coarse = [item for item in self.attempts if item["phase"] != "verify"]
        verify = [item for item in self.attempts if item["phase"] == "verify"]
        completed_coarse = [item for item in coarse if item["status"] == "completed"]
        failed_coarse = [item for item in coarse if item["status"] == "failed"]
        return {
            "schema_version": 1,
            "search_strategy": self.args.search_strategy,
            "shape": {"M": self.args.M, "N": self.args.N, "K": self.args.K},
            "nproc_per_node": self.args.nproc_per_node,
            "candidate_plan_csv": self.args.candidate_plan_csv or None,
            "candidate_plan_sha256": getattr(self.args, "candidate_plan_sha256", "") or None,
            "completion_status": self.completion_status,
            "failure_reason": self.failure_reason,
            "candidate_space_total": self.candidate_space_total,
            "planned_coarse_runs": self.planned_coarse_runs,
            "attempted_candidate_runs": len(coarse),
            "successful_candidate_runs": len(completed_coarse),
            "failed_candidate_runs": len(failed_coarse),
            "timeout_candidate_runs": sum(item["returncode"] == 124 for item in coarse),
            "correctness_failure_markers": sum(int(item["correctness_failure_marker"]) for item in coarse),
            "verification_runs": len(verify),
            "successful_verification_runs": sum(item["status"] == "completed" for item in verify),
            "failed_verification_runs": sum(item["status"] == "failed" for item in verify),
            "best_completed_coarse": self._best_completed_coarse(),
        }

    def flush(self) -> None:
        if self.progress_csv is not None:
            self.progress_csv.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = self.progress_csv.with_suffix(self.progress_csv.suffix + ".tmp")
            with tmp_path.open("w", newline="", encoding="utf-8") as fout:
                writer = csv.DictWriter(fout, fieldnames=PROGRESS_FIELDS)
                writer.writeheader()
                for item in self.attempts:
                    writer.writerow({name: item.get(name) for name in PROGRESS_FIELDS})
            tmp_path.replace(self.progress_csv)
        if self.status_json is not None:
            self.status_json.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = self.status_json.with_suffix(self.status_json.suffix + ".tmp")
            tmp_path.write_text(
                json.dumps(json_safe(self._status_payload()), ensure_ascii=True, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            tmp_path.replace(self.status_json)


def print_topk(title: str, results: list[dict[str, object]], topk: int) -> None:
    print(f"[search] {title}", flush=True)
    for idx, item in enumerate(results[:topk], start=1):
        print(f"  #{idx}: {format_result(item)} | {format_candidate(item)}", flush=True)


def print_best_command(args, best: dict[str, object]) -> None:
    cmd = build_bench_cmd(args, best, iters=args.iters, warmup_iters=args.warmup_iters, profile=False)
    print("[search] best command:", flush=True)
    print(" ".join(cmd), flush=True)


def evaluate_candidate(args,
                       candidate: dict[str, int],
                       ledger: SearchLedger,
                       *,
                       idx: int,
                       total: int,
                       iters: int,
                       warmup_iters: int,
                       tag: str,
                       autotune_override: bool | None = None) -> dict[str, object] | None:
    cmd = build_bench_cmd(
        args,
        candidate,
        iters=iters,
        warmup_iters=warmup_iters,
        profile=False,
        autotune_override=autotune_override,
    )
    started_at = time.perf_counter()
    returncode, output = run_and_capture(cmd, quiet=args.quiet_subprocess, timeout_sec=args.candidate_timeout_sec)
    elapsed_sec = time.perf_counter() - started_at
    if returncode != 0:
        print(
            f"[search][{tag}] candidate {idx}/{total} failed(rc={returncode}): {format_candidate(candidate)}",
            flush=True,
        )
        ledger.record(
            phase=tag,
            attempt_index=idx,
            planned_in_phase=total,
            candidate=candidate,
            returncode=returncode,
            elapsed_sec=elapsed_sec,
            summary=None,
            failure_kind="timeout" if returncode == 124 else "nonzero_returncode",
            correctness_failure_marker=bool(CORRECTNESS_FAILURE_RE.search(output)),
        )
        return None

    per_rank = parse_rank_metrics(output)
    if not per_rank:
        print(
            f"[search][{tag}] candidate {idx}/{total} failed(parse): {format_candidate(candidate)}",
            flush=True,
        )
        ledger.record(
            phase=tag,
            attempt_index=idx,
            planned_in_phase=total,
            candidate=candidate,
            returncode=returncode,
            elapsed_sec=elapsed_sec,
            summary=None,
            failure_kind="rank_metrics_parse_failed",
            correctness_failure_marker=bool(CORRECTNESS_FAILURE_RE.search(output)),
        )
        return None

    summary = summarize_metrics(per_rank, candidate)
    print(
        f"[search][{tag}] candidate {idx}/{total}: {format_result(summary)} | {format_candidate(candidate)}",
        flush=True,
    )
    ledger.record(
        phase=tag,
        attempt_index=idx,
        planned_in_phase=total,
        candidate=candidate,
        returncode=returncode,
        elapsed_sec=elapsed_sec,
        summary=summary,
    )
    return summary


def run_search(args: argparse.Namespace,
               ledger: SearchLedger,
               candidate_space: list[dict[str, int]]) -> None:

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

    if args.search_strategy == "exhaustive":
        candidates = candidate_space
        ledger.set_plan(candidate_space_total=len(candidates), planned_coarse_runs=len(candidates))
        print(
            f"[search] exhaustive search: {len(candidates)} candidates for M={args.M}, N={args.N}, K={args.K}, nproc={args.nproc_per_node}",
            flush=True,
        )
        coarse_results = []
        total_candidates = len(candidates)
        for idx, candidate in enumerate(candidates, start=1):
            summary = evaluate_candidate(
                args,
                candidate,
                ledger,
                idx=idx,
                total=total_candidates,
                iters=args.fast_iters,
                warmup_iters=args.fast_warmup_iters,
                tag="coarse",
                autotune_override=False if args.stage1_no_autotune else None,
            )
            if summary is not None:
                coarse_results.append(summary)
    else:
        all_structural_candidates, search_lists = generate_structural_candidates(args)
        structural_candidates, sms_pairs, bucket_count = reduce_structural_candidates(
            args,
            all_structural_candidates,
            search_lists,
        )
        stage1_total = len(structural_candidates)
        if args.fast_budget > 0:
            remaining_fast_budget = max(0, args.fast_budget - stage1_total)
            actual_structural_topk = min(
                args.structural_topk,
                len(structural_candidates),
                remaining_fast_budget // max(1, len(sms_pairs)),
            )
        else:
            actual_structural_topk = min(args.structural_topk, len(structural_candidates))
        stage2_total = actual_structural_topk * len(sms_pairs)
        ledger.set_plan(
            candidate_space_total=len(candidate_space),
            planned_coarse_runs=stage1_total + stage2_total,
        )
        print(
            f"[search] two-stage search for M={args.M}, N={args.N}, K={args.K}, nproc={args.nproc_per_node}",
            flush=True,
        )
        print(
            f"[search] structural candidates total={len(all_structural_candidates)}, buckets={bucket_count}, "
            f"bucket_topk={max(1, args.structural_bucket_topk)}, stage1 structural candidates={stage1_total}, "
            f"stage2 sms_pairs={len(sms_pairs)}, stage2 refine_topk={actual_structural_topk}, "
            f"fast_budget={'off' if args.fast_budget <= 0 else args.fast_budget}, verify_topk={args.verify_topk}",
            flush=True,
        )

        structural_results = []
        for idx, candidate in enumerate(structural_candidates, start=1):
            summary = evaluate_candidate(
                args,
                candidate,
                ledger,
                idx=idx,
                total=stage1_total,
                iters=args.fast_iters,
                warmup_iters=args.fast_warmup_iters,
                tag="stage1",
                autotune_override=False if args.stage1_no_autotune else None,
            )
            if summary is not None:
                structural_results.append(summary)

        if not structural_results:
            raise SystemExit("[search] all stage1 candidates failed")

        structural_results.sort(key=result_sort_key)
        print_topk("Stage1 Top Structural Candidates", structural_results, min(args.topk, len(structural_results)))

        top_structural = structural_results[:actual_structural_topk]
        coarse_results = []
        seen = set()
        global_idx = 0
        total_sms_candidates = len(top_structural) * len(sms_pairs)
        for structural_result in top_structural:
            structural_candidate = {
                "chunk_rows": int(structural_result["chunk_rows"]),
                "active_chunk_window": int(structural_result["active_chunk_window"]),
                "stage_slots": int(structural_result["stage_slots"]),
                "steady_sms": int(structural_result["steady_sms"]),
                "tail_sms": int(structural_result["tail_sms"]),
                "comm_lanes": int(structural_result["comm_lanes"]),
                "n_bands": int(structural_result["n_bands"]),
                "frontier_chunks": int(structural_result["frontier_chunks"]),
            }
            for candidate in expand_sms_candidates(structural_candidate, sms_pairs):
                key = candidate_key(candidate)
                if key in seen:
                    continue
                seen.add(key)
                global_idx += 1
                summary = evaluate_candidate(
                    args,
                    candidate,
                    ledger,
                    idx=global_idx,
                    total=total_sms_candidates,
                    iters=args.fast_iters,
                    warmup_iters=args.fast_warmup_iters,
                    tag="stage2",
                    autotune_override=False if args.stage1_no_autotune else None,
                )
                if summary is not None:
                    coarse_results.append(summary)

    if not coarse_results:
        raise SystemExit("[search] all coarse candidates failed")

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
        started_at = time.perf_counter()
        returncode, output = run_and_capture(cmd, quiet=args.quiet_subprocess, timeout_sec=args.candidate_timeout_sec)
        elapsed_sec = time.perf_counter() - started_at
        if returncode != 0:
            print(f"[search] verify failed(rc={returncode}): {format_candidate(candidate)}", flush=True)
            ledger.record(
                phase="verify",
                attempt_index=idx + 1,
                planned_in_phase=verify_count,
                candidate=candidate,
                returncode=returncode,
                elapsed_sec=elapsed_sec,
                summary=None,
                failure_kind="timeout" if returncode == 124 else "nonzero_returncode",
                correctness_failure_marker=bool(CORRECTNESS_FAILURE_RE.search(output)),
            )
            continue
        per_rank = parse_rank_metrics(output)
        if not per_rank:
            print(f"[search] verify failed(parse): {format_candidate(candidate)}", flush=True)
            ledger.record(
                phase="verify",
                attempt_index=idx + 1,
                planned_in_phase=verify_count,
                candidate=candidate,
                returncode=returncode,
                elapsed_sec=elapsed_sec,
                summary=None,
                failure_kind="rank_metrics_parse_failed",
                correctness_failure_marker=bool(CORRECTNESS_FAILURE_RE.search(output)),
            )
            continue
        verified_summary = summarize_metrics(per_rank, candidate)
        verified_results.append(verified_summary)
        ledger.record(
            phase="verify",
            attempt_index=idx + 1,
            planned_in_phase=verify_count,
            candidate=candidate,
            returncode=returncode,
            elapsed_sec=elapsed_sec,
            summary=verified_summary,
        )

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
        returncode, _ = run_and_capture(cmd, quiet=args.quiet_subprocess, timeout_sec=args.candidate_timeout_sec)
        if returncode != 0:
            raise SystemExit(f"[search] best-candidate profile run failed with rc={returncode}")


def main() -> None:
    args = parse_args()
    candidate_space = prepare_candidate_space(args)
    if args.plan_only:
        return
    ledger = SearchLedger(args)
    try:
        run_search(args, ledger, candidate_space)
    except KeyboardInterrupt:
        ledger.finish("interrupted", "keyboard_interrupt")
        raise
    except SystemExit as exc:
        if exc.code in (None, 0):
            ledger.finish("completed")
        else:
            ledger.finish("failed", str(exc.code))
        raise
    except Exception as exc:
        ledger.finish("failed", f"{type(exc).__name__}: {exc}")
        raise
    else:
        ledger.finish("completed")


if __name__ == "__main__":
    main()
