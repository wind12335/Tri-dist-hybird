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
import gc
import os
from pathlib import Path

import torch
import torch.distributed
import triton
import triton_dist.tune

from triton_dist.layers.nvidia import GemmARLayer
from triton_dist.kernels.nvidia import (create_frontier_windowed_panel_gemm_ar_context_v23,
                                        frontier_windowed_panel_allreduce_v23,
                                        frontier_windowed_panel_gemm_allreduce_v23,
                                        frontier_windowed_panel_gemm_only_op_v23,
                                        frontier_windowed_panel_gemm_allreduce_op_v23)
from triton_dist.kernels.nvidia.gemm import get_config_space
from triton_dist.profiler_utils import group_profile, perf_func
from triton_dist.test.utils import LAYER_CONFIGS, assert_allclose
from triton_dist.utils import (dist_print, finalize_distributed, initialize_distributed, nvshmem_barrier_all_on_stream,
                               rand_tensor, sleep_async, wait_until_max_gpu_clock_or_warning)
# torchrun --nproc_per_node=4 python/triton_dist/benchmark/bench_new_windowed_panel_gemm_allreduce_v23.py --M 8192 --N 49152 --K 12288 --dtype bfloat16 --chunk_rows 1024 --stripe_rows 256 --active_chunk_window 4 --n_bands 2 --frontier_chunks 2 --stage_slots 16 --comm_lanes 2 --num_comm_sms 24 --autotune

AUTOTUNE_CACHE: dict[tuple, dict] = {}


def _parse_int_list_arg(value: str | None) -> list[int] | None:
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def _unique_preserve_order(values):
    seen = set()
    result = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--M", type=int, default=8192)
    parser.add_argument("--N", type=int, default=None)
    parser.add_argument("--K", type=int, default=None)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--warmup_iters", type=int, default=5)
    parser.add_argument("--autotune", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--profile", action="store_true", default=False)
    parser.add_argument("--dump_csv", action="store_true", default=False)
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16"])
    parser.add_argument("--check", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--run_baseline", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--baseline_num_comm_sms", type=int, default=16)
    parser.add_argument("--baseline_row_wise", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--baseline_low_latency", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--chunk_rows", type=int, default=0)
    parser.add_argument("--stripe_rows", type=int, default=256)
    parser.add_argument("--target_chunks", type=int, default=4)
    parser.add_argument("--min_chunk_rows", type=int, default=512)
    parser.add_argument("--active_chunk_window", type=int, default=2)
    parser.add_argument("--n_bands", type=int, default=1)
    parser.add_argument("--frontier_chunks", type=int, default=1)
    parser.add_argument("--producer_order", choices=["logical", "stripe_frontier", "panel_bulk", "panel_frontier",
                                                    "panel_recursive_doubling", "panel_recursive_doubling_cublas",
                                                    "panel_recursive_doubling_bulk_cublas"],
                        default="logical", help="panel_* uses whole panels and fused FP32 reduction; set "
                        "stripe_rows=chunk_rows. panel_frontier submits handoff immediately; "
                        "panel_bulk is the window-batched submission control. The recursive-doubling bulk mode "
                        "is the matched control for panel_recursive_doubling_cublas. frontier_chunks only affects stripes.")
    parser.add_argument("--stage_slots", type=int, default=4)
    parser.add_argument("--num_comm_sms", type=int, default=16)
    parser.add_argument("--comm_lanes", type=int, default=4)
    parser.add_argument("--streaming_depth", type=int, default=1)
    parser.add_argument("--search", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--search_topk", type=int, default=5)
    parser.add_argument("--search_verify_topk", type=int, default=1)
    parser.add_argument("--search_chunk_rows_list", type=str, default="")
    parser.add_argument("--search_stripe_rows_list", type=str, default="")
    parser.add_argument("--search_n_bands_list", type=str, default="")
    parser.add_argument("--search_active_chunk_window_list", type=str, default="")
    parser.add_argument("--search_frontier_chunks_list", type=str, default="")
    parser.add_argument("--search_stage_slots_list", type=str, default="")
    parser.add_argument("--search_comm_lanes_list", type=str, default="")
    parser.add_argument("--search_num_comm_sms_list", type=str, default="")
    args = parser.parse_args()
    if args.producer_order.endswith("_cublas") and args.autotune:
        parser.error("The cuBLAS producer does not use Triton GEMM configurations; use --no-autotune")
    if args.producer_order.startswith("panel_"):
        if args.streaming_depth != 1:
            parser.error("panel_* currently requires --streaming_depth 1")
        if not args.search and (args.chunk_rows <= 0 or args.stripe_rows != args.chunk_rows):
            parser.error("panel_* requires explicit positive --chunk_rows and equal --stripe_rows")
        args.frontier_chunks = 0
    return args


def get_test_configs(args):
    if args.N is not None or args.K is not None:
        if args.N is None or args.K is None:
            raise ValueError("`--N` and `--K` must be set together.")
        return {"custom": {"N": args.N, "K": args.K}}
    return LAYER_CONFIGS


def build_kernel_params(args, overrides=None):
    params = {
        "chunk_rows": args.chunk_rows,
        "stripe_rows": args.stripe_rows,
        "target_chunks": args.target_chunks,
        "min_chunk_rows": args.min_chunk_rows,
        "active_chunk_window": args.active_chunk_window,
        "n_bands": args.n_bands,
        "frontier_chunks": args.frontier_chunks,
        "producer_order": args.producer_order,
        "stage_slots": args.stage_slots,
        "num_comm_sms": args.num_comm_sms,
        "comm_lanes": args.comm_lanes,
        "streaming_depth": args.streaming_depth,
    }
    if overrides:
        params.update(overrides)
    if params["producer_order"].startswith("panel_"):
        params["frontier_chunks"] = 0  # Stripe-specific parameter, not a panel-order control.
    return params


def build_alloc_envelope(search_candidates, N: int):
    if not search_candidates:
        return None
    return {
        "alloc_scatter_rows": max(
            params["stage_slots"] * LOCAL_WORLD_SIZE * params["stripe_rows"] for params in search_candidates
        ),
        "alloc_max_band_cols": max(
            triton.cdiv(N, max(1, params["n_bands"])) for params in search_candidates
        ),
        "alloc_stage_slots": max(params["stage_slots"] for params in search_candidates),
    }


def candidate_alloc_key(params, N: int):
    return (
        params["stage_slots"] * LOCAL_WORLD_SIZE * params["stripe_rows"],
        triton.cdiv(N, max(1, params["n_bands"])),
        params["stage_slots"],
    )


def bucket_search_candidates(search_candidates, N: int):
    buckets = {}
    for params in search_candidates:
        key = candidate_alloc_key(params, N)
        buckets.setdefault(key, []).append(params)

    ordered = []
    for key in sorted(buckets.keys(), key=lambda x: (x[0] * x[1], x[0], x[1], x[2])):
        bucket_params = buckets[key]
        ordered.append({
            "alloc_envelope": {
                "alloc_scatter_rows": key[0],
                "alloc_max_band_cols": key[1],
                "alloc_stage_slots": key[2],
            },
            "candidates": bucket_params,
        })
    return ordered


def kernel_params_key(params):
    return (
        params["chunk_rows"],
        params["stripe_rows"],
        params["active_chunk_window"],
        params["n_bands"],
        params["frontier_chunks"],
        params["producer_order"],
        params["stage_slots"],
        params["comm_lanes"],
        params["num_comm_sms"],
    )


def estimate_max_stripes(chunk_rows: int, stripe_rows: int) -> int:
    return max(1, triton.cdiv(chunk_rows, max(1, stripe_rows)))


def estimate_num_chunks(M: int, chunk_rows: int) -> int:
    return max(1, triton.cdiv(M, max(1, chunk_rows)))


def default_search_lists(args, M: int, N: int):
    chunk_rows_list = _parse_int_list_arg(args.search_chunk_rows_list)
    stripe_rows_list = _parse_int_list_arg(args.search_stripe_rows_list)
    n_bands_list = _parse_int_list_arg(args.search_n_bands_list)
    active_window_list = _parse_int_list_arg(args.search_active_chunk_window_list)
    frontier_chunks_list = _parse_int_list_arg(args.search_frontier_chunks_list)
    stage_slots_list = _parse_int_list_arg(args.search_stage_slots_list)
    comm_lanes_list = _parse_int_list_arg(args.search_comm_lanes_list)
    num_comm_sms_list = _parse_int_list_arg(args.search_num_comm_sms_list)

    if chunk_rows_list is None:
        if args.chunk_rows > 0:
            chunk_rows_list = [args.chunk_rows]
        elif M <= 8192:
            chunk_rows_list = [512, 1024]
        else:
            chunk_rows_list = [1024, 2048]

    if n_bands_list is None:
        if N <= 32768:
            n_bands_list = [1, 2]
        elif N <= 65536:
            n_bands_list = [1, 2]
        else:
            n_bands_list = [2, 4]

    if stage_slots_list is None:
        stage_slots_list = [4, 8, 12]

    if comm_lanes_list is None:
        comm_lanes_list = [1, 2]

    if num_comm_sms_list is None:
        num_comm_sms_list = [16, 24]

    return {
        "chunk_rows_list": _unique_preserve_order([x for x in chunk_rows_list if x > 0 and x <= M]),
        "stripe_rows_list": stripe_rows_list,
        "n_bands_list": _unique_preserve_order([x for x in n_bands_list if x > 0]),
        "active_window_list": active_window_list,
        "frontier_chunks_list": frontier_chunks_list,
        "stage_slots_list": _unique_preserve_order([x for x in stage_slots_list if x > 0]),
        "comm_lanes_list": _unique_preserve_order([x for x in comm_lanes_list if x > 0]),
        "num_comm_sms_list": _unique_preserve_order([x for x in num_comm_sms_list if x > 0]),
    }


def generate_search_candidates(args, M: int, N: int):
    search_lists = default_search_lists(args, M, N)
    candidates = []
    seen = set()
    for chunk_rows in search_lists["chunk_rows_list"]:
        stripe_rows_candidates = search_lists["stripe_rows_list"]
        if args.producer_order.startswith("panel_"):
            stripe_rows_candidates = [chunk_rows]
        if stripe_rows_candidates is None:
            stripe_rows_candidates = [256, 512]
        stripe_rows_candidates = _unique_preserve_order(
            [x for x in stripe_rows_candidates if x > 0 and x <= chunk_rows]
        )
        if not stripe_rows_candidates:
            stripe_rows_candidates = [chunk_rows]

        num_chunks = estimate_num_chunks(M, chunk_rows)
        for stripe_rows in stripe_rows_candidates:
            max_stripes_per_chunk = estimate_max_stripes(chunk_rows, stripe_rows)
            for n_bands in search_lists["n_bands_list"]:
                effective_n_bands = max(1, min(n_bands, N))
                for stage_slots in search_lists["stage_slots_list"]:
                    if search_lists["active_window_list"] is None:
                        active_window_candidates = [1, 2]
                    else:
                        active_window_candidates = search_lists["active_window_list"]
                    active_window_candidates = _unique_preserve_order(
                        [max(1, min(num_chunks, x)) for x in active_window_candidates]
                    )
                    for active_chunk_window in active_window_candidates:
                        if search_lists["frontier_chunks_list"] is None:
                            frontier_chunk_candidates = _unique_preserve_order([1, min(2, active_chunk_window)])
                        else:
                            frontier_chunk_candidates = _unique_preserve_order(
                                [max(1, min(num_chunks, x)) for x in search_lists["frontier_chunks_list"]]
                            )
                        for frontier_chunks in frontier_chunk_candidates:
                            for comm_lanes in search_lists["comm_lanes_list"]:
                                for num_comm_sms in search_lists["num_comm_sms_list"]:
                                    params = build_kernel_params(
                                        args,
                                        {
                                            "chunk_rows": chunk_rows,
                                            "stripe_rows": stripe_rows,
                                            "active_chunk_window": active_chunk_window,
                                            "n_bands": effective_n_bands,
                                            "frontier_chunks": frontier_chunks,
                                            "stage_slots": stage_slots,
                                            "comm_lanes": comm_lanes,
                                            "num_comm_sms": num_comm_sms,
                                        },
                                    )
                                    key = kernel_params_key(params)
                                    if key in seen:
                                        continue
                                    seen.add(key)
                                    params["max_stripes_per_chunk"] = max_stripes_per_chunk
                                    params["num_chunks"] = num_chunks
                                    params["lead_ratio"] = (
                                        active_chunk_window * effective_n_bands * max_stripes_per_chunk / max(stage_slots, 1)
                                    )
                                    candidates.append(params)
    return candidates


def compute_pareto_front(results):
    pareto = []
    for item in results:
        dominated = False
        for other in results:
            if other is item:
                continue
            better_or_equal_speed = other["new_speedup_vs_torch"] >= item["new_speedup_vs_torch"]
            better_or_equal_compact = other["compaction_ratio"] >= item["compaction_ratio"]
            strictly_better = (
                other["new_speedup_vs_torch"] > item["new_speedup_vs_torch"]
                or other["compaction_ratio"] > item["compaction_ratio"]
            )
            if better_or_equal_speed and better_or_equal_compact and strictly_better:
                dominated = True
                break
        if not dominated:
            pareto.append(item)
    pareto.sort(key=lambda x: (-x["new_speedup_vs_torch"], -x["compaction_ratio"], x["new_total_ms"]))
    return pareto


def make_data(M, N, K, dtype: torch.dtype, tp_group: torch.distributed.ProcessGroup):
    rank = tp_group.rank()
    world_size = tp_group.size()
    assert K % world_size == 0
    local_k = K // world_size
    scale = 0.01 * (rank + 1)
    device = torch.cuda.current_device()
    a = rand_tensor((M, local_k), dtype=dtype, device=device) * scale
    weight = rand_tensor((N, local_k), dtype=dtype, device=device) * scale
    return a, weight


def torch_gemm_ar(a: torch.Tensor, weight: torch.Tensor, tp_group: torch.distributed.ProcessGroup):
    output = torch.matmul(a, weight.T)
    torch.distributed.all_reduce(output, group=tp_group)
    return output


def sync_all(pg: torch.distributed.ProcessGroup):
    nvshmem_barrier_all_on_stream(torch.cuda.current_stream())
    torch.cuda.synchronize()
    torch.distributed.barrier(pg, device_ids=[torch.cuda.current_device()])


def check_allclose_across_ranks(
    ref: torch.Tensor,
    out: torch.Tensor,
    pg: torch.distributed.ProcessGroup,
    *,
    atol: float,
    rtol: float,
    baseline: torch.Tensor | None = None,
) -> None:
    rank = pg.rank()
    world_size = pg.size()
    local_ok = torch.tensor(
        [
            int(
                torch.allclose(ref, out, atol=atol, rtol=rtol)
                and (baseline is None or torch.allclose(ref, baseline, atol=atol, rtol=rtol))
            )
        ],
        dtype=torch.int32,
        device=ref.device,
    )
    torch.distributed.all_reduce(local_ok, op=torch.distributed.ReduceOp.MIN, group=pg)
    if int(local_ok.item()) == 1:
        return

    for i in range(world_size):
        torch.distributed.barrier(pg, device_ids=[torch.cuda.current_device()])
        if rank == i:
            if not torch.allclose(ref, out, atol=atol, rtol=rtol):
                assert_allclose(ref, out, atol=atol, rtol=rtol)
            if baseline is not None and not torch.allclose(ref, baseline, atol=atol, rtol=rtol):
                assert_allclose(ref, baseline, atol=atol, rtol=rtol)

    torch.distributed.barrier(pg, device_ids=[torch.cuda.current_device()])
    raise RuntimeError("Distributed correctness check failed")


def choose_gemm_config():
    return get_config_space(False)[0]


def get_autotuned_gemm_ar_config(A: torch.Tensor, B: torch.Tensor, ctx, pg: torch.distributed.ProcessGroup):
    base_key = frontier_windowed_panel_gemm_allreduce_v23.key_fn(A, B, ctx)
    cache_key = (base_key, "frontier_windowed_panel_gemm_allreduce_v23")
    best_config = AUTOTUNE_CACHE.get(cache_key)
    if best_config is None:
        config_space = frontier_windowed_panel_gemm_allreduce_v23.get_pruned_config(A, B, ctx)
        timings = frontier_windowed_panel_gemm_allreduce_v23.tune(config_space, pg, A, B, ctx)
        timings.sort(key=lambda x: x[0])
        assert len(timings) > 0, "autotune returned empty timing list"
        best_config = timings[0][1]
        AUTOTUNE_CACHE[cache_key] = best_config
    return best_config["gemm_config"]


def create_baseline_layer(pg, M, N, K):
    return GemmARLayer(
        pg,
        M,
        N,
        K,
        dtype,
        dtype,
        LOCAL_WORLD_SIZE,
        persistent=False,
        use_ll_kernel=args.baseline_low_latency,
        copy_to_local=True,
        NUM_COMM_SMS=args.baseline_num_comm_sms,
        TILE_MAP_LEVEL=int(args.baseline_row_wise),
    )


def create_new_ctx(M, N, kernel_params, alloc_envelope=None):
    alloc_envelope = alloc_envelope or {}
    return create_frontier_windowed_panel_gemm_ar_context_v23(
        M,
        N,
        RANK,
        WORLD_SIZE,
        LOCAL_WORLD_SIZE,
        dtype,
        chunk_rows=kernel_params["chunk_rows"],
        stripe_rows=kernel_params["stripe_rows"],
        target_chunks=kernel_params["target_chunks"],
        min_chunk_rows=kernel_params["min_chunk_rows"],
        active_chunk_window=kernel_params["active_chunk_window"],
        n_bands=kernel_params["n_bands"],
        frontier_chunks=kernel_params["frontier_chunks"],
        producer_order=kernel_params["producer_order"],
        stage_slots=kernel_params["stage_slots"],
        num_comm_sms=kernel_params["num_comm_sms"],
        comm_lanes=kernel_params["comm_lanes"],
        alloc_scatter_rows=alloc_envelope.get("alloc_scatter_rows"),
        alloc_max_band_cols=alloc_envelope.get("alloc_max_band_cols"),
        alloc_stage_slots=alloc_envelope.get("alloc_stage_slots"),
    )


def perf_test(
    model_name: str,
    M: int,
    config,
    pg: torch.distributed.ProcessGroup,
    *,
    kernel_params=None,
    enable_check: bool | None = None,
    enable_baseline: bool | None = None,
    verbose: bool = True,
    run_profile_pass: bool = True,
    print_shape: bool = True,
    alloc_envelope=None,
):
    N = config["N"]
    K = config["K"]
    rank = pg.rank()
    world_size = pg.size()
    if rank == 0 and print_shape:
        print(f"[{model_name}] test shape: M={M}, N={N}, K={K}")

    kernel_params = kernel_params or build_kernel_params(args)
    enable_check = args.check if enable_check is None else enable_check
    enable_baseline = args.run_baseline if enable_baseline is None else enable_baseline

    a, weight = make_data(M, N, K, dtype, pg)
    partial = torch.matmul(a, weight.T)
    gemm_config = choose_gemm_config()

    baseline = None
    new_ctx = None
    baseline_status = "disabled"
    if enable_baseline:
        try:
            baseline = create_baseline_layer(pg, M, N, K)
            baseline_status = "ok"
        except Exception as exc:
            if "Failed to allocate memory" in str(exc):
                baseline_status = "oom"
                dist_print(
                    f"Rank {rank} [{model_name}] baseline skipped: NVSHMEM symmetric allocation failed",
                    need_sync=True,
                    allowed_ranks=list(range(world_size)),
                )
            else:
                raise

    try:
        new_ctx = create_new_ctx(M, N, kernel_params, alloc_envelope=alloc_envelope)
    except Exception as exc:
        error_text = str(exc)
        if "Failed to allocate memory" in error_text:
            if verbose:
                dist_print(
                    f"Rank {rank} [{model_name}] new kernel skipped: NVSHMEM symmetric allocation failed",
                    need_sync=True,
                    allowed_ranks=list(range(world_size)),
                )
            torch.cuda.empty_cache()
            return {
                "skipped": True,
                "reason": "nvshmem_oom",
                "kernel_params": kernel_params,
            }
        if "invalid for input of size" in error_text:
            if verbose:
                dist_print(
                    f"Rank {rank} [{model_name}] new kernel skipped: NVSHMEM peer tensor shape mismatch",
                    need_sync=True,
                    allowed_ranks=list(range(world_size)),
                )
            torch.cuda.empty_cache()
            return {
                "skipped": True,
                "reason": "nvshmem_shape_mismatch",
                "kernel_params": kernel_params,
            }
        raise

    runtime_gemm_config = get_autotuned_gemm_ar_config(a, weight.T, new_ctx, pg) if args.autotune else gemm_config

    def _launch_new_total_once(*, drain: bool):
        return frontier_windowed_panel_gemm_allreduce_op_v23(a, weight.T, new_ctx, runtime_gemm_config, drain=drain)

    def _new_gemm_only():
        return frontier_windowed_panel_gemm_only_op_v23(a, weight.T, new_ctx, runtime_gemm_config)

    def _launch_new_ar_once(*, drain: bool):
        return frontier_windowed_panel_allreduce_v23(partial, new_ctx, drain=drain)

    def _torch_total():
        return torch_gemm_ar(a, weight, pg)

    def _torch_gemm_only():
        return torch.matmul(a, weight.T)

    def _torch_ar_only():
        output = partial.clone()
        torch.distributed.all_reduce(output, group=pg)
        return output

    def _new_total():
        return _launch_new_total_once(drain=True)

    def _new_ar_only():
        return _launch_new_ar_once(drain=True)

    def _new_total_streaming():
        last_output = None
        for launch_id in range(args.streaming_depth):
            last_output = _launch_new_total_once(drain=launch_id == args.streaming_depth - 1)
        return last_output

    def _new_ar_only_streaming():
        last_output = None
        for launch_id in range(args.streaming_depth):
            last_output = _launch_new_ar_once(drain=launch_id == args.streaming_depth - 1)
        return last_output

    def _baseline_total():
        return baseline.forward(a, weight)

    def _baseline_gemm_only():
        return baseline.forward_gemm(a, weight)

    def _baseline_ar_only():
        return baseline.forward_ar(partial)

    atol = 6e-2 if dtype == torch.bfloat16 else 1e-2
    rtol = atol
    metrics = {
        "baseline_status": baseline_status,
        "new_status": "ok",
        "streaming_depth": kernel_params["streaming_depth"],
        "chunk_rows": new_ctx.chunk_rows,
        "stripe_rows": new_ctx.stripe_rows,
        "active_chunk_window": kernel_params["active_chunk_window"],
        "n_bands": kernel_params["n_bands"],
        "frontier_chunks": kernel_params["frontier_chunks"],
        "producer_order": new_ctx.producer_order,
        "stage_slots": new_ctx.stage_slots,
        "comm_lanes": kernel_params["comm_lanes"],
        "num_comm_sms": kernel_params["num_comm_sms"],
        "compact_scatter_rows": float(new_ctx.compact_scatter_rows),
        "baseline_scatter_rows_est": float(new_ctx.baseline_scatter_rows_estimate),
        "compaction_ratio": float(new_ctx.scatter_compaction_ratio),
        "lead_ratio": (
            kernel_params["active_chunk_window"]
            * kernel_params["n_bands"]
            * estimate_max_stripes(new_ctx.chunk_rows, new_ctx.stripe_rows)
            / max(new_ctx.stage_slots, 1)
        ),
    }

    try:
        for _ in range(3):
            sync_all(pg)
            new_out = _new_total()
            if baseline is not None:
                sync_all(pg)
                baseline_out = _baseline_total()

        if rank == 0 and verbose:
            print(
                "[custom] v2.3 symmetric scratch:",
                {
                    "compact_scatter_rows": new_ctx.compact_scatter_rows,
                    "baseline_scatter_rows_est": new_ctx.baseline_scatter_rows_estimate,
                    "compaction_ratio": round(new_ctx.scatter_compaction_ratio, 4),
                    "chunk_rows": new_ctx.chunk_rows,
                    "stripe_rows": new_ctx.stripe_rows,
                    "stage_slots": new_ctx.stage_slots,
                    "lead_ratio": round(metrics["lead_ratio"], 4),
                },
                flush=True,
            )

        if enable_check:
            sync_all(pg)
            torch_out = _torch_total()
            check_allclose_across_ranks(
                torch_out,
                new_out,
                pg,
                atol=atol,
                rtol=rtol,
                baseline=baseline_out if baseline is not None else None,
            )

        run_id = os.environ.get("TORCHELASTIC_RUN_ID", "local")
        if run_profile_pass:
            with group_profile(f"new_gemm_ar_perf_m_{M}_n_{N}_k_{K}_{run_id}", args.profile, group=TP_GROUP):
                sync_all(pg)
                sleep_async(100)
                perf_func(_new_total_streaming, iters=args.iters, warmup_iters=args.warmup_iters)
                if baseline is not None:
                    sync_all(pg)
                    sleep_async(100)
                    perf_func(_baseline_total, iters=args.iters, warmup_iters=args.warmup_iters)
                sync_all(pg)
                sleep_async(100)
                perf_func(_torch_total, iters=args.iters, warmup_iters=args.warmup_iters)

        sync_all(pg)
        sleep_async(100)
        wait_until_max_gpu_clock_or_warning(torch.cuda.current_device())
        _, metrics["new_total_ms"] = perf_func(_new_total_streaming, iters=args.iters, warmup_iters=args.warmup_iters)
        metrics["new_total_ms"] /= kernel_params["streaming_depth"]

        sync_all(pg)
        sleep_async(100)
        wait_until_max_gpu_clock_or_warning(torch.cuda.current_device())
        _, metrics["new_ar_only_ms"] = perf_func(_new_ar_only_streaming, iters=args.iters, warmup_iters=args.warmup_iters)
        metrics["new_ar_only_ms"] /= kernel_params["streaming_depth"]

        sync_all(pg)
        sleep_async(100)
        wait_until_max_gpu_clock_or_warning(torch.cuda.current_device())
        _, metrics["new_gemm_only_ms"] = perf_func(_new_gemm_only, iters=args.iters, warmup_iters=args.warmup_iters)

        sync_all(pg)
        sleep_async(100)
        wait_until_max_gpu_clock_or_warning(torch.cuda.current_device())
        _, metrics["torch_total_ms"] = perf_func(_torch_total, iters=args.iters, warmup_iters=args.warmup_iters)

        sync_all(pg)
        wait_until_max_gpu_clock_or_warning(torch.cuda.current_device())
        _, metrics["torch_gemm_only_ms"] = perf_func(_torch_gemm_only, iters=args.iters, warmup_iters=args.warmup_iters)

        sync_all(pg)
        wait_until_max_gpu_clock_or_warning(torch.cuda.current_device())
        _, metrics["torch_ar_only_ms"] = perf_func(_torch_ar_only, iters=args.iters, warmup_iters=args.warmup_iters)

        if baseline is not None:
            sync_all(pg)
            sleep_async(100)
            wait_until_max_gpu_clock_or_warning(torch.cuda.current_device())
            _, metrics["baseline_total_ms"] = perf_func(_baseline_total, iters=args.iters, warmup_iters=args.warmup_iters)

            sync_all(pg)
            wait_until_max_gpu_clock_or_warning(torch.cuda.current_device())
            _, metrics["baseline_gemm_only_ms"] = perf_func(_baseline_gemm_only,
                                                            iters=args.iters,
                                                            warmup_iters=args.warmup_iters)

            sync_all(pg)
            wait_until_max_gpu_clock_or_warning(torch.cuda.current_device())
            _, metrics["baseline_ar_only_ms"] = perf_func(_baseline_ar_only,
                                                          iters=args.iters,
                                                          warmup_iters=args.warmup_iters)
        else:
            metrics["baseline_total_ms"] = float("nan")
            metrics["baseline_gemm_only_ms"] = float("nan")
            metrics["baseline_ar_only_ms"] = float("nan")

        serial_torch_ms = metrics["torch_gemm_only_ms"] + metrics["torch_ar_only_ms"]
        metrics["new_speedup_vs_torch"] = metrics["torch_total_ms"] / metrics["new_total_ms"]
        metrics["new_overlap_ratio_vs_torch_serial"] = (serial_torch_ms - metrics["new_total_ms"]) / max(
            serial_torch_ms, 1e-6)
        metrics["new_internal_overlap_ratio"] = 1.0 - metrics["new_total_ms"] / max(
            metrics["new_gemm_only_ms"] + metrics["new_ar_only_ms"], 1e-6)

        if baseline is not None:
            metrics["new_speedup_vs_baseline"] = metrics["baseline_total_ms"] / metrics["new_total_ms"]
        else:
            metrics["new_speedup_vs_baseline"] = float("nan")

        partial_nbytes = M * N * dtype.itemsize
        metrics["new_ar_gbps"] = partial_nbytes / (1024**3) / metrics["new_ar_only_ms"] * 1e3
        metrics["torch_ar_gbps"] = partial_nbytes / (1024**3) / metrics["torch_ar_only_ms"] * 1e3
        if baseline is not None:
            metrics["baseline_ar_gbps"] = partial_nbytes / (1024**3) / metrics["baseline_ar_only_ms"] * 1e3
        else:
            metrics["baseline_ar_gbps"] = float("nan")

        if verbose:
            dist_print(
                f"Rank {rank} [{model_name}] new latency (ms): "
                f"total={metrics['new_total_ms']:.4f}, gemm_only={metrics['new_gemm_only_ms']:.4f}, "
                f"ar_only={metrics['new_ar_only_ms']:.4f}, "
                f"speedup_vs_torch={metrics['new_speedup_vs_torch']:.4f}, "
                f"speedup_vs_baseline={metrics['new_speedup_vs_baseline']:.4f}, "
                f"overlap_ratio={metrics['new_overlap_ratio_vs_torch_serial']:.4f}, "
                f"internal_overlap={metrics['new_internal_overlap_ratio']:.4f}, "
                f"streaming_depth={metrics['streaming_depth']}, producer_order={new_ctx.producer_order}",
                need_sync=True,
                allowed_ranks=list(range(world_size)),
            )
            if baseline is not None:
                dist_print(
                    f"Rank {rank} [{model_name}] baseline latency (ms): "
                    f"total={metrics['baseline_total_ms']:.4f}, "
                    f"gemm_only={metrics['baseline_gemm_only_ms']:.4f}, "
                    f"ar_only={metrics['baseline_ar_only_ms']:.4f}",
                    need_sync=True,
                    allowed_ranks=list(range(world_size)),
                )
            dist_print(
                f"Rank {rank} [{model_name}] torch latency (ms): "
                f"total={metrics['torch_total_ms']:.4f}, "
                f"gemm_only={metrics['torch_gemm_only_ms']:.4f}, "
                f"ar_only={metrics['torch_ar_only_ms']:.4f}",
                need_sync=True,
                allowed_ranks=list(range(world_size)),
            )
            dist_print(
                f"Rank {rank} [{model_name}] allreduce BW (GB/s): "
                f"new={metrics['new_ar_gbps']:.4f}, baseline={metrics['baseline_ar_gbps']:.4f}, torch={metrics['torch_ar_gbps']:.4f}",
                need_sync=True,
                allowed_ranks=list(range(world_size)),
            )

        return metrics
    finally:
        if new_ctx is not None:
            new_ctx.finalize()
            del new_ctx
        if baseline is not None:
            baseline.finalize()
            del baseline
        gc.collect()
        torch.cuda.empty_cache()


def print_search_summary(model_name: str, results, *, topk: int, title: str) -> None:
    print(f"[search][{model_name}] {title}", flush=True)
    for idx, item in enumerate(results[:topk], start=1):
        print(
            f"  #{idx}: speedup={item['new_speedup_vs_torch']:.4f}, total={item['new_total_ms']:.4f} ms, "
            f"gemm={item['new_gemm_only_ms']:.4f} ms, ar={item['new_ar_only_ms']:.4f} ms, "
            f"bw={item['new_ar_gbps']:.2f} GB/s, compaction={item['compaction_ratio']:.3f}, "
            f"lead_ratio={item['lead_ratio']:.3f}, "
            f"chunk={int(item['chunk_rows'])}, stripe={int(item['stripe_rows'])}, "
            f"window={int(item['active_chunk_window'])}, bands={int(item['n_bands'])}, "
            f"frontier={int(item['frontier_chunks'])}, "
            f"producer_order={item['producer_order']}, "
            f"stage={int(item['stage_slots'])}, lanes={int(item['comm_lanes'])}, sms={int(item['num_comm_sms'])}",
            flush=True,
        )


def write_search_csv(csv_file: Path, results) -> None:
    with open(csv_file, "w", encoding="utf-8") as fout:
        print(
            ",".join([
                "rank_world",
                "M",
                "N",
                "K",
                "chunk_rows",
                "stripe_rows",
                "active_chunk_window",
                "n_bands",
                "frontier_chunks",
                "producer_order",
                "stage_slots",
                "comm_lanes",
                "num_comm_sms",
                "compact_scatter_rows",
                "baseline_scatter_rows_est",
                "compaction_ratio",
                "lead_ratio",
                "new_total_ms",
                "new_gemm_only_ms",
                "new_ar_only_ms",
                "new_ar_gbps",
                "torch_total_ms",
                "torch_gemm_only_ms",
                "torch_ar_only_ms",
                "torch_ar_gbps",
                "new_speedup_vs_torch",
                "new_overlap_ratio_vs_torch_serial",
                "new_internal_overlap_ratio",
            ]),
            file=fout,
        )
        for item in results:
            print(
                ",".join([
                    str(WORLD_SIZE),
                    str(args.M),
                    str(item["N"]),
                    str(item["K"]),
                    str(int(item["chunk_rows"])),
                    str(int(item["stripe_rows"])),
                    str(int(item["active_chunk_window"])),
                    str(int(item["n_bands"])),
                    str(int(item["frontier_chunks"])),
                    str(item["producer_order"]),
                    str(int(item["stage_slots"])),
                    str(int(item["comm_lanes"])),
                    str(int(item["num_comm_sms"])),
                    str(int(item["compact_scatter_rows"])),
                    str(int(item["baseline_scatter_rows_est"])),
                    f"{item['compaction_ratio']:.6f}",
                    f"{item['lead_ratio']:.6f}",
                    f"{item['new_total_ms']:.6f}",
                    f"{item['new_gemm_only_ms']:.6f}",
                    f"{item['new_ar_only_ms']:.6f}",
                    f"{item['new_ar_gbps']:.6f}",
                    f"{item['torch_total_ms']:.6f}",
                    f"{item['torch_gemm_only_ms']:.6f}",
                    f"{item['torch_ar_only_ms']:.6f}",
                    f"{item['torch_ar_gbps']:.6f}",
                    f"{item['new_speedup_vs_torch']:.6f}",
                    f"{item['new_overlap_ratio_vs_torch_serial']:.6f}",
                    f"{item['new_internal_overlap_ratio']:.6f}",
                ]),
                file=fout,
            )


def search_best_params(model_name: str, M: int, config, pg: torch.distributed.ProcessGroup):
    N = config["N"]
    K = config["K"]
    candidates = generate_search_candidates(args, M, N)
    candidate_buckets = bucket_search_candidates(candidates, N)
    if RANK == 0:
        print(
            f"[search][{model_name}] evaluating {len(candidates)} candidates for M={M}, N={N}, K={K}",
            flush=True,
        )
        print(
            "[search] fast pass disables baseline and per-candidate correctness; top candidates will be re-verified.",
            flush=True,
        )
        print(f"[search] candidate buckets: {len(candidate_buckets)}", flush=True)

    coarse_results = []
    global_idx = 0
    for bucket_idx, bucket in enumerate(candidate_buckets, start=1):
        alloc_envelope = bucket["alloc_envelope"]
        bucket_candidates = bucket["candidates"]
        if RANK == 0:
            print(
                f"[search][{model_name}] bucket {bucket_idx}/{len(candidate_buckets)}: "
                f"{len(bucket_candidates)} candidates, alloc={alloc_envelope}",
                flush=True,
            )
        for kernel_params in bucket_candidates:
            global_idx += 1
            metrics = perf_test(
                model_name,
                M,
                config,
                pg,
                kernel_params=kernel_params,
                enable_check=False,
                enable_baseline=False,
                verbose=False,
                run_profile_pass=False,
                print_shape=(global_idx == 1),
                alloc_envelope=alloc_envelope,
            )
            if not metrics.get("skipped", False):
                metrics["N"] = N
                metrics["K"] = K
                coarse_results.append(metrics)
            if RANK == 0:
                if metrics.get("skipped", False):
                    reason = metrics.get("reason", "unknown")
                    print(
                        f"[search][{model_name}] candidate {global_idx}/{len(candidates)} skipped({reason}): "
                        f"chunk={kernel_params['chunk_rows']}, stripe={kernel_params['stripe_rows']}, "
                        f"window={kernel_params['active_chunk_window']}, bands={kernel_params['n_bands']}, "
                        f"frontier={kernel_params['frontier_chunks']}, "
                        f"stage={kernel_params['stage_slots']}, lanes={kernel_params['comm_lanes']}, sms={kernel_params['num_comm_sms']}",
                        flush=True,
                    )
                else:
                    print(
                        f"[search][{model_name}] candidate {global_idx}/{len(candidates)}: "
                        f"speedup={metrics['new_speedup_vs_torch']:.4f}, total={metrics['new_total_ms']:.4f} ms, "
                        f"compaction={metrics['compaction_ratio']:.3f}, lead_ratio={metrics['lead_ratio']:.3f}, "
                        f"chunk={kernel_params['chunk_rows']}, stripe={kernel_params['stripe_rows']}, "
                        f"window={kernel_params['active_chunk_window']}, bands={kernel_params['n_bands']}, "
                        f"frontier={kernel_params['frontier_chunks']}, "
                        f"stage={kernel_params['stage_slots']}, lanes={kernel_params['comm_lanes']}, sms={kernel_params['num_comm_sms']}",
                        flush=True,
                    )

    if not coarse_results:
        return {"search_results": [], "verified_results": [], "pareto_results": []}

    coarse_results.sort(key=lambda x: (-x["new_speedup_vs_torch"], x["new_total_ms"], -x["compaction_ratio"]))
    pareto_results = compute_pareto_front(coarse_results)

    verify_count = min(args.search_verify_topk, len(coarse_results))
    verified_results = []
    for idx in range(verify_count):
        kernel_params = {
            "chunk_rows": int(coarse_results[idx]["chunk_rows"]),
            "stripe_rows": int(coarse_results[idx]["stripe_rows"]),
            "target_chunks": args.target_chunks,
            "min_chunk_rows": args.min_chunk_rows,
            "active_chunk_window": int(coarse_results[idx]["active_chunk_window"]),
            "n_bands": int(coarse_results[idx]["n_bands"]),
            "frontier_chunks": int(coarse_results[idx]["frontier_chunks"]),
            "producer_order": coarse_results[idx]["producer_order"],
            "stage_slots": int(coarse_results[idx]["stage_slots"]),
            "num_comm_sms": int(coarse_results[idx]["num_comm_sms"]),
            "comm_lanes": int(coarse_results[idx]["comm_lanes"]),
            "streaming_depth": args.streaming_depth,
        }
        if RANK == 0:
            print(f"[search][{model_name}] verifying top candidate #{idx + 1}", flush=True)
        verified = perf_test(
            model_name,
            M,
            config,
            pg,
            kernel_params=kernel_params,
            enable_check=args.check,
            enable_baseline=args.run_baseline,
            verbose=True,
            run_profile_pass=False,
            print_shape=False,
            alloc_envelope={
                "alloc_scatter_rows": int(coarse_results[idx]["stage_slots"]) * LOCAL_WORLD_SIZE * int(coarse_results[idx]["stripe_rows"]),
                "alloc_max_band_cols": triton.cdiv(N, max(1, int(coarse_results[idx]["n_bands"]))),
                "alloc_stage_slots": int(coarse_results[idx]["stage_slots"]),
            },
        )
        if not verified.get("skipped", False):
            verified["N"] = N
            verified["K"] = K
            verified_results.append(verified)

    if RANK == 0:
        print_search_summary(model_name, coarse_results, topk=args.search_topk, title="Top By Speedup")
        print_search_summary(model_name, pareto_results, topk=args.search_topk, title="Pareto Front")
        best = coarse_results[0]
        print(
            "[search] best command:\n"
            f"torchrun --nproc_per_node={pg.size()} python/triton_dist/benchmark/bench_new_windowed_panel_gemm_allreduce_v23.py "
            f"--M {M} --N {N} --K {K} --dtype {args.dtype} "
            f"--chunk_rows {int(best['chunk_rows'])} --stripe_rows {int(best['stripe_rows'])} "
            f"--active_chunk_window {int(best['active_chunk_window'])} --n_bands {int(best['n_bands'])} "
            f"--frontier_chunks {int(best['frontier_chunks'])} --stage_slots {int(best['stage_slots'])} "
            f"--producer_order {best['producer_order']} "
            f"--comm_lanes {int(best['comm_lanes'])} --num_comm_sms {int(best['num_comm_sms'])} "
            f"{'--autotune ' if args.autotune else ''}"
            f"{'--no-check' if not args.check else ''}",
            flush=True,
        )
        if args.dump_csv:
            csv_dir = Path("csv")
            csv_dir.mkdir(exist_ok=True)
            csv_file = csv_dir / f"perf_new_windowed_panel_gemm_allreduce_v23_search_{args.producer_order}_{pg.size()}_ranks.csv"
            write_search_csv(csv_file, coarse_results)
            print(f"[search] csv file is dumped into {csv_file}", flush=True)

    return {
        "search_results": coarse_results,
        "verified_results": verified_results,
        "pareto_results": pareto_results,
    }


if __name__ == "__main__":
    args = parse_args()
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    TP_GROUP = initialize_distributed()
    RANK = int(os.environ.get("RANK", 0))
    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
    LOCAL_WORLD_SIZE = int(os.environ.get("LOCAL_WORLD_SIZE", TP_GROUP.size()))

    configs = get_test_configs(args)
    perf_res = {}
    if RANK == 0:
        print(
            "new v2.3 kernel params:",
            {
                "chunk_rows": args.chunk_rows,
                "stripe_rows": args.stripe_rows,
                "target_chunks": args.target_chunks,
                "min_chunk_rows": args.min_chunk_rows,
                "active_chunk_window": args.active_chunk_window,
                "n_bands": args.n_bands,
                "frontier_chunks": args.frontier_chunks,
                "producer_order": args.producer_order,
                "stage_slots": args.stage_slots,
                "num_comm_sms": args.num_comm_sms,
                "comm_lanes": args.comm_lanes,
                "autotune": args.autotune,
                "streaming_depth": args.streaming_depth,
                "search": args.search,
            },
            flush=True,
        )

    for model_name, config in configs.items():
        if args.search:
            perf_res[model_name] = search_best_params(model_name, args.M, config, TP_GROUP)
        else:
            perf_res[model_name] = perf_test(model_name, args.M, config, TP_GROUP)

    if args.dump_csv and TP_GROUP.rank() == 0 and not args.search:
        csv_dir = Path("csv")
        csv_dir.mkdir(exist_ok=True)
        csv_file = csv_dir / f"perf_new_windowed_panel_gemm_allreduce_v23_{args.producer_order}_{TP_GROUP.size()}_ranks.csv"
        with open(csv_file, "w", encoding="utf-8") as fout:
            print(
                ",".join([
                    "Model",
                    "M",
                    "N",
                    "K",
                    "baseline_status",
                    "producer_order",
                    "new_total_ms",
                    "new_gemm_only_ms",
                    "new_ar_only_ms",
                    "baseline_total_ms",
                    "baseline_gemm_only_ms",
                    "baseline_ar_only_ms",
                    "torch_total_ms",
                    "torch_gemm_only_ms",
                    "torch_ar_only_ms",
                    "new_speedup_vs_torch",
                    "new_speedup_vs_baseline",
                    "new_overlap_ratio_vs_torch_serial",
                    "new_internal_overlap_ratio",
                ]),
                file=fout,
            )
            for model_name, config in configs.items():
                m = perf_res[model_name]
                if m.get("skipped", False):
                    print(
                        ",".join([
                            model_name,
                            str(args.M),
                            str(config["N"]),
                            str(config["K"]),
                            "skipped",
                            args.producer_order,
                            "nan",
                            "nan",
                            "nan",
                            "nan",
                            "nan",
                            "nan",
                            "nan",
                            "nan",
                            "nan",
                            "nan",
                            "nan",
                            "nan",
                            "nan",
                        ]),
                        file=fout,
                        flush=True,
                    )
                    continue
                print(
                    ",".join([
                        model_name,
                        str(args.M),
                        str(config["N"]),
                        str(config["K"]),
                        str(m["baseline_status"]),
                        str(m["producer_order"]),
                        f"{m['new_total_ms']:.6f}",
                        f"{m['new_gemm_only_ms']:.6f}",
                        f"{m['new_ar_only_ms']:.6f}",
                        f"{m['baseline_total_ms']:.6f}",
                        f"{m['baseline_gemm_only_ms']:.6f}",
                        f"{m['baseline_ar_only_ms']:.6f}",
                        f"{m['torch_total_ms']:.6f}",
                        f"{m['torch_gemm_only_ms']:.6f}",
                        f"{m['torch_ar_only_ms']:.6f}",
                        f"{m['new_speedup_vs_torch']:.6f}",
                        f"{m['new_speedup_vs_baseline']:.6f}",
                        f"{m['new_overlap_ratio_vs_torch_serial']:.6f}",
                        f"{m['new_internal_overlap_ratio']:.6f}",
                    ]),
                    file=fout,
                    flush=True,
                )
        print(f"csv file is dumped into {csv_file}")

    finalize_distributed()



