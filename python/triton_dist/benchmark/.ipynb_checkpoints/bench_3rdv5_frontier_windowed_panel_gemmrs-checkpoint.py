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
from typing import Callable, Dict, Tuple

import torch
import torch.distributed
import triton

from triton_dist.kernels.nvidia.gemm import get_config_space
from triton_dist.kernels.nvidia.new_3rd_v5_frontier_windowed_panel_rsgemm import (
    create_new_3rd_v5_frontier_windowed_panel_gemm_rs_context,
    launch_v5_frontier_panelized_gemm_producer,
    new_3rd_v5_frontier_windowed_panel_gemm_rs,
)
from triton_dist.kernels.nvidia.new_3rd_v5_frontier_windowed_panel_rs import new_3rd_v3_windowed_panel_rs_op
from triton_dist.kernels.nvidia.new_3rdgemm_reducescatterv3 import (
    create_new_3rd_gemm_rs_context,
    gemm_rs_producer_non_persistent_chunked,
    new_3rd_gemm_rs,
)
from triton_dist.kernels.nvidia.new_3rdreducescatter import new_3rd_reduce_scatter_2d_op
from triton_dist.profiler_utils import group_profile, perf_func
from triton_dist.test.utils import LAYER_CONFIGS, assert_allclose
from triton_dist.utils import (dist_print, finalize_distributed, initialize_distributed, nvshmem_barrier_all_on_stream,
                               rand_tensor, wait_until_max_gpu_clock_or_warning)


V2_AUTOTUNE_CACHE: dict[Tuple, dict] = {}
OLD_AUTOTUNE_CACHE: dict[Tuple, dict] = {}


def debug_log(msg: str, rank: int | None = None) -> None:
    if args.debug:
        prefix = f"[bench-v5-frontier][rank{rank}] " if rank is not None else "[bench-v5-frontier] "
        print(prefix + msg, flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--M", type=int, default=8192)
    parser.add_argument("--N", type=int, default=None)
    parser.add_argument("--K", type=int, default=None)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--warmup_iters", type=int, default=5)
    parser.add_argument("--autotune", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--mode", type=str, default="all", choices=["all", "v2", "new_3rd", "torch"])
    parser.add_argument("--profile", action="store_true", default=False)
    parser.add_argument("--profile_target", type=str, default="all", choices=["all", "v2", "new_3rd", "torch"])
    parser.add_argument("--profile_merge_group", default=None, action=argparse.BooleanOptionalAction)
    parser.add_argument("--profile_with_stack", default=None, action=argparse.BooleanOptionalAction)
    parser.add_argument("--profile_barrier_after_merge", default=None, action=argparse.BooleanOptionalAction)
    parser.add_argument("--dump_csv", action="store_true", default=False)
    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    parser.add_argument("--trans_b", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--persistent", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--chunk_rows", type=int, default=0)
    parser.add_argument("--target_chunks_per_rank", type=int, default=2)
    parser.add_argument("--min_chunk_rows", type=int, default=512)
    parser.add_argument("--active_chunk_window", type=int, default=4)
    parser.add_argument("--stage_slots", type=int, default=4)
    parser.add_argument("--steady_sms", type=int, default=6)
    parser.add_argument("--tail_sms", type=int, default=12)
    parser.add_argument("--tail_chunk_window", type=int, default=1)
    parser.add_argument("--comm_lanes", type=int, default=2)
    parser.add_argument("--n_bands", type=int, default=1, help="number of N-direction panels for 2D panel-ready")
    parser.add_argument("--frontier_chunks", type=int, default=1)
    parser.add_argument("--local_seed_direct", default=True, action=argparse.BooleanOptionalAction)
    return parser.parse_args()


def get_test_configs(args):
    if args.N is not None or args.K is not None:
        if args.N is None or args.K is None:
            raise ValueError("`--N` and `--K` must be set together.")
        return {"custom": {"N": args.N, "K": args.K}}
    return LAYER_CONFIGS


def make_data(M, N, K, dtype: torch.dtype, trans_b: bool, tp_group: torch.distributed.ProcessGroup):
    rank = tp_group.rank()
    world_size = tp_group.size()
    assert K % world_size == 0
    K_per_rank = K // world_size
    scale = (rank + 1) * 0.01

    device = torch.cuda.current_device()
    A = rand_tensor([M, K_per_rank], dtype=dtype, device=device) * scale
    if trans_b:
        B = (rand_tensor([N, K_per_rank], dtype=dtype, device=device) * scale).T.contiguous()
    else:
        B = (rand_tensor([K_per_rank, N], dtype=dtype, device=device) * scale).contiguous()
    return A, B


def torch_gemm_rs(pg: torch.distributed.ProcessGroup, A: torch.Tensor, B: torch.Tensor):
    M, _ = A.shape
    _, N = B.shape
    partial = torch.matmul(A, B)
    output = torch.empty((M // pg.size(), N), dtype=partial.dtype, device=A.device)
    torch.distributed.reduce_scatter_tensor(output, partial, group=pg)
    return output


def sync_all(pg: torch.distributed.ProcessGroup):
    nvshmem_barrier_all_on_stream(torch.cuda.current_stream())
    torch.cuda.synchronize()
    torch.distributed.barrier(pg, device_ids=[torch.cuda.current_device()])


def release_python_cuda_refs() -> None:
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    try:
        torch.cuda.ipc_collect()
    except Exception:
        pass


def profile_section(label: str, enabled: bool, fn: Callable[[], torch.Tensor], iters: int, warmup_iters: int) -> None:
    if not enabled:
        return
    with torch.profiler.record_function(label):
        perf_func(fn, iters=iters, warmup_iters=warmup_iters)


def perf_func_lockstep(func: Callable[[], torch.Tensor], pg: torch.distributed.ProcessGroup, iters: int, warmup_iters: int):
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    stop_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    output = None
    for n in range(iters + warmup_iters):
        sync_all(pg)
        if n >= warmup_iters:
            start_events[n - warmup_iters].record()
        output = func()
        if n >= warmup_iters:
            stop_events[n - warmup_iters].record()
        sync_all(pg)
    duration_ms = 0.0
    for i in range(iters):
        stop_events[i].synchronize()
        duration_ms += start_events[i].elapsed_time(stop_events[i])
    return output, duration_ms / iters


def profile_section_lockstep(label: str,
                             enabled: bool,
                             fn: Callable[[], torch.Tensor],
                             pg: torch.distributed.ProcessGroup,
                             iters: int,
                             warmup_iters: int) -> None:
    if not enabled:
        return
    with torch.profiler.record_function(label):
        perf_func_lockstep(fn, pg=pg, iters=iters, warmup_iters=warmup_iters)


def choose_gemm_config() -> triton.Config:
    return get_config_space(False)[0]


def get_autotuned_v2_config(A: torch.Tensor, B: torch.Tensor, ctx, pg: torch.distributed.ProcessGroup):
    base_key = new_3rd_v5_frontier_windowed_panel_gemm_rs.key_fn(A, B, ctx, persistent=False)
    cache_key = (base_key, False)
    best_config = V2_AUTOTUNE_CACHE.get(cache_key)
    if best_config is None:
        debug_log("enter v2 autotune", pg.rank())
        config_space = new_3rd_v5_frontier_windowed_panel_gemm_rs.get_pruned_config(A, B, ctx, persistent=False)
        timings = new_3rd_v5_frontier_windowed_panel_gemm_rs.tune(config_space, pg, A, B, ctx, persistent=False)
        timings.sort(key=lambda x: x[0])
        assert len(timings) > 0, "v2 autotune returned empty timing list"
        best_config = timings[0][1]
        V2_AUTOTUNE_CACHE[cache_key] = best_config
        debug_log("leave v2 autotune", pg.rank())
    return best_config["gemm_config"]


def get_autotuned_old_config(A: torch.Tensor, B: torch.Tensor, ctx, pg: torch.distributed.ProcessGroup):
    base_key = new_3rd_gemm_rs.key_fn(A, B, ctx, persistent=False)
    cache_key = (base_key, False)
    best_config = OLD_AUTOTUNE_CACHE.get(cache_key)
    if best_config is None:
        debug_log("enter old new_3rd autotune", pg.rank())
        config_space = new_3rd_gemm_rs.get_pruned_config(A, B, ctx, persistent=False)
        timings = new_3rd_gemm_rs.tune(config_space, pg, A, B, ctx, persistent=False)
        timings.sort(key=lambda x: x[0])
        assert len(timings) > 0, "old new_3rd autotune returned empty timing list"
        best_config = timings[0][1]
        OLD_AUTOTUNE_CACHE[cache_key] = best_config
        debug_log("leave old new_3rd autotune", pg.rank())
    return best_config["gemm_config"]


def perf_test(model_name: str, M: int, config: Dict[str, int], pg: torch.distributed.ProcessGroup):
    N = config["N"]
    K = config["K"]
    rank = pg.rank()
    world_size = pg.size()
    local_world_size = LOCAL_WORLD_SIZE

    run_old = args.mode in ["all", "new_3rd"]
    run_v2 = args.mode in ["all", "v2"]

    if (run_old or run_v2) and world_size != local_world_size:
        raise AssertionError("windowed_panel/new_3rd GEMM-RS benchmarks currently only support single-node runs")
    if args.persistent:
        raise AssertionError("windowed_panel/new_3rd GEMM-RS benchmarks currently support only --no-persistent")
    if rank == 0:
        print(f"[{model_name}] test shape: M {M}, N {N}, K {K}")
    debug_log(f"perf_test enter: mode={args.mode}, autotune={args.autotune}", rank)

    assert M % world_size == 0
    assert K % world_size == 0

    A, B = make_data(M, N, K, dtype, args.trans_b, pg)
    M_per_rank = M // world_size
    atol = 6e-2 if dtype == torch.bfloat16 else 1e-2
    rtol = atol
    torch_partial = torch.matmul(A, B)

    metrics = {
        "torch_total_ms": float("nan"),
        "torch_gemm_only_ms": float("nan"),
        "torch_rs_only_ms": float("nan"),
        "old_total_ms": float("nan"),
        "old_gemm_only_ms": float("nan"),
        "old_rs_only_ms": float("nan"),
        "v2_total_ms": float("nan"),
        "v2_gemm_only_ms": float("nan"),
        "v2_rs_only_ms": float("nan"),
        "v2_chunk_rows": float("nan"),
        "v2_num_chunks": float("nan"),
        "v2_active_chunk_window": float("nan"),
        "v2_stage_slots": float("nan"),
        "v2_comm_lanes": float("nan"),
        "v2_n_bands": float("nan"),
        "v2_frontier_chunks": float("nan"),
    }

    def _torch_total():
        return torch_gemm_rs(pg, A, B)

    def _torch_gemm_only():
        return torch.matmul(A, B)

    def _torch_rs_only():
        output = torch.empty((M_per_rank, N), dtype=dtype, device=A.device)
        torch.distributed.reduce_scatter_tensor(output, torch_partial, group=pg)
        return output

    sync_all(pg)
    C_torch = _torch_total()

    def _run_old_stage():
        ctx = None
        try:
            sync_all(pg)
            rs_stream = torch.cuda.Stream(priority=-1)
            ctx = create_new_3rd_gemm_rs_context(
                M,
                N,
                rank,
                world_size,
                local_world_size,
                dtype,
                rs_stream,
                chunk_rows=args.chunk_rows,
                target_chunks_per_rank=args.target_chunks_per_rank,
                min_chunk_rows=args.min_chunk_rows,
                helper_num_sms=args.steady_sms,
                steady_sms=args.steady_sms,
                tail_sms=args.tail_sms,
                stage_slots=args.stage_slots,
                tail_chunk_window=args.tail_chunk_window,
                local_seed_direct=args.local_seed_direct,
            )
            workspace = torch.zeros((world_size * ctx.rs_ctx.num_chunks,), dtype=torch.int32, device=A.device)
            rs_output = torch.empty((M_per_rank, N), dtype=dtype, device=A.device)
            gemm_out = ctx.get_gemm_out_buf(A)

            def _get_config():
                if args.autotune:
                    return get_autotuned_old_config(A, B, ctx, pg)
                return choose_gemm_config()

            def _old_total():
                return new_3rd_gemm_rs.fn(A, B, ctx, gemm_config=_get_config(), persistent=False)

            def _old_gemm_only():
                workspace.zero_()
                signal_value = ctx.rs_ctx.begin_round()
                gemm_rs_producer_non_persistent_chunked(
                    A,
                    B,
                    gemm_out,
                    ctx.rs_ctx.chunk_signal,
                    workspace,
                    ctx.rs_ctx.world_size,
                    ctx.rs_ctx.local_world_size,
                    ctx.rs_ctx.num_chunks,
                    ctx.rs_ctx.chunk_rows,
                    signal_value,
                    _get_config(),
                )
                return gemm_out

            def _old_rs_only():
                gemm_out.copy_(torch_partial)
                signal_value = ctx.rs_ctx.begin_round()
                ctx.rs_ctx.chunk_signal.fill_(signal_value)
                return new_3rd_reduce_scatter_2d_op(gemm_out, ctx.rs_ctx, output=rs_output)

            C_old = None
            for _ in range(3):
                sync_all(pg)
                C_old = _old_total()
            for i in range(world_size):
                torch.distributed.barrier(pg, device_ids=[torch.cuda.current_device()])
                if rank == i and C_old is not None:
                    assert_allclose(C_torch, C_old, atol=atol, rtol=rtol)

            if args.profile_target in ["all", "new_3rd"]:
                profile_section_lockstep("triton/old_total", args.profile, _old_total, pg, args.iters, args.warmup_iters)
                profile_section_lockstep("triton/old_gemm_only",
                                         args.profile,
                                         _old_gemm_only,
                                         pg,
                                         args.iters,
                                         args.warmup_iters)
                profile_section_lockstep("triton/old_rs_only", args.profile, _old_rs_only, pg, args.iters, args.warmup_iters)

            wait_until_max_gpu_clock_or_warning(torch.cuda.current_device())
            _, metrics["old_total_ms"] = perf_func_lockstep(_old_total,
                                                            pg=pg,
                                                            iters=args.iters,
                                                            warmup_iters=args.warmup_iters)
            wait_until_max_gpu_clock_or_warning(torch.cuda.current_device())
            _, metrics["old_gemm_only_ms"] = perf_func_lockstep(_old_gemm_only,
                                                                pg=pg,
                                                                iters=args.iters,
                                                                warmup_iters=args.warmup_iters)
            wait_until_max_gpu_clock_or_warning(torch.cuda.current_device())
            _, metrics["old_rs_only_ms"] = perf_func_lockstep(_old_rs_only,
                                                              pg=pg,
                                                              iters=args.iters,
                                                              warmup_iters=args.warmup_iters)
        finally:
            sync_all(pg)
            if ctx is not None:
                ctx.finalize()
            sync_all(pg)
            release_python_cuda_refs()
            sync_all(pg)

    def _run_v2_stage():
        ctx = None
        try:
            sync_all(pg)
            ctx = create_new_3rd_v5_frontier_windowed_panel_gemm_rs_context(
                M,
                N,
                rank,
                world_size,
                local_world_size,
                dtype,
                chunk_rows=args.chunk_rows,
                target_chunks_per_rank=args.target_chunks_per_rank,
                min_chunk_rows=args.min_chunk_rows,
                active_chunk_window=args.active_chunk_window,
                comm_lanes=args.comm_lanes,
                n_bands=args.n_bands,
                frontier_chunks=args.frontier_chunks,
                steady_sms=args.steady_sms,
                tail_sms=args.tail_sms,
                stage_slots=args.stage_slots,
                tail_chunk_window=args.tail_chunk_window,
                local_seed_direct=args.local_seed_direct,
            )
            metrics["v2_chunk_rows"] = float(ctx.rs_ctx.chunk_rows)
            metrics["v2_num_chunks"] = float(ctx.rs_ctx.num_chunks)
            metrics["v2_active_chunk_window"] = float(ctx.rs_ctx.active_chunk_window)
            metrics["v2_stage_slots"] = float(ctx.rs_ctx.stage_slots)
            metrics["v2_comm_lanes"] = float(len(ctx.rs_ctx.comm_streams))
            metrics["v2_n_bands"] = float(ctx.rs_ctx.n_bands)
            metrics["v2_frontier_chunks"] = float(ctx.frontier_chunks)

            workspace = torch.zeros((ctx.rs_ctx.n_bands * world_size * ctx.rs_ctx.num_chunks,),
                                    dtype=torch.int32,
                                    device=A.device)
            rs_output = torch.empty((M_per_rank, N), dtype=dtype, device=A.device)
            gemm_out = ctx.get_gemm_out_buf(A)

            def _get_config():
                if args.autotune:
                    return get_autotuned_v2_config(A, B, ctx, pg)
                return choose_gemm_config()

            def _reset_v2_runtime():
                ctx.rs_ctx.reset_runtime_state()
                sync_all(pg)

            def _v2_total():
                return new_3rd_v5_frontier_windowed_panel_gemm_rs.fn(A, B, ctx, gemm_config=_get_config(), persistent=False)

            def _v2_gemm_only():
                num_runtime_chunks = triton.cdiv(M_per_rank, ctx.rs_ctx.chunk_rows)
                signal_value = ctx.rs_ctx.begin_round(num_runtime_chunks)
                launch_v5_frontier_panelized_gemm_producer(A, B, gemm_out, ctx, workspace, signal_value, _get_config())
                return gemm_out

            def _v2_rs_only():
                gemm_out.copy_(torch_partial)
                num_runtime_chunks = triton.cdiv(M_per_rank, ctx.rs_ctx.chunk_rows)
                signal_value = ctx.rs_ctx.begin_round(num_runtime_chunks)
                ctx.rs_ctx.chunk_signal.fill_(signal_value)
                return new_3rd_v3_windowed_panel_rs_op(gemm_out, ctx.rs_ctx, output=rs_output, prepare_round=False)

            _reset_v2_runtime()
            C_v2 = None
            for _ in range(3):
                sync_all(pg)
                C_v2 = _v2_total()
            for i in range(world_size):
                torch.distributed.barrier(pg, device_ids=[torch.cuda.current_device()])
                if rank == i and C_v2 is not None:
                    assert_allclose(C_torch, C_v2, atol=atol, rtol=rtol)

            if args.profile_target in ["all", "v2"]:
                _reset_v2_runtime()
                profile_section_lockstep("triton/v2_total", args.profile, _v2_total, pg, args.iters, args.warmup_iters)
                _reset_v2_runtime()
                profile_section_lockstep("triton/v2_gemm_only",
                                         args.profile,
                                         _v2_gemm_only,
                                         pg,
                                         args.iters,
                                         args.warmup_iters)
                _reset_v2_runtime()
                profile_section_lockstep("triton/v2_rs_only", args.profile, _v2_rs_only, pg, args.iters, args.warmup_iters)

            _reset_v2_runtime()
            wait_until_max_gpu_clock_or_warning(torch.cuda.current_device())
            _, metrics["v2_total_ms"] = perf_func_lockstep(_v2_total,
                                                           pg=pg,
                                                           iters=args.iters,
                                                           warmup_iters=args.warmup_iters)
            _reset_v2_runtime()
            wait_until_max_gpu_clock_or_warning(torch.cuda.current_device())
            _, metrics["v2_gemm_only_ms"] = perf_func_lockstep(_v2_gemm_only,
                                                               pg=pg,
                                                               iters=args.iters,
                                                               warmup_iters=args.warmup_iters)
            _reset_v2_runtime()
            wait_until_max_gpu_clock_or_warning(torch.cuda.current_device())
            _, metrics["v2_rs_only_ms"] = perf_func_lockstep(_v2_rs_only,
                                                             pg=pg,
                                                             iters=args.iters,
                                                             warmup_iters=args.warmup_iters)
        finally:
            sync_all(pg)
            if ctx is not None:
                ctx.finalize()
            sync_all(pg)
            release_python_cuda_refs()
            sync_all(pg)

    run_id = os.environ.get("TORCHELASTIC_RUN_ID", "local")
    merge_profile_group = args.profile_merge_group if args.profile_merge_group is not None else pg.size() <= 4
    profile_with_stack = args.profile_with_stack if args.profile_with_stack is not None else pg.size() <= 4
    profile_barrier_after_merge = (
        args.profile_barrier_after_merge if args.profile_barrier_after_merge is not None else pg.size() <= 4
    )
    try:
        with group_profile(f"v5_frontier_windowed_panel_gemm_rs_perf_m_{M}_n_{N}_k_{K}_{run_id}",
                           args.profile,
                           merge_group=merge_profile_group,
                           barrier_after_merge=profile_barrier_after_merge,
                           with_stack=profile_with_stack,
                           group=TP_GROUP):
            if run_v2:
                _run_v2_stage()
            if run_old:
                _run_old_stage()
            if args.profile_target in ["all", "torch"]:
                sync_all(pg)
                profile_section_lockstep("torch/reference_total", args.profile, _torch_total, pg, args.iters,
                                         args.warmup_iters)
                sync_all(pg)
                profile_section_lockstep("torch/reference_gemm_only", args.profile, _torch_gemm_only, pg, args.iters,
                                         args.warmup_iters)
                sync_all(pg)
                profile_section_lockstep("torch/reference_rs_only", args.profile, _torch_rs_only, pg, args.iters,
                                         args.warmup_iters)

        sync_all(pg)
        wait_until_max_gpu_clock_or_warning(torch.cuda.current_device())
        _, metrics["torch_total_ms"] = perf_func_lockstep(_torch_total,
                                                          pg=pg,
                                                          iters=args.iters,
                                                          warmup_iters=args.warmup_iters)
        sync_all(pg)
        wait_until_max_gpu_clock_or_warning(torch.cuda.current_device())
        _, metrics["torch_gemm_only_ms"] = perf_func_lockstep(_torch_gemm_only,
                                                              pg=pg,
                                                              iters=args.iters,
                                                              warmup_iters=args.warmup_iters)
        sync_all(pg)
        wait_until_max_gpu_clock_or_warning(torch.cuda.current_device())
        _, metrics["torch_rs_only_ms"] = perf_func_lockstep(_torch_rs_only,
                                                            pg=pg,
                                                            iters=args.iters,
                                                            warmup_iters=args.warmup_iters)
    finally:
        sync_all(pg)

    metrics["old_internal_overlap_ratio"] = 1.0 - metrics["old_total_ms"] / max(metrics["old_gemm_only_ms"] + metrics["old_rs_only_ms"], 1e-6)
    metrics["v2_internal_overlap_ratio"] = 1.0 - metrics["v2_total_ms"] / max(metrics["v2_gemm_only_ms"] + metrics["v2_rs_only_ms"], 1e-6)
    metrics["old_speedup_vs_torch"] = metrics["torch_total_ms"] / metrics["old_total_ms"]
    metrics["v2_speedup_vs_torch"] = metrics["torch_total_ms"] / metrics["v2_total_ms"]
    metrics["v2_speedup_vs_old"] = metrics["old_total_ms"] / metrics["v2_total_ms"]

    msg = (
        f"Rank {rank} [{model_name}] latency (ms): "
        f"torch_total={metrics['torch_total_ms']:.2f}, "
        f"torch_gemm_only={metrics['torch_gemm_only_ms']:.2f}, "
        f"torch_rs_only={metrics['torch_rs_only_ms']:.2f}"
    )
    if run_old:
        msg += (
            f", old_new_3rd_total={metrics['old_total_ms']:.2f}, "
            f"old_new_3rd_gemm_only={metrics['old_gemm_only_ms']:.2f}, "
            f"old_new_3rd_rs_only={metrics['old_rs_only_ms']:.2f}, "
            f"old_new_3rd_internal_overlap={metrics['old_internal_overlap_ratio']:.2%}, "
            f"old_new_3rd_speedup_vs_torch={metrics['old_speedup_vs_torch']:.2f}"
        )
    if run_v2:
        msg += (
            f", v2_total={metrics['v2_total_ms']:.2f}, "
            f"v2_gemm_only={metrics['v2_gemm_only_ms']:.2f}, "
            f"v2_rs_only={metrics['v2_rs_only_ms']:.2f}, "
            f"v2_internal_overlap={metrics['v2_internal_overlap_ratio']:.2%}, "
            f"v2_speedup_vs_torch={metrics['v2_speedup_vs_torch']:.2f}, "
            f"v2_speedup_vs_old={metrics['v2_speedup_vs_old']:.2f}, "
            f"chunk_rows={metrics['v2_chunk_rows']:.0f}, "
            f"num_chunks={metrics['v2_num_chunks']:.0f}, "
            f"active_chunk_window={metrics['v2_active_chunk_window']:.0f}, "
            f"stage_slots={metrics['v2_stage_slots']:.0f}, "
            f"comm_lanes={metrics['v2_comm_lanes']:.0f}, "
            f"n_bands={metrics['v2_n_bands']:.0f}, "
            f"frontier_chunks={metrics['v2_frontier_chunks']:.0f}"
        )
    dist_print(msg, need_sync=True, allowed_ranks=list(range(world_size)))
    return metrics


if __name__ == "__main__":
    args = parse_args()
    if args.debug:
        os.environ["TRITON_DIST_NEW_3RD_DEBUG"] = "1"
        os.environ["TRITON_DIST_NEW_3RD_V2_DEBUG"] = "1"
    if args.persistent:
        raise AssertionError("persistent is not supported in this benchmark")

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    TP_GROUP = initialize_distributed()
    LOCAL_WORLD_SIZE = int(os.environ.get("LOCAL_WORLD_SIZE", TP_GROUP.size()))

    perf_res = []
    configs = get_test_configs(args)
    for model_name, config in configs.items():
        metrics = perf_test(model_name, args.M, config, TP_GROUP)
        perf_res.append((model_name, config, metrics))

    if args.dump_csv and TP_GROUP.rank() == 0:
        os.makedirs("csv", exist_ok=True)
        csv_file = Path("csv") / f"perf_3rdv5_frontier_windowed_panel_gemm_rs_{TP_GROUP.size()}_ranks.csv"
        with open(csv_file, "w") as fout:
            print(
                ",".join([
                    "Model",
                    "M",
                    "N",
                    "K",
                    "torch_total_ms",
                    "torch_gemm_only_ms",
                    "torch_rs_only_ms",
                    "old_total_ms",
                    "old_gemm_only_ms",
                    "old_rs_only_ms",
                    "v2_total_ms",
                    "v2_gemm_only_ms",
                    "v2_rs_only_ms",
                    "old_speedup_vs_torch",
                    "v2_speedup_vs_torch",
                    "v2_speedup_vs_old",
                    "old_internal_overlap_ratio",
                    "v2_internal_overlap_ratio",
                ]),
                file=fout,
            )
            for model_name, config, metrics in perf_res:
                print(
                    ",".join(
                        [model_name] +
                        list(
                            map(
                                str,
                                [
                                    args.M,
                                    config["N"],
                                    config["K"],
                                    f"{metrics['torch_total_ms']:.4f}",
                                    f"{metrics['torch_gemm_only_ms']:.4f}",
                                    f"{metrics['torch_rs_only_ms']:.4f}",
                                    f"{metrics['old_total_ms']:.4f}",
                                    f"{metrics['old_gemm_only_ms']:.4f}",
                                    f"{metrics['old_rs_only_ms']:.4f}",
                                    f"{metrics['v2_total_ms']:.4f}",
                                    f"{metrics['v2_gemm_only_ms']:.4f}",
                                    f"{metrics['v2_rs_only_ms']:.4f}",
                                    f"{metrics['old_speedup_vs_torch']:.4f}",
                                    f"{metrics['v2_speedup_vs_torch']:.4f}",
                                    f"{metrics['v2_speedup_vs_old']:.4f}",
                                    f"{metrics['old_internal_overlap_ratio']:.4f}",
                                    f"{metrics['v2_internal_overlap_ratio']:.4f}",
                                ],
                            ))),
                    file=fout,
                    flush=True,
                )
        print(f"csv file is dumped into {csv_file}")

    finalize_distributed()
