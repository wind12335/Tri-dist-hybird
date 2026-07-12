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
Frontier scheduling ablation benchmark for RS-GEMM.

This file is intentionally created as a separate benchmark instead of modifying
bench_3rdv5_frontier_windowed_panel_gemmrs.py. It keeps the same panelized /
windowed benchmark skeleton, but exposes a clean schedule ablation entry:

    --schedule_policy uniform
    --schedule_policy frontier_first
    --schedule_policy both

The "uniform" path uses the existing v3 windowed-panel producer as the
no-frontier baseline within the same panelized / windowed operator family.
The "frontier_first" path uses the existing v5 frontier dual-phase producer.
"""

import argparse
import gc
import os
from pathlib import Path
from typing import Callable, Dict, Tuple

import torch
import torch.distributed
import triton

from triton_dist.kernels.nvidia.gemm import get_config_space
from triton_dist.kernels.nvidia.new_3rd_v3_windowed_panel_rsgemm import (
    create_new_3rd_v3_windowed_panel_gemm_rs_context,
    launch_v2_panelized_gemm_producer,
    new_3rd_v3_windowed_panel_gemm_rs,
)
from triton_dist.kernels.nvidia.new_3rd_v3_windowed_panel_rs import new_3rd_v3_windowed_panel_rs_op
from triton_dist.kernels.nvidia.new_3rd_v5_frontier_windowed_panel_rsgemm import (
    create_new_3rd_v5_frontier_windowed_panel_gemm_rs_context,
    launch_v5_frontier_panelized_gemm_producer,
    new_3rd_v5_frontier_windowed_panel_gemm_rs,
)
from triton_dist.profiler_utils import group_profile, perf_func
from triton_dist.test.utils import LAYER_CONFIGS, assert_allclose
from triton_dist.utils import (
    dist_print,
    finalize_distributed,
    initialize_distributed,
    nvshmem_barrier_all_on_stream,
    rand_tensor,
    wait_until_max_gpu_clock_or_warning,
)


UNIFORM_AUTOTUNE_CACHE: dict[Tuple, dict] = {}
FRONTIER_AUTOTUNE_CACHE: dict[Tuple, dict] = {}


def debug_log(msg: str, rank: int | None = None) -> None:
    if args.debug:
        prefix = f"[bench-frontier-ablation][rank{rank}] " if rank is not None else "[bench-frontier-ablation] "
        print(prefix + msg, flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--M", type=int, default=8192)
    parser.add_argument("--N", type=int, default=None)
    parser.add_argument("--K", type=int, default=None)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--warmup_iters", type=int, default=5)
    parser.add_argument("--autotune", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--profile", action="store_true", default=False)
    parser.add_argument("--profile_target", type=str, default="all", choices=["all", "uniform", "frontier_first", "torch"])
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
    parser.add_argument(
        "--schedule_policy",
        type=str,
        default="both",
        choices=["uniform", "frontier_first", "both"],
        help=(
            "uniform uses the v3 panelized/windowed producer as the no-frontier baseline; "
            "frontier_first uses the v5 frontier dual-phase producer."
        ),
    )
    return parser.parse_args()


def get_test_configs(parsed_args):
    if parsed_args.N is not None or parsed_args.K is not None:
        if parsed_args.N is None or parsed_args.K is None:
            raise ValueError("`--N` and `--K` must be set together.")
        return {"custom": {"N": parsed_args.N, "K": parsed_args.K}}
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


def profile_section_lockstep(
    label: str,
    enabled: bool,
    fn: Callable[[], torch.Tensor],
    pg: torch.distributed.ProcessGroup,
    iters: int,
    warmup_iters: int,
) -> None:
    if not enabled:
        return
    with torch.profiler.record_function(label):
        perf_func_lockstep(fn, pg=pg, iters=iters, warmup_iters=warmup_iters)


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


def choose_gemm_config() -> triton.Config:
    return get_config_space(False)[0]


def get_autotuned_uniform_config(A: torch.Tensor, B: torch.Tensor, ctx, pg: torch.distributed.ProcessGroup):
    base_key = new_3rd_v3_windowed_panel_gemm_rs.key_fn(A, B, ctx, persistent=False)
    cache_key = (base_key, False)
    best_config = UNIFORM_AUTOTUNE_CACHE.get(cache_key)
    if best_config is None:
        debug_log("enter uniform autotune", pg.rank())
        config_space = new_3rd_v3_windowed_panel_gemm_rs.get_pruned_config(A, B, ctx, persistent=False)
        timings = new_3rd_v3_windowed_panel_gemm_rs.tune(config_space, pg, A, B, ctx, persistent=False)
        timings.sort(key=lambda x: x[0])
        assert len(timings) > 0, "uniform autotune returned empty timing list"
        best_config = timings[0][1]
        UNIFORM_AUTOTUNE_CACHE[cache_key] = best_config
        debug_log("leave uniform autotune", pg.rank())
    return best_config["gemm_config"]


def get_autotuned_frontier_config(A: torch.Tensor, B: torch.Tensor, ctx, pg: torch.distributed.ProcessGroup):
    base_key = new_3rd_v5_frontier_windowed_panel_gemm_rs.key_fn(A, B, ctx, persistent=False)
    cache_key = (base_key, False)
    best_config = FRONTIER_AUTOTUNE_CACHE.get(cache_key)
    if best_config is None:
        debug_log("enter frontier autotune", pg.rank())
        config_space = new_3rd_v5_frontier_windowed_panel_gemm_rs.get_pruned_config(A, B, ctx, persistent=False)
        timings = new_3rd_v5_frontier_windowed_panel_gemm_rs.tune(config_space, pg, A, B, ctx, persistent=False)
        timings.sort(key=lambda x: x[0])
        assert len(timings) > 0, "frontier autotune returned empty timing list"
        best_config = timings[0][1]
        FRONTIER_AUTOTUNE_CACHE[cache_key] = best_config
        debug_log("leave frontier autotune", pg.rank())
    return best_config["gemm_config"]


def run_uniform_stage(
    A: torch.Tensor,
    B: torch.Tensor,
    torch_partial: torch.Tensor,
    M: int,
    M_per_rank: int,
    pg: torch.distributed.ProcessGroup,
    atol: float,
    rtol: float,
    C_torch: torch.Tensor,
    metrics: dict[str, float],
) -> None:
    rank = pg.rank()
    world_size = pg.size()
    local_world_size = LOCAL_WORLD_SIZE
    ctx = None
    try:
        sync_all(pg)
        ctx = create_new_3rd_v3_windowed_panel_gemm_rs_context(
            M,
            B.shape[1],
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
            steady_sms=args.steady_sms,
            tail_sms=args.tail_sms,
            stage_slots=args.stage_slots,
            tail_chunk_window=args.tail_chunk_window,
            local_seed_direct=args.local_seed_direct,
        )
        metrics["uniform_chunk_rows"] = float(ctx.rs_ctx.chunk_rows)
        metrics["uniform_num_chunks"] = float(ctx.rs_ctx.num_chunks)
        metrics["uniform_active_chunk_window"] = float(ctx.rs_ctx.active_chunk_window)
        metrics["uniform_stage_slots"] = float(ctx.rs_ctx.stage_slots)
        metrics["uniform_comm_lanes"] = float(len(ctx.rs_ctx.comm_streams))
        metrics["uniform_n_bands"] = float(ctx.rs_ctx.n_bands)

        workspace = torch.zeros((ctx.rs_ctx.n_bands * world_size * ctx.rs_ctx.num_chunks,), dtype=torch.int32, device=A.device)
        rs_output = torch.empty((M_per_rank, B.shape[1]), dtype=dtype, device=A.device)
        gemm_out = ctx.get_gemm_out_buf(A)

        def _get_config():
            if args.autotune:
                return get_autotuned_uniform_config(A, B, ctx, pg)
            return choose_gemm_config()

        def _reset_runtime():
            ctx.rs_ctx.reset_runtime_state()
            sync_all(pg)

        def _total():
            return new_3rd_v3_windowed_panel_gemm_rs.fn(A, B, ctx, gemm_config=_get_config(), persistent=False)

        def _gemm_only():
            num_runtime_chunks = triton.cdiv(M_per_rank, ctx.rs_ctx.chunk_rows)
            signal_value = ctx.rs_ctx.begin_round(num_runtime_chunks)
            launch_v2_panelized_gemm_producer(A, B, gemm_out, ctx, workspace, signal_value, _get_config())
            return gemm_out

        def _rs_only():
            gemm_out.copy_(torch_partial)
            num_runtime_chunks = triton.cdiv(M_per_rank, ctx.rs_ctx.chunk_rows)
            signal_value = ctx.rs_ctx.begin_round(num_runtime_chunks)
            ctx.rs_ctx.chunk_signal.fill_(signal_value)
            return new_3rd_v3_windowed_panel_rs_op(gemm_out, ctx.rs_ctx, output=rs_output, prepare_round=False)

        _reset_runtime()
        C_uniform = None
        for _ in range(3):
            sync_all(pg)
            C_uniform = _total()
        for i in range(world_size):
            torch.distributed.barrier(pg, device_ids=[torch.cuda.current_device()])
            if rank == i and C_uniform is not None:
                assert_allclose(C_torch, C_uniform, atol=atol, rtol=rtol)

        if args.profile_target in ["all", "uniform"]:
            _reset_runtime()
            profile_section_lockstep("triton/uniform_total", args.profile, _total, pg, args.iters, args.warmup_iters)
            _reset_runtime()
            profile_section_lockstep("triton/uniform_gemm_only", args.profile, _gemm_only, pg, args.iters, args.warmup_iters)
            _reset_runtime()
            profile_section_lockstep("triton/uniform_rs_only", args.profile, _rs_only, pg, args.iters, args.warmup_iters)

        _reset_runtime()
        wait_until_max_gpu_clock_or_warning(torch.cuda.current_device())
        _, metrics["uniform_total_ms"] = perf_func_lockstep(_total, pg=pg, iters=args.iters, warmup_iters=args.warmup_iters)
        _reset_runtime()
        wait_until_max_gpu_clock_or_warning(torch.cuda.current_device())
        _, metrics["uniform_gemm_only_ms"] = perf_func_lockstep(_gemm_only, pg=pg, iters=args.iters, warmup_iters=args.warmup_iters)
        _reset_runtime()
        wait_until_max_gpu_clock_or_warning(torch.cuda.current_device())
        _, metrics["uniform_rs_only_ms"] = perf_func_lockstep(_rs_only, pg=pg, iters=args.iters, warmup_iters=args.warmup_iters)
    finally:
        sync_all(pg)
        if ctx is not None:
            ctx.finalize()
        sync_all(pg)
        release_python_cuda_refs()
        sync_all(pg)


def run_frontier_stage(
    A: torch.Tensor,
    B: torch.Tensor,
    torch_partial: torch.Tensor,
    M: int,
    M_per_rank: int,
    pg: torch.distributed.ProcessGroup,
    atol: float,
    rtol: float,
    C_torch: torch.Tensor,
    metrics: dict[str, float],
) -> None:
    rank = pg.rank()
    world_size = pg.size()
    local_world_size = LOCAL_WORLD_SIZE
    ctx = None
    try:
        sync_all(pg)
        ctx = create_new_3rd_v5_frontier_windowed_panel_gemm_rs_context(
            M,
            B.shape[1],
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
        metrics["frontier_chunk_rows"] = float(ctx.rs_ctx.chunk_rows)
        metrics["frontier_num_chunks"] = float(ctx.rs_ctx.num_chunks)
        metrics["frontier_active_chunk_window"] = float(ctx.rs_ctx.active_chunk_window)
        metrics["frontier_stage_slots"] = float(ctx.rs_ctx.stage_slots)
        metrics["frontier_comm_lanes"] = float(len(ctx.rs_ctx.comm_streams))
        metrics["frontier_n_bands"] = float(ctx.rs_ctx.n_bands)
        metrics["frontier_frontier_chunks"] = float(ctx.frontier_chunks)

        workspace = torch.zeros((ctx.rs_ctx.n_bands * world_size * ctx.rs_ctx.num_chunks,), dtype=torch.int32, device=A.device)
        rs_output = torch.empty((M_per_rank, B.shape[1]), dtype=dtype, device=A.device)
        gemm_out = ctx.get_gemm_out_buf(A)

        def _get_config():
            if args.autotune:
                return get_autotuned_frontier_config(A, B, ctx, pg)
            return choose_gemm_config()

        def _reset_runtime():
            ctx.rs_ctx.reset_runtime_state()
            sync_all(pg)

        def _total():
            return new_3rd_v5_frontier_windowed_panel_gemm_rs.fn(A, B, ctx, gemm_config=_get_config(), persistent=False)

        def _gemm_only():
            num_runtime_chunks = triton.cdiv(M_per_rank, ctx.rs_ctx.chunk_rows)
            signal_value = ctx.rs_ctx.begin_round(num_runtime_chunks)
            launch_v5_frontier_panelized_gemm_producer(A, B, gemm_out, ctx, workspace, signal_value, _get_config())
            return gemm_out

        def _rs_only():
            gemm_out.copy_(torch_partial)
            num_runtime_chunks = triton.cdiv(M_per_rank, ctx.rs_ctx.chunk_rows)
            signal_value = ctx.rs_ctx.begin_round(num_runtime_chunks)
            ctx.rs_ctx.chunk_signal.fill_(signal_value)
            return new_3rd_v3_windowed_panel_rs_op(gemm_out, ctx.rs_ctx, output=rs_output, prepare_round=False)

        _reset_runtime()
        C_frontier = None
        for _ in range(3):
            sync_all(pg)
            C_frontier = _total()
        for i in range(world_size):
            torch.distributed.barrier(pg, device_ids=[torch.cuda.current_device()])
            if rank == i and C_frontier is not None:
                assert_allclose(C_torch, C_frontier, atol=atol, rtol=rtol)

        if args.profile_target in ["all", "frontier_first"]:
            _reset_runtime()
            profile_section_lockstep("triton/frontier_total", args.profile, _total, pg, args.iters, args.warmup_iters)
            _reset_runtime()
            profile_section_lockstep("triton/frontier_gemm_only", args.profile, _gemm_only, pg, args.iters, args.warmup_iters)
            _reset_runtime()
            profile_section_lockstep("triton/frontier_rs_only", args.profile, _rs_only, pg, args.iters, args.warmup_iters)

        _reset_runtime()
        wait_until_max_gpu_clock_or_warning(torch.cuda.current_device())
        _, metrics["frontier_total_ms"] = perf_func_lockstep(_total, pg=pg, iters=args.iters, warmup_iters=args.warmup_iters)
        _reset_runtime()
        wait_until_max_gpu_clock_or_warning(torch.cuda.current_device())
        _, metrics["frontier_gemm_only_ms"] = perf_func_lockstep(_gemm_only, pg=pg, iters=args.iters, warmup_iters=args.warmup_iters)
        _reset_runtime()
        wait_until_max_gpu_clock_or_warning(torch.cuda.current_device())
        _, metrics["frontier_rs_only_ms"] = perf_func_lockstep(_rs_only, pg=pg, iters=args.iters, warmup_iters=args.warmup_iters)
    finally:
        sync_all(pg)
        if ctx is not None:
            ctx.finalize()
        sync_all(pg)
        release_python_cuda_refs()
        sync_all(pg)


def perf_test(model_name: str, M: int, config: Dict[str, int], pg: torch.distributed.ProcessGroup):
    N = config["N"]
    K = config["K"]
    rank = pg.rank()
    world_size = pg.size()
    local_world_size = LOCAL_WORLD_SIZE

    if world_size != local_world_size:
        raise AssertionError("frontier schedule ablation benchmark currently supports single-node runs only")
    if args.persistent:
        raise AssertionError("frontier schedule ablation benchmark currently supports only --no-persistent")
    if rank == 0:
        print(f"[{model_name}] test shape: M {M}, N {N}, K {K}, schedule_policy={args.schedule_policy}")
    debug_log(f"perf_test enter: autotune={args.autotune}, schedule_policy={args.schedule_policy}", rank)

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
        "uniform_total_ms": float("nan"),
        "uniform_gemm_only_ms": float("nan"),
        "uniform_rs_only_ms": float("nan"),
        "uniform_chunk_rows": float("nan"),
        "uniform_num_chunks": float("nan"),
        "uniform_active_chunk_window": float("nan"),
        "uniform_stage_slots": float("nan"),
        "uniform_comm_lanes": float("nan"),
        "uniform_n_bands": float("nan"),
        "frontier_total_ms": float("nan"),
        "frontier_gemm_only_ms": float("nan"),
        "frontier_rs_only_ms": float("nan"),
        "frontier_chunk_rows": float("nan"),
        "frontier_num_chunks": float("nan"),
        "frontier_active_chunk_window": float("nan"),
        "frontier_stage_slots": float("nan"),
        "frontier_comm_lanes": float("nan"),
        "frontier_n_bands": float("nan"),
        "frontier_frontier_chunks": float("nan"),
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

    run_id = os.environ.get("TORCHELASTIC_RUN_ID", "local")
    merge_profile_group = args.profile_merge_group if args.profile_merge_group is not None else pg.size() <= 4
    profile_with_stack = args.profile_with_stack if args.profile_with_stack is not None else pg.size() <= 4
    profile_barrier_after_merge = (
        args.profile_barrier_after_merge if args.profile_barrier_after_merge is not None else pg.size() <= 4
    )
    try:
        with group_profile(
            f"frontier_schedule_ablation_gemm_rs_perf_m_{M}_n_{N}_k_{K}_{run_id}",
            args.profile,
            merge_group=merge_profile_group,
            barrier_after_merge=profile_barrier_after_merge,
            with_stack=profile_with_stack,
            group=TP_GROUP,
        ):
            if args.schedule_policy in ["uniform", "both"]:
                run_uniform_stage(A, B, torch_partial, M, M_per_rank, pg, atol, rtol, C_torch, metrics)
            if args.schedule_policy in ["frontier_first", "both"]:
                run_frontier_stage(A, B, torch_partial, M, M_per_rank, pg, atol, rtol, C_torch, metrics)
            if args.profile_target in ["all", "torch"]:
                sync_all(pg)
                profile_section_lockstep("torch/reference_total", args.profile, _torch_total, pg, args.iters, args.warmup_iters)
                sync_all(pg)
                profile_section_lockstep("torch/reference_gemm_only", args.profile, _torch_gemm_only, pg, args.iters, args.warmup_iters)
                sync_all(pg)
                profile_section_lockstep("torch/reference_rs_only", args.profile, _torch_rs_only, pg, args.iters, args.warmup_iters)

        sync_all(pg)
        wait_until_max_gpu_clock_or_warning(torch.cuda.current_device())
        _, metrics["torch_total_ms"] = perf_func_lockstep(_torch_total, pg=pg, iters=args.iters, warmup_iters=args.warmup_iters)
        sync_all(pg)
        wait_until_max_gpu_clock_or_warning(torch.cuda.current_device())
        _, metrics["torch_gemm_only_ms"] = perf_func_lockstep(_torch_gemm_only, pg=pg, iters=args.iters, warmup_iters=args.warmup_iters)
        sync_all(pg)
        wait_until_max_gpu_clock_or_warning(torch.cuda.current_device())
        _, metrics["torch_rs_only_ms"] = perf_func_lockstep(_torch_rs_only, pg=pg, iters=args.iters, warmup_iters=args.warmup_iters)
    finally:
        sync_all(pg)

    if metrics["uniform_total_ms"] == metrics["uniform_total_ms"]:
        metrics["uniform_internal_overlap_ratio"] = 1.0 - metrics["uniform_total_ms"] / max(
            metrics["uniform_gemm_only_ms"] + metrics["uniform_rs_only_ms"],
            1e-6,
        )
        metrics["uniform_speedup_vs_torch"] = metrics["torch_total_ms"] / metrics["uniform_total_ms"]
    else:
        metrics["uniform_internal_overlap_ratio"] = float("nan")
        metrics["uniform_speedup_vs_torch"] = float("nan")

    if metrics["frontier_total_ms"] == metrics["frontier_total_ms"]:
        metrics["frontier_internal_overlap_ratio"] = 1.0 - metrics["frontier_total_ms"] / max(
            metrics["frontier_gemm_only_ms"] + metrics["frontier_rs_only_ms"],
            1e-6,
        )
        metrics["frontier_speedup_vs_torch"] = metrics["torch_total_ms"] / metrics["frontier_total_ms"]
    else:
        metrics["frontier_internal_overlap_ratio"] = float("nan")
        metrics["frontier_speedup_vs_torch"] = float("nan")

    if (
        metrics["uniform_total_ms"] == metrics["uniform_total_ms"]
        and metrics["frontier_total_ms"] == metrics["frontier_total_ms"]
    ):
        metrics["frontier_speedup_vs_uniform"] = metrics["uniform_total_ms"] / metrics["frontier_total_ms"]
    else:
        metrics["frontier_speedup_vs_uniform"] = float("nan")

    msg = (
        f"Rank {rank} [{model_name}] latency (ms): "
        f"torch_total={metrics['torch_total_ms']:.2f}, "
        f"torch_gemm_only={metrics['torch_gemm_only_ms']:.2f}, "
        f"torch_rs_only={metrics['torch_rs_only_ms']:.2f}"
    )
    if args.schedule_policy in ["uniform", "both"]:
        msg += (
            f", uniform_total={metrics['uniform_total_ms']:.2f}, "
            f"uniform_gemm_only={metrics['uniform_gemm_only_ms']:.2f}, "
            f"uniform_rs_only={metrics['uniform_rs_only_ms']:.2f}, "
            f"uniform_internal_overlap={metrics['uniform_internal_overlap_ratio']:.2%}, "
            f"uniform_speedup_vs_torch={metrics['uniform_speedup_vs_torch']:.2f}"
        )
    if args.schedule_policy in ["frontier_first", "both"]:
        msg += (
            f", frontier_total={metrics['frontier_total_ms']:.2f}, "
            f"frontier_gemm_only={metrics['frontier_gemm_only_ms']:.2f}, "
            f"frontier_rs_only={metrics['frontier_rs_only_ms']:.2f}, "
            f"frontier_internal_overlap={metrics['frontier_internal_overlap_ratio']:.2%}, "
            f"frontier_speedup_vs_torch={metrics['frontier_speedup_vs_torch']:.2f}, "
            f"frontier_chunks={metrics['frontier_frontier_chunks']:.0f}"
        )
    if args.schedule_policy == "both":
        msg += f", frontier_speedup_vs_uniform={metrics['frontier_speedup_vs_uniform']:.2f}"
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
        csv_file = Path("csv") / f"perf_frontier_schedule_ablation_gemm_rs_{TP_GROUP.size()}_ranks.csv"
        with open(csv_file, "w") as fout:
            print(
                ",".join(
                    [
                        "Model",
                        "M",
                        "N",
                        "K",
                        "schedule_policy",
                        "torch_total_ms",
                        "torch_gemm_only_ms",
                        "torch_rs_only_ms",
                        "uniform_total_ms",
                        "uniform_gemm_only_ms",
                        "uniform_rs_only_ms",
                        "uniform_internal_overlap_ratio",
                        "uniform_speedup_vs_torch",
                        "frontier_total_ms",
                        "frontier_gemm_only_ms",
                        "frontier_rs_only_ms",
                        "frontier_internal_overlap_ratio",
                        "frontier_speedup_vs_torch",
                        "frontier_speedup_vs_uniform",
                        "chunk_rows",
                        "active_chunk_window",
                        "stage_slots",
                        "comm_lanes",
                        "n_bands",
                        "frontier_chunks",
                    ]
                ),
                file=fout,
            )
            for model_name, config, metrics in perf_res:
                print(
                    ",".join(
                        [model_name]
                        + list(
                            map(
                                str,
                                [
                                    args.M,
                                    config["N"],
                                    config["K"],
                                    args.schedule_policy,
                                    f"{metrics['torch_total_ms']:.4f}",
                                    f"{metrics['torch_gemm_only_ms']:.4f}",
                                    f"{metrics['torch_rs_only_ms']:.4f}",
                                    f"{metrics['uniform_total_ms']:.4f}",
                                    f"{metrics['uniform_gemm_only_ms']:.4f}",
                                    f"{metrics['uniform_rs_only_ms']:.4f}",
                                    f"{metrics['uniform_internal_overlap_ratio']:.4f}",
                                    f"{metrics['uniform_speedup_vs_torch']:.4f}",
                                    f"{metrics['frontier_total_ms']:.4f}",
                                    f"{metrics['frontier_gemm_only_ms']:.4f}",
                                    f"{metrics['frontier_rs_only_ms']:.4f}",
                                    f"{metrics['frontier_internal_overlap_ratio']:.4f}",
                                    f"{metrics['frontier_speedup_vs_torch']:.4f}",
                                    f"{metrics['frontier_speedup_vs_uniform']:.4f}",
                                    args.chunk_rows,
                                    args.active_chunk_window,
                                    args.stage_slots,
                                    args.comm_lanes,
                                    args.n_bands,
                                    args.frontier_chunks,
                                ],
                            )
                        )
                    ),
                    file=fout,
                    flush=True,
                )
        print(f"csv file is dumped into {csv_file}")

    finalize_distributed()
