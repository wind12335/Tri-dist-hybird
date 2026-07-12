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
import os
from pathlib import Path
from typing import Callable, Dict, Tuple

import torch
import torch.distributed

from triton_dist.kernels.nvidia import create_gemm_rs_context, gemm_rs
from triton_dist.kernels.nvidia.gemm import get_config_space
from triton_dist.kernels.nvidia.gemm_reduce_scatter import (gemm_rs_producer_non_persistent,
                                                            gemm_rs_producer_persistent)
from triton_dist.kernels.nvidia.new_3rdgemm_reducescatter import (create_new_3rd_gemm_rs_context, new_3rd_gemm_rs,
                                                                  gemm_rs_producer_non_persistent_chunked)
from triton_dist.kernels.nvidia.new_3rdreducescatter import new_3rd_reduce_scatter_2d_op
from triton_dist.kernels.nvidia.reduce_scatter import reduce_scatter_2d_op
from triton_dist.profiler_utils import group_profile, perf_func
from triton_dist.test.utils import LAYER_CONFIGS, assert_allclose
from triton_dist.utils import (dist_print, finalize_distributed, initialize_distributed, nvshmem_barrier_all_on_stream,
                               rand_tensor, wait_until_max_gpu_clock_or_warning)

# torchrun --nproc_per_node=4 python/triton_dist/benchmark/bench_new_3rd_gemm_reducescatter.py \
#   --M 8192 --N 29568 --K 8192 \
#   --mode new_3rd --no-persistent \
#   --iters 10 --warmup_iters 5 --profile \
#   --chunk_rows 512 --stage_slots 4 \
#   --steady_sms 6 --tail_sms 12 \
#   --accum_dtype fp16 --use_scratch --local_seed_direct


BASE_AUTOTUNE_CACHE: dict[Tuple, dict] = {}
NEW_3RD_AUTOTUNE_CACHE: dict[Tuple, dict] = {}


def debug_log(msg: str, rank: int | None = None) -> None:
    if args.debug:
        prefix = f"[bench][rank{rank}] " if rank is not None else "[bench] "
        print(prefix + msg, flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--M", type=int, default=8192)
    parser.add_argument("--N", type=int, default=None)
    parser.add_argument("--K", type=int, default=None)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--warmup_iters", type=int, default=5)
    parser.add_argument("--autotune", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--mode", type=str, default="all", choices=["all", "nonfused", "new_3rd", "torch"])
    parser.add_argument("--profile", action="store_true", default=False)
    parser.add_argument("--profile_target",
                        type=str,
                        default="all",
                        choices=["all", "nonfused", "new_3rd", "torch"])
    parser.add_argument("--dump_csv", action="store_true", default=False)
    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    parser.add_argument("--trans_b", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--persistent",
                        action=argparse.BooleanOptionalAction,
                        default=torch.cuda.get_device_capability() >= (9, 0))
    parser.add_argument("--chunk_rows", type=int, default=0, help="chunk rows for new_3rd; 0 means heuristic auto")
    parser.add_argument("--target_chunks_per_rank",
                        type=int,
                        default=2,
                        help="heuristic target chunk count per rank when --chunk_rows=0")
    parser.add_argument("--min_chunk_rows",
                        type=int,
                        default=512,
                        help="minimum heuristic chunk rows for new_3rd")
    parser.add_argument("--helper_num_sms",
                        type=int,
                        default=4,
                        help="deprecated alias for steady-stage new_3rd reduce SM budget")
    parser.add_argument("--steady_sms",
                        type=int,
                        default=None,
                        help="SM budget for steady-phase chunk-local reduce kernels")
    parser.add_argument("--tail_sms",
                        type=int,
                        default=None,
                        help="SM budget for tail-phase chunk-local reduce kernels")
    parser.add_argument("--stage_slots",
                        type=int,
                        default=2,
                        help="number of slot streams used by elastic new_3rd reduce")
    parser.add_argument("--accum_dtype",
                        type=str,
                        default="fp32",
                        choices=["fp16", "bf16", "fp32", "output"],
                        help="legacy compatibility flag; single-kernel chunk reduce no longer uses scratch accumulation")
    parser.add_argument("--use_scratch",
                        default=True,
                        action=argparse.BooleanOptionalAction,
                        help="legacy compatibility flag; single-kernel chunk reduce ignores scratch mode")
    parser.add_argument("--local_seed_direct", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--tail_chunk_window",
                        type=int,
                        default=1,
                        help="last N chunks use tail_sms for chunk-local reduce launch")
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
    torch.distributed.barrier(pg)


def profile_section(label: str,
                    enabled: bool,
                    fn: Callable[[], torch.Tensor],
                    iters: int,
                    warmup_iters: int) -> None:
    if not enabled:
        return
    with torch.profiler.record_function(label):
        perf_func(fn, iters=iters, warmup_iters=warmup_iters)


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


def perf_func_lockstep(func: Callable[[], torch.Tensor],
                       pg: torch.distributed.ProcessGroup,
                       iters: int,
                       warmup_iters: int):
    """Measure an iteration-coupled distributed op without letting ranks drift.

    `new_3rd` uses rank-to-rank chunk signals and remote arrival flags. Unlike a
    standard collective, consecutive benchmark iterations are not independent if
    different ranks enter iteration `n+1` before peers have fully finished
    iteration `n`. We therefore align ranks at iteration boundaries while timing
    only the device work between CUDA events.
    """
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


def choose_gemm_config(persistent: bool):
    return get_config_space(persistent)[0]


def choose_accum_dtype(output_dtype: torch.dtype) -> torch.dtype:
    if args.accum_dtype == "output":
        return output_dtype
    return {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }[args.accum_dtype]


def get_autotuned_nonfused_config(A: torch.Tensor, B: torch.Tensor, ctx, persistent: bool,
                                  pg: torch.distributed.ProcessGroup):
    base_key = gemm_rs.key_fn(A, B, ctx)
    cache_key = (base_key, persistent, False)
    best_config = BASE_AUTOTUNE_CACHE.get(cache_key)
    if best_config is None:
        debug_log("enter nonfused autotune", pg.rank())
        config_space = gemm_rs.get_pruned_config(A, B, ctx, persistent=persistent, fuse_scatter=False)
        timings = gemm_rs.tune(config_space, pg, A, B, ctx, persistent=persistent, fuse_scatter=False)
        timings.sort(key=lambda x: x[0])
        assert len(timings) > 0, "nonfused autotune returned empty timing list"
        best_config = timings[0][1]
        BASE_AUTOTUNE_CACHE[cache_key] = best_config
        debug_log("leave nonfused autotune", pg.rank())
    return best_config["gemm_config"]


def get_autotuned_new_3rd_config(A: torch.Tensor, B: torch.Tensor, ctx, persistent: bool,
                                 pg: torch.distributed.ProcessGroup):
    base_key = new_3rd_gemm_rs.key_fn(A, B, ctx, persistent=persistent)
    cache_key = (base_key, persistent)
    best_config = NEW_3RD_AUTOTUNE_CACHE.get(cache_key)
    if best_config is None:
        debug_log("enter new_3rd autotune", pg.rank())
        config_space = new_3rd_gemm_rs.get_pruned_config(A, B, ctx, persistent=persistent)
        timings = new_3rd_gemm_rs.tune(config_space, pg, A, B, ctx, persistent=persistent)
        timings.sort(key=lambda x: x[0])
        assert len(timings) > 0, "new_3rd autotune returned empty timing list"
        best_config = timings[0][1]
        NEW_3RD_AUTOTUNE_CACHE[cache_key] = best_config
        debug_log("leave new_3rd autotune", pg.rank())
    return best_config["gemm_config"]


def launch_nonfused_gemm_producer(A: torch.Tensor, B: torch.Tensor, ctx, workspace: torch.Tensor, persistent: bool,
                                  gemm_config):
    gemm_out = ctx.get_gemm_out_buf(A)
    scatter_signal = ctx.rs_ctx.scatter_signal_buf
    workspace.zero_()
    if hasattr(ctx.rs_ctx, "reset_runtime_state"):
        ctx.rs_ctx.reset_runtime_state()
    else:
        ctx.rs_ctx.reset_barriers()
    if persistent:
        gemm_rs_producer_persistent(
            A,
            B,
            gemm_out,
            scatter_signal,
            workspace,
            ctx.rs_ctx.world_size,
            ctx.rs_ctx.local_world_size,
            False,
            ctx.num_gemm_sms,
            gemm_config,
        )
    else:
        gemm_rs_producer_non_persistent(
            A,
            B,
            gemm_out,
            scatter_signal,
            workspace,
            ctx.rs_ctx.world_size,
            ctx.rs_ctx.local_world_size,
            False,
            gemm_config,
        )
    return gemm_out


def launch_new_3rd_chunked_gemm_producer(A: torch.Tensor, B: torch.Tensor, ctx, workspace: torch.Tensor, gemm_config):
    gemm_out = ctx.get_gemm_out_buf(A)
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
        gemm_config,
    )
    return gemm_out


def perf_test(model_name: str, M: int, config: Dict[str, int], pg: torch.distributed.ProcessGroup):
    N = config["N"]
    K = config["K"]
    rank = pg.rank()
    world_size = pg.size()
    local_world_size = LOCAL_WORLD_SIZE

    run_nonfused = args.mode in ["all", "nonfused"]
    run_new_3rd = args.mode in ["all", "new_3rd"]

    if run_new_3rd and world_size != local_world_size:
        raise AssertionError("new_3rd GEMM-RS benchmark currently only supports single-node runs")
    if run_new_3rd and args.persistent:
        raise AssertionError("new_3rd GEMM-RS benchmark currently supports only --no-persistent")

    if rank == 0:
        print(f"[{model_name}] test shape: M {M}, N {N}, K {K}")
    debug_log(f"perf_test enter: mode={args.mode}, autotune={args.autotune}, persistent={args.persistent}", rank)

    assert M % world_size == 0
    assert K % world_size == 0

    A, B = make_data(M, N, K, dtype, args.trans_b, pg)
    M_per_rank = M // world_size
    default_gemm_config = choose_gemm_config(args.persistent)

    atol = 6e-2 if dtype == torch.bfloat16 else 1e-2
    rtol = atol
    torch_partial = torch.matmul(A, B)

    metrics = {
        "torch_total_ms": float("nan"),
        "torch_gemm_only_ms": float("nan"),
        "torch_rs_only_ms": float("nan"),
        "triton_nonfused_total_ms": float("nan"),
        "triton_nonfused_gemm_only_ms": float("nan"),
        "triton_nonfused_rs_only_ms": float("nan"),
        "new_3rd_total_ms": float("nan"),
        "new_3rd_gemm_only_ms": float("nan"),
        "new_3rd_rs_only_ms": float("nan"),
        "new_3rd_chunk_rows": float("nan"),
        "new_3rd_num_chunks": float("nan"),
        "new_3rd_steady_sms": float("nan"),
        "new_3rd_tail_sms": float("nan"),
        "new_3rd_stage_slots": float("nan"),
        "new_3rd_use_scratch": float("nan"),
    }
    contexts_to_finalize = []

    def _torch_total():
        return torch_gemm_rs(pg, A, B)

    def _torch_gemm_only():
        return torch.matmul(A, B)

    def _torch_rs_only():
        output = torch.empty((M_per_rank, N), dtype=dtype, device=A.device)
        torch.distributed.reduce_scatter_tensor(output, torch_partial, group=pg)
        return output

    sync_all(pg)
    debug_log("about to run torch reference once for correctness baseline", rank)
    C_torch = _torch_total()
    debug_log("torch reference baseline done", rank)

    def _run_nonfused_stage():
        ctx = None
        try:
            sync_all(pg)
            rs_stream = torch.cuda.Stream(priority=-1)
            ctx = create_gemm_rs_context(M, N, rank, world_size, local_world_size, dtype, rs_stream)
            contexts_to_finalize.append(ctx)
            workspace = torch.zeros((world_size,), dtype=torch.int32, device=A.device)
            rs_output = torch.empty((M_per_rank, N), dtype=dtype, device=A.device)
            gemm_out = ctx.get_gemm_out_buf(A)

            def _get_config():
                if args.autotune:
                    return get_autotuned_nonfused_config(A, B, ctx, args.persistent, pg)
                return default_gemm_config

            def _nonfused_total():
                return gemm_rs.fn(
                    A,
                    B,
                    ctx,
                    gemm_config=_get_config(),
                    persistent=args.persistent,
                    fuse_scatter=False,
                )

            def _nonfused_gemm_only():
                return launch_nonfused_gemm_producer(A, B, ctx, workspace, args.persistent, _get_config())

            def _nonfused_rs_only():
                gemm_out.copy_(torch_partial)
                ctx.rs_ctx.reset_barriers()
                ctx.rs_ctx.scatter_signal_buf.fill_(1)
                return reduce_scatter_2d_op(gemm_out, ctx.rs_ctx, output=rs_output)

            C_nonfused = None
            for _ in range(3):
                debug_log("nonfused warmup iteration start", rank)
                sync_all(pg)
                C_nonfused = _nonfused_total()
                debug_log("nonfused warmup iteration end", rank)

            for i in range(world_size):
                torch.distributed.barrier(pg)
                if rank == i and C_nonfused is not None:
                    assert_allclose(C_torch, C_nonfused, atol=atol, rtol=rtol)

            if args.profile_target in ["all", "nonfused"]:
                sync_all(pg)
                profile_section("triton/nonfused_total", args.profile, _nonfused_total, args.iters, args.warmup_iters)
                sync_all(pg)
                profile_section("triton/nonfused_gemm_only", args.profile, _nonfused_gemm_only, args.iters,
                                args.warmup_iters)
                sync_all(pg)
                profile_section("triton/nonfused_rs_only", args.profile, _nonfused_rs_only, args.iters,
                                args.warmup_iters)

            sync_all(pg)
            debug_log("measuring nonfused_total", rank)
            wait_until_max_gpu_clock_or_warning(torch.cuda.current_device())
            _, metrics["triton_nonfused_total_ms"] = perf_func(_nonfused_total,
                                                               iters=args.iters,
                                                               warmup_iters=args.warmup_iters)
            sync_all(pg)
            debug_log("measuring nonfused_gemm_only", rank)
            wait_until_max_gpu_clock_or_warning(torch.cuda.current_device())
            _, metrics["triton_nonfused_gemm_only_ms"] = perf_func(_nonfused_gemm_only,
                                                                   iters=args.iters,
                                                                   warmup_iters=args.warmup_iters)
            sync_all(pg)
            debug_log("measuring nonfused_rs_only", rank)
            wait_until_max_gpu_clock_or_warning(torch.cuda.current_device())
            _, metrics["triton_nonfused_rs_only_ms"] = perf_func(_nonfused_rs_only,
                                                                 iters=args.iters,
                                                                 warmup_iters=args.warmup_iters)
        finally:
            sync_all(pg)

    def _run_new_3rd_stage():
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
                helper_num_sms=args.helper_num_sms,
                steady_sms=args.steady_sms,
                tail_sms=args.tail_sms,
                stage_slots=args.stage_slots,
                accum_dtype=choose_accum_dtype(dtype),
                use_scratch=args.use_scratch,
                tail_chunk_window=args.tail_chunk_window,
                local_seed_direct=args.local_seed_direct,
            )
            contexts_to_finalize.append(ctx)
            metrics["new_3rd_chunk_rows"] = float(ctx.rs_ctx.chunk_rows)
            metrics["new_3rd_num_chunks"] = float(ctx.rs_ctx.num_chunks)
            metrics["new_3rd_steady_sms"] = float(ctx.rs_ctx.steady_sms)
            metrics["new_3rd_tail_sms"] = float(ctx.rs_ctx.tail_sms)
            metrics["new_3rd_stage_slots"] = float(ctx.rs_ctx.stage_slots)
            metrics["new_3rd_use_scratch"] = float(int(ctx.rs_ctx.use_scratch))
            workspace = torch.zeros((world_size * ctx.rs_ctx.num_chunks,), dtype=torch.int32, device=A.device)
            rs_output = torch.empty((M_per_rank, N), dtype=dtype, device=A.device)
            gemm_out = ctx.get_gemm_out_buf(A)

            def _get_config():
                if args.autotune:
                    return get_autotuned_new_3rd_config(A, B, ctx, args.persistent, pg)
                return default_gemm_config

            def _new_3rd_total():
                return new_3rd_gemm_rs.fn(
                    A,
                    B,
                    ctx,
                    gemm_config=_get_config(),
                    persistent=args.persistent,
                )

            def _new_3rd_gemm_only():
                return launch_new_3rd_chunked_gemm_producer(A, B, ctx, workspace, _get_config())

            def _new_3rd_rs_only():
                gemm_out.copy_(torch_partial)
                signal_value = ctx.rs_ctx.begin_round()
                ctx.rs_ctx.chunk_signal.fill_(signal_value)
                return new_3rd_reduce_scatter_2d_op(gemm_out, ctx.rs_ctx, output=rs_output)

            C_new_3rd = None
            for _ in range(3):
                debug_log("new_3rd warmup iteration start", rank)
                sync_all(pg)
                C_new_3rd = _new_3rd_total()
                debug_log("new_3rd warmup iteration end", rank)

            for i in range(world_size):
                torch.distributed.barrier(pg)
                if rank == i and C_new_3rd is not None:
                    assert_allclose(C_torch, C_new_3rd, atol=atol, rtol=rtol)

            if args.profile_target in ["all", "new_3rd"]:
                sync_all(pg)
                profile_section_lockstep("triton/new_3rd_total", args.profile, _new_3rd_total, pg, args.iters,
                                         args.warmup_iters)
                sync_all(pg)
                profile_section_lockstep("triton/new_3rd_gemm_only", args.profile, _new_3rd_gemm_only, pg,
                                         args.iters,
                                         args.warmup_iters)
                sync_all(pg)
                profile_section_lockstep("triton/new_3rd_rs_only", args.profile, _new_3rd_rs_only, pg, args.iters,
                                         args.warmup_iters)

            sync_all(pg)
            debug_log("measuring new_3rd_total", rank)
            wait_until_max_gpu_clock_or_warning(torch.cuda.current_device())
            _, metrics["new_3rd_total_ms"] = perf_func_lockstep(_new_3rd_total,
                                                                pg=pg,
                                                                iters=args.iters,
                                                                warmup_iters=args.warmup_iters)
            sync_all(pg)
            debug_log("measuring new_3rd_gemm_only", rank)
            wait_until_max_gpu_clock_or_warning(torch.cuda.current_device())
            _, metrics["new_3rd_gemm_only_ms"] = perf_func_lockstep(_new_3rd_gemm_only,
                                                                    pg=pg,
                                                                    iters=args.iters,
                                                                    warmup_iters=args.warmup_iters)
            sync_all(pg)
            debug_log("measuring new_3rd_rs_only", rank)
            wait_until_max_gpu_clock_or_warning(torch.cuda.current_device())
            _, metrics["new_3rd_rs_only_ms"] = perf_func_lockstep(_new_3rd_rs_only,
                                                                  pg=pg,
                                                                  iters=args.iters,
                                                                  warmup_iters=args.warmup_iters)
        finally:
            sync_all(pg)

    run_id = os.environ.get("TORCHELASTIC_RUN_ID", "local")
    try:
        with group_profile(f"new_3rd_gemm_rs_perf_m_{M}_n_{N}_k_{K}_{run_id}", args.profile, group=TP_GROUP):
            if run_nonfused:
                _run_nonfused_stage()
            if run_new_3rd:
                _run_new_3rd_stage()
            if args.profile_target in ["all", "torch"]:
                sync_all(pg)
                profile_section("torch/reference_total", args.profile, _torch_total, args.iters, args.warmup_iters)
                sync_all(pg)
                profile_section("torch/reference_gemm_only", args.profile, _torch_gemm_only, args.iters,
                                args.warmup_iters)
                sync_all(pg)
                profile_section("torch/reference_rs_only", args.profile, _torch_rs_only, args.iters,
                                args.warmup_iters)

        sync_all(pg)
        wait_until_max_gpu_clock_or_warning(torch.cuda.current_device())
        _, metrics["torch_total_ms"] = perf_func(_torch_total, iters=args.iters, warmup_iters=args.warmup_iters)
        sync_all(pg)
        wait_until_max_gpu_clock_or_warning(torch.cuda.current_device())
        _, metrics["torch_gemm_only_ms"] = perf_func(_torch_gemm_only, iters=args.iters, warmup_iters=args.warmup_iters)
        sync_all(pg)
        wait_until_max_gpu_clock_or_warning(torch.cuda.current_device())
        _, metrics["torch_rs_only_ms"] = perf_func(_torch_rs_only, iters=args.iters, warmup_iters=args.warmup_iters)
    finally:
        for ctx in reversed(contexts_to_finalize):
            sync_all(pg)
            ctx.finalize()
        sync_all(pg)

    serial_torch_ms = metrics["torch_gemm_only_ms"] + metrics["torch_rs_only_ms"]
    metrics["triton_nonfused_overlap_ratio"] = ((serial_torch_ms - metrics["triton_nonfused_total_ms"]) /
                                                max(serial_torch_ms, 1e-6))
    metrics["new_3rd_internal_overlap_ratio"] = 1.0 - metrics["new_3rd_total_ms"] / max(
        metrics["new_3rd_gemm_only_ms"] + metrics["new_3rd_rs_only_ms"], 1e-6)
    metrics["nonfused_speedup_vs_torch"] = metrics["torch_total_ms"] / metrics["triton_nonfused_total_ms"]
    metrics["new_3rd_speedup_vs_torch"] = metrics["torch_total_ms"] / metrics["new_3rd_total_ms"]
    metrics["new_3rd_speedup_vs_nonfused"] = metrics["triton_nonfused_total_ms"] / metrics["new_3rd_total_ms"]

    flops = 2 * M * N * (K // world_size)
    reduce_scatter_gb = M * N * dtype.itemsize / 2**30 * (world_size - 1) / world_size
    metrics["triton_nonfused_tflops"] = flops / metrics["triton_nonfused_total_ms"] * 1e-9
    metrics["new_3rd_tflops"] = flops / metrics["new_3rd_total_ms"] * 1e-9
    metrics["torch_gemm_tflops"] = flops / metrics["torch_gemm_only_ms"] * 1e-9
    metrics["torch_rs_gbps"] = reduce_scatter_gb / metrics["torch_rs_only_ms"] * 1e3
    metrics["triton_nonfused_rs_gbps"] = reduce_scatter_gb / metrics["triton_nonfused_rs_only_ms"] * 1e3
    metrics["new_3rd_rs_gbps"] = reduce_scatter_gb / metrics["new_3rd_rs_only_ms"] * 1e3

    msg = (
        f"Rank {rank} [{model_name}] latency (ms): "
        f"torch_total={metrics['torch_total_ms']:.2f}, "
        f"torch_gemm_only={metrics['torch_gemm_only_ms']:.2f}, "
        f"torch_rs_only={metrics['torch_rs_only_ms']:.2f}"
    )
    if run_nonfused:
        msg += (
            f", triton_nonfused_total={metrics['triton_nonfused_total_ms']:.2f}, "
            f"triton_nonfused_gemm_only={metrics['triton_nonfused_gemm_only_ms']:.2f}, "
            f"triton_nonfused_rs_only={metrics['triton_nonfused_rs_only_ms']:.2f}, "
            f"nonfused_speedup_vs_torch={metrics['nonfused_speedup_vs_torch']:.2f}, "
            f"nonfused_overlap_ratio={metrics['triton_nonfused_overlap_ratio']:.2%}"
        )
    if run_new_3rd:
        msg += (
            f", new_3rd_total={metrics['new_3rd_total_ms']:.2f}, "
            f"new_3rd_gemm_only={metrics['new_3rd_gemm_only_ms']:.2f}, "
            f"new_3rd_rs_only={metrics['new_3rd_rs_only_ms']:.2f}, "
            f"new_3rd_internal_overlap={metrics['new_3rd_internal_overlap_ratio']:.2%}, "
            f"new_3rd_speedup_vs_torch={metrics['new_3rd_speedup_vs_torch']:.2f}, "
            f"new_3rd_speedup_vs_nonfused={metrics['new_3rd_speedup_vs_nonfused']:.2f}, "
            f"chunk_rows={metrics['new_3rd_chunk_rows']:.0f}, "
            f"num_chunks={metrics['new_3rd_num_chunks']:.0f}, "
            f"steady_sms={metrics['new_3rd_steady_sms']:.0f}, "
            f"tail_sms={metrics['new_3rd_tail_sms']:.0f}, "
            f"stage_slots={metrics['new_3rd_stage_slots']:.0f}, "
            f"use_scratch={int(metrics['new_3rd_use_scratch'])}"
        )

    dist_print(msg, need_sync=True, allowed_ranks=list(range(world_size)))
    return metrics


if __name__ == "__main__":
    args = parse_args()
    if args.debug:
        os.environ["TRITON_DIST_NEW_3RD_DEBUG"] = "1"
    if torch.cuda.get_device_capability()[0] < 9 and args.persistent:
        raise AssertionError("persistent GEMM-RS is not supported on cuda capability < 9.0")

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
        csv_file = Path("csv") / f"perf_new_3rd_gemm_rs_{TP_GROUP.size()}_ranks.csv"
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
                    "triton_nonfused_total_ms",
                    "triton_nonfused_gemm_only_ms",
                    "triton_nonfused_rs_only_ms",
                    "new_3rd_total_ms",
                    "new_3rd_gemm_only_ms",
                    "new_3rd_rs_only_ms",
                    "nonfused_speedup_vs_torch",
                    "new_3rd_speedup_vs_torch",
                    "new_3rd_speedup_vs_nonfused",
                    "triton_nonfused_overlap_ratio",
                    "new_3rd_internal_overlap_ratio",
                    "triton_nonfused_tflops",
                    "new_3rd_tflops",
                    "torch_gemm_tflops",
                    "torch_rs_gbps",
                    "triton_nonfused_rs_gbps",
                    "new_3rd_rs_gbps",
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
                                    f"{metrics['triton_nonfused_total_ms']:.4f}",
                                    f"{metrics['triton_nonfused_gemm_only_ms']:.4f}",
                                    f"{metrics['triton_nonfused_rs_only_ms']:.4f}",
                                    f"{metrics['new_3rd_total_ms']:.4f}",
                                    f"{metrics['new_3rd_gemm_only_ms']:.4f}",
                                    f"{metrics['new_3rd_rs_only_ms']:.4f}",
                                    f"{metrics['nonfused_speedup_vs_torch']:.4f}",
                                    f"{metrics['new_3rd_speedup_vs_torch']:.4f}",
                                    f"{metrics['new_3rd_speedup_vs_nonfused']:.4f}",
                                    f"{metrics['triton_nonfused_overlap_ratio']:.4f}",
                                    f"{metrics['new_3rd_internal_overlap_ratio']:.4f}",
                                    f"{metrics['triton_nonfused_tflops']:.4f}",
                                    f"{metrics['new_3rd_tflops']:.4f}",
                                    f"{metrics['torch_gemm_tflops']:.4f}",
                                    f"{metrics['torch_rs_gbps']:.4f}",
                                    f"{metrics['triton_nonfused_rs_gbps']:.4f}",
                                    f"{metrics['new_3rd_rs_gbps']:.4f}",
                                ],
                            ))),
                    file=fout,
                    flush=True,
                )
        print(f"csv file is dumped into {csv_file}")

    finalize_distributed()
