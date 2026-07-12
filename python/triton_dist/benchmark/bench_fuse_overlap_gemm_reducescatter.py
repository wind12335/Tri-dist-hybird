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

from triton_dist.kernels.nvidia import (create_fuse_overlap_gemm_rs_context, create_gemm_rs_context,
                                        fuse_overlap_gemm_rs, gemm_rs)
from triton_dist.kernels.nvidia.fuse_overlap_gemm_reducescatter import (
    fuse_overlap_gemm_rs_producer_non_persistent,
)
from triton_dist.kernels.nvidia.fuse_overlap_reducescatter import fuse_overlap_reduce_scatter_2d_op
from triton_dist.kernels.nvidia.gemm import get_config_space
from triton_dist.kernels.nvidia.gemm_reduce_scatter import (gemm_rs_producer_non_persistent,
                                                            gemm_rs_producer_persistent)
from triton_dist.kernels.nvidia.reduce_scatter import reduce_scatter_2d_op, ring_reduce
from triton_dist.profiler_utils import group_profile, perf_func
from triton_dist.test.utils import LAYER_CONFIGS, assert_allclose
from triton_dist.utils import (dist_print, finalize_distributed, initialize_distributed, nvshmem_barrier_all_on_stream,
                               rand_tensor, wait_until_max_gpu_clock_or_warning)

#torchrun --nproc_per_node=4 python/triton_dist/benchmark/bench_fuse_overlap_gemm_reducescatter.py   --M 8192 --N 8192 --K 8192   --mode fuse_overlap   --autotune --no-persistent      --iters 10 --warmup_iters 5
# torchrun --nproc_per_node=4 python/triton_dist/benchmark/bench_fuse_overlap_gemm_reducescatter.py   --M 8192 --N 8192 --K 8192   --mode fused   --autotune --no-persistent      --iters 10 --warmup_iters 5
TRITON_AUTOTUNE_CACHE: dict[Tuple, dict] = {}
FUSE_OVERLAP_AUTOTUNE_CACHE: dict[Tuple, dict] = {}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--M", type=int, default=8192)
    parser.add_argument("--N", type=int, default=None)
    parser.add_argument("--K", type=int, default=None)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--warmup_iters", type=int, default=5)
    parser.add_argument("--autotune", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--mode",
                        type=str,
                        default="all",
                        choices=["all", "nonfused", "fused", "fuse_overlap", "torch"])
    parser.add_argument("--profile", action="store_true", default=False)
    parser.add_argument("--profile_target",
                        type=str,
                        default="all",
                        choices=["all", "nonfused", "fused", "fuse_overlap", "torch"])
    parser.add_argument("--dump_csv", action="store_true", default=False)
    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    parser.add_argument("--trans_b", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--persistent",
                        action=argparse.BooleanOptionalAction,
                        default=torch.cuda.get_device_capability() >= (9, 0))
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


def choose_gemm_config(persistent: bool):
    return get_config_space(persistent)[0]


def choose_non_persistent_gemm_config():
    return get_config_space(False)[0]


def get_autotuned_gemm_rs_config(A: torch.Tensor,
                                 B: torch.Tensor,
                                 ctx,
                                 persistent: bool,
                                 fuse_scatter: bool,
                                 pg: torch.distributed.ProcessGroup):
    base_key = gemm_rs.key_fn(A, B, ctx)
    cache_key = (base_key, persistent, fuse_scatter)
    best_config = TRITON_AUTOTUNE_CACHE.get(cache_key)
    if best_config is None:
        config_space = gemm_rs.get_pruned_config(A, B, ctx, persistent=persistent, fuse_scatter=fuse_scatter)
        timings = gemm_rs.tune(config_space, pg, A, B, ctx, persistent=persistent, fuse_scatter=fuse_scatter)
        timings.sort(key=lambda x: x[0])
        assert len(timings) > 0, "gemm_rs autotune returned empty timing list"
        best_config = timings[0][1]
        TRITON_AUTOTUNE_CACHE[cache_key] = best_config
    return best_config["gemm_config"]


def get_autotuned_fuse_overlap_config(A: torch.Tensor, B: torch.Tensor, ctx, pg: torch.distributed.ProcessGroup):
    base_key = fuse_overlap_gemm_rs.key_fn(A, B, ctx)
    best_config = FUSE_OVERLAP_AUTOTUNE_CACHE.get(base_key)
    if best_config is None:
        config_space = fuse_overlap_gemm_rs.get_pruned_config(A, B, ctx)
        timings = fuse_overlap_gemm_rs.tune(config_space, pg, A, B, ctx)
        timings.sort(key=lambda x: x[0])
        assert len(timings) > 0, "fuse_overlap autotune returned empty timing list"
        best_config = timings[0][1]
        FUSE_OVERLAP_AUTOTUNE_CACHE[base_key] = best_config
    return best_config["gemm_config"]


def launch_triton_gemm_producer(A: torch.Tensor,
                                B: torch.Tensor,
                                ctx,
                                workspace: torch.Tensor,
                                persistent: bool,
                                fuse_scatter: bool,
                                gemm_config):
    gemm_out = ctx.get_gemm_out_buf(A)
    scatter_signal = ctx.rs_ctx.scatter_signal_buf
    workspace.zero_()
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
            fuse_scatter,
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
            fuse_scatter,
            gemm_config,
        )
    return gemm_out


def launch_fuse_overlap_gemm_producer(A: torch.Tensor, B: torch.Tensor, ctx, workspace: torch.Tensor, gemm_config):
    gemm_out = ctx.get_gemm_out_buf(A)
    workspace.zero_()
    ctx.rs_ctx.reset_runtime_state()
    fuse_overlap_gemm_rs_producer_non_persistent(
        A,
        B,
        gemm_out,
        ctx.rs_ctx.ready_buf,
        workspace,
        ctx.rs_ctx.world_size,
        ctx.rs_ctx.local_world_size,
        gemm_config,
    )
    return gemm_out


def build_dummy_fused_layout(torch_partial: torch.Tensor, local_rank: int, local_world_size: int) -> torch.Tensor:
    M, N = torch_partial.shape
    M_per_rank = M // local_world_size
    shard = torch_partial[local_rank * M_per_rank:(local_rank + 1) * M_per_rank].contiguous()
    fused_layout = torch.empty((M, N), dtype=torch_partial.dtype, device=torch_partial.device)
    for src_local_rank in range(local_world_size):
        fused_layout[src_local_rank * M_per_rank:(src_local_rank + 1) * M_per_rank].copy_(shard)
    return fused_layout


def mark_all_fuse_overlap_ready(ctx) -> None:
    ctx.rs_ctx.ready_buf.fill_(1)


def perf_test(model_name: str, M: int, config: Dict[str, int], pg: torch.distributed.ProcessGroup):
    N = config["N"]
    K = config["K"]
    rank = pg.rank()
    world_size = pg.size()
    local_world_size = LOCAL_WORLD_SIZE

    run_nonfused = args.mode in ["all", "nonfused"]
    run_fused = args.mode in ["all", "fused"]
    run_fuse_overlap = args.mode in ["all", "fuse_overlap"]

    if (run_fused or run_fuse_overlap) and world_size != local_world_size:
        raise AssertionError("fused/fuse_overlap GEMM-RS benchmarks currently only support single-node runs")
    if run_fuse_overlap and args.persistent:
        raise AssertionError("fuse_overlap GEMM-RS currently only supports non-persistent producer path")

    if rank == 0:
        print(f"[{model_name}] test shape: M {M}, N {N}, K {K}")

    assert M % world_size == 0
    assert K % world_size == 0

    A, B = make_data(M, N, K, dtype, args.trans_b, pg)
    M_per_rank = M // world_size
    gemm_config = choose_gemm_config(args.persistent)
    fuse_overlap_gemm_config = choose_non_persistent_gemm_config()

    atol = 6e-2 if dtype == torch.bfloat16 else 1e-2
    rtol = atol
    torch_partial = torch.matmul(A, B)
    dummy_fused_layout = build_dummy_fused_layout(torch_partial, rank % local_world_size, local_world_size)

    def _torch_total():
        return torch_gemm_rs(pg, A, B)

    def _torch_gemm_only():
        return torch.matmul(A, B)

    def _torch_rs_only():
        output = torch.empty((M_per_rank, N), dtype=dtype, device=A.device)
        torch.distributed.reduce_scatter_tensor(output, torch_partial, group=pg)
        return output

    metrics = {
        "triton_nonfused_total_ms": float("nan"),
        "triton_nonfused_gemm_only_ms": float("nan"),
        "triton_nonfused_rs_only_ms": float("nan"),
        "triton_fused_total_ms": float("nan"),
        "triton_fused_producer_only_ms": float("nan"),
        "triton_fused_reduce_only_ms": float("nan"),
        "fuse_overlap_total_ms": float("nan"),
        "fuse_overlap_gemm_only_ms": float("nan"),
        "fuse_overlap_reduce_only_ms": float("nan"),
        "torch_total_ms": float("nan"),
        "torch_gemm_only_ms": float("nan"),
        "torch_rs_only_ms": float("nan"),
    }

    sync_all(pg)
    C_torch = _torch_total()

    def _run_triton_stage():
        ctx = None
        try:
            rs_stream = torch.cuda.Stream(priority=-1)
            ctx = create_gemm_rs_context(M, N, rank, world_size, local_world_size, dtype, rs_stream)
            workspace_nonfused = torch.zeros((world_size,), dtype=torch.int32, device=A.device)
            workspace_fused = torch.zeros((world_size,), dtype=torch.int32, device=A.device)
            triton_nonfused_rs_output = torch.empty((M_per_rank, N), dtype=dtype, device=A.device)
            triton_fused_reduce_output = torch.empty((M_per_rank, N), dtype=dtype, device=A.device)
            gemm_out = ctx.get_gemm_out_buf(A)

            def _get_nonfused_config():
                if args.autotune:
                    return get_autotuned_gemm_rs_config(A, B, ctx, args.persistent, False, pg)
                return gemm_config

            def _get_fused_config():
                if args.autotune:
                    return get_autotuned_gemm_rs_config(A, B, ctx, args.persistent, True, pg)
                return gemm_config

            def _triton_nonfused_total():
                return gemm_rs.fn(
                    A,
                    B,
                    ctx,
                    gemm_config=_get_nonfused_config(),
                    persistent=args.persistent,
                    fuse_scatter=False,
                )

            def _triton_fused_total():
                return gemm_rs.fn(
                    A,
                    B,
                    ctx,
                    gemm_config=_get_fused_config(),
                    persistent=args.persistent,
                    fuse_scatter=True,
                )

            def _triton_nonfused_gemm_only():
                return launch_triton_gemm_producer(
                    A, B, ctx, workspace_nonfused, args.persistent, False, _get_nonfused_config())

            def _triton_nonfused_rs_only():
                gemm_out.copy_(torch_partial)
                ctx.rs_ctx.scatter_signal_buf.fill_(1)
                return reduce_scatter_2d_op(gemm_out, ctx.rs_ctx, output=triton_nonfused_rs_output)

            def _prepare_fused_layout():
                sync_all(pg)
                launch_triton_gemm_producer(A, B, ctx, workspace_fused, args.persistent, True, _get_fused_config())
                sync_all(pg)

            def _triton_fused_producer_only():
                return launch_triton_gemm_producer(A, B, ctx, workspace_fused, args.persistent, True, _get_fused_config())

            def _triton_fused_reduce_only():
                gemm_out.copy_(dummy_fused_layout)
                return ring_reduce(gemm_out, triton_fused_reduce_output, ctx.rs_ctx.local_rank, ctx.rs_ctx.local_world_size)

            C_nonfused = None
            C_fused = None
            for _ in range(3):
                if run_nonfused:
                    sync_all(pg)
                    C_nonfused = _triton_nonfused_total()
                if run_fused:
                    sync_all(pg)
                    C_fused = _triton_fused_total()

            for i in range(world_size):
                torch.distributed.barrier(pg)
                if rank == i:
                    if C_nonfused is not None:
                        assert_allclose(C_torch, C_nonfused, atol=atol, rtol=rtol)
                    if C_fused is not None:
                        assert_allclose(C_torch, C_fused, atol=atol, rtol=rtol)

            if args.profile_target in ["all", "nonfused"] and run_nonfused:
                sync_all(pg)
                profile_section("triton/nonfused_total", True, _triton_nonfused_total, args.iters, args.warmup_iters)
                sync_all(pg)
                profile_section("triton/nonfused_gemm_only", True, _triton_nonfused_gemm_only, args.iters,
                                args.warmup_iters)
                sync_all(pg)
                profile_section("triton/nonfused_rs_only", True, _triton_nonfused_rs_only, args.iters,
                                args.warmup_iters)

            if args.profile_target in ["all", "fused"] and run_fused:
                sync_all(pg)
                profile_section("triton/fused_total", True, _triton_fused_total, args.iters, args.warmup_iters)
                sync_all(pg)
                profile_section("triton/fused_producer_only", True, _triton_fused_producer_only, args.iters,
                                args.warmup_iters)
                _prepare_fused_layout()
                profile_section("triton/fused_reduce_only", True, _triton_fused_reduce_only, args.iters,
                                args.warmup_iters)

            if run_nonfused:
                sync_all(pg)
                wait_until_max_gpu_clock_or_warning(torch.cuda.current_device())
                _, metrics["triton_nonfused_total_ms"] = perf_func(_triton_nonfused_total,
                                                                   iters=args.iters,
                                                                   warmup_iters=args.warmup_iters)
                sync_all(pg)
                wait_until_max_gpu_clock_or_warning(torch.cuda.current_device())
                _, metrics["triton_nonfused_gemm_only_ms"] = perf_func(_triton_nonfused_gemm_only,
                                                                       iters=args.iters,
                                                                       warmup_iters=args.warmup_iters)
                sync_all(pg)
                wait_until_max_gpu_clock_or_warning(torch.cuda.current_device())
                _, metrics["triton_nonfused_rs_only_ms"] = perf_func(_triton_nonfused_rs_only,
                                                                     iters=args.iters,
                                                                     warmup_iters=args.warmup_iters)

            if run_fused:
                sync_all(pg)
                wait_until_max_gpu_clock_or_warning(torch.cuda.current_device())
                _, metrics["triton_fused_total_ms"] = perf_func(_triton_fused_total,
                                                                iters=args.iters,
                                                                warmup_iters=args.warmup_iters)
                sync_all(pg)
                wait_until_max_gpu_clock_or_warning(torch.cuda.current_device())
                _, metrics["triton_fused_producer_only_ms"] = perf_func(_triton_fused_producer_only,
                                                                        iters=args.iters,
                                                                        warmup_iters=args.warmup_iters)
                _prepare_fused_layout()
                wait_until_max_gpu_clock_or_warning(torch.cuda.current_device())
                _, metrics["triton_fused_reduce_only_ms"] = perf_func(_triton_fused_reduce_only,
                                                                      iters=args.iters,
                                                                      warmup_iters=args.warmup_iters)
        finally:
            if ctx is not None:
                sync_all(pg)
                ctx.finalize()
                sync_all(pg)

    def _run_fuse_overlap_stage():
        ctx = None
        try:
            rs_stream = torch.cuda.Stream(priority=-1)
            ctx = create_fuse_overlap_gemm_rs_context(M, N, rank, world_size, local_world_size, dtype, rs_stream)
            workspace_overlap = torch.zeros((local_world_size,), dtype=torch.int32, device=A.device)
            fuse_overlap_reduce_output = torch.empty((M_per_rank, N), dtype=dtype, device=A.device)

            def _get_overlap_config():
                if args.autotune:
                    return get_autotuned_fuse_overlap_config(A, B, ctx, pg)
                return fuse_overlap_gemm_config

            def _fuse_overlap_total():
                return fuse_overlap_gemm_rs.fn(A, B, ctx, gemm_config=_get_overlap_config())

            def _fuse_overlap_gemm_only():
                return launch_fuse_overlap_gemm_producer(A, B, ctx, workspace_overlap, _get_overlap_config())

            def _fuse_overlap_reduce_only():
                gemm_out = ctx.get_gemm_out_buf(A)
                gemm_out.copy_(dummy_fused_layout)
                ctx.rs_ctx.reset_runtime_state()
                mark_all_fuse_overlap_ready(ctx)
                return fuse_overlap_reduce_scatter_2d_op(gemm_out, ctx.rs_ctx, output=fuse_overlap_reduce_output)

            C_overlap = None
            for _ in range(3):
                sync_all(pg)
                C_overlap = _fuse_overlap_total()

            for i in range(world_size):
                torch.distributed.barrier(pg)
                if rank == i and C_overlap is not None:
                    assert_allclose(C_torch, C_overlap, atol=atol, rtol=rtol)

            if args.profile_target in ["all", "fuse_overlap"]:
                sync_all(pg)
                profile_section("triton/fuse_overlap_total", True, _fuse_overlap_total, args.iters, args.warmup_iters)
                sync_all(pg)
                profile_section("triton/fuse_overlap_gemm_only", True, _fuse_overlap_gemm_only, args.iters,
                                args.warmup_iters)
                sync_all(pg)
                profile_section("triton/fuse_overlap_reduce_only", True, _fuse_overlap_reduce_only, args.iters,
                                args.warmup_iters)

            sync_all(pg)
            wait_until_max_gpu_clock_or_warning(torch.cuda.current_device())
            _, metrics["fuse_overlap_total_ms"] = perf_func(_fuse_overlap_total,
                                                            iters=args.iters,
                                                            warmup_iters=args.warmup_iters)
            sync_all(pg)
            wait_until_max_gpu_clock_or_warning(torch.cuda.current_device())
            _, metrics["fuse_overlap_gemm_only_ms"] = perf_func(_fuse_overlap_gemm_only,
                                                                iters=args.iters,
                                                                warmup_iters=args.warmup_iters)
            sync_all(pg)
            wait_until_max_gpu_clock_or_warning(torch.cuda.current_device())
            _, metrics["fuse_overlap_reduce_only_ms"] = perf_func(_fuse_overlap_reduce_only,
                                                                  iters=args.iters,
                                                                  warmup_iters=args.warmup_iters)
        finally:
            if ctx is not None:
                sync_all(pg)
                ctx.finalize()
                sync_all(pg)

    run_id = os.environ.get("TORCHELASTIC_RUN_ID", "local")
    with group_profile(f"fuse_overlap_gemm_rs_perf_m_{M}_n_{N}_k_{K}_{run_id}", args.profile, group=TP_GROUP):
        if run_nonfused or run_fused:
            _run_triton_stage()

        if run_fuse_overlap:
            _run_fuse_overlap_stage()

        if args.profile_target in ["all", "torch"]:
            sync_all(pg)
            profile_section("torch/reference_total", True, _torch_total, args.iters, args.warmup_iters)
            sync_all(pg)
            profile_section("torch/reference_gemm_only", True, _torch_gemm_only, args.iters, args.warmup_iters)
            sync_all(pg)
            profile_section("torch/reference_rs_only", True, _torch_rs_only, args.iters, args.warmup_iters)

    sync_all(pg)
    wait_until_max_gpu_clock_or_warning(torch.cuda.current_device())
    _, metrics["torch_total_ms"] = perf_func(_torch_total, iters=args.iters, warmup_iters=args.warmup_iters)
    sync_all(pg)
    wait_until_max_gpu_clock_or_warning(torch.cuda.current_device())
    _, metrics["torch_gemm_only_ms"] = perf_func(_torch_gemm_only, iters=args.iters, warmup_iters=args.warmup_iters)
    sync_all(pg)
    wait_until_max_gpu_clock_or_warning(torch.cuda.current_device())
    _, metrics["torch_rs_only_ms"] = perf_func(_torch_rs_only, iters=args.iters, warmup_iters=args.warmup_iters)

    serial_torch_ms = metrics["torch_gemm_only_ms"] + metrics["torch_rs_only_ms"]
    metrics["triton_nonfused_overlap_ratio"] = (
        (serial_torch_ms - metrics["triton_nonfused_total_ms"]) / max(serial_torch_ms, 1e-6))
    metrics["triton_fused_overlap_ratio"] = (
        (serial_torch_ms - metrics["triton_fused_total_ms"]) / max(serial_torch_ms, 1e-6))
    metrics["fuse_overlap_speedup_vs_torch"] = metrics["torch_total_ms"] / metrics["fuse_overlap_total_ms"]
    metrics["nonfused_speedup_vs_torch"] = metrics["torch_total_ms"] / metrics["triton_nonfused_total_ms"]
    metrics["fused_speedup_vs_torch"] = metrics["torch_total_ms"] / metrics["triton_fused_total_ms"]
    metrics["fuse_overlap_speedup_vs_fused"] = metrics["triton_fused_total_ms"] / metrics["fuse_overlap_total_ms"]
    metrics["fuse_overlap_internal_overlap_ratio"] = 1.0 - metrics["fuse_overlap_total_ms"] / max(
        metrics["fuse_overlap_gemm_only_ms"] + metrics["fuse_overlap_reduce_only_ms"], 1e-6)

    flops = 2 * M * N * (K // world_size)
    reduce_scatter_gb = M * N * dtype.itemsize / 2**30 * (world_size - 1) / world_size
    metrics["triton_nonfused_tflops"] = flops / metrics["triton_nonfused_total_ms"] * 1e-9
    metrics["triton_fused_tflops"] = flops / metrics["triton_fused_total_ms"] * 1e-9
    metrics["fuse_overlap_tflops"] = flops / metrics["fuse_overlap_total_ms"] * 1e-9
    metrics["torch_gemm_tflops"] = flops / metrics["torch_gemm_only_ms"] * 1e-9
    metrics["torch_rs_gbps"] = reduce_scatter_gb / metrics["torch_rs_only_ms"] * 1e3
    metrics["triton_nonfused_rs_gbps"] = reduce_scatter_gb / metrics["triton_nonfused_rs_only_ms"] * 1e3
    metrics["triton_fused_reduce_gbps"] = reduce_scatter_gb / metrics["triton_fused_reduce_only_ms"] * 1e3
    metrics["fuse_overlap_reduce_gbps"] = reduce_scatter_gb / metrics["fuse_overlap_reduce_only_ms"] * 1e3

    msg = (f"Rank {rank} [{model_name}] latency (ms): torch_total={metrics['torch_total_ms']:.2f}, "
           f"torch_gemm_only={metrics['torch_gemm_only_ms']:.2f}, "
           f"torch_rs_only={metrics['torch_rs_only_ms']:.2f}")
    if run_nonfused:
        msg += (
            f", triton_nonfused_total={metrics['triton_nonfused_total_ms']:.2f}, "
            f"triton_nonfused_gemm_only={metrics['triton_nonfused_gemm_only_ms']:.2f}, "
            f"triton_nonfused_rs_only={metrics['triton_nonfused_rs_only_ms']:.2f}, "
            f"nonfused_speedup={metrics['nonfused_speedup_vs_torch']:.2f}, "
            f"nonfused_overlap_ratio={metrics['triton_nonfused_overlap_ratio']:.2%}"
        )
    if run_fused:
        msg += (
            f", triton_fused_total={metrics['triton_fused_total_ms']:.2f}, "
            f"triton_fused_producer_only={metrics['triton_fused_producer_only_ms']:.2f}, "
            f"triton_fused_reduce_only={metrics['triton_fused_reduce_only_ms']:.2f}, "
            f"fused_speedup={metrics['fused_speedup_vs_torch']:.2f}, "
            f"fused_overlap_ratio={metrics['triton_fused_overlap_ratio']:.2%}"
        )
    if run_fuse_overlap:
        msg += (
            f", fuse_overlap_total={metrics['fuse_overlap_total_ms']:.2f}, "
            f"fuse_overlap_gemm_only={metrics['fuse_overlap_gemm_only_ms']:.2f}, "
            f"fuse_overlap_reduce_only={metrics['fuse_overlap_reduce_only_ms']:.2f}, "
            f"fuse_overlap_internal_overlap={metrics['fuse_overlap_internal_overlap_ratio'] * 100:.2f}%, "
            f"fuse_overlap_speedup_vs_torch={metrics['fuse_overlap_speedup_vs_torch']:.2f}, "
            f"fuse_overlap_speedup_vs_fused={metrics['fuse_overlap_speedup_vs_fused']:.2f}"
        )
    dist_print(msg, need_sync=True, allowed_ranks=list(range(world_size)))
    return metrics


if __name__ == "__main__":
    args = parse_args()
    if torch.cuda.get_device_capability()[0] < 9 and args.persistent:
        raise AssertionError("persistent GEMM-RS is not supported on cuda capability < 9.0")

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]

    TP_GROUP = initialize_distributed()
    LOCAL_WORLD_SIZE = int(os.environ.get("LOCAL_WORLD_SIZE", TP_GROUP.size()))

    perf_rows = []
    test_configs = get_test_configs(args)
    for model_name, config in test_configs.items():
        metrics = perf_test(model_name, args.M, config, TP_GROUP)
        perf_rows.append((model_name, config, metrics))

    if args.dump_csv and TP_GROUP.rank() == 0:
        os.makedirs("csv", exist_ok=True)
        csv_file = Path("csv") / f"perf_fuse_overlap_gemm_rs_{TP_GROUP.size()}_ranks.csv"
        with open(csv_file, "w") as fout:
            print(
                ",".join([
                    "Model",
                    "M",
                    "N",
                    "K",
                    "torch_total_ms",
                    "triton_nonfused_total_ms",
                    "triton_fused_total_ms",
                    "fuse_overlap_total_ms",
                    "nonfused_speedup_vs_torch",
                    "fused_speedup_vs_torch",
                    "fuse_overlap_speedup_vs_torch",
                    "fuse_overlap_speedup_vs_fused",
                    "triton_nonfused_overlap_ratio",
                    "triton_fused_overlap_ratio",
                    "fuse_overlap_internal_overlap_ratio",
                ]),
                file=fout,
            )
            for model_name, config, metrics in perf_rows:
                print(
                    ",".join([
                        model_name,
                        str(args.M),
                        str(config["N"]),
                        str(config["K"]),
                        f"{metrics['torch_total_ms']:.4f}",
                        f"{metrics['triton_nonfused_total_ms']:.4f}",
                        f"{metrics['triton_fused_total_ms']:.4f}",
                        f"{metrics['fuse_overlap_total_ms']:.4f}",
                        f"{metrics['nonfused_speedup_vs_torch']:.4f}",
                        f"{metrics['fused_speedup_vs_torch']:.4f}",
                        f"{metrics['fuse_overlap_speedup_vs_torch']:.4f}",
                        f"{metrics['fuse_overlap_speedup_vs_fused']:.4f}",
                        f"{metrics['triton_nonfused_overlap_ratio']:.4f}",
                        f"{metrics['triton_fused_overlap_ratio']:.4f}",
                        f"{metrics['fuse_overlap_internal_overlap_ratio']:.4f}",
                    ]),
                    file=fout,
                    flush=True,
                )
        print(f"csv file is dumped into {csv_file}")

    finalize_distributed()
