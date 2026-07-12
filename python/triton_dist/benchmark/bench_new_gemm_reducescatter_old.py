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

import torch
import torch.distributed

from triton_dist.kernels.nvidia.gemm import get_config_space
from triton_dist.kernels.nvidia.gemm_reduce_scatter import create_gemm_rs_context, gemm_rs
from triton_dist.kernels.nvidia.new_gemm_reducescatter import create_new_gemm_rs_context, new_gemm_rs
from triton_dist.profiler_utils import group_profile, perf_func
from triton_dist.test.utils import assert_allclose
from triton_dist.utils import (dist_print, finalize_distributed, initialize_distributed, nvshmem_barrier_all_on_stream,
                               rand_tensor, wait_until_max_gpu_clock_or_warning)
# 你可以直接这样跑

# 基本对比：
# torchrun --nproc_per_node=4 python/triton_dist/benchmark/bench_new_gemm_reducescatter.py --M 8192 --N 28672 --K 8192 --profile
# 固定 chunk：
# torchrun --nproc_per_node=4 python/triton_dist/benchmark/bench_new_gemm_reducescatter.py --M 8192 --N 28672 --K 8192 --profile --chunk_rows 512
# 自动调优：
# torchrun --nproc_per_node=4 python/triton_dist/benchmark/bench_new_gemm_reducescatter.py --M 8192 --N 28672 --K 8192 --profile --autotune

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--M", type=int, default=8192)
    parser.add_argument("--N", type=int, required=True)
    parser.add_argument("--K", type=int, required=True)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--warmup_iters", type=int, default=5)
    parser.add_argument("--autotune", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--profile", action="store_true", default=False)
    parser.add_argument("--dump_csv", action="store_true", default=False)
    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    parser.add_argument("--trans_b", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--persistent",
                        action=argparse.BooleanOptionalAction,
                        default=torch.cuda.get_device_capability() >= (9, 0))
    parser.add_argument("--fuse_scatter", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--chunk_rows",
                        type=int,
                        default=0,
                        help="0 means auto-select for new_gemm_rs; >0 fixes per-rank chunk rows")
    parser.add_argument("--target_chunks_per_rank", type=int, default=4)
    parser.add_argument("--min_chunk_rows", type=int, default=256)
    return parser.parse_args()


def make_data(M, N, K, dtype: torch.dtype, trans_b: bool, tp_group: torch.distributed.ProcessGroup):
    rank = tp_group.rank()
    world_size = tp_group.size()
    assert K % world_size == 0
    K_per_rank = K // world_size
    scale = (rank + 1) * 0.01
    device = torch.cuda.current_device()
    A = rand_tensor([M, K_per_rank], dtype=dtype, device=device) * scale
    if trans_b:
        B = rand_tensor([N, K_per_rank], dtype=dtype, device=device).T * scale
    else:
        B = rand_tensor([K_per_rank, N], dtype=dtype, device=device) * scale
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


def choose_base_gemm_config(persistent: bool):
    return get_config_space(persistent)[0]


def choose_new_gemm_config():
    return get_config_space(False)[0]


def perf_test(M: int, N: int, K: int, pg: torch.distributed.ProcessGroup):
    rank = pg.rank()
    world_size = pg.size()
    local_world_size = LOCAL_WORLD_SIZE

    if rank == 0:
        print(f"test shape: M {M}, N {N}, K {K}")

    assert M % world_size == 0
    assert K % world_size == 0
    if args.fuse_scatter and world_size != local_world_size:
        raise AssertionError("base gemm_rs fused path currently only supports single-node runs")

    A, B = make_data(M, N, K, dtype, args.trans_b, pg)
    base_gemm_config = choose_base_gemm_config(args.persistent)
    new_gemm_config = choose_new_gemm_config()
    atol = 6e-2 if dtype == torch.bfloat16 else 1e-2
    rtol = atol

    base_ctx = None
    new_ctx = None
    try:
        base_rs_stream = torch.cuda.Stream(priority=-1)
        new_rs_stream = torch.cuda.Stream(priority=-1)
        base_ctx = create_gemm_rs_context(M, N, rank, world_size, local_world_size, dtype, base_rs_stream)
        new_ctx = create_new_gemm_rs_context(
            M,
            N,
            rank,
            world_size,
            local_world_size,
            dtype,
            new_rs_stream,
            chunk_rows=args.chunk_rows,
            target_chunks_per_rank=args.target_chunks_per_rank,
            min_chunk_rows=args.min_chunk_rows,
        )

        if rank == 0:
            print(
                "new_gemm_rs config: "
                f"chunk_rows={new_ctx.rs_ctx.chunk_rows}, "
                f"num_chunks={new_ctx.rs_ctx.num_chunks}"
            )

        def _torch_func():
            return torch_gemm_rs(pg, A, B)

        def _base_triton_func():
            if args.autotune:
                return gemm_rs(A, B, base_ctx, autotune=True, persistent=args.persistent, fuse_scatter=args.fuse_scatter)
            return gemm_rs.fn(
                A,
                B,
                base_ctx,
                gemm_config=base_gemm_config,
                persistent=args.persistent,
                fuse_scatter=args.fuse_scatter,
            )

        def _new_triton_func():
            if args.autotune:
                return new_gemm_rs(A, B, new_ctx, autotune=True)
            return new_gemm_rs.fn(A, B, new_ctx, gemm_config=new_gemm_config)

        for _ in range(3):
            sync_all(pg)
            C_base = _base_triton_func()
            sync_all(pg)
            C_new = _new_triton_func()

        C_torch = _torch_func()
        for i in range(world_size):
            torch.distributed.barrier(pg)
            if rank == i:
                assert_allclose(C_torch, C_base, atol=atol, rtol=rtol)
                assert_allclose(C_torch, C_new, atol=atol, rtol=rtol)

        with group_profile(f"new_gemm_rs_perf_m_{M}_n_{N}_k_{K}_{os.environ['TORCHELASTIC_RUN_ID']}", args.profile,
                           group=TP_GROUP):
            with torch.profiler.record_function("base/gemm_rs"):
                perf_func(_base_triton_func, iters=args.iters, warmup_iters=args.warmup_iters)
            with torch.profiler.record_function("new/new_gemm_rs"):
                perf_func(_new_triton_func, iters=args.iters, warmup_iters=args.warmup_iters)
            with torch.profiler.record_function("torch/serial_gemm_reduce_scatter"):
                perf_func(_torch_func, iters=args.iters, warmup_iters=args.warmup_iters)

        wait_until_max_gpu_clock_or_warning(torch.cuda.current_device())
        _, base_ms = perf_func(_base_triton_func, iters=args.iters, warmup_iters=args.warmup_iters)
        wait_until_max_gpu_clock_or_warning(torch.cuda.current_device())
        _, new_ms = perf_func(_new_triton_func, iters=args.iters, warmup_iters=args.warmup_iters)
    finally:
        if new_ctx is not None:
            new_ctx.finalize()
        if base_ctx is not None:
            base_ctx.finalize()

    wait_until_max_gpu_clock_or_warning(torch.cuda.current_device())
    _, torch_ms = perf_func(_torch_func, iters=args.iters, warmup_iters=args.warmup_iters)

    dist_print(
        f"Rank {rank} latency (ms): "
        f"base_triton total={base_ms:.2f}, new_triton total={new_ms:.2f}, "
        f"torch total={torch_ms:.2f}, new_speedup {torch_ms / new_ms:.2f}, "
        f"new_vs_base {base_ms / new_ms:.2f}",
        need_sync=True,
        allowed_ranks=list(range(world_size)),
    )
    return base_ms, new_ms, torch_ms


if __name__ == "__main__":
    args = parse_args()
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    TP_GROUP = initialize_distributed()
    LOCAL_WORLD_SIZE = int(os.environ.get("LOCAL_WORLD_SIZE", TP_GROUP.size()))

    base_ms, new_ms, torch_ms = perf_test(args.M, args.N, args.K, TP_GROUP)

    if args.dump_csv and TP_GROUP.rank() == 0:
        os.makedirs("csv", exist_ok=True)
        csv_path = os.path.join("csv", f"perf_new_gemm_rs_{TP_GROUP.size()}_ranks.csv")
        with open(csv_path, "w", encoding="utf-8") as fout:
            print(
                "Model,M,N,K,base_triton_ms,new_triton_ms,torch_ms,new_speedup,new_vs_base,chunk_rows,target_chunks_per_rank,min_chunk_rows",
                file=fout,
            )
            print(
                ",".join(
                    map(
                        str,
                        [
                            "custom",
                            args.M,
                            args.N,
                            args.K,
                            f"{base_ms:.4f}",
                            f"{new_ms:.4f}",
                            f"{torch_ms:.4f}",
                            f"{torch_ms / new_ms:.4f}",
                            f"{base_ms / new_ms:.4f}",
                            args.chunk_rows,
                            args.target_chunks_per_rank,
                            args.min_chunk_rows,
                        ],
                    )),
                file=fout,
            )
        print(f"csv file is dumped into {csv_path}")

    finalize_distributed()
