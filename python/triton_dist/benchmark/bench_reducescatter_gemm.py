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

import torch
import torch.distributed

from triton_dist.kernels.nvidia import create_gemm_rs_context, gemm_rs
from triton_dist.kernels.nvidia.reduce_scatter import reduce_scatter_2d_op
from triton_dist.profiler_utils import group_profile, perf_func
from triton_dist.test.utils import LAYER_CONFIGS, assert_allclose
from triton_dist.utils import (dist_print, finalize_distributed, initialize_distributed, nvshmem_barrier_all_on_stream,
                               rand_tensor, wait_until_max_gpu_clock_or_warning)
"codex生成 "
# 1.source ./scripts/setenv.sh
# 2.export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
# torchrun --standalone --nproc_per_node=2 python/triton_dist/benchmark/bench_reducescatter_gemm2.py --M 8192 --N 14336 --K 4096 --dtype bfloat16 --iters 20 --warmup_iters 10 --autotune --dump_csv
# 最常用 torchrun --nproc_per_node=2 python/triton_dist/benchmark/bench_reducescatter_gemm2.py --M 8192 --N 14336 --K 4096 --dtype bfloat16 --iters 10 --warmup_iters 5
# torchrun --nproc_per_node=2 python/triton_dist/benchmark/bench_reducescatter_gemm2_2.py --dtype bfloat16 --iters 10 --warmup_iters 5
# overlap_gain_ms = torch_gemm_ms + torch_rs_ms - triton_total_ms
# overlap_ratio = overlap_gain_ms / (torch_gemm_ms + torch_rs_ms)
 

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--M", type=int, default=8192, help="Sequence dimension before reduce-scatter.")
    parser.add_argument("--N", type=int, default=None, help="Optional custom N. If unset, use LAYER_CONFIGS.")
    parser.add_argument("--K", type=int, default=None, help="Optional custom K. If unset, use LAYER_CONFIGS.")
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--warmup_iters", type=int, default=5)
    parser.add_argument("--autotune", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--profile", action="store_true", default=False)
    parser.add_argument("--dump_csv", action="store_true", default=False)
    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    parser.add_argument("--trans_b", default=True, action=argparse.BooleanOptionalAction)
    return parser.parse_args()


def get_test_configs(args):
    if args.N is not None or args.K is not None:
        if args.N is None or args.K is None:
            raise ValueError("`--N` and `--K` must be set together.")
        return {"custom": {"N": args.N, "K": args.K}}
    return LAYER_CONFIGS


def torch_gemm_rs(
    pg: torch.distributed.ProcessGroup,
    A: torch.Tensor,  # [M, K_per_rank]
    B: torch.Tensor,  # [K_per_rank, N]
):
    M, _ = A.shape
    _, N = B.shape
    gemm_output = torch.matmul(A, B)
    rs_output = torch.empty([M // pg.size(), N], dtype=gemm_output.dtype, device=gemm_output.device)
    torch.distributed.reduce_scatter_tensor(rs_output, gemm_output, group=pg)
    return rs_output


def make_data(M, N, K, dtype: torch.dtype, trans_b: bool, tp_group: torch.distributed.ProcessGroup):
    rank = tp_group.rank()
    num_ranks = tp_group.size()
    K_per_rank = K // num_ranks
    scale = (rank + 1) * 0.01

    current_device = torch.cuda.current_device()
    A = rand_tensor([M, K_per_rank], dtype=dtype, device=current_device) * scale
    if trans_b:
        B = rand_tensor([N, K_per_rank], dtype=dtype, device=current_device).T * scale
    else:
        B = rand_tensor([K_per_rank, N], dtype=dtype, device=current_device) * scale
    return A, B


def perf_test(M, config, pg: torch.distributed.ProcessGroup):
    N = config["N"]
    K = config["K"]
    rank = pg.rank()
    world_size = pg.size()

    if rank == 0:
        print(f"test shape: M {M}, N {N}, K {K}, K_per_rank {K // world_size}")

    assert M % world_size == 0
    assert K % world_size == 0

    A, B = make_data(M, N, K, dtype, args.trans_b, pg)
    rs_stream = torch.cuda.Stream(priority=-1)
    ctx = create_gemm_rs_context(M, N, rank, world_size, LOCAL_WORLD_SIZE, dtype, rs_stream)
    atol = 6e-2 if dtype == torch.bfloat16 else 1e-2
    rtol = atol

    def _torch_total_func():
        return torch_gemm_rs(pg, A, B)

    def _triton_total_func():
        return gemm_rs(A, B, ctx=ctx, autotune=args.autotune)

    gemm_out_for_rs = torch.matmul(A, B)
    torch_rs_output = torch.empty((M // world_size, N), dtype=A.dtype, device=A.device)
    triton_rs_output = torch.empty((M // world_size, N), dtype=A.dtype, device=A.device)

    def _torch_gemm_func():
        return torch.matmul(A, B)

    def _torch_rs_func():
        torch.distributed.reduce_scatter_tensor(torch_rs_output, gemm_out_for_rs, group=pg)
        return torch_rs_output

    def _triton_rs_func():
        current_stream = torch.cuda.current_stream()
        nvshmem_barrier_all_on_stream(current_stream)
        # Standalone RS path needs all scatter signals ready; gemm_rs kernel usually sets them.
        ctx.rs_ctx.scatter_signal_buf.fill_(1)
        return reduce_scatter_2d_op(gemm_out_for_rs, ctx.rs_ctx, triton_rs_output)

    try:
        for _ in range(3):
            nvshmem_barrier_all_on_stream(torch.cuda.current_stream())
            C = _triton_total_func()

        C_golden = _torch_total_func()
        for i in range(world_size):
            torch.distributed.barrier(pg)
            if rank == i:
                assert_allclose(C_golden, C, atol=atol, rtol=rtol)

        run_id = os.environ.get("TORCHELASTIC_RUN_ID", "local")
        with group_profile(f"gemm_rs_perf_m_{M}_n_{N}_k_{K}_{run_id}", args.profile, group=TP_GROUP):
            perf_func(_triton_total_func, iters=args.iters, warmup_iters=args.warmup_iters)
            perf_func(_torch_total_func, iters=args.iters, warmup_iters=args.warmup_iters)

        wait_until_max_gpu_clock_or_warning(torch.cuda.current_device())
        _, triton_total_ms = perf_func(_triton_total_func, iters=args.iters, warmup_iters=args.warmup_iters)
        wait_until_max_gpu_clock_or_warning(torch.cuda.current_device())
        _, triton_rs_ms = perf_func(_triton_rs_func, iters=args.iters, warmup_iters=args.warmup_iters)
        wait_until_max_gpu_clock_or_warning(torch.cuda.current_device())
        _, torch_total_ms = perf_func(_torch_total_func, iters=args.iters, warmup_iters=args.warmup_iters)
        wait_until_max_gpu_clock_or_warning(torch.cuda.current_device())
        _, torch_gemm_ms = perf_func(_torch_gemm_func, iters=args.iters, warmup_iters=args.warmup_iters)
        wait_until_max_gpu_clock_or_warning(torch.cuda.current_device())
        _, torch_rs_ms = perf_func(_torch_rs_func, iters=args.iters, warmup_iters=args.warmup_iters)

        overlap_gain_ms = torch_gemm_ms + torch_rs_ms - triton_total_ms
        overlap_ratio = overlap_gain_ms / max(torch_gemm_ms + torch_rs_ms, 1e-6)
        dist_print(
            f"Rank {rank} latency (ms): "
            f"triton total={triton_total_ms:.2f}, triton_rs_only={triton_rs_ms:.2f}, "
            f"torch total={torch_total_ms:.2f}, torch_gemm_only={torch_gemm_ms:.2f}, torch_rs_only={torch_rs_ms:.2f}, "
            f"speedup={torch_total_ms / triton_total_ms:.2f}, overlap_ratio={overlap_ratio:.2%}",
            need_sync=True, allowed_ranks=list(range(world_size)))

        return {
            "triton_total_ms": triton_total_ms,
            "triton_rs_ms": triton_rs_ms,
            "torch_total_ms": torch_total_ms,
            "torch_gemm_ms": torch_gemm_ms,
            "torch_rs_ms": torch_rs_ms,
            "speedup": torch_total_ms / triton_total_ms,
            "overlap_ratio": overlap_ratio,
        }
    finally:
        ctx.finalize()


if __name__ == "__main__":
    args = parse_args()
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    TP_GROUP = initialize_distributed()
    LOCAL_WORLD_SIZE = int(os.environ.get("LOCAL_WORLD_SIZE", TP_GROUP.size()))
    configs = get_test_configs(args)
    perf_res = []

    for model, config in configs.items():
        metrics = perf_test(args.M, config, TP_GROUP)
        perf_res.append((model, config, metrics))

    if args.dump_csv and TP_GROUP.rank() == 0:
        os.makedirs("csv", exist_ok=True)
        csv_file = Path("csv") / f"perf_gemm_rs2_{TP_GROUP.size()}_ranks.csv"
        with open(csv_file, "w") as fout:
            print(
                ",".join([
                    "Model",
                    "M",
                    "N",
                    "K",
                    "dist-triton gemm-rs latency (ms)",
                    "dist-triton rs-only latency (ms)",
                    "torch gemm-rs latency (ms)",
                    "torch gemm-only latency (ms)",
                    "torch rs-only latency (ms)",
                    "speedup",
                    "overlap_ratio",
                ]),
                file=fout,
            )
            for model, config, metrics in perf_res:
                print(
                    ",".join(
                        [model] + list(
                            map(
                                str,
                                [
                                    args.M,
                                    config["N"],
                                    config["K"],
                                ],
                            )) + list(
                                map(
                                    "{:0.4f}".format,
                                    [
                                        metrics["triton_total_ms"],
                                        metrics["triton_rs_ms"],
                                        metrics["torch_total_ms"],
                                        metrics["torch_gemm_ms"],
                                        metrics["torch_rs_ms"],
                                        metrics["speedup"],
                                        metrics["overlap_ratio"],
                                    ],
                                ))),
                    file=fout,
                    flush=True,
                )
        print(f"csv file is dumped into {csv_file}")

    finalize_distributed()
