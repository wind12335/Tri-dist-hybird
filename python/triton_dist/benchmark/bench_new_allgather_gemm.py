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

from triton_dist.profiler_utils import group_profile, perf_func
from triton_dist.test.utils import assert_allclose
from triton_dist.kernels.nvidia import (ag_gemm, create_ag_gemm_context, create_new_ag_gemm_context, new_ag_gemm)
from triton_dist.utils import (dist_print, finalize_distributed, initialize_distributed, nvshmem_barrier_all_on_stream,
                               wait_until_max_gpu_clock_or_warning, rand_tensor)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--M", type=int, default=8192)
    parser.add_argument("--N", type=int, required=True)
    parser.add_argument("--K", type=int, required=True)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--warmup_iters", type=int, default=5)
    parser.add_argument("--autotune", action="store_true", default=False)
    parser.add_argument("--profile", action="store_true", default=False)
    parser.add_argument("--dump_csv", action="store_true", default=False)
    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    parser.add_argument("--trans_b", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--cooperative_copy", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--copy_sms", type=int, default=0, help="<=0 means auto(about 1/4 SMs for copy)")
    args = parser.parse_args()
    return args


def torch_ag_gemm(
    pg: torch.distributed.ProcessGroup,
    A: torch.Tensor,
    B: torch.Tensor,
):
    M_per_rank, K = A.shape
    A_full = torch.empty([M_per_rank * pg.size(), K], dtype=A.dtype, device=A.device)
    torch.distributed.all_gather_into_tensor(A_full, A, pg)
    ag_gemm_output = torch.matmul(A_full, B)
    return ag_gemm_output


def make_data(M, N, K, dtype: torch.dtype, trans_b, tp_group: torch.distributed.ProcessGroup):
    rank = tp_group.rank()
    num_ranks = tp_group.size()
    M_per_rank = M // num_ranks
    N_per_rank = N // num_ranks
    scale = (rank + 1) * 0.01

    current_device = torch.cuda.current_device()
    A = rand_tensor([M_per_rank, K], dtype=dtype, device=current_device) * scale
    if trans_b:
        B = rand_tensor([N_per_rank, K], dtype=dtype, device=current_device).T * scale
    else:
        B = rand_tensor([K, N_per_rank], dtype=dtype, device=current_device) * scale

    return A, B


def perf_test(M: int, N: int, K: int, pg: torch.distributed.ProcessGroup):
    rank = pg.rank()
    world_size = pg.size()

    if rank == 0:
        print(f"test shape: M {M}, N {N}, K {K}")

    assert M % world_size == 0
    assert N % world_size == 0

    A, B = make_data(M, N, K, dtype, args.trans_b, pg)

    def _torch_func():
        return torch_ag_gemm(pg, A, B)

    def _sync_all():
        nvshmem_barrier_all_on_stream(torch.cuda.current_stream())
        torch.cuda.synchronize()
        torch.distributed.barrier(pg)

    base_ctx = None
    new_ctx = None
    try:
        # Create both contexts once for this single shape.
        base_ctx = create_ag_gemm_context(M, N, K, dtype, rank, world_size, LOCAL_WORLD_SIZE)
        new_ctx = create_new_ag_gemm_context(M, N, K, dtype, rank, world_size, LOCAL_WORLD_SIZE, copy_sms=args.copy_sms)

        def _base_triton_func():
            return ag_gemm(A, B, ctx=base_ctx, autotune=args.autotune, debug=args.debug)

        def _new_triton_func():
            return new_ag_gemm(
                A,
                B,
                ctx=new_ctx,
                autotune=args.autotune,
                debug=args.debug,
                use_cooperative=args.cooperative_copy,
            )

        # Warmup with fresh inputs.
        for _ in range(5):
            A, B = make_data(M, N, K, dtype, args.trans_b, pg)
            _sync_all()
            C_base = _base_triton_func()
            _sync_all()
            C_new = _new_triton_func()

        C_golden = _torch_func()
        for i in range(world_size):
            torch.distributed.barrier(pg)
            if rank == i:
                assert_allclose(C_golden, C_base, atol=1e-3, rtol=1e-3)
                assert_allclose(C_golden, C_new, atol=1e-3, rtol=1e-3)

        # IMPORTANT:
        # Use a single profiling context for base/new/torch.
        # If we open multiple `group_profile` blocks with the same name, the
        # second one will overwrite the merged trace and you may only see the
        # last section (often torch) in Perfetto.
        with group_profile(f"new_ag_gemm_perf_m_{M}_n_{N}_k_{K}_{os.environ['TORCHELASTIC_RUN_ID']}", args.profile,
                           group=TP_GROUP):
            with torch.profiler.record_function("base/ag_gemm"):
                perf_func(_base_triton_func, iters=args.iters, warmup_iters=args.warmup_iters)
            with torch.profiler.record_function("new/new_ag_gemm"):
                perf_func(_new_triton_func, iters=args.iters, warmup_iters=args.warmup_iters)
            with torch.profiler.record_function("torch/serial_allgather_gemm"):
                perf_func(_torch_func, iters=args.iters, warmup_iters=args.warmup_iters)

        wait_until_max_gpu_clock_or_warning(torch.cuda.current_device())
        _, base_triton_duration_ms = perf_func(_base_triton_func, iters=args.iters, warmup_iters=args.warmup_iters)
        wait_until_max_gpu_clock_or_warning(torch.cuda.current_device())
        _, new_triton_duration_ms = perf_func(_new_triton_func, iters=args.iters, warmup_iters=args.warmup_iters)
    finally:
        if new_ctx is not None:
            new_ctx.finalize()
        if base_ctx is not None:
            base_ctx.finalize()

    wait_until_max_gpu_clock_or_warning(torch.cuda.current_device())
    _, torch_duration_ms = perf_func(_torch_func, iters=args.iters, warmup_iters=args.warmup_iters)

    dist_print(
        f"Rank {rank} latency (ms): " \
        f"base_triton total={base_triton_duration_ms:.2f}, new_triton total={new_triton_duration_ms:.2f}, " \
        f"torch total={torch_duration_ms:.2f}, new_speedup {torch_duration_ms / new_triton_duration_ms:.2f}, " \
        f"new_vs_base {base_triton_duration_ms / new_triton_duration_ms:.2f}",
        need_sync=True,
        allowed_ranks=list(range(world_size)),
    )

    return base_triton_duration_ms, new_triton_duration_ms, torch_duration_ms


if __name__ == "__main__":
    args = parse_args()

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    TP_GROUP = initialize_distributed()
    LOCAL_WORLD_SIZE = int(os.environ.get("LOCAL_WORLD_SIZE", TP_GROUP.size()))
    base_ms, new_ms, torch_ms = perf_test(args.M, args.N, args.K, TP_GROUP)

    if args.dump_csv and TP_GROUP.rank() == 0:
        if not os.path.exists("csv"):
            os.makedirs("csv")
        csv_file = Path("csv") / f"perf_new_ag_gemm_{TP_GROUP.size()}_ranks.csv"

        with open(csv_file, "w") as fout:
            print(
                ",".join(
                    map(
                        str,
                        [
                            "Model",
                            "M",
                            "N",
                            "K",
                            "dist-triton ag gemm latency (ms)",
                            "new dist-triton ag gemm latency (ms)",
                            "torch ag gemm latency (ms)",
                            "new speed up",
                            "new vs base",
                        ],
                    )),
                file=fout,
            )
            print(
                ",".join(["custom"] + list(map(
                    "{:d}".format,
                    [
                        args.M,
                        args.N,
                        args.K,
                    ],
                )) + list(
                    map(
                        "{:02f}".format,
                        [
                            base_ms,
                            new_ms,
                            torch_ms,
                            torch_ms / new_ms,
                            base_ms / new_ms,
                        ],
                    ))),
                file=fout,
                flush=True,
            )
        print(f"csv file is dumped into {csv_file}")

    finalize_distributed()
