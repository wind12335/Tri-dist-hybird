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
import time
from pathlib import Path

import torch
import torch.distributed

from triton_dist.kernels.nvidia import ag_gemm, create_ag_gemm_context
from triton_dist.kernels.nvidia.new_allgather_gemm_hurastic import create_new_ag_gemm_context, new_ag_gemm
from triton_dist.profiler_utils import group_profile, perf_func
from triton_dist.test.utils import assert_allclose
from triton_dist.utils import (
    dist_print,
    finalize_distributed,
    initialize_distributed,
    nvshmem_barrier_all_on_stream,
    rand_tensor,
    wait_until_max_gpu_clock_or_warning,
)

# 默认 heuristic：
# torchrun --nproc_per_node=4 python/triton_dist/benchmark/bench_new_allgather_gemm_8rank.py --M 8192 --N 28672 --K 8192 --profile

# 固定 1024 行 super-chunk：
# torchrun --nproc_per_node=4 python/triton_dist/benchmark/bench_new_allgather_gemm_8rank.py --M 8192 --N 28672 --K 8192 --profile --tile_rows_per_chunk 1024
# 强制退回 rank-ready 做对照：
# torchrun --nproc_per_node=4 python/triton_dist/benchmark/bench_new_allgather_gemm.py --M 8192 --N 28672 --K 8192 --profile --no-enable_tile_ready


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
    parser.add_argument(
        "--atol",
        type=float,
        default=None,
        help="Correctness-check absolute tolerance; defaults to a dtype-specific value.",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=None,
        help="Correctness-check relative tolerance; defaults to a dtype-specific value.",
    )
    parser.add_argument("--trans_b", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--cooperative_copy", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--copy_sms", type=int, default=0, help="<=0 means auto(about 1/4 SMs for copy)")
    parser.add_argument("--enable_tile_ready", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument(
        "--tile_rows_per_chunk",
        type=int,
        default=0,
        help=">0 uses a fixed super-chunk row size; 0 enables heuristic selection",
    )
    parser.add_argument(
        "--min_m_per_rank_for_tile_ready",
        type=int,
        default=4096,
        help="Disable chunk-ready when M_per_rank is smaller than this threshold",
    )
    parser.add_argument(
        "--target_chunks_per_rank",
        type=int,
        default=2,
        help="Heuristic target number of ready chunks per rank when tile_rows_per_chunk=0",
    )
    parser.add_argument(
        "--min_tile_rows_per_chunk",
        type=int,
        default=1024,
        help="Minimum super-chunk rows used by the heuristic",
    )
    return parser.parse_args()


def torch_ag_gemm(
    pg: torch.distributed.ProcessGroup,
    A: torch.Tensor,
    B: torch.Tensor,
):
    M_per_rank, K = A.shape
    A_full = torch.empty([M_per_rank * pg.size(), K], dtype=A.dtype, device=A.device)
    torch.distributed.all_gather_into_tensor(A_full, A, pg)
    return torch.matmul(A_full, B)


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


def resolve_check_tolerances(data_type: torch.dtype) -> tuple[float, float]:
    # At the observed output magnitude near 0.5, one BF16 ULP is 0.00390625.
    default = 4e-3 if data_type == torch.bfloat16 else 1e-3
    atol = default if args.atol is None else args.atol
    rtol = default if args.rtol is None else args.rtol
    if atol < 0 or rtol < 0:
        raise ValueError(f"--atol and --rtol must be non-negative, got {atol}, {rtol}")
    return atol, rtol


def perf_test(M: int, N: int, K: int, pg: torch.distributed.ProcessGroup):
    rank = pg.rank()
    world_size = pg.size()
    base_ctx = None
    new_ctx = None
    check_atol, check_rtol = resolve_check_tolerances(dtype)

    if rank == 0:
        print(f"test shape: M {M}, N {N}, K {K}")
        print(f"correctness tolerance: atol={check_atol:g}, rtol={check_rtol:g}")

    assert M % world_size == 0
    assert N % world_size == 0

    A, B = make_data(M, N, K, dtype, args.trans_b, pg)
    metrics = {
        "enable_row_tile_barrier": 0,
        "tile_rows_per_chunk": 0,
        "num_tile_chunks": 0,
        "tile_barrier_present": 0,
        "first_ready_ts_ms": float("nan"),
        "first_consumer_ts_ms": float("nan"),
        "last_completion_ts_ms": float("nan"),
        "consumer_wait_ms": float("nan"),
        "consumer_ts_is_proxy": 1,
        "check_atol": check_atol,
        "check_rtol": check_rtol,
    }

    def _torch_func():
        return torch_ag_gemm(pg, A, B)

    def _sync_all():
        nvshmem_barrier_all_on_stream(torch.cuda.current_stream())
        torch.cuda.synchronize()
        torch.distributed.barrier(pg)

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

    def _record_ag_timing_evidence() -> None:
        if not metrics["enable_row_tile_barrier"]:
            return
        tile_barrier = getattr(new_ctx, "symm_tile_barrier", None)
        if tile_barrier is None:
            return

        metrics["tile_barrier_present"] = 1
        tile_barrier.zero_()
        _sync_all()

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        wall_t0 = time.perf_counter()
        _new_triton_func()
        end_event.record()

        observed_ready = False
        while not end_event.query():
            ready_count = int((tile_barrier != 0).sum().item())
            if ready_count > 0 and not observed_ready:
                observed_ready = True
                ready_ts_ms = (time.perf_counter() - wall_t0) * 1000.0
                metrics["first_ready_ts_ms"] = ready_ts_ms
                # The benchmark layer cannot observe the first consumer tile launch
                # directly without kernel hooks. Use the first ready event as a
                # conservative consumer-activation proxy for Figure 3-2 timelines.
                metrics["first_consumer_ts_ms"] = ready_ts_ms
                metrics["consumer_wait_ms"] = 0.0
            torch.cuda._sleep(10000)

        end_event.synchronize()
        metrics["last_completion_ts_ms"] = start_event.elapsed_time(end_event)
        if not observed_ready:
            metrics["first_ready_ts_ms"] = metrics["last_completion_ts_ms"]
            metrics["first_consumer_ts_ms"] = metrics["last_completion_ts_ms"]
            metrics["consumer_wait_ms"] = 0.0

    try:
        base_ctx = create_ag_gemm_context(M, N, K, dtype, rank, world_size, LOCAL_WORLD_SIZE)
        new_ctx = create_new_ag_gemm_context(
            max_M=M,
            N=N,
            K=K,
            dtype=dtype,
            rank=rank,
            num_ranks=world_size,
            num_local_ranks=LOCAL_WORLD_SIZE,
            copy_sms=args.copy_sms,
            enable_row_tile_barrier=args.enable_tile_ready,
            tile_rows_per_chunk=args.tile_rows_per_chunk,
            min_m_per_rank_for_tile_ready=args.min_m_per_rank_for_tile_ready,
            target_chunks_per_rank=args.target_chunks_per_rank,
            min_tile_rows_per_chunk=args.min_tile_rows_per_chunk,
        )

        metrics["enable_row_tile_barrier"] = int(bool(getattr(new_ctx, "enable_row_tile_barrier", False)))
        metrics["tile_rows_per_chunk"] = int(getattr(new_ctx, "tile_rows_per_chunk", 0))
        metrics["num_tile_chunks"] = int(getattr(new_ctx, "num_tile_chunks", 0))
        metrics["tile_barrier_present"] = int(getattr(new_ctx, "symm_tile_barrier", None) is not None)

        if rank == 0:
            print(
                "tile-ready config: "
                f"enabled={bool(metrics['enable_row_tile_barrier'])}, "
                f"tile_rows_per_chunk={metrics['tile_rows_per_chunk']}, "
                f"num_tile_chunks={metrics['num_tile_chunks']}"
            )

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
                assert_allclose(C_golden, C_base, atol=check_atol, rtol=check_rtol)
                assert_allclose(C_golden, C_new, atol=check_atol, rtol=check_rtol)

        _record_ag_timing_evidence()

        with group_profile(
            f"new_ag_gemm_perf_m_{M}_n_{N}_k_{K}_{os.environ.get('TORCHELASTIC_RUN_ID', 'local')}",
            args.profile,
            group=TP_GROUP,
        ):
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

    metrics.update(
        {
            "base_triton_duration_ms": base_triton_duration_ms,
            "new_triton_duration_ms": new_triton_duration_ms,
            "torch_duration_ms": torch_duration_ms,
            "new_speedup": torch_duration_ms / new_triton_duration_ms,
            "new_vs_base": base_triton_duration_ms / new_triton_duration_ms,
        }
    )

    dist_print(
        f"Rank {rank} latency (ms): "
        f"base_triton total={metrics['base_triton_duration_ms']:.2f}, "
        f"new_triton total={metrics['new_triton_duration_ms']:.2f}, "
        f"torch total={metrics['torch_duration_ms']:.2f}, "
        f"new_speedup {metrics['new_speedup']:.2f}, "
        f"new_vs_base {metrics['new_vs_base']:.2f}, "
        f"first_ready_ts={metrics['first_ready_ts_ms']:.2f}, "
        f"first_consumer_ts={metrics['first_consumer_ts_ms']:.2f}, "
        f"last_completion_ts={metrics['last_completion_ts_ms']:.2f}, "
        f"consumer_wait_ms={metrics['consumer_wait_ms']:.2f}",
        need_sync=True,
        allowed_ranks=list(range(world_size)),
    )

    return metrics


if __name__ == "__main__":
    args = parse_args()

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    TP_GROUP = initialize_distributed()
    LOCAL_WORLD_SIZE = int(os.environ.get("LOCAL_WORLD_SIZE", TP_GROUP.size()))
    metrics = perf_test(args.M, args.N, args.K, TP_GROUP)

    if args.dump_csv and TP_GROUP.rank() == 0:
        if not os.path.exists("csv"):
            os.makedirs("csv")
        csv_file = Path("csv") / f"perf_new_ag_gemm_{TP_GROUP.size()}_ranks.csv"

        with open(csv_file, "w", encoding="utf-8") as fout:
            print(
                ",".join(
                    [
                        "Model",
                        "M",
                        "N",
                        "K",
                        "check_atol",
                        "check_rtol",
                        "dist-triton ag gemm latency (ms)",
                        "new dist-triton ag gemm latency (ms)",
                        "torch ag gemm latency (ms)",
                        "new speed up",
                        "new vs base",
                        "enable_row_tile_barrier",
                        "tile_rows_per_chunk",
                        "num_tile_chunks",
                        "tile_barrier_present",
                        "first_ready_ts_ms",
                        "first_consumer_ts_ms",
                        "last_completion_ts_ms",
                        "consumer_wait_ms",
                        "consumer_ts_is_proxy",
                    ]
                ),
                file=fout,
            )
            print(
                ",".join(
                    [
                        "custom",
                        str(args.M),
                        str(args.N),
                        str(args.K),
                        f"{metrics['check_atol']:.9g}",
                        f"{metrics['check_rtol']:.9g}",
                    ]
                    + [
                        f"{metrics['base_triton_duration_ms']:.6f}",
                        f"{metrics['new_triton_duration_ms']:.6f}",
                        f"{metrics['torch_duration_ms']:.6f}",
                        f"{metrics['new_speedup']:.6f}",
                        f"{metrics['new_vs_base']:.6f}",
                        str(metrics["enable_row_tile_barrier"]),
                        str(metrics["tile_rows_per_chunk"]),
                        str(metrics["num_tile_chunks"]),
                        str(metrics["tile_barrier_present"]),
                        f"{metrics['first_ready_ts_ms']:.6f}",
                        f"{metrics['first_consumer_ts_ms']:.6f}",
                        f"{metrics['last_completion_ts_ms']:.6f}",
                        f"{metrics['consumer_wait_ms']:.6f}",
                        str(metrics["consumer_ts_is_proxy"]),
                    ]
                ),
                file=fout,
                flush=True,
            )
        print(f"csv file is dumped into {csv_file}")

    finalize_distributed()
