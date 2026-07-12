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

from triton_dist.kernels.nvidia import (ag_gemm, build_dynamic_k_schedule, chunk_ag_gemm, create_ag_gemm_context,
                                        create_chunk_ag_gemm_context)
from triton_dist.profiler_utils import group_profile, perf_func
from triton_dist.test.utils import LAYER_CONFIGS, assert_allclose
from triton_dist.utils import (dist_print, finalize_distributed, initialize_distributed, nvshmem_barrier_all_on_stream,
                               rand_tensor, wait_until_max_gpu_clock_or_warning)
# cd /data/coding/Triton-distributed
# 1.source ./scripts/setenv.sh
# 2.export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
#$env:PYTHONPATH="python"
# torchrun --standalone --nproc_per_node=2 python/triton_dist/benchmark/bench_reducescatter_gemm2.py --M 8192 --N 14336 --K 4096 --dtype bfloat16 --iters 20 --warmup_iters 10 --autotune --dump_csv
# torchrun --nproc_per_node=2 python/triton_dist/benchmark/bench_chunk_allgather_gemm.py --M 8192 --N 53248 --K 16384 --dtype bfloat16 --iters 10 --warmup_iters 5 --profile



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--M", type=int, default=8192)
    parser.add_argument("--N", type=int, default=None)
    parser.add_argument("--K", type=int, default=None)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--warmup_iters", type=int, default=5)
    parser.add_argument("--autotune", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--compare_base", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--profile", action="store_true", default=False)
    parser.add_argument("--dump_csv", action="store_true", default=False)
    parser.add_argument("--dump_schedule", action="store_true", default=False)
    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    parser.add_argument("--trans_b", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--k_alignment", type=int, default=64)
    parser.add_argument("--min_prefetch_k", type=int, default=128)
    parser.add_argument("--max_k_chunks", type=int, default=8)
    parser.add_argument("--overlap_target", type=float, default=0.85)
    parser.add_argument("--target_intra_bw_gbps", type=float, default=250.0)
    parser.add_argument("--target_gemm_tflops", type=float, default=120.0)
    parser.add_argument(
        "--reorder_policy",
        type=str,
        default="rank_swizzle",
        choices=["rank_swizzle", "in_order", "largest_first"],
    )
    parser.add_argument("--triton_chunk_compute", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--cooperative_copy", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--copy_sms", type=int, default=0, help="<=0 means auto(about 1/4 SMs for copy)")
    return parser.parse_args()


def get_test_configs(args):
    if args.N is not None or args.K is not None:
        if args.N is None or args.K is None:
            raise ValueError("`--N` and `--K` must be set together.")
        return {"custom": {"N": args.N, "K": args.K}}
    return LAYER_CONFIGS


def torch_ag_gemm(
    pg: torch.distributed.ProcessGroup,
    A: torch.Tensor,  # [M_per_rank, K]
    B: torch.Tensor,  # [K, N_per_rank]
):
    M_per_rank, K = A.shape
    A_full = torch.empty([M_per_rank * pg.size(), K], dtype=A.dtype, device=A.device)
    torch.distributed.all_gather_into_tensor(A_full, A, group=pg)
    return torch.matmul(A_full, B)


def make_data(M, N, K, dtype: torch.dtype, trans_b: bool, tp_group: torch.distributed.ProcessGroup):
    rank = tp_group.rank()
    world_size = tp_group.size()
    M_per_rank = M // world_size
    N_per_rank = N // world_size
    scale = (rank + 1) * 0.01

    device = torch.cuda.current_device()
    A = rand_tensor([M_per_rank, K], dtype=dtype, device=device) * scale
    if trans_b:
        B = rand_tensor([N_per_rank, K], dtype=dtype, device=device).T * scale
    else:
        B = rand_tensor([K, N_per_rank], dtype=dtype, device=device) * scale
    return A, B


def perf_test(M, config, pg: torch.distributed.ProcessGroup):
    N = config["N"]
    K = config["K"]
    rank = pg.rank()
    world_size = pg.size()

    if rank == 0:
        print(f"test shape: M {M}, N {N}, K {K}")

    assert M % world_size == 0
    assert N % world_size == 0

    A, B = make_data(M, N, K, dtype, args.trans_b, pg)
    A_gathered = torch.empty((M, K), dtype=A.dtype, device=A.device)
    torch.distributed.all_gather_into_tensor(A_gathered, A, group=pg)
    atol = 6e-2 if dtype == torch.bfloat16 else 1e-2
    rtol = atol

    schedule = build_dynamic_k_schedule(
        K=K,
        rank=rank,
        num_ranks=world_size,
        M_per_rank=M // world_size,
        N_per_rank=N // world_size,
        dtype=dtype,
        k_alignment=args.k_alignment,
        min_prefetch_k=args.min_prefetch_k,
        max_k_chunks=args.max_k_chunks,
        overlap_target=args.overlap_target,
        intra_node_bw_gbps=args.target_intra_bw_gbps,
        gemm_tflops=args.target_gemm_tflops,
        reorder_policy=args.reorder_policy,
    )
    if args.dump_schedule and rank == 0:
        chunk_shapes = [f"[{ks},{ke})" for ks, ke in schedule.ordered_chunks()]
        print(
            f"chunk schedule: prefetch={schedule.prefetch_chunk} order={chunk_shapes} "
            f"triton_chunk_compute={args.triton_chunk_compute} cooperative_copy={args.cooperative_copy} copy_sms={args.copy_sms}"
        )

    chunk_ctx = create_chunk_ag_gemm_context(
        max_M=M,
        N=N,
        K=K,
        dtype=dtype,
        rank=rank,
        num_ranks=world_size,
        num_local_ranks=LOCAL_WORLD_SIZE,
        k_alignment=args.k_alignment,
        min_prefetch_k=args.min_prefetch_k,
        max_k_chunks=args.max_k_chunks,
        overlap_target=args.overlap_target,
        target_intra_bw_gbps=args.target_intra_bw_gbps,
        target_gemm_tflops=args.target_gemm_tflops,
        reorder_policy=args.reorder_policy,
        copy_sms=args.copy_sms,
    )
    base_ctx = None
    if args.compare_base:
        base_ctx = create_ag_gemm_context(M, N, K, dtype, rank, world_size, LOCAL_WORLD_SIZE)

    def _chunk_func():
        return chunk_ag_gemm(
            A,
            B,
            ctx=chunk_ctx,
            schedule=schedule,
            debug=args.debug,
            use_cooperative_copy=args.cooperative_copy,
            use_triton_compute=args.triton_chunk_compute,
            autotune=args.autotune,
        )

    def _torch_func():
        return torch_ag_gemm(pg, A, B)

    def _torch_ag_func():
        torch.distributed.all_gather_into_tensor(A_gathered, A, group=pg)
        return A_gathered

    def _torch_gemm_func():
        return torch.matmul(A_gathered, B)

    def _base_func():
        return ag_gemm(A, B, ctx=base_ctx, autotune=args.autotune)

    try:
        for _ in range(3):
            nvshmem_barrier_all_on_stream(torch.cuda.current_stream())
            C_chunk = _chunk_func()

        C_golden = _torch_func()
        for i in range(world_size):
            torch.distributed.barrier(pg)
            if rank == i:
                assert_allclose(C_golden, C_chunk, atol=atol, rtol=rtol)

        if args.compare_base:
            C_base = _base_func()
            for i in range(world_size):
                torch.distributed.barrier(pg)
                if rank == i:
                    assert_allclose(C_golden, C_base, atol=atol, rtol=rtol)

        run_id = os.environ.get("TORCHELASTIC_RUN_ID", "local")
        with group_profile(f"chunk_ag_gemm_perf_m_{M}_n_{N}_k_{K}_{run_id}", args.profile, group=TP_GROUP):
            perf_func(_chunk_func, iters=args.iters, warmup_iters=args.warmup_iters)
            if args.compare_base:
                perf_func(_base_func, iters=args.iters, warmup_iters=args.warmup_iters)
            perf_func(_torch_func, iters=args.iters, warmup_iters=args.warmup_iters)

        wait_until_max_gpu_clock_or_warning(torch.cuda.current_device())
        _, chunk_total_ms = perf_func(_chunk_func, iters=args.iters, warmup_iters=args.warmup_iters)
        base_total_ms = float("nan")
        if args.compare_base:
            wait_until_max_gpu_clock_or_warning(torch.cuda.current_device())
            _, base_total_ms = perf_func(_base_func, iters=args.iters, warmup_iters=args.warmup_iters)
        wait_until_max_gpu_clock_or_warning(torch.cuda.current_device())
        _, torch_total_ms = perf_func(_torch_func, iters=args.iters, warmup_iters=args.warmup_iters)
        wait_until_max_gpu_clock_or_warning(torch.cuda.current_device())
        _, torch_ag_ms = perf_func(_torch_ag_func, iters=args.iters, warmup_iters=args.warmup_iters)
        wait_until_max_gpu_clock_or_warning(torch.cuda.current_device())
        _, torch_gemm_ms = perf_func(_torch_gemm_func, iters=args.iters, warmup_iters=args.warmup_iters)

        serial_torch_ms = torch_ag_ms + torch_gemm_ms
        chunk_overlap_ratio = (serial_torch_ms - chunk_total_ms) / max(serial_torch_ms, 1e-6)
        base_overlap_ratio = (serial_torch_ms - base_total_ms) / max(serial_torch_ms, 1e-6) if args.compare_base else float(
            "nan")
        msg = (
            f"Rank {rank} latency (ms): "
            f"chunk_total={chunk_total_ms:.2f}, "
            f"torch total={torch_total_ms:.2f}, torch_ag_only={torch_ag_ms:.2f}, torch_gemm_only={torch_gemm_ms:.2f}, "
            f"chunk_speedup={torch_total_ms / chunk_total_ms:.2f}, chunk_overlap_ratio={chunk_overlap_ratio:.2%}"
        )
        if args.compare_base:
            msg += (
                f", base_total={base_total_ms:.2f}, base_speedup={torch_total_ms / base_total_ms:.2f}, "
                f"chunk_vs_base={base_total_ms / chunk_total_ms:.2f}, base_overlap_ratio={base_overlap_ratio:.2%}"
            )
        dist_print(msg, need_sync=True, allowed_ranks=list(range(world_size)))

        return {
            "chunk_total_ms": chunk_total_ms,
            "base_total_ms": base_total_ms,
            "torch_total_ms": torch_total_ms,
            "torch_ag_ms": torch_ag_ms,
            "torch_gemm_ms": torch_gemm_ms,
            "chunk_speedup_vs_torch": torch_total_ms / chunk_total_ms,
            "base_speedup_vs_torch": torch_total_ms / base_total_ms if args.compare_base else float("nan"),
            "chunk_speedup_vs_base": base_total_ms / chunk_total_ms if args.compare_base else float("nan"),
            "chunk_overlap_ratio": chunk_overlap_ratio,
            "base_overlap_ratio": base_overlap_ratio,
            "prefetch_k": schedule.prefetch_chunk[1] - schedule.prefetch_chunk[0],
            "num_chunks": schedule.num_chunks,
        }
    finally:
        chunk_ctx.finalize()
        if base_ctx is not None:
            base_ctx.finalize()


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
        csv_file = Path("csv") / f"perf_chunk_ag_gemm_{TP_GROUP.size()}_ranks.csv"
        with open(csv_file, "w") as fout:
            print(
                ",".join([
                    "Model",
                    "M",
                    "N",
                    "K",
                    "chunk_ag_gemm_ms",
                    "base_ag_gemm_ms",
                    "torch_ag_gemm_ms",
                    "torch_ag_only_ms",
                    "torch_gemm_only_ms",
                    "chunk_speedup_vs_torch",
                    "base_speedup_vs_torch",
                    "chunk_speedup_vs_base",
                    "chunk_overlap_ratio",
                    "base_overlap_ratio",
                    "prefetch_k",
                    "num_chunks",
                ]),
                file=fout,
            )
            for model, config, metrics in perf_res:
                print(
                    ",".join([model] + list(
                        map(
                            str,
                            [
                                args.M,
                                config["N"],
                                config["K"],
                                f"{metrics['chunk_total_ms']:.4f}",
                                f"{metrics['base_total_ms']:.4f}",
                                f"{metrics['torch_total_ms']:.4f}",
                                f"{metrics['torch_ag_ms']:.4f}",
                                f"{metrics['torch_gemm_ms']:.4f}",
                                f"{metrics['chunk_speedup_vs_torch']:.4f}",
                                f"{metrics['base_speedup_vs_torch']:.4f}",
                                f"{metrics['chunk_speedup_vs_base']:.4f}",
                                f"{metrics['chunk_overlap_ratio']:.4f}",
                                f"{metrics['base_overlap_ratio']:.4f}",
                                metrics["prefetch_k"],
                                metrics["num_chunks"],
                            ],
                        ))),
                    file=fout,
                    flush=True,
                )
        print(f"csv file is dumped into {csv_file}")

    finalize_distributed()
