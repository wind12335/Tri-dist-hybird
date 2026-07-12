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

from triton_dist.kernels.nvidia import create_gemm_rs_context, create_new_gemm_rs_context, gemm_rs, new_gemm_rs
from triton_dist.kernels.nvidia.gemm import get_config_space
from triton_dist.kernels.nvidia.gemm_reduce_scatter import (gemm_rs_producer_non_persistent,
                                                            gemm_rs_producer_persistent)
from triton_dist.kernels.nvidia.new_gemm_reducescatter import gemm_rs_producer_non_persistent_chunked
from triton_dist.kernels.nvidia.new_reducescatter import new_reduce_scatter_2d_op
from triton_dist.kernels.nvidia.reduce_scatter import reduce_scatter_2d_op
from triton_dist.profiler_utils import group_profile, perf_func
from triton_dist.test.utils import LAYER_CONFIGS, assert_allclose
from triton_dist.utils import (dist_print, finalize_distributed, initialize_distributed, nvshmem_barrier_all_on_stream,
                               rand_tensor, wait_until_max_gpu_clock_or_warning)
# torchrun --nproc_per_node=4 python/triton_dist/benchmark/bench_new_gemm_reducescatter.py --M 8192 --N 29568 --K 8192 --mode all --profile --autotune --no-persistent
# torchrun --nproc_per_node=4 python/triton_dist/benchmark/bench_new_gemm_reducescatter.py --M 8192 --N 29568 --K 8192 --mode new --profile --autotune --no-persistent
# torchrun --nproc_per_node=4 python/triton_dist/benchmark/bench_new_gemm_reducescatter.py --M 8192 --N 29568 --K 8192 --mode base --profile --autotune --no-persistent


BASE_AUTOTUNE_CACHE: dict[Tuple, dict] = {}
NEW_AUTOTUNE_CACHE: dict[Tuple, dict] = {}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--M", type=int, default=8192)
    parser.add_argument("--N", type=int, default=None)
    parser.add_argument("--K", type=int, default=None)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--warmup_iters", type=int, default=5)
    parser.add_argument("--autotune", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--mode", type=str, default="all", choices=["all", "base", "new", "torch"])
    parser.add_argument("--profile", action="store_true", default=False)
    parser.add_argument("--profile_target",
                        type=str,
                        default="all",
                        choices=["all", "base", "new", "torch"])
    parser.add_argument("--dump_csv", action="store_true", default=False)
    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    parser.add_argument("--trans_b", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--persistent",
                        action=argparse.BooleanOptionalAction,
                        default=torch.cuda.get_device_capability() >= (9, 0))
    parser.add_argument("--chunk_rows",
                        type=int,
                        default=0,
                        help=">0 使用固定 chunk 行数；0 使用启发式自动选择")
    parser.add_argument("--target_chunks_per_rank",
                        type=int,
                        default=4,
                        help="启发式目标：每个 rank 期望切成多少个 chunk")
    parser.add_argument("--min_chunk_rows",
                        type=int,
                        default=256,
                        help="启发式自动选 chunk 时的最小行数")
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


def torch_gemm_rs(
    pg: torch.distributed.ProcessGroup,
    A: torch.Tensor,
    B: torch.Tensor,
):
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


def choose_base_gemm_config(persistent: bool):
    return get_config_space(persistent)[0]


def choose_new_gemm_config():
    return get_config_space(False)[0]


def get_autotuned_base_gemm_rs_config(
    A: torch.Tensor,
    B: torch.Tensor,
    ctx,
    persistent: bool,
    pg: torch.distributed.ProcessGroup,
):
    base_key = gemm_rs.key_fn(A, B, ctx)
    cache_key = (base_key, persistent, False)
    best_config = BASE_AUTOTUNE_CACHE.get(cache_key)
    if best_config is None:
        config_space = gemm_rs.get_pruned_config(A, B, ctx, persistent=persistent, fuse_scatter=False)
        timings = gemm_rs.tune(config_space, pg, A, B, ctx, persistent=persistent, fuse_scatter=False)
        timings.sort(key=lambda x: x[0])
        assert len(timings) > 0, "baseline autotune returned empty timing list"
        best_config = timings[0][1]
        BASE_AUTOTUNE_CACHE[cache_key] = best_config
    return best_config["gemm_config"]


def get_autotuned_new_gemm_rs_config(
    A: torch.Tensor,
    B: torch.Tensor,
    ctx,
    pg: torch.distributed.ProcessGroup,
):
    base_key = new_gemm_rs.key_fn(A, B, ctx)
    best_config = NEW_AUTOTUNE_CACHE.get(base_key)
    if best_config is None:
        config_space = new_gemm_rs.get_pruned_config(A, B, ctx)
        timings = new_gemm_rs.tune(config_space, pg, A, B, ctx)
        timings.sort(key=lambda x: x[0])
        assert len(timings) > 0, "new autotune returned empty timing list"
        best_config = timings[0][1]
        NEW_AUTOTUNE_CACHE[base_key] = best_config
    return best_config["gemm_config"]


def launch_base_gemm_producer(A: torch.Tensor,
                              B: torch.Tensor,
                              ctx,
                              workspace: torch.Tensor,
                              persistent: bool,
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


def launch_new_gemm_producer(A: torch.Tensor, B: torch.Tensor, ctx, workspace: torch.Tensor, gemm_config):
    gemm_out = ctx.get_gemm_out_buf(A)
    workspace.zero_()
    ctx.rs_ctx.reset_runtime_state()
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
        gemm_config,
    )
    return gemm_out


def perf_test(model_name: str, M: int, config: Dict[str, int], pg: torch.distributed.ProcessGroup):
    N = config["N"]
    K = config["K"]
    rank = pg.rank()
    world_size = pg.size()
    local_world_size = LOCAL_WORLD_SIZE

    if world_size != local_world_size and args.mode in ["all", "new"]:
        raise AssertionError("new_gemm_rs benchmark currently only supports single-node runs")
    if args.persistent and args.mode in ["all", "new"]:
        raise AssertionError("new_gemm_rs currently only supports non-persistent producer path")

    if rank == 0:
        print(f"[{model_name}] test shape: M {M}, N {N}, K {K}")

    assert M % world_size == 0
    assert K % world_size == 0

    A, B = make_data(M, N, K, dtype, args.trans_b, pg)
    M_per_rank = M // world_size
    base_gemm_config = choose_base_gemm_config(args.persistent)
    new_gemm_config = choose_new_gemm_config()

    atol = 6e-2 if dtype == torch.bfloat16 else 1e-2
    rtol = atol
    torch_partial = torch.matmul(A, B)

    def _torch_total():
        return torch_gemm_rs(pg, A, B)

    def _torch_gemm_only():
        return torch.matmul(A, B)

    def _torch_rs_only():
        output = torch.empty((M_per_rank, N), dtype=dtype, device=A.device)
        torch.distributed.reduce_scatter_tensor(output, torch_partial, group=pg)
        return output
    metrics = {
        "base_total_ms": float("nan"),
        "base_gemm_only_ms": float("nan"),
        "base_rs_only_ms": float("nan"),
        "new_total_ms": float("nan"),
        "new_gemm_only_ms": float("nan"),
        "new_rs_only_ms": float("nan"),
        "torch_total_ms": float("nan"),
        "torch_gemm_only_ms": float("nan"),
        "torch_rs_only_ms": float("nan"),
    }

    # 先算 torch reference，后面 base/new 都用它做 correctness 对照。
    sync_all(pg)
    C_torch = _torch_total()

    def _run_base_stage():
        base_ctx = None
        try:
            base_rs_stream = torch.cuda.Stream(priority=-1)
            base_ctx = create_gemm_rs_context(M, N, rank, world_size, local_world_size, dtype, base_rs_stream)
            workspace_base = torch.zeros((world_size,), dtype=torch.int32, device=A.device)
            base_rs_output = torch.empty((M_per_rank, N), dtype=dtype, device=A.device)

            def _get_base_config():
                if args.autotune:
                    return get_autotuned_base_gemm_rs_config(A, B, base_ctx, args.persistent, pg)
                return base_gemm_config

            def _base_total():
                return gemm_rs.fn(
                    A,
                    B,
                    base_ctx,
                    gemm_config=_get_base_config(),
                    persistent=args.persistent,
                    fuse_scatter=False,
                )

            def _base_gemm_only():
                return launch_base_gemm_producer(A, B, base_ctx, workspace_base, args.persistent, _get_base_config())

            def _base_rs_only():
                gemm_out = base_ctx.get_gemm_out_buf(A)
                gemm_out.copy_(torch_partial)
                base_ctx.rs_ctx.scatter_signal_buf.fill_(1)
                return reduce_scatter_2d_op(gemm_out, base_ctx.rs_ctx, output=base_rs_output)

            for _ in range(3):
                sync_all(pg)
                C_base = _base_total()

            for i in range(world_size):
                torch.distributed.barrier(pg)
                if rank == i:
                    assert_allclose(C_torch, C_base, atol=atol, rtol=rtol)

            if args.profile_target in ["all", "base"]:
                sync_all(pg)
                profile_section("baseline_gemm_rs/total", True, _base_total, args.iters, args.warmup_iters)
                sync_all(pg)
                profile_section("baseline_gemm_rs/gemm_only", True, _base_gemm_only, args.iters, args.warmup_iters)
                sync_all(pg)
                profile_section("baseline_gemm_rs/rs_only", True, _base_rs_only, args.iters, args.warmup_iters)

            sync_all(pg)
            wait_until_max_gpu_clock_or_warning(torch.cuda.current_device())
            _, metrics["base_total_ms"] = perf_func(_base_total, iters=args.iters, warmup_iters=args.warmup_iters)
            sync_all(pg)
            wait_until_max_gpu_clock_or_warning(torch.cuda.current_device())
            _, metrics["base_gemm_only_ms"] = perf_func(_base_gemm_only, iters=args.iters, warmup_iters=args.warmup_iters)
            sync_all(pg)
            wait_until_max_gpu_clock_or_warning(torch.cuda.current_device())
            _, metrics["base_rs_only_ms"] = perf_func(_base_rs_only, iters=args.iters, warmup_iters=args.warmup_iters)
        finally:
            if base_ctx is not None:
                base_ctx.finalize()

    def _run_new_stage():
        new_ctx = None
        try:
            new_rs_stream = torch.cuda.Stream(priority=-1)
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
                    "[new] chunk config: "
                    f"chunk_rows={new_ctx.rs_ctx.chunk_rows}, "
                    f"num_chunks={new_ctx.rs_ctx.num_chunks}, "
                    f"target_chunks_per_rank={args.target_chunks_per_rank}, "
                    f"min_chunk_rows={args.min_chunk_rows}"
                )

            workspace_new = torch.zeros((world_size * new_ctx.rs_ctx.num_chunks,), dtype=torch.int32, device=A.device)
            new_rs_output = torch.empty((M_per_rank, N), dtype=dtype, device=A.device)

            def _get_new_config():
                if args.autotune:
                    return get_autotuned_new_gemm_rs_config(A, B, new_ctx, pg)
                return new_gemm_config

            def _new_total():
                return new_gemm_rs.fn(
                    A,
                    B,
                    new_ctx,
                    gemm_config=_get_new_config(),
                )

            def _new_gemm_only():
                return launch_new_gemm_producer(A, B, new_ctx, workspace_new, _get_new_config())

            def _new_rs_only():
                gemm_out = new_ctx.get_gemm_out_buf(A)
                gemm_out.copy_(torch_partial)
                new_ctx.rs_ctx.reset_runtime_state()
                new_ctx.rs_ctx.chunk_signal.fill_(1)
                return new_reduce_scatter_2d_op(gemm_out, new_ctx.rs_ctx, output=new_rs_output)

            for _ in range(3):
                sync_all(pg)
                C_new = _new_total()

            for i in range(world_size):
                torch.distributed.barrier(pg)
                if rank == i:
                    assert_allclose(C_torch, C_new, atol=atol, rtol=rtol)

            if args.profile_target in ["all", "new"]:
                sync_all(pg)
                profile_section("new_gemm_rs/total", True, _new_total, args.iters, args.warmup_iters)
                sync_all(pg)
                profile_section("new_gemm_rs/producer_only", True, _new_gemm_only, args.iters, args.warmup_iters)
                sync_all(pg)
                profile_section("new_gemm_rs/rs_only", True, _new_rs_only, args.iters, args.warmup_iters)

            sync_all(pg)
            wait_until_max_gpu_clock_or_warning(torch.cuda.current_device())
            _, metrics["new_total_ms"] = perf_func(_new_total, iters=args.iters, warmup_iters=args.warmup_iters)
            sync_all(pg)
            wait_until_max_gpu_clock_or_warning(torch.cuda.current_device())
            _, metrics["new_gemm_only_ms"] = perf_func(_new_gemm_only, iters=args.iters, warmup_iters=args.warmup_iters)
            sync_all(pg)
            wait_until_max_gpu_clock_or_warning(torch.cuda.current_device())
            _, metrics["new_rs_only_ms"] = perf_func(_new_rs_only, iters=args.iters, warmup_iters=args.warmup_iters)
        finally:
            if new_ctx is not None:
                new_ctx.finalize()

    run_id = os.environ.get("TORCHELASTIC_RUN_ID", "local")
    with group_profile(f"new_gemm_rs_perf_m_{M}_n_{N}_k_{K}_{run_id}", args.profile, group=TP_GROUP):
        if args.mode in ["all", "base"]:
            _run_base_stage()

        if args.mode in ["all", "new"]:
            _run_new_stage()

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

    if args.mode in ["all", "base"]:
        metrics["base_internal_overlap_ratio"] = 1.0 - metrics["base_total_ms"] / max(
            metrics["base_gemm_only_ms"] + metrics["base_rs_only_ms"], 1e-6)
        metrics["base_visible_comm_overhead_ms"] = metrics["base_total_ms"] - metrics["base_gemm_only_ms"]
        metrics["base_speedup_vs_torch"] = metrics["torch_total_ms"] / metrics["base_total_ms"]
    else:
        metrics["base_internal_overlap_ratio"] = float("nan")
        metrics["base_visible_comm_overhead_ms"] = float("nan")
        metrics["base_speedup_vs_torch"] = float("nan")

    if args.mode in ["all", "new"]:
        metrics["new_internal_overlap_ratio"] = 1.0 - metrics["new_total_ms"] / max(
            metrics["new_gemm_only_ms"] + metrics["new_rs_only_ms"], 1e-6)
        metrics["new_visible_comm_overhead_ms"] = metrics["new_total_ms"] - metrics["new_gemm_only_ms"]
        metrics["new_speedup_vs_torch"] = metrics["torch_total_ms"] / metrics["new_total_ms"]
    else:
        metrics["new_internal_overlap_ratio"] = float("nan")
        metrics["new_visible_comm_overhead_ms"] = float("nan")
        metrics["new_speedup_vs_torch"] = float("nan")

    if args.mode == "all":
        metrics["new_speedup_vs_base"] = metrics["base_total_ms"] / metrics["new_total_ms"]
    else:
        metrics["new_speedup_vs_base"] = float("nan")

    flops = 2 * M * N * (K // world_size)
    reduce_scatter_gb = M * N * dtype.itemsize / 2**30 * (world_size - 1) / world_size
    metrics["base_tflops"] = flops / metrics["base_total_ms"] * 1e-9
    metrics["new_tflops"] = flops / metrics["new_total_ms"] * 1e-9
    metrics["torch_gemm_tflops"] = flops / metrics["torch_gemm_only_ms"] * 1e-9
    metrics["base_rs_gbps"] = reduce_scatter_gb / metrics["base_rs_only_ms"] * 1e3
    metrics["new_rs_gbps"] = reduce_scatter_gb / metrics["new_rs_only_ms"] * 1e3
    metrics["torch_rs_gbps"] = reduce_scatter_gb / metrics["torch_rs_only_ms"] * 1e3

    msg = (
        f"Rank {rank} [{model_name}] latency (ms): "
        f"torch_total={metrics['torch_total_ms']:.2f}, "
        f"torch_gemm_only={metrics['torch_gemm_only_ms']:.2f}, "
        f"torch_rs_only={metrics['torch_rs_only_ms']:.2f}"
    )
    if args.mode in ["all", "base"]:
        msg += (
            f", base_total={metrics['base_total_ms']:.2f}, "
            f"base_gemm_only={metrics['base_gemm_only_ms']:.2f}, "
            f"base_rs_only={metrics['base_rs_only_ms']:.2f}, "
            f"base_internal_overlap={metrics['base_internal_overlap_ratio'] * 100:.2f}%, "
            f"base_visible_comm_overhead={metrics['base_visible_comm_overhead_ms']:.2f}, "
            f"base_speedup_vs_torch={metrics['base_speedup_vs_torch']:.2f}"
        )
    if args.mode in ["all", "new"]:
        msg += (
            f", new_total={metrics['new_total_ms']:.2f}, "
            f"new_gemm_only={metrics['new_gemm_only_ms']:.2f}, "
            f"new_rs_only={metrics['new_rs_only_ms']:.2f}, "
            f"new_internal_overlap={metrics['new_internal_overlap_ratio'] * 100:.2f}%, "
            f"new_visible_comm_overhead={metrics['new_visible_comm_overhead_ms']:.2f}, "
            f"new_speedup_vs_base={metrics['new_speedup_vs_base']:.2f}, "
            f"new_speedup_vs_torch={metrics['new_speedup_vs_torch']:.2f}"
        )
    dist_print(msg, need_sync=True, allowed_ranks=list(range(world_size)))
    return metrics


if __name__ == "__main__":
    args = parse_args()

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    TP_GROUP = initialize_distributed()
    LOCAL_WORLD_SIZE = int(os.environ.get("LOCAL_WORLD_SIZE", 8))

    perf_rows = []
    test_configs = get_test_configs(args)

    for model_name, config in test_configs.items():
        metrics = perf_test(model_name, args.M, config, TP_GROUP)
        perf_rows.append((model_name, config, metrics))

    if args.dump_csv and TP_GROUP.rank() == 0:
        os.makedirs("csv", exist_ok=True)
        csv_file = Path("csv") / f"perf_new_gemm_rs_{TP_GROUP.size()}_ranks.csv"
        with open(csv_file, "w") as fout:
            print(
                ",".join([
                    "Model",
                    "M",
                    "N",
                    "K",
                    "torch_total_ms",
                    "base_total_ms",
                    "new_total_ms",
                    "base_internal_overlap_ratio",
                    "new_internal_overlap_ratio",
                    "new_speedup_vs_base",
                    "new_speedup_vs_torch",
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
                        f"{metrics['base_total_ms']:.4f}",
                        f"{metrics['new_total_ms']:.4f}",
                        f"{metrics['base_internal_overlap_ratio']:.6f}",
                        f"{metrics['new_internal_overlap_ratio']:.6f}",
                        f"{metrics['new_speedup_vs_base']:.6f}",
                        f"{metrics['new_speedup_vs_torch']:.6f}",
                    ]),
                    file=fout,
                    flush=True,
                )
        print(f"csv file is dumped into {csv_file}")

    finalize_distributed()
