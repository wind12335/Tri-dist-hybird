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
from triton_dist.kernels.nvidia.reduce_scatter import reduce_scatter_2d_op, ring_reduce
from triton_dist.profiler_utils import group_profile, perf_func
from triton_dist.test.utils import LAYER_CONFIGS, assert_allclose
from triton_dist.utils import (dist_print, finalize_distributed, initialize_distributed, nvshmem_barrier_all_on_stream,
                               rand_tensor, wait_until_max_gpu_clock_or_warning)


AUTOTUNE_CACHE: dict[Tuple, dict] = {}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--M", type=int, default=8192)
    parser.add_argument("--N", type=int, default=None)
    parser.add_argument("--K", type=int, default=None)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--warmup_iters", type=int, default=5)
    parser.add_argument("--autotune", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--mode", type=str, default="all", choices=["all", "nonfused", "fused"])
    parser.add_argument("--profile", action="store_true", default=False)
    parser.add_argument("--profile_target",
                        type=str,
                        default="all",
                        choices=["all", "nonfused", "fused", "torch"])
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
        B = rand_tensor([N, K_per_rank], dtype=dtype, device=device).T * scale
    else:
        B = rand_tensor([K_per_rank, N], dtype=dtype, device=device) * scale
    return A, B


def torch_gemm_rs(
    pg: torch.distributed.ProcessGroup,
    A: torch.Tensor,  # [M, K_per_rank]
    B: torch.Tensor,  # [K_per_rank, N]
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


def choose_gemm_config(persistent: bool):
    return get_config_space(persistent)[0]


def get_autotuned_gemm_rs_config(A: torch.Tensor,
                                 B: torch.Tensor,
                                 ctx,
                                 persistent: bool,
                                 fuse_scatter: bool,
                                 pg: torch.distributed.ProcessGroup):
    """
    `gemm_rs` 的 autotune config_space 本身包含 `persistent/fuse_scatter`。
    benchmark 如果再把这两个参数直接传给 `gemm_rs(...)`，AutoTuner 会因为重复参数报错。

    这里在 benchmark 内部手动做一层 autotune：
    - cache key 显式纳入 `persistent/fuse_scatter`
    - 直接调用 AutoTuner 的 `get_pruned_config()/tune()`
    - 最终返回 best_config 中的 `gemm_config`
    """
    base_key = gemm_rs.key_fn(A, B, ctx)
    cache_key = (base_key, persistent, fuse_scatter)
    best_config = AUTOTUNE_CACHE.get(cache_key)
    if best_config is None:
        config_space = gemm_rs.get_pruned_config(A, B, ctx, persistent=persistent, fuse_scatter=fuse_scatter)
        timings = gemm_rs.tune(config_space, pg, A, B, ctx, persistent=persistent, fuse_scatter=fuse_scatter)
        timings.sort(key=lambda x: x[0])
        assert len(timings) > 0, "autotune returned empty timing list"
        best_config = timings[0][1]
        AUTOTUNE_CACHE[cache_key] = best_config
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


def perf_test(model_name: str, M: int, config: Dict[str, int], pg: torch.distributed.ProcessGroup):
    N = config["N"]
    K = config["K"]
    rank = pg.rank()
    world_size = pg.size()
    local_world_size = LOCAL_WORLD_SIZE
    if args.mode in ["all", "fused"] and world_size != local_world_size:
        raise AssertionError("fused GEMM-RS benchmark currently only supports single-node runs")

    if rank == 0:
        print(f"[{model_name}] test shape: M {M}, N {N}, K {K}")

    assert M % world_size == 0
    assert K % world_size == 0

    A, B = make_data(M, N, K, dtype, args.trans_b, pg)
    M_per_rank = M // world_size
    gemm_config = choose_gemm_config(args.persistent)

    atol = 6e-2 if dtype == torch.bfloat16 else 1e-2
    rtol = atol

    rs_stream = torch.cuda.Stream(priority=-1)
    ctx = create_gemm_rs_context(M, N, rank, world_size, local_world_size, dtype, rs_stream)
    workspace_nonfused = torch.zeros((world_size,), dtype=torch.int32, device=A.device)
    workspace_fused = torch.zeros((world_size,), dtype=torch.int32, device=A.device)
    triton_nonfused_rs_output = torch.empty((M_per_rank, N), dtype=dtype, device=A.device)
    triton_fused_reduce_output = torch.empty((M_per_rank, N), dtype=dtype, device=A.device)

    torch_partial = torch.matmul(A, B)
    gemm_out = ctx.get_gemm_out_buf(A)

    def _torch_total():
        return torch_gemm_rs(pg, A, B)

    def _torch_gemm_only():
        return torch.matmul(A, B)

    def _torch_rs_only():
        output = torch.empty((M_per_rank, N), dtype=dtype, device=A.device)
        torch.distributed.reduce_scatter_tensor(output, torch_partial, group=pg)
        return output

    def _triton_nonfused_total():
        if args.autotune:
            tuned_gemm_config = get_autotuned_gemm_rs_config(A, B, ctx, args.persistent, False, pg)
            return gemm_rs.fn(A,
                              B,
                              ctx,
                              gemm_config=tuned_gemm_config,
                              persistent=args.persistent,
                              fuse_scatter=False)
        return gemm_rs.fn(A,
                          B,
                          ctx,
                          gemm_config=gemm_config,
                          persistent=args.persistent,
                          fuse_scatter=False)

    def _triton_fused_total():
        if args.autotune:
            tuned_gemm_config = get_autotuned_gemm_rs_config(A, B, ctx, args.persistent, True, pg)
            return gemm_rs.fn(A,
                              B,
                              ctx,
                              gemm_config=tuned_gemm_config,
                              persistent=args.persistent,
                              fuse_scatter=True)
        return gemm_rs.fn(A,
                          B,
                          ctx,
                          gemm_config=gemm_config,
                          persistent=args.persistent,
                          fuse_scatter=True)

    def _triton_nonfused_gemm_only():
        return launch_triton_gemm_producer(A, B, ctx, workspace_nonfused, args.persistent, False, gemm_config)

    def _triton_nonfused_rs_only():
        gemm_out.copy_(torch_partial)
        ctx.rs_ctx.scatter_signal_buf.fill_(1)
        return reduce_scatter_2d_op(gemm_out, ctx.rs_ctx, output=triton_nonfused_rs_output)

    def _prepare_fused_layout():
        sync_all(pg)
        launch_triton_gemm_producer(A, B, ctx, workspace_fused, args.persistent, True, gemm_config)
        sync_all(pg)

    def _triton_fused_producer_only():
        return launch_triton_gemm_producer(A, B, ctx, workspace_fused, args.persistent, True, gemm_config)

    def _triton_fused_reduce_only():
        return ring_reduce(gemm_out, triton_fused_reduce_output, ctx.rs_ctx.local_rank, ctx.rs_ctx.local_world_size)

    try:
        # Warmup and correctness
        for _ in range(3):
            sync_all(pg)
            C_nonfused = _triton_nonfused_total()
            if args.mode in ["all", "fused"]:
                sync_all(pg)
                C_fused = _triton_fused_total()

        C_torch = _torch_total()
        for i in range(world_size):
            torch.distributed.barrier(pg)
            if rank == i:
                assert_allclose(C_torch, C_nonfused, atol=atol, rtol=rtol)
                if args.mode in ["all", "fused"]:
                    assert_allclose(C_torch, C_fused, atol=atol, rtol=rtol)

        run_id = os.environ.get("TORCHELASTIC_RUN_ID", "local")
        with group_profile(f"gemm_rs_perf_m_{M}_n_{N}_k_{K}_{run_id}", args.profile, group=TP_GROUP):
            if args.profile_target in ["all", "nonfused"] and args.mode in ["all", "nonfused"]:
                sync_all(pg)
                profile_section("triton/nonfused_total", True, _triton_nonfused_total, args.iters, args.warmup_iters)
                sync_all(pg)
                profile_section("triton/nonfused_gemm_only", True, _triton_nonfused_gemm_only, args.iters,
                                args.warmup_iters)
                sync_all(pg)
                profile_section("triton/nonfused_rs_only", True, _triton_nonfused_rs_only, args.iters,
                                args.warmup_iters)

            if args.profile_target in ["all", "fused"] and args.mode in ["all", "fused"]:
                sync_all(pg)
                profile_section("triton/fused_total", True, _triton_fused_total, args.iters, args.warmup_iters)
                sync_all(pg)
                profile_section("triton/fused_producer_only", True, _triton_fused_producer_only, args.iters,
                                args.warmup_iters)
                _prepare_fused_layout()
                profile_section("triton/fused_reduce_only", True, _triton_fused_reduce_only, args.iters,
                                args.warmup_iters)

            if args.profile_target in ["all", "torch"]:
                sync_all(pg)
                profile_section("torch/reference_total", True, _torch_total, args.iters, args.warmup_iters)
                sync_all(pg)
                profile_section("torch/reference_gemm_only", True, _torch_gemm_only, args.iters, args.warmup_iters)
                sync_all(pg)
                profile_section("torch/reference_rs_only", True, _torch_rs_only, args.iters, args.warmup_iters)

        metrics = {
            "triton_nonfused_total_ms": float("nan"),
            "triton_nonfused_gemm_only_ms": float("nan"),
            "triton_nonfused_rs_only_ms": float("nan"),
            "triton_fused_total_ms": float("nan"),
            "triton_fused_producer_only_ms": float("nan"),
            "triton_fused_reduce_only_ms": float("nan"),
            "torch_total_ms": float("nan"),
            "torch_gemm_only_ms": float("nan"),
            "torch_rs_only_ms": float("nan"),
        }

        if args.mode in ["all", "nonfused"]:
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

        if args.mode in ["all", "fused"]:
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

        sync_all(pg)
        wait_until_max_gpu_clock_or_warning(torch.cuda.current_device())
        _, metrics["torch_total_ms"] = perf_func(_torch_total, iters=args.iters, warmup_iters=args.warmup_iters)
        sync_all(pg)
        wait_until_max_gpu_clock_or_warning(torch.cuda.current_device())
        _, metrics["torch_gemm_only_ms"] = perf_func(_torch_gemm_only,
                                                     iters=args.iters,
                                                     warmup_iters=args.warmup_iters)
        sync_all(pg)
        wait_until_max_gpu_clock_or_warning(torch.cuda.current_device())
        _, metrics["torch_rs_only_ms"] = perf_func(_torch_rs_only, iters=args.iters, warmup_iters=args.warmup_iters)

        serial_torch_ms = metrics["torch_gemm_only_ms"] + metrics["torch_rs_only_ms"]
        metrics["triton_nonfused_overlap_ratio"] = ((serial_torch_ms - metrics["triton_nonfused_total_ms"]) /
                                                    max(serial_torch_ms, 1e-6))
        metrics["triton_fused_overlap_ratio"] = ((serial_torch_ms - metrics["triton_fused_total_ms"]) /
                                                 max(serial_torch_ms, 1e-6))
        metrics["nonfused_speedup_vs_torch"] = metrics["torch_total_ms"] / metrics["triton_nonfused_total_ms"]
        metrics["fused_speedup_vs_torch"] = metrics["torch_total_ms"] / metrics["triton_fused_total_ms"]
        metrics["fused_speedup_vs_nonfused"] = metrics["triton_nonfused_total_ms"] / metrics["triton_fused_total_ms"]

        flops = 2 * M * N * (K // world_size)
        reduce_scatter_gb = M * N * dtype.itemsize / 2**30 * (world_size - 1) / world_size
        metrics["triton_nonfused_tflops"] = flops / metrics["triton_nonfused_total_ms"] * 1e-9
        metrics["triton_fused_tflops"] = flops / metrics["triton_fused_total_ms"] * 1e-9
        metrics["torch_gemm_tflops"] = flops / metrics["torch_gemm_only_ms"] * 1e-9
        metrics["torch_rs_gbps"] = reduce_scatter_gb / metrics["torch_rs_only_ms"] * 1e3
        metrics["triton_nonfused_rs_gbps"] = reduce_scatter_gb / metrics["triton_nonfused_rs_only_ms"] * 1e3
        metrics["triton_fused_reduce_gbps"] = reduce_scatter_gb / metrics["triton_fused_reduce_only_ms"] * 1e3

        msg = (
            f"Rank {rank} [{model_name}] latency (ms): "
            f"torch_total={metrics['torch_total_ms']:.2f}, torch_gemm_only={metrics['torch_gemm_only_ms']:.2f}, "
            f"torch_rs_only={metrics['torch_rs_only_ms']:.2f}"
        )
        if args.mode in ["all", "nonfused"]:
            msg += (
                f", triton_nonfused_total={metrics['triton_nonfused_total_ms']:.2f}, "
                f"triton_nonfused_gemm_only={metrics['triton_nonfused_gemm_only_ms']:.2f}, "
                f"triton_nonfused_rs_only={metrics['triton_nonfused_rs_only_ms']:.2f}, "
                f"nonfused_speedup={metrics['nonfused_speedup_vs_torch']:.2f}, "
                f"nonfused_overlap_ratio={metrics['triton_nonfused_overlap_ratio']:.2%}"
            )
        if args.mode in ["all", "fused"]:
            msg += (
                f", triton_fused_total={metrics['triton_fused_total_ms']:.2f}, "
                f"triton_fused_producer_only={metrics['triton_fused_producer_only_ms']:.2f}, "
                f"triton_fused_reduce_only={metrics['triton_fused_reduce_only_ms']:.2f}, "
                f"fused_speedup={metrics['fused_speedup_vs_torch']:.2f}, "
                f"fused_overlap_ratio={metrics['triton_fused_overlap_ratio']:.2%}"
            )
        if args.mode == "all":
            msg += f", fused_vs_nonfused={metrics['fused_speedup_vs_nonfused']:.2f}"

        dist_print(msg, need_sync=True, allowed_ranks=list(range(world_size)))
        return metrics
    finally:
        ctx.finalize()


if __name__ == "__main__":
    args = parse_args()
    if torch.cuda.get_device_capability()[0] < 9 and args.persistent:
        raise AssertionError("persistent GEMM-RS is not supported on cuda capability < 9.0")

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    TP_GROUP = initialize_distributed()
    LOCAL_WORLD_SIZE = int(os.environ.get("LOCAL_WORLD_SIZE", TP_GROUP.size()))

    configs = get_test_configs(args)
    perf_res = []
    for model_name, config in configs.items():
        metrics = perf_test(model_name, args.M, config, TP_GROUP)
        perf_res.append((model_name, config, metrics))

    if args.dump_csv and TP_GROUP.rank() == 0:
        os.makedirs("csv", exist_ok=True)
        csv_file = Path("csv") / f"perf_gemm_rs_{TP_GROUP.size()}_ranks.csv"
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
                    "triton_fused_total_ms",
                    "triton_fused_producer_only_ms",
                    "triton_fused_reduce_only_ms",
                    "nonfused_speedup_vs_torch",
                    "fused_speedup_vs_torch",
                    "fused_speedup_vs_nonfused",
                    "triton_nonfused_overlap_ratio",
                    "triton_fused_overlap_ratio",
                    "triton_nonfused_tflops",
                    "triton_fused_tflops",
                    "torch_gemm_tflops",
                    "torch_rs_gbps",
                    "triton_nonfused_rs_gbps",
                    "triton_fused_reduce_gbps",
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
                                    f"{metrics['triton_fused_total_ms']:.4f}",
                                    f"{metrics['triton_fused_producer_only_ms']:.4f}",
                                    f"{metrics['triton_fused_reduce_only_ms']:.4f}",
                                    f"{metrics['nonfused_speedup_vs_torch']:.4f}",
                                    f"{metrics['fused_speedup_vs_torch']:.4f}",
                                    f"{metrics['fused_speedup_vs_nonfused']:.4f}",
                                    f"{metrics['triton_nonfused_overlap_ratio']:.4f}",
                                    f"{metrics['triton_fused_overlap_ratio']:.4f}",
                                    f"{metrics['triton_nonfused_tflops']:.4f}",
                                    f"{metrics['triton_fused_tflops']:.4f}",
                                    f"{metrics['torch_gemm_tflops']:.4f}",
                                    f"{metrics['torch_rs_gbps']:.4f}",
                                    f"{metrics['triton_nonfused_rs_gbps']:.4f}",
                                    f"{metrics['triton_fused_reduce_gbps']:.4f}",
                                ],
                            ))),
                    file=fout,
                    flush=True,
                )
        print(f"csv file is dumped into {csv_file}")

    finalize_distributed()
