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
import triton

from triton_dist.layers.nvidia import GemmARLayer
from triton_dist.profiler_utils import group_profile, perf_func
from triton_dist.test.utils import LAYER_CONFIGS, assert_allclose
from triton_dist.utils import (dist_print, finalize_distributed, initialize_distributed, nvshmem_barrier_all_on_stream,
                               rand_tensor, sleep_async, wait_until_max_gpu_clock_or_warning)
# torchrun --nproc_per_node=4 python/triton_dist/benchmark/benchmark_gemm_allreduce.py --M 8192 --N 11008 --K 4096 --dtype bfloat16


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--M", type=int, default=8192)
    parser.add_argument("--N", type=int, default=None)
    parser.add_argument("--K", type=int, default=None)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--warmup_iters", type=int, default=5)
    parser.add_argument("--profile", action="store_true", default=False)
    parser.add_argument("--dump_csv", action="store_true", default=False)
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16"])
    parser.add_argument("--persistent",
                        action=argparse.BooleanOptionalAction,
                        default=torch.cuda.get_device_capability() >= (9, 0))
    parser.add_argument("--copy_to_local", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num_comm_sms", type=int, default=16)
    parser.add_argument("--row_wise", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--low_latency", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--check", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--block_size_n",
        type=int,
        choices=[0, 64, 128, 256],
        default=0,
        help="Override GEMM BLOCK_SIZE_N; 0 keeps GemmARLayer's default (256).",
    )
    return parser.parse_args()


def get_test_configs(args):
    if args.N is not None or args.K is not None:
        if args.N is None or args.K is None:
            raise ValueError("`--N` and `--K` must be set together.")
        return {"custom": {"N": args.N, "K": args.K}}
    return LAYER_CONFIGS


def make_data(M, N, K, dtype: torch.dtype, tp_group: torch.distributed.ProcessGroup):
    rank = tp_group.rank()
    world_size = tp_group.size()
    assert K % world_size == 0
    local_k = K // world_size
    scale = 0.01 * (rank + 1)
    device = torch.cuda.current_device()
    a = rand_tensor((M, local_k), dtype=dtype, device=device) * scale
    weight = rand_tensor((N, local_k), dtype=dtype, device=device) * scale
    return a, weight


def torch_gemm_ar(
    a: torch.Tensor,
    weight: torch.Tensor,
    tp_group: torch.distributed.ProcessGroup,
):
    output = torch.matmul(a, weight.T)
    torch.distributed.all_reduce(output, group=tp_group)
    return output


def sync_all(pg: torch.distributed.ProcessGroup):
    nvshmem_barrier_all_on_stream(torch.cuda.current_stream())
    torch.cuda.synchronize()
    torch.distributed.barrier(pg)


def perf_test(model_name: str, M: int, config, pg: torch.distributed.ProcessGroup):
    N = config["N"]
    K = config["K"]
    rank = pg.rank()
    world_size = pg.size()

    if rank == 0:
        print(f"[{model_name}] test shape: M={M}, N={N}, K={K}")

    assert K % world_size == 0
    a, weight = make_data(M, N, K, dtype, pg)
    partial = torch.matmul(a, weight.T)

    user_gemm_config = None
    if args.block_size_n:
        num_sms = torch.cuda.get_device_properties("cuda").multi_processor_count
        user_gemm_config = triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": args.block_size_n,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 1,
                "NUM_GEMM_SMS": num_sms - args.num_comm_sms,
            },
            num_stages=3,
            num_warps=8,
        )

    try:
        gemm_ar = GemmARLayer(
            pg,
            M,
            N,
            K,
            dtype,
            dtype,
            LOCAL_WORLD_SIZE,
            persistent=args.persistent,
            use_ll_kernel=args.low_latency,
            copy_to_local=args.copy_to_local,
            NUM_COMM_SMS=args.num_comm_sms,
            TILE_MAP_LEVEL=int(args.row_wise),
            user_gemm_config=user_gemm_config,
        )
    except Exception as exc:
        err_msg = str(exc)
        if "Failed to allocate memory" in err_msg:
            dist_print(
                f"Rank {rank} [{model_name}] skipped: NVSHMEM symmetric allocation failed for M={M}, N={N}, K={K}",
                need_sync=True,
                allowed_ranks=list(range(world_size)),
            )
            torch.cuda.empty_cache()
            return {
                "skipped": True,
                "reason": "nvshmem_oom",
                "triton_total_ms": float("nan"),
                "triton_gemm_only_ms": float("nan"),
                "triton_ar_only_ms": float("nan"),
                "torch_total_ms": float("nan"),
                "torch_gemm_only_ms": float("nan"),
                "torch_ar_only_ms": float("nan"),
                "speedup_vs_torch": float("nan"),
                "overlap_ratio_vs_torch_serial": float("nan"),
            }
        raise

    def _torch_total():
        return torch_gemm_ar(a, weight, pg)

    def _torch_gemm_only():
        return torch.matmul(a, weight.T)

    def _torch_ar_only():
        output = partial.clone()
        torch.distributed.all_reduce(output, group=pg)
        return output

    def _triton_total():
        return gemm_ar.forward(a, weight)

    def _triton_gemm_only():
        return gemm_ar.forward_gemm(a, weight)

    def _triton_ar_only():
        return gemm_ar.forward_ar(partial)

    atol = 6e-2 if dtype == torch.bfloat16 else 1e-2
    rtol = atol

    try:
        for _ in range(3):
            sync_all(pg)
            triton_out = _triton_total()

        if args.check:
            sync_all(pg)
            torch_out = _torch_total()
            for i in range(world_size):
                torch.distributed.barrier(pg)
                if rank == i:
                    assert_allclose(torch_out, triton_out, atol=atol, rtol=rtol)

        run_id = os.environ.get("TORCHELASTIC_RUN_ID", "local")
        with group_profile(f"gemm_ar_perf_m_{M}_n_{N}_k_{K}_{run_id}", args.profile, group=TP_GROUP):
            sync_all(pg)
            sleep_async(100)
            perf_func(_triton_total, iters=args.iters, warmup_iters=args.warmup_iters)
            sync_all(pg)
            sleep_async(100)
            perf_func(_torch_total, iters=args.iters, warmup_iters=args.warmup_iters)

        metrics = {}

        sync_all(pg)
        sleep_async(100)
        wait_until_max_gpu_clock_or_warning(torch.cuda.current_device())
        _, metrics["triton_total_ms"] = perf_func(_triton_total, iters=args.iters, warmup_iters=args.warmup_iters)

        torch.cuda.synchronize()
        sleep_async(100)
        sync_all(pg)
        wait_until_max_gpu_clock_or_warning(torch.cuda.current_device())
        _, metrics["triton_gemm_only_ms"] = perf_func(_triton_gemm_only, iters=args.iters, warmup_iters=args.warmup_iters)

        torch.cuda.synchronize()
        sleep_async(100)
        sync_all(pg)
        wait_until_max_gpu_clock_or_warning(torch.cuda.current_device())
        _, metrics["triton_ar_only_ms"] = perf_func(_triton_ar_only, iters=args.iters, warmup_iters=args.warmup_iters)

        sync_all(pg)
        sleep_async(100)
        wait_until_max_gpu_clock_or_warning(torch.cuda.current_device())
        _, metrics["torch_total_ms"] = perf_func(_torch_total, iters=args.iters, warmup_iters=args.warmup_iters)

        sync_all(pg)
        wait_until_max_gpu_clock_or_warning(torch.cuda.current_device())
        _, metrics["torch_gemm_only_ms"] = perf_func(_torch_gemm_only, iters=args.iters, warmup_iters=args.warmup_iters)

        sync_all(pg)
        wait_until_max_gpu_clock_or_warning(torch.cuda.current_device())
        _, metrics["torch_ar_only_ms"] = perf_func(_torch_ar_only, iters=args.iters, warmup_iters=args.warmup_iters)

        serial_torch_ms = metrics["torch_gemm_only_ms"] + metrics["torch_ar_only_ms"]
        metrics["speedup_vs_torch"] = metrics["torch_total_ms"] / metrics["triton_total_ms"]
        metrics["overlap_ratio_vs_torch_serial"] = (serial_torch_ms - metrics["triton_total_ms"]) / max(
            serial_torch_ms, 1e-6)

        partial_nbytes = M * N * dtype.itemsize
        metrics["triton_ar_gbps"] = partial_nbytes / (1024**3) / metrics["triton_ar_only_ms"] * 1e3
        metrics["torch_ar_gbps"] = partial_nbytes / (1024**3) / metrics["torch_ar_only_ms"] * 1e3

        dist_print(
            f"Rank {rank} [{model_name}] latency (ms): "
            f"triton_total={metrics['triton_total_ms']:.4f}, "
            f"triton_gemm_only={metrics['triton_gemm_only_ms']:.4f}, "
            f"triton_ar_only={metrics['triton_ar_only_ms']:.4f}, "
            f"torch_total={metrics['torch_total_ms']:.4f}, "
            f"torch_gemm_only={metrics['torch_gemm_only_ms']:.4f}, "
            f"torch_ar_only={metrics['torch_ar_only_ms']:.4f}, "
            f"speedup={metrics['speedup_vs_torch']:.4f}, "
            f"overlap_ratio={metrics['overlap_ratio_vs_torch_serial']:.4f}",
            need_sync=True,
            allowed_ranks=list(range(world_size)),
        )

        dist_print(
            f"Rank {rank} [{model_name}] allreduce BW (GB/s): "
            f"triton={metrics['triton_ar_gbps']:.4f}, torch={metrics['torch_ar_gbps']:.4f}",
            need_sync=True,
            allowed_ranks=list(range(world_size)),
        )

        return metrics
    finally:
        gemm_ar.finalize()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    args = parse_args()

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    TP_GROUP = initialize_distributed()
    LOCAL_WORLD_SIZE = int(os.environ.get("LOCAL_WORLD_SIZE", TP_GROUP.size()))

    configs = get_test_configs(args)
    perf_res = {}

    for model_name, config in configs.items():
        perf_res[model_name] = perf_test(model_name, args.M, config, TP_GROUP)

    if args.dump_csv and TP_GROUP.rank() == 0:
        csv_dir = Path("csv")
        csv_dir.mkdir(exist_ok=True)
        csv_file = csv_dir / f"perf_gemm_allreduce_{TP_GROUP.size()}_ranks.csv"
        with open(csv_file, "w", encoding="utf-8") as fout:
            print(
                ",".join([
                    "Model",
                    "M",
                    "N",
                    "K",
                    "triton_total_ms",
                    "triton_gemm_only_ms",
                    "triton_ar_only_ms",
                    "torch_total_ms",
                    "torch_gemm_only_ms",
                    "torch_ar_only_ms",
                    "speedup_vs_torch",
                    "overlap_ratio_vs_torch_serial",
                ]),
                file=fout,
            )
            for model_name, config in configs.items():
                metrics = perf_res[model_name]
                if metrics.get("skipped", False):
                    print(
                        ",".join([
                            model_name,
                            str(args.M),
                            str(config["N"]),
                            str(config["K"]),
                            "nan",
                            "nan",
                            "nan",
                            "nan",
                            "nan",
                            "nan",
                            "nan",
                            "nan",
                        ]),
                        file=fout,
                        flush=True,
                    )
                    continue
                print(
                    ",".join([
                        model_name,
                        str(args.M),
                        str(config["N"]),
                        str(config["K"]),
                        f"{metrics['triton_total_ms']:.6f}",
                        f"{metrics['triton_gemm_only_ms']:.6f}",
                        f"{metrics['triton_ar_only_ms']:.6f}",
                        f"{metrics['torch_total_ms']:.6f}",
                        f"{metrics['torch_gemm_only_ms']:.6f}",
                        f"{metrics['torch_ar_only_ms']:.6f}",
                        f"{metrics['speedup_vs_torch']:.6f}",
                        f"{metrics['overlap_ratio_vs_torch_serial']:.6f}",
                    ]),
                    file=fout,
                    flush=True,
                )
        print(f"csv file is dumped into {csv_file}")

    finalize_distributed()
