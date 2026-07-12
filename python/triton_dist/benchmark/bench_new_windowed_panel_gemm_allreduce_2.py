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
import triton_dist.tune

from triton_dist.layers.nvidia import GemmARLayer
from triton_dist.kernels.nvidia import (create_frontier_windowed_panel_gemm_ar_context,
                                        frontier_windowed_panel_allreduce,
                                        frontier_windowed_panel_gemm_allreduce,
                                        frontier_windowed_panel_gemm_allreduce_op)
from triton_dist.kernels.nvidia.gemm import get_config_space
from triton_dist.profiler_utils import group_profile, perf_func
from triton_dist.test.utils import LAYER_CONFIGS, assert_allclose
from triton_dist.utils import (dist_print, finalize_distributed, initialize_distributed, nvshmem_barrier_all_on_stream,
                               rand_tensor, sleep_async, wait_until_max_gpu_clock_or_warning)
# torchrun --nproc_per_node=4 python/triton_dist/benchmark/bench_new_windowed_panel_gemm_allreduce.py --M 8192 --N 11008 --K 4096 --dtype bfloat16 --chunk_rows 1024 --active_chunk_window 2 --n_bands 1 --frontier_chunks 1 --stage_slots 4

AUTOTUNE_CACHE: dict[tuple, dict] = {}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--M", type=int, default=8192)
    parser.add_argument("--N", type=int, default=None)
    parser.add_argument("--K", type=int, default=None)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--warmup_iters", type=int, default=5)
    parser.add_argument("--autotune", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--profile", action="store_true", default=False)
    parser.add_argument("--dump_csv", action="store_true", default=False)
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16"])
    parser.add_argument("--check", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--run_baseline", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--baseline_num_comm_sms", type=int, default=16)
    parser.add_argument("--baseline_row_wise", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--baseline_low_latency", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--chunk_rows", type=int, default=0)
    parser.add_argument("--target_chunks", type=int, default=4)
    parser.add_argument("--min_chunk_rows", type=int, default=512)
    parser.add_argument("--active_chunk_window", type=int, default=2)
    parser.add_argument("--n_bands", type=int, default=1)
    parser.add_argument("--frontier_chunks", type=int, default=1)
    parser.add_argument("--stage_slots", type=int, default=4)
    parser.add_argument("--num_comm_sms", type=int, default=16)
    parser.add_argument("--streaming_depth", type=int, default=1)
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


def torch_gemm_ar(a: torch.Tensor, weight: torch.Tensor, tp_group: torch.distributed.ProcessGroup):
    output = torch.matmul(a, weight.T)
    torch.distributed.all_reduce(output, group=tp_group)
    return output


def sync_all(pg: torch.distributed.ProcessGroup):
    nvshmem_barrier_all_on_stream(torch.cuda.current_stream())
    torch.cuda.synchronize()
    torch.distributed.barrier(pg)


def choose_gemm_config():
    return get_config_space(False)[0]


def get_autotuned_gemm_ar_config(A: torch.Tensor, B: torch.Tensor, ctx, pg: torch.distributed.ProcessGroup):
    base_key = frontier_windowed_panel_gemm_allreduce.key_fn(A, B, ctx)
    cache_key = (base_key, "frontier_windowed_panel_gemm_allreduce")
    best_config = AUTOTUNE_CACHE.get(cache_key)
    if best_config is None:
        config_space = frontier_windowed_panel_gemm_allreduce.get_pruned_config(A, B, ctx)
        timings = frontier_windowed_panel_gemm_allreduce.tune(config_space, pg, A, B, ctx)
        timings.sort(key=lambda x: x[0])
        assert len(timings) > 0, "autotune returned empty timing list"
        best_config = timings[0][1]
        AUTOTUNE_CACHE[cache_key] = best_config
    return best_config["gemm_config"]


def create_baseline_layer(pg, M, N, K):
    return GemmARLayer(
        pg,
        M,
        N,
        K,
        dtype,
        dtype,
        LOCAL_WORLD_SIZE,
        persistent=False,
        use_ll_kernel=args.baseline_low_latency,
        copy_to_local=True,
        NUM_COMM_SMS=args.baseline_num_comm_sms,
        TILE_MAP_LEVEL=int(args.baseline_row_wise),
    )


def create_new_ctx(M, N):
    return create_frontier_windowed_panel_gemm_ar_context(
        M,
        N,
        RANK,
        WORLD_SIZE,
        LOCAL_WORLD_SIZE,
        dtype,
        chunk_rows=args.chunk_rows,
        target_chunks=args.target_chunks,
        min_chunk_rows=args.min_chunk_rows,
        active_chunk_window=args.active_chunk_window,
        n_bands=args.n_bands,
        frontier_chunks=args.frontier_chunks,
        stage_slots=args.stage_slots,
        num_comm_sms=args.num_comm_sms,
    )


def perf_test(model_name: str, M: int, config, pg: torch.distributed.ProcessGroup):
    N = config["N"]
    K = config["K"]
    rank = pg.rank()
    world_size = pg.size()
    if rank == 0:
        print(f"[{model_name}] test shape: M={M}, N={N}, K={K}")

    a, weight = make_data(M, N, K, dtype, pg)
    partial = torch.matmul(a, weight.T)
    gemm_config = choose_gemm_config()

    baseline = None
    baseline_status = "disabled"
    if args.run_baseline:
        try:
            baseline = create_baseline_layer(pg, M, N, K)
            baseline_status = "ok"
        except Exception as exc:
            if "Failed to allocate memory" in str(exc):
                baseline_status = "oom"
                dist_print(
                    f"Rank {rank} [{model_name}] baseline skipped: NVSHMEM symmetric allocation failed",
                    need_sync=True,
                    allowed_ranks=list(range(world_size)),
                )
            else:
                raise

    try:
        new_ctx = create_new_ctx(M, N)
    except Exception as exc:
        if "Failed to allocate memory" in str(exc):
            dist_print(
                f"Rank {rank} [{model_name}] new kernel skipped: NVSHMEM symmetric allocation failed",
                need_sync=True,
                allowed_ranks=list(range(world_size)),
            )
            torch.cuda.empty_cache()
            return {
                "skipped": True,
                "reason": "nvshmem_oom",
            }
        raise

    runtime_gemm_config = get_autotuned_gemm_ar_config(a, weight.T, new_ctx, pg) if args.autotune else gemm_config

    def _launch_new_total_once(*, drain: bool):
        return frontier_windowed_panel_gemm_allreduce_op(a, weight.T, new_ctx, runtime_gemm_config, drain=drain)

    def _launch_new_ar_once(*, drain: bool):
        return frontier_windowed_panel_allreduce(partial, new_ctx, drain=drain)

    def _torch_total():
        return torch_gemm_ar(a, weight, pg)

    def _torch_gemm_only():
        return torch.matmul(a, weight.T)

    def _torch_ar_only():
        output = partial.clone()
        torch.distributed.all_reduce(output, group=pg)
        return output

    def _new_total():
        return _launch_new_total_once(drain=True)

    def _new_ar_only():
        return _launch_new_ar_once(drain=True)

    def _new_total_streaming():
        last_output = None
        for launch_id in range(args.streaming_depth):
            last_output = _launch_new_total_once(drain=launch_id == args.streaming_depth - 1)
        return last_output

    def _new_ar_only_streaming():
        last_output = None
        for launch_id in range(args.streaming_depth):
            last_output = _launch_new_ar_once(drain=launch_id == args.streaming_depth - 1)
        return last_output

    def _baseline_total():
        return baseline.forward(a, weight)

    def _baseline_gemm_only():
        return baseline.forward_gemm(a, weight)

    def _baseline_ar_only():
        return baseline.forward_ar(partial)

    atol = 6e-2 if dtype == torch.bfloat16 else 1e-2
    rtol = atol
    metrics = {
        "baseline_status": baseline_status,
        "new_status": "ok",
        "streaming_depth": args.streaming_depth,
    }

    try:
        for _ in range(3):
            sync_all(pg)
            new_out = _new_total()
            if baseline is not None:
                sync_all(pg)
                baseline_out = _baseline_total()

        if args.check:
            sync_all(pg)
            torch_out = _torch_total()
            for i in range(world_size):
                torch.distributed.barrier(pg)
                if rank == i:
                    assert_allclose(torch_out, new_out, atol=atol, rtol=rtol)
                    if baseline is not None:
                        assert_allclose(torch_out, baseline_out, atol=atol, rtol=rtol)

        run_id = os.environ.get("TORCHELASTIC_RUN_ID", "local")
        with group_profile(f"new_gemm_ar_perf_m_{M}_n_{N}_k_{K}_{run_id}", args.profile, group=TP_GROUP):
            sync_all(pg)
            sleep_async(100)
            perf_func(_new_total_streaming, iters=args.iters, warmup_iters=args.warmup_iters)
            if baseline is not None:
                sync_all(pg)
                sleep_async(100)
                perf_func(_baseline_total, iters=args.iters, warmup_iters=args.warmup_iters)
            sync_all(pg)
            sleep_async(100)
            perf_func(_torch_total, iters=args.iters, warmup_iters=args.warmup_iters)

        sync_all(pg)
        sleep_async(100)
        wait_until_max_gpu_clock_or_warning(torch.cuda.current_device())
        _, metrics["new_total_ms"] = perf_func(_new_total_streaming, iters=args.iters, warmup_iters=args.warmup_iters)
        metrics["new_total_ms"] /= args.streaming_depth

        sync_all(pg)
        sleep_async(100)
        wait_until_max_gpu_clock_or_warning(torch.cuda.current_device())
        _, metrics["new_ar_only_ms"] = perf_func(_new_ar_only_streaming, iters=args.iters, warmup_iters=args.warmup_iters)
        metrics["new_ar_only_ms"] /= args.streaming_depth

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

        if baseline is not None:
            sync_all(pg)
            sleep_async(100)
            wait_until_max_gpu_clock_or_warning(torch.cuda.current_device())
            _, metrics["baseline_total_ms"] = perf_func(_baseline_total, iters=args.iters, warmup_iters=args.warmup_iters)

            sync_all(pg)
            wait_until_max_gpu_clock_or_warning(torch.cuda.current_device())
            _, metrics["baseline_gemm_only_ms"] = perf_func(_baseline_gemm_only,
                                                            iters=args.iters,
                                                            warmup_iters=args.warmup_iters)

            sync_all(pg)
            wait_until_max_gpu_clock_or_warning(torch.cuda.current_device())
            _, metrics["baseline_ar_only_ms"] = perf_func(_baseline_ar_only,
                                                          iters=args.iters,
                                                          warmup_iters=args.warmup_iters)
        else:
            metrics["baseline_total_ms"] = float("nan")
            metrics["baseline_gemm_only_ms"] = float("nan")
            metrics["baseline_ar_only_ms"] = float("nan")

        serial_torch_ms = metrics["torch_gemm_only_ms"] + metrics["torch_ar_only_ms"]
        metrics["new_speedup_vs_torch"] = metrics["torch_total_ms"] / metrics["new_total_ms"]
        metrics["new_overlap_ratio_vs_torch_serial"] = (serial_torch_ms - metrics["new_total_ms"]) / max(
            serial_torch_ms, 1e-6)

        if baseline is not None:
            metrics["new_speedup_vs_baseline"] = metrics["baseline_total_ms"] / metrics["new_total_ms"]
        else:
            metrics["new_speedup_vs_baseline"] = float("nan")

        partial_nbytes = M * N * dtype.itemsize
        metrics["new_ar_gbps"] = partial_nbytes / (1024**3) / metrics["new_ar_only_ms"] * 1e3
        metrics["torch_ar_gbps"] = partial_nbytes / (1024**3) / metrics["torch_ar_only_ms"] * 1e3
        if baseline is not None:
            metrics["baseline_ar_gbps"] = partial_nbytes / (1024**3) / metrics["baseline_ar_only_ms"] * 1e3
        else:
            metrics["baseline_ar_gbps"] = float("nan")

        dist_print(
            f"Rank {rank} [{model_name}] new latency (ms): "
            f"total={metrics['new_total_ms']:.4f}, ar_only={metrics['new_ar_only_ms']:.4f}, "
            f"speedup_vs_torch={metrics['new_speedup_vs_torch']:.4f}, "
            f"speedup_vs_baseline={metrics['new_speedup_vs_baseline']:.4f}, "
            f"overlap_ratio={metrics['new_overlap_ratio_vs_torch_serial']:.4f}, "
            f"streaming_depth={metrics['streaming_depth']}",
            need_sync=True,
            allowed_ranks=list(range(world_size)),
        )
        if baseline is not None:
            dist_print(
                f"Rank {rank} [{model_name}] baseline latency (ms): "
                f"total={metrics['baseline_total_ms']:.4f}, "
                f"gemm_only={metrics['baseline_gemm_only_ms']:.4f}, "
                f"ar_only={metrics['baseline_ar_only_ms']:.4f}",
                need_sync=True,
                allowed_ranks=list(range(world_size)),
            )
        dist_print(
            f"Rank {rank} [{model_name}] torch latency (ms): "
            f"total={metrics['torch_total_ms']:.4f}, "
            f"gemm_only={metrics['torch_gemm_only_ms']:.4f}, "
            f"ar_only={metrics['torch_ar_only_ms']:.4f}",
            need_sync=True,
            allowed_ranks=list(range(world_size)),
        )
        dist_print(
            f"Rank {rank} [{model_name}] allreduce BW (GB/s): "
            f"new={metrics['new_ar_gbps']:.4f}, baseline={metrics['baseline_ar_gbps']:.4f}, torch={metrics['torch_ar_gbps']:.4f}",
            need_sync=True,
            allowed_ranks=list(range(world_size)),
        )

        return metrics
    finally:
        new_ctx.finalize()
        if baseline is not None:
            baseline.finalize()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    args = parse_args()
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    TP_GROUP = initialize_distributed()
    RANK = int(os.environ.get("RANK", 0))
    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
    LOCAL_WORLD_SIZE = int(os.environ.get("LOCAL_WORLD_SIZE", TP_GROUP.size()))

    configs = get_test_configs(args)
    perf_res = {}
    if RANK == 0:
        print(
            "new kernel params:",
            {
                "chunk_rows": args.chunk_rows,
                "target_chunks": args.target_chunks,
                "min_chunk_rows": args.min_chunk_rows,
                "active_chunk_window": args.active_chunk_window,
                "n_bands": args.n_bands,
                "frontier_chunks": args.frontier_chunks,
                "stage_slots": args.stage_slots,
                "num_comm_sms": args.num_comm_sms,
                "autotune": args.autotune,
                "streaming_depth": args.streaming_depth,
            },
            flush=True,
        )

    for model_name, config in configs.items():
        perf_res[model_name] = perf_test(model_name, args.M, config, TP_GROUP)

    if args.dump_csv and TP_GROUP.rank() == 0:
        csv_dir = Path("csv")
        csv_dir.mkdir(exist_ok=True)
        csv_file = csv_dir / f"perf_new_windowed_panel_gemm_allreduce_{TP_GROUP.size()}_ranks.csv"
        with open(csv_file, "w", encoding="utf-8") as fout:
            print(
                ",".join([
                    "Model",
                    "M",
                    "N",
                    "K",
                    "baseline_status",
                    "new_total_ms",
                    "new_ar_only_ms",
                    "baseline_total_ms",
                    "baseline_gemm_only_ms",
                    "baseline_ar_only_ms",
                    "torch_total_ms",
                    "torch_gemm_only_ms",
                    "torch_ar_only_ms",
                    "new_speedup_vs_torch",
                    "new_speedup_vs_baseline",
                    "new_overlap_ratio_vs_torch_serial",
                ]),
                file=fout,
            )
            for model_name, config in configs.items():
                m = perf_res[model_name]
                if m.get("skipped", False):
                    print(
                        ",".join([
                            model_name,
                            str(args.M),
                            str(config["N"]),
                            str(config["K"]),
                            "skipped",
                            "nan",
                            "nan",
                            "nan",
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
                        str(m["baseline_status"]),
                        f"{m['new_total_ms']:.6f}",
                        f"{m['new_ar_only_ms']:.6f}",
                        f"{m['baseline_total_ms']:.6f}",
                        f"{m['baseline_gemm_only_ms']:.6f}",
                        f"{m['baseline_ar_only_ms']:.6f}",
                        f"{m['torch_total_ms']:.6f}",
                        f"{m['torch_gemm_only_ms']:.6f}",
                        f"{m['torch_ar_only_ms']:.6f}",
                        f"{m['new_speedup_vs_torch']:.6f}",
                        f"{m['new_speedup_vs_baseline']:.6f}",
                        f"{m['new_overlap_ratio_vs_torch_serial']:.6f}",
                    ]),
                    file=fout,
                    flush=True,
                )
        print(f"csv file is dumped into {csv_file}")

    finalize_distributed()
