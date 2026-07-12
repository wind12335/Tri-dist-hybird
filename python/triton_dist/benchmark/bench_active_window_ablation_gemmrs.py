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

"""Active-window ablation benchmark for RS-GEMM.

This benchmark keeps the final v5 frontier/windowed RS operator family fixed and
ablates only the bounded active-window mechanism:

- `with_active_window`: use the requested `active_chunk_window`
- `unbounded_window`: set `active_chunk_window = num_chunks`, which removes
  slot reuse and lets the staging footprint grow with the full logical workset

This is intentionally different from comparing against the original RS
benchmark, because the goal here is a mechanism ablation rather than a full
system baseline comparison.
"""

import argparse
import gc
import math
import os
from pathlib import Path
from typing import Callable, Dict, Tuple

import torch
import torch.distributed
import triton

from triton_dist.kernels.nvidia.gemm import get_config_space
from triton_dist.kernels.nvidia.new_3rd_v5_frontier_windowed_panel_rsgemm import (
    create_new_3rd_v5_frontier_windowed_panel_gemm_rs_context,
    launch_v5_frontier_panelized_gemm_producer,
    new_3rd_v5_frontier_windowed_panel_gemm_rs,
)
from triton_dist.kernels.nvidia.new_3rd_v5_frontier_windowed_panel_rs import new_3rd_v3_windowed_panel_rs_op
from triton_dist.profiler_utils import group_profile
from triton_dist.test.utils import LAYER_CONFIGS, assert_allclose
from triton_dist.utils import (NVSHMEM_SIGNAL_DTYPE, dist_print, finalize_distributed, initialize_distributed,
                               nvshmem_barrier_all_on_stream, rand_tensor, wait_until_max_gpu_clock_or_warning)


WINDOWED_AUTOTUNE_CACHE: dict[Tuple, dict] = {}
UNBOUNDED_AUTOTUNE_CACHE: dict[Tuple, dict] = {}


def debug_log(msg: str, rank: int | None = None) -> None:
    if args.debug:
        prefix = f"[bench-active-window][rank{rank}] " if rank is not None else "[bench-active-window] "
        print(prefix + msg, flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shapes",
                        type=str,
                        default=None,
                        help="Comma-separated MxNxK shapes, e.g. 8192x29568x8192,8192x49152x12288")
    parser.add_argument("--M", type=int, default=8192)
    parser.add_argument("--N", type=int, default=None)
    parser.add_argument("--K", type=int, default=None)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--warmup_iters", type=int, default=5)
    parser.add_argument("--autotune", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--profile", action="store_true", default=False)
    parser.add_argument("--profile_target",
                        type=str,
                        default="all",
                        choices=["all", "with_active_window", "unbounded_window", "torch"])
    parser.add_argument("--profile_merge_group", default=None, action=argparse.BooleanOptionalAction)
    parser.add_argument("--profile_with_stack", default=None, action=argparse.BooleanOptionalAction)
    parser.add_argument("--profile_barrier_after_merge", default=None, action=argparse.BooleanOptionalAction)
    parser.add_argument("--dump_csv", action="store_true", default=False)
    parser.add_argument("--output_csv", type=str, default=None)
    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    parser.add_argument("--trans_b", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--persistent", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--chunk_rows", type=int, default=0)
    parser.add_argument("--target_chunks_per_rank", type=int, default=2)
    parser.add_argument("--min_chunk_rows", type=int, default=512)
    parser.add_argument("--active_chunk_window", type=int, default=4)
    parser.add_argument("--stage_slots", type=int, default=4)
    parser.add_argument("--steady_sms", type=int, default=6)
    parser.add_argument("--tail_sms", type=int, default=12)
    parser.add_argument("--tail_chunk_window", type=int, default=1)
    parser.add_argument("--comm_lanes", type=int, default=2)
    parser.add_argument("--n_bands", type=int, default=1)
    parser.add_argument("--frontier_chunks", type=int, default=1)
    parser.add_argument("--local_seed_direct", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--window_policy",
                        type=str,
                        default="both",
                        choices=["with_active_window", "unbounded_window", "both"])
    return parser.parse_args()


def parse_shape_list(text: str) -> list[tuple[int, int, int]]:
    shapes: list[tuple[int, int, int]] = []
    for raw in text.split(","):
        raw = raw.strip().lower().replace(" ", "")
        if not raw:
            continue
        parts = raw.split("x")
        if len(parts) != 3:
            raise ValueError(f"invalid shape '{raw}', expected MxNxK")
        M, N, K = map(int, parts)
        shapes.append((M, N, K))
    if not shapes:
        raise ValueError("no valid shapes provided via --shapes")
    return shapes


def get_test_configs(parsed_args):
    if parsed_args.shapes is not None:
        configs = {}
        for M, N, K in parse_shape_list(parsed_args.shapes):
            label = f"{M}x{N}x{K}"
            configs[label] = {"M": M, "N": N, "K": K}
        return configs
    if parsed_args.N is not None or parsed_args.K is not None:
        if parsed_args.N is None or parsed_args.K is None:
            raise ValueError("`--N` and `--K` must be set together.")
        return {"custom": {"M": parsed_args.M, "N": parsed_args.N, "K": parsed_args.K}}
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
    torch.distributed.barrier(pg, device_ids=[torch.cuda.current_device()])


def release_python_cuda_refs() -> None:
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    try:
        torch.cuda.ipc_collect()
    except Exception:
        pass


def perf_func_lockstep(func: Callable[[], torch.Tensor], pg: torch.distributed.ProcessGroup, iters: int, warmup_iters: int):
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    stop_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    output = None
    for n in range(iters + warmup_iters):
        sync_all(pg)
        if n >= warmup_iters:
            start_events[n - warmup_iters].record()
        output = func()
        if n >= warmup_iters:
            stop_events[n - warmup_iters].record()
        sync_all(pg)
    duration_ms = 0.0
    for i in range(iters):
        stop_events[i].synchronize()
        duration_ms += start_events[i].elapsed_time(stop_events[i])
    return output, duration_ms / iters


def profile_section_lockstep(label: str,
                             enabled: bool,
                             fn: Callable[[], torch.Tensor],
                             pg: torch.distributed.ProcessGroup,
                             iters: int,
                             warmup_iters: int) -> None:
    if not enabled:
        return
    with torch.profiler.record_function(label):
        perf_func_lockstep(fn, pg=pg, iters=iters, warmup_iters=warmup_iters)


def choose_gemm_config() -> triton.Config:
    return get_config_space(False)[0]


def get_autotuned_config(cache: dict[Tuple, dict], A: torch.Tensor, B: torch.Tensor, ctx, pg: torch.distributed.ProcessGroup):
    base_key = new_3rd_v5_frontier_windowed_panel_gemm_rs.key_fn(A, B, ctx, persistent=False)
    cache_key = (base_key, False)
    best_config = cache.get(cache_key)
    if best_config is None:
        config_space = new_3rd_v5_frontier_windowed_panel_gemm_rs.get_pruned_config(A, B, ctx, persistent=False)
        timings = new_3rd_v5_frontier_windowed_panel_gemm_rs.tune(config_space, pg, A, B, ctx, persistent=False)
        timings.sort(key=lambda x: x[0])
        assert len(timings) > 0, "active-window autotune returned empty timing list"
        best_config = timings[0][1]
        cache[cache_key] = best_config
    return best_config["gemm_config"]


def compute_symmetric_staging_bytes(ctx) -> int:
    itemsize = torch.empty((), dtype=ctx.rs_ctx.dtype, device="cuda").element_size()
    signal_itemsize = torch.empty((), dtype=NVSHMEM_SIGNAL_DTYPE, device="cuda").element_size()
    rs_ctx = ctx.rs_ctx
    scatter_rows = rs_ctx.active_chunk_window * rs_ctx.n_bands * rs_ctx.local_world_size * rs_ctx.chunk_rows
    scatter_bytes = scatter_rows * rs_ctx.max_band_cols * itemsize
    arrival_flag_bytes = rs_ctx.local_world_size * rs_ctx.active_chunk_window * rs_ctx.n_bands * signal_itemsize
    free_flag_bytes = rs_ctx.active_chunk_window * rs_ctx.n_bands * signal_itemsize
    return scatter_bytes + arrival_flag_bytes + free_flag_bytes


def gib(nbytes: int) -> float:
    return nbytes / float(1024**3)


def run_stage(label: str,
              active_window_request: int,
              autotune_cache: dict[Tuple, dict],
              A: torch.Tensor,
              B: torch.Tensor,
              torch_partial: torch.Tensor,
              M: int,
              M_per_rank: int,
              pg: torch.distributed.ProcessGroup,
              atol: float,
              rtol: float,
              C_torch: torch.Tensor,
              metrics: dict[str, float]) -> None:
    rank = pg.rank()
    world_size = pg.size()
    local_world_size = LOCAL_WORLD_SIZE
    ctx = None
    try:
        sync_all(pg)
        ctx = create_new_3rd_v5_frontier_windowed_panel_gemm_rs_context(
            M,
            B.shape[1],
            rank,
            world_size,
            local_world_size,
            dtype,
            chunk_rows=args.chunk_rows,
            target_chunks_per_rank=args.target_chunks_per_rank,
            min_chunk_rows=args.min_chunk_rows,
            active_chunk_window=active_window_request,
            comm_lanes=args.comm_lanes,
            n_bands=args.n_bands,
            frontier_chunks=args.frontier_chunks,
            steady_sms=args.steady_sms,
            tail_sms=args.tail_sms,
            stage_slots=args.stage_slots,
            tail_chunk_window=args.tail_chunk_window,
            local_seed_direct=args.local_seed_direct,
        )
        metrics[f"{label}_chunk_rows"] = float(ctx.rs_ctx.chunk_rows)
        metrics[f"{label}_num_chunks"] = float(ctx.rs_ctx.num_chunks)
        metrics[f"{label}_active_chunk_window"] = float(ctx.rs_ctx.active_chunk_window)
        metrics[f"{label}_stage_slots"] = float(ctx.rs_ctx.stage_slots)
        metrics[f"{label}_comm_lanes"] = float(len(ctx.rs_ctx.comm_streams))
        metrics[f"{label}_n_bands"] = float(ctx.rs_ctx.n_bands)
        metrics[f"{label}_frontier_chunks"] = float(ctx.frontier_chunks)
        metrics[f"{label}_symmetric_staging_bytes"] = float(compute_symmetric_staging_bytes(ctx))
        metrics[f"{label}_symmetric_staging_gib"] = gib(int(metrics[f"{label}_symmetric_staging_bytes"]))

        workspace = torch.zeros((ctx.rs_ctx.n_bands * world_size * ctx.rs_ctx.num_chunks,), dtype=torch.int32, device=A.device)
        rs_output = torch.empty((M_per_rank, B.shape[1]), dtype=dtype, device=A.device)
        gemm_out = ctx.get_gemm_out_buf(A)

        def _get_config():
            if args.autotune:
                return get_autotuned_config(autotune_cache, A, B, ctx, pg)
            return choose_gemm_config()

        def _reset_runtime():
            ctx.rs_ctx.reset_runtime_state()
            sync_all(pg)

        def _total():
            return new_3rd_v5_frontier_windowed_panel_gemm_rs.fn(A, B, ctx, gemm_config=_get_config(), persistent=False)

        def _gemm_only():
            num_runtime_chunks = triton.cdiv(M_per_rank, ctx.rs_ctx.chunk_rows)
            signal_value = ctx.rs_ctx.begin_round(num_runtime_chunks)
            launch_v5_frontier_panelized_gemm_producer(A, B, gemm_out, ctx, workspace, signal_value, _get_config())
            return gemm_out

        def _rs_only():
            gemm_out.copy_(torch_partial)
            num_runtime_chunks = triton.cdiv(M_per_rank, ctx.rs_ctx.chunk_rows)
            signal_value = ctx.rs_ctx.begin_round(num_runtime_chunks)
            ctx.rs_ctx.chunk_signal.fill_(signal_value)
            return new_3rd_v3_windowed_panel_rs_op(gemm_out, ctx.rs_ctx, output=rs_output, prepare_round=False)

        _reset_runtime()
        C_stage = None
        for _ in range(3):
            sync_all(pg)
            C_stage = _total()
        for i in range(world_size):
            torch.distributed.barrier(pg, device_ids=[torch.cuda.current_device()])
            if rank == i and C_stage is not None:
                assert_allclose(C_torch, C_stage, atol=atol, rtol=rtol)

        target_name = "with_active_window" if label == "windowed" else "unbounded_window"
        if args.profile_target in ["all", target_name]:
            _reset_runtime()
            profile_section_lockstep(f"triton/{label}_total", args.profile, _total, pg, args.iters, args.warmup_iters)
            _reset_runtime()
            profile_section_lockstep(f"triton/{label}_gemm_only", args.profile, _gemm_only, pg, args.iters,
                                     args.warmup_iters)
            _reset_runtime()
            profile_section_lockstep(f"triton/{label}_rs_only", args.profile, _rs_only, pg, args.iters, args.warmup_iters)

        _reset_runtime()
        wait_until_max_gpu_clock_or_warning(torch.cuda.current_device())
        _, metrics[f"{label}_total_ms"] = perf_func_lockstep(_total, pg=pg, iters=args.iters, warmup_iters=args.warmup_iters)
        _reset_runtime()
        wait_until_max_gpu_clock_or_warning(torch.cuda.current_device())
        _, metrics[f"{label}_gemm_only_ms"] = perf_func_lockstep(_gemm_only,
                                                                 pg=pg,
                                                                 iters=args.iters,
                                                                 warmup_iters=args.warmup_iters)
        _reset_runtime()
        wait_until_max_gpu_clock_or_warning(torch.cuda.current_device())
        _, metrics[f"{label}_rs_only_ms"] = perf_func_lockstep(_rs_only,
                                                               pg=pg,
                                                               iters=args.iters,
                                                               warmup_iters=args.warmup_iters)
    finally:
        sync_all(pg)
        if ctx is not None:
            ctx.finalize()
        sync_all(pg)
        release_python_cuda_refs()
        sync_all(pg)


def perf_test(model_name: str, M: int, config: Dict[str, int], pg: torch.distributed.ProcessGroup):
    N = config["N"]
    K = config["K"]
    rank = pg.rank()
    world_size = pg.size()
    local_world_size = LOCAL_WORLD_SIZE

    if world_size != local_world_size:
        raise AssertionError("active-window ablation benchmark currently supports single-node runs only")
    if args.persistent:
        raise AssertionError("active-window ablation benchmark currently supports only --no-persistent")
    if rank == 0:
        print(f"[{model_name}] test shape: M {M}, N {N}, K {K}, window_policy={args.window_policy}")
    debug_log(f"perf_test enter: autotune={args.autotune}, window_policy={args.window_policy}", rank)

    assert M % world_size == 0
    assert K % world_size == 0

    A, B = make_data(M, N, K, dtype, args.trans_b, pg)
    M_per_rank = M // world_size
    atol = 6e-2 if dtype == torch.bfloat16 else 1e-2
    rtol = atol
    torch_partial = torch.matmul(A, B)

    metrics = {
        "torch_total_ms": float("nan"),
        "torch_gemm_only_ms": float("nan"),
        "torch_rs_only_ms": float("nan"),
        "windowed_total_ms": float("nan"),
        "windowed_gemm_only_ms": float("nan"),
        "windowed_rs_only_ms": float("nan"),
        "windowed_chunk_rows": float("nan"),
        "windowed_num_chunks": float("nan"),
        "windowed_active_chunk_window": float("nan"),
        "windowed_stage_slots": float("nan"),
        "windowed_comm_lanes": float("nan"),
        "windowed_n_bands": float("nan"),
        "windowed_frontier_chunks": float("nan"),
        "windowed_symmetric_staging_bytes": float("nan"),
        "windowed_symmetric_staging_gib": float("nan"),
        "unbounded_total_ms": float("nan"),
        "unbounded_gemm_only_ms": float("nan"),
        "unbounded_rs_only_ms": float("nan"),
        "unbounded_chunk_rows": float("nan"),
        "unbounded_num_chunks": float("nan"),
        "unbounded_active_chunk_window": float("nan"),
        "unbounded_stage_slots": float("nan"),
        "unbounded_comm_lanes": float("nan"),
        "unbounded_n_bands": float("nan"),
        "unbounded_frontier_chunks": float("nan"),
        "unbounded_symmetric_staging_bytes": float("nan"),
        "unbounded_symmetric_staging_gib": float("nan"),
    }

    def _torch_total():
        return torch_gemm_rs(pg, A, B)

    def _torch_gemm_only():
        return torch.matmul(A, B)

    def _torch_rs_only():
        output = torch.empty((M_per_rank, N), dtype=dtype, device=A.device)
        torch.distributed.reduce_scatter_tensor(output, torch_partial, group=pg)
        return output

    sync_all(pg)
    C_torch = _torch_total()

    effective_chunk_rows = args.chunk_rows
    if effective_chunk_rows <= 0:
        rows = math.ceil(M_per_rank / args.target_chunks_per_rank)
        rows = max(rows, args.min_chunk_rows)
        rows = min(rows, M_per_rank)
        rows = ((rows + 255) // 256) * 256
        effective_chunk_rows = min(rows, M_per_rank)
    num_chunks = triton.cdiv(M_per_rank, effective_chunk_rows)
    bounded_active_window = max(1, min(args.active_chunk_window, num_chunks))
    unbounded_active_window = num_chunks

    run_id = os.environ.get("TORCHELASTIC_RUN_ID", "local")
    merge_profile_group = args.profile_merge_group if args.profile_merge_group is not None else pg.size() <= 4
    profile_with_stack = args.profile_with_stack if args.profile_with_stack is not None else pg.size() <= 4
    profile_barrier_after_merge = (
        args.profile_barrier_after_merge if args.profile_barrier_after_merge is not None else pg.size() <= 4
    )
    try:
        with group_profile(f"active_window_ablation_gemm_rs_perf_m_{M}_n_{N}_k_{K}_{run_id}",
                           args.profile,
                           merge_group=merge_profile_group,
                           barrier_after_merge=profile_barrier_after_merge,
                           with_stack=profile_with_stack,
                           group=TP_GROUP):
            if args.window_policy in ["with_active_window", "both"]:
                run_stage("windowed",
                          bounded_active_window,
                          WINDOWED_AUTOTUNE_CACHE,
                          A,
                          B,
                          torch_partial,
                          M,
                          M_per_rank,
                          pg,
                          atol,
                          rtol,
                          C_torch,
                          metrics)
            if args.window_policy in ["unbounded_window", "both"]:
                run_stage("unbounded",
                          unbounded_active_window,
                          UNBOUNDED_AUTOTUNE_CACHE,
                          A,
                          B,
                          torch_partial,
                          M,
                          M_per_rank,
                          pg,
                          atol,
                          rtol,
                          C_torch,
                          metrics)
            if args.profile_target in ["all", "torch"]:
                sync_all(pg)
                profile_section_lockstep("torch/reference_total", args.profile, _torch_total, pg, args.iters,
                                         args.warmup_iters)
                sync_all(pg)
                profile_section_lockstep("torch/reference_gemm_only", args.profile, _torch_gemm_only, pg, args.iters,
                                         args.warmup_iters)
                sync_all(pg)
                profile_section_lockstep("torch/reference_rs_only", args.profile, _torch_rs_only, pg, args.iters,
                                         args.warmup_iters)

        sync_all(pg)
        wait_until_max_gpu_clock_or_warning(torch.cuda.current_device())
        _, metrics["torch_total_ms"] = perf_func_lockstep(_torch_total, pg=pg, iters=args.iters, warmup_iters=args.warmup_iters)
        sync_all(pg)
        wait_until_max_gpu_clock_or_warning(torch.cuda.current_device())
        _, metrics["torch_gemm_only_ms"] = perf_func_lockstep(_torch_gemm_only,
                                                              pg=pg,
                                                              iters=args.iters,
                                                              warmup_iters=args.warmup_iters)
        sync_all(pg)
        wait_until_max_gpu_clock_or_warning(torch.cuda.current_device())
        _, metrics["torch_rs_only_ms"] = perf_func_lockstep(_torch_rs_only,
                                                            pg=pg,
                                                            iters=args.iters,
                                                            warmup_iters=args.warmup_iters)
    finally:
        sync_all(pg)

    for label in ["windowed", "unbounded"]:
        if metrics[f"{label}_total_ms"] == metrics[f"{label}_total_ms"]:
            metrics[f"{label}_internal_overlap_ratio"] = 1.0 - metrics[f"{label}_total_ms"] / max(
                metrics[f"{label}_gemm_only_ms"] + metrics[f"{label}_rs_only_ms"], 1e-6)
            metrics[f"{label}_speedup_vs_torch"] = metrics["torch_total_ms"] / metrics[f"{label}_total_ms"]
        else:
            metrics[f"{label}_internal_overlap_ratio"] = float("nan")
            metrics[f"{label}_speedup_vs_torch"] = float("nan")

    if metrics["windowed_total_ms"] == metrics["windowed_total_ms"] and metrics["unbounded_total_ms"] == metrics[
            "unbounded_total_ms"]:
        metrics["window_speedup_vs_unbounded"] = metrics["unbounded_total_ms"] / metrics["windowed_total_ms"]
        metrics["window_latency_ratio_vs_unbounded"] = metrics["windowed_total_ms"] / metrics["unbounded_total_ms"]
        metrics["window_symmetric_ratio_vs_unbounded"] = metrics["windowed_symmetric_staging_bytes"] / max(
            metrics["unbounded_symmetric_staging_bytes"], 1.0)
    else:
        metrics["window_speedup_vs_unbounded"] = float("nan")
        metrics["window_latency_ratio_vs_unbounded"] = float("nan")
        metrics["window_symmetric_ratio_vs_unbounded"] = float("nan")

    msg = (
        f"Rank {rank} [{model_name}] latency (ms): "
        f"torch_total={metrics['torch_total_ms']:.2f}, "
        f"torch_gemm_only={metrics['torch_gemm_only_ms']:.2f}, "
        f"torch_rs_only={metrics['torch_rs_only_ms']:.2f}"
    )
    if args.window_policy in ["with_active_window", "both"]:
        msg += (
            f", with_window_total={metrics['windowed_total_ms']:.2f}, "
            f"with_window_gemm_only={metrics['windowed_gemm_only_ms']:.2f}, "
            f"with_window_rs_only={metrics['windowed_rs_only_ms']:.2f}, "
            f"with_window_internal_overlap={metrics['windowed_internal_overlap_ratio']:.2%}, "
            f"with_window_symm_gib={metrics['windowed_symmetric_staging_gib']:.3f}"
        )
    if args.window_policy in ["unbounded_window", "both"]:
        msg += (
            f", unbounded_total={metrics['unbounded_total_ms']:.2f}, "
            f"unbounded_gemm_only={metrics['unbounded_gemm_only_ms']:.2f}, "
            f"unbounded_rs_only={metrics['unbounded_rs_only_ms']:.2f}, "
            f"unbounded_internal_overlap={metrics['unbounded_internal_overlap_ratio']:.2%}, "
            f"unbounded_symm_gib={metrics['unbounded_symmetric_staging_gib']:.3f}"
        )
    if args.window_policy == "both":
        msg += (
            f", window_speedup_vs_unbounded={metrics['window_speedup_vs_unbounded']:.2f}, "
            f"window_symmetric_ratio_vs_unbounded={metrics['window_symmetric_ratio_vs_unbounded']:.2f}"
        )
    dist_print(msg, need_sync=True, allowed_ranks=list(range(world_size)))
    return metrics


if __name__ == "__main__":
    args = parse_args()
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]

    if args.persistent:
        raise AssertionError("persistent is not supported in this benchmark")

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    TP_GROUP = initialize_distributed()
    LOCAL_WORLD_SIZE = int(os.environ.get("LOCAL_WORLD_SIZE", TP_GROUP.size()))

    perf_res = []
    configs = get_test_configs(args)
    for model_name, config in configs.items():
        shape_M = config.get("M", args.M)
        metrics = perf_test(model_name, shape_M, config, TP_GROUP)
        perf_res.append((model_name, config, metrics))

    if args.dump_csv and TP_GROUP.rank() == 0:
        if args.output_csv is not None:
            csv_file = Path(args.output_csv)
            csv_file.parent.mkdir(parents=True, exist_ok=True)
        else:
            os.makedirs("csv", exist_ok=True)
            csv_file = Path("csv") / f"perf_active_window_ablation_gemm_rs_{TP_GROUP.size()}_ranks.csv"
        header = [
            "Model",
            "M",
            "N",
            "K",
            "torch_total_ms",
            "torch_gemm_only_ms",
            "torch_rs_only_ms",
            "windowed_total_ms",
            "windowed_gemm_only_ms",
            "windowed_rs_only_ms",
            "windowed_internal_overlap_ratio",
            "windowed_symmetric_staging_gib",
            "windowed_num_chunks",
            "windowed_active_chunk_window",
            "unbounded_total_ms",
            "unbounded_gemm_only_ms",
            "unbounded_rs_only_ms",
            "unbounded_internal_overlap_ratio",
            "unbounded_symmetric_staging_gib",
            "unbounded_num_chunks",
            "unbounded_active_chunk_window",
            "window_speedup_vs_unbounded",
            "window_latency_ratio_vs_unbounded",
            "window_symmetric_ratio_vs_unbounded",
        ]
        with open(csv_file, "w") as fout:
            print(",".join(header), file=fout)
            for model_name, config, metrics in perf_res:
                row = [
                    model_name,
                    str(config.get("M", args.M)),
                    str(config["N"]),
                    str(config["K"]),
                    f"{metrics['torch_total_ms']:.4f}",
                    f"{metrics['torch_gemm_only_ms']:.4f}",
                    f"{metrics['torch_rs_only_ms']:.4f}",
                    f"{metrics['windowed_total_ms']:.4f}",
                    f"{metrics['windowed_gemm_only_ms']:.4f}",
                    f"{metrics['windowed_rs_only_ms']:.4f}",
                    f"{metrics['windowed_internal_overlap_ratio']:.4f}",
                    f"{metrics['windowed_symmetric_staging_gib']:.6f}",
                    f"{metrics['windowed_num_chunks']:.0f}",
                    f"{metrics['windowed_active_chunk_window']:.0f}",
                    f"{metrics['unbounded_total_ms']:.4f}",
                    f"{metrics['unbounded_gemm_only_ms']:.4f}",
                    f"{metrics['unbounded_rs_only_ms']:.4f}",
                    f"{metrics['unbounded_internal_overlap_ratio']:.4f}",
                    f"{metrics['unbounded_symmetric_staging_gib']:.6f}",
                    f"{metrics['unbounded_num_chunks']:.0f}",
                    f"{metrics['unbounded_active_chunk_window']:.0f}",
                    f"{metrics['window_speedup_vs_unbounded']:.4f}",
                    f"{metrics['window_latency_ratio_vs_unbounded']:.4f}",
                    f"{metrics['window_symmetric_ratio_vs_unbounded']:.4f}",
                ]
                print(",".join(row), file=fout, flush=True)
        print(f"csv file is dumped into {csv_file}")

    finalize_distributed()
