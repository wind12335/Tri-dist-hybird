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

"""
Collect the real RS frontier-scheduling timeline events needed by
plot_rs_frontier_timeline.py.

For each policy, this script measures three GPU-timestamped events for the
first wake-up panel (chunk 0, band 0 on the local destination rank):

1. first_panel_ready_ts_ms
   Time from producer launch until the local wake-up panel is producer-ready,
   i.e. the local chunk-ready signal becomes visible.
2. first_consumer_start_ts_ms
   Time until the RS consumer stream actually starts the first reduction after
   all waits are satisfied.
3. first_output_commit_ts_ms
   Time until that first wake-up panel finishes reduction and commits output.

This is intentionally implemented as a separate collector benchmark so that the
original RS benchmarks remain unchanged.
"""

from __future__ import annotations

import argparse
import csv
import gc
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable

import torch
import torch.distributed
import triton

from triton_dist.kernels.nvidia.common_ops import _set_signal_cuda, _wait_eq_cuda
from triton_dist.kernels.nvidia.gemm import get_config_space
from triton_dist.kernels.nvidia.new_3rd_v3_windowed_panel_rs import (
    _arrival_flag_view,
    _band_col_range,
    _chunk_row_range,
    _chunk_signal_view,
    _free_flag_view_local,
    _issue_windowed_panel_scatter_and_arrival,
    _num_sms_or_default,
    _reduce_window_slot_from_scatter_direct,
    _slot_for_panel,
)
from triton_dist.kernels.nvidia.new_3rd_v3_windowed_panel_rsgemm import (
    create_new_3rd_v3_windowed_panel_gemm_rs_context,
    launch_v2_panelized_gemm_producer,
    new_3rd_v3_windowed_panel_gemm_rs,
)
from triton_dist.kernels.nvidia.new_3rd_v5_frontier_windowed_panel_rsgemm import (
    create_new_3rd_v5_frontier_windowed_panel_gemm_rs_context,
    launch_v5_frontier_panelized_gemm_producer,
    new_3rd_v5_frontier_windowed_panel_gemm_rs,
)
from triton_dist.test.utils import LAYER_CONFIGS, assert_allclose
from triton_dist.utils import (
    finalize_distributed,
    initialize_distributed,
    nvshmem_barrier_all_on_stream,
    rand_tensor,
    wait_until_max_gpu_clock_or_warning,
)


UNIFORM_AUTOTUNE_CACHE: dict[tuple, dict] = {}
FRONTIER_AUTOTUNE_CACHE: dict[tuple, dict] = {}
LOCAL_WORLD_SIZE = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--M", type=int, default=8192)
    parser.add_argument("--N", type=int, default=None)
    parser.add_argument("--K", type=int, default=None)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--warmup_iters", type=int, default=5)
    parser.add_argument("--autotune", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--plot", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--dump_csv", default=True, action=argparse.BooleanOptionalAction)
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
    parser.add_argument("--n_bands", type=int, default=2)
    parser.add_argument("--frontier_chunks", type=int, default=1)
    parser.add_argument("--local_seed_direct", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--output_dir", type=str, default="")
    return parser.parse_args()


def get_test_configs(parsed_args):
    if parsed_args.N is not None or parsed_args.K is not None:
        if parsed_args.N is None or parsed_args.K is None:
            raise ValueError("`--N` and `--K` must be set together.")
        return {"custom": {"N": parsed_args.N, "K": parsed_args.K}}
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
    partial = torch.matmul(A, B)
    output = torch.empty((M // pg.size(), B.shape[1]), dtype=partial.dtype, device=A.device)
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


def choose_gemm_config() -> triton.Config:
    return get_config_space(False)[0]


def get_autotuned_uniform_config(A: torch.Tensor, B: torch.Tensor, ctx, pg: torch.distributed.ProcessGroup):
    base_key = new_3rd_v3_windowed_panel_gemm_rs.key_fn(A, B, ctx, persistent=False)
    cache_key = (base_key, False)
    best_config = UNIFORM_AUTOTUNE_CACHE.get(cache_key)
    if best_config is None:
        config_space = new_3rd_v3_windowed_panel_gemm_rs.get_pruned_config(A, B, ctx, persistent=False)
        timings = new_3rd_v3_windowed_panel_gemm_rs.tune(config_space, pg, A, B, ctx, persistent=False)
        timings.sort(key=lambda x: x[0])
        assert len(timings) > 0, "uniform autotune returned empty timing list"
        best_config = timings[0][1]
        UNIFORM_AUTOTUNE_CACHE[cache_key] = best_config
    return best_config["gemm_config"]


def get_autotuned_frontier_config(A: torch.Tensor, B: torch.Tensor, ctx, pg: torch.distributed.ProcessGroup):
    base_key = new_3rd_v5_frontier_windowed_panel_gemm_rs.key_fn(A, B, ctx, persistent=False)
    cache_key = (base_key, False)
    best_config = FRONTIER_AUTOTUNE_CACHE.get(cache_key)
    if best_config is None:
        config_space = new_3rd_v5_frontier_windowed_panel_gemm_rs.get_pruned_config(A, B, ctx, persistent=False)
        timings = new_3rd_v5_frontier_windowed_panel_gemm_rs.tune(config_space, pg, A, B, ctx, persistent=False)
        timings.sort(key=lambda x: x[0])
        assert len(timings) > 0, "frontier autotune returned empty timing list"
        best_config = timings[0][1]
        FRONTIER_AUTOTUNE_CACHE[cache_key] = best_config
    return best_config["gemm_config"]


def _enqueue_windowed_chunk_recipe_with_events(
    input_intra_node: torch.Tensor,
    rs_ctx,
    output: torch.Tensor,
    chunk_id: int,
    band_id: int,
    num_runtime_chunks: int,
    *,
    first_consumer_start_event: torch.cuda.Event | None,
    first_output_commit_event: torch.cuda.Event | None,
) -> None:
    slot = _slot_for_panel(rs_ctx, chunk_id, band_id)
    stream = slot.stream
    slot.chunk_id_host = chunk_id * rs_ctx.n_bands + band_id

    m_per_rank = output.shape[0]
    row_start, row_end = _chunk_row_range(rs_ctx, chunk_id, m_per_rank)
    rows = row_end - row_start
    if rows <= 0:
        return

    col_start, col_end = _band_col_range(rs_ctx, band_id)
    band_cols = col_end - col_start
    if band_cols <= 0:
        return

    out_chunk = output[row_start:row_end, col_start:col_end]
    window_slot = chunk_id % rs_ctx.active_chunk_window
    ticket = rs_ctx.ticket_for_panel(chunk_id, band_id)
    local_src = None
    if rs_ctx.local_seed_direct:
        local_segment_start = rs_ctx.local_rank * m_per_rank + row_start
        local_segment_end = local_segment_start + rows
        local_src = input_intra_node[local_segment_start:local_segment_end, col_start:col_end]

    is_first_wakeup_panel = chunk_id == 0 and band_id == 0
    with torch.cuda.stream(stream):
        if rs_ctx.local_seed_direct:
            _wait_eq_cuda(_chunk_signal_view(rs_ctx, rs_ctx.local_rank, chunk_id, band_id), rs_ctx.signal_value, stream)
        for src_local_rank in range(rs_ctx.local_world_size):
            if rs_ctx.local_seed_direct and src_local_rank == rs_ctx.local_rank:
                continue
            _wait_eq_cuda(_arrival_flag_view(rs_ctx, src_local_rank, window_slot, band_id), ticket, stream)

        if is_first_wakeup_panel and first_consumer_start_event is not None:
            first_consumer_start_event.record(stream)

        from triton_dist.kernels.nvidia.new_3rd_v3_windowed_panel_rs import _window_slot_view

        scatter_slot = _window_slot_view(rs_ctx, window_slot, band_id, band_cols)
        use_tail_budget = chunk_id >= max(0, num_runtime_chunks - rs_ctx.tail_chunk_window)
        sms = _num_sms_or_default(rs_ctx.tail_sms if use_tail_budget else rs_ctx.steady_sms)
        _reduce_window_slot_from_scatter_direct(
            scatter_slot,
            local_src,
            out_chunk,
            slot_rows=rs_ctx.chunk_rows,
            local_rank=rs_ctx.local_rank,
            num_splits=rs_ctx.local_world_size,
            rows_to_reduce=rows,
            num_sms=sms,
        )
        _set_signal_cuda(_free_flag_view_local(rs_ctx, window_slot, band_id), ticket, stream)
        if is_first_wakeup_panel and first_output_commit_event is not None:
            first_output_commit_event.record(stream)
        slot.done_event.record(stream)


def run_instrumented_rs_consumer(
    input_intra_node: torch.Tensor,
    rs_ctx,
    output: torch.Tensor,
    num_runtime_chunks: int,
    *,
    first_consumer_start_event: torch.cuda.Event | None,
    first_output_commit_event: torch.cuda.Event | None,
) -> torch.Tensor:
    for chunk_id in range(num_runtime_chunks):
        for band_id in range(rs_ctx.n_bands):
            _issue_windowed_panel_scatter_and_arrival(input_intra_node, rs_ctx, chunk_id, band_id)
            _enqueue_windowed_chunk_recipe_with_events(
                input_intra_node,
                rs_ctx,
                output,
                chunk_id,
                band_id,
                num_runtime_chunks,
                first_consumer_start_event=first_consumer_start_event,
                first_output_commit_event=first_output_commit_event,
            )
    rs_ctx.wait_all(torch.cuda.current_stream())
    return output


def average_across_ranks(pg: torch.distributed.ProcessGroup, values_ms: list[float]) -> list[float]:
    tensor = torch.tensor(values_ms, device="cuda", dtype=torch.float64)
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM, group=pg)
    tensor /= pg.size()
    return [float(x) for x in tensor.cpu().tolist()]


def collect_policy_timeline(
    *,
    policy: str,
    A: torch.Tensor,
    B: torch.Tensor,
    M: int,
    dtype: torch.dtype,
    pg: torch.distributed.ProcessGroup,
    args,
) -> dict[str, float]:
    rank = pg.rank()
    world_size = pg.size()
    local_world_size = LOCAL_WORLD_SIZE
    ctx = None
    try:
        if policy == "uniform":
            ctx = create_new_3rd_v3_windowed_panel_gemm_rs_context(
                M,
                B.shape[1],
                rank,
                world_size,
                local_world_size,
                dtype,
                chunk_rows=args.chunk_rows,
                target_chunks_per_rank=args.target_chunks_per_rank,
                min_chunk_rows=args.min_chunk_rows,
                active_chunk_window=args.active_chunk_window,
                comm_lanes=args.comm_lanes,
                n_bands=args.n_bands,
                steady_sms=args.steady_sms,
                tail_sms=args.tail_sms,
                stage_slots=args.stage_slots,
                tail_chunk_window=args.tail_chunk_window,
                local_seed_direct=args.local_seed_direct,
            )
            get_config: Callable[[], triton.Config]
            get_config = lambda: (
                get_autotuned_uniform_config(A, B, ctx, pg) if args.autotune else choose_gemm_config()
            )
            launch_producer = launch_v2_panelized_gemm_producer
        elif policy == "frontier_first":
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
                active_chunk_window=args.active_chunk_window,
                comm_lanes=args.comm_lanes,
                n_bands=args.n_bands,
                frontier_chunks=args.frontier_chunks,
                steady_sms=args.steady_sms,
                tail_sms=args.tail_sms,
                stage_slots=args.stage_slots,
                tail_chunk_window=args.tail_chunk_window,
                local_seed_direct=args.local_seed_direct,
            )
            get_config = lambda: (
                get_autotuned_frontier_config(A, B, ctx, pg) if args.autotune else choose_gemm_config()
            )
            launch_producer = launch_v5_frontier_panelized_gemm_producer
        else:
            raise ValueError(f"unsupported policy: {policy}")

        M_per_rank = M // world_size
        torch_ref = torch_gemm_rs(pg, A, B)
        atol = 6e-2 if dtype == torch.bfloat16 else 1e-2
        rtol = atol
        workspace = torch.zeros((ctx.rs_ctx.n_bands * world_size * ctx.rs_ctx.num_chunks,), dtype=torch.int32, device=A.device)
        gemm_out = ctx.get_gemm_out_buf(A)
        output = torch.empty((M_per_rank, B.shape[1]), dtype=dtype, device=A.device)
        gemm_config = get_config()

        sync_all(pg)
        ctx.rs_ctx.reset_runtime_state()
        num_runtime_chunks = triton.cdiv(M_per_rank, ctx.rs_ctx.chunk_rows)
        signal_value = ctx.rs_ctx.begin_round(num_runtime_chunks)
        launch_producer(A, B, gemm_out, ctx, workspace, signal_value, gemm_config)
        out = run_instrumented_rs_consumer(
            gemm_out,
            ctx.rs_ctx,
            output,
            num_runtime_chunks,
            first_consumer_start_event=None,
            first_output_commit_event=None,
        )
        sync_all(pg)
        for i in range(world_size):
            torch.distributed.barrier(pg, device_ids=[torch.cuda.current_device()])
            if rank == i:
                assert_allclose(torch_ref, out, atol=atol, rtol=rtol)

        wait_until_max_gpu_clock_or_warning(torch.cuda.current_device())
        ready_times = []
        consumer_times = []
        commit_times = []
        for step in range(args.warmup_iters + args.iters):
            sync_all(pg)
            ctx.rs_ctx.reset_runtime_state()

            num_runtime_chunks = triton.cdiv(M_per_rank, ctx.rs_ctx.chunk_rows)
            signal_value = ctx.rs_ctx.begin_round(num_runtime_chunks)

            launch_event = torch.cuda.Event(enable_timing=True)
            first_panel_ready_event = torch.cuda.Event(enable_timing=True)
            first_consumer_start_event = torch.cuda.Event(enable_timing=True)
            first_output_commit_event = torch.cuda.Event(enable_timing=True)
            ready_watch_stream = torch.cuda.Stream(priority=-1)

            launch_event.record(torch.cuda.current_stream())
            with torch.cuda.stream(ready_watch_stream):
                _wait_eq_cuda(_chunk_signal_view(ctx.rs_ctx, ctx.rs_ctx.local_rank, 0, 0), signal_value, ready_watch_stream)
                first_panel_ready_event.record(ready_watch_stream)

            launch_producer(A, B, gemm_out, ctx, workspace, signal_value, gemm_config)
            run_instrumented_rs_consumer(
                gemm_out,
                ctx.rs_ctx,
                output,
                num_runtime_chunks,
                first_consumer_start_event=first_consumer_start_event,
                first_output_commit_event=first_output_commit_event,
            )

            torch.cuda.synchronize()
            if step >= args.warmup_iters:
                ready_times.append(launch_event.elapsed_time(first_panel_ready_event))
                consumer_times.append(launch_event.elapsed_time(first_consumer_start_event))
                commit_times.append(launch_event.elapsed_time(first_output_commit_event))
            sync_all(pg)

        avg_ready = sum(ready_times) / len(ready_times)
        avg_consumer = sum(consumer_times) / len(consumer_times)
        avg_commit = sum(commit_times) / len(commit_times)
        avg_ready, avg_consumer, avg_commit = average_across_ranks(pg, [avg_ready, avg_consumer, avg_commit])

        return {
            "policy": policy,
            "first_panel_ready_ts_ms": avg_ready,
            "first_consumer_start_ts_ms": avg_consumer,
            "first_output_commit_ts_ms": avg_commit,
            "chunk_rows": float(ctx.rs_ctx.chunk_rows),
            "n_bands": float(ctx.rs_ctx.n_bands),
            "active_chunk_window": float(ctx.rs_ctx.active_chunk_window),
            "stage_slots": float(ctx.rs_ctx.stage_slots),
            "comm_lanes": float(len(ctx.rs_ctx.comm_streams)),
            "frontier_chunks": float(getattr(ctx, "frontier_chunks", 0)),
        }
    finally:
        sync_all(pg)
        if ctx is not None:
            ctx.finalize()
        sync_all(pg)
        release_python_cuda_refs()
        sync_all(pg)


def default_output_dir() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(__file__).resolve().parent / "rs_frontier_timeline_results" / stamp


def dump_csv(csv_path: Path, rows: list[dict[str, float]], args, M: int, N: int, K: int) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "policy",
        "M",
        "N",
        "K",
        "dtype",
        "iters",
        "warmup_iters",
        "first_panel_ready_ts_ms",
        "first_consumer_start_ts_ms",
        "first_output_commit_ts_ms",
        "chunk_rows",
        "n_bands",
        "active_chunk_window",
        "stage_slots",
        "comm_lanes",
        "frontier_chunks",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out.update(
                {
                    "M": M,
                    "N": N,
                    "K": K,
                    "dtype": args.dtype,
                    "iters": args.iters,
                    "warmup_iters": args.warmup_iters,
                }
            )
            writer.writerow(out)


def maybe_plot(csv_path: Path, output_dir: Path) -> None:
    plot_script = Path(__file__).resolve().parent / "plot_rs_frontier_timeline.py"
    subprocess.run(
        [
            sys.executable,
            str(plot_script),
            "--input_csv",
            str(csv_path),
            "--output_dir",
            str(output_dir),
        ],
        check=True,
    )


def main():
    args = parse_args()
    pg = initialize_distributed()
    rank = pg.rank()

    try:
        dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
        test_configs = get_test_configs(args)
        if len(test_configs) != 1:
            raise ValueError("timeline collector expects exactly one shape; pass --M, --N, --K explicitly")
        model_name, config = next(iter(test_configs.items()))
        M = args.M
        N = config["N"]
        K = config["K"]

        if pg.size() != LOCAL_WORLD_SIZE:
            raise AssertionError("timeline collector currently supports single-node runs only")
        if args.persistent:
            raise AssertionError("timeline collector currently supports only --no-persistent")
        if rank == 0:
            print(f"[{model_name}] collect timeline: M={M}, N={N}, K={K}")

        A, B = make_data(M, N, K, dtype, args.trans_b, pg)
        rows = []
        for policy in ["uniform", "frontier_first"]:
            row = collect_policy_timeline(policy=policy, A=A, B=B, M=M, dtype=dtype, pg=pg, args=args)
            rows.append(row)
            if rank == 0:
                print(
                    f"[timeline] {policy}: ready={row['first_panel_ready_ts_ms']:.3f} ms, "
                    f"consumer={row['first_consumer_start_ts_ms']:.3f} ms, "
                    f"commit={row['first_output_commit_ts_ms']:.3f} ms"
                )

        if rank == 0 and args.dump_csv:
            output_dir = Path(args.output_dir) if args.output_dir else default_output_dir()
            csv_path = output_dir / "rs_frontier_timeline.csv"
            dump_csv(csv_path, rows, args, M, N, K)
            print(f"[timeline] csv: {csv_path}")
            if args.plot:
                maybe_plot(csv_path, output_dir)
                print(f"[timeline] figure dir: {output_dir}")
    finally:
        finalize_distributed()


if __name__ == "__main__":
    main()
