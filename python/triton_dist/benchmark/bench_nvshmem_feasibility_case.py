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
from __future__ import annotations

import argparse
import json
import os
from typing import Any

import torch

from triton_dist.utils import finalize_distributed, initialize_distributed


JSON_PREFIX = "[probe-json]"


def parse_args():
    parser = argparse.ArgumentParser(description="Probe NVSHMEM feasibility for a single implementation/shape.")
    parser.add_argument("--impl",
                        required=True,
                        choices=["baseline_rs", "new_rs_v5", "baseline_ar", "new_ar_v23"])
    parser.add_argument("--M", type=int, required=True)
    parser.add_argument("--N", type=int, required=True)
    parser.add_argument("--K", type=int, required=True)
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16"])

    parser.add_argument("--baseline_persistent",
                        action=argparse.BooleanOptionalAction,
                        default=torch.cuda.get_device_capability() >= (9, 0))
    parser.add_argument("--baseline_num_comm_sms", type=int, default=16)
    parser.add_argument("--baseline_row_wise", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--baseline_low_latency", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--rs_chunk_rows", type=int, default=0)
    parser.add_argument("--rs_target_chunks_per_rank", type=int, default=2)
    parser.add_argument("--rs_min_chunk_rows", type=int, default=512)
    parser.add_argument("--rs_active_chunk_window", type=int, default=4)
    parser.add_argument("--rs_comm_lanes", type=int, default=2)
    parser.add_argument("--rs_n_bands", type=int, default=1)
    parser.add_argument("--rs_frontier_chunks", type=int, default=1)
    parser.add_argument("--rs_steady_sms", type=int, default=6)
    parser.add_argument("--rs_tail_sms", type=int, default=12)
    parser.add_argument("--rs_stage_slots", type=int, default=4)
    parser.add_argument("--rs_tail_chunk_window", type=int, default=1)
    parser.add_argument("--rs_local_seed_direct", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--ar_chunk_rows", type=int, default=0)
    parser.add_argument("--ar_stripe_rows", type=int, default=256)
    parser.add_argument("--ar_target_chunks", type=int, default=4)
    parser.add_argument("--ar_min_chunk_rows", type=int, default=512)
    parser.add_argument("--ar_active_chunk_window", type=int, default=2)
    parser.add_argument("--ar_n_bands", type=int, default=1)
    parser.add_argument("--ar_frontier_chunks", type=int, default=1)
    parser.add_argument("--ar_stage_slots", type=int, default=4)
    parser.add_argument("--ar_num_comm_sms", type=int, default=16)
    parser.add_argument("--ar_comm_lanes", type=int, default=4)
    return parser.parse_args()


def cdiv(x: int, y: int) -> int:
    return (x + y - 1) // y


def round_up(x: int, align: int) -> int:
    return cdiv(x, align) * align


def auto_chunk_rows_per_rank(max_m_per_rank: int, target_chunks_per_rank: int, min_chunk_rows: int) -> int:
    rows = cdiv(max_m_per_rank, target_chunks_per_rank)
    rows = max(rows, min_chunk_rows)
    rows = min(rows, max_m_per_rank)
    rows = round_up(rows, 256)
    return min(rows, max_m_per_rank)


def auto_chunk_rows_global(max_m: int, target_chunks: int, min_chunk_rows: int) -> int:
    rows = cdiv(max_m, target_chunks)
    rows = max(rows, min_chunk_rows)
    rows = min(rows, max_m)
    rows = round_up(rows, 256)
    return min(rows, max_m)


def classify_exception(exc: Exception) -> tuple[str, str]:
    err = str(exc).replace("\n", " ").strip()
    lower = err.lower()
    if "failed to allocate memory" in lower:
        return "nvshmem_oom", err
    if "invalid for input of size" in lower:
        return "shape_mismatch", err
    return "error", err


def estimate_baseline_rs(args, world_size: int, local_world_size: int, dtype: torch.dtype) -> dict[str, Any]:
    if args.M % world_size != 0:
        raise ValueError("baseline_rs requires M divisible by world_size")
    bytes_per_elem = dtype.itemsize
    m_per_local_rank = args.M // local_world_size
    scatter_buf_bytes = args.M * args.N * bytes_per_elem
    gemm_out_buf_bytes = args.M * args.N * bytes_per_elem
    rs_per_node_buf_bytes = m_per_local_rank * args.N * bytes_per_elem
    p2p_buf_bytes = m_per_local_rank * args.N * bytes_per_elem
    data_bytes = scatter_buf_bytes + gemm_out_buf_bytes + rs_per_node_buf_bytes + p2p_buf_bytes
    signal_bytes = (world_size * 2) * torch.int64.itemsize
    return {
        "impl_family": "gemm_reducescatter",
        "estimated_scatter_buf_bytes": int(scatter_buf_bytes),
        "estimated_gemm_out_buf_bytes": int(gemm_out_buf_bytes),
        "estimated_rs_per_node_buf_bytes": int(rs_per_node_buf_bytes),
        "estimated_p2p_buf_bytes": int(p2p_buf_bytes),
        "estimated_symm_data_bytes": int(data_bytes),
        "estimated_symm_signal_bytes": int(signal_bytes),
        "estimated_symm_total_bytes": int(data_bytes + signal_bytes),
        "estimated_chunk_rows": None,
        "estimated_num_chunks": None,
        "estimated_scatter_rows": args.M,
        "estimated_max_band_cols": args.N,
        "estimated_compaction_ratio": None,
    }


def estimate_new_rs_v5(args, world_size: int, local_world_size: int, dtype: torch.dtype) -> dict[str, Any]:
    if args.M % world_size != 0:
        raise ValueError("new_rs_v5 requires M divisible by world_size")
    bytes_per_elem = dtype.itemsize
    max_m_per_rank = args.M // world_size
    chunk_rows = args.rs_chunk_rows if args.rs_chunk_rows > 0 else auto_chunk_rows_per_rank(
        max_m_per_rank,
        target_chunks_per_rank=args.rs_target_chunks_per_rank,
        min_chunk_rows=args.rs_min_chunk_rows,
    )
    num_chunks = cdiv(max_m_per_rank, chunk_rows)
    active_chunk_window = max(1, min(args.rs_active_chunk_window, num_chunks))
    n_bands = max(1, min(args.rs_n_bands, args.N))
    stage_slots = max(1, min(args.rs_stage_slots, num_chunks * n_bands))
    max_band_cols = cdiv(args.N, n_bands)
    scatter_rows = active_chunk_window * n_bands * local_world_size * chunk_rows
    data_bytes = scatter_rows * max_band_cols * bytes_per_elem
    signal_bytes = (local_world_size * active_chunk_window * n_bands + active_chunk_window * n_bands) * torch.int32.itemsize
    baseline_equiv_rows = args.M
    compaction_ratio = baseline_equiv_rows / max(scatter_rows, 1)
    return {
        "impl_family": "gemm_reducescatter",
        "estimated_symm_data_bytes": int(data_bytes),
        "estimated_symm_signal_bytes": int(signal_bytes),
        "estimated_symm_total_bytes": int(data_bytes + signal_bytes),
        "estimated_chunk_rows": int(chunk_rows),
        "estimated_num_chunks": int(num_chunks),
        "estimated_stage_slots": int(stage_slots),
        "estimated_scatter_rows": int(scatter_rows),
        "estimated_max_band_cols": int(max_band_cols),
        "estimated_compaction_ratio": float(compaction_ratio),
    }


def estimate_baseline_ar(args, world_size: int, local_world_size: int, dtype: torch.dtype) -> dict[str, Any]:
    if world_size != local_world_size:
        raise ValueError("baseline_ar probe expects single-node execution")
    bytes_per_elem = dtype.itemsize
    phases = 2 if args.baseline_low_latency else 1
    data_bytes = phases * ((world_size * args.M * args.N + args.M * args.N) * bytes_per_elem)
    barrier_bytes = phases * (
        world_size * cdiv(args.M, 16) * cdiv(args.N, 16) * torch.int32.itemsize
        + world_size * args.baseline_num_comm_sms * torch.int32.itemsize
    )
    return {
        "impl_family": "gemm_allreduce",
        "estimated_symm_data_bytes": int(data_bytes),
        "estimated_symm_signal_bytes": int(barrier_bytes),
        "estimated_symm_total_bytes": int(data_bytes + barrier_bytes),
        "estimated_chunk_rows": None,
        "estimated_num_chunks": None,
        "estimated_scatter_rows": args.M,
        "estimated_max_band_cols": args.N,
        "estimated_compaction_ratio": None,
    }


def estimate_new_ar_v23(args, world_size: int, local_world_size: int, dtype: torch.dtype) -> dict[str, Any]:
    if world_size != local_world_size:
        raise ValueError("new_ar_v23 probe expects single-node execution")
    bytes_per_elem = dtype.itemsize
    chunk_rows = args.ar_chunk_rows if args.ar_chunk_rows > 0 else auto_chunk_rows_global(
        args.M,
        target_chunks=args.ar_target_chunks,
        min_chunk_rows=args.ar_min_chunk_rows,
    )
    stripe_rows = max(1, min(args.ar_stripe_rows if args.ar_stripe_rows > 0 else min(256, chunk_rows), chunk_rows))
    num_chunks = cdiv(args.M, chunk_rows)
    max_stripes_per_chunk = cdiv(chunk_rows, stripe_rows)
    active_chunk_window = max(1, min(args.ar_active_chunk_window, num_chunks))
    n_bands = max(1, min(args.ar_n_bands, args.N))
    max_band_cols = cdiv(args.N, n_bands)
    total_max_tasks = max(1, num_chunks * n_bands * max_stripes_per_chunk)
    stage_slots = max(1, min(args.ar_stage_slots, total_max_tasks))
    scatter_rows = stage_slots * local_world_size * stripe_rows
    data_bytes = scatter_rows * max_band_cols * bytes_per_elem
    signal_bytes = (local_world_size * stage_slots + stage_slots) * torch.int32.itemsize
    baseline_equiv_rows = active_chunk_window * n_bands * local_world_size * chunk_rows
    compaction_ratio = baseline_equiv_rows / max(scatter_rows, 1)
    return {
        "impl_family": "gemm_allreduce",
        "estimated_symm_data_bytes": int(data_bytes),
        "estimated_symm_signal_bytes": int(signal_bytes),
        "estimated_symm_total_bytes": int(data_bytes + signal_bytes),
        "estimated_chunk_rows": int(chunk_rows),
        "estimated_num_chunks": int(num_chunks),
        "estimated_stage_slots": int(stage_slots),
        "estimated_scatter_rows": int(scatter_rows),
        "estimated_max_band_cols": int(max_band_cols),
        "estimated_compaction_ratio": float(compaction_ratio),
        "estimated_stripe_rows": int(stripe_rows),
        "estimated_max_stripes_per_chunk": int(max_stripes_per_chunk),
        "estimated_baseline_scatter_rows": int(baseline_equiv_rows),
    }


def collect_baseline_rs_actual(ctx) -> dict[str, Any]:
    local_rank = ctx.rs_ctx.local_rank
    scatter_buf_bytes = ctx.rs_ctx.scatter_bufs[local_rank].nbytes
    gemm_out_buf_bytes = ctx.gemm_out_bufs[local_rank].nbytes
    rs_per_node_buf_bytes = ctx.rs_ctx.rs_per_node_bufs[local_rank].nbytes
    p2p_buf_bytes = ctx.rs_ctx.p2p_bufs[local_rank].nbytes
    data_bytes = scatter_buf_bytes + gemm_out_buf_bytes + rs_per_node_buf_bytes + p2p_buf_bytes
    signal_bytes = ctx.rs_ctx.signal_bufs[local_rank].nbytes
    return {
        "actual_scatter_buf_bytes": int(scatter_buf_bytes),
        "actual_gemm_out_buf_bytes": int(gemm_out_buf_bytes),
        "actual_rs_per_node_buf_bytes": int(rs_per_node_buf_bytes),
        "actual_p2p_buf_bytes": int(p2p_buf_bytes),
        "actual_symm_data_bytes": int(data_bytes),
        "actual_symm_signal_bytes": int(signal_bytes),
        "actual_symm_total_bytes": int(data_bytes + signal_bytes),
    }


def collect_new_rs_actual(ctx) -> dict[str, Any]:
    local_rank = ctx.rs_ctx.local_rank
    data_bytes = ctx.rs_ctx.scatter_bufs[local_rank].nbytes
    signal_bytes = ctx.rs_ctx.arrival_flag_bufs[local_rank].nbytes + ctx.rs_ctx.free_flag_bufs[local_rank].nbytes
    return {
        "actual_symm_data_bytes": int(data_bytes),
        "actual_symm_signal_bytes": int(signal_bytes),
        "actual_symm_total_bytes": int(data_bytes + signal_bytes),
        "effective_chunk_rows": int(ctx.rs_ctx.chunk_rows),
        "effective_num_chunks": int(ctx.rs_ctx.num_chunks),
        "effective_active_chunk_window": int(ctx.rs_ctx.active_chunk_window),
        "effective_n_bands": int(ctx.rs_ctx.n_bands),
        "effective_stage_slots": int(ctx.rs_ctx.stage_slots),
        "effective_max_band_cols": int(ctx.rs_ctx.max_band_cols),
        "effective_frontier_chunks": int(ctx.frontier_chunks),
        "local_gemm_out_bytes": int(ctx.gemm_out.nbytes),
    }


def collect_baseline_ar_actual(layer: GemmARLayer) -> dict[str, Any]:
    ctx = layer.ctx
    if hasattr(ctx, "ctxs"):
        data_bytes = 0
        signal_bytes = 0
        for subctx in ctx.ctxs:
            data_bytes += subctx.symm_gemm_out_buf.nbytes + subctx.symm_ar_out_buf.nbytes
            signal_bytes += subctx.gemm_barrier_buf.nbytes + subctx.multi_st_barrier_buf.nbytes
    else:
        data_bytes = ctx.symm_gemm_out_buf.nbytes + ctx.symm_ar_out_buf.nbytes
        signal_bytes = ctx.gemm_barrier_buf.nbytes + ctx.multi_st_barrier_buf.nbytes
    return {
        "actual_symm_data_bytes": int(data_bytes),
        "actual_symm_signal_bytes": int(signal_bytes),
        "actual_symm_total_bytes": int(data_bytes + signal_bytes),
    }


def collect_new_ar_actual(ctx) -> dict[str, Any]:
    data_bytes = ctx.local_scatter_buf.nbytes
    signal_bytes = ctx.local_arrival_flag.nbytes + ctx.local_free_flag.nbytes
    return {
        "actual_symm_data_bytes": int(data_bytes),
        "actual_symm_signal_bytes": int(signal_bytes),
        "actual_symm_total_bytes": int(data_bytes + signal_bytes),
        "effective_chunk_rows": int(ctx.chunk_rows),
        "effective_num_chunks": int(ctx.num_chunks),
        "effective_active_chunk_window": int(ctx.active_chunk_window),
        "effective_n_bands": int(ctx.n_bands),
        "effective_stage_slots": int(ctx.stage_slots),
        "effective_max_band_cols": int(ctx.max_band_cols),
        "effective_frontier_chunks": int(ctx.frontier_chunks),
        "effective_stripe_rows": int(ctx.stripe_rows),
        "effective_compact_scatter_rows": int(ctx.compact_scatter_rows),
        "effective_baseline_scatter_rows": int(ctx.baseline_scatter_rows_estimate),
        "effective_compaction_ratio": float(ctx.scatter_compaction_ratio),
        "local_gemm_out_bytes": int(ctx.gemm_out.nbytes),
        "local_stripe_ready_bytes": int(ctx.stripe_ready_buf.nbytes),
    }


def emit_json(payload: dict[str, Any], rank: int) -> None:
    if rank == 0:
        print(f"{JSON_PREFIX}{json.dumps(payload, ensure_ascii=False, sort_keys=True)}", flush=True)


def main():
    args = parse_args()
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    pg = initialize_distributed()
    rank = pg.rank()
    world_size = pg.size()
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", world_size))

    base_payload: dict[str, Any] = {
        "impl": args.impl,
        "M": int(args.M),
        "N": int(args.N),
        "K": int(args.K),
        "dtype": args.dtype,
        "world_size": int(world_size),
        "local_world_size": int(local_world_size),
    }

    try:
        if args.impl == "baseline_rs":
            base_payload.update(estimate_baseline_rs(args, world_size, local_world_size, dtype))
            ctx = None
            try:
                from triton_dist.kernels.nvidia import create_gemm_rs_context

                rs_stream = torch.cuda.Stream(priority=-1)
                ctx = create_gemm_rs_context(args.M, args.N, rank, world_size, local_world_size, dtype, rs_stream)
                base_payload.update(collect_baseline_rs_actual(ctx))
                base_payload["status"] = "success"
            except Exception as exc:
                status, reason = classify_exception(exc)
                base_payload["status"] = status
                base_payload["reason"] = reason
            finally:
                if ctx is not None:
                    try:
                        ctx.finalize()
                    except Exception:
                        pass
                torch.cuda.empty_cache()
        elif args.impl == "new_rs_v5":
            base_payload.update(estimate_new_rs_v5(args, world_size, local_world_size, dtype))
            ctx = None
            try:
                from triton_dist.kernels.nvidia.new_3rd_v5_frontier_windowed_panel_rsgemm import (
                    create_new_3rd_v5_frontier_windowed_panel_gemm_rs_context,
                )

                ctx = create_new_3rd_v5_frontier_windowed_panel_gemm_rs_context(
                    args.M,
                    args.N,
                    rank,
                    world_size,
                    local_world_size,
                    dtype,
                    chunk_rows=args.rs_chunk_rows,
                    target_chunks_per_rank=args.rs_target_chunks_per_rank,
                    min_chunk_rows=args.rs_min_chunk_rows,
                    active_chunk_window=args.rs_active_chunk_window,
                    comm_lanes=args.rs_comm_lanes,
                    n_bands=args.rs_n_bands,
                    frontier_chunks=args.rs_frontier_chunks,
                    steady_sms=args.rs_steady_sms,
                    tail_sms=args.rs_tail_sms,
                    stage_slots=args.rs_stage_slots,
                    tail_chunk_window=args.rs_tail_chunk_window,
                    local_seed_direct=args.rs_local_seed_direct,
                )
                base_payload.update(collect_new_rs_actual(ctx))
                base_payload["status"] = "success"
            except Exception as exc:
                status, reason = classify_exception(exc)
                base_payload["status"] = status
                base_payload["reason"] = reason
            finally:
                if ctx is not None:
                    try:
                        ctx.finalize()
                    except Exception:
                        pass
                torch.cuda.empty_cache()
        elif args.impl == "baseline_ar":
            base_payload.update(estimate_baseline_ar(args, world_size, local_world_size, dtype))
            layer = None
            try:
                from triton_dist.layers.nvidia import GemmARLayer

                layer = GemmARLayer(
                    pg,
                    args.M,
                    args.N,
                    args.K,
                    dtype,
                    dtype,
                    local_world_size,
                    persistent=args.baseline_persistent,
                    use_ll_kernel=args.baseline_low_latency,
                    copy_to_local=True,
                    NUM_COMM_SMS=args.baseline_num_comm_sms,
                    TILE_MAP_LEVEL=int(args.baseline_row_wise),
                )
                base_payload.update(collect_baseline_ar_actual(layer))
                base_payload["status"] = "success"
            except Exception as exc:
                status, reason = classify_exception(exc)
                base_payload["status"] = status
                base_payload["reason"] = reason
            finally:
                if layer is not None:
                    try:
                        layer.finalize()
                    except Exception:
                        pass
                torch.cuda.empty_cache()
        else:
            base_payload.update(estimate_new_ar_v23(args, world_size, local_world_size, dtype))
            ctx = None
            try:
                from triton_dist.kernels.nvidia.new_windowed_panel_gemm_allreduce_v23 import (
                    create_frontier_windowed_panel_gemm_ar_context_v23,
                )

                ctx = create_frontier_windowed_panel_gemm_ar_context_v23(
                    args.M,
                    args.N,
                    rank,
                    world_size,
                    local_world_size,
                    dtype,
                    chunk_rows=args.ar_chunk_rows,
                    stripe_rows=args.ar_stripe_rows,
                    target_chunks=args.ar_target_chunks,
                    min_chunk_rows=args.ar_min_chunk_rows,
                    active_chunk_window=args.ar_active_chunk_window,
                    n_bands=args.ar_n_bands,
                    frontier_chunks=args.ar_frontier_chunks,
                    stage_slots=args.ar_stage_slots,
                    num_comm_sms=args.ar_num_comm_sms,
                    comm_lanes=args.ar_comm_lanes,
                )
                base_payload.update(collect_new_ar_actual(ctx))
                base_payload["status"] = "success"
            except Exception as exc:
                status, reason = classify_exception(exc)
                base_payload["status"] = status
                base_payload["reason"] = reason
            finally:
                if ctx is not None:
                    try:
                        ctx.finalize()
                    except Exception:
                        pass
                torch.cuda.empty_cache()

        emit_json(base_payload, rank)
    finally:
        finalize_distributed()


if __name__ == "__main__":
    main()
