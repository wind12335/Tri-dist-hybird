"""Paired Figure-13 ablation for AR whole-panel handoff policy.

Both treatments use the same cuBLAS panel producer, recursive-doubling
AllReduce, panel order, active window, staging allocation, and input tensors.
Only the host submission policy differs:

* bulk: submit every producer in the active window before its consumers;
* frontier: submit each panel consumer immediately after its producer.

One four-rank torchrun launch is one independent experimental replicate.
Repeated timings inside the launch are paired measurements, not replicates.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.distributed

from triton_dist.kernels.nvidia import (
    create_frontier_windowed_panel_gemm_ar_context_v23,
    frontier_windowed_panel_gemm_allreduce_op_v23,
)
from triton_dist.kernels.nvidia.gemm import get_config_space
from triton_dist.utils import (
    finalize_distributed,
    initialize_distributed,
    nvshmem_barrier_all_on_stream,
    rand_tensor,
)


MODE_TO_PRODUCER_ORDER = {
    "bulk": "panel_recursive_doubling_bulk_cublas",
    "frontier": "panel_recursive_doubling_cublas",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--M", type=int, required=True)
    parser.add_argument("--N", type=int, required=True)
    parser.add_argument("--K", type=int, required=True)
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--chunk_rows", type=int, default=1024)
    parser.add_argument("--active_chunk_window", type=int, default=4)
    parser.add_argument("--n_bands", type=int, default=1)
    parser.add_argument("--stage_slots", type=int, default=4)
    parser.add_argument("--comm_lanes", type=int, default=2)
    parser.add_argument("--num_comm_sms", type=int, default=64)
    parser.add_argument("--warmup_pairs", type=int, default=5)
    parser.add_argument("--measured_pairs", type=int, default=20)
    parser.add_argument("--first_mode", choices=("bulk", "frontier"), default="frontier")
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--repetition", type=int, default=1)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--atol", type=float, default=0.0625)
    parser.add_argument("--rtol", type=float, default=0.0625)
    return parser.parse_args()


def sync_all(pg: torch.distributed.ProcessGroup) -> None:
    nvshmem_barrier_all_on_stream(torch.cuda.current_stream())
    torch.cuda.synchronize()
    torch.distributed.barrier(pg, device_ids=[torch.cuda.current_device()])


def ordered_pair(first_mode: str, pair_index: int) -> tuple[str, str]:
    first = first_mode if pair_index % 2 == 0 else ("bulk" if first_mode == "frontier" else "frontier")
    second = "bulk" if first == "frontier" else "frontier"
    return first, second


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    pg = initialize_distributed()
    rank = pg.rank()
    world_size = pg.size()
    if world_size != 4:
        raise ValueError(f"Figure-13 AR ablation requires TP=4, got {world_size}")
    if args.K % world_size:
        raise ValueError(f"K={args.K} must be divisible by TP={world_size}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.distributed.barrier(pg, device_ids=[torch.cuda.current_device()])
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed(args.seed + rank)
    local_k = args.K // world_size
    scale = 0.01 * (rank + 1)
    a = rand_tensor((args.M, local_k), dtype=dtype, device=torch.cuda.current_device()) * scale
    b = rand_tensor((local_k, args.N), dtype=dtype, device=torch.cuda.current_device()) * scale

    ctx = create_frontier_windowed_panel_gemm_ar_context_v23(
        args.M,
        args.N,
        rank,
        world_size,
        int(os.environ.get("LOCAL_WORLD_SIZE", world_size)),
        dtype,
        chunk_rows=args.chunk_rows,
        stripe_rows=args.chunk_rows,
        active_chunk_window=args.active_chunk_window,
        n_bands=args.n_bands,
        frontier_chunks=0,
        stage_slots=args.stage_slots,
        num_comm_sms=args.num_comm_sms,
        comm_lanes=args.comm_lanes,
        producer_order=MODE_TO_PRODUCER_ORDER["frontier"],
    )
    gemm_config = get_config_space(False)[0]

    def launch(mode: str) -> torch.Tensor:
        ctx.producer_order = MODE_TO_PRODUCER_ORDER[mode]
        return frontier_windowed_panel_gemm_allreduce_op_v23(a, b, ctx, gemm_config, drain=True)

    try:
        sync_all(pg)
        reference = torch.mm(a, b)
        torch.distributed.all_reduce(reference, group=pg)
        correctness = {}
        for mode in ("bulk", "frontier"):
            sync_all(pg)
            output = launch(mode)
            torch.cuda.synchronize()
            diff = (reference.float() - output.float()).abs()
            max_abs = float(diff.max().item())
            denom = reference.float().abs().clamp_min(1e-6)
            max_rel = float((diff / denom).max().item())
            local_ok = torch.tensor(
                [int(torch.allclose(reference, output, atol=args.atol, rtol=args.rtol))],
                dtype=torch.int32,
                device=reference.device,
            )
            torch.distributed.all_reduce(local_ok, op=torch.distributed.ReduceOp.MIN, group=pg)
            correctness[mode] = {"ok": bool(local_ok.item()), "max_abs": max_abs, "max_rel": max_rel}
            if not local_ok.item():
                raise AssertionError(
                    f"{mode} correctness failed on rank {rank}: max_abs={max_abs}, max_rel={max_rel}"
                )

        for pair_index in range(args.warmup_pairs):
            for mode in ordered_pair(args.first_mode, pair_index):
                sync_all(pg)
                launch(mode)
                torch.cuda.synchronize()

        local_samples = {"bulk": [], "frontier": []}
        rank_max_samples = {"bulk": [], "frontier": []}
        all_rank_samples = {"bulk": [], "frontier": []}
        execution_order = []
        for pair_index in range(args.measured_pairs):
            pair_order = ordered_pair(args.first_mode, pair_index)
            execution_order.append(list(pair_order))
            for mode in pair_order:
                sync_all(pg)
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record(torch.cuda.current_stream())
                launch(mode)
                end.record(torch.cuda.current_stream())
                end.synchronize()
                local_ms = float(start.elapsed_time(end))
                local_tensor = torch.tensor([local_ms], dtype=torch.float64, device=a.device)
                gathered = [torch.empty_like(local_tensor) for _ in range(world_size)]
                torch.distributed.all_gather(gathered, local_tensor, group=pg)
                rank_values = [float(value.item()) for value in gathered]
                local_samples[mode].append(local_ms)
                all_rank_samples[mode].append(rank_values)
                rank_max_samples[mode].append(max(rank_values))

        medians = {mode: statistics.median(values) for mode, values in rank_max_samples.items()}
        summary = {
            "schema": "ar-handoff-ablation-v1",
            "timestamp_unix": time.time(),
            "shape": {"M": args.M, "N": args.N, "K": args.K},
            "dtype": args.dtype,
            "tp": world_size,
            "repetition": args.repetition,
            "first_mode": args.first_mode,
            "execution_order": execution_order,
            "modes": MODE_TO_PRODUCER_ORDER,
            "fixed_parameters": {
                "chunk_rows": args.chunk_rows,
                "stripe_rows": args.chunk_rows,
                "active_chunk_window": args.active_chunk_window,
                "n_bands": args.n_bands,
                "stage_slots": args.stage_slots,
                "comm_lanes": args.comm_lanes,
                "num_comm_sms": args.num_comm_sms,
                "autotune": False,
                "warmup_pairs": args.warmup_pairs,
                "measured_pairs": args.measured_pairs,
            },
            "correctness": correctness,
            "rank_max_samples_ms": rank_max_samples,
            "all_rank_samples_ms": all_rank_samples,
            "median_rank_max_ms": medians,
            "frontier_speedup_over_bulk": medians["bulk"] / medians["frontier"],
            "symmetric_staging": {
                "scatter_rows": ctx.compact_scatter_rows,
                "scatter_bytes_per_pe": ctx.compact_scatter_rows * ctx.max_band_cols * a.element_size(),
                "arrival_flag_bytes_per_pe": world_size * ctx.stage_slots * ctx.local_arrival_flag.element_size(),
                "free_flag_bytes_per_pe": ctx.stage_slots * ctx.local_free_flag.element_size(),
            },
            "environment": {
                "hostname": socket.gethostname(),
                "platform": platform.platform(),
                "python": sys.version,
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(local_rank),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "nvshmem_symmetric_size": os.environ.get("NVSHMEM_SYMMETRIC_SIZE"),
            },
        }
        rank_payload = {
            "rank": rank,
            "local_rank": local_rank,
            "correctness": correctness,
            "local_samples_ms": local_samples,
            "summary_file": "summary.json",
        }
        write_json(args.output_dir / f"rank_{rank}.json", rank_payload)
        if rank == 0:
            write_json(args.output_dir / "summary.json", summary)
            print("AR_HANDOFF_SUMMARY " + json.dumps({
                "shape": summary["shape"],
                "repetition": args.repetition,
                "bulk_ms": medians["bulk"],
                "frontier_ms": medians["frontier"],
                "frontier_speedup_over_bulk": summary["frontier_speedup_over_bulk"],
            }, sort_keys=True), flush=True)
    finally:
        ctx.finalize()
        finalize_distributed()


if __name__ == "__main__":
    main()
