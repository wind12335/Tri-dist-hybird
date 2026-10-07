################################################################################
#
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#
################################################################################

"""Controlled frontier-first producer-order ablation for panel-ready GEMM--RS.

Unlike ``bench_frontier_schedule_ablation_gemmrs.py``, this benchmark does not
compare the legacy v3 and v5 implementations.  It holds the following fixed
between ``uniform`` and ``frontier_first``:

* v5 windowed RS context and its ready/free-ticket protocol;
* GEMM tile configuration, complete output-tile set, and two launch grids;
* chunking, active window, physical slots, bands, lanes, and SM parameters.

The experimental factor is only the producer work permutation.  ``uniform``
uses a contiguous segment-major sequence, while ``frontier_first`` schedules
the first ``frontier_chunks`` from every destination segment before the tail.

Every measured iteration is synchronized before and after execution.  The
script records both rank-local CUDA-event latency and its ``all_reduce(MAX)``
rank-max counterpart.  It writes raw per-rank samples, a rank-0 aggregate CSV,
and a manifest with the exact command and controlled-condition contract.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Callable, Iterable

import torch
import torch.distributed
import triton

from triton_dist.kernels.nvidia.experimental_frontier_order_controlled import (
    launch_controlled_frontier_order_producer,
)
from triton_dist.kernels.nvidia.gemm import get_config_space
from triton_dist.kernels.nvidia.new_3rd_v5_frontier_windowed_panel_rsgemm import (
    create_new_3rd_v5_frontier_windowed_panel_gemm_rs_context,
)
from triton_dist.kernels.nvidia.new_3rd_v5_frontier_windowed_panel_rs import new_3rd_v3_windowed_panel_rs_op
from triton_dist.test.utils import assert_allclose
from triton_dist.utils import (
    dist_print,
    finalize_distributed,
    initialize_distributed,
    nvshmem_barrier_all_on_stream,
    rand_tensor,
    wait_until_max_gpu_clock_or_warning,
)


POLICIES = ("uniform", "frontier_first")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--M", type=int, default=8192)
    parser.add_argument("--N", type=int, required=False)
    parser.add_argument("--K", type=int, required=False)
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--trans_b", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup_iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--atol", type=float, default=None)
    parser.add_argument("--rtol", type=float, default=None)
    parser.add_argument("--chunk_rows", type=int, default=512)
    parser.add_argument("--target_chunks_per_rank", type=int, default=2)
    parser.add_argument("--min_chunk_rows", type=int, default=512)
    parser.add_argument("--active_chunk_window", type=int, default=4)
    parser.add_argument("--stage_slots", type=int, default=4)
    parser.add_argument("--steady_sms", type=int, default=8)
    parser.add_argument("--tail_sms", type=int, default=20)
    parser.add_argument("--tail_chunk_window", type=int, default=1)
    parser.add_argument("--comm_lanes", type=int, default=2)
    parser.add_argument("--n_bands", type=int, default=2)
    parser.add_argument("--frontier_chunks", type=int, default=2)
    parser.add_argument("--local_seed_direct", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--gemm_config_index", type=int, default=0,
                        help="Shared index into get_config_space(False); no policy-specific autotuning is performed.")
    parser.add_argument("--no-autotune", action="store_true",
                        help="Documentation flag: this benchmark always uses the shared fixed config index.")
    parser.add_argument("--producer_order", choices=[*POLICIES, "both"], default="both")
    parser.add_argument("--measurement_order", choices=["uniform_first", "frontier_first"], default="uniform_first",
                        help="Execution order when --producer_order=both; alternate it between independent launches.")
    parser.add_argument(
        "--measure_torch_reference",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Measure the optional Torch GEMM--RS reference after the controlled policy pair.",
    )
    parser.add_argument("--run_id", type=str, default="controlled_order")
    parser.add_argument("--output_csv", type=Path, default=None,
                        help="Rank-0 aggregate CSV. Required for actual GPU execution.")
    parser.add_argument("--raw_samples_dir", type=Path, default=None,
                        help="Directory for one raw timing CSV per rank; defaults beside --output_csv.")
    parser.add_argument("--manifest_json", type=Path, default=None,
                        help="Rank-0 metadata manifest; defaults beside --output_csv.")
    parser.add_argument("--validate_schedule_only", action="store_true",
                        help="CPU-only preflight: prove the two policies are permutations of the same tile set.")
    parser.add_argument("--schedule_preview_tiles", type=int, default=24)
    parser.add_argument(
        "--progress_every",
        type=int,
        default=0,
        help="Diagnostic-only progress interval. Zero disables progress output and is the measurement default.",
    )
    return parser.parse_args()


def _config_or_raise(index: int) -> triton.Config:
    config_space = get_config_space(False)
    if index < 0 or index >= len(config_space):
        raise ValueError(f"gemm_config_index={index} is outside [0, {len(config_space) - 1}]")
    return config_space[index]


def _frontier_tile_count(M: int, world_size: int, chunk_rows: int, frontier_chunks: int, block_size_m: int) -> tuple[int, int]:
    if M % world_size:
        raise ValueError(f"M={M} must be divisible by world_size={world_size}")
    tiles_per_segment = triton.cdiv(M // world_size, block_size_m)
    selected_rows = min(chunk_rows * frontier_chunks, M // world_size)
    frontier_tiles = min(triton.cdiv(selected_rows, block_size_m), tiles_per_segment)
    return tiles_per_segment, frontier_tiles


def policy_tile_order(
    policy: str,
    *,
    M: int,
    world_size: int,
    chunk_rows: int,
    frontier_chunks: int,
    block_size_m: int,
) -> tuple[list[int], list[int]]:
    """Return phase-0 and phase-1 logical M-tile IDs for a policy."""
    if policy not in POLICIES:
        raise ValueError(f"unsupported policy {policy!r}")
    tiles_per_segment, frontier_tiles = _frontier_tile_count(
        M, world_size, chunk_rows, frontier_chunks, block_size_m
    )
    if (M // world_size) % block_size_m:
        raise ValueError(
            "controlled order currently requires M/world_size to be BLOCK_SIZE_M aligned; "
            f"got M/world_size={M // world_size}, BLOCK_SIZE_M={block_size_m}"
        )
    first_phase_count = frontier_tiles * world_size
    total_tiles = tiles_per_segment * world_size

    if policy == "uniform":
        return list(range(first_phase_count)), list(range(first_phase_count, total_tiles))

    frontier = [
        segment * tiles_per_segment + local_tile
        for segment in range(world_size)
        for local_tile in range(frontier_tiles)
    ]
    tail = [
        segment * tiles_per_segment + frontier_tiles + local_tile
        for segment in range(world_size)
        for local_tile in range(tiles_per_segment - frontier_tiles)
    ]
    return frontier, tail


def validate_schedule_contract(args: argparse.Namespace, world_size: int) -> None:
    if args.N is None or args.K is None:
        raise ValueError("--validate_schedule_only still requires --N and --K to avoid an ambiguous shape record")
    config = _config_or_raise(args.gemm_config_index)
    block_size_m = config.kwargs["BLOCK_SIZE_M"]
    if args.frontier_chunks < 1:
        raise ValueError("--frontier_chunks must be >= 1")
    tiles_per_segment, frontier_tiles = _frontier_tile_count(
        args.M, world_size, args.chunk_rows, args.frontier_chunks, block_size_m
    )
    expected = list(range(tiles_per_segment * world_size))
    uniform = sum(policy_tile_order("uniform", M=args.M, world_size=world_size, chunk_rows=args.chunk_rows,
                                    frontier_chunks=args.frontier_chunks, block_size_m=block_size_m), [])
    frontier = sum(policy_tile_order("frontier_first", M=args.M, world_size=world_size, chunk_rows=args.chunk_rows,
                                     frontier_chunks=args.frontier_chunks, block_size_m=block_size_m), [])
    if sorted(uniform) != expected or sorted(frontier) != expected:
        raise AssertionError("controlled schedules do not cover exactly the same logical tile set")
    if len(uniform) != len(set(uniform)) or len(frontier) != len(set(frontier)):
        raise AssertionError("controlled schedules contain duplicate logical tiles")
    if uniform == frontier:
        raise AssertionError("controlled schedules are identical; no order factor remains")

    preview = max(1, args.schedule_preview_tiles)
    uniform_phase0, uniform_phase1 = policy_tile_order(
        "uniform", M=args.M, world_size=world_size, chunk_rows=args.chunk_rows,
        frontier_chunks=args.frontier_chunks, block_size_m=block_size_m,
    )
    frontier_phase0, frontier_phase1 = policy_tile_order(
        "frontier_first", M=args.M, world_size=world_size, chunk_rows=args.chunk_rows,
        frontier_chunks=args.frontier_chunks, block_size_m=block_size_m,
    )
    print("[controlled-order preflight] PASS", flush=True)
    print(f"shape={args.M}x{args.N}x{args.K}, world_size={world_size}, BLOCK_SIZE_M={block_size_m}", flush=True)
    print(f"tiles_per_segment={tiles_per_segment}, frontier_tiles_per_segment={frontier_tiles}", flush=True)
    print(f"uniform phase0={uniform_phase0[:preview]}; phase1={uniform_phase1[:preview]}", flush=True)
    print(f"frontier_first phase0={frontier_phase0[:preview]}; phase1={frontier_phase1[:preview]}", flush=True)
    print("contract=same v5 context, same ready tickets, same complete tile set, same two phase grid sizes; only tile order differs", flush=True)


def sync_all(pg: torch.distributed.ProcessGroup) -> None:
    nvshmem_barrier_all_on_stream(torch.cuda.current_stream())
    torch.cuda.synchronize()
    torch.distributed.barrier(group=pg, device_ids=[torch.cuda.current_device()])


def make_data(M: int, N: int, K: int, dtype: torch.dtype, trans_b: bool, pg: torch.distributed.ProcessGroup):
    rank = pg.rank()
    world_size = pg.size()
    if K % world_size:
        raise ValueError(f"K={K} must be divisible by world_size={world_size}")
    local_K = K // world_size
    scale = (rank + 1) * 0.01
    A = rand_tensor([M, local_K], dtype=dtype, device=torch.cuda.current_device()) * scale
    if trans_b:
        B = (rand_tensor([N, local_K], dtype=dtype, device=torch.cuda.current_device()) * scale).T.contiguous()
    else:
        B = (rand_tensor([local_K, N], dtype=dtype, device=torch.cuda.current_device()) * scale).contiguous()
    return A, B


def torch_gemm_rs(pg: torch.distributed.ProcessGroup, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    partial = torch.matmul(A, B)
    output = torch.empty((A.shape[0] // pg.size(), B.shape[1]), dtype=partial.dtype, device=A.device)
    torch.distributed.reduce_scatter_tensor(output, partial, group=pg)
    return output


def measure_rankmax(
    func: Callable[[], torch.Tensor],
    pg: torch.distributed.ProcessGroup,
    *,
    iters: int,
    warmup_iters: int,
    progress_every: int = 0,
    progress_label: str = "",
) -> tuple[torch.Tensor, list[float], list[float]]:
    """Measure local CUDA events and rank-max latency outside the timed region."""
    if iters < 1:
        raise ValueError("--iters must be >= 1")
    output = None
    for warmup_index in range(warmup_iters):
        sync_all(pg)
        output = func()
        sync_all(pg)
        if progress_every and (warmup_index + 1) % progress_every == 0 and pg.rank() == 0:
            print(f"[progress] {progress_label} warmup {warmup_index + 1}/{warmup_iters}", flush=True)

    local_samples: list[float] = []
    rankmax_samples: list[float] = []
    for iteration_index in range(iters):
        sync_all(pg)
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        output = func()
        stop.record()
        sync_all(pg)
        stop.synchronize()
        local_ms = float(start.elapsed_time(stop))
        local_samples.append(local_ms)
        rank_max = torch.tensor([local_ms], dtype=torch.float64, device=output.device)
        torch.distributed.all_reduce(rank_max, op=torch.distributed.ReduceOp.MAX, group=pg)
        rankmax_samples.append(float(rank_max.item()))
        if progress_every and (iteration_index + 1) % progress_every == 0 and pg.rank() == 0:
            print(f"[progress] {progress_label} measurement {iteration_index + 1}/{iters}", flush=True)
    assert output is not None
    return output, local_samples, rankmax_samples


def _stats(prefix: str, values: Iterable[float]) -> dict[str, float]:
    samples = list(values)
    return {
        f"{prefix}_mean_ms": mean(samples),
        f"{prefix}_median_ms": median(samples),
        f"{prefix}_min_ms": min(samples),
        f"{prefix}_max_ms": max(samples),
    }


def _write_rank_raw(path: Path, run_id: str, policy: str, local_samples: list[float], rankmax_samples: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=["run_id", "policy", "iteration", "local_latency_ms", "rank_max_latency_ms"])
        writer.writeheader()
        for iteration, (local_ms, rankmax_ms) in enumerate(zip(local_samples, rankmax_samples), start=1):
            writer.writerow({
                "run_id": run_id,
                "policy": policy,
                "iteration": iteration,
                "local_latency_ms": f"{local_ms:.8f}",
                "rank_max_latency_ms": f"{rankmax_ms:.8f}",
            })


def _controlled_op(A, B, ctx, workspace, gemm_config, producer_order: str) -> torch.Tensor:
    output = torch.empty((A.shape[0] // ctx.rs_ctx.world_size, B.shape[1]), dtype=ctx.output_dtype, device=A.device)
    num_runtime_chunks = triton.cdiv(A.shape[0] // ctx.rs_ctx.world_size, ctx.rs_ctx.chunk_rows)
    signal_value = ctx.rs_ctx.begin_round(num_runtime_chunks)
    launch_controlled_frontier_order_producer(
        A, B, ctx.get_gemm_out_buf(A), ctx, workspace, signal_value, gemm_config, producer_order=producer_order
    )
    return new_3rd_v3_windowed_panel_rs_op(ctx.get_gemm_out_buf(A), ctx.rs_ctx, output, prepare_round=False)


def _reset_runtime(ctx, pg: torch.distributed.ProcessGroup) -> None:
    ctx.rs_ctx.reset_runtime_state()
    sync_all(pg)


def _policy_sequence(args: argparse.Namespace) -> list[str]:
    if args.producer_order != "both":
        return [args.producer_order]
    return ["uniform", "frontier_first"] if args.measurement_order == "uniform_first" else ["frontier_first", "uniform"]


def _resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    if args.output_csv is None:
        raise ValueError("--output_csv is required for a GPU measurement run")
    output_csv = args.output_csv
    raw_dir = args.raw_samples_dir or output_csv.parent / "raw_per_rank"
    manifest = args.manifest_json or output_csv.with_suffix(".manifest.json")
    return output_csv, raw_dir, manifest


def _write_aggregate(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        raise AssertionError("no records to write")
    with path.open("w", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


def _jsonable_config(config: triton.Config) -> dict[str, object]:
    values: dict[str, object] = {}
    for key, value in config.all_kwargs().items():
        values[key] = value
    return values


def run_benchmark(args: argparse.Namespace) -> None:
    if args.N is None or args.K is None:
        raise ValueError("GPU measurement requires both --N and --K")
    if args.n_bands < 2:
        raise ValueError("controlled producer-order ablation requires --n_bands >= 2")
    if args.frontier_chunks < 1:
        raise ValueError("--frontier_chunks must be >= 1")
    if args.iters < 1 or args.warmup_iters < 0 or args.progress_every < 0:
        raise ValueError("--iters must be >= 1, --warmup_iters must be >= 0, and --progress_every must be >= 0")

    output_csv, raw_dir, manifest_path = _resolve_paths(args)
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    atol = args.atol if args.atol is not None else (2e-2 if dtype == torch.bfloat16 else 1e-2)
    rtol = args.rtol if args.rtol is not None else atol
    gemm_config = _config_or_raise(args.gemm_config_index)

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    pg = initialize_distributed(seed=args.seed)
    rank = pg.rank()
    world_size = pg.size()
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", str(world_size)))

    if local_world_size != world_size:
        finalize_distributed()
        raise AssertionError("controlled frontier-order ablation currently supports a single node only")
    if args.M % world_size or args.K % world_size:
        finalize_distributed()
        raise ValueError("M and K must both be divisible by the tensor-parallel world size")

    if rank == 0:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        raw_dir.mkdir(parents=True, exist_ok=True)
    torch.distributed.barrier(group=pg, device_ids=[torch.cuda.current_device()])

    ctx = None
    try:
        A, B = make_data(args.M, args.N, args.K, dtype, args.trans_b, pg)
        sync_all(pg)
        C_torch = torch_gemm_rs(pg, A, B)
        sync_all(pg)
        ctx = create_new_3rd_v5_frontier_windowed_panel_gemm_rs_context(
            args.M,
            args.N,
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
        workspace = torch.zeros(
            (ctx.rs_ctx.n_bands * world_size * ctx.rs_ctx.num_chunks,), dtype=torch.int32, device=A.device
        )
        policies = _policy_sequence(args)
        results: dict[str, dict[str, object]] = {}

        for policy in policies:
            _reset_runtime(ctx, pg)
            candidate = _controlled_op(A, B, ctx, workspace, gemm_config, policy)
            assert_allclose(C_torch, candidate, rtol=rtol, atol=atol, verbose=rank == 0)
            sync_all(pg)

            _reset_runtime(ctx, pg)
            wait_until_max_gpu_clock_or_warning(torch.cuda.current_device())
            _, local_samples, rankmax_samples = measure_rankmax(
                lambda selected=policy: _controlled_op(A, B, ctx, workspace, gemm_config, selected),
                pg,
                iters=args.iters,
                warmup_iters=args.warmup_iters,
                progress_every=args.progress_every,
                progress_label=policy,
            )
            _write_rank_raw(raw_dir / f"{args.run_id}_{policy}_rank{rank}.csv", args.run_id, policy, local_samples, rankmax_samples)
            results[policy] = {
                "correctness": "passed",
                **_stats("local", local_samples),
                **_stats("rank_max", rankmax_samples),
            }
            sync_all(pg)

        torch_stats: dict[str, float] = {}
        raw_sample_files = [
            str(raw_dir / f"{args.run_id}_{policy}_rank{worker_rank}.csv")
            for policy in policies
            for worker_rank in range(world_size)
        ]
        if args.measure_torch_reference:
            # The reference is diagnostic context, not part of the controlled policy comparison.
            wait_until_max_gpu_clock_or_warning(torch.cuda.current_device())
            _, torch_local_samples, torch_rankmax_samples = measure_rankmax(
                lambda: torch_gemm_rs(pg, A, B),
                pg,
                iters=args.iters,
                warmup_iters=args.warmup_iters,
                progress_every=args.progress_every,
                progress_label="torch_reference",
            )
            torch_reference_path = raw_dir / f"{args.run_id}_torch_reference_rank{rank}.csv"
            _write_rank_raw(
                torch_reference_path,
                args.run_id,
                "torch_reference",
                torch_local_samples,
                torch_rankmax_samples,
            )
            torch_stats = {
                **_stats("torch_local", torch_local_samples),
                **_stats("torch_rank_max", torch_rankmax_samples),
            }
            raw_sample_files = [
                *raw_sample_files,
                *(str(raw_dir / f"{args.run_id}_torch_reference_rank{worker_rank}.csv")
                  for worker_rank in range(world_size)),
            ]

        if rank == 0:
            pair_ratio = ""
            if all(policy in results for policy in POLICIES):
                pair_ratio = (
                    results["uniform"]["rank_max_mean_ms"] / results["frontier_first"]["rank_max_mean_ms"]
                )
            common: dict[str, object] = {
                "run_id": args.run_id,
                "shape": f"{args.M}x{args.N}x{args.K}",
                "M": args.M,
                "N": args.N,
                "K": args.K,
                "dtype": args.dtype,
                "world_size": world_size,
                "measurement_order": args.measurement_order,
                "iters": args.iters,
                "warmup_iters": args.warmup_iters,
                "atol": atol,
                "rtol": rtol,
                "gemm_config_index": args.gemm_config_index,
                "gemm_config": json.dumps(_jsonable_config(gemm_config), sort_keys=True),
                "autotune_enabled": False,
                "chunk_rows": ctx.rs_ctx.chunk_rows,
                "num_chunks": ctx.rs_ctx.num_chunks,
                "active_chunk_window": ctx.rs_ctx.active_chunk_window,
                "stage_slots": ctx.rs_ctx.stage_slots,
                "comm_lanes": len(ctx.rs_ctx.comm_streams),
                "n_bands": ctx.rs_ctx.n_bands,
                "frontier_chunks": ctx.frontier_chunks,
                "steady_sms": args.steady_sms,
                "tail_sms": args.tail_sms,
                "tail_chunk_window": args.tail_chunk_window,
                "torch_reference_measured": args.measure_torch_reference,
                "controlled_variable": "logical producer tile order only",
                "correctness_protocol": "Torch GEMM-RS all-close before timing",
                "raw_samples_dir": str(raw_dir),
            }
            records = []
            for policy in policies:
                record = {
                    **common,
                    "policy": policy,
                    "correctness": results[policy]["correctness"],
                    "paired_uniform_over_frontier_rank_max_mean": pair_ratio,
                    **{key: value for key, value in results[policy].items() if key != "correctness"},
                    **torch_stats,
                }
                records.append(record)
            _write_aggregate(output_csv, records)
            manifest = {
                "schema_version": 1,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "run_id": args.run_id,
                "command": shlex.join([sys.executable, *sys.argv]),
                "controlled_condition": {
                    "shared_kernel": "_controlled_order_banded_ready_kernel",
                    "shared_v5_context": True,
                    "same_complete_tile_set": True,
                    "same_two_phase_grid_sizes": True,
                    "same_ready_ticket_protocol": True,
                    "only_changed_factor": "logical producer tile permutation",
                    "uniform_definition": "contiguous segment-major prefix followed by remainder",
                    "frontier_first_definition": "frontier chunks from every destination segment followed by remainder",
                },
                "configuration": common,
                "policy_sequence": policies,
                "output_csv": str(output_csv),
                "raw_sample_files": raw_sample_files,
            }
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
            print(f"[aggregate] {output_csv}", flush=True)
            print(f"[manifest] {manifest_path}", flush=True)

        dist_print(
            "[controlled-order] " + ", ".join(
                f"{policy}: rank-max={results[policy]['rank_max_mean_ms']:.4f} ms"
                for policy in policies
            ),
            need_sync=True,
            allowed_ranks=[0],
        )
    finally:
        try:
            sync_all(pg)
        except Exception:
            pass
        if ctx is not None:
            ctx.finalize()
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        finalize_distributed()


if __name__ == "__main__":
    parsed_args = parse_args()
    if parsed_args.validate_schedule_only:
        requested_world_size = int(os.environ.get("WORLD_SIZE", os.environ.get("LOCAL_WORLD_SIZE", "4")))
        validate_schedule_contract(parsed_args, requested_world_size)
    else:
        run_benchmark(parsed_args)
