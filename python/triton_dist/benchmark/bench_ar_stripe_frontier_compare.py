"""Paired, fixed-config v23 stripe/whole-panel experiment.

Run under torchrun on a single full-mesh NVLink node. No autotuning, CUDA
graph replay, or undrained rounds. Input and GEMM config are shared; an optional
first-mode stripe size uses separate contexts with equal symmetric data budgets.
Comparisons can change production, submission, granularity, and reduction;
see AR_PANEL_FRONTIER.md for the precise mode definitions.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics

import torch
import torch.distributed as dist


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--M", type=int, default=1025)
    p.add_argument("--N", type=int, default=1030)
    p.add_argument("--K", type=int, default=1024)
    p.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    p.add_argument("--chunk_rows", type=int, default=512)
    p.add_argument("--stripe_rows", type=int, default=128)
    p.add_argument("--n_bands", type=int, default=2)
    p.add_argument("--frontier_chunks", type=int, default=2)
    p.add_argument("--active_chunk_window", type=int, default=2)
    p.add_argument("--stage_slots", type=int, default=2)
    p.add_argument("--comm_lanes", type=int, default=2)
    p.add_argument("--num_comm_sms", type=int, default=16)
    p.add_argument("--modes", nargs=2, default=["logical", "stripe_frontier"],
                   choices=["logical", "stripe_frontier", "panel_bulk", "panel_frontier",
                            "panel_recursive_doubling", "panel_recursive_doubling_cublas"],
                   help="Two modes with identical allocation/config. For panel_* set stripe_rows=chunk_rows.")
    p.add_argument("--first_mode_stripe_rows", type=int,
                   help="Optional stripe size for the first mode; scale its slots to preserve scatter bytes. "
                   "Uses a separate context, with the same input/GEMM config.")
    p.add_argument("--check_rounds", type=int, default=4)
    p.add_argument("--warmup_iters", type=int, default=5)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--seed", type=int, default=20260916)
    p.add_argument("--output_json", type=Path)
    args = p.parse_args()
    for key in ("M", "N", "K", "chunk_rows", "stripe_rows", "n_bands", "active_chunk_window",
                "stage_slots", "comm_lanes", "num_comm_sms", "check_rounds", "iters"):
        if getattr(args, key) <= 0:
            p.error(f"--{key} must be positive")
    if args.warmup_iters < 0 or args.frontier_chunks < 0:
        p.error("warmup_iters and frontier_chunks must be nonnegative")
    if args.output_json and args.output_json.exists():
        p.error(f"Refusing to overwrite {args.output_json}")
    if args.modes[0] == args.modes[1]:
        p.error("Choose two distinct modes")
    if any(mode.startswith("panel_") for mode in args.modes) and args.stripe_rows != args.chunk_rows:
        p.error("panel_* requires --stripe_rows equal to --chunk_rows")
    if args.first_mode_stripe_rows is not None:
        stripe = args.first_mode_stripe_rows
        if not 0 < stripe <= args.chunk_rows or (args.stage_slots * args.stripe_rows) % stripe:
            p.error("first_mode_stripe_rows must divide stage_slots * stripe_rows and be <= chunk_rows")
        if args.modes[0].startswith("panel_") and stripe != args.chunk_rows:
            p.error("The first panel_* mode must also use whole panels")
    return args


def main():
    args = parse_args()
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not torch.cuda.is_available() or torch.cuda.device_count() < world:
        raise SystemExit(f"GPU preflight failed: visible CUDA devices={torch.cuda.device_count()}, requested ranks={world}. "
                         "Run on a GPU-enabled single-node allocation; no measurements were performed.")
    if world < 2 or int(os.environ.get("LOCAL_WORLD_SIZE", "1")) != world:
        raise SystemExit("Use torchrun with at least 2 ranks on one node.")
    if args.K % world:
        raise SystemExit("K must be divisible by WORLD_SIZE.")

    # Import the GPU stack only after the explicit hardware preflight.
    from triton_dist.kernels.nvidia.new_windowed_panel_gemm_allreduce_v23 import (
        create_frontier_windowed_panel_gemm_ar_context_v23 as create_context,
        frontier_windowed_panel_gemm_allreduce_op_v23 as gemm_ar,
    )
    from triton_dist.kernels.nvidia.gemm import get_config_space
    from triton_dist.utils import (initialize_distributed, finalize_distributed,
                                   nvshmem_barrier_all_on_stream, has_fullmesh_nvlink)

    torch.cuda.set_device(local_rank)
    pg = initialize_distributed()
    rank = pg.rank()
    ctx = None
    first_ctx = None
    try:
        if not has_fullmesh_nvlink():
            raise RuntimeError("v23 requires full-mesh NVLink; do not bypass this guard for the comparison.")
        torch.manual_seed(args.seed + rank)
        torch.backends.cuda.matmul.allow_tf32 = False
        dtype = getattr(torch, args.dtype)
        local_k = args.K // world
        config = get_config_space(False)[0]
        ctx = create_context(args.M, args.N, rank, world, world, dtype,
                             chunk_rows=args.chunk_rows, stripe_rows=args.stripe_rows,
                             n_bands=args.n_bands, frontier_chunks=args.frontier_chunks,
                             active_chunk_window=args.active_chunk_window, stage_slots=args.stage_slots,
                             comm_lanes=args.comm_lanes, num_comm_sms=args.num_comm_sms)
        if args.first_mode_stripe_rows is not None:
            first_ctx = create_context(
                args.M, args.N, rank, world, world, dtype,
                chunk_rows=args.chunk_rows, stripe_rows=args.first_mode_stripe_rows,
                n_bands=args.n_bands, frontier_chunks=args.frontier_chunks,
                active_chunk_window=args.active_chunk_window,
                stage_slots=args.stage_slots * args.stripe_rows // args.first_mode_stripe_rows,
                comm_lanes=args.comm_lanes, num_comm_sms=args.num_comm_sms)
            if first_ctx.local_scatter_buf.numel() != ctx.local_scatter_buf.numel():
                raise ValueError("Effective task/slot clamping changed the allocation: scatter budgets do not match")
        contexts = {args.modes[0]: first_ctx if first_ctx is not None else ctx, args.modes[1]: ctx}

        def sync():
            nvshmem_barrier_all_on_stream(torch.cuda.current_stream())
            torch.cuda.synchronize()
            dist.barrier(group=pg, device_ids=[local_rank])

        def launch(mode, a, b):
            selected = contexts[mode]
            selected.producer_order = mode
            return gemm_ar(a, b, selected, config, drain=True)

        def rank_max(value):
            tensor = torch.tensor(value, dtype=torch.float64, device="cuda")
            dist.all_reduce(tensor, op=dist.ReduceOp.MAX, group=pg)
            return tensor.cpu().tolist()

        checks = []
        modes = tuple(args.modes)
        # Changing data between drained rounds catches stale flags/data, unlike
        # repeating one tiny-amplitude input. Force slot reuse via the defaults.
        for round_id in range(args.check_rounds):
            a = (torch.randn((args.M, local_k), device="cuda") / math.sqrt(local_k)).to(dtype)
            b = torch.randn((args.N, local_k), device="cuda").to(dtype).T
            reference = a.float() @ b.float()
            dist.all_reduce(reference, group=pg)
            for mode in modes:
                sync()
                out = launch(mode, a, b)
                sync()
                error = (out.float() - reference).abs()
                # FP32 reference includes unrounded GEMM partials. Legacy AR
                # rounds each addition; panel AR uses FP32 accumulation with
                # a final cast. Both elementwise and norm tests are required.
                atol, rtol, norm_limit = ((0.05, 0.03, 0.02) if dtype == torch.bfloat16
                                         else (0.008, 0.004, 0.003))
                relative_l2 = (error.norm() / reference.norm().clamp_min(1e-12)).item()
                failed = not bool(torch.isfinite(out).all().item()) or not bool(
                    (error <= atol + rtol * reference.abs()).all().item()) or not relative_l2 <= norm_limit
                worst = rank_max([float(failed), error.max().item(), relative_l2])
                row = dict(round=round_id, mode=mode, failed=bool(worst[0]),
                           max_abs_error=worst[1], relative_l2=worst[2])
                checks.append(row)
                if rank == 0:
                    print(json.dumps({"check": row}), flush=True)
                if worst[0]:
                    raise RuntimeError(f"Numerical check failed: {row}")

        for _ in range(args.warmup_iters):
            for mode in modes:
                sync()
                launch(mode, a, b)
        sync()
        samples = {mode: [] for mode in modes}
        for sample in range(args.iters):
            for mode in modes if sample % 2 == 0 else modes[::-1]:
                sync()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                result = launch(mode, a, b)
                end.record()
                end.synchronize()
                elapsed = rank_max(start.elapsed_time(end))
                samples[mode].append(elapsed)
                if rank == 0:
                    print(json.dumps(dict(sample=sample, mode=mode, rank_max_ms=elapsed)), flush=True)
                del result

        if rank == 0:
            kernel_file = Path(__file__).resolve().parents[1] / "kernels/nvidia/new_windowed_panel_gemm_allreduce_v23.py"
            medians = {mode: statistics.median(values) for mode, values in samples.items()}
            report = dict(
                experiment=f"v23 {modes[0]} vs {modes[1]} (see mode definitions; not necessarily order-only)",
                args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                world_size=world, device=torch.cuda.get_device_name(), torch_version=torch.__version__,
                cuda_version=torch.version.cuda, gemm_config=str(config),
                kernel_sha256=hashlib.sha256(kernel_file.read_bytes()).hexdigest(),
                symmetric_scatter_bytes=ctx.local_scatter_buf.numel() * ctx.local_scatter_buf.element_size(),
                allocation_by_mode={mode: dict(stripe_rows=selected.stripe_rows, stage_slots=selected.stage_slots,
                                               scatter_bytes=selected.local_scatter_buf.numel() *
                                               selected.local_scatter_buf.element_size())
                                    for mode, selected in contexts.items()},
                checks=checks, rank_max_samples_ms=samples, median_ms=medians,
                ratio_of_medians=medians[modes[0]] / medians[modes[1]],
                median_paired_speedup=statistics.median(
                    x / y for x, y in zip(samples[modes[0]], samples[modes[1]])),
            )
            print(json.dumps(report, indent=2), flush=True)
            if args.output_json:
                args.output_json.parent.mkdir(parents=True, exist_ok=True)
                with args.output_json.open("x") as handle:
                    json.dump(report, handle, indent=2)
                    handle.write("\n")
    finally:
        if first_ctx is not None:
            first_ctx.finalize()
        if ctx is not None:
            ctx.finalize()
        finalize_distributed()


if __name__ == "__main__":
    main()
