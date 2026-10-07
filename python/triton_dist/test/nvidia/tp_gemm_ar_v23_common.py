"""Shared helpers for eager GEMM-AllReduce v2.3 end-to-end experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import torch

from triton_dist.profiler_utils import perf_func
from triton_dist.utils import nvshmem_barrier_all_on_stream


AR_V23_PRODUCER_ORDERS = (
    "panel_recursive_doubling",
    "panel_recursive_doubling_cublas",
)


def _ar_arg_name(prefix: str | None, name: str) -> str:
    if prefix is None:
        return name if name == "autotune" else f"ar_{name}"
    return f"{prefix}_{name}"


def add_ar_v23_args(
    parser: argparse.ArgumentParser,
    *,
    prefix: str | None = None,
    defaults: dict[str, Any] | None = None,
) -> None:
    defaults = defaults or {}
    parser.add_argument(f"--{_ar_arg_name(prefix, 'chunk_rows')}", type=int,
                        default=defaults.get("chunk_rows", 1024),
                        help="Rows in one complete AR panel. The compatibility stripe size is set to the same value.")
    parser.add_argument(f"--{_ar_arg_name(prefix, 'target_chunks')}", type=int,
                        default=defaults.get("target_chunks", 4))
    parser.add_argument(f"--{_ar_arg_name(prefix, 'min_chunk_rows')}", type=int,
                        default=defaults.get("min_chunk_rows", 512))
    parser.add_argument(f"--{_ar_arg_name(prefix, 'active_chunk_window')}", type=int,
                        default=defaults.get("active_chunk_window", 4))
    parser.add_argument(f"--{_ar_arg_name(prefix, 'n_bands')}", type=int,
                        default=defaults.get("n_bands", 1))
    parser.add_argument(f"--{_ar_arg_name(prefix, 'stage_slots')}", type=int,
                        default=defaults.get("stage_slots", 4))
    parser.add_argument(f"--{_ar_arg_name(prefix, 'num_comm_sms')}", type=int,
                        default=defaults.get("num_comm_sms", 64))
    parser.add_argument(f"--{_ar_arg_name(prefix, 'comm_lanes')}", type=int,
                        default=defaults.get("comm_lanes", 2))
    parser.add_argument(f"--{_ar_arg_name(prefix, 'producer_order')}", choices=AR_V23_PRODUCER_ORDERS,
                        default=defaults.get("producer_order", "panel_recursive_doubling_cublas"))
    parser.add_argument(f"--{_ar_arg_name(prefix, 'autotune')}",
                        default=defaults.get("autotune", False), action=argparse.BooleanOptionalAction,
                        help="Tune the Triton panel producer. It must remain disabled for the cuBLAS producer.")


def validate_ar_v23_args(args: argparse.Namespace, world_size: int, *, prefix: str | None = None) -> None:
    if world_size < 2 or world_size & (world_size - 1):
        raise ValueError(f"Recursive-doubling AR requires a power-of-two world size >= 2, got {world_size}.")
    for name in (
        "chunk_rows",
        "target_chunks",
        "min_chunk_rows",
        "active_chunk_window",
        "n_bands",
        "stage_slots",
        "num_comm_sms",
        "comm_lanes",
    ):
        arg_name = _ar_arg_name(prefix, name)
        value = getattr(args, arg_name)
        if value <= 0:
            raise ValueError(f"--{arg_name} must be positive, got {value}.")
    producer_order = getattr(args, _ar_arg_name(prefix, "producer_order"))
    autotune = getattr(args, _ar_arg_name(prefix, "autotune"))
    if producer_order.endswith("_cublas") and autotune:
        raise ValueError("The cuBLAS panel producer does not use Triton autotuning; pass --no-autotune.")


def build_ar_v23_kwargs(args: argparse.Namespace, *, prefix: str | None = None) -> dict[str, Any]:
    def value(name: str):
        return getattr(args, _ar_arg_name(prefix, name))

    return {
        "chunk_rows": value("chunk_rows"),
        # Whole-panel AR does not subdivide a panel into stripes. v2.3 keeps
        # stripe_rows as an allocation/API compatibility field, so it equals
        # chunk_rows by construction.
        "stripe_rows": value("chunk_rows"),
        "target_chunks": value("target_chunks"),
        "min_chunk_rows": value("min_chunk_rows"),
        "active_chunk_window": value("active_chunk_window"),
        "n_bands": value("n_bands"),
        "frontier_chunks": 0,
        "stage_slots": value("stage_slots"),
        "num_comm_sms": value("num_comm_sms"),
        "comm_lanes": value("comm_lanes"),
        "producer_order": value("producer_order"),
    }


def _distributed_max(value: float, group) -> float:
    tensor = torch.tensor(value, dtype=torch.float64, device="cuda")
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.MAX, group=group)
    return float(tensor.item())


def _time_synchronized_iterations(
    func: Callable[[], torch.Tensor],
    *,
    group,
    warmup: int,
    iters: int,
) -> float:
    """Measure isolated distributed invocations without queueing later protocol rounds."""

    def align_ranks() -> None:
        torch.distributed.barrier(group=group, device_ids=[torch.cuda.current_device()])
        nvshmem_barrier_all_on_stream(torch.cuda.current_stream())
        torch.cuda.synchronize()

    for _ in range(warmup):
        align_ranks()
        func()
        torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    elapsed_ms = 0.0
    for _ in range(iters):
        align_ranks()
        start.record()
        func()
        stop.record()
        stop.synchronize()
        elapsed_ms += float(start.elapsed_time(stop))
    align_ranks()
    return elapsed_ms / iters


def run_eager_pair(
    torch_func: Callable[[], torch.Tensor],
    v23_func: Callable[[], torch.Tensor],
    *,
    group,
    warmup: int,
    iters: int,
    before_torch: Callable[[], None] | None = None,
    before_v23: Callable[[], None] | None = None,
    synchronize_each_iter: bool = False,
) -> dict[str, float | str]:
    """Time both paths in eager mode and report the slowest rank."""
    if before_torch is not None:
        before_torch()
    nvshmem_barrier_all_on_stream(torch.cuda.current_stream())
    torch.cuda.synchronize()
    if synchronize_each_iter:
        torch_local_ms = _time_synchronized_iterations(
            torch_func, group=group, iters=iters, warmup=warmup)
    else:
        _, torch_local_ms = perf_func(torch_func, iters=iters, warmup_iters=warmup)
    nvshmem_barrier_all_on_stream(torch.cuda.current_stream())
    torch.cuda.synchronize()

    if before_v23 is not None:
        before_v23()
    nvshmem_barrier_all_on_stream(torch.cuda.current_stream())
    torch.cuda.synchronize()
    if synchronize_each_iter:
        v23_local_ms = _time_synchronized_iterations(
            v23_func, group=group, iters=iters, warmup=warmup)
    else:
        _, v23_local_ms = perf_func(v23_func, iters=iters, warmup_iters=warmup)
    nvshmem_barrier_all_on_stream(torch.cuda.current_stream())
    torch.cuda.synchronize()

    torch_max_ms = _distributed_max(torch_local_ms, group)
    v23_max_ms = _distributed_max(v23_local_ms, group)
    return {
        "timing_mode": "synchronized_eager" if synchronize_each_iter else "eager",
        "torch_local_ms": float(torch_local_ms),
        "v23_local_ms": float(v23_local_ms),
        "torch_rank_max_ms": torch_max_ms,
        "v23_rank_max_ms": v23_max_ms,
        "rank_max_speedup": torch_max_ms / v23_max_ms,
    }


def run_single_pair(
    torch_func: Callable[[], torch.Tensor],
    v23_func: Callable[[], torch.Tensor],
    *,
    group,
    before_torch: Callable[[], None] | None = None,
    before_v23: Callable[[], None] | None = None,
) -> dict[str, float | str]:
    """Time one drained invocation of each path using the slowest rank."""

    def time_once(func: Callable[[], torch.Tensor]) -> float:
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        func()
        stop.record()
        stop.synchronize()
        return float(start.elapsed_time(stop))

    if before_torch is not None:
        before_torch()
    nvshmem_barrier_all_on_stream(torch.cuda.current_stream())
    torch.cuda.synchronize()
    torch_local_ms = time_once(torch_func)
    nvshmem_barrier_all_on_stream(torch.cuda.current_stream())
    torch.cuda.synchronize()

    if before_v23 is not None:
        before_v23()
    nvshmem_barrier_all_on_stream(torch.cuda.current_stream())
    torch.cuda.synchronize()
    v23_local_ms = time_once(v23_func)
    nvshmem_barrier_all_on_stream(torch.cuda.current_stream())
    torch.cuda.synchronize()

    torch_max_ms = _distributed_max(torch_local_ms, group)
    v23_max_ms = _distributed_max(v23_local_ms, group)
    return {
        "timing_mode": "single_shot_eager",
        "torch_local_ms": torch_local_ms,
        "v23_local_ms": v23_local_ms,
        "torch_rank_max_ms": torch_max_ms,
        "v23_rank_max_ms": v23_max_ms,
        "rank_max_speedup": torch_max_ms / v23_max_ms,
    }


def correctness_metrics(actual: torch.Tensor, expected: torch.Tensor, *, atol: float, rtol: float) -> dict[str, Any]:
    if actual.shape != expected.shape:
        raise AssertionError(f"Shape mismatch: actual={tuple(actual.shape)}, expected={tuple(expected.shape)}")
    actual_float = actual.float()
    expected_float = expected.float()
    abs_diff = (actual_float - expected_float).abs()
    passed = bool(torch.allclose(actual_float, expected_float, atol=atol, rtol=rtol))
    metrics = {
        "passed": passed,
        "shape": list(actual.shape),
        "atol": atol,
        "rtol": rtol,
        "max_abs_diff": float(abs_diff.max().item()),
        "max_rel_diff": float((abs_diff / expected_float.abs().clamp_min(1e-12)).max().item()),
    }
    if not passed:
        raise AssertionError(f"V23 output mismatch: {metrics}")
    return metrics


def write_result(path: str | None, payload: dict[str, Any], *, rank: int) -> None:
    if path is None:
        return
    target = Path(path)
    if target.suffix.lower() != ".json":
        target.mkdir(parents=True, exist_ok=True)
        target = target / f"rank_{rank}.json"
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        if rank != 0:
            target = target.with_name(f"{target.stem}_rank_{rank}{target.suffix}")
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
