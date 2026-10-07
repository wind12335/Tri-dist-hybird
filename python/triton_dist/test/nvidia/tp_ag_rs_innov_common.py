"""Shared timing and result helpers for TP AllGather/ReduceScatter experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import torch

from triton_dist.profiler_utils import perf_func
from triton_dist.utils import nvshmem_barrier_all_on_stream


TensorFunc = Callable[[], torch.Tensor]
Hook = Callable[[], None]


def align_distributed(group) -> None:
    """Drain CUDA/NVSHMEM work and place all ranks at the same protocol round."""
    torch.distributed.barrier(group=group, device_ids=[torch.cuda.current_device()])
    nvshmem_barrier_all_on_stream(torch.cuda.current_stream())
    torch.cuda.synchronize()


def distributed_max(value: float, group) -> float:
    value_tensor = torch.tensor(value, dtype=torch.float64, device="cuda")
    torch.distributed.all_reduce(value_tensor, op=torch.distributed.ReduceOp.MAX, group=group)
    return float(value_tensor.item())


def make_cuda_graph(
    mempool,
    func: TensorFunc,
    *,
    warmup: int,
    before_each: Hook | None = None,
) -> torch.cuda.CUDAGraph:
    """Capture one invocation after a bounded, configurable warmup."""
    if warmup < 0:
        raise ValueError(f"CUDA Graph warmup must be non-negative, got {warmup}.")

    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(capture_stream):
        for _ in range(warmup):
            if before_each is not None:
                before_each()
            func()
    capture_stream.synchronize()

    if before_each is not None:
        before_each()
        torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, pool=mempool):
        func()
    return graph


def time_iterations(
    func: TensorFunc,
    *,
    group,
    warmup: int,
    iters: int,
    synchronize_each_iter: bool,
    before_each: Hook | None = None,
) -> float:
    """Time one rank locally, optionally isolating every distributed invocation."""
    if warmup < 0:
        raise ValueError(f"warmup must be non-negative, got {warmup}.")
    if iters <= 0:
        raise ValueError(f"iters must be positive, got {iters}.")

    if not synchronize_each_iter:
        if before_each is not None:
            before_each()
            torch.cuda.synchronize()
        _, local_ms = perf_func(func, iters=iters, warmup_iters=warmup)
        return float(local_ms)

    for _ in range(warmup):
        align_distributed(group)
        if before_each is not None:
            before_each()
            torch.cuda.synchronize()
        func()
        torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    elapsed_ms = 0.0
    for _ in range(iters):
        align_distributed(group)
        if before_each is not None:
            before_each()
            torch.cuda.synchronize()
        start.record()
        func()
        stop.record()
        stop.synchronize()
        elapsed_ms += float(start.elapsed_time(stop))
    align_distributed(group)
    return elapsed_ms / iters


def run_pair(
    torch_func: TensorFunc,
    selected_func: TensorFunc,
    *,
    group,
    warmup: int,
    iters: int,
    use_cuda_graph: bool,
    graph_warmup: int,
    synchronize_each_iter: bool,
    before_torch_mode: Hook | None = None,
    before_selected_mode: Hook | None = None,
    before_torch_each: Hook | None = None,
    before_selected_each: Hook | None = None,
) -> tuple[dict[str, float | str | bool], tuple[Any, ...]]:
    """Measure Torch and selected AG/RS paths with a common rank-max protocol."""
    torch_runner = torch_func
    selected_runner = selected_func
    graph_state: tuple[Any, ...] = ()
    used_cuda_graph = False

    if use_cuda_graph:
        mempool = torch.cuda.graph_pool_handle()
        try:
            if before_torch_mode is not None:
                before_torch_mode()
            align_distributed(group)
            torch_graph = make_cuda_graph(
                mempool,
                torch_func,
                warmup=graph_warmup,
                before_each=before_torch_each,
            )

            if before_selected_mode is not None:
                before_selected_mode()
            align_distributed(group)
            selected_graph = make_cuda_graph(
                mempool,
                selected_func,
                warmup=graph_warmup,
                before_each=before_selected_each,
            )
            torch_runner = torch_graph.replay
            selected_runner = selected_graph.replay
            graph_state = (torch_graph, selected_graph, mempool)
            used_cuda_graph = True
        except RuntimeError:
            # The caller prints the experiment context; keep this helper focused
            # on returning a safe eager fallback rather than hiding the failure.
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            graph_state = ()

    if before_torch_mode is not None:
        before_torch_mode()
    align_distributed(group)
    torch_local_ms = time_iterations(
        torch_runner,
        group=group,
        warmup=warmup,
        iters=iters,
        synchronize_each_iter=synchronize_each_iter,
        before_each=before_torch_each,
    )

    if before_selected_mode is not None:
        before_selected_mode()
    align_distributed(group)
    selected_local_ms = time_iterations(
        selected_runner,
        group=group,
        warmup=warmup,
        iters=iters,
        synchronize_each_iter=synchronize_each_iter,
        before_each=before_selected_each,
    )

    torch_rank_max_ms = distributed_max(torch_local_ms, group)
    selected_rank_max_ms = distributed_max(selected_local_ms, group)
    timing_prefix = "synchronized_" if synchronize_each_iter else ""
    timing_backend = "cuda_graph" if used_cuda_graph else "eager"
    return ({
        "timing_mode": f"{timing_prefix}{timing_backend}",
        "cuda_graph_requested": bool(use_cuda_graph),
        "cuda_graph_used": used_cuda_graph,
        "torch_local_ms": torch_local_ms,
        "selected_local_ms": selected_local_ms,
        "local_speedup": torch_local_ms / selected_local_ms,
        "torch_rank_max_ms": torch_rank_max_ms,
        "selected_rank_max_ms": selected_rank_max_ms,
        "rank_max_speedup": torch_rank_max_ms / selected_rank_max_ms,
    }, graph_state)


def gather_rank_payloads(payload: dict[str, Any], *, group, world_size: int) -> list[dict[str, Any]]:
    gathered: list[dict[str, Any] | None] = [None for _ in range(world_size)]
    torch.distributed.all_gather_object(gathered, payload, group=group)
    return [item for item in gathered if item is not None]


def write_ranked_json(
    path: str | None,
    payload: dict[str, Any],
    *,
    group,
    rank: int,
    world_size: int,
    stem: str,
) -> None:
    if path is None:
        return

    gathered = gather_rank_payloads(payload, group=group, world_size=world_size)
    target = Path(path)
    if target.suffix.lower() == ".json":
        output_dir = target.parent
        rank_target = target if rank == 0 else target.with_name(f"{target.stem}_rank_{rank}.json")
        summary_target = target
    else:
        output_dir = target
        rank_target = output_dir / f"{stem}_rank_{rank}.json"
        summary_target = output_dir / f"{stem}.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    rank_target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if rank == 0:
        summary_target.write_text(json.dumps({"ranks": gathered}, indent=2), encoding="utf-8")
