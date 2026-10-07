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

import dataclasses
from typing import Dict, List, Optional, Tuple

import torch
import triton
import triton.language as tl
import triton_dist
import triton_dist.tune
from cuda import cudart

from triton_dist.kernels.nvidia.common_ops import _set_signal_cuda, _wait_eq_cuda
from triton_dist.kernels.nvidia.gemm import get_config_space
from triton_dist.kernels.nvidia.gemm_reduce_scatter import _matmul_launch_metadata, update_triton_config
from triton_dist.kernels.nvidia.new_3rdgemm_reducescatterv3 import swizzle_2d
from triton_dist.utils import (
    CUDA_CHECK,
    NVSHMEM_SIGNAL_DTYPE,
    get_device_max_shared_memory_size,
    has_fullmesh_nvlink,
    nvshmem_barrier_all_on_stream,
    nvshmem_create_tensors,
    nvshmem_free_tensor_sync,
)


TaskKey = Tuple[int, int, int]


def _round_up(x: int, align: int) -> int:
    return ((x + align - 1) // align) * align


def _auto_chunk_rows(
    max_m: int,
    *,
    target_chunks: int,
    min_chunk_rows: int,
    align: int = 256,
) -> int:
    rows = triton.cdiv(max_m, target_chunks)
    rows = max(rows, min_chunk_rows)
    rows = min(rows, max_m)
    rows = _round_up(rows, align)
    return min(rows, max_m)


@triton_dist.jit(launch_metadata=_matmul_launch_metadata)
def kernel_gemm_ar_producer_windowed_chunk_panel_v23(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    chunk_row_start,
    rows_this_panel,
    band_col_start,
    band_cols,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(rows_this_panel, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(band_cols, BLOCK_SIZE_N)
    pid_m, pid_n = swizzle_2d(pid, num_pid_m, num_pid_n, GROUP_SIZE_M)

    offs_am = chunk_row_start + pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = band_col_start + pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    if a_ptr.dtype.element_ty == tl.int8:
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.int32)
    else:
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=(offs_am[:, None] < M) & (offs_k[None, :] < K - k * BLOCK_SIZE_K), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K - k * BLOCK_SIZE_K) & (offs_bn[None, :] < N), other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    local_offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    local_offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * local_offs_m[:, None] + stride_cn * local_offs_n[None, :]
    c_mask = (local_offs_m[:, None] < rows_this_panel) & (local_offs_n[None, :] < band_cols)
    tl.store(c_ptrs, accumulator.to(c_ptr.dtype.element_ty), mask=c_mask)


@triton.jit(do_not_specialize=["local_rank"])
def kernel_reduce_window_slot_from_scatter_with_local_v23(
    scatter_ptr,
    local_ptr,
    out_ptr,
    SLOT_ROWS,
    N,
    local_rank,
    rows_to_reduce,
    stride_scatter_m,
    stride_scatter_n,
    stride_local_m,
    stride_local_n,
    stride_outm,
    stride_outn,
    NUM_SPLITS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid = tl.num_programs(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_m = tl.cdiv(rows_to_reduce, BLOCK_SIZE_M)
    total_tiles = num_pid_m * num_pid_n
    for tile_id in range(pid, total_tiles, num_pid):
        pid_m = tile_id // num_pid_n
        pid_n = tile_id % num_pid_n
        offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        mask = (offs_m[:, None] < rows_to_reduce) & (offs_n[None, :] < N)

        local_ptrs = local_ptr + offs_m[:, None] * stride_local_m + offs_n[None, :] * stride_local_n
        acc = tl.load(local_ptrs, mask=mask, other=0.0).to(tl.float32)
        for split in range(NUM_SPLITS):
            if split != local_rank:
                src_rows = split * SLOT_ROWS + offs_m
                ptrs = scatter_ptr + src_rows[:, None] * stride_scatter_m + offs_n[None, :] * stride_scatter_n
                acc += tl.load(ptrs, mask=mask, other=0.0)

        out_ptrs = out_ptr + offs_m[:, None] * stride_outm + offs_n[None, :] * stride_outn
        tl.store(out_ptrs, acc.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def kernel_accumulate_strided_inplace_v23(
    src_ptr,
    out_ptr,
    M,
    N,
    stride_src_m,
    stride_src_n,
    stride_outm,
    stride_outn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid = tl.num_programs(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    total_tiles = num_pid_m * num_pid_n
    for tile_id in range(pid, total_tiles, num_pid):
        pid_m = tile_id // num_pid_n
        pid_n = tile_id % num_pid_n
        offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

        src_ptrs = src_ptr + offs_m[:, None] * stride_src_m + offs_n[None, :] * stride_src_n
        out_ptrs = out_ptr + offs_m[:, None] * stride_outm + offs_n[None, :] * stride_outn
        src = tl.load(src_ptrs, mask=mask, other=0.0).to(tl.float32)
        dst = tl.load(out_ptrs, mask=mask, other=0.0).to(tl.float32)
        tl.store(out_ptrs, (dst + src).to(out_ptr.dtype.element_ty), mask=mask)


@dataclasses.dataclass
class CompactStripeTaskMetaV23:
    slot_id: int
    ticket: int
    prev_free_ticket: int


@dataclasses.dataclass
class CompactStripeAllReduceSlotV23:
    slot_id: int
    stream: torch.cuda.Stream
    done_event: torch.cuda.Event
    task_id_host: int = -1


@dataclasses.dataclass
class FrontierWindowedPanelGEMMARContextV23:
    rank: int
    world_size: int
    local_world_size: int
    output_dtype: torch.dtype
    max_M: int
    N: int
    chunk_rows: int
    stripe_rows: int
    max_stripes_per_chunk: int
    num_chunks: int
    active_chunk_window: int
    n_bands: int
    max_band_cols: int
    frontier_chunks: int
    stage_slots: int
    num_comm_sms: int
    gemm_out: torch.Tensor
    output_buf: torch.Tensor
    stripe_ready_buf: torch.Tensor
    scatter_bufs: List[torch.Tensor]
    arrival_flag_bufs: List[torch.Tensor]
    free_flag_bufs: List[torch.Tensor]
    comm_streams: List[torch.cuda.Stream]
    reduce_slots: List[CompactStripeAllReduceSlotV23]
    round_base: int = 0
    initial_free_ticket_per_slot: List[int] = dataclasses.field(default_factory=list)
    prev_round_last_ticket_per_slot: List[int] = dataclasses.field(default_factory=list)
    current_round_tasks: List[TaskKey] = dataclasses.field(default_factory=list)
    current_round_meta: Dict[TaskKey, CompactStripeTaskMetaV23] = dataclasses.field(default_factory=dict)
    task_schedule_cache: Dict[int, List[TaskKey]] = dataclasses.field(default_factory=dict)
    panel_schedule_cache: Dict[int, List[tuple[int, int]]] = dataclasses.field(default_factory=dict)
    task_meta_cache: Dict[int, Dict[TaskKey, CompactStripeTaskMetaV23]] = dataclasses.field(default_factory=dict)
    # Opt-in experimental producer; legacy callers retain the v23 launch path.
    producer_order: str = "logical"

    @property
    def local_rank(self) -> int:
        return self.rank % self.local_world_size

    @property
    def nnodes(self) -> int:
        return self.world_size // self.local_world_size

    @property
    def local_scatter_buf(self) -> torch.Tensor:
        return self.scatter_bufs[self.local_rank]

    @property
    def local_arrival_flag(self) -> torch.Tensor:
        return self.arrival_flag_bufs[self.local_rank]

    @property
    def local_free_flag(self) -> torch.Tensor:
        return self.free_flag_bufs[self.local_rank]

    @property
    def compact_scatter_rows(self) -> int:
        return self.stage_slots * self.local_world_size * self.stripe_rows

    @property
    def baseline_scatter_rows_estimate(self) -> int:
        return self.active_chunk_window * self.n_bands * self.local_world_size * self.chunk_rows

    @property
    def scatter_compaction_ratio(self) -> float:
        return self.baseline_scatter_rows_estimate / max(self.compact_scatter_rows, 1)

    def get_gemm_out_buf(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return self.gemm_out[:input_tensor.shape[0]]

    def get_output_buf(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return self.output_buf[:input_tensor.shape[0]]

    def begin_round(self, M: int) -> None:
        max_round_tasks = self.num_chunks * self.n_bands * self.max_stripes_per_chunk
        self.round_base += max_round_tasks + 1
        self.initial_free_ticket_per_slot = self.prev_round_last_ticket_per_slot.copy()
        last_ticket_per_slot = self.prev_round_last_ticket_per_slot.copy()
        task_schedule_cache = getattr(self, "task_schedule_cache", None)
        if task_schedule_cache is None:
            task_schedule_cache = {}
            self.task_schedule_cache = task_schedule_cache
        self.current_round_tasks = task_schedule_cache.get(M)
        if self.current_round_tasks is None:
            self.current_round_tasks = _build_task_schedule_v23(self, M)
            task_schedule_cache[M] = self.current_round_tasks

        task_meta_cache = getattr(self, "task_meta_cache", None)
        if task_meta_cache is None:
            task_meta_cache = {}
            self.task_meta_cache = task_meta_cache
        self.current_round_meta = task_meta_cache.get(M)
        if self.current_round_meta is None:
            self.current_round_meta = {
                task: CompactStripeTaskMetaV23(
                    slot_id=task_index % self.stage_slots,
                    ticket=0,
                    prev_free_ticket=0,
                )
                for task_index, task in enumerate(self.current_round_tasks)
            }
            task_meta_cache[M] = self.current_round_meta

        for task_index, task in enumerate(self.current_round_tasks):
            meta = self.current_round_meta[task]
            slot_id = meta.slot_id
            ticket = self.round_base + task_index + 1
            meta.ticket = ticket
            meta.prev_free_ticket = last_ticket_per_slot[slot_id]
            last_ticket_per_slot[slot_id] = ticket
        self.prev_round_last_ticket_per_slot = last_ticket_per_slot
        for slot in self.reduce_slots:
            slot.task_id_host = -1

    def wait_all(self, current_stream: Optional[torch.cuda.Stream] = None) -> None:
        current_stream = current_stream or torch.cuda.current_stream()
        for stream in self.comm_streams:
            current_stream.wait_stream(stream)
        for slot in self.reduce_slots:
            current_stream.wait_stream(slot.stream)

    def finalize(self) -> None:
        self.wait_all()
        torch.cuda.synchronize()
        nvshmem_free_tensor_sync(self.scatter_bufs[self.local_rank])
        nvshmem_free_tensor_sync(self.arrival_flag_bufs[self.local_rank])
        nvshmem_free_tensor_sync(self.free_flag_bufs[self.local_rank])
        self.scatter_bufs = []
        self.arrival_flag_bufs = []
        self.free_flag_bufs = []
        self.comm_streams = []
        self.reduce_slots = []


def _num_sms_or_default_v23(num_sms: int) -> int:
    total_sms = torch.cuda.get_device_properties(0).multi_processor_count
    return max(1, min(total_sms, num_sms))


def _num_reduce_ctas_v23(rows: int, ncols: int, requested_sms: int) -> int:
    total_tiles = triton.cdiv(rows, 128) * triton.cdiv(ncols, 128)
    return max(1, min(total_tiles, _num_sms_or_default_v23(requested_sms)))


def _chunk_row_range_v23(ctx: FrontierWindowedPanelGEMMARContextV23, chunk_id: int, M: int) -> tuple[int, int]:
    row_start = chunk_id * ctx.chunk_rows
    row_end = min(row_start + ctx.chunk_rows, M)
    return row_start, row_end


def _band_col_range_v23(ctx: FrontierWindowedPanelGEMMARContextV23, band_id: int, N: int) -> tuple[int, int]:
    col_start = band_id * ctx.max_band_cols
    col_end = min(col_start + ctx.max_band_cols, N)
    return col_start, col_end


def _stripe_row_range_v23(
    ctx: FrontierWindowedPanelGEMMARContextV23,
    chunk_id: int,
    stripe_id: int,
    M: int,
) -> tuple[int, int]:
    chunk_row_start, chunk_row_end = _chunk_row_range_v23(ctx, chunk_id, M)
    stripe_row_start = chunk_row_start + stripe_id * ctx.stripe_rows
    stripe_row_end = min(stripe_row_start + ctx.stripe_rows, chunk_row_end)
    return stripe_row_start, stripe_row_end


def _num_runtime_stripes_v23(ctx: FrontierWindowedPanelGEMMARContextV23, chunk_id: int, M: int) -> int:
    row_start, row_end = _chunk_row_range_v23(ctx, chunk_id, M)
    return max(1, triton.cdiv(row_end - row_start, ctx.stripe_rows))


def _build_chunk_schedule_v23(num_runtime_chunks: int, frontier_chunks: int) -> List[int]:
    # This is intentionally the logical consumer order, NOT a frontier reorder.
    # AR has no RS-style output-owner segments. Stripe-frontier production below
    # exposes the first consumer dependency without changing slot/ticket order.
    frontier = list(range(min(num_runtime_chunks, max(0, frontier_chunks))))
    tail = list(range(len(frontier), num_runtime_chunks))
    return frontier + tail


def _build_task_schedule_v23(ctx: FrontierWindowedPanelGEMMARContextV23, M: int) -> List[TaskKey]:
    num_runtime_chunks = triton.cdiv(M, ctx.chunk_rows)
    chunk_schedule = _build_chunk_schedule_v23(num_runtime_chunks, ctx.frontier_chunks)
    tasks: List[TaskKey] = []
    for chunk_id in chunk_schedule:
        stripe_count = _num_runtime_stripes_v23(ctx, chunk_id, M)
        for band_id in range(ctx.n_bands):
            for stripe_id in range(stripe_count):
                tasks.append((chunk_id, band_id, stripe_id))
    return tasks


def _build_panel_schedule_v23(ctx: FrontierWindowedPanelGEMMARContextV23, M: int) -> List[tuple[int, int]]:
    num_runtime_chunks = triton.cdiv(M, ctx.chunk_rows)
    chunk_schedule = _build_chunk_schedule_v23(num_runtime_chunks, ctx.frontier_chunks)
    panels: List[tuple[int, int]] = []
    for chunk_id in chunk_schedule:
        for band_id in range(ctx.n_bands):
            panels.append((chunk_id, band_id))
    return panels


def _get_panel_schedule_v23(ctx: FrontierWindowedPanelGEMMARContextV23, M: int) -> List[tuple[int, int]]:
    panel_schedule_cache = getattr(ctx, "panel_schedule_cache", None)
    if panel_schedule_cache is None:
        panel_schedule_cache = {}
        ctx.panel_schedule_cache = panel_schedule_cache
    panels = panel_schedule_cache.get(M)
    if panels is None:
        panels = _build_panel_schedule_v23(ctx, M)
        panel_schedule_cache[M] = panels
    return panels


def _producer_window_panel_count_v23(
    ctx: FrontierWindowedPanelGEMMARContextV23,
    panel_count: int,
) -> int:
    return max(1, min(panel_count, ctx.active_chunk_window * ctx.n_bands))


def _uses_bulk_panel_handoff_v23(ctx: FrontierWindowedPanelGEMMARContextV23) -> bool:
    """Whether whole-panel consumers are delayed until all window producers submit."""
    return ctx.producer_order in ("panel_bulk", "panel_recursive_doubling_bulk_cublas")


def _run_whole_panel_pipeline_v23(ctx, panels, produce, consume) -> None:
    """Identical panel order, two host submission policies for controlled tests.

    All ranks use the same panel frontier; there is no RS output-owner reorder.
    A local consumer completion bounds producer lookahead. Remote slot reuse
    remains protected independently by the existing per-destination free ticket.
    Bulk modes delay handoff until a window's producers have been submitted;
    frontier modes submit each consumer immediately after its producer.
    """
    window = max(1, min(ctx.stage_slots, _producer_window_panel_count_v23(ctx, len(panels))))
    stream = torch.cuda.current_stream()

    def issue_producer(index):
        if index >= window:
            old_chunk, old_band = panels[index - window]
            old_meta = _task_meta_v23(ctx, old_chunk, old_band, 0)
            # This event has already been recorded by an earlier consume call.
            # CUDA wait_event captures that record even if the slot is reused.
            stream.wait_event(ctx.reduce_slots[old_meta.slot_id].done_event)
        produce(*panels[index])

    for start in range(0, len(panels), window):
        end = min(start + window, len(panels))
        if not _uses_bulk_panel_handoff_v23(ctx):
            for index in range(start, end):
                issue_producer(index)
                consume(*panels[index])
        else:
            for index in range(start, end):
                issue_producer(index)
            for index in range(start, end):
                consume(*panels[index])


def _task_meta_v23(
    ctx: FrontierWindowedPanelGEMMARContextV23,
    chunk_id: int,
    band_id: int,
    stripe_id: int,
) -> CompactStripeTaskMetaV23:
    return ctx.current_round_meta[(chunk_id, band_id, stripe_id)]


def _stripe_ready_view_v23(
    ctx: FrontierWindowedPanelGEMMARContextV23,
    chunk_id: int,
    band_id: int,
    stripe_id: int,
) -> torch.Tensor:
    idx = ((band_id * ctx.num_chunks + chunk_id) * ctx.max_stripes_per_chunk) + stripe_id
    return ctx.stripe_ready_buf[idx:idx + 1]


def _arrival_flag_view_v23(
    ctx: FrontierWindowedPanelGEMMARContextV23,
    src_local_rank: int,
    slot_id: int,
) -> torch.Tensor:
    idx = src_local_rank * ctx.stage_slots + slot_id
    return ctx.local_arrival_flag[idx:idx + 1]


def _free_flag_view_for_dest_v23(
    ctx: FrontierWindowedPanelGEMMARContextV23,
    dest_local_rank: int,
    slot_id: int,
) -> torch.Tensor:
    return ctx.free_flag_bufs[dest_local_rank][slot_id:slot_id + 1]


def _free_flag_view_local_v23(ctx: FrontierWindowedPanelGEMMARContextV23, slot_id: int) -> torch.Tensor:
    return ctx.local_free_flag[slot_id:slot_id + 1]


def _slot_rows_offset_v23(
    ctx: FrontierWindowedPanelGEMMARContextV23,
    slot_id: int,
    src_local_rank: int = 0,
) -> int:
    return (slot_id * ctx.local_world_size + src_local_rank) * ctx.stripe_rows


def _slot_view_v23(
    ctx: FrontierWindowedPanelGEMMARContextV23,
    slot_id: int,
    band_cols: int,
) -> torch.Tensor:
    row_start = _slot_rows_offset_v23(ctx, slot_id, 0)
    row_end = row_start + ctx.local_world_size * ctx.stripe_rows
    return ctx.local_scatter_buf[row_start:row_end, :band_cols]


def _slot_rank_view_v23(
    ctx: FrontierWindowedPanelGEMMARContextV23,
    slot_id: int,
    src_local_rank: int,
    band_cols: int,
    rows: int,
) -> torch.Tensor:
    row_start = _slot_rows_offset_v23(ctx, slot_id, src_local_rank)
    row_end = row_start + rows
    return ctx.local_scatter_buf[row_start:row_end, :band_cols]


def _comm_stream_for_copy_v23(
    ctx: FrontierWindowedPanelGEMMARContextV23,
    dest_local_rank: int,
    slot_id: int,
) -> torch.cuda.Stream:
    lane = (dest_local_rank + slot_id) % len(ctx.comm_streams)
    return ctx.comm_streams[lane]


def create_frontier_windowed_panel_gemm_ar_context_v23(
    max_M: int,
    N: int,
    rank: int,
    world_size: int,
    local_world_size: int,
    output_dtype: torch.dtype,
    *,
    chunk_rows: int = 0,
    stripe_rows: int = 0,
    target_chunks: int = 4,
    min_chunk_rows: int = 512,
    active_chunk_window: int = 2,
    n_bands: int = 1,
    frontier_chunks: int = 1,
    stage_slots: int = 4,
    num_comm_sms: int = 16,
    comm_lanes: int = 2,
    alloc_scatter_rows: int | None = None,
    alloc_max_band_cols: int | None = None,
    alloc_stage_slots: int | None = None,
    producer_order: str = "logical",
) -> FrontierWindowedPanelGEMMARContextV23:
    if producer_order not in ("logical", "stripe_frontier", "panel_bulk", "panel_frontier",
                              "panel_recursive_doubling", "panel_recursive_doubling_cublas",
                              "panel_recursive_doubling_bulk_cublas"):
        raise ValueError(f"Unknown AR producer_order: {producer_order}")
    if "recursive_doubling" in producer_order and world_size & (world_size - 1):
        raise ValueError("Recursive doubling requires a power-of-two world_size")
    if world_size != local_world_size:
        raise NotImplementedError("frontier_windowed_panel_gemm_allreduce_v23 currently supports single-node only")

    effective_chunk_rows = chunk_rows if chunk_rows > 0 else _auto_chunk_rows(
        max_M,
        target_chunks=target_chunks,
        min_chunk_rows=min_chunk_rows,
    )
    if producer_order.startswith("panel_") and stripe_rows != effective_chunk_rows:
        raise ValueError("Whole-panel AR requires stripe_rows == effective chunk_rows; "
                         "set both explicitly. stripe_rows is only a compatibility field in this mode.")
    effective_stripe_rows = max(1, min(
        stripe_rows if stripe_rows > 0 else min(256, effective_chunk_rows),
        effective_chunk_rows,
    ))
    num_chunks = triton.cdiv(max_M, effective_chunk_rows)
    max_stripes_per_chunk = triton.cdiv(effective_chunk_rows, effective_stripe_rows)
    active_chunk_window = max(1, min(active_chunk_window, num_chunks))
    n_bands = max(1, min(n_bands, N))
    max_band_cols = triton.cdiv(N, n_bands)
    total_max_tasks = max(1, num_chunks * n_bands * max_stripes_per_chunk)
    stage_slots = max(1, min(stage_slots, total_max_tasks))
    comm_lanes = max(1, min(comm_lanes, local_world_size))

    scatter_rows = stage_slots * local_world_size * effective_stripe_rows
    alloc_scatter_rows = max(scatter_rows, alloc_scatter_rows or 0)
    alloc_max_band_cols = max(max_band_cols, alloc_max_band_cols or 0)
    alloc_stage_slots = max(stage_slots, alloc_stage_slots or 0)

    scatter_bufs = None
    arrival_flag_bufs = None
    free_flag_bufs = None
    try:
        scatter_bufs = nvshmem_create_tensors((alloc_scatter_rows, alloc_max_band_cols), output_dtype, rank,
                                              local_world_size)
        arrival_flag_bufs = nvshmem_create_tensors((local_world_size * alloc_stage_slots,), NVSHMEM_SIGNAL_DTYPE, rank,
                                                   local_world_size)
        free_flag_bufs = nvshmem_create_tensors((alloc_stage_slots,), NVSHMEM_SIGNAL_DTYPE, rank, local_world_size)
        gemm_out = torch.empty((max_M, N), dtype=output_dtype, device="cuda")
        output_buf = torch.empty((max_M, N), dtype=output_dtype, device="cuda")
        stripe_ready_buf = torch.zeros((n_bands * num_chunks * max_stripes_per_chunk,), dtype=torch.int32, device="cuda")

        arrival_flag_bufs[rank % local_world_size].zero_()
        free_flag_bufs[rank % local_world_size].zero_()
        nvshmem_barrier_all_on_stream(torch.cuda.current_stream())

        comm_streams = [torch.cuda.Stream(priority=-1) for _ in range(comm_lanes)]
        reduce_slots = [
            CompactStripeAllReduceSlotV23(
                slot_id=slot_id,
                stream=torch.cuda.Stream(priority=-1),
                done_event=torch.cuda.Event(),
            )
            for slot_id in range(stage_slots)
        ]
        return FrontierWindowedPanelGEMMARContextV23(
            rank=rank,
            world_size=world_size,
            local_world_size=local_world_size,
            output_dtype=output_dtype,
            max_M=max_M,
            N=N,
            chunk_rows=effective_chunk_rows,
            stripe_rows=effective_stripe_rows,
            max_stripes_per_chunk=max_stripes_per_chunk,
            num_chunks=num_chunks,
            active_chunk_window=active_chunk_window,
            n_bands=n_bands,
            max_band_cols=max_band_cols,
            frontier_chunks=max(0, frontier_chunks),
            producer_order=producer_order,
            stage_slots=stage_slots,
            num_comm_sms=num_comm_sms,
            gemm_out=gemm_out,
            output_buf=output_buf,
            stripe_ready_buf=stripe_ready_buf,
            scatter_bufs=scatter_bufs,
            arrival_flag_bufs=arrival_flag_bufs,
            free_flag_bufs=free_flag_bufs,
            comm_streams=comm_streams,
            reduce_slots=reduce_slots,
            initial_free_ticket_per_slot=[0 for _ in range(stage_slots)],
            prev_round_last_ticket_per_slot=[0 for _ in range(stage_slots)],
        )
    except Exception:
        if free_flag_bufs is not None:
            nvshmem_free_tensor_sync(free_flag_bufs[rank % local_world_size])
        if arrival_flag_bufs is not None:
            nvshmem_free_tensor_sync(arrival_flag_bufs[rank % local_world_size])
        if scatter_bufs is not None:
            nvshmem_free_tensor_sync(scatter_bufs[rank % local_world_size])
        raise


def _signal_panel_stripes_ready_v23(
    ctx: FrontierWindowedPanelGEMMARContextV23,
    chunk_id: int,
    band_id: int,
    M: int,
) -> None:
    for stripe_id in range(_num_runtime_stripes_v23(ctx, chunk_id, M)):
        meta = _task_meta_v23(ctx, chunk_id, band_id, stripe_id)
        _set_signal_cuda(_stripe_ready_view_v23(ctx, chunk_id, band_id, stripe_id), meta.ticket, torch.cuda.current_stream())


def _panel_production_steps_v23(
    ctx: FrontierWindowedPanelGEMMARContextV23,
    chunk_id: int,
    M: int,
) -> List[tuple[int, int, int, int]]:
    """(global row start, end, first stripe, stripe end) for each launch.

    For each band of the first F chunks, finish and publish stripe 0 before
    launching the remaining panel rows. This is a two-phase *stripe* frontier,
    not RS's cross-owner chunk frontier. F=0 or a single-stripe panel is a no-op.
    Publication after each launch relies on same-stream CUDA ordering.
    """
    start, end = _chunk_row_range_v23(ctx, chunk_id, M)
    if start >= end:
        return []
    stripes = _num_runtime_stripes_v23(ctx, chunk_id, M)
    if ctx.producer_order == "stripe_frontier" and chunk_id < ctx.frontier_chunks and stripes > 1:
        return [(start, start + ctx.stripe_rows, 0, 1),
                (start + ctx.stripe_rows, end, 1, stripes)]
    return [(start, end, 0, stripes)]


def _launch_windowed_chunk_panel_producer_v23(
    A: torch.Tensor,
    B: torch.Tensor,
    ctx: FrontierWindowedPanelGEMMARContextV23,
    gemm_config: triton.Config,
    chunk_id: int,
    band_id: int,
    *,
    signal_tasks: bool = True,
) -> None:
    for row_start, row_end, first_stripe, stripe_end in _panel_production_steps_v23(ctx, chunk_id, A.shape[0]):
        _launch_panel_rows_v23(A, B, ctx, gemm_config, row_start, row_end, band_id)
        if signal_tasks:
            for stripe_id in range(first_stripe, stripe_end):
                meta = _task_meta_v23(ctx, chunk_id, band_id, stripe_id)
                _set_signal_cuda(_stripe_ready_view_v23(ctx, chunk_id, band_id, stripe_id),
                                 meta.ticket, torch.cuda.current_stream())


def _launch_panel_rows_v23(
    A: torch.Tensor,
    B: torch.Tensor,
    ctx: FrontierWindowedPanelGEMMARContextV23,
    gemm_config: triton.Config,
    chunk_row_start: int,
    chunk_row_end: int,
    band_id: int,
) -> None:
    M, K = A.shape
    _, N = B.shape
    col_start, col_end = _band_col_range_v23(ctx, band_id, N)
    chunk_rows = chunk_row_end - chunk_row_start
    band_cols = col_end - col_start
    if chunk_rows <= 0 or band_cols <= 0:
        return

    gemm_out = ctx.get_gemm_out_buf(A)
    out_chunk = gemm_out[chunk_row_start:chunk_row_end, col_start:col_end]
    if ctx.producer_order.endswith("_cublas"):
        torch.mm(A[chunk_row_start:chunk_row_end], B[:, col_start:col_end], out=out_chunk)
        return
    grid = (
        triton.cdiv(chunk_rows, gemm_config.kwargs["BLOCK_SIZE_M"]) *
        triton.cdiv(band_cols, gemm_config.kwargs["BLOCK_SIZE_N"]),
    )
    kernel_gemm_ar_producer_windowed_chunk_panel_v23[grid](
        A,
        B,
        out_chunk,
        M,
        N,
        K,
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(1),
        out_chunk.stride(0),
        out_chunk.stride(1),
        chunk_row_start,
        chunk_rows,
        col_start,
        band_cols,
        **gemm_config.all_kwargs(),
    )


def _issue_stripe_panel_copies_from_local_tensor_v23(
    local_tensor: torch.Tensor,
    ctx: FrontierWindowedPanelGEMMARContextV23,
    chunk_id: int,
    band_id: int,
    stripe_id: int,
) -> None:
    if "recursive_doubling" in ctx.producer_order:
        # This mode sends intermediate sums in its consumer, not all peers'
        # raw contributions here.
        return
    M, N = local_tensor.shape
    stripe_row_start, stripe_row_end = _stripe_row_range_v23(ctx, chunk_id, stripe_id, M)
    col_start, col_end = _band_col_range_v23(ctx, band_id, N)
    stripe_rows = stripe_row_end - stripe_row_start
    band_cols = col_end - col_start
    if stripe_rows <= 0 or band_cols <= 0:
        return

    meta = _task_meta_v23(ctx, chunk_id, band_id, stripe_id)
    local_stripe = local_tensor[stripe_row_start:stripe_row_end, col_start:col_end]
    itemsize = local_tensor.element_size()
    row_nbytes = band_cols * itemsize

    for step in range(ctx.local_world_size):
        dest_local_rank = (ctx.local_rank + step + 1) % ctx.local_world_size
        if dest_local_rank == ctx.local_rank:
            continue

        scatter_stream = _comm_stream_for_copy_v23(ctx, dest_local_rank, meta.slot_id)
        _wait_eq_cuda(_free_flag_view_for_dest_v23(ctx, dest_local_rank, meta.slot_id), meta.prev_free_ticket,
                      scatter_stream)
        _wait_eq_cuda(_stripe_ready_view_v23(ctx, chunk_id, band_id, stripe_id), meta.ticket, scatter_stream)

        remote_panel = ctx.scatter_bufs[dest_local_rank]
        remote_row_start = _slot_rows_offset_v23(ctx, meta.slot_id, ctx.local_rank)
        remote_buf_ptr = remote_panel.data_ptr() + (remote_row_start * remote_panel.stride(0)) * itemsize
        (err,) = cudart.cudaMemcpy2DAsync(
            remote_buf_ptr,
            remote_panel.stride(0) * itemsize,
            local_stripe.data_ptr(),
            local_stripe.stride(0) * itemsize,
            row_nbytes,
            stripe_rows,
            cudart.cudaMemcpyKind.cudaMemcpyDefault,
            scatter_stream.cuda_stream,
        )
        CUDA_CHECK(err)

        remote_arrival = ctx.arrival_flag_bufs[dest_local_rank][
            ctx.local_rank * ctx.stage_slots + meta.slot_id:
            ctx.local_rank * ctx.stage_slots + meta.slot_id + 1
        ]
        _set_signal_cuda(remote_arrival, meta.ticket, scatter_stream)


def _recursive_doubling_partners_v23(rank: int, world_size: int):
    if world_size < 1 or world_size & (world_size - 1) or not 0 <= rank < world_size:
        raise ValueError("Recursive doubling requires a valid rank and power-of-two world_size")
    return [rank ^ (1 << phase) for phase in range(world_size.bit_length() - 1)]


def _enqueue_recursive_doubling_panel_v23(local_tensor, ctx, output, chunk_id, band_id):
    """XOR partner exchanges; every rank retains the full panel.

    A phase's outgoing read finishes before its input is updated in-place.
    Partners differ in each phase, so source-indexed arrival cells identify the
    phase without extra flags. Free is published only after all phases finish.
    """
    M, N = local_tensor.shape
    row_start, row_end = _chunk_row_range_v23(ctx, chunk_id, M)
    col_start, col_end = _band_col_range_v23(ctx, band_id, N)
    rows, cols = row_end - row_start, col_end - col_start
    if rows <= 0 or cols <= 0:
        return
    meta = _task_meta_v23(ctx, chunk_id, band_id, 0)
    slot = ctx.reduce_slots[meta.slot_id]
    stream = slot.stream
    out = output[row_start:row_end, col_start:col_end]
    ctas = _num_reduce_ctas_v23(rows, cols, ctx.num_comm_sms)
    with torch.cuda.stream(stream):
        _wait_eq_cuda(_stripe_ready_view_v23(ctx, chunk_id, band_id, 0), meta.ticket, stream)
        out.copy_(local_tensor[row_start:row_end, col_start:col_end])
        for peer in _recursive_doubling_partners_v23(ctx.local_rank, ctx.local_world_size):
            send_stream = _comm_stream_for_copy_v23(ctx, peer, meta.slot_id)
            send_stream.wait_stream(stream)
            _wait_eq_cuda(_free_flag_view_for_dest_v23(ctx, peer, meta.slot_id), meta.prev_free_ticket, send_stream)
            remote = ctx.scatter_bufs[peer]
            remote_offset = _slot_rows_offset_v23(ctx, meta.slot_id, ctx.local_rank)
            itemsize = out.element_size()
            (err,) = cudart.cudaMemcpy2DAsync(
                remote.data_ptr() + remote_offset * remote.stride(0) * itemsize,
                remote.stride(0) * itemsize, out.data_ptr(), out.stride(0) * itemsize,
                cols * itemsize, rows, cudart.cudaMemcpyKind.cudaMemcpyDefault, send_stream.cuda_stream)
            CUDA_CHECK(err)
            arrival = ctx.arrival_flag_bufs[peer][ctx.local_rank * ctx.stage_slots + meta.slot_id:
                                                ctx.local_rank * ctx.stage_slots + meta.slot_id + 1]
            _set_signal_cuda(arrival, meta.ticket, send_stream)
            stream.wait_stream(send_stream)
            _wait_eq_cuda(_arrival_flag_view_v23(ctx, peer, meta.slot_id), meta.ticket, stream)
            incoming = _slot_rank_view_v23(ctx, meta.slot_id, peer, cols, rows)
            kernel_accumulate_strided_inplace_v23[(ctas,)](
                incoming, out, rows, cols, incoming.stride(0), incoming.stride(1),
                out.stride(0), out.stride(1), BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, num_warps=8)
        _set_signal_cuda(_free_flag_view_local_v23(ctx, meta.slot_id), meta.ticket, stream)
        slot.done_event.record(stream)


def _enqueue_windowed_stripe_allreduce_from_local_tensor_v23(
    local_tensor: torch.Tensor,
    ctx: FrontierWindowedPanelGEMMARContextV23,
    output: torch.Tensor,
    chunk_id: int,
    band_id: int,
    stripe_id: int,
) -> None:
    if "recursive_doubling" in ctx.producer_order:
        _enqueue_recursive_doubling_panel_v23(local_tensor, ctx, output, chunk_id, band_id)
        return
    M, N = local_tensor.shape
    row_start, row_end = _stripe_row_range_v23(ctx, chunk_id, stripe_id, M)
    col_start, col_end = _band_col_range_v23(ctx, band_id, N)
    rows = row_end - row_start
    band_cols = col_end - col_start
    if rows <= 0 or band_cols <= 0:
        return

    meta = _task_meta_v23(ctx, chunk_id, band_id, stripe_id)
    slot = ctx.reduce_slots[meta.slot_id]
    stream = slot.stream
    local_src = local_tensor[row_start:row_end, col_start:col_end]
    out_chunk = output[row_start:row_end, col_start:col_end]
    ctas = _num_reduce_ctas_v23(rows, band_cols, ctx.num_comm_sms)

    with torch.cuda.stream(stream):
        _wait_eq_cuda(_stripe_ready_view_v23(ctx, chunk_id, band_id, stripe_id), meta.ticket, stream)
        if ctx.producer_order in ("panel_bulk", "panel_frontier"):
            # AR needs every rank's contribution. Wait for the full arrival
            # frontier, then read each contribution once, accumulate in FP32,
            # and store once. The local scatter slice is deliberately skipped.
            for src_local_rank in range(ctx.local_world_size):
                if src_local_rank != ctx.local_rank:
                    _wait_eq_cuda(_arrival_flag_view_v23(ctx, src_local_rank, meta.slot_id), meta.ticket, stream)
            scatter = _slot_view_v23(ctx, meta.slot_id, band_cols)
            kernel_reduce_window_slot_from_scatter_with_local_v23[(ctas,)](
                scatter, local_src, out_chunk, ctx.stripe_rows, band_cols,
                ctx.local_rank, rows,
                scatter.stride(0), scatter.stride(1),
                local_src.stride(0), local_src.stride(1),
                out_chunk.stride(0), out_chunk.stride(1),
                NUM_SPLITS=ctx.local_world_size,
                BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, num_warps=8,
            )
            _set_signal_cuda(_free_flag_view_local_v23(ctx, meta.slot_id), meta.ticket, stream)
            slot.done_event.record(stream)
            return
        out_chunk.copy_(local_src)
        for src_local_rank in range(ctx.local_world_size):
            if src_local_rank == ctx.local_rank:
                continue
            _wait_eq_cuda(_arrival_flag_view_v23(ctx, src_local_rank, meta.slot_id), meta.ticket, stream)
            remote_src = _slot_rank_view_v23(ctx, meta.slot_id, src_local_rank, band_cols, rows)
            kernel_accumulate_strided_inplace_v23[(ctas,)](
                remote_src,
                out_chunk,
                rows,
                band_cols,
                remote_src.stride(0),
                remote_src.stride(1),
                out_chunk.stride(0),
                out_chunk.stride(1),
                BLOCK_SIZE_M=128,
                BLOCK_SIZE_N=128,
                num_warps=8,
            )
        _set_signal_cuda(_free_flag_view_local_v23(ctx, meta.slot_id), meta.ticket, stream)
        slot.done_event.record(stream)


def frontier_windowed_panel_gemm_allreduce_v23_key_fn(
    A: torch.Tensor,
    B: torch.Tensor,
    ctx: FrontierWindowedPanelGEMMARContextV23,
    *args,
    **kwargs,
):
    return (
        triton_dist.tune.to_hashable(A),
        triton_dist.tune.to_hashable(B),
        ctx.world_size,
        ctx.chunk_rows,
        ctx.stripe_rows,
        ctx.active_chunk_window,
        ctx.n_bands,
        ctx.frontier_chunks,
        ctx.producer_order,
        ctx.stage_slots,
        len(ctx.comm_streams),
        ctx.num_comm_sms,
    )


def frontier_windowed_panel_gemm_allreduce_v23_prune_fn(config, A, B, *args, **kwargs):
    ctx = args[0]
    itemsize = A.itemsize
    gemm_config = config["gemm_config"].all_kwargs()
    num_stages = gemm_config["num_stages"]
    block_size_m = gemm_config["BLOCK_SIZE_M"]
    block_size_n = gemm_config["BLOCK_SIZE_N"]
    block_size_k = gemm_config["BLOCK_SIZE_K"]
    shared_memory = (itemsize * block_size_m * block_size_k + itemsize * block_size_n * block_size_k) * num_stages
    return shared_memory < get_device_max_shared_memory_size(torch.cuda.current_device()) and block_size_m <= ctx.chunk_rows


def get_frontier_windowed_panel_gemm_allreduce_v23_config_space():
    return [{"gemm_config": config} for config in get_config_space(False)]


def frontier_windowed_panel_gemm_only_op_v23(
    A: torch.Tensor,
    B: torch.Tensor,
    ctx: FrontierWindowedPanelGEMMARContextV23,
    gemm_config: triton.Config,
) -> torch.Tensor:
    if ctx.nnodes != 1:
        raise NotImplementedError("frontier_windowed_panel_gemm_only_v23 currently supports single-node only")

    M, local_K = A.shape
    _, N = B.shape
    assert N == ctx.N, f"B should be of shape [{local_K}, {ctx.N}]"

    tuned_config = update_triton_config(M, N, local_K, A.dtype, ctx.world_size, ctx.local_world_size, gemm_config)
    if ctx.chunk_rows < tuned_config.kwargs["BLOCK_SIZE_M"]:
        raise ValueError(
            "frontier_windowed_panel_gemm_only_v23 requires chunk_rows >= BLOCK_SIZE_M, "
            f"but got chunk_rows={ctx.chunk_rows}, BLOCK_SIZE_M={tuned_config.kwargs['BLOCK_SIZE_M']}"
        )

    local_partial = ctx.get_gemm_out_buf(A)
    panel_schedule = _build_panel_schedule_v23(ctx, M)
    for chunk_id, band_id in panel_schedule:
        _launch_windowed_chunk_panel_producer_v23(A, B, ctx, tuned_config, chunk_id, band_id, signal_tasks=False)

    return local_partial[:M, :N]


def frontier_windowed_panel_gemm_allreduce_op_v23(
    A: torch.Tensor,
    B: torch.Tensor,
    ctx: FrontierWindowedPanelGEMMARContextV23,
    gemm_config: triton.Config,
    *,
    drain: bool = True,
) -> torch.Tensor:
    if ctx.producer_order.startswith("panel_") and not drain:
        raise NotImplementedError("Whole-panel AR currently requires drain=True to protect cross-round source reuse")
    if ctx.nnodes != 1:
        raise NotImplementedError("frontier_windowed_panel_gemm_allreduce_v23 currently supports single-node only")
    if not has_fullmesh_nvlink():
        raise NotImplementedError("frontier_windowed_panel_gemm_allreduce_v23 currently expects full-mesh NVLink")

    M, local_K = A.shape
    _, N = B.shape
    assert N == ctx.N, f"B should be of shape [{local_K}, {ctx.N}]"

    tuned_config = update_triton_config(M, N, local_K, A.dtype, ctx.world_size, ctx.local_world_size, gemm_config)
    if ctx.chunk_rows < tuned_config.kwargs["BLOCK_SIZE_M"]:
        raise ValueError(
            "frontier_windowed_panel_gemm_allreduce_v23 requires chunk_rows >= BLOCK_SIZE_M, "
            f"but got chunk_rows={ctx.chunk_rows}, BLOCK_SIZE_M={tuned_config.kwargs['BLOCK_SIZE_M']}"
        )

    output = ctx.get_output_buf(A)
    local_partial = ctx.get_gemm_out_buf(A)
    ctx.begin_round(M)
    panel_schedule = _get_panel_schedule_v23(ctx, M)
    if ctx.producer_order.startswith("panel_"):
        if ctx.stripe_rows != ctx.chunk_rows:
            raise ValueError("Whole-panel mode requires stripe_rows == chunk_rows")

        def produce(chunk_id, band_id):
            _launch_windowed_chunk_panel_producer_v23(A, B, ctx, tuned_config, chunk_id, band_id)

        def consume(chunk_id, band_id):
            _issue_stripe_panel_copies_from_local_tensor_v23(local_partial, ctx, chunk_id, band_id, 0)
            _enqueue_windowed_stripe_allreduce_from_local_tensor_v23(local_partial, ctx, output, chunk_id, band_id, 0)

        _run_whole_panel_pipeline_v23(ctx, panel_schedule, produce, consume)
        if drain:
            ctx.wait_all(torch.cuda.current_stream())
        return output
    producer_window_panels = _producer_window_panel_count_v23(ctx, len(panel_schedule))
    launched_panels: set[tuple[int, int]] = set()

    def _launch_panel_if_needed(panel_index: int) -> None:
        if panel_index >= len(panel_schedule):
            return
        chunk_id, band_id = panel_schedule[panel_index]
        panel = (chunk_id, band_id)
        if panel in launched_panels:
            return
        _launch_windowed_chunk_panel_producer_v23(A, B, ctx, tuned_config, chunk_id, band_id, signal_tasks=True)
        launched_panels.add(panel)

    for panel_index in range(producer_window_panels):
        _launch_panel_if_needed(panel_index)

    for panel_index, (chunk_id, band_id) in enumerate(panel_schedule):
        _launch_panel_if_needed(panel_index + producer_window_panels)
        for stripe_id in range(_num_runtime_stripes_v23(ctx, chunk_id, M)):
            _issue_stripe_panel_copies_from_local_tensor_v23(local_partial, ctx, chunk_id, band_id, stripe_id)
            _enqueue_windowed_stripe_allreduce_from_local_tensor_v23(local_partial, ctx, output, chunk_id, band_id,
                                                                     stripe_id)

    if drain:
        ctx.wait_all(torch.cuda.current_stream())
    return output


@triton_dist.tune.autotune(
    config_space=get_frontier_windowed_panel_gemm_allreduce_v23_config_space(),
    key_fn=frontier_windowed_panel_gemm_allreduce_v23_key_fn,
    prune_fn=frontier_windowed_panel_gemm_allreduce_v23_prune_fn,
)
def frontier_windowed_panel_gemm_allreduce_v23(
    A: torch.Tensor,
    B: torch.Tensor,
    ctx: FrontierWindowedPanelGEMMARContextV23,
    gemm_config: triton.Config,
    *,
    drain: bool = True,
):
    return frontier_windowed_panel_gemm_allreduce_op_v23(A, B, ctx, gemm_config, drain=drain)


def frontier_windowed_panel_allreduce_v23(
    input_tensor: torch.Tensor,
    ctx: FrontierWindowedPanelGEMMARContextV23,
    *,
    drain: bool = True,
) -> torch.Tensor:
    if ctx.producer_order.startswith("panel_") and not drain:
        raise NotImplementedError("Whole-panel AR currently requires drain=True")
    M, N = input_tensor.shape
    output = ctx.get_output_buf(input_tensor)
    ctx.begin_round(M)
    panel_schedule = _get_panel_schedule_v23(ctx, M)
    if ctx.producer_order.startswith("panel_"):
        if ctx.stripe_rows != ctx.chunk_rows:
            raise ValueError("Whole-panel mode requires stripe_rows == chunk_rows")

        def produce(chunk_id, band_id):
            _signal_panel_stripes_ready_v23(ctx, chunk_id, band_id, M)

        def consume(chunk_id, band_id):
            _issue_stripe_panel_copies_from_local_tensor_v23(input_tensor, ctx, chunk_id, band_id, 0)
            _enqueue_windowed_stripe_allreduce_from_local_tensor_v23(input_tensor, ctx, output, chunk_id, band_id, 0)

        _run_whole_panel_pipeline_v23(ctx, panel_schedule, produce, consume)
        if drain:
            ctx.wait_all(torch.cuda.current_stream())
        return output
    ready_window_panels = _producer_window_panel_count_v23(ctx, len(panel_schedule))
    signaled_panels: set[tuple[int, int]] = set()

    def _signal_panel_if_needed(panel_index: int) -> None:
        if panel_index >= len(panel_schedule):
            return
        chunk_id, band_id = panel_schedule[panel_index]
        panel = (chunk_id, band_id)
        if panel in signaled_panels:
            return
        _signal_panel_stripes_ready_v23(ctx, chunk_id, band_id, M)
        signaled_panels.add(panel)

    for panel_index in range(ready_window_panels):
        _signal_panel_if_needed(panel_index)

    for panel_index, (chunk_id, band_id) in enumerate(panel_schedule):
        _signal_panel_if_needed(panel_index + ready_window_panels)
        for stripe_id in range(_num_runtime_stripes_v23(ctx, chunk_id, M)):
            _issue_stripe_panel_copies_from_local_tensor_v23(input_tensor, ctx, chunk_id, band_id, stripe_id)
            _enqueue_windowed_stripe_allreduce_from_local_tensor_v23(input_tensor, ctx, output, chunk_id, band_id,
                                                                     stripe_id)

    if drain:
        ctx.wait_all(torch.cuda.current_stream())
    return output


__all__ = [
    "FrontierWindowedPanelGEMMARContextV23",
    "create_frontier_windowed_panel_gemm_ar_context_v23",
    "frontier_windowed_panel_allreduce_v23",
    "frontier_windowed_panel_gemm_allreduce_v23",
    "frontier_windowed_panel_gemm_only_op_v23",
    "frontier_windowed_panel_gemm_allreduce_op_v23",
]



