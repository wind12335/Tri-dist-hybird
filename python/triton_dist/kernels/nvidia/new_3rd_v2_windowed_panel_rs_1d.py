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
import os
from typing import List, Optional

import torch
import triton
import triton.language as tl
from cuda import cudart

from triton_dist.kernels.nvidia.common_ops import _set_signal_cuda, _wait_eq_cuda
from triton_dist.utils import (CUDA_CHECK, NVSHMEM_SIGNAL_DTYPE, has_fullmesh_nvlink, nvshmem_barrier_all_on_stream,
                               nvshmem_create_tensors, nvshmem_free_tensor_sync)


def _debug_enabled() -> bool:
    return os.environ.get("TRITON_DIST_NEW_3RD_V2_DEBUG", "0") == "1"


def _debug_log(ctx: "New3rdV2WindowedPanelRSContext", msg: str) -> None:
    if _debug_enabled():
        print(f"[new_3rd_v2][rank{ctx.rank}] {msg}", flush=True)


def _round_up(x: int, align: int) -> int:
    return ((x + align - 1) // align) * align


def _auto_chunk_rows(
    max_m_per_rank: int,
    *,
    target_chunks_per_rank: int,
    min_chunk_rows: int,
    align: int = 256,
) -> int:
    rows = triton.cdiv(max_m_per_rank, target_chunks_per_rank)
    rows = max(rows, min_chunk_rows)
    rows = min(rows, max_m_per_rank)
    rows = _round_up(rows, align)
    return min(rows, max_m_per_rank)


@triton.jit(do_not_specialize=["local_rank"])
def kernel_reduce_window_slot_from_scatter_with_local(
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
def kernel_reduce_window_slot_from_scatter(
    scatter_ptr,
    out_ptr,
    SLOT_ROWS,
    N,
    rows_to_reduce,
    stride_scatter_m,
    stride_scatter_n,
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

        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for split in range(NUM_SPLITS):
            src_rows = split * SLOT_ROWS + offs_m
            ptrs = scatter_ptr + src_rows[:, None] * stride_scatter_m + offs_n[None, :] * stride_scatter_n
            acc += tl.load(ptrs, mask=mask, other=0.0)

        out_ptrs = out_ptr + offs_m[:, None] * stride_outm + offs_n[None, :] * stride_outn
        tl.store(out_ptrs, acc.to(out_ptr.dtype.element_ty), mask=mask)


@dataclasses.dataclass
class WindowedReduceSlot:
    slot_id: int
    stream: torch.cuda.Stream
    done_event: torch.cuda.Event
    chunk_id_host: int = -1


@dataclasses.dataclass
class New3rdV2WindowedPanelRSContext:
    rank: int
    world_size: int
    local_world_size: int
    dtype: torch.dtype
    max_M: int
    N: int
    chunk_rows: int
    num_chunks: int
    active_chunk_window: int
    n_bands: int
    local_seed_direct: bool
    steady_sms: int
    tail_sms: int
    stage_slots: int
    tail_chunk_window: int
    chunk_signal: torch.Tensor
    scatter_bufs: List[torch.Tensor]
    arrival_flag_bufs: List[torch.Tensor]
    free_flag_bufs: List[torch.Tensor]
    comm_streams: List[torch.cuda.Stream]
    reduce_slots: List[WindowedReduceSlot]
    signal_value: int = 0
    round_base: int = 0
    initial_free_ticket_per_window_slot: List[int] = dataclasses.field(default_factory=list)
    prev_round_last_ticket_per_window_slot: List[int] = dataclasses.field(default_factory=list)

    @property
    def local_rank(self) -> int:
        return self.rank % self.local_world_size

    @property
    def nnodes(self) -> int:
        return self.world_size // self.local_world_size

    @property
    def node_id(self) -> int:
        return self.rank // self.local_world_size

    @property
    def arrival_flag_buf(self) -> torch.Tensor:
        return self.arrival_flag_bufs[self.local_rank]

    @property
    def free_flag_buf(self) -> torch.Tensor:
        return self.free_flag_bufs[self.local_rank]

    @property
    def local_scatter_buf(self) -> torch.Tensor:
        return self.scatter_bufs[self.local_rank]

    def begin_round(self, num_runtime_chunks: int) -> int:
        self.signal_value += 1
        self.round_base += self.num_chunks + 1
        self.initial_free_ticket_per_window_slot = self.prev_round_last_ticket_per_window_slot.copy()
        for slot_id in range(self.active_chunk_window):
            if slot_id < num_runtime_chunks:
                last_chunk = slot_id + self.active_chunk_window * ((num_runtime_chunks - 1 - slot_id) //
                                                                   self.active_chunk_window)
                self.prev_round_last_ticket_per_window_slot[slot_id] = self.ticket_for_chunk(last_chunk)
        for slot in self.reduce_slots:
            slot.chunk_id_host = -1
        return self.signal_value

    def reset_runtime_state(self) -> None:
        self.signal_value = 0
        self.round_base = 0
        self.initial_free_ticket_per_window_slot = [0 for _ in range(self.active_chunk_window)]
        self.prev_round_last_ticket_per_window_slot = [0 for _ in range(self.active_chunk_window)]
        self.chunk_signal.zero_()
        self.arrival_flag_buf.zero_()
        self.free_flag_buf.zero_()
        for slot in self.reduce_slots:
            slot.chunk_id_host = -1
        nvshmem_barrier_all_on_stream(torch.cuda.current_stream())

    def ticket_for_chunk(self, chunk_id: int) -> int:
        return self.round_base + chunk_id + 1

    def wait_all(self, current_stream: Optional[torch.cuda.Stream] = None) -> None:
        current_stream = current_stream or torch.cuda.current_stream()
        for stream in self.comm_streams:
            current_stream.wait_stream(stream)
        for slot in self.reduce_slots:
            current_stream.wait_stream(slot.stream)

    def finalize(self) -> None:
        nvshmem_free_tensor_sync(self.scatter_bufs[self.local_rank])
        nvshmem_free_tensor_sync(self.arrival_flag_buf)
        nvshmem_free_tensor_sync(self.free_flag_buf)


def _num_sms_or_default(num_sms: int) -> int:
    total_sms = torch.cuda.get_device_properties(0).multi_processor_count
    return max(1, min(total_sms, num_sms))


def _num_reduce_ctas(rows: int, ncols: int, requested_sms: int) -> int:
    total_tiles = triton.cdiv(rows, 128) * triton.cdiv(ncols, 128)
    return max(1, min(total_tiles, _num_sms_or_default(requested_sms)))


def _chunk_row_range(ctx: New3rdV2WindowedPanelRSContext, chunk_id: int, m_per_rank: int) -> tuple[int, int]:
    row_start = chunk_id * ctx.chunk_rows
    row_end = min(row_start + ctx.chunk_rows, m_per_rank)
    return row_start, row_end


def _chunk_signal_view(ctx: New3rdV2WindowedPanelRSContext, dest_rank: int, chunk_id: int) -> torch.Tensor:
    idx = dest_rank * ctx.num_chunks + chunk_id
    return ctx.chunk_signal[idx:idx + 1]


def _arrival_flag_view(ctx: New3rdV2WindowedPanelRSContext, src_local_rank: int, window_slot: int) -> torch.Tensor:
    idx = src_local_rank * ctx.active_chunk_window + window_slot
    return ctx.arrival_flag_buf[idx:idx + 1]


def _free_flag_view_for_dest(ctx: New3rdV2WindowedPanelRSContext, dest_local_rank: int, window_slot: int) -> torch.Tensor:
    return ctx.free_flag_bufs[dest_local_rank][window_slot:window_slot + 1]


def _free_flag_view_local(ctx: New3rdV2WindowedPanelRSContext, window_slot: int) -> torch.Tensor:
    return ctx.free_flag_buf[window_slot:window_slot + 1]


def _slot_for_chunk(ctx: New3rdV2WindowedPanelRSContext, chunk_id: int) -> WindowedReduceSlot:
    return ctx.reduce_slots[chunk_id % ctx.stage_slots]


def _comm_stream_for_copy(ctx: New3rdV2WindowedPanelRSContext, dest_local_rank: int, chunk_id: int) -> torch.cuda.Stream:
    lane = (dest_local_rank + chunk_id) % len(ctx.comm_streams)
    return ctx.comm_streams[lane]


def _window_slot_rows_offset(ctx: New3rdV2WindowedPanelRSContext, window_slot: int, src_local_rank: int = 0) -> int:
    return (window_slot * ctx.local_world_size + src_local_rank) * ctx.chunk_rows


def _window_slot_view(ctx: New3rdV2WindowedPanelRSContext, window_slot: int) -> torch.Tensor:
    row_start = _window_slot_rows_offset(ctx, window_slot, 0)
    row_end = row_start + ctx.local_world_size * ctx.chunk_rows
    return ctx.local_scatter_buf[row_start:row_end]


def _expected_free_ticket(ctx: New3rdV2WindowedPanelRSContext, chunk_id: int, window_slot: int) -> int:
    prev_chunk = chunk_id - ctx.active_chunk_window
    if prev_chunk >= 0:
        return ctx.ticket_for_chunk(prev_chunk)
    return ctx.initial_free_ticket_per_window_slot[window_slot]


def _reduce_window_slot_from_scatter_direct(
    scatter_slot: torch.Tensor,
    local_src: Optional[torch.Tensor],
    output_chunk: torch.Tensor,
    *,
    slot_rows: int,
    local_rank: int,
    num_splits: int,
    rows_to_reduce: int,
    num_sms: int,
) -> None:
    ctas = _num_reduce_ctas(rows_to_reduce, output_chunk.shape[1], num_sms)
    if local_src is None:
        kernel_reduce_window_slot_from_scatter[(ctas,)](
            scatter_slot,
            output_chunk,
            slot_rows,
            output_chunk.shape[1],
            rows_to_reduce,
            scatter_slot.stride(0),
            scatter_slot.stride(1),
            output_chunk.stride(0),
            output_chunk.stride(1),
            NUM_SPLITS=num_splits,
            BLOCK_SIZE_M=128,
            BLOCK_SIZE_N=128,
            num_warps=8,
        )
        return

    kernel_reduce_window_slot_from_scatter_with_local[(ctas,)](
        scatter_slot,
        local_src,
        output_chunk,
        slot_rows,
        output_chunk.shape[1],
        local_rank,
        rows_to_reduce,
        scatter_slot.stride(0),
        scatter_slot.stride(1),
        local_src.stride(0),
        local_src.stride(1),
        output_chunk.stride(0),
        output_chunk.stride(1),
        NUM_SPLITS=num_splits,
        BLOCK_SIZE_M=128,
        BLOCK_SIZE_N=128,
        num_warps=8,
    )


def create_new_3rd_v2_windowed_panel_rs_context(
    max_M: int,
    N: int,
    rank: int,
    world_size: int,
    local_world_size: int,
    dtype: torch.dtype,
    *,
    chunk_rows: int = 0,
    target_chunks_per_rank: int = 2,
    min_chunk_rows: int = 512,
    active_chunk_window: int = 2,
    comm_lanes: int = 2,
    n_bands: int = 1,
    steady_sms: int = 6,
    tail_sms: int = 12,
    stage_slots: int = 4,
    tail_chunk_window: int = 1,
    local_seed_direct: bool = True,
) -> New3rdV2WindowedPanelRSContext:
    if world_size != local_world_size:
        raise NotImplementedError("windowed_panel_v2 currently supports single-node only")
    if n_bands != 1:
        raise NotImplementedError("Phase 1 windowed_panel_v2 currently supports only n_bands=1")

    max_m_per_rank = max_M // world_size
    effective_chunk_rows = chunk_rows if chunk_rows > 0 else _auto_chunk_rows(
        max_m_per_rank,
        target_chunks_per_rank=target_chunks_per_rank,
        min_chunk_rows=min_chunk_rows,
    )
    num_chunks = triton.cdiv(max_m_per_rank, effective_chunk_rows)
    active_chunk_window = max(1, min(active_chunk_window, num_chunks))
    stage_slots = max(1, min(stage_slots, num_chunks))
    comm_lanes = max(1, min(comm_lanes, local_world_size))

    scatter_rows = active_chunk_window * local_world_size * effective_chunk_rows
    scatter_bufs = nvshmem_create_tensors((scatter_rows, N), dtype, rank, local_world_size)
    arrival_flag_bufs = nvshmem_create_tensors((local_world_size * active_chunk_window,), NVSHMEM_SIGNAL_DTYPE, rank,
                                               local_world_size)
    free_flag_bufs = nvshmem_create_tensors((active_chunk_window,), NVSHMEM_SIGNAL_DTYPE, rank, local_world_size)
    chunk_signal = torch.zeros((world_size * num_chunks,), dtype=torch.int32, device="cuda")

    arrival_flag_bufs[rank % local_world_size].zero_()
    free_flag_bufs[rank % local_world_size].zero_()
    nvshmem_barrier_all_on_stream(torch.cuda.current_stream())

    comm_streams = [torch.cuda.Stream(priority=-1) for _ in range(comm_lanes)]
    reduce_slots = [
        WindowedReduceSlot(slot_id=slot_id, stream=torch.cuda.Stream(priority=-1), done_event=torch.cuda.Event())
        for slot_id in range(stage_slots)
    ]
    return New3rdV2WindowedPanelRSContext(
        rank=rank,
        world_size=world_size,
        local_world_size=local_world_size,
        dtype=dtype,
        max_M=max_M,
        N=N,
        chunk_rows=effective_chunk_rows,
        num_chunks=num_chunks,
        active_chunk_window=active_chunk_window,
        n_bands=n_bands,
        local_seed_direct=local_seed_direct,
        steady_sms=steady_sms,
        tail_sms=tail_sms,
        stage_slots=stage_slots,
        tail_chunk_window=max(1, tail_chunk_window),
        chunk_signal=chunk_signal,
        scatter_bufs=scatter_bufs,
        arrival_flag_bufs=arrival_flag_bufs,
        free_flag_bufs=free_flag_bufs,
        comm_streams=comm_streams,
        reduce_slots=reduce_slots,
        initial_free_ticket_per_window_slot=[0 for _ in range(active_chunk_window)],
        prev_round_last_ticket_per_window_slot=[0 for _ in range(active_chunk_window)],
    )


def _issue_windowed_chunk_scatter_and_arrival(
    input_intra_node: torch.Tensor,
    ctx: New3rdV2WindowedPanelRSContext,
) -> None:
    local_rank = ctx.local_rank
    local_world_size = ctx.local_world_size
    M, N = input_intra_node.shape
    m_per_rank = M // local_world_size
    nbytes_per_row = N * input_intra_node.dtype.itemsize
    local_buf_base_ptr = input_intra_node.data_ptr()
    num_runtime_chunks = triton.cdiv(m_per_rank, ctx.chunk_rows)

    for chunk_id in range(num_runtime_chunks):
        row_start, row_end = _chunk_row_range(ctx, chunk_id, m_per_rank)
        rows = row_end - row_start
        if rows <= 0:
            continue

        chunk_nbytes = rows * nbytes_per_row
        window_slot = chunk_id % ctx.active_chunk_window
        ticket = ctx.ticket_for_chunk(chunk_id)
        free_ticket = _expected_free_ticket(ctx, chunk_id, window_slot)
        for step in range(local_world_size):
            dest_local_rank = (local_rank + step + 1) % local_world_size
            if ctx.local_seed_direct and dest_local_rank == local_rank:
                continue

            scatter_stream = _comm_stream_for_copy(ctx, dest_local_rank, chunk_id)
            _wait_eq_cuda(_free_flag_view_for_dest(ctx, dest_local_rank, window_slot), free_ticket, scatter_stream)
            _wait_eq_cuda(_chunk_signal_view(ctx, dest_local_rank, chunk_id), ctx.signal_value, scatter_stream)

            remote_row_start = _window_slot_rows_offset(ctx, window_slot, local_rank)
            remote_buf_ptr = ctx.scatter_bufs[dest_local_rank].data_ptr() + remote_row_start * nbytes_per_row
            local_buf_ptr = local_buf_base_ptr + (dest_local_rank * m_per_rank + row_start) * nbytes_per_row
            (err,) = cudart.cudaMemcpyAsync(
                remote_buf_ptr,
                local_buf_ptr,
                chunk_nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyDefault,
                scatter_stream.cuda_stream,
            )
            CUDA_CHECK(err)
            remote_arrival = ctx.arrival_flag_bufs[dest_local_rank][local_rank * ctx.active_chunk_window +
                                                                   window_slot:local_rank * ctx.active_chunk_window +
                                                                   window_slot + 1]
            _set_signal_cuda(remote_arrival, ticket, scatter_stream)


def _enqueue_windowed_chunk_recipe(
    input_intra_node: torch.Tensor,
    ctx: New3rdV2WindowedPanelRSContext,
    output: torch.Tensor,
    chunk_id: int,
    num_runtime_chunks: int,
) -> None:
    slot = _slot_for_chunk(ctx, chunk_id)
    stream = slot.stream
    slot.chunk_id_host = chunk_id

    m_per_rank = output.shape[0]
    row_start, row_end = _chunk_row_range(ctx, chunk_id, m_per_rank)
    rows = row_end - row_start
    if rows <= 0:
        return

    out_chunk = output[row_start:row_end]
    window_slot = chunk_id % ctx.active_chunk_window
    ticket = ctx.ticket_for_chunk(chunk_id)
    local_src = None
    if ctx.local_seed_direct:
        local_segment_start = ctx.local_rank * m_per_rank + row_start
        local_segment_end = local_segment_start + rows
        local_src = input_intra_node[local_segment_start:local_segment_end]

    with torch.cuda.stream(stream):
        if ctx.local_seed_direct:
            _wait_eq_cuda(_chunk_signal_view(ctx, ctx.local_rank, chunk_id), ctx.signal_value, stream)
        for src_local_rank in range(ctx.local_world_size):
            if ctx.local_seed_direct and src_local_rank == ctx.local_rank:
                continue
            _wait_eq_cuda(_arrival_flag_view(ctx, src_local_rank, window_slot), ticket, stream)

        scatter_slot = _window_slot_view(ctx, window_slot)
        use_tail_budget = chunk_id >= max(0, num_runtime_chunks - ctx.tail_chunk_window)
        sms = _num_sms_or_default(ctx.tail_sms if use_tail_budget else ctx.steady_sms)
        _reduce_window_slot_from_scatter_direct(
            scatter_slot,
            local_src,
            out_chunk,
            slot_rows=ctx.chunk_rows,
            local_rank=ctx.local_rank,
            num_splits=ctx.local_world_size,
            rows_to_reduce=rows,
            num_sms=sms,
        )
        _set_signal_cuda(_free_flag_view_local(ctx, window_slot), ticket, stream)
        slot.done_event.record(stream)


def new_3rd_v2_windowed_panel_rs_op(
    input: torch.Tensor,
    ctx: New3rdV2WindowedPanelRSContext,
    output: Optional[torch.Tensor] = None,
    *,
    prepare_round: bool = False,
) -> torch.Tensor:
    if ctx.nnodes != 1:
        raise NotImplementedError("windowed_panel_v2 currently supports single-node only")
    if not has_fullmesh_nvlink():
        raise NotImplementedError("windowed_panel_v2 currently expects full-mesh NVLink")

    M, N = input.shape
    if output is None:
        output = torch.empty((M // ctx.world_size, N), dtype=input.dtype, device=input.device)

    m_per_rank = output.shape[0]
    num_runtime_chunks = triton.cdiv(m_per_rank, ctx.chunk_rows)
    if prepare_round:
        ctx.begin_round(num_runtime_chunks)
    elif ctx.signal_value <= 0:
        raise RuntimeError("new_3rd_v2_windowed_panel_rs_op requires ctx.begin_round(...) before launch")

    _debug_log(
        ctx,
        f"rs enter: M={M}, N={N}, chunk_rows={ctx.chunk_rows}, num_runtime_chunks={num_runtime_chunks}, "
        f"active_chunk_window={ctx.active_chunk_window}, comm_lanes={len(ctx.comm_streams)}, "
        f"stage_slots={ctx.stage_slots}",
    )
    _issue_windowed_chunk_scatter_and_arrival(input, ctx)
    for chunk_id in range(num_runtime_chunks):
        _enqueue_windowed_chunk_recipe(input, ctx, output, chunk_id, num_runtime_chunks)
    ctx.wait_all(torch.cuda.current_stream())
    _debug_log(ctx, "rs exit")
    return output


__all__ = [
    "New3rdV2WindowedPanelRSContext",
    "create_new_3rd_v2_windowed_panel_rs_context",
    "new_3rd_v2_windowed_panel_rs_op",
]
