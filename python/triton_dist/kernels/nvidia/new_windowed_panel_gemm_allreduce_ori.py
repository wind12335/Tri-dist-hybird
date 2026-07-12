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
from typing import List, Optional

import torch
import triton
import triton.language as tl
import triton_dist
import triton_dist.language as dl
import triton_dist.tune

from triton_dist.kernels.nvidia.common_ops import _set_signal_cuda, _wait_eq_cuda
from triton_dist.kernels.nvidia.gemm import get_config_space
from triton_dist.kernels.nvidia.gemm_reduce_scatter import _matmul_launch_metadata, update_triton_config
from triton_dist.kernels.nvidia.new_3rdgemm_reducescatterv3 import swizzle_2d
from triton_dist.utils import (NVSHMEM_SIGNAL_DTYPE, get_device_max_shared_memory_size, has_fullmesh_nvlink,
                               nvshmem_barrier_all_on_stream, nvshmem_create_tensors, nvshmem_free_tensor_sync)


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
def kernel_gemm_ar_producer_windowed_chunk_panel(
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
    rows_this_chunk,
    band_col_start,
    band_cols,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(rows_this_chunk, BLOCK_SIZE_M)
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
    c_mask = (local_offs_m[:, None] < rows_this_chunk) & (local_offs_n[None, :] < band_cols)
    tl.store(c_ptrs, accumulator.to(c_ptr.dtype.element_ty), mask=c_mask)


@triton_dist.jit(do_not_specialize=[])
def kernel_reduce_window_slot_allreduce(
    symm_window_ptr,
    out_ptr,
    slot_row_offset,
    rows_to_reduce,
    band_cols,
    stride_symm_m,
    stride_symm_n,
    stride_outm,
    stride_outn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    world_size = dl.num_ranks()
    pid = tl.program_id(0)
    num_pid = tl.num_programs(0)
    num_pid_m = tl.cdiv(rows_to_reduce, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(band_cols, BLOCK_SIZE_N)
    total_tiles = num_pid_m * num_pid_n

    for tile_id in range(pid, total_tiles, num_pid):
        pid_m = tile_id // num_pid_n
        pid_n = tile_id % num_pid_n
        offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        mask = (offs_m[:, None] < rows_to_reduce) & (offs_n[None, :] < band_cols)

        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for peer in range(world_size):
            peer_ptr = dl.symm_at(symm_window_ptr, peer)
            ptrs = peer_ptr + (slot_row_offset + offs_m)[:, None] * stride_symm_m + offs_n[None, :] * stride_symm_n
            acc += tl.load(ptrs, mask=mask, other=0.0)

        out_ptrs = out_ptr + offs_m[:, None] * stride_outm + offs_n[None, :] * stride_outn
        tl.store(out_ptrs, acc.to(out_ptr.dtype.element_ty), mask=mask)


@dataclasses.dataclass
class AllReduceWindowSlot:
    slot_id: int
    stream: torch.cuda.Stream
    done_event: torch.cuda.Event
    task_id_host: int = -1


@dataclasses.dataclass
class FrontierWindowedPanelGEMMARContext:
    rank: int
    world_size: int
    local_world_size: int
    output_dtype: torch.dtype
    max_M: int
    N: int
    chunk_rows: int
    num_chunks: int
    active_chunk_window: int
    n_bands: int
    max_band_cols: int
    frontier_chunks: int
    stage_slots: int
    num_comm_sms: int
    symm_window_bufs: List[torch.Tensor]
    ready_signal_bufs: List[torch.Tensor]
    free_flag_bufs: List[torch.Tensor]
    reduce_slots: List[AllReduceWindowSlot]
    round_base: int = 0
    initial_free_ticket_per_panel_slot: List[int] = dataclasses.field(default_factory=list)
    prev_round_last_ticket_per_panel_slot: List[int] = dataclasses.field(default_factory=list)

    @property
    def local_rank(self) -> int:
        return self.rank % self.local_world_size

    @property
    def nnodes(self) -> int:
        return self.world_size // self.local_world_size

    @property
    def local_window_buf(self) -> torch.Tensor:
        return self.symm_window_bufs[self.local_rank]

    @property
    def local_ready_signal(self) -> torch.Tensor:
        return self.ready_signal_bufs[self.local_rank]

    @property
    def local_free_flag(self) -> torch.Tensor:
        return self.free_flag_bufs[self.local_rank]

    def begin_round(self, num_runtime_chunks: int) -> None:
        self.round_base += self.num_chunks * self.n_bands + 1
        self.initial_free_ticket_per_panel_slot = self.prev_round_last_ticket_per_panel_slot.copy()
        for band_id in range(self.n_bands):
            for slot_id in range(self.active_chunk_window):
                panel_slot = _panel_slot_id(self, window_slot=slot_id, band_id=band_id)
                if slot_id < num_runtime_chunks:
                    last_chunk = slot_id + self.active_chunk_window * ((num_runtime_chunks - 1 - slot_id) //
                                                                       self.active_chunk_window)
                    self.prev_round_last_ticket_per_panel_slot[panel_slot] = self.ticket_for_panel(last_chunk, band_id)
        for slot in self.reduce_slots:
            slot.task_id_host = -1

    def ticket_for_panel(self, chunk_id: int, band_id: int) -> int:
        return self.round_base + chunk_id * self.n_bands + band_id + 1

    def wait_all(self, current_stream: Optional[torch.cuda.Stream] = None) -> None:
        current_stream = current_stream or torch.cuda.current_stream()
        for slot in self.reduce_slots:
            current_stream.wait_stream(slot.stream)

    def finalize(self) -> None:
        nvshmem_free_tensor_sync(self.symm_window_bufs[self.local_rank])
        nvshmem_free_tensor_sync(self.ready_signal_bufs[self.local_rank])
        nvshmem_free_tensor_sync(self.free_flag_bufs[self.local_rank])


def _num_sms_or_default(num_sms: int) -> int:
    total_sms = torch.cuda.get_device_properties(0).multi_processor_count
    return max(1, min(total_sms, num_sms))


def _num_panel_slots(ctx: FrontierWindowedPanelGEMMARContext) -> int:
    return ctx.active_chunk_window * ctx.n_bands


def _panel_slot_id(ctx: FrontierWindowedPanelGEMMARContext, *, window_slot: int, band_id: int) -> int:
    return band_id * ctx.active_chunk_window + window_slot


def _slot_row_offset(ctx: FrontierWindowedPanelGEMMARContext, window_slot: int, band_id: int) -> int:
    return _panel_slot_id(ctx, window_slot=window_slot, band_id=band_id) * ctx.chunk_rows


def _ready_signal_view(ctx: FrontierWindowedPanelGEMMARContext, peer_local_rank: int, chunk_id: int,
                       band_id: int) -> torch.Tensor:
    signal_idx = band_id * ctx.num_chunks + chunk_id
    return ctx.ready_signal_bufs[peer_local_rank][signal_idx:signal_idx + 1]


def _free_flag_remote_view(ctx: FrontierWindowedPanelGEMMARContext, peer_local_rank: int, consumer_local_rank: int,
                           window_slot: int, band_id: int) -> torch.Tensor:
    panel_slot = _panel_slot_id(ctx, window_slot=window_slot, band_id=band_id)
    idx = consumer_local_rank * _num_panel_slots(ctx) + panel_slot
    return ctx.free_flag_bufs[peer_local_rank][idx:idx + 1]


def _free_flag_local_view(ctx: FrontierWindowedPanelGEMMARContext, consumer_local_rank: int, window_slot: int,
                          band_id: int) -> torch.Tensor:
    panel_slot = _panel_slot_id(ctx, window_slot=window_slot, band_id=band_id)
    idx = consumer_local_rank * _num_panel_slots(ctx) + panel_slot
    return ctx.local_free_flag[idx:idx + 1]


def _expected_free_ticket(ctx: FrontierWindowedPanelGEMMARContext, chunk_id: int, window_slot: int, band_id: int) -> int:
    prev_chunk = chunk_id - ctx.active_chunk_window
    if prev_chunk >= 0:
        return ctx.ticket_for_panel(prev_chunk, band_id)
    panel_slot = _panel_slot_id(ctx, window_slot=window_slot, band_id=band_id)
    return ctx.initial_free_ticket_per_panel_slot[panel_slot]


def _slot_for_panel(ctx: FrontierWindowedPanelGEMMARContext, chunk_id: int, band_id: int) -> AllReduceWindowSlot:
    return ctx.reduce_slots[(chunk_id * ctx.n_bands + band_id) % ctx.stage_slots]


def _chunk_row_range(ctx: FrontierWindowedPanelGEMMARContext, chunk_id: int, M: int) -> tuple[int, int]:
    row_start = chunk_id * ctx.chunk_rows
    row_end = min(row_start + ctx.chunk_rows, M)
    return row_start, row_end


def _band_col_range(ctx: FrontierWindowedPanelGEMMARContext, band_id: int, N: int) -> tuple[int, int]:
    col_start = band_id * ctx.max_band_cols
    col_end = min(col_start + ctx.max_band_cols, N)
    return col_start, col_end


def _build_chunk_schedule(num_runtime_chunks: int, frontier_chunks: int) -> List[int]:
    frontier = list(range(min(num_runtime_chunks, max(0, frontier_chunks))))
    tail = list(range(len(frontier), num_runtime_chunks))
    return frontier + tail


def create_frontier_windowed_panel_gemm_ar_context(
    max_M: int,
    N: int,
    rank: int,
    world_size: int,
    local_world_size: int,
    output_dtype: torch.dtype,
    *,
    chunk_rows: int = 0,
    target_chunks: int = 4,
    min_chunk_rows: int = 512,
    active_chunk_window: int = 2,
    n_bands: int = 1,
    frontier_chunks: int = 1,
    stage_slots: int = 4,
    num_comm_sms: int = 16,
) -> FrontierWindowedPanelGEMMARContext:
    if world_size != local_world_size:
        raise NotImplementedError("frontier_windowed_panel_gemm_allreduce currently supports single-node only")

    effective_chunk_rows = chunk_rows if chunk_rows > 0 else _auto_chunk_rows(
        max_M, target_chunks=target_chunks, min_chunk_rows=min_chunk_rows)
    num_chunks = triton.cdiv(max_M, effective_chunk_rows)
    active_chunk_window = max(1, min(active_chunk_window, num_chunks))
    n_bands = max(1, min(n_bands, N))
    max_band_cols = triton.cdiv(N, n_bands)
    stage_slots = max(1, min(stage_slots, num_chunks * n_bands))

    window_rows = active_chunk_window * n_bands * effective_chunk_rows
    symm_window_bufs = nvshmem_create_tensors((window_rows, max_band_cols), output_dtype, rank, local_world_size)
    ready_signal_bufs = nvshmem_create_tensors((n_bands * num_chunks,), NVSHMEM_SIGNAL_DTYPE, rank, local_world_size)
    free_flag_bufs = nvshmem_create_tensors((local_world_size * active_chunk_window * n_bands,), NVSHMEM_SIGNAL_DTYPE,
                                            rank, local_world_size)

    ready_signal_bufs[rank % local_world_size].zero_()
    free_flag_bufs[rank % local_world_size].zero_()
    nvshmem_barrier_all_on_stream(torch.cuda.current_stream())

    reduce_slots = [
        AllReduceWindowSlot(slot_id=i, stream=torch.cuda.Stream(priority=-1), done_event=torch.cuda.Event())
        for i in range(stage_slots)
    ]
    return FrontierWindowedPanelGEMMARContext(
        rank=rank,
        world_size=world_size,
        local_world_size=local_world_size,
        output_dtype=output_dtype,
        max_M=max_M,
        N=N,
        chunk_rows=effective_chunk_rows,
        num_chunks=num_chunks,
        active_chunk_window=active_chunk_window,
        n_bands=n_bands,
        max_band_cols=max_band_cols,
        frontier_chunks=max(0, frontier_chunks),
        stage_slots=stage_slots,
        num_comm_sms=num_comm_sms,
        symm_window_bufs=symm_window_bufs,
        ready_signal_bufs=ready_signal_bufs,
        free_flag_bufs=free_flag_bufs,
        reduce_slots=reduce_slots,
        initial_free_ticket_per_panel_slot=[0 for _ in range(active_chunk_window * n_bands)],
        prev_round_last_ticket_per_panel_slot=[0 for _ in range(active_chunk_window * n_bands)],
    )


def _launch_windowed_chunk_panel_producer(
    A: torch.Tensor,
    B: torch.Tensor,
    ctx: FrontierWindowedPanelGEMMARContext,
    gemm_config: triton.Config,
    chunk_id: int,
    band_id: int,
) -> None:
    M, K = A.shape
    _, N = B.shape
    row_start, row_end = _chunk_row_range(ctx, chunk_id, M)
    col_start, col_end = _band_col_range(ctx, band_id, N)
    rows = row_end - row_start
    band_cols = col_end - col_start
    if rows <= 0 or band_cols <= 0:
        return

    window_slot = chunk_id % ctx.active_chunk_window
    ticket = ctx.ticket_for_panel(chunk_id, band_id)
    free_ticket = _expected_free_ticket(ctx, chunk_id, window_slot, band_id)
    producer_stream = torch.cuda.current_stream()
    for consumer_local_rank in range(ctx.local_world_size):
        _wait_eq_cuda(_free_flag_local_view(ctx, consumer_local_rank, window_slot, band_id), free_ticket, producer_stream)

    slot_row_offset = _slot_row_offset(ctx, window_slot, band_id)
    local_window = ctx.local_window_buf[slot_row_offset:slot_row_offset + ctx.chunk_rows, :band_cols]
    grid = (
        triton.cdiv(rows, gemm_config.kwargs["BLOCK_SIZE_M"]) *
        triton.cdiv(band_cols, gemm_config.kwargs["BLOCK_SIZE_N"]),
    )
    kernel_gemm_ar_producer_windowed_chunk_panel[grid](
        A,
        B,
        local_window,
        M,
        N,
        K,
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(1),
        local_window.stride(0),
        local_window.stride(1),
        row_start,
        rows,
        col_start,
        band_cols,
        **gemm_config.all_kwargs(),
    )
    _set_signal_cuda(_ready_signal_view(ctx, ctx.local_rank, chunk_id, band_id), ticket, producer_stream)


def _enqueue_windowed_panel_allreduce(
    ctx: FrontierWindowedPanelGEMMARContext,
    output: torch.Tensor,
    chunk_id: int,
    band_id: int,
) -> None:
    M, N = output.shape
    row_start, row_end = _chunk_row_range(ctx, chunk_id, M)
    col_start, col_end = _band_col_range(ctx, band_id, N)
    rows = row_end - row_start
    band_cols = col_end - col_start
    if rows <= 0 or band_cols <= 0:
        return

    window_slot = chunk_id % ctx.active_chunk_window
    ticket = ctx.ticket_for_panel(chunk_id, band_id)
    slot = _slot_for_panel(ctx, chunk_id, band_id)
    stream = slot.stream
    out_chunk = output[row_start:row_end, col_start:col_end]
    slot_row_offset = _slot_row_offset(ctx, window_slot, band_id)

    with torch.cuda.stream(stream):
        for peer in range(ctx.local_world_size):
            _wait_eq_cuda(_ready_signal_view(ctx, peer, chunk_id, band_id), ticket, stream)

        grid = (_num_sms_or_default(ctx.num_comm_sms),)
        kernel_reduce_window_slot_allreduce[grid](
            ctx.local_window_buf,
            out_chunk,
            slot_row_offset,
            rows,
            band_cols,
            ctx.local_window_buf.stride(0),
            ctx.local_window_buf.stride(1),
            out_chunk.stride(0),
            out_chunk.stride(1),
            BLOCK_SIZE_M=32,
            BLOCK_SIZE_N=128,
            num_warps=8,
        )

        for peer in range(ctx.local_world_size):
            _set_signal_cuda(_free_flag_remote_view(ctx, peer, ctx.local_rank, window_slot, band_id), ticket, stream)
        slot.done_event.record(stream)


def frontier_windowed_panel_gemm_allreduce_key_fn(
    A: torch.Tensor,
    B: torch.Tensor,
    ctx: FrontierWindowedPanelGEMMARContext,
    *args,
    **kwargs,
):
    return (
        triton_dist.tune.to_hashable(A),
        triton_dist.tune.to_hashable(B),
        ctx.world_size,
        ctx.chunk_rows,
        ctx.active_chunk_window,
        ctx.n_bands,
        ctx.frontier_chunks,
        ctx.stage_slots,
    )


def frontier_windowed_panel_gemm_allreduce_prune_fn(config, A, B, *args, **kwargs):
    itemsize = A.itemsize
    gemm_config = config["gemm_config"].all_kwargs()
    num_stages = gemm_config["num_stages"]
    block_size_m = gemm_config["BLOCK_SIZE_M"]
    block_size_n = gemm_config["BLOCK_SIZE_N"]
    block_size_k = gemm_config["BLOCK_SIZE_K"]
    shared_memory = (itemsize * block_size_m * block_size_k + itemsize * block_size_n * block_size_k) * num_stages
    return shared_memory < get_device_max_shared_memory_size(torch.cuda.current_device())


def get_frontier_windowed_panel_gemm_allreduce_config_space():
    return [{"gemm_config": config} for config in get_config_space(False)]


def frontier_windowed_panel_gemm_allreduce_op(
    A: torch.Tensor,
    B: torch.Tensor,
    ctx: FrontierWindowedPanelGEMMARContext,
    gemm_config: triton.Config,
) -> torch.Tensor:
    if ctx.nnodes != 1:
        raise NotImplementedError("frontier_windowed_panel_gemm_allreduce currently supports single-node only")
    if not has_fullmesh_nvlink():
        raise NotImplementedError("frontier_windowed_panel_gemm_allreduce currently expects full-mesh NVLink")

    M, local_K = A.shape
    _, N = B.shape
    assert N == ctx.N, f"B should be of shape [{local_K}, {ctx.N}]"

    num_runtime_chunks = triton.cdiv(M, ctx.chunk_rows)
    tuned_config = update_triton_config(M, N, local_K, A.dtype, ctx.world_size, ctx.local_world_size, gemm_config)
    if ctx.chunk_rows < tuned_config.kwargs["BLOCK_SIZE_M"]:
        raise ValueError(
            "frontier_windowed_panel_gemm_allreduce requires chunk_rows >= BLOCK_SIZE_M, "
            f"but got chunk_rows={ctx.chunk_rows}, BLOCK_SIZE_M={tuned_config.kwargs['BLOCK_SIZE_M']}"
        )

    output = torch.empty((M, N), dtype=ctx.output_dtype, device=A.device)
    ctx.begin_round(num_runtime_chunks)
    chunk_schedule = _build_chunk_schedule(num_runtime_chunks, ctx.frontier_chunks)

    # Producer-first launch avoids immediately parking consumer streams on
    # not-yet-emitted ready tickets, which was the main deadlock risk.
    for chunk_id in chunk_schedule:
        for band_id in range(ctx.n_bands):
            _launch_windowed_chunk_panel_producer(A, B, ctx, tuned_config, chunk_id, band_id)
            _enqueue_windowed_panel_allreduce(ctx, output, chunk_id, band_id)

    ctx.wait_all(torch.cuda.current_stream())
    return output


@triton_dist.tune.autotune(
    config_space=get_frontier_windowed_panel_gemm_allreduce_config_space(),
    key_fn=frontier_windowed_panel_gemm_allreduce_key_fn,
    prune_fn=frontier_windowed_panel_gemm_allreduce_prune_fn,
)
def frontier_windowed_panel_gemm_allreduce(
    A: torch.Tensor,
    B: torch.Tensor,
    ctx: FrontierWindowedPanelGEMMARContext,
    gemm_config: triton.Config,
):
    return frontier_windowed_panel_gemm_allreduce_op(A, B, ctx, gemm_config)


def frontier_windowed_panel_allreduce(
    input_tensor: torch.Tensor,
    ctx: FrontierWindowedPanelGEMMARContext,
) -> torch.Tensor:
    M, N = input_tensor.shape
    num_runtime_chunks = triton.cdiv(M, ctx.chunk_rows)
    output = torch.empty_like(input_tensor)
    ctx.begin_round(num_runtime_chunks)
    chunk_schedule = _build_chunk_schedule(num_runtime_chunks, ctx.frontier_chunks)

    for chunk_id in chunk_schedule:
        for band_id in range(ctx.n_bands):
            row_start, row_end = _chunk_row_range(ctx, chunk_id, M)
            col_start, col_end = _band_col_range(ctx, band_id, N)
            rows = row_end - row_start
            band_cols = col_end - col_start
            if rows <= 0 or band_cols <= 0:
                continue
            window_slot = chunk_id % ctx.active_chunk_window
            ticket = ctx.ticket_for_panel(chunk_id, band_id)
            free_ticket = _expected_free_ticket(ctx, chunk_id, window_slot, band_id)
            for consumer_local_rank in range(ctx.local_world_size):
                _wait_eq_cuda(
                    _free_flag_local_view(ctx, consumer_local_rank, window_slot, band_id),
                    free_ticket,
                    torch.cuda.current_stream(),
                )
            slot_row_offset = _slot_row_offset(ctx, window_slot, band_id)
            local_window = ctx.local_window_buf[slot_row_offset:slot_row_offset + ctx.chunk_rows, :band_cols]
            local_window[:rows, :band_cols].copy_(input_tensor[row_start:row_end, col_start:col_end])
            _set_signal_cuda(_ready_signal_view(ctx, ctx.local_rank, chunk_id, band_id), ticket, torch.cuda.current_stream())
            _enqueue_windowed_panel_allreduce(ctx, output, chunk_id, band_id)

    ctx.wait_all(torch.cuda.current_stream())
    return output


__all__ = [
    "FrontierWindowedPanelGEMMARContext",
    "create_frontier_windowed_panel_gemm_ar_context",
    "frontier_windowed_panel_allreduce",
    "frontier_windowed_panel_gemm_allreduce",
    "frontier_windowed_panel_gemm_allreduce_op",
]
