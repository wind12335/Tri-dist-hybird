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
from triton_dist.kernels.nvidia.reduce_scatter import (ReduceScatter2DContext, add_continuous, copy_continous,
                                                       create_reduce_scater_2d_ctx, reduce_scatter_2d_op)
from triton_dist.utils import (CUDA_CHECK, NVSHMEM_SIGNAL_DTYPE, has_fullmesh_nvlink, nvshmem_barrier_all_on_stream,
                               nvshmem_create_tensors, nvshmem_free_tensor_sync)


_SIGNAL_WRAP = 2**30


def _debug_enabled() -> bool:
    return os.environ.get("TRITON_DIST_NEW_3RD_DEBUG", "0") == "1"


def _debug_log(ctx: "New3rdReduceScatterContext", msg: str) -> None:
    if _debug_enabled():
        print(f"[new_3rd][rank{ctx.rank}] {msg}", flush=True)


def _round_up(x: int, align: int) -> int:
    return ((x + align - 1) // align) * align


def _auto_chunk_rows(
    max_m_per_rank: int,
    *,
    target_chunks_per_rank: int,
    min_chunk_rows: int,
    align: int = 256,
) -> int:
    if max_m_per_rank <= 0:
        raise ValueError(f"max_m_per_rank must be > 0, got {max_m_per_rank}")
    if target_chunks_per_rank <= 0:
        raise ValueError(f"target_chunks_per_rank must be > 0, got {target_chunks_per_rank}")
    if min_chunk_rows <= 0:
        raise ValueError(f"min_chunk_rows must be > 0, got {min_chunk_rows}")
    rows = triton.cdiv(max_m_per_rank, target_chunks_per_rank)
    rows = max(rows, min_chunk_rows)
    rows = min(rows, max_m_per_rank)
    rows = _round_up(rows, align)
    return min(rows, max_m_per_rank)


@triton.jit
def kernel_copy_cast_contiguous(
    src_ptr,
    dst_ptr,
    numel,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid = tl.num_programs(axis=0)
    nblocks = tl.cdiv(numel, BLOCK_SIZE)
    for block in range(pid, nblocks, num_pid):
        offs = block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < numel
        data = tl.load(src_ptr + offs, mask=mask, other=0.0)
        tl.store(dst_ptr + offs, data.to(dst_ptr.dtype.element_ty), mask=mask)


@triton.jit
def kernel_add_cast_contiguous(
    src_ptr,
    accum_ptr,
    numel,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid = tl.num_programs(axis=0)
    nblocks = tl.cdiv(numel, BLOCK_SIZE)
    for block in range(pid, nblocks, num_pid):
        offs = block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < numel
        src = tl.load(src_ptr + offs, mask=mask, other=0.0)
        accum = tl.load(accum_ptr + offs, mask=mask, other=0.0)
        tl.store(accum_ptr + offs, accum + src.to(accum_ptr.dtype.element_ty), mask=mask)


@triton.jit
def kernel_final_add_store_contiguous(
    src_ptr,
    accum_ptr,
    out_ptr,
    numel,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid = tl.num_programs(axis=0)
    nblocks = tl.cdiv(numel, BLOCK_SIZE)
    for block in range(pid, nblocks, num_pid):
        offs = block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < numel
        src = tl.load(src_ptr + offs, mask=mask, other=0.0)
        accum = tl.load(accum_ptr + offs, mask=mask, other=0.0)
        out = accum + src.to(accum_ptr.dtype.element_ty)
        tl.store(out_ptr + offs, out.to(out_ptr.dtype.element_ty), mask=mask)


@dataclasses.dataclass
class ElasticReduceSlot:
    slot_id: int
    stream: torch.cuda.Stream
    done_event: torch.cuda.Event
    accum_buf: Optional[torch.Tensor]
    chunk_id_host: int = -1


@dataclasses.dataclass
class New3rdReduceScatterContext:
    """Signal-driven third-stream reduce-scatter.

    Main ideas:
    - wait for readiness with stream wait-value, not a resident helper kernel
    - use a small pool of reduction streams ("slots") for per-chunk recipes
    - accumulate arrivals into scratch buffers to avoid a tail full-chunk reread
    - use small-SM kernels during the steady phase and a larger tail budget on
      the last chunk(s)
    """

    base_ctx: ReduceScatter2DContext
    chunk_rows: int
    num_chunks: int
    chunk_signal: torch.Tensor
    arrival_flag_bufs: List[torch.Tensor]
    comm_stream: torch.cuda.Stream
    slots: List[ElasticReduceSlot]
    accum_dtype: torch.dtype
    use_scratch: bool
    steady_sms: int
    tail_sms: int
    stage_slots: int
    tail_chunk_window: int
    local_seed_direct: bool
    signal_value: int = 0

    @property
    def rank(self) -> int:
        return self.base_ctx.rank

    @property
    def world_size(self) -> int:
        return self.base_ctx.world_size

    @property
    def local_world_size(self) -> int:
        return self.base_ctx.local_world_size

    @property
    def local_rank(self) -> int:
        return self.base_ctx.local_rank

    @property
    def node_id(self) -> int:
        return self.base_ctx.node_id

    @property
    def nnodes(self) -> int:
        return self.base_ctx.nnodes

    @property
    def dtype(self) -> torch.dtype:
        return self.base_ctx.dtype

    @property
    def arrival_flag_buf(self) -> torch.Tensor:
        return self.arrival_flag_bufs[self.local_rank]

    def begin_round(self) -> int:
        self.signal_value += 1
        if self.signal_value >= _SIGNAL_WRAP:
            self.signal_value = 1
            self.chunk_signal.zero_()
            self.arrival_flag_buf.zero_()
        for slot in self.slots:
            slot.chunk_id_host = -1
        return self.signal_value

    def wait_all(self, current_stream: Optional[torch.cuda.Stream] = None) -> None:
        current_stream = current_stream or torch.cuda.current_stream()
        current_stream.wait_stream(self.comm_stream)
        for slot in self.slots:
            current_stream.wait_stream(slot.stream)

    def finalize(self) -> None:
        nvshmem_free_tensor_sync(self.arrival_flag_buf)
        self.base_ctx.finalize()


def _num_sms_or_default(num_sms: int) -> int:
    total_sms = torch.cuda.get_device_properties(0).multi_processor_count
    return max(1, min(total_sms, num_sms))


def _num_ctas_for_tensor(tensor: torch.Tensor, requested_sms: int) -> int:
    block_size = max(1, 32 * 8 * 16 // tensor.itemsize)
    total_blocks = triton.cdiv(tensor.numel(), block_size)
    return max(1, min(total_blocks, _num_sms_or_default(requested_sms)))


def _copy_or_cast(src: torch.Tensor, dst: torch.Tensor, num_sms: int) -> None:
    src = src.contiguous().reshape(-1)
    if not dst.is_contiguous():
        raise ValueError("destination chunk must be contiguous")
    dst = dst.reshape(-1)
    if src.dtype == dst.dtype:
        copy_continous(src, dst, num_ctas=_num_ctas_for_tensor(src, num_sms))
        return
    ctas = _num_ctas_for_tensor(dst, num_sms)
    kernel_copy_cast_contiguous[(ctas, )](src, dst, dst.numel(), BLOCK_SIZE=4096, num_warps=8)


def _add_into_accum(src: torch.Tensor, accum: torch.Tensor, num_sms: int) -> None:
    src = src.contiguous().reshape(-1)
    if not accum.is_contiguous():
        raise ValueError("accum chunk must be contiguous")
    accum = accum.reshape(-1)
    if src.dtype == accum.dtype:
        add_continuous(src, accum, accum, num_ctas=_num_ctas_for_tensor(accum, num_sms))
        return
    ctas = _num_ctas_for_tensor(accum, num_sms)
    kernel_add_cast_contiguous[(ctas, )](src, accum, accum.numel(), BLOCK_SIZE=4096, num_warps=8)


def _final_add_to_output(src: torch.Tensor, accum: torch.Tensor, output: torch.Tensor, num_sms: int) -> None:
    src = src.contiguous().reshape(-1)
    if not accum.is_contiguous():
        raise ValueError("accum chunk must be contiguous")
    if not output.is_contiguous():
        raise ValueError("output chunk must be contiguous")
    accum = accum.reshape(-1)
    output = output.reshape(-1)
    if src.dtype == accum.dtype and accum.dtype == output.dtype:
        add_continuous(src, accum, output, num_ctas=_num_ctas_for_tensor(output, num_sms))
        return
    ctas = _num_ctas_for_tensor(output, num_sms)
    kernel_final_add_store_contiguous[(ctas, )](src, accum, output, output.numel(), BLOCK_SIZE=4096, num_warps=8)


def _copy_to_output(src: torch.Tensor, output: torch.Tensor, num_sms: int) -> None:
    _copy_or_cast(src, output, num_sms)


def _add_to_output(src: torch.Tensor, output: torch.Tensor, num_sms: int) -> None:
    src = src.contiguous().reshape(-1)
    if not output.is_contiguous():
        raise ValueError("output chunk must be contiguous")
    output = output.reshape(-1)
    add_continuous(src, output, output, num_ctas=_num_ctas_for_tensor(output, num_sms))


def _chunk_row_range(ctx: New3rdReduceScatterContext, chunk_id: int, m_per_rank: int) -> tuple[int, int]:
    row_start = chunk_id * ctx.chunk_rows
    row_end = min(row_start + ctx.chunk_rows, m_per_rank)
    return row_start, row_end


def _chunk_signal_view(ctx: New3rdReduceScatterContext, dest_local_rank: int, chunk_id: int) -> torch.Tensor:
    signal_idx = dest_local_rank * ctx.num_chunks + chunk_id
    return ctx.chunk_signal[signal_idx:signal_idx + 1]


def _arrival_flag_view(ctx: New3rdReduceScatterContext, src_local_rank: int, chunk_id: int) -> torch.Tensor:
    arrival_idx = src_local_rank * ctx.num_chunks + chunk_id
    return ctx.arrival_flag_buf[arrival_idx:arrival_idx + 1]


def _slot_for_chunk(ctx: New3rdReduceScatterContext, chunk_id: int) -> ElasticReduceSlot:
    return ctx.slots[chunk_id % ctx.stage_slots]


def _seed_sms(ctx: New3rdReduceScatterContext) -> int:
    return max(1, min(2, ctx.steady_sms))


def _steady_sms(ctx: New3rdReduceScatterContext) -> int:
    return _num_sms_or_default(ctx.steady_sms)


def _tail_sms(ctx: New3rdReduceScatterContext) -> int:
    return _num_sms_or_default(ctx.tail_sms)


def create_new_3rd_reducescatter_2d_ctx(
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
    helper_num_sms: int = 4,
    steady_sms: Optional[int] = None,
    tail_sms: Optional[int] = None,
    stage_slots: int = 2,
    accum_dtype: Optional[torch.dtype] = None,
    use_scratch: bool = True,
    tail_chunk_window: int = 1,
    local_seed_direct: bool = True,
    comm_stream: Optional[torch.cuda.Stream] = None,
) -> New3rdReduceScatterContext:
    base_ctx = create_reduce_scater_2d_ctx(max_M, N, rank, world_size, local_world_size, dtype)
    max_m_per_rank = max_M // world_size
    effective_chunk_rows = chunk_rows if chunk_rows > 0 else _auto_chunk_rows(
        max_m_per_rank,
        target_chunks_per_rank=target_chunks_per_rank,
        min_chunk_rows=min_chunk_rows,
    )
    num_chunks = triton.cdiv(max_m_per_rank, effective_chunk_rows)
    stage_slots = max(1, min(stage_slots, num_chunks))

    if accum_dtype is None:
        accum_dtype = torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype

    total_sms = torch.cuda.get_device_properties(0).multi_processor_count
    steady_sms = helper_num_sms if steady_sms is None else steady_sms
    tail_sms = max(steady_sms, helper_num_sms * 4) if tail_sms is None else tail_sms
    steady_sms = max(1, min(total_sms, steady_sms))
    tail_sms = max(steady_sms, min(total_sms, tail_sms))

    chunk_signal = torch.zeros((world_size * num_chunks,), dtype=torch.int32, device="cuda")

    nvshmem_barrier_all_on_stream(torch.cuda.current_stream())
    arrival_flag_bufs = nvshmem_create_tensors((local_world_size * num_chunks,), NVSHMEM_SIGNAL_DTYPE, rank,
                                               local_world_size)
    arrival_flag_bufs[rank % local_world_size].zero_()

    if comm_stream is None:
        comm_stream = torch.cuda.Stream(priority=-1)
    slots: List[ElasticReduceSlot] = []
    for slot_id in range(stage_slots):
        slot_stream = torch.cuda.Stream(priority=-1)
        done_event = torch.cuda.Event()
        accum_buf = None
        if use_scratch:
            accum_buf = torch.empty((effective_chunk_rows, N), dtype=accum_dtype, device="cuda")
        slots.append(ElasticReduceSlot(slot_id=slot_id, stream=slot_stream, done_event=done_event, accum_buf=accum_buf))

    nvshmem_barrier_all_on_stream(torch.cuda.current_stream())
    return New3rdReduceScatterContext(
        base_ctx=base_ctx,
        chunk_rows=effective_chunk_rows,
        num_chunks=num_chunks,
        chunk_signal=chunk_signal,
        arrival_flag_bufs=arrival_flag_bufs,
        comm_stream=comm_stream,
        slots=slots,
        accum_dtype=accum_dtype,
        use_scratch=use_scratch,
        steady_sms=steady_sms,
        tail_sms=tail_sms,
        stage_slots=stage_slots,
        tail_chunk_window=max(1, tail_chunk_window),
        local_seed_direct=local_seed_direct,
    )


def _enqueue_chunk_recipe(
    input_intra_node: torch.Tensor,
    local_scatter: torch.Tensor,
    ctx: New3rdReduceScatterContext,
    output: torch.Tensor,
    chunk_id: int,
    num_runtime_chunks: int,
    gemm_done_event: Optional[torch.cuda.Event] = None,
) -> None:
    del gemm_done_event  # current implementation uses a deterministic phase budget
    slot = _slot_for_chunk(ctx, chunk_id)
    stream = slot.stream
    slot.chunk_id_host = chunk_id

    m_per_rank = output.shape[0]
    row_start, row_end = _chunk_row_range(ctx, chunk_id, m_per_rank)
    rows = row_end - row_start
    if rows <= 0:
        return

    local_segment_start = ctx.local_rank * m_per_rank + row_start
    local_segment_end = local_segment_start + rows
    local_src = input_intra_node[local_segment_start:local_segment_end]
    out_chunk = output[row_start:row_end]
    accum_chunk = None if slot.accum_buf is None else slot.accum_buf[:rows]

    remote_sources = [src_local_rank for src_local_rank in range(ctx.local_world_size)
                      if not (ctx.local_seed_direct and src_local_rank == ctx.local_rank)]

    with torch.cuda.stream(stream):
        if ctx.local_seed_direct:
            _wait_eq_cuda(_chunk_signal_view(ctx, ctx.local_rank, chunk_id), ctx.signal_value, stream)
            if ctx.use_scratch:
                _copy_or_cast(local_src, accum_chunk, _seed_sms(ctx))
            else:
                _copy_to_output(local_src, out_chunk, _seed_sms(ctx))

        if len(remote_sources) == 0:
            if ctx.use_scratch:
                _copy_or_cast(accum_chunk, out_chunk, _tail_sms(ctx))
            slot.done_event.record(stream)
            return

        for ridx, src_local_rank in enumerate(remote_sources):
            _wait_eq_cuda(_arrival_flag_view(ctx, src_local_rank, chunk_id), ctx.signal_value, stream)
            src_chunk = local_scatter[src_local_rank * m_per_rank + row_start:src_local_rank * m_per_rank + row_end]
            is_last = ridx == len(remote_sources) - 1
            use_tail_budget = is_last and chunk_id >= max(0, num_runtime_chunks - ctx.tail_chunk_window)
            sms = _tail_sms(ctx) if use_tail_budget else _steady_sms(ctx)

            if ctx.use_scratch:
                if is_last:
                    _final_add_to_output(src_chunk, accum_chunk, out_chunk, sms)
                else:
                    _add_into_accum(src_chunk, accum_chunk, sms)
            else:
                _add_to_output(src_chunk, out_chunk, sms)

        slot.done_event.record(stream)


def _issue_chunk_scatter_and_arrival(
    input_intra_node: torch.Tensor,
    ctx: New3rdReduceScatterContext,
) -> torch.Tensor:
    local_rank = ctx.local_rank
    local_world_size = ctx.local_world_size
    scatter_stream = ctx.comm_stream

    scatter_bufs_intra_node, _ = ctx.base_ctx.get_scatter_bufs_and_signal_for_each_node(input_intra_node, ctx.node_id)
    M, N = input_intra_node.shape
    m_per_rank = M // local_world_size
    nbytes_per_row = N * input_intra_node.dtype.itemsize
    local_buf_base_ptr = input_intra_node.data_ptr()
    remote_offset_base = local_rank * m_per_rank * nbytes_per_row
    num_runtime_chunks = triton.cdiv(m_per_rank, ctx.chunk_rows)

    _debug_log(
        ctx,
        f"scatter start: local_world_size={local_world_size}, M={M}, N={N}, chunk_rows={ctx.chunk_rows}, "
        f"num_runtime_chunks={num_runtime_chunks}, stage_slots={ctx.stage_slots}, steady_sms={ctx.steady_sms}, "
        f"tail_sms={ctx.tail_sms}, use_scratch={ctx.use_scratch}",
    )

    for chunk_id in range(num_runtime_chunks):
        row_start, row_end = _chunk_row_range(ctx, chunk_id, m_per_rank)
        chunk_nbytes = (row_end - row_start) * nbytes_per_row

        for step in range(local_world_size):
            dest_local_rank = (local_rank + step + 1) % local_world_size
            if ctx.local_seed_direct and dest_local_rank == local_rank:
                continue

            _debug_log(ctx, f"scatter chunk {chunk_id} step {step}: waiting chunk-ready for dest_local_rank={dest_local_rank}")
            _wait_eq_cuda(_chunk_signal_view(ctx, dest_local_rank, chunk_id), ctx.signal_value, scatter_stream)

            remote_buf_ptr = scatter_bufs_intra_node[dest_local_rank].data_ptr() + remote_offset_base + row_start * nbytes_per_row
            local_buf_ptr = local_buf_base_ptr + (dest_local_rank * m_per_rank + row_start) * nbytes_per_row
            (err,) = cudart.cudaMemcpyAsync(
                remote_buf_ptr,
                local_buf_ptr,
                chunk_nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyDefault,
                scatter_stream.cuda_stream,
            )
            CUDA_CHECK(err)

            arrival_idx = local_rank * ctx.num_chunks + chunk_id
            remote_ready = ctx.arrival_flag_bufs[dest_local_rank][arrival_idx:arrival_idx + 1]
            _set_signal_cuda(remote_ready, ctx.signal_value, scatter_stream)
            _debug_log(
                ctx,
                f"scatter chunk {chunk_id} step {step}: memcpy+arrival issued to dest_local_rank={dest_local_rank}",
            )

    return scatter_bufs_intra_node[local_rank]


def new_3rd_reduce_scatter_2d_op(
    input: torch.Tensor,
    ctx: New3rdReduceScatterContext,
    output: Optional[torch.Tensor] = None,
    *,
    gemm_done_event: Optional[torch.cuda.Event] = None,
) -> torch.Tensor:
    if ctx.nnodes != 1 or not has_fullmesh_nvlink():
        return reduce_scatter_2d_op(input, ctx.base_ctx, output)

    M, N = input.shape
    if output is None:
        output = torch.empty((M // ctx.world_size, N), dtype=input.dtype, device=input.device)

    current_stream = torch.cuda.current_stream()
    _debug_log(ctx, f"new_3rd_reduce_scatter_2d_op enter: input_shape={tuple(input.shape)}, signal_value={ctx.signal_value}")
    local_scatter = _issue_chunk_scatter_and_arrival(input, ctx)

    m_per_rank = output.shape[0]
    num_runtime_chunks = triton.cdiv(m_per_rank, ctx.chunk_rows)
    for chunk_id in range(num_runtime_chunks):
        _enqueue_chunk_recipe(input, local_scatter, ctx, output, chunk_id, num_runtime_chunks, gemm_done_event)

    _debug_log(ctx, "new_3rd_reduce_scatter_2d_op waiting slot streams")
    ctx.wait_all(current_stream)
    _debug_log(ctx, "new_3rd_reduce_scatter_2d_op exit")
    return output
