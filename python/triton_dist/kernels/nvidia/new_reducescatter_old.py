from __future__ import annotations

import dataclasses
from typing import List, Optional

import torch
import triton
import triton.language as tl
from cuda import cudart

from triton_dist.kernels.nvidia.common_ops import _set_signal_cuda, _wait_eq_cuda
from triton_dist.kernels.nvidia.reduce_scatter import (ReduceScatter2DContext, create_reduce_scater_2d_ctx,
                                                       reduce_scatter_2d_op)
from triton_dist.utils import CUDA_CHECK, NVSHMEM_SIGNAL_DTYPE, has_fullmesh_nvlink, nvshmem_barrier_all_on_stream, \
    nvshmem_create_tensors, nvshmem_free_tensor_sync


def _round_up(x: int, align: int) -> int:
    return ((x + align - 1) // align) * align


def _auto_chunk_rows(max_m_per_rank: int, target_chunks_per_rank: int, min_chunk_rows: int, align: int = 256) -> int:
    rows = max(triton.cdiv(max_m_per_rank, target_chunks_per_rank), min_chunk_rows)
    rows = min(rows, max_m_per_rank)
    return min(_round_up(rows, align), max_m_per_rank)


@triton.jit(do_not_specialize=["chunk_row_start", "chunk_rows"])
def kernel_reduce_chunk_from_scatter(
    in_ptr,
    out_ptr,
    M_per_rank,
    N,
    chunk_row_start,
    chunk_rows,
    stride_inm,
    stride_inn,
    stride_outm,
    stride_outn,
    NUM_SPLITS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    rows = chunk_row_start + offs_m
    mask = (offs_m[:, None] < chunk_rows) & (offs_n[None, :] < N)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for split in range(NUM_SPLITS):
        src_rows = split * M_per_rank + rows
        ptrs = in_ptr + src_rows[:, None] * stride_inm + offs_n[None, :] * stride_inn
        acc += tl.load(ptrs, mask=mask, other=0.0)

    out_ptrs = out_ptr + offs_m[:, None] * stride_outm + offs_n[None, :] * stride_outn
    tl.store(out_ptrs, acc.to(out_ptr.dtype.element_ty), mask=mask)


def reduce_chunk_from_scatter(
    input_scatter: torch.Tensor,
    output_chunk: torch.Tensor,
    *,
    M_per_rank: int,
    chunk_row_start: int,
    num_splits: int,
):
    rows, N = output_chunk.shape
    grid = lambda META: (triton.cdiv(rows, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),)
    kernel_reduce_chunk_from_scatter[grid](
        input_scatter,
        output_chunk,
        M_per_rank,
        N,
        chunk_row_start,
        rows,
        input_scatter.stride(0),
        input_scatter.stride(1),
        output_chunk.stride(0),
        output_chunk.stride(1),
        NUM_SPLITS=num_splits,
        BLOCK_SIZE_M=128,
        BLOCK_SIZE_N=128,
        num_warps=8,
    )


@dataclasses.dataclass
class NewReduceScatterContext:
    base_ctx: ReduceScatter2DContext
    chunk_rows: int
    num_chunks: int
    chunk_signal: torch.Tensor
    arrival_flag_bufs: List[torch.Tensor]

    @property
    def rank(self):
        return self.base_ctx.rank

    @property
    def world_size(self):
        return self.base_ctx.world_size

    @property
    def local_world_size(self):
        return self.base_ctx.local_world_size

    @property
    def local_rank(self):
        return self.base_ctx.local_rank

    @property
    def nnodes(self):
        return self.base_ctx.nnodes

    @property
    def dtype(self):
        return self.base_ctx.dtype

    @property
    def reduction_stream(self):
        return self.base_ctx.reduction_stream

    @property
    def arrival_flag_buf(self):
        return self.arrival_flag_bufs[self.local_rank]

    def reset_runtime_state(self):
        self.chunk_signal.zero_()
        self.arrival_flag_buf.zero_()
        self.base_ctx.reset_barriers()

    def finalize(self):
        nvshmem_free_tensor_sync(self.arrival_flag_buf)
        self.base_ctx.finalize()


def create_new_reducescatter_2d_ctx(
    max_M: int,
    N: int,
    rank: int,
    world_size: int,
    local_world_size: int,
    dtype: torch.dtype,
    *,
    chunk_rows: int = 0,
    target_chunks_per_rank: int = 4,
    min_chunk_rows: int = 256,
) -> NewReduceScatterContext:
    base_ctx = create_reduce_scater_2d_ctx(max_M, N, rank, world_size, local_world_size, dtype)
    max_m_per_rank = max_M // world_size
    effective_chunk_rows = chunk_rows if chunk_rows > 0 else _auto_chunk_rows(max_m_per_rank, target_chunks_per_rank,
                                                                              min_chunk_rows)
    num_chunks = triton.cdiv(max_m_per_rank, effective_chunk_rows)
    chunk_signal = torch.zeros((world_size * num_chunks,), dtype=torch.int32, device="cuda")
    arrival_flag_bufs = nvshmem_create_tensors((local_world_size * num_chunks,), NVSHMEM_SIGNAL_DTYPE, rank,
                                               local_world_size)
    arrival_flag_bufs[rank % local_world_size].zero_()
    ctx = NewReduceScatterContext(
        base_ctx=base_ctx,
        chunk_rows=effective_chunk_rows,
        num_chunks=num_chunks,
        chunk_signal=chunk_signal,
        arrival_flag_bufs=arrival_flag_bufs,
    )
    nvshmem_barrier_all_on_stream(torch.cuda.current_stream())
    return ctx


def intra_node_chunked_scatter_reduce(
    input_intra_node: torch.Tensor,
    ctx: NewReduceScatterContext,
    output: torch.Tensor,
):
    local_rank = ctx.local_rank
    local_world_size = ctx.local_world_size
    M, N = input_intra_node.shape
    M_per_rank = M // local_world_size
    scatter_bufs_intra_node, _ = ctx.base_ctx.get_scatter_bufs_and_signal_for_each_node(input_intra_node, 0)
    stream = torch.cuda.current_stream()
    nbytes_per_row = N * input_intra_node.dtype.itemsize
    remote_offset_base = local_rank * M_per_rank * nbytes_per_row
    local_buf_base_ptr = input_intra_node.data_ptr()

    for chunk_id in range(ctx.num_chunks):
        row_start = chunk_id * ctx.chunk_rows
        row_end = min(row_start + ctx.chunk_rows, M_per_rank)
        if row_start >= row_end:
            continue
        chunk_nbytes = (row_end - row_start) * nbytes_per_row

        for i in range(local_world_size):
            remote_local_rank = (local_rank + i + 1) % local_world_size
            signal_idx = remote_local_rank * ctx.num_chunks + chunk_id
            _wait_eq_cuda(ctx.chunk_signal[signal_idx], 1, stream)
            remote_buf_ptr = scatter_bufs_intra_node[remote_local_rank].data_ptr() + remote_offset_base + row_start * nbytes_per_row
            local_buf_ptr = local_buf_base_ptr + (remote_local_rank * M_per_rank + row_start) * nbytes_per_row
            (err,) = cudart.cudaMemcpyAsync(
                remote_buf_ptr,
                local_buf_ptr,
                chunk_nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyDefault,
                stream.cuda_stream,
            )
            CUDA_CHECK(err)
            _set_signal_cuda(ctx.arrival_flag_bufs[remote_local_rank][local_rank * ctx.num_chunks + chunk_id], 1, stream)

        chunk_out = output[row_start:row_end]
        with torch.cuda.stream(ctx.reduction_stream):
            for src_rank in range(local_world_size):
                _wait_eq_cuda(ctx.arrival_flag_buf[src_rank * ctx.num_chunks + chunk_id], 1, ctx.reduction_stream)
            reduce_chunk_from_scatter(
                scatter_bufs_intra_node[local_rank],
                chunk_out,
                M_per_rank=M_per_rank,
                chunk_row_start=row_start,
                num_splits=local_world_size,
            )


def new_reduce_scatter_2d_op(input: torch.Tensor, ctx: NewReduceScatterContext, output: Optional[torch.Tensor] = None):
    if ctx.nnodes != 1 or not has_fullmesh_nvlink():
        return reduce_scatter_2d_op(input, ctx.base_ctx, output)

    M, N = input.shape
    output = output or torch.empty((M // ctx.world_size, N), dtype=input.dtype, device=input.device)
    stream = torch.cuda.current_stream()
    intra_node_chunked_scatter_reduce(input, ctx, output)
    stream.wait_stream(ctx.reduction_stream)
    nvshmem_barrier_all_on_stream(stream)
    ctx.reset_runtime_state()
    return output
