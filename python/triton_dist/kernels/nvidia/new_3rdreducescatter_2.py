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
from triton_dist.kernels.nvidia.reduce_scatter import (ReduceScatter2DContext, create_reduce_scater_2d_ctx,
                                                       reduce_scatter_2d_op)
from triton_dist.language.extra.language_extra import __syncthreads, ld, tid
from triton_dist.utils import (CUDA_CHECK, NVSHMEM_SIGNAL_DTYPE, has_fullmesh_nvlink, nvshmem_barrier_all_on_stream,
                               nvshmem_create_tensors, nvshmem_free_tensor_sync)


def _debug_enabled() -> bool:
    return os.environ.get("TRITON_DIST_NEW_3RD_DEBUG", "0") == "1"


def _debug_log(ctx: New3rdReduceScatterContext, msg: str) -> None:
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


@triton.jit(do_not_specialize=["M_per_rank", "N", "num_runtime_chunks"])
def kernel_persistent_chunk_reduce_from_scatter(
    scatter_ptr,
    arrival_flag_ptr,
    output_ptr,
    M_per_rank,
    N,
    num_runtime_chunks,
    stride_sm,
    stride_sn,
    stride_om,
    stride_on,
    NUM_SPLITS: tl.constexpr,
    NUM_CHUNKS: tl.constexpr,
    CHUNK_ROWS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """Persistent helper that waits chunk-by-chunk and reduces locally.

    Each CTA stays alive for the whole RS stage:
    - wait for all source arrivals of chunk k
    - reduce chunk k from the local scatter layout into final output
    - continue with chunk k+1

    This removes the old "copy + add + add + add" tail made of multiple host-side
    kernel launches and turns the third stream into a real arrival-driven helper.
    """
    pid = tl.program_id(axis=0)
    num_pid = tl.num_programs(axis=0)
    thread_idx = tid(0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    max_tiles_per_chunk = tl.cdiv(CHUNK_ROWS, BLOCK_SIZE_M) * num_pid_n

    for chunk_id in range(NUM_CHUNKS):
        if chunk_id < num_runtime_chunks:
            if thread_idx == 0:
                for src_local_rank in range(NUM_SPLITS):
                    wait_idx = src_local_rank * NUM_CHUNKS + chunk_id
                    while ld(arrival_flag_ptr + wait_idx, scope="sys", semantic="acquire") != 1:
                        pass
            __syncthreads()

            row_start = chunk_id * CHUNK_ROWS
            for tile_id in range(pid, max_tiles_per_chunk, num_pid):
                pid_m = tile_id // num_pid_n
                pid_n = tile_id % num_pid_n
                offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
                offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
                rows = row_start + offs_m
                mask = (rows[:, None] < M_per_rank) & (offs_n[None, :] < N)

                acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
                for src_local_rank in range(NUM_SPLITS):
                    src_rows = src_local_rank * M_per_rank + rows
                    src_ptrs = scatter_ptr + src_rows[:, None] * stride_sm + offs_n[None, :] * stride_sn
                    acc += tl.load(src_ptrs, mask=mask, other=0.0)

                out_ptrs = output_ptr + rows[:, None] * stride_om + offs_n[None, :] * stride_on
                tl.store(out_ptrs, acc.to(output_ptr.dtype.element_ty), mask=mask)


@dataclasses.dataclass
class New3rdReduceScatterContext:
    """Single-node chunk-arrival reduce-scatter with a persistent helper stream.

    The design matches the intended "third stream" idea more closely than the
    previous bring-up prototype:
    - GEMM emits chunk-ready signals instead of rank-level segment-ready signals
    - scatter copies each destination chunk as soon as it is ready
    - each destination rank exposes `[src_local_rank, chunk_id]` arrival flags
    - a low-SM persistent helper kernel waits on those flags and immediately
      reduces the arrived chunk into final output
    """

    base_ctx: ReduceScatter2DContext
    chunk_rows: int
    num_chunks: int
    chunk_signal: torch.Tensor
    arrival_flag_bufs: List[torch.Tensor]
    helper_num_sms: int = 4

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
    def reduction_stream(self) -> torch.cuda.Stream:
        return self.base_ctx.reduction_stream

    @property
    def arrival_flag_buf(self) -> torch.Tensor:
        return self.arrival_flag_bufs[self.local_rank]

    def reset_runtime_state(self) -> None:
        self.chunk_signal.zero_()
        self.arrival_flag_buf.zero_()
        self.base_ctx.reset_barriers()

    def finalize(self) -> None:
        nvshmem_free_tensor_sync(self.arrival_flag_buf)
        self.base_ctx.finalize()


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
) -> New3rdReduceScatterContext:
    base_ctx = create_reduce_scater_2d_ctx(max_M, N, rank, world_size, local_world_size, dtype)
    max_m_per_rank = max_M // world_size
    effective_chunk_rows = chunk_rows if chunk_rows > 0 else _auto_chunk_rows(
        max_m_per_rank,
        target_chunks_per_rank=target_chunks_per_rank,
        min_chunk_rows=min_chunk_rows,
    )
    num_chunks = triton.cdiv(max_m_per_rank, effective_chunk_rows)
    chunk_signal = torch.zeros((world_size * num_chunks,), dtype=torch.int32, device="cuda")

    nvshmem_barrier_all_on_stream(torch.cuda.current_stream())
    arrival_flag_bufs = nvshmem_create_tensors((local_world_size * num_chunks,), NVSHMEM_SIGNAL_DTYPE, rank,
                                               local_world_size)
    arrival_flag_bufs[rank % local_world_size].zero_()
    nvshmem_barrier_all_on_stream(torch.cuda.current_stream())
    return New3rdReduceScatterContext(
        base_ctx=base_ctx,
        chunk_rows=effective_chunk_rows,
        num_chunks=num_chunks,
        chunk_signal=chunk_signal,
        arrival_flag_bufs=arrival_flag_bufs,
        helper_num_sms=helper_num_sms,
    )


def intra_node_3rd_scatter_reduce(
    input_intra_node: torch.Tensor,
    ctx: New3rdReduceScatterContext,
    output: torch.Tensor,
) -> None:
    local_rank = ctx.local_rank
    local_world_size = ctx.local_world_size
    scatter_stream = torch.cuda.current_stream()

    scatter_bufs_intra_node, _ = ctx.base_ctx.get_scatter_bufs_and_signal_for_each_node(input_intra_node, ctx.node_id)
    M, N = input_intra_node.shape
    M_per_rank = M // local_world_size
    nbytes_per_row = N * input_intra_node.dtype.itemsize
    local_buf_base_ptr = input_intra_node.data_ptr()
    remote_offset_base = local_rank * M_per_rank * nbytes_per_row
    num_runtime_chunks = triton.cdiv(M_per_rank, ctx.chunk_rows)
    local_scatter = scatter_bufs_intra_node[local_rank]

    _debug_log(
        ctx,
        f"scatter start: local_world_size={local_world_size}, M={M}, N={N}, chunk_rows={ctx.chunk_rows}, "
        f"num_runtime_chunks={num_runtime_chunks}, helper_num_sms={ctx.helper_num_sms}",
    )

    with torch.cuda.stream(ctx.reduction_stream):
        kernel_persistent_chunk_reduce_from_scatter[(ctx.helper_num_sms,)](
            local_scatter,
            ctx.arrival_flag_buf,
            output,
            M_per_rank,
            N,
            num_runtime_chunks,
            local_scatter.stride(0),
            local_scatter.stride(1),
            output.stride(0),
            output.stride(1),
            NUM_SPLITS=local_world_size,
            NUM_CHUNKS=ctx.num_chunks,
            CHUNK_ROWS=ctx.chunk_rows,
            BLOCK_SIZE_M=128,
            BLOCK_SIZE_N=128,
            num_warps=4,
        )
    _debug_log(ctx, "persistent helper launched")

    for chunk_id in range(num_runtime_chunks):
        row_start = chunk_id * ctx.chunk_rows
        row_end = min(row_start + ctx.chunk_rows, M_per_rank)
        chunk_nbytes = (row_end - row_start) * nbytes_per_row

        for step in range(local_world_size):
            dest_local_rank = (local_rank + step + 1) % local_world_size
            signal_idx = dest_local_rank * ctx.num_chunks + chunk_id
            _debug_log(ctx, f"scatter chunk {chunk_id} step {step}: waiting chunk-ready for dest_local_rank={dest_local_rank}")
            _wait_eq_cuda(ctx.chunk_signal[signal_idx], 1, scatter_stream)

            remote_buf_ptr = scatter_bufs_intra_node[dest_local_rank].data_ptr() + remote_offset_base + row_start * nbytes_per_row
            local_buf_ptr = local_buf_base_ptr + (dest_local_rank * M_per_rank + row_start) * nbytes_per_row
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
            _set_signal_cuda(remote_ready, 1, scatter_stream)
            _debug_log(
                ctx,
                f"scatter chunk {chunk_id} step {step}: memcpy+arrival issued to dest_local_rank={dest_local_rank}",
            )


def new_3rd_reduce_scatter_2d_op(
    input: torch.Tensor,
    ctx: New3rdReduceScatterContext,
    output: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if ctx.nnodes != 1 or not has_fullmesh_nvlink():
        return reduce_scatter_2d_op(input, ctx.base_ctx, output)

    M, N = input.shape
    if output is None:
        output = torch.empty((M // ctx.world_size, N), dtype=input.dtype, device=input.device)

    _debug_log(ctx, f"new_3rd_reduce_scatter_2d_op enter: input_shape={tuple(input.shape)}")
    intra_node_3rd_scatter_reduce(input, ctx, output)
    _debug_log(ctx, "new_3rd_reduce_scatter_2d_op waiting helper stream")
    torch.cuda.current_stream().wait_stream(ctx.reduction_stream)
    _debug_log(ctx, "new_3rd_reduce_scatter_2d_op helper stream finished")
    ctx.reset_runtime_state()
    _debug_log(ctx, "new_3rd_reduce_scatter_2d_op exit")
    return output
