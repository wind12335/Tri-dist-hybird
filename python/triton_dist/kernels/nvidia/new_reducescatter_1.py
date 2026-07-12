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
from cuda import cudart

from triton_dist.kernels.nvidia.common_ops import _set_signal_cuda, _wait_eq_cuda
from triton_dist.kernels.nvidia.reduce_scatter import (ReduceScatter2DContext, create_reduce_scater_2d_ctx,
                                                       reduce_scatter_2d_op)
from triton_dist.utils import (CUDA_CHECK, NVSHMEM_SIGNAL_DTYPE, has_fullmesh_nvlink, nvshmem_barrier_all_on_stream,
                               nvshmem_create_tensors, nvshmem_free_tensor_sync)


def _round_up(x: int, align: int) -> int:
    return ((x + align - 1) // align) * align


def _auto_chunk_rows(
    max_m_per_rank: int,
    *,
    target_chunks_per_rank: int,
    min_chunk_rows: int,
    align: int = 256,
) -> int:
    """为单机 NVLink 路径选择一个较稳妥的 chunk 行数。

    设计目标：
    1. chunk 不能太小，否则会把 P2P copy 切得过碎，DIL 会很明显。
    2. chunk 也不能太大，否则又会退化回 rank-level signal，scatter 启动太晚。
    """
    if max_m_per_rank <= 0:
        raise ValueError(f"max_m_per_rank must be > 0, got {max_m_per_rank}")
    if target_chunks_per_rank <= 0:
        raise ValueError(f"target_chunks_per_rank must be > 0, got {target_chunks_per_rank}")
    if min_chunk_rows <= 0:
        raise ValueError(f"min_chunk_rows must be > 0, got {min_chunk_rows}")
    if align <= 0:
        raise ValueError(f"align must be > 0, got {align}")

    rows = triton.cdiv(max_m_per_rank, target_chunks_per_rank)
    rows = max(rows, min_chunk_rows)
    rows = min(rows, max_m_per_rank)
    rows = _round_up(rows, align)
    return min(rows, max_m_per_rank)


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
    """把 scatter buffer 中一个 chunk 对应的所有 split 做本地 reduce。"""
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
) -> None:
    rows, N = output_chunk.shape
    grid = lambda META: (triton.cdiv(rows, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]), )
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
    """单机 chunk-ready reduce-scatter 的扩展上下文。

    `base_ctx` 继续复用官方的 symmetric buffer 与基础 barrier 布局；
    新增的状态只负责更细粒度的 chunk overlap。
    """
    base_ctx: ReduceScatter2DContext
    chunk_rows: int
    num_chunks: int
    chunk_signal: torch.Tensor
    arrival_flag_bufs: List[torch.Tensor]

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
        """一次调用结束后需要清空所有 chunk 级运行时状态。"""
        self.chunk_signal.zero_()
        self.arrival_flag_buf.zero_()
        self.base_ctx.reset_barriers()

    def finalize(self) -> None:
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
    """创建面向单机 NVLink chunk overlap 的新 reduce-scatter 上下文。"""
    base_ctx = create_reduce_scater_2d_ctx(max_M, N, rank, world_size, local_world_size, dtype)
    max_m_per_rank = max_M // world_size
    effective_chunk_rows = chunk_rows if chunk_rows > 0 else _auto_chunk_rows(
        max_m_per_rank,
        target_chunks_per_rank=target_chunks_per_rank,
        min_chunk_rows=min_chunk_rows,
    )
    num_chunks = triton.cdiv(max_m_per_rank, effective_chunk_rows)

    # `chunk_signal[rank_id, chunk_id]`:
    # 由 GEMM producer 在本 GPU 上写入，scatter 在本 GPU 上等待。
    chunk_signal = torch.zeros((world_size * num_chunks, ), dtype=torch.int32, device="cuda")

    # `arrival_flag_bufs[dst_local_rank][src_local_rank, chunk_id]`:
    # copy 完成后由 src GPU 通过对称内存告诉 dst GPU：该 chunk 已到达，可以参与 reduce。
    arrival_flag_bufs = nvshmem_create_tensors((local_world_size * num_chunks, ), NVSHMEM_SIGNAL_DTYPE, rank,
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
) -> None:
    """单机 NVLink 路径的核心流程。

    执行顺序：
    1. 对每个 chunk、每个 src_rank 等待 GEMM 的 chunk-ready signal。
    2. 一旦 ready，立刻把这个 chunk 拷到目标对称 scatter buffer。
    3. copy 完成后，用 arrival flag 告诉目标 GPU：这个 chunk 可以被 reduce。
    4. reduction_stream 在所有 src_rank 的 chunk 都到达后，立刻做该 chunk 的 local reduce。
    """
    local_rank = ctx.local_rank
    local_world_size = ctx.local_world_size
    M, N = input_intra_node.shape
    M_per_rank = M // local_world_size

    # 当前新路径只面向单机；因此 node_id 必然是 0，但这里仍显式写成通用形式，便于后续扩展。
    scatter_bufs_intra_node, _ = ctx.base_ctx.get_scatter_bufs_and_signal_for_each_node(input_intra_node, ctx.node_id)

    scatter_stream = torch.cuda.current_stream()
    nbytes_per_row = N * input_intra_node.dtype.itemsize
    local_buf_base_ptr = input_intra_node.data_ptr()
    remote_offset_base = local_rank * M_per_rank * nbytes_per_row

    for chunk_id in range(ctx.num_chunks):
        row_start = chunk_id * ctx.chunk_rows
        row_end = min(row_start + ctx.chunk_rows, M_per_rank)
        if row_start >= row_end:
            continue
        chunk_nbytes = (row_end - row_start) * nbytes_per_row

        # 依次把每个 src_rank 的这个 chunk 搬到“dst=local_rank”的 scatter 位置上。
        # 这里保留 self-copy，是为了让 local reduce 能统一读取 `[num_splits, M_per_rank, N]` 布局。
        for step in range(local_world_size):
            src_local_rank = (local_rank + step + 1) % local_world_size
            signal_idx = src_local_rank * ctx.num_chunks + chunk_id
            _wait_eq_cuda(ctx.chunk_signal[signal_idx], 1, scatter_stream)

            remote_buf_ptr = scatter_bufs_intra_node[src_local_rank].data_ptr() + remote_offset_base + row_start * nbytes_per_row
            local_buf_ptr = local_buf_base_ptr + (src_local_rank * M_per_rank + row_start) * nbytes_per_row
            (err, ) = cudart.cudaMemcpyAsync(
                remote_buf_ptr,
                local_buf_ptr,
                chunk_nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyDefault,
                scatter_stream.cuda_stream,
            )
            CUDA_CHECK(err)

            arrival_idx = local_rank * ctx.num_chunks + chunk_id
            _set_signal_cuda(ctx.arrival_flag_bufs[src_local_rank][arrival_idx], 1, scatter_stream)

        # reduce 侧放到独立 stream，让 chunk k 的 reduce 能和 chunk k+1 的 scatter 重叠。
        chunk_out = output[row_start:row_end]
        with torch.cuda.stream(ctx.reduction_stream):
            for src_local_rank in range(local_world_size):
                wait_idx = src_local_rank * ctx.num_chunks + chunk_id
                _wait_eq_cuda(ctx.arrival_flag_buf[wait_idx], 1, ctx.reduction_stream)

            reduce_chunk_from_scatter(
                scatter_bufs_intra_node[local_rank],
                chunk_out,
                M_per_rank=M_per_rank,
                chunk_row_start=row_start,
                num_splits=local_world_size,
            )


def new_reduce_scatter_2d_op(
    input: torch.Tensor,
    ctx: NewReduceScatterContext,
    output: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """新的 reduce-scatter 入口。

    目前重点支持：
    - 单机
    - full-mesh NVLink
    - chunk-ready overlap

    其他场景自动回退到官方 `reduce_scatter_2d_op`。
    """
    if ctx.nnodes != 1 or not has_fullmesh_nvlink():
        return reduce_scatter_2d_op(input, ctx.base_ctx, output)

    M, N = input.shape
    if output is None:
        output = torch.empty((M // ctx.world_size, N), dtype=input.dtype, device=input.device)

    stream = torch.cuda.current_stream()
    intra_node_chunked_scatter_reduce(input, ctx, output)
    stream.wait_stream(ctx.reduction_stream)
    nvshmem_barrier_all_on_stream(stream)
    ctx.reset_runtime_state()
    return output
