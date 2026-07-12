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

import torch
import triton
import triton.language as tl
import triton_dist
import triton_dist.language as dl
from triton_dist.language.extra.language_extra import __syncthreads, atomic_add, st

from triton_dist.kernels.nvidia.gemm_reduce_scatter import _matmul_launch_metadata
from triton_dist.kernels.nvidia.gemm_rs_threadblock_swizzle import threadblock_swizzle_gemm_reduce_scatter_kernel
from triton_dist.kernels.nvidia.new_3rdgemm_reducescatterv3 import (
    gemm_rs_producer_non_persistent_chunked,
    swizzle_2d,
)


@triton_dist.jit(launch_metadata=_matmul_launch_metadata)
def kernel_gemm_rs_producer_non_persistent_chunked_banded_ready_frontier_dualphase(
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
    chunk_signal_ptr,
    counter_ptr,
    signal_value,
    LOCAL_WORLD_SIZE: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    NUM_CHUNKS: tl.constexpr,
    CHUNK_ROWS: tl.constexpr,
    N_BANDS: tl.constexpr,
    MAX_BAND_COLS: tl.constexpr,
    FRONTIER_CHUNKS: tl.constexpr,
    PHASE_KIND: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    tl.static_assert(a_ptr.dtype.is_ptr(), "A should be a pointer")
    tl.static_assert(b_ptr.dtype.is_ptr(), "B should be a pointer")
    tl.static_assert(c_ptr.dtype.is_ptr(), "C should be a pointer")
    tl.static_assert(CHUNK_ROWS >= BLOCK_SIZE_M, "Current chunked producer assumes CHUNK_ROWS >= BLOCK_SIZE_M")
    tl.static_assert(N_BANDS >= 1, "N_BANDS should be >= 1")
    tl.static_assert(MAX_BAND_COLS >= 1, "MAX_BAND_COLS should be >= 1")

    a_dtype = a_ptr.dtype.element_ty
    b_dtype = b_ptr.dtype.element_ty
    c_dtype = c_ptr.dtype.element_ty
    tl.static_assert(a_dtype == b_dtype, "A and B should have the same dtype")

    rank = dl.rank()
    nnodes = WORLD_SIZE // LOCAL_WORLD_SIZE

    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_m_total = tl.cdiv(M, BLOCK_SIZE_M)
    m_per_rank = M // WORLD_SIZE
    tiles_per_segment = tl.cdiv(m_per_rank, BLOCK_SIZE_M)
    frontier_rows = tl.minimum(CHUNK_ROWS * FRONTIER_CHUNKS, m_per_rank)
    frontier_tiles_per_segment = tl.minimum(tl.cdiv(frontier_rows, BLOCK_SIZE_M), tiles_per_segment)
    tail_tiles_per_segment = tiles_per_segment - frontier_tiles_per_segment

    phase_tiles_per_segment = tl.where(PHASE_KIND == 0, frontier_tiles_per_segment, tail_tiles_per_segment)
    phase_total_pid_m = phase_tiles_per_segment * WORLD_SIZE
    if phase_total_pid_m <= 0:
        return

    pid_m_phase, pid_n = swizzle_2d(pid, phase_total_pid_m, num_pid_n, GROUP_SIZE_M)
    if nnodes != 1:
        # v5 focuses on the single-node path. Keep the existing inter-node mapping behavior.
        global_pid_m = threadblock_swizzle_gemm_reduce_scatter_kernel(pid_m_phase, M, rank, WORLD_SIZE, nnodes,
                                                                      BLOCK_SIZE_M)
    else:
        if PHASE_KIND == 0:
            tail_space_pid_m = pid_m_phase
        else:
            tail_phase_tile_offset = ((rank + 1) % WORLD_SIZE) * tail_tiles_per_segment
            tail_space_pid_m = (pid_m_phase + tail_phase_tile_offset) % phase_total_pid_m

        segment_id = tail_space_pid_m // phase_tiles_per_segment
        local_tile_in_segment = tail_space_pid_m % phase_tiles_per_segment
        local_tile_base = tl.where(PHASE_KIND == 0, 0, frontier_tiles_per_segment)
        global_pid_m = segment_id * tiles_per_segment + local_tile_base + local_tile_in_segment

    if global_pid_m >= num_pid_m_total:
        return

    offs_am = (global_pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    if a_dtype == tl.int8:
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.int32)
    else:
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    offs_cm = global_pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    out_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator.to(c_dtype), mask=out_mask)

    num_pid_n_per_band = tl.cdiv(MAX_BAND_COLS, BLOCK_SIZE_N)
    band_id = pid_n // num_pid_n_per_band
    local_pid_n = pid_n - band_id * num_pid_n_per_band
    band_col_start = band_id * MAX_BAND_COLS
    band_cols = tl.maximum(0, tl.minimum(MAX_BAND_COLS, N - band_col_start))
    num_pid_n_in_band = tl.cdiv(band_cols, BLOCK_SIZE_N)
    valid_band_tile = local_pid_n < num_pid_n_in_band

    tile_row_start = global_pid_m * BLOCK_SIZE_M
    tile_row_end = min((global_pid_m + 1) * BLOCK_SIZE_M, M) - 1
    segment_start = tile_row_start // m_per_rank
    segment_end = tile_row_end // m_per_rank
    __syncthreads()

    for seg_offset in tl.static_range(0, 2):
        segment = segment_start + seg_offset
        if segment <= segment_end:
            seg_global_row_start = segment * m_per_rank
            seg_global_row_end = seg_global_row_start + m_per_rank - 1
            seg_local_start = tl.maximum(tile_row_start, seg_global_row_start) - seg_global_row_start
            seg_local_end = tl.minimum(tile_row_end, seg_global_row_end) - seg_global_row_start
            chunk_start = seg_local_start // CHUNK_ROWS
            chunk_end = seg_local_end // CHUNK_ROWS

            for chunk_offset in tl.static_range(0, 2):
                chunk_id = chunk_start + chunk_offset
                if chunk_id <= chunk_end and valid_band_tile:
                    chunk_row_start = seg_global_row_start + chunk_id * CHUNK_ROWS
                    chunk_row_end = tl.minimum(chunk_row_start + CHUNK_ROWS, seg_global_row_end + 1) - 1
                    tiled_m_start = chunk_row_start // BLOCK_SIZE_M
                    tiled_m_end = chunk_row_end // BLOCK_SIZE_M
                    tiled_m_size = tiled_m_end - tiled_m_start + 1

                    signal_idx = band_id * WORLD_SIZE * NUM_CHUNKS + segment * NUM_CHUNKS + chunk_id
                    val = atomic_add(counter_ptr + signal_idx, 1, semantic="release", scope="gpu")
                    if val == num_pid_n_in_band * tiled_m_size - 1:
                        st(chunk_signal_ptr + signal_idx, signal_value, scope="gpu", semantic="release")


def gemm_rs_producer_non_persistent_chunked_banded_ready_frontier_dualphase(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    chunk_signal: torch.Tensor,
    workspace: torch.Tensor,
    world_size: int,
    local_world_size: int,
    num_chunks: int,
    chunk_rows: int,
    n_bands: int,
    max_band_cols: int,
    frontier_chunks: int,
    signal_value: int,
    gemm_config: triton.Config,
) -> None:
    assert A.shape[1] == B.shape[0], "Incompatible dimensions"
    assert A.dtype == B.dtype, "Incompatible dtypes"
    assert n_bands >= 1, "n_bands must be >= 1"
    assert max_band_cols >= 1, "max_band_cols must be >= 1"

    M, K_per_rank = A.shape
    _, N = B.shape
    block_size_m = gemm_config.kwargs["BLOCK_SIZE_M"]
    block_size_n = gemm_config.kwargs["BLOCK_SIZE_N"]
    m_per_rank = M // world_size
    tiles_per_segment = triton.cdiv(m_per_rank, block_size_m)
    frontier_chunks = max(0, min(frontier_chunks, num_chunks))
    frontier_rows = min(chunk_rows * frontier_chunks, m_per_rank)
    frontier_tiles_per_segment = min(triton.cdiv(frontier_rows, block_size_m), tiles_per_segment)
    tail_tiles_per_segment = tiles_per_segment - frontier_tiles_per_segment

    workspace.zero_()

    if frontier_tiles_per_segment > 0:
        frontier_grid = (frontier_tiles_per_segment * world_size * triton.cdiv(N, block_size_n),)
        kernel_gemm_rs_producer_non_persistent_chunked_banded_ready_frontier_dualphase[frontier_grid](
            A,
            B,
            C,
            M,
            N,
            K_per_rank,
            A.stride(0),
            A.stride(1),
            B.stride(0),
            B.stride(1),
            C.stride(0),
            C.stride(1),
            chunk_signal,
            workspace,
            signal_value,
            local_world_size,
            world_size,
            NUM_CHUNKS=num_chunks,
            CHUNK_ROWS=chunk_rows,
            N_BANDS=n_bands,
            MAX_BAND_COLS=max_band_cols,
            FRONTIER_CHUNKS=frontier_chunks,
            PHASE_KIND=0,
            **gemm_config.all_kwargs(),
        )

    if tail_tiles_per_segment > 0:
        tail_grid = (tail_tiles_per_segment * world_size * triton.cdiv(N, block_size_n),)
        kernel_gemm_rs_producer_non_persistent_chunked_banded_ready_frontier_dualphase[tail_grid](
            A,
            B,
            C,
            M,
            N,
            K_per_rank,
            A.stride(0),
            A.stride(1),
            B.stride(0),
            B.stride(1),
            C.stride(0),
            C.stride(1),
            chunk_signal,
            workspace,
            signal_value,
            local_world_size,
            world_size,
            NUM_CHUNKS=num_chunks,
            CHUNK_ROWS=chunk_rows,
            N_BANDS=n_bands,
            MAX_BAND_COLS=max_band_cols,
            FRONTIER_CHUNKS=frontier_chunks,
            PHASE_KIND=1,
            **gemm_config.all_kwargs(),
        )


__all__ = [
    "gemm_rs_producer_non_persistent_chunked",
    "gemm_rs_producer_non_persistent_chunked_banded_ready_frontier_dualphase",
]
