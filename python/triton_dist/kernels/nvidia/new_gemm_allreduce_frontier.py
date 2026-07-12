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
from triton_dist.language.extra.language_extra import __syncthreads, atomic_add, st

from triton_dist.kernels.nvidia.gemm_reduce_scatter import _matmul_launch_metadata
from triton_dist.kernels.nvidia.new_3rdgemm_reducescatterv3 import swizzle_2d


@triton_dist.jit(launch_metadata=_matmul_launch_metadata)
def kernel_gemm_ar_producer_non_persistent_chunked_banded_ready_frontier_dualphase(
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
    ready_signal_ptr,
    counter_ptr,
    signal_value,
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
    tl.static_assert(CHUNK_ROWS >= BLOCK_SIZE_M, "CHUNK_ROWS must be >= BLOCK_SIZE_M")
    tl.static_assert(N_BANDS >= 1, "N_BANDS must be >= 1")
    tl.static_assert(MAX_BAND_COLS >= 1, "MAX_BAND_COLS must be >= 1")

    a_dtype = a_ptr.dtype.element_ty
    c_dtype = c_ptr.dtype.element_ty

    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_pid_m = tl.cdiv(M, BLOCK_SIZE_M)

    frontier_rows = tl.minimum(CHUNK_ROWS * FRONTIER_CHUNKS, M)
    frontier_pid_m = tl.minimum(tl.cdiv(frontier_rows, BLOCK_SIZE_M), total_pid_m)
    tail_pid_m = total_pid_m - frontier_pid_m
    phase_pid_m = tl.where(PHASE_KIND == 0, frontier_pid_m, tail_pid_m)
    if phase_pid_m <= 0:
        return

    pid_m_phase, pid_n = swizzle_2d(pid, phase_pid_m, num_pid_n, GROUP_SIZE_M)
    global_pid_m = tl.where(PHASE_KIND == 0, pid_m_phase, frontier_pid_m + pid_m_phase)
    if global_pid_m >= total_pid_m:
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
    tile_row_end = tl.minimum((global_pid_m + 1) * BLOCK_SIZE_M, M) - 1
    chunk_start = tile_row_start // CHUNK_ROWS
    chunk_end = tile_row_end // CHUNK_ROWS
    __syncthreads()

    for chunk_offset in tl.static_range(0, 2):
        chunk_id = chunk_start + chunk_offset
        if chunk_id <= chunk_end and valid_band_tile:
            chunk_row_start = chunk_id * CHUNK_ROWS
            chunk_row_end = tl.minimum(chunk_row_start + CHUNK_ROWS, M) - 1
            tiled_m_start = chunk_row_start // BLOCK_SIZE_M
            tiled_m_end = chunk_row_end // BLOCK_SIZE_M
            tiled_m_size = tiled_m_end - tiled_m_start + 1
            signal_idx = band_id * NUM_CHUNKS + chunk_id
            val = atomic_add(counter_ptr + signal_idx, 1, semantic="release", scope="gpu")
            if val == num_pid_n_in_band * tiled_m_size - 1:
                st(ready_signal_ptr + signal_idx, signal_value, scope="gpu", semantic="release")


def gemm_ar_producer_non_persistent_chunked_banded_ready_frontier_dualphase(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    ready_signal: torch.Tensor,
    workspace: torch.Tensor,
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

    M, K = A.shape
    _, N = B.shape
    block_size_m = gemm_config.kwargs["BLOCK_SIZE_M"]
    block_size_n = gemm_config.kwargs["BLOCK_SIZE_N"]
    total_pid_m = triton.cdiv(M, block_size_m)
    frontier_chunks = max(0, min(frontier_chunks, num_chunks))
    frontier_rows = min(chunk_rows * frontier_chunks, M)
    frontier_pid_m = min(triton.cdiv(frontier_rows, block_size_m), total_pid_m)
    tail_pid_m = total_pid_m - frontier_pid_m

    workspace.zero_()
    if frontier_pid_m > 0:
        frontier_grid = (frontier_pid_m * triton.cdiv(N, block_size_n),)
        kernel_gemm_ar_producer_non_persistent_chunked_banded_ready_frontier_dualphase[frontier_grid](
            A,
            B,
            C,
            M,
            N,
            K,
            A.stride(0),
            A.stride(1),
            B.stride(0),
            B.stride(1),
            C.stride(0),
            C.stride(1),
            ready_signal,
            workspace,
            signal_value,
            NUM_CHUNKS=num_chunks,
            CHUNK_ROWS=chunk_rows,
            N_BANDS=n_bands,
            MAX_BAND_COLS=max_band_cols,
            FRONTIER_CHUNKS=frontier_chunks,
            PHASE_KIND=0,
            **gemm_config.all_kwargs(),
        )

    if tail_pid_m > 0:
        tail_grid = (tail_pid_m * triton.cdiv(N, block_size_n),)
        kernel_gemm_ar_producer_non_persistent_chunked_banded_ready_frontier_dualphase[tail_grid](
            A,
            B,
            C,
            M,
            N,
            K,
            A.stride(0),
            A.stride(1),
            B.stride(0),
            B.stride(1),
            C.stride(0),
            C.stride(1),
            ready_signal,
            workspace,
            signal_value,
            NUM_CHUNKS=num_chunks,
            CHUNK_ROWS=chunk_rows,
            N_BANDS=n_bands,
            MAX_BAND_COLS=max_band_cols,
            FRONTIER_CHUNKS=frontier_chunks,
            PHASE_KIND=1,
            **gemm_config.all_kwargs(),
        )


__all__ = [
    "gemm_ar_producer_non_persistent_chunked_banded_ready_frontier_dualphase",
]
