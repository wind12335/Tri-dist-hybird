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
from typing import Optional

import torch
import triton
import triton.language as tl
import triton_dist
import triton_dist.language as dl
import triton_dist.tune
from triton_dist.language.extra.language_extra import __syncthreads, atomic_add, st

from triton_dist.kernels.nvidia.gemm import get_config_space
from triton_dist.kernels.nvidia.gemm_reduce_scatter import update_triton_config
from triton_dist.kernels.nvidia.new_3rdreducescatter import (New3rdReduceScatterContext,
                                                             create_new_3rd_reducescatter_2d_ctx,
                                                             new_3rd_reduce_scatter_2d_op)
from triton_dist.kernels.nvidia.gemm_reduce_scatter import _matmul_launch_metadata
from triton_dist.kernels.nvidia.gemm_rs_threadblock_swizzle import threadblock_swizzle_gemm_reduce_scatter_kernel
from triton_dist.utils import (get_device_max_shared_memory_size, has_fullmesh_nvlink, nvshmem_barrier_all_on_stream,
                               nvshmem_create_tensors, nvshmem_free_tensor_sync)


def _debug_enabled() -> bool:
    return os.environ.get("TRITON_DIST_NEW_3RD_DEBUG", "0") == "1"


def _debug_log(ctx: New3rdGEMMReduceScatterTensorParallelContext, msg: str) -> None:
    if _debug_enabled():
        print(f"[new_3rd][rank{ctx.rs_ctx.rank}] {msg}", flush=True)


@triton.jit
def swizzle_2d(tile_id, num_pid_m, num_pid_n, GROUP_SIZE_M: tl.constexpr):
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (tile_id % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n


@triton_dist.jit(launch_metadata=_matmul_launch_metadata)
def kernel_gemm_rs_producer_non_persistent_chunked(
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
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    tl.static_assert(a_ptr.dtype.is_ptr(), "A should be a pointer")
    tl.static_assert(b_ptr.dtype.is_ptr(), "B should be a pointer")
    tl.static_assert(c_ptr.dtype.is_ptr(), "C should be a pointer")
    tl.static_assert(CHUNK_ROWS >= BLOCK_SIZE_M, "Current chunked producer assumes CHUNK_ROWS >= BLOCK_SIZE_M")

    a_dtype = a_ptr.dtype.element_ty
    b_dtype = b_ptr.dtype.element_ty
    c_dtype = c_ptr.dtype.element_ty
    tl.static_assert(a_dtype == b_dtype, "A and B should have the same dtype")

    rank = dl.rank()
    nnodes = WORLD_SIZE // LOCAL_WORLD_SIZE

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    M_per_rank = M // WORLD_SIZE

    pid_m, pid_n = swizzle_2d(pid, num_pid_m, num_pid_n, GROUP_SIZE_M)
    if nnodes != 1:
        pid_m = threadblock_swizzle_gemm_reduce_scatter_kernel(pid_m, M, rank, WORLD_SIZE, nnodes, BLOCK_SIZE_M)
    else:
        # Unlike the upstream GEMM+RS path, the new_3rd consumer is destination-local:
        # chunk d on GPU d only becomes reducible after *all* source ranks have produced
        # that same destination chunk. Reusing the upstream per-rank M rotation would make
        # different GPUs start from different destination segments, which delays the first
        # all-source-ready chunk and pushes overlap too far to the right in the timeline.
        #
        # Keep the single-node producer aligned across ranks so each destination chunk
        # reaches "all peers ready" earlier.
        pid_m_offset = 0
        pid_m = (pid_m + pid_m_offset) % num_pid_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
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

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    out_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator.to(c_dtype), mask=out_mask)

    tile_row_start = pid_m * BLOCK_SIZE_M
    tile_row_end = min((pid_m + 1) * BLOCK_SIZE_M, M) - 1
    segment_start = tile_row_start // M_per_rank
    segment_end = tile_row_end // M_per_rank
    __syncthreads()

    for seg_offset in tl.static_range(0, 2):
        segment = segment_start + seg_offset
        if segment <= segment_end:
            seg_global_row_start = segment * M_per_rank
            seg_global_row_end = seg_global_row_start + M_per_rank - 1
            seg_local_start = tl.maximum(tile_row_start, seg_global_row_start) - seg_global_row_start
            seg_local_end = tl.minimum(tile_row_end, seg_global_row_end) - seg_global_row_start
            chunk_start = seg_local_start // CHUNK_ROWS
            chunk_end = seg_local_end // CHUNK_ROWS

            for chunk_offset in tl.static_range(0, 2):
                chunk_id = chunk_start + chunk_offset
                if chunk_id <= chunk_end:
                    chunk_row_start = seg_global_row_start + chunk_id * CHUNK_ROWS
                    chunk_row_end = tl.minimum(chunk_row_start + CHUNK_ROWS, seg_global_row_end + 1) - 1
                    tiled_m_start = chunk_row_start // BLOCK_SIZE_M
                    tiled_m_end = chunk_row_end // BLOCK_SIZE_M
                    tiled_m_size = tiled_m_end - tiled_m_start + 1

                    signal_idx = segment * NUM_CHUNKS + chunk_id
                    val = atomic_add(counter_ptr + signal_idx, 1, semantic="release", scope="gpu")
                    if val == num_pid_n * tiled_m_size - 1:
                        st(chunk_signal_ptr + signal_idx, signal_value, scope="gpu", semantic="release")


def gemm_rs_producer_non_persistent_chunked(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    chunk_signal: torch.Tensor,
    workspace: torch.Tensor,
    world_size: int,
    local_world_size: int,
    num_chunks: int,
    chunk_rows: int,
    signal_value: int,
    gemm_config: triton.Config,
) -> None:
    assert A.shape[1] == B.shape[0], "Incompatible dimensions"
    assert A.dtype == B.dtype, "Incompatible dtypes"

    M, K_per_rank = A.shape
    _, N = B.shape
    block_size_m = gemm_config.kwargs["BLOCK_SIZE_M"]
    block_size_n = gemm_config.kwargs["BLOCK_SIZE_N"]
    grid = (triton.cdiv(M, block_size_m) * triton.cdiv(N, block_size_n),)

    kernel_gemm_rs_producer_non_persistent_chunked[grid](
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
        **gemm_config.all_kwargs(),
    )


@triton_dist.jit(launch_metadata=_matmul_launch_metadata)
def kernel_gemm_rs_producer_non_persistent_chunked_banded_ready(
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
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    M_per_rank = M // WORLD_SIZE

    pid_m, pid_n = swizzle_2d(pid, num_pid_m, num_pid_n, GROUP_SIZE_M)
    if nnodes != 1:
        pid_m = threadblock_swizzle_gemm_reduce_scatter_kernel(pid_m, M, rank, WORLD_SIZE, nnodes, BLOCK_SIZE_M)
    else:
        pid_m_offset = 0
        pid_m = (pid_m + pid_m_offset) % num_pid_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
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

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
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

    tile_row_start = pid_m * BLOCK_SIZE_M
    tile_row_end = min((pid_m + 1) * BLOCK_SIZE_M, M) - 1
    segment_start = tile_row_start // M_per_rank
    segment_end = tile_row_end // M_per_rank
    __syncthreads()

    for seg_offset in tl.static_range(0, 2):
        segment = segment_start + seg_offset
        if segment <= segment_end:
            seg_global_row_start = segment * M_per_rank
            seg_global_row_end = seg_global_row_start + M_per_rank - 1
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


def gemm_rs_producer_non_persistent_chunked_banded_ready(
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
    grid = (triton.cdiv(M, block_size_m) * triton.cdiv(N, block_size_n),)

    kernel_gemm_rs_producer_non_persistent_chunked_banded_ready[grid](
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
        **gemm_config.all_kwargs(),
    )


@dataclasses.dataclass
class New3rdGEMMReduceScatterTensorParallelContext:
    """GEMM + third-stream reduce-scatter context.

    The GEMM producer path stays identical to the upstream nonfused version.
    The innovation is entirely in `rs_ctx`, which replaces the tail one-shot
    reduction with destination-local arrival-driven incremental reduction.
    """

    rs_ctx: New3rdReduceScatterContext
    output_dtype: torch.dtype
    gemm_out_bufs: list[torch.Tensor]
    rs_stream: torch.cuda.Stream
    num_gemm_sms: int

    def finalize(self) -> None:
        self.rs_ctx.finalize()
        nvshmem_free_tensor_sync(self.gemm_out_bufs[self.rs_ctx.local_rank])

    def get_gemm_out_buf(self, input: torch.Tensor) -> torch.Tensor:
        M, _ = input.shape
        return self.gemm_out_bufs[self.rs_ctx.local_rank][:M]


def create_new_3rd_gemm_rs_context(
    max_M: int,
    N: int,
    rank: int,
    world_size: int,
    local_world_size: int,
    output_dtype: torch.dtype,
    rs_stream: torch.cuda.Stream,
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
) -> New3rdGEMMReduceScatterTensorParallelContext:
    rs_ctx = create_new_3rd_reducescatter_2d_ctx(
        max_M,
        N,
        rank,
        world_size,
        local_world_size,
        output_dtype,
        chunk_rows=chunk_rows,
        target_chunks_per_rank=target_chunks_per_rank,
        min_chunk_rows=min_chunk_rows,
        helper_num_sms=helper_num_sms,
        steady_sms=steady_sms,
        tail_sms=tail_sms,
        stage_slots=stage_slots,
        accum_dtype=accum_dtype,
        use_scratch=use_scratch,
        tail_chunk_window=tail_chunk_window,
        local_seed_direct=local_seed_direct,
        comm_stream=rs_stream,
    )
    num_sms = torch.cuda.get_device_properties("cuda").multi_processor_count
    gemm_out_bufs = nvshmem_create_tensors((max_M, N), output_dtype, rank, local_world_size)
    ctx = New3rdGEMMReduceScatterTensorParallelContext(
        rs_ctx=rs_ctx,
        output_dtype=output_dtype,
        gemm_out_bufs=gemm_out_bufs,
        rs_stream=rs_stream,
        num_gemm_sms=num_sms - rs_ctx.base_ctx.num_rs_sms,
    )
    nvshmem_barrier_all_on_stream(torch.cuda.current_stream())
    return ctx


def new_3rd_key_fn(A, B, ctx: New3rdGEMMReduceScatterTensorParallelContext, *args, **kwargs):
    return (
        triton_dist.tune.to_hashable(A),
        triton_dist.tune.to_hashable(B),
        ctx.rs_ctx.world_size,
        ctx.rs_ctx.local_world_size,
        ctx.rs_ctx.chunk_rows,
        ctx.rs_ctx.stage_slots,
        int(ctx.rs_ctx.use_scratch),
        kwargs.get("persistent", True),
    )


def new_3rd_prune_fn(config, A, B, *args, **kwargs):
    itemsize = A.itemsize
    gemm_config = config["gemm_config"].all_kwargs()
    num_stages = gemm_config["num_stages"]
    block_size_m = gemm_config["BLOCK_SIZE_M"]
    block_size_n = gemm_config["BLOCK_SIZE_N"]
    block_size_k = gemm_config["BLOCK_SIZE_K"]
    shared_memory = (itemsize * block_size_m * block_size_k + itemsize * block_size_n * block_size_k) * num_stages

    M, _ = A.shape
    _, N = B.shape
    tiled_m = triton.cdiv(M, block_size_m)
    tiled_n = triton.cdiv(N, block_size_n)
    tiles = tiled_m * tiled_n
    num_sms = torch.cuda.get_device_properties(0).multi_processor_count
    persistent = kwargs.get("persistent", True)

    if torch.cuda.get_device_capability()[0] >= 9:
        ntiles_per_sm = tiles // num_sms
        if ntiles_per_sm < 4 and persistent:
            return False
        if ntiles_per_sm > 10 and not persistent:
            return False

    return shared_memory < get_device_max_shared_memory_size(0)


def get_new_3rd_gemm_rs_config_space():
    # The chunk-ready producer is currently implemented only for non-persistent GEMM.
    return [{"gemm_config": c} for c in get_config_space(False)]


def new_3rd_gemm_rs_op(
    A: torch.Tensor,
    B: torch.Tensor,
    ctx: New3rdGEMMReduceScatterTensorParallelContext,
    gemm_config: triton.Config,
    persistent: bool = True,
) -> torch.Tensor:
    """Run chunk-ready GEMM producer plus elastic signal-driven third-stream RS."""
    if ctx.rs_ctx.nnodes != 1:
        raise NotImplementedError("new_3rd_gemm_rs currently focuses on single-node only")
    if not has_fullmesh_nvlink():
        raise NotImplementedError("new_3rd_gemm_rs currently expects full-mesh NVLink")
    if persistent:
        raise NotImplementedError("new_3rd_gemm_rs currently supports only --no-persistent")

    world_size = ctx.rs_ctx.world_size
    local_world_size = ctx.rs_ctx.local_world_size
    current_stream = torch.cuda.current_stream()
    ctx.rs_stream.wait_stream(current_stream)

    M, local_K = A.shape
    _, N = B.shape
    assert B.shape == (local_K, ctx.rs_ctx.base_ctx.N), (
        f"B should be of shape [{local_K}, {ctx.rs_ctx.base_ctx.N}], but got {list(B.shape)}"
    )
    assert M % world_size == 0, "M must be divisible by world_size"

    output = torch.empty((M // world_size, N), dtype=ctx.output_dtype, device=A.device)
    workspace = torch.zeros((world_size * ctx.rs_ctx.num_chunks,), dtype=torch.int32, device=A.device)
    gemm_out = ctx.get_gemm_out_buf(A)
    signal_value = ctx.rs_ctx.begin_round()
    _debug_log(
        ctx,
        f"new_3rd_gemm_rs_op enter: M={M}, N={N}, local_K={local_K}, "
        f"chunk_rows={ctx.rs_ctx.chunk_rows}, num_chunks={ctx.rs_ctx.num_chunks}, "
        f"stage_slots={ctx.rs_ctx.stage_slots}, signal_value={signal_value}",
    )

    gemm_config = update_triton_config(M, N, local_K, A.dtype, world_size, local_world_size, gemm_config)
    if ctx.rs_ctx.chunk_rows < gemm_config.kwargs["BLOCK_SIZE_M"]:
        raise ValueError(
            "new_3rd_gemm_rs requires chunk_rows >= BLOCK_SIZE_M, "
            f"but got chunk_rows={ctx.rs_ctx.chunk_rows}, BLOCK_SIZE_M={gemm_config.kwargs['BLOCK_SIZE_M']}"
        )

    _debug_log(ctx, "launching chunk-ready non-persistent GEMM producer")
    gemm_rs_producer_non_persistent_chunked(
        A,
        B,
        gemm_out,
        ctx.rs_ctx.chunk_signal,
        workspace,
        world_size,
        local_world_size,
        ctx.rs_ctx.num_chunks,
        ctx.rs_ctx.chunk_rows,
        signal_value,
        gemm_config,
    )
    gemm_done_event = torch.cuda.Event()
    gemm_done_event.record(current_stream)
    _debug_log(ctx, "GEMM producer launched; entering elastic new_3rd reduce-scatter stage")

    new_3rd_reduce_scatter_2d_op(gemm_out, ctx.rs_ctx, output, gemm_done_event=gemm_done_event)
    _debug_log(ctx, "elastic new_3rd reduce-scatter launched; waiting slot streams")
    _debug_log(ctx, "new_3rd_gemm_rs_op exit")
    return output


@triton_dist.tune.autotune(
    config_space=get_new_3rd_gemm_rs_config_space(),
    key_fn=new_3rd_key_fn,
    prune_fn=new_3rd_prune_fn,
)
def new_3rd_gemm_rs(
    A: torch.Tensor,
    B: torch.Tensor,
    ctx: New3rdGEMMReduceScatterTensorParallelContext,
    gemm_config: triton.Config,
    persistent: bool = True,
):
    return new_3rd_gemm_rs_op(A, B, ctx, gemm_config, persistent)
