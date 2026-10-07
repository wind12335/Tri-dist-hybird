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

from dataclasses import dataclass
from typing import Optional

import torch

import triton
import triton.language as tl
import triton_dist
import triton_dist.language as dl
import triton_dist.tune

from triton_dist.kernels.nvidia.allgather import AllGatherMethod, get_auto_all_gather_method
from triton_dist.kernels.nvidia.ag_gemm_threadblock_swizzle import threadblock_swizzle_allgather_gemm_kernel
from triton_dist.kernels.nvidia.allgather_gemm import (AllGatherGEMMTensorParallelContext, _matmul_launch_metadata,
                                                       ag_gemm_config_space, create_ag_gemm_context, key_fn,
                                                       prune_fn, swizzle_2d)
from triton_dist.kernels.nvidia.gemm import get_config_space, matmul_kernel, prune_fn as gemm_prune_fn
from triton_dist.kernels.nvidia.new_allgather_tileready import launch_new_allgather_intra_node
from triton_dist.utils import nvshmem_create_tensors, nvshmem_free_tensor_sync


@triton_dist.jit(launch_metadata=_matmul_launch_metadata)
def kernel_consumer_gemm_persistent_skip_local(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    rank: tl.constexpr,
    num_ranks: tl.constexpr,
    ready_ptr,
    skip_rank,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    EPILOGUE_SUBTILE: tl.constexpr,
    NUM_SMS: tl.constexpr,
    ready_value: tl.constexpr = 1,
    LOCAL_WORLD_SIZE: tl.constexpr = 8,
):
    dtype = c_ptr.dtype.element_ty
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n
    nnodes = num_ranks // LOCAL_WORLD_SIZE

    a_desc = tl.make_tensor_descriptor(
        a_ptr,
        shape=[M, K],
        strides=[K, 1],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K],
    )
    b_desc = tl.make_tensor_descriptor(
        b_ptr,
        shape=[N, K],
        strides=[K, 1],
        block_shape=[BLOCK_SIZE_N, BLOCK_SIZE_K],
    )
    c_desc = tl.make_tensor_descriptor(
        c_ptr,
        shape=[M, N],
        strides=[N, 1],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N if not EPILOGUE_SUBTILE else BLOCK_SIZE_N // 2],
    )

    tiles_per_sm = num_tiles // NUM_SMS
    if start_pid < num_tiles % NUM_SMS:
        tiles_per_sm += 1

    tile_id = start_pid - NUM_SMS
    ki = -1
    pid_m = 0
    pid_n = 0
    offs_am = 0
    offs_bn = 0
    tile_rank_beg = 0
    tile_rank_end = 0

    M_per_rank = M // num_ranks
    pid_ms_per_rank = tl.cdiv(M_per_rank, BLOCK_SIZE_M)
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for _ in range(0, k_tiles * tiles_per_sm):
        ki = tl.where(ki == k_tiles - 1, 0, ki + 1)
        if ki == 0:
            tile_id += NUM_SMS
            pid_m, pid_n = swizzle_2d(tile_id, num_pid_m, num_pid_n, GROUP_SIZE_M)
            if nnodes == 1:
                pid_m = (pid_m + (rank % num_ranks) * pid_ms_per_rank) % num_pid_m
            else:
                pid_m = threadblock_swizzle_allgather_gemm_kernel(pid_m, M, rank, num_ranks, nnodes, BLOCK_SIZE_M)

            offs_am = pid_m * BLOCK_SIZE_M
            offs_bn = pid_n * BLOCK_SIZE_N
            tile_rank_beg = offs_am // M_per_rank
            tile_rank_end = (min(offs_am + BLOCK_SIZE_M, M) - 1) // M_per_rank

            # If this tile lies entirely inside the local shard, we can skip
            # the whole compute (not only skipping store). This is the key to
            # realizing the expected ~ (world_size-1)/world_size speedup.
            skip_tile = (tile_rank_beg == skip_rank) & (tile_rank_end == skip_rank)
            if not skip_tile:
                token = dl.wait(ready_ptr + tile_rank_beg, tile_rank_end - tile_rank_beg + 1, "gpu", "acquire",
                                waitValue=ready_value)
                a_desc = dl.consume_token(a_desc, token)

        offs_k = ki * BLOCK_SIZE_K
        # In persistent mode, we cannot `return` early (each CTA processes many
        # tiles). Instead, we branch per-tile and skip dot+store for the local
        # shard tiles.
        if not skip_tile:
            a = a_desc.load([offs_am, offs_k])
            b = b_desc.load([offs_bn, offs_k])
            accumulator = tl.dot(a, b.T, accumulator)

        if ki == k_tiles - 1:
            if not skip_tile:
                offs_cm = offs_am + tl.arange(0, BLOCK_SIZE_M)
                row_rank = offs_cm // M_per_rank
                valid_rows = (offs_cm < M) & (row_rank != skip_rank)

                if EPILOGUE_SUBTILE:
                    acc = tl.reshape(accumulator, (BLOCK_SIZE_M, 2, BLOCK_SIZE_N // 2))
                    acc = tl.permute(acc, (0, 2, 1))
                    acc0, acc1 = tl.split(acc)
                    c0 = acc0.to(dtype)
                    c1 = acc1.to(dtype)
                    c_desc.store([offs_am, offs_bn], c0, mask=valid_rows[:, None])
                    c_desc.store([offs_am, offs_bn + BLOCK_SIZE_N // 2], c1, mask=valid_rows[:, None])
                else:
                    c = accumulator.to(dtype)
                    c_desc.store([offs_am, offs_bn], c, mask=valid_rows[:, None])

            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)


@triton_dist.jit(do_not_specialize=["rank", "skip_rank"], launch_metadata=_matmul_launch_metadata)
def kernel_consumer_gemm_non_persistent_skip_local(
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
    rank,
    skip_rank,
    WORLD_SIZE: tl.constexpr,
    barrier_ptr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    a_dtype = a_ptr.dtype.element_ty
    b_dtype = b_ptr.dtype.element_ty
    c_dtype = c_ptr.dtype.element_ty
    tl.static_assert(a_dtype == b_dtype, "A and B must have the same dtype")

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    m_per_rank = M // WORLD_SIZE
    pid_m_offset = tl.cdiv(m_per_rank * rank, BLOCK_SIZE_M)
    pid_m = (pid_m + pid_m_offset) % num_pid_m

    offs_am_tile = pid_m * BLOCK_SIZE_M
    rank_beg = offs_am_tile // m_per_rank
    rank_end = (min(offs_am_tile + BLOCK_SIZE_M, M) - 1) // m_per_rank
    # If this tile belongs fully to the local shard, skip the whole program.
    # This avoids both the barrier wait and the dot() FLOPs.
    if (rank_beg == skip_rank) & (rank_end == skip_rank):
        return
    token = dl.wait(barrier_ptr + rank_beg, rank_end - rank_beg + 1, "gpu", "acquire", waitValue=1)

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    a_ptrs = dl.consume_token(a_ptrs, token)

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
    row_rank = offs_cm // m_per_rank
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N) & (row_rank[:, None] != skip_rank)
    tl.store(c_ptrs, accumulator.to(c_dtype), mask=c_mask)


@triton_dist.jit(do_not_specialize=["rank", "skip_rank"], launch_metadata=_matmul_launch_metadata)
def kernel_consumer_gemm_non_persistent_skip_local_tile_barrier(
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
    rank,
    skip_rank,
    WORLD_SIZE: tl.constexpr,
    rank_barrier_ptr,
    tile_barrier_ptr,
    NUM_TILE_CHUNKS: tl.constexpr,
    TILE_ROWS_PER_CHUNK: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    a_dtype = a_ptr.dtype.element_ty
    b_dtype = b_ptr.dtype.element_ty
    c_dtype = c_ptr.dtype.element_ty
    tl.static_assert(a_dtype == b_dtype, "A and B must have the same dtype")

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    m_per_rank = M // WORLD_SIZE
    pid_m_offset = tl.cdiv(m_per_rank * rank, BLOCK_SIZE_M)
    pid_m = (pid_m + pid_m_offset) % num_pid_m

    offs_am_tile = pid_m * BLOCK_SIZE_M
    rank_beg = offs_am_tile // m_per_rank
    rank_end = (min(offs_am_tile + BLOCK_SIZE_M, M) - 1) // m_per_rank
    if (rank_beg == skip_rank) & (rank_end == skip_rank):
        return

    if rank_beg == rank_end:
        rank_row_offset = offs_am_tile - rank_beg * m_per_rank
        chunk_beg = rank_row_offset // TILE_ROWS_PER_CHUNK
        tile_last_row = min(rank_row_offset + BLOCK_SIZE_M, m_per_rank) - 1
        chunk_end = tile_last_row // TILE_ROWS_PER_CHUNK
        tile_barrier_offset = rank_beg * NUM_TILE_CHUNKS + chunk_beg
        token = dl.wait(tile_barrier_ptr + tile_barrier_offset, chunk_end - chunk_beg + 1, "gpu", "acquire", waitValue=1)
    else:
        token = dl.wait(rank_barrier_ptr + rank_beg, rank_end - rank_beg + 1, "gpu", "acquire", waitValue=1)

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    a_ptrs = dl.consume_token(a_ptrs, token)

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
    row_rank = offs_cm // m_per_rank
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N) & (row_rank[:, None] != skip_rank)
    tl.store(c_ptrs, accumulator.to(c_dtype), mask=c_mask)


@dataclass
class NewAllGatherGEMMContext:
    base_ctx: AllGatherGEMMTensorParallelContext
    local_compute_stream: torch.cuda.Stream
    output_buf: torch.Tensor
    all_gather_method: AllGatherMethod
    persistent: bool
    use_row_tile_barrier: bool
    copy_sms: int = 0
    tile_rows_per_chunk: int = 256
    enable_row_tile_barrier: bool = True
    num_tile_chunks: int = 0
    symm_tile_barriers: list[torch.Tensor] | None = None
    symm_tile_barrier: torch.Tensor | None = None

    @property
    def rank(self):
        return self.base_ctx.rank

    @property
    def num_ranks(self):
        return self.base_ctx.num_ranks

    @property
    def num_local_ranks(self):
        return self.base_ctx.num_local_ranks

    def get_output_buf(self, M: int) -> torch.Tensor:
        return self.output_buf[:M]

    def finalize(self):
        if self.symm_tile_barrier is not None:
            nvshmem_free_tensor_sync(self.symm_tile_barrier)
        self.base_ctx.finalize()


def create_new_ag_gemm_context(
    max_M: int,
    N: int,
    K: int,
    dtype: torch.dtype,
    rank: int,
    num_ranks: int,
    num_local_ranks: int = 8,
    ag_intranode_stream: Optional[torch.cuda.Stream] = None,
    ag_internode_stream: Optional[torch.cuda.Stream] = None,
    local_compute_stream: Optional[torch.cuda.Stream] = None,
    copy_sms: int = 0,
) -> NewAllGatherGEMMContext:
    base_ctx = create_ag_gemm_context(
        max_M,
        N,
        K,
        dtype,
        rank,
        num_ranks,
        num_local_ranks=num_local_ranks,
        ag_intranode_stream=ag_intranode_stream,
        ag_internode_stream=ag_internode_stream,
    )
    tile_rows_per_chunk = 256
    max_m_per_rank = max_M // num_ranks
    num_tile_chunks = triton.cdiv(max_m_per_rank, tile_rows_per_chunk)
    barrier_dtype = torch.int32
    symm_tile_barriers = nvshmem_create_tensors((num_ranks * num_tile_chunks, ), barrier_dtype, rank, num_local_ranks)
    symm_tile_barrier = symm_tile_barriers[rank % num_local_ranks]
    symm_tile_barrier.zero_()
    method = base_ctx.all_gather_method
    if method == AllGatherMethod.Auto:
        method = get_auto_all_gather_method(base_ctx.num_ranks, base_ctx.num_local_ranks)
    persistent = torch.cuda.get_device_capability()[0] >= 9
    use_row_tile_barrier = (
        not persistent
        and not base_ctx.is_multinode
        and method == AllGatherMethod.All2All_IntraNode
        and symm_tile_barrier is not None
    )
    return NewAllGatherGEMMContext(
        base_ctx=base_ctx,
        local_compute_stream=local_compute_stream or torch.cuda.Stream(priority=0),
        output_buf=torch.empty((max_M, base_ctx.N_per_rank), dtype=dtype, device="cuda"),
        all_gather_method=method,
        persistent=persistent,
        use_row_tile_barrier=use_row_tile_barrier,
        copy_sms=copy_sms,
        tile_rows_per_chunk=tile_rows_per_chunk,
        enable_row_tile_barrier=True,
        num_tile_chunks=num_tile_chunks,
        symm_tile_barriers=symm_tile_barriers,
        symm_tile_barrier=symm_tile_barrier,
    )


def _local_gemm_key_fn(A: torch.Tensor, B: torch.Tensor, C_out: torch.Tensor, *args, **kwargs):
    return (triton_dist.tune.to_hashable(A), triton_dist.tune.to_hashable(B))


@triton_dist.tune.autotune(
    config_space=[{"local_gemm_config": config} for config in get_config_space(False)],
    key_fn=_local_gemm_key_fn,
    prune_fn=lambda config, A, B, C_out, *args, **kwargs: gemm_prune_fn({"config": config["local_gemm_config"]}, A, B),
)
def _local_matmul_into(
    A: torch.Tensor,
    B: torch.Tensor,
    C_out: torch.Tensor,
    local_gemm_config: triton.Config,
):
    M, K = A.shape
    K2, N = B.shape
    if K2 != K:
        raise ValueError(f"incompatible shapes: A={list(A.shape)}, B={list(B.shape)}")
    if C_out.shape != (M, N):
        raise ValueError(f"C_out must be {(M, N)}, got {tuple(C_out.shape)}")

    grid = lambda META: (triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]), )
    matmul_kernel[grid](
        A,
        B,
        C_out,
        M,
        N,
        K,
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(1),
        C_out.stride(0),
        C_out.stride(1),
        **local_gemm_config.all_kwargs(),
    )


def _run_local_shard_gemm_into(
    A: torch.Tensor,
    B: torch.Tensor,
    C_out: torch.Tensor,
    gemm_config: triton.Config,
    autotune: bool,
):
    # Use a dedicated local-GEMM autotune path rather than reusing the
    # allgather-consumer config. The local shard has a different shape regime
    # (smaller M), and reusing the consumer config can leave a couple percent
    # of performance on the table.
    _local_matmul_into(
        A,
        B,
        C_out,
        autotune=autotune,
    )


@triton_dist.tune.autotune(
    config_space=ag_gemm_config_space(),
    key_fn=key_fn,
    prune_fn=prune_fn,
)
def new_ag_gemm(
    A: torch.Tensor,
    B: torch.Tensor,
    ctx: NewAllGatherGEMMContext,
    gemm_config: triton.Config,
    debug: bool = False,
    use_cooperative: bool = False,
    autotune: bool = False,
):
    base = ctx.base_ctx
    if base.is_multinode:
        raise NotImplementedError("new_ag_gemm currently focuses on intra-node only")

    M_per_rank, K = A.shape
    _, N_per_rank = B.shape

    assert B.shape == (base.K, base.N_per_rank), f"B should be of shape [{base.K}, {base.N_per_rank}], but get [{B.shape}]"
    assert M_per_rank * base.num_ranks <= base.max_M and K == base.K, (
        f"Shape of A must not exceed ctx max_M: A shape [{A.shape}], ctx shape [{base.max_M}, {base.K}]"
    )
    assert base.dtype == A.dtype == B.dtype, f"dtype mismatch: A {A.dtype}, B {B.dtype}, ctx {base.dtype}"

    M = M_per_rank * base.num_ranks
    C = ctx.get_output_buf(M)

    # IMPORTANT: A/B are typically produced on the current stream. If we launch
    # the local GEMM on another stream without a dependency, it may read
    # uninitialized data and produce wrong results.
    current_stream = torch.cuda.current_stream()
    method = ctx.all_gather_method
    use_row_tile_barrier = ctx.use_row_tile_barrier

    ag_stream = launch_new_allgather_intra_node(
        A,
        base,
        debug=debug,
        use_cooperative=use_cooperative,
        copy_sms=ctx.copy_sms,
        all_gather_method=method,
        use_row_tile_barrier=use_row_tile_barrier,
        row_tile_barrier_buffers=ctx.symm_tile_barriers,
        num_tile_chunks=ctx.num_tile_chunks,
        tile_rows_per_chunk=ctx.tile_rows_per_chunk,
    )

    local_row_start = base.rank * M_per_rank
    local_row_end = local_row_start + M_per_rank
    ctx.local_compute_stream.wait_stream(current_stream)
    with torch.cuda.stream(ctx.local_compute_stream):
        C_local_out = C[local_row_start:local_row_end, :]
        _run_local_shard_gemm_into(A, B, C_local_out, gemm_config, autotune)

    if ctx.persistent:
        def alloc_fn(size: int, alignment: int, stream: Optional[int]):
            return torch.empty(size, device="cuda", dtype=torch.int8)

        triton.set_allocator(alloc_fn)
        gemm_sm = base.max_gemm_sm
        grid = lambda META: (min(
            gemm_sm,
            triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N_per_rank, META["BLOCK_SIZE_N"]),
        ), )
        kernel_consumer_gemm_persistent_skip_local[grid](
            base.symm_workspace[:M],
            B,
            C,
            M,
            N_per_rank,
            base.K,
            base.rank,
            base.num_ranks,
            base.symm_barrier,
            base.rank,
            NUM_SMS=gemm_sm,
            ready_value=base.barrier_target,
            LOCAL_WORLD_SIZE=base.num_local_ranks,
            **gemm_config.all_kwargs(),
        )
    else:
        grid = lambda META: (triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N_per_rank, META["BLOCK_SIZE_N"]), )
        if use_row_tile_barrier:
            kernel_consumer_gemm_non_persistent_skip_local_tile_barrier[grid](
                base.symm_workspace[:M],
                B,
                C,
                M,
                N_per_rank,
                base.K,
                base.symm_workspace.stride(0),
                base.symm_workspace.stride(1),
                B.stride(0),
                B.stride(1),
                C.stride(0),
                C.stride(1),
                base.rank,
                base.rank,
                base.num_ranks,
                base.symm_barrier,
                ctx.symm_tile_barrier,
                NUM_TILE_CHUNKS=ctx.num_tile_chunks,
                TILE_ROWS_PER_CHUNK=ctx.tile_rows_per_chunk,
                **gemm_config.all_kwargs(),
            )
        else:
            kernel_consumer_gemm_non_persistent_skip_local[grid](
                base.symm_workspace[:M],
                B,
                C,
                M,
                N_per_rank,
                base.K,
                base.symm_workspace.stride(0),
                base.symm_workspace.stride(1),
                B.stride(0),
                B.stride(1),
                C.stride(0),
                C.stride(1),
                base.rank,
                base.rank,
                base.num_ranks,
                base.symm_barrier,
                **gemm_config.all_kwargs(),
            )

    # IMPORTANT:
    # Do NOT wait for `local_compute_stream` before launching the consumer GEMM.
    # The consumer kernel does not write local rows (`skip_rank=rank`), so it can
    # overlap with the local GEMM and, more importantly, overlap its internal
    # `dl.wait(...)` stalls with useful local compute. We only need to ensure the
    # *caller* sees a fully written `C`, so we add the stream dependency here.
    current_stream.wait_stream(ctx.local_compute_stream)
    current_stream.wait_stream(ag_stream)
    return C
