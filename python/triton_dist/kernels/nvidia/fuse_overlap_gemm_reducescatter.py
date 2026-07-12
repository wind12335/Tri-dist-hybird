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

import torch
import triton
import triton.language as tl
import triton_dist
import triton_dist.language as dl
import triton_dist.tune
from triton_dist.language.extra.language_extra import __syncthreads, atomic_add, st, tid

from triton_dist.kernels.nvidia.fuse_overlap_reducescatter import (
    FuseOverlapReduceScatterContext,
    create_fuse_overlap_reducescatter_2d_ctx,
    fuse_overlap_reduce_scatter_2d_op,
)
from triton_dist.kernels.nvidia.gemm import get_config_space
from triton_dist.kernels.nvidia.gemm_reduce_scatter import _matmul_launch_metadata, swizzle_2d, update_triton_config
from triton_dist.kernels.nvidia.gemm_rs_threadblock_swizzle import threadblock_swizzle_gemm_reduce_scatter_kernel
from triton_dist.utils import (get_device_max_shared_memory_size, has_fullmesh_nvlink, nvshmem_barrier_all_on_stream,
                               nvshmem_create_tensors, nvshmem_free_tensor_sync)


@triton_dist.jit(launch_metadata=_matmul_launch_metadata)
def kernel_fuse_overlap_gemm_rs_producer_non_persistent(
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
    ready_ptr,
    counter_ptr,
    LOCAL_WORLD_SIZE: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """Single-node fused-scatter producer with per-destination ready signaling.

    Compared with the upstream fused path, this kernel keeps the direct remote
    layout store, but also tracks when each destination segment becomes complete.
    Once a destination segment is fully produced, it raises a source-owned ready
    flag that the destination rank can wait on from a side stream.
    """
    tl.static_assert(a_ptr.dtype.is_ptr(), "A should be a pointer")
    tl.static_assert(b_ptr.dtype.is_ptr(), "B should be a pointer")
    tl.static_assert(c_ptr.dtype.is_ptr(), "C should be a pointer")
    a_dtype = a_ptr.dtype.element_ty
    b_dtype = b_ptr.dtype.element_ty
    c_dtype = c_ptr.dtype.element_ty
    tl.static_assert(a_dtype == b_dtype, "A and B should have the same dtype")

    rank = dl.rank()
    local_rank = rank % LOCAL_WORLD_SIZE
    nnodes = WORLD_SIZE // LOCAL_WORLD_SIZE
    tl.static_assert(WORLD_SIZE == LOCAL_WORLD_SIZE, "fuse_overlap path currently assumes single-node only")

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    M_per_rank = M // WORLD_SIZE

    pid_m, pid_n = swizzle_2d(pid, num_pid_m, num_pid_n, GROUP_SIZE_M)
    if nnodes != 1:
        pid_m = threadblock_swizzle_gemm_reduce_scatter_kernel(pid_m, M, rank, WORLD_SIZE, nnodes, BLOCK_SIZE_M)
    else:
        pid_m_offset = (rank + 1) * M_per_rank // BLOCK_SIZE_M
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
    rank_start = pid_m * BLOCK_SIZE_M // M_per_rank
    rank_end = (min((pid_m + 1) * BLOCK_SIZE_M, M) - 1) // M_per_rank

    for cur_rank in range(rank_start, rank_end + 1):
        m_start = max(M_per_rank * cur_rank, pid_m * BLOCK_SIZE_M)
        m_end = min(M_per_rank * (cur_rank + 1) - 1, min((pid_m + 1) * BLOCK_SIZE_M, M) - 1)
        remote_c_ptr = dl.symm_at(c_ptr, cur_rank)
        mask_offset = m_start - pid_m * BLOCK_SIZE_M
        remote_offs_cm = m_start % M_per_rank + rank * M_per_rank + tl.arange(0, BLOCK_SIZE_M) - mask_offset
        remote_c_ptrs = remote_c_ptr + stride_cm * remote_offs_cm[:, None] + stride_cn * offs_cn[None, :]
        remote_mask = (offs_cm[:, None] >= m_start) & (offs_cm[:, None] <= m_end) & (offs_cn[None, :] < N)
        tl.store(remote_c_ptrs, accumulator.to(c_dtype), mask=remote_mask)

    # Publish per-destination ready only after the whole destination segment is done.
    __syncthreads()
    segment = rank_start + tid(axis=0)
    if segment <= rank_end:
        seg_m_start = M_per_rank * segment
        seg_m_end = M_per_rank * (segment + 1) - 1
        tiled_m_start = seg_m_start // BLOCK_SIZE_M
        tiled_m_end = seg_m_end // BLOCK_SIZE_M
        tiled_m_size = tiled_m_end - tiled_m_start + 1
        val = atomic_add(counter_ptr + segment, 1, semantic="release", scope="gpu")
        if val == num_pid_n * tiled_m_size - 1:
            # Write the ready bit into the destination rank's local ready vector.
            remote_ready_ptr = dl.symm_at(ready_ptr, segment)
            st(remote_ready_ptr + local_rank, 1, semantic="release", scope="sys")


def fuse_overlap_gemm_rs_producer_non_persistent(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    ready: torch.Tensor,
    workspace: torch.Tensor,
    world_size: int,
    local_world_size: int,
    gemm_config: triton.Config,
) -> None:
    assert A.shape[1] == B.shape[0], "Incompatible dimensions"
    assert A.dtype == B.dtype, "Incompatible dtypes"

    M, K_per_rank = A.shape
    _, N = B.shape
    block_size_m = gemm_config.kwargs["BLOCK_SIZE_M"]
    block_size_n = gemm_config.kwargs["BLOCK_SIZE_N"]
    grid = (triton.cdiv(M, block_size_m) * triton.cdiv(N, block_size_n), )

    kernel_fuse_overlap_gemm_rs_producer_non_persistent[grid](
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
        ready,
        workspace,
        local_world_size,
        world_size,
        **gemm_config.all_kwargs(),
    )


@dataclasses.dataclass
class FuseOverlapGEMMReduceScatterTensorParallelContext:
    rs_ctx: FuseOverlapReduceScatterContext
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


def create_fuse_overlap_gemm_rs_context(
    max_M: int,
    N: int,
    rank: int,
    world_size: int,
    local_world_size: int,
    output_dtype: torch.dtype,
    rs_stream: torch.cuda.Stream,
) -> FuseOverlapGEMMReduceScatterTensorParallelContext:
    rs_ctx = create_fuse_overlap_reducescatter_2d_ctx(max_M, N, rank, world_size, local_world_size, output_dtype)
    num_sms = torch.cuda.get_device_properties("cuda").multi_processor_count
    gemm_out_bufs = nvshmem_create_tensors((max_M, N), output_dtype, rank, local_world_size)
    ctx = FuseOverlapGEMMReduceScatterTensorParallelContext(
        rs_ctx=rs_ctx,
        output_dtype=output_dtype,
        gemm_out_bufs=gemm_out_bufs,
        rs_stream=rs_stream,
        num_gemm_sms=num_sms - rs_ctx.base_ctx.num_rs_sms,
    )
    nvshmem_barrier_all_on_stream(torch.cuda.current_stream())
    return ctx


def fuse_overlap_key_fn(A, B, ctx: FuseOverlapGEMMReduceScatterTensorParallelContext, *args, **kwargs):
    return (
        triton_dist.tune.to_hashable(A),
        triton_dist.tune.to_hashable(B),
        ctx.rs_ctx.world_size,
        ctx.rs_ctx.local_world_size,
    )


def fuse_overlap_prune_fn(config, A, B, *args, **kwargs):
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
    if tiles < max(1, num_sms // 2) and gemm_config["GROUP_SIZE_M"] > 4:
        return False
    return shared_memory < get_device_max_shared_memory_size(0)


def get_fuse_overlap_gemm_rs_config_space():
    return [{"gemm_config": c} for c in get_config_space(False)]


def fuse_overlap_gemm_rs_op(
    A: torch.Tensor,
    B: torch.Tensor,
    ctx: FuseOverlapGEMMReduceScatterTensorParallelContext,
    gemm_config: triton.Config,
) -> torch.Tensor:
    if ctx.rs_ctx.nnodes != 1:
        raise NotImplementedError("fuse_overlap_gemm_rs currently focuses on single-node only")
    if not has_fullmesh_nvlink():
        raise NotImplementedError("fuse_overlap_gemm_rs currently expects full-mesh NVLink")

    world_size = ctx.rs_ctx.world_size
    local_world_size = ctx.rs_ctx.local_world_size
    M, local_K = A.shape
    _, N = B.shape
    if B.shape != (local_K, N):
        raise ValueError(f"B should be of shape [{local_K}, {N}], but got {list(B.shape)}")
    if M % world_size != 0:
        raise ValueError(f"M must be divisible by world_size, got M={M}, world_size={world_size}")

    gemm_config = update_triton_config(M, N, local_K, A.dtype, world_size, local_world_size, gemm_config)

    current_stream = torch.cuda.current_stream()
    nvshmem_barrier_all_on_stream(current_stream)
    ctx.rs_ctx.reset_runtime_state()
    ctx.rs_stream.wait_stream(current_stream)

    output = torch.empty((M // world_size, N), dtype=ctx.output_dtype, device=A.device)
    workspace = torch.zeros((local_world_size, ), dtype=torch.int32, device=A.device)
    gemm_out = ctx.get_gemm_out_buf(A)

    fuse_overlap_gemm_rs_producer_non_persistent(
        A,
        B,
        gemm_out,
        ctx.rs_ctx.ready_buf,
        workspace,
        world_size,
        local_world_size,
        gemm_config,
    )
    return fuse_overlap_reduce_scatter_2d_op(gemm_out, ctx.rs_ctx, output)


@triton_dist.tune.autotune(
    config_space=get_fuse_overlap_gemm_rs_config_space(),
    key_fn=fuse_overlap_key_fn,
    prune_fn=fuse_overlap_prune_fn,
)
def fuse_overlap_gemm_rs(
    A: torch.Tensor,
    B: torch.Tensor,
    ctx: FuseOverlapGEMMReduceScatterTensorParallelContext,
    gemm_config: triton.Config,
):
    return fuse_overlap_gemm_rs_op(A, B, ctx, gemm_config)
