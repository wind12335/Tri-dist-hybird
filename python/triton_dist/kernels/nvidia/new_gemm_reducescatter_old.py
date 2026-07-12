from __future__ import annotations

import dataclasses
from typing import Optional

import torch
import triton
import triton.language as tl
import triton_dist
import triton_dist.language as dl
import triton_dist.tune
from triton_dist.language.extra.language_extra import __syncthreads, atomic_add, st, tid

from triton_dist.kernels.nvidia.gemm import get_config_space
from triton_dist.kernels.nvidia.gemm_reduce_scatter import _matmul_launch_metadata, update_triton_config
from triton_dist.kernels.nvidia.gemm_rs_threadblock_swizzle import threadblock_swizzle_gemm_reduce_scatter_kernel
from triton_dist.kernels.nvidia.new_reducescatter import NewReduceScatterContext, create_new_reducescatter_2d_ctx, \
    new_reduce_scatter_2d_op
from triton_dist.utils import get_device_max_shared_memory_size, nvshmem_barrier_all_on_stream, nvshmem_create_tensors, \
    nvshmem_free_tensor_sync


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
    LOCAL_WORLD_SIZE: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    NUM_CHUNKS: tl.constexpr,
    CHUNK_ROWS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    tl.static_assert(a_ptr.dtype.is_ptr())
    tl.static_assert(b_ptr.dtype.is_ptr())
    tl.static_assert(c_ptr.dtype.is_ptr())
    a_dtype = a_ptr.dtype.element_ty
    b_dtype = b_ptr.dtype.element_ty
    tl.static_assert(a_dtype == b_dtype)

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
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    out_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=out_mask)

    segment_start = pid_m * BLOCK_SIZE_M // M_per_rank
    segment_end = (min((pid_m + 1) * BLOCK_SIZE_M, M) - 1) // M_per_rank
    __syncthreads()
    segment = segment_start + tid(axis=0)
    if segment <= segment_end:
        tile_row_start = pid_m * BLOCK_SIZE_M
        tile_row_end = min((pid_m + 1) * BLOCK_SIZE_M, M) - 1
        seg_row_start = segment * M_per_rank
        local_tile_start = tl.maximum(tile_row_start, seg_row_start) - seg_row_start
        local_tile_end = tl.minimum(tile_row_end, seg_row_start + M_per_rank - 1) - seg_row_start
        chunk_start = local_tile_start // CHUNK_ROWS
        chunk_end = local_tile_end // CHUNK_ROWS
        for chunk_id in range(chunk_start, chunk_end + 1):
            chunk_row_start = segment * M_per_rank + chunk_id * CHUNK_ROWS
            chunk_row_end = tl.minimum(chunk_row_start + CHUNK_ROWS, (segment + 1) * M_per_rank) - 1
            tiled_m_start = chunk_row_start // BLOCK_SIZE_M
            tiled_m_end = chunk_row_end // BLOCK_SIZE_M
            tiled_m_size = tiled_m_end - tiled_m_start + 1
            signal_idx = segment * NUM_CHUNKS + chunk_id
            val = atomic_add(counter_ptr + signal_idx, 1, semantic="release", scope="gpu")
            if val == num_pid_n * tiled_m_size - 1:
                st(chunk_signal_ptr + signal_idx, 1, scope="gpu", semantic="release")


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
    gemm_config: triton.Config,
):
    assert A.shape[1] == B.shape[0]
    assert A.dtype == B.dtype
    M, K_per_rank = A.shape
    _, N = B.shape
    BLOCK_SIZE_M = gemm_config.kwargs["BLOCK_SIZE_M"]
    BLOCK_SIZE_N = gemm_config.kwargs["BLOCK_SIZE_N"]
    grid = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N),)
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
        local_world_size,
        world_size,
        NUM_CHUNKS=num_chunks,
        CHUNK_ROWS=chunk_rows,
        **gemm_config.all_kwargs(),
    )


@dataclasses.dataclass
class NewGEMMReduceScatterTensorParallelContext:
    rs_ctx: NewReduceScatterContext
    output_dtype: torch.dtype
    gemm_out_bufs: list[torch.Tensor]
    rs_stream: torch.cuda.Stream
    num_gemm_sms: int

    def finalize(self):
        self.rs_ctx.finalize()
        nvshmem_free_tensor_sync(self.gemm_out_bufs[self.rs_ctx.local_rank])

    def get_gemm_out_buf(self, input: torch.Tensor):
        M, _ = input.shape
        return self.gemm_out_bufs[self.rs_ctx.local_rank][:M]


def create_new_gemm_rs_context(
    max_M: int,
    N: int,
    rank: int,
    world_size: int,
    local_world_size: int,
    output_dtype: torch.dtype,
    rs_stream: torch.cuda.Stream,
    *,
    chunk_rows: int = 0,
    target_chunks_per_rank: int = 4,
    min_chunk_rows: int = 256,
) -> NewGEMMReduceScatterTensorParallelContext:
    rs_ctx = create_new_reducescatter_2d_ctx(
        max_M,
        N,
        rank,
        world_size,
        local_world_size,
        output_dtype,
        chunk_rows=chunk_rows,
        target_chunks_per_rank=target_chunks_per_rank,
        min_chunk_rows=min_chunk_rows,
    )
    num_sms = torch.cuda.get_device_properties("cuda").multi_processor_count
    gemm_out_bufs = nvshmem_create_tensors((max_M, N), output_dtype, rank, local_world_size)
    ctx = NewGEMMReduceScatterTensorParallelContext(
        rs_ctx=rs_ctx,
        output_dtype=output_dtype,
        gemm_out_bufs=gemm_out_bufs,
        rs_stream=rs_stream,
        num_gemm_sms=num_sms - rs_ctx.base_ctx.num_rs_sms,
    )
    nvshmem_barrier_all_on_stream(torch.cuda.current_stream())
    return ctx


def new_key_fn(A, B, ctx: NewGEMMReduceScatterTensorParallelContext, *args, **kwargs):
    return (
        triton_dist.tune.to_hashable(A),
        triton_dist.tune.to_hashable(B),
        ctx.rs_ctx.world_size,
        ctx.rs_ctx.local_world_size,
        ctx.rs_ctx.chunk_rows,
    )


def new_prune_fn(config, A, B, *args, **kwargs):
    itemsize = A.itemsize
    gemm_config = config["gemm_config"].all_kwargs()
    num_stages = gemm_config["num_stages"]
    BLOCK_SIZE_M = gemm_config["BLOCK_SIZE_M"]
    BLOCK_SIZE_N = gemm_config["BLOCK_SIZE_N"]
    BLOCK_SIZE_K = gemm_config["BLOCK_SIZE_K"]
    shared_memory = (itemsize * BLOCK_SIZE_M * BLOCK_SIZE_K + itemsize * BLOCK_SIZE_N * BLOCK_SIZE_K) * num_stages
    M, _ = A.shape
    _, N = B.shape
    tiled_m = (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    tiled_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    tiles = tiled_m * tiled_n
    num_sms = torch.cuda.get_device_properties(0).multi_processor_count
    if tiles < num_sms // 2 and gemm_config["GROUP_SIZE_M"] > 4:
        return False
    return shared_memory < get_device_max_shared_memory_size(0)


def get_new_gemm_rs_config_space():
    return [{"gemm_config": c} for c in get_config_space(False)]


def new_gemm_rs_op(A: torch.Tensor, B: torch.Tensor, ctx: NewGEMMReduceScatterTensorParallelContext,
                   gemm_config: triton.Config):
    if ctx.rs_ctx.nnodes != 1:
        raise NotImplementedError("new_gemm_rs currently focuses on intra-node only")

    world_size = ctx.rs_ctx.world_size
    local_world_size = ctx.rs_ctx.local_world_size
    rs_stream = ctx.rs_stream
    output_dtype = ctx.output_dtype
    M, local_K = A.shape
    _, N = B.shape
    assert B.shape == (local_K, ctx.rs_ctx.base_ctx.N)
    assert M % world_size == 0

    current_stream = torch.cuda.current_stream()
    rs_stream.wait_stream(current_stream)
    output = torch.empty((M // world_size, N), dtype=output_dtype, device=A.device)
    workspace = torch.zeros((world_size * ctx.rs_ctx.num_chunks,), dtype=torch.int32, device=A.device)
    gemm_out = ctx.get_gemm_out_buf(A)
    ctx.rs_ctx.reset_runtime_state()

    gemm_config = update_triton_config(M, N, local_K, A.dtype, world_size, local_world_size, gemm_config)
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
        gemm_config,
    )

    with torch.cuda.stream(rs_stream):
        new_reduce_scatter_2d_op(gemm_out, ctx.rs_ctx, output)
    current_stream.wait_stream(rs_stream)
    return output


@triton_dist.tune.autotune(
    config_space=get_new_gemm_rs_config_space(),
    key_fn=new_key_fn,
    prune_fn=new_prune_fn,
)
def new_gemm_rs(A: torch.Tensor, B: torch.Tensor, ctx: NewGEMMReduceScatterTensorParallelContext,
                gemm_config: triton.Config):
    return new_gemm_rs_op(A, B, ctx, gemm_config)
