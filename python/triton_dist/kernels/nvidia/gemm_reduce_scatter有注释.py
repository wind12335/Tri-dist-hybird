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
"""面向 tensor parallel（张量并行）工作负载的 GEMM + reduce-scatter 协同调度实现。

这个文件最重要的思想是 producer-consumer overlap（生产者-消费者重叠）：
- GEMM 是 producer（生产者）：负责计算本地部分输出矩阵的各个 tile
- reduce-scatter 是 consumer（消费者）：一旦 GEMM 把某个 rank 对应的 M 分段标记为 ready，
  就立刻开始通信与归约

对你当前的单机 A100 NVLink 环境来说：
- 最值得关注的重叠主要是 GEMM 和 intra-node reduce-scatter（节点内 reduce-scatter）的重叠
- 文件中仍保留了 multi-node threadblock swizzle（多节点线程块重排）和跨节点通信逻辑，
  因为它们有助于理解完整设计；但当 `world_size == local_world_size` 时，这些分支通常不是主路径
"""

import dataclasses
from typing import List, Optional

import torch

import triton
import triton_dist
import triton.language as tl
import triton_dist.language as dl
from triton_dist.language.extra.language_extra import (__syncthreads, atomic_add, tid, st)
from triton_dist.kernels.nvidia.reduce_scatter import (ReduceScatter2DContext, create_reduce_scater_2d_ctx,
                                                       reduce_scatter_2d_op, ring_reduce)
from triton_dist.kernels.nvidia.gemm_rs_threadblock_swizzle import threadblock_swizzle_gemm_reduce_scatter_kernel
from triton_dist.utils import has_tma, nvshmem_barrier_all_on_stream, nvshmem_create_tensors, nvshmem_free_tensor_sync, requires, get_device_max_shared_memory_size
import triton_dist.tune
from triton_dist.kernels.nvidia.gemm import get_config_space
from triton_dist.kernels.nvidia.comm_perf_model import (estimate_reduce_scatter_time_ms, get_nic_gbps_per_gpu)
from triton_dist.kernels.nvidia.gemm_perf_model import estimate_gemm_sol_time_ms
from triton_dist.nv_utils import get_intranode_max_speed_gbps


################### context ###################
@dataclasses.dataclass
class GEMMReduceScatterTensorParallelContext:
    """GEMM 与 reduce-scatter 共享的运行时状态。

    - `gemm_out_bufs`：symmetric producer buffer（对称生产者缓冲区），GEMM 把部分 C tile 写到这里
    - `rs_ctx`：通信/归约上下文，内部持有 signal buffer 和通信暂存区
    - `rs_stream`：执行 reduce-scatter 的 side stream（侧边流），让 GEMM 可以继续占用主流
    - `num_gemm_sms`：在为 reduce-scatter 预留辅助 SM 后，剩余可分给 GEMM 的 SM 数量
    """
    rs_ctx: ReduceScatter2DContext
    output_dtype: torch.dtype

    # gemm bufs (symm address)
    gemm_out_bufs: List[torch.Tensor]

    # stream
    rs_stream: torch.cuda.Stream

    # gemm kernel config
    num_gemm_sms: int

    def finalize(self):
        """同时释放 reduce-scatter context 和对称 GEMM 输出缓冲区。"""
        self.rs_ctx.finalize()
        nvshmem_free_tensor_sync(self.gemm_out_bufs[self.rs_ctx.local_rank])

    def get_gemm_out_buf(self, input):
        """返回当前 batch 对应的、长度为 M 的对称 GEMM 输出缓冲区前缀。"""
        M, _ = input.shape
        local_rank = self.rs_ctx.local_rank
        return self.gemm_out_bufs[local_rank][:M]


def create_gemm_rs_context(max_M, N, rank, world_size, local_world_size, output_dtype: torch.dtype,
                           rs_stream: torch.cuda.Stream) -> GEMMReduceScatterTensorParallelContext:
    """创建用于 GEMM + reduce-scatter 重叠执行的共享 context。

    这里有一套 SM budgeting（SM 预算分配）逻辑：
    - `rs_ctx.num_rs_sms` 表示为通信/归约阶段预留的 SM
    - 剩余的 SM 分配给 GEMM

    在单机环境下，`rs_ctx.num_rs_sms` 往往是 0，因为没有额外的跨节点 helper kernel 需要预留 SM，
    所以 GEMM 基本可以使用几乎全部设备 SM。
    """
    rs_ctx = create_reduce_scater_2d_ctx(max_M, N, rank, world_size, local_world_size, output_dtype)
    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
    num_gemm_sms = NUM_SMS - rs_ctx.num_rs_sms
    gemm_out_bufs = nvshmem_create_tensors((max_M, N), output_dtype, rank, local_world_size)
    ctx = GEMMReduceScatterTensorParallelContext(rs_ctx=rs_ctx, output_dtype=output_dtype, gemm_out_bufs=gemm_out_bufs,
                                                 rs_stream=rs_stream, num_gemm_sms=num_gemm_sms)
    nvshmem_barrier_all_on_stream(torch.cuda.current_stream())
    return ctx


@triton.jit
def swizzle_2d(tile_id, num_pid_m, num_pid_n, GROUP_SIZE_M: tl.constexpr):
    """标准的 Triton grouped-M swizzle（分组 M 维重排），作为 GEMM tile 的基础遍历顺序。"""
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (tile_id % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n


# TMA related test
def _matmul_launch_metadata(grid, kernel, args):
    """供 Triton profiling / tracing（性能分析/跟踪）使用的元数据辅助函数。"""
    ret = {}
    M, N, K = args["M"], args["N"], args["K"]
    ret["name"] = f"{kernel.name} [M={M}, N={N}, K={K}]"
    if "c_ptr" in args:
        bytes_per_elem = args["c_ptr"].element_size()
    else:
        bytes_per_elem = args["a_ptr"].element_size()
    ret[f"flops{bytes_per_elem * 8}"] = 2.0 * M * N * K
    ret["bytes"] = bytes_per_elem * (M * K + N * K + M * N)
    return ret


def _gemm_rs_persistent_repr(proxy):
    """为 persistent GEMM+RS producer（常驻式 GEMM+RS 生产者）生成可读的 kernel 名称。"""
    constexprs = proxy.constants
    cap_major, cap_minor = torch.cuda.get_device_capability()
    a_dtype = proxy.signature["a_ptr"].lstrip("*")
    b_dtype = proxy.signature["b_ptr"].lstrip("*")
    c_dtype = proxy.signature["c_ptr"].lstrip("*")
    BM, BN, BK = constexprs["BLOCK_SIZE_M"], constexprs["BLOCK_SIZE_N"], constexprs["BLOCK_SIZE_K"]
    fuse_scatter = constexprs["FUSE_SCATTER"]
    suffix = "_fuse_scatter" if fuse_scatter else ""

    return f"triton3x_sm{cap_major}{cap_minor}_gemm_rs_persistent_tensorop_{a_dtype}_{b_dtype}_{c_dtype}_{BM}x{BN}x{BK}_ntn{suffix}"


@triton_dist.jit(launch_metadata=_matmul_launch_metadata, repr=_gemm_rs_persistent_repr)
def kernel_gemm_rs_producer_persistent(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    barrier_ptr,
    counter_ptr,
    FUSE_SCATTER: tl.constexpr,
    LOCAL_WORLD_SIZE: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    EPILOGUE_SUBTILE: tl.constexpr,
    NUM_SMS: tl.constexpr,
):  #
    """为 reduce-scatter 提供输入的 persistent GEMM producer。

    它的核心职责有三件事：
    - 按有利于 reduce-scatter 消费的顺序计算 GEMM tile
    - 把完成的 tile 写入 symmetric output buffer（对称输出缓冲区）
    - 当某个 rank 拥有的整个 M segment 都完成后，发布 ready signal（就绪信号），使
      reduce-scatter 能立刻消费这个 segment

    如果 `FUSE_SCATTER=True`，这个 kernel 会跳过显式的 post-GEMM scatter（GEMM 后散播）阶段，
    直接把 tile 写到目标 rank 的对称缓冲区中。Python 包装层里，这种 fused 模式只允许单机使用。
    """
    # 使用 TMA 和 device-side descriptor creation（设备侧描述符构造）实现 matmul。
    rank = dl.rank()
    dtype = c_ptr.dtype.element_ty
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n
    NNODES: tl.constexpr = WORLD_SIZE // LOCAL_WORLD_SIZE

    a_desc = tl.make_tensor_descriptor(a_ptr, shape=[M, K], strides=[K, 1], block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K])
    b_desc = tl.make_tensor_descriptor(b_ptr, shape=[N, K], strides=[K, 1], block_shape=[BLOCK_SIZE_N, BLOCK_SIZE_K])
    c_desc = tl.make_tensor_descriptor(
        c_ptr, shape=[M, N], strides=[N, 1], block_shape=[
            BLOCK_SIZE_M,
            BLOCK_SIZE_N if not EPILOGUE_SUBTILE else BLOCK_SIZE_N // 2,
        ])

    # persistent scheduling（常驻调度）：每个 CTA 会反复领取新 tile，而不是一个 CTA 只算一个 tile。
    # 这样可以让 CTA 驻留更稳定，也更利于 GEMM 和通信并存。
    tiles_per_SM = num_tiles // NUM_SMS
    if start_pid < num_tiles % NUM_SMS:
        tiles_per_SM += 1

    tile_id = start_pid - NUM_SMS
    ki = -1

    pid_m = 0
    pid_n = 0
    offs_am = 0
    offs_bn = 0

    M_per_rank = M // WORLD_SIZE

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    pid_m_offset = (rank + 1) * M_per_rank // BLOCK_SIZE_M

    for _ in range(0, k_tiles * tiles_per_SM):
        ki = tl.where(ki == k_tiles - 1, 0, ki + 1)
        if ki == 0:
            tile_id += NUM_SMS
            pid_m, pid_n = swizzle_2d(tile_id, num_pid_m, num_pid_n, GROUP_SIZE_M)
            if NNODES != 1:  # 仅多节点使用：为了让后续 RS 顺序更友好，对 tile 做重新映射
                pid_m = threadblock_swizzle_gemm_reduce_scatter_kernel(pid_m, M, rank, WORLD_SIZE, NNODES, BLOCK_SIZE_M)
            else:
                # 单机快速路径：旋转 M 维 tile 顺序，让每个 rank 更早产出 peer 所需的 M segment，
                # 从而提升与 intra-node scatter 的重叠效果。
                pid_m = (pid_m + pid_m_offset) % num_pid_m

            offs_am = pid_m * BLOCK_SIZE_M
            offs_bn = pid_n * BLOCK_SIZE_N

        offs_k = ki * BLOCK_SIZE_K

        a = a_desc.load([offs_am, offs_k])
        b = b_desc.load([offs_bn, offs_k])
        accumulator = tl.dot(a, b.T, accumulator)

        if ki == k_tiles - 1:
            if not FUSE_SCATTER:
                if EPILOGUE_SUBTILE:
                    acc = tl.reshape(accumulator, (BLOCK_SIZE_M, 2, BLOCK_SIZE_N // 2))
                    acc = tl.permute(acc, (0, 2, 1))
                    acc0, acc1 = tl.split(acc)
                    c0 = acc0.to(dtype)
                    c_desc.store([offs_am, offs_bn], c0)
                    c1 = acc1.to(dtype)
                    c_desc.store([offs_am, offs_bn + BLOCK_SIZE_N // 2], c1)
                else:
                    c = accumulator.to(dtype)
                    c_desc.store([offs_am, offs_bn], c)

                # 一个 GEMM tile 可能会覆盖一个或多个 reduce-scatter segment。
                # 这里按 segment 统计 tile 完成数；只有当该 segment 的所有 N 方向 tile 全部完成时，
                # 才通知 reduce-scatter 可以消费。
                counter_start = offs_am // M_per_rank
                counter_end = (offs_am + BLOCK_SIZE_M - 1) // M_per_rank
                counter_end = min(counter_end, WORLD_SIZE - 1)
                for counter_id in range(counter_start, counter_end + 1):
                    m_start = M_per_rank * counter_id
                    m_end = M_per_rank * (counter_id + 1) - 1
                    tiled_m_start = m_start // BLOCK_SIZE_M
                    tiled_m_end = m_end // BLOCK_SIZE_M
                    tiled_m_size = tiled_m_end - tiled_m_start + 1
                    val = tl.atomic_add(counter_ptr + counter_id, 1, sem="release", scope="gpu")
                    if val == tiled_m_size * num_pid_n - 1:
                        # 这是给 `intra_node_scatter(...)` / ring-push reduce-scatter 消费的
                        # segment-ready notification（分段就绪通知）。
                        dl.notify(barrier_ptr + counter_id, rank, signal=1, comm_scope="gpu")
            else:
                # 单机 fused 模式：不再先写本地 GEMM buffer、后续再 scatter，
                # 而是直接把 tile 写到目标 rank 的对称缓冲区。
                rank_start = pid_m * BLOCK_SIZE_M // M_per_rank
                rank_end = (min((pid_m + 1) * BLOCK_SIZE_M, M) - 1) // M_per_rank
                offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
                offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
                for cur_rank in range(rank_start, rank_end + 1):
                    m_start = max(M_per_rank * cur_rank, pid_m * BLOCK_SIZE_M)
                    m_end = min(M_per_rank * (cur_rank + 1) - 1, (min((pid_m + 1) * BLOCK_SIZE_M, M) - 1))
                    remote_c_ptr = dl.symm_at(c_ptr, cur_rank)
                    mask_offset = m_start - pid_m * BLOCK_SIZE_M
                    remote_offs_cm = m_start % M_per_rank + rank * M_per_rank + tl.arange(0, BLOCK_SIZE_M) - mask_offset
                    remote_c_ptrs = remote_c_ptr + N * remote_offs_cm[:, None] + offs_cn[None, :]
                    remote_mask = (offs_cm[:, None] <= m_end) & (offs_cm[:, None] >= m_start) & (offs_cn[None, :] < N)
                    tl.store(remote_c_ptrs, accumulator, mask=remote_mask)

            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)


def _gemm_rs_non_persistent_repr(proxy):
    """为 non-persistent GEMM+RS producer（非常驻式 GEMM+RS 生产者）生成可读的 kernel 名称。"""
    constexprs = proxy.constants
    cap_major, cap_minor = torch.cuda.get_device_capability()
    a_dtype = proxy.signature["a_ptr"].lstrip("*")
    b_dtype = proxy.signature["b_ptr"].lstrip("*")
    c_dtype = proxy.signature["c_ptr"].lstrip("*")
    BM, BN, BK = constexprs["BLOCK_SIZE_M"], constexprs["BLOCK_SIZE_N"], constexprs["BLOCK_SIZE_K"]
    if constexprs.get("stride_am", None) == 1:  # column major => n
        a_trans = "n"
    elif constexprs.get("stride_ak", None) == 1:  # row-major => t
        a_trans = "t"
    else:
        raise Exception("both stride_am/stride_ak != 1")

    if constexprs.get("stride_bk", None) == 1:
        b_trans = "n"
    elif constexprs.get("stride_bn", None) == 1:
        b_trans = "t"
    else:
        raise Exception("both stride_am/stride_ak != 1")

    if constexprs.get("stride_cm", None) == 1:
        c_trans = "n"
    elif constexprs.get("stride_cn", None) == 1:
        c_trans = "t"
    else:
        raise Exception("both stride_am/stride_ak != 1")

    fuse_scatter = constexprs["FUSE_SCATTER"]
    suffix = "_fuse_scatter" if fuse_scatter else ""
    return f"triton3x_sm{cap_major}{cap_minor}_gemm_rs_tensorop_{a_dtype}_{b_dtype}_{c_dtype}_{BM}x{BN}x{BK}_{a_trans}{b_trans}{c_trans}{suffix}"


@triton_dist.jit(launch_metadata=_matmul_launch_metadata, repr=_gemm_rs_non_persistent_repr)
def kernel_gemm_rs_producer_non_persistent(
    # Pointers to matrices
    a_ptr,  # [M, K]_Ti
    b_ptr,  # [K, N]_Ti
    c_ptr,  # [M, N]_To
    # Matrix dimensions
    M,
    N,
    K,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
    # by to get the element one row down (A has M rows).
    stride_am,
    stride_ak,  #
    stride_bk,
    stride_bn,  #
    stride_cm,
    stride_cn,
    barrier_ptr,
    counter_ptr,
    FUSE_SCATTER: tl.constexpr,
    LOCAL_WORLD_SIZE: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,  #
    GROUP_SIZE_M: tl.constexpr,
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)

    相比 persistent 版本，这个 kernel 采用的是“每个输出 tile 启动一个 CTA”的方式。
    但它的重叠机制是一样的：
    - 先把 tile 写入 symmetric GEMM output buffer（对称 GEMM 输出缓冲区）
    - 再用原子操作统计每个 rank 所拥有的 M segment 已完成多少 tile
    - 当整个 segment 完成后，发布对应的 ready flag（就绪标志）
    """
    tl.static_assert(a_ptr.dtype.is_ptr(), "A should be a pointer")
    tl.static_assert(b_ptr.dtype.is_ptr(), "B should be a pointer")
    tl.static_assert(c_ptr.dtype.is_ptr(), "C should be a pointer")
    a_dtype = a_ptr.dtype.element_ty
    b_dtype = b_ptr.dtype.element_ty
    tl.static_assert(a_dtype == b_dtype, "A and B should have the same dtype")
    # IS_FP8 = tl.constexpr(a_dtype == tl.float8e4nv) or tl.constexpr(a_dtype == tl.float8e5)

    rank = dl.rank()
    NNODES = WORLD_SIZE // LOCAL_WORLD_SIZE

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    M_per_rank = M // WORLD_SIZE
    pid_m, pid_n = swizzle_2d(pid, num_pid_m, num_pid_n, GROUP_SIZE_M)

    if NNODES != 1:  # 仅多节点使用：为了更适合 reduce-scatter 消费顺序，对 tile 重新映射
        pid_m = threadblock_swizzle_gemm_reduce_scatter_kernel(pid_m, M, rank, WORLD_SIZE, NNODES, BLOCK_SIZE_M)
    else:
        # 单机快速路径：旋转 M 维 tile 顺序，让 ready segment 更早暴露给通信侧。
        pid_m_offset = (rank + 1) * M_per_rank // BLOCK_SIZE_M
        pid_m = (pid_m + pid_m_offset) % num_pid_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    # See above `Pointer Arithmetic` section for details
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    if a_ptr.dtype.element_ty == tl.int8:
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.int32)
    else:
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        # We accumulate along the K dimension.
        accumulator += tl.dot(a, b)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    out_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

    if not FUSE_SCATTER:
        tl.store(c_ptrs, accumulator, mask=out_mask)

        # 一个 CTA 可能会覆盖多个 reduce-scatter segment 的行。
        # 因此这里要对所有涉及到的 segment 分别计数；只有当该 segment 的所有 N 向 tile
        # 都完成后，才能拉起 ready flag。
        segment_start = pid_m * BLOCK_SIZE_M // M_per_rank
        segment_end = (min((pid_m + 1) * BLOCK_SIZE_M, M) - 1) // M_per_rank
        __syncthreads()
        segment = segment_start + tid(axis=0)
        if segment <= segment_end:
            m_start = M_per_rank * segment
            m_end = M_per_rank * (segment + 1) - 1
            tiled_m_start = m_start // BLOCK_SIZE_M
            tiled_m_end = m_end // BLOCK_SIZE_M
            tiled_m_size = tiled_m_end - tiled_m_start + 1
            val = atomic_add(counter_ptr + segment, 1, semantic="release", scope="gpu")
            if val == num_pid_n * tiled_m_size - 1:
                # or use other signal op semantic
                st(barrier_ptr + segment, 1, scope="gpu", semantic="release")
    else:
        # 单机 fused 模式：直接把每个 tile scatter 到目标对称缓冲区。
        rank_start = pid_m * BLOCK_SIZE_M // M_per_rank
        rank_end = (min((pid_m + 1) * BLOCK_SIZE_M, M) - 1) // M_per_rank
        for cur_rank in range(rank_start, rank_end + 1):
            m_start = max(M_per_rank * cur_rank, pid_m * BLOCK_SIZE_M)
            m_end = min(M_per_rank * (cur_rank + 1) - 1, (min((pid_m + 1) * BLOCK_SIZE_M, M) - 1))
            remote_c_ptr = dl.symm_at(c_ptr, cur_rank)
            mask_offset = m_start - pid_m * BLOCK_SIZE_M
            remote_offs_cm = m_start % M_per_rank + rank * M_per_rank + tl.arange(0, BLOCK_SIZE_M) - mask_offset
            remote_c_ptrs = remote_c_ptr + stride_cm * remote_offs_cm[:, None] + stride_cn * offs_cn[None, :]
            remote_mask = (offs_cm[:, None] <= m_end) & (offs_cm[:, None] >= m_start) & (offs_cn[None, :] < N)
            tl.store(remote_c_ptrs, accumulator, mask=remote_mask)


@requires(has_tma)
def gemm_rs_producer_persistent(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, barrier: torch.Tensor,
                                workspace: torch.Tensor, world_size: int, local_world_size: int, fuse_scatter: bool,
                                num_gemm_sms: int, gemm_config: triton.Config):
    """persistent GEMM producer 的 host 端启动包装函数。"""
    # Check constraints.
    assert A.shape[1] == B.shape[0], "Incompatible dimensions"
    assert A.dtype == B.dtype, "Incompatible dtypes"

    M, local_K = A.shape
    _, N = B.shape

    grid = lambda META: (min(
        num_gemm_sms,
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    ), )
    assert B.stride(0) == 1
    kernel_gemm_rs_producer_persistent[grid](
        A,
        B,
        C,
        M,
        N,
        local_K,
        barrier,
        workspace,
        fuse_scatter,
        local_world_size,
        world_size,
        NUM_SMS=num_gemm_sms,
        **gemm_config.all_kwargs(),
    )


def gemm_rs_producer_non_persistent(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, barrier: torch.Tensor,
                                    workspace: torch.Tensor, world_size: int, local_world_size: int, fuse_scatter: bool,
                                    gemm_config: triton.Config):
    """non-persistent GEMM producer 的 host 端启动包装函数。"""
    # Check constraints.
    assert A.shape[1] == B.shape[0], "Incompatible dimensions"  # b is transposed
    assert A.dtype == B.dtype, "Incompatible dtypes"

    M, K_per_rank = A.shape
    _, N = B.shape

    BLOCK_SIZE_M = gemm_config.kwargs["BLOCK_SIZE_M"]
    BLOCK_SIZE_N = gemm_config.kwargs["BLOCK_SIZE_N"]
    grid = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N), )
    kernel_gemm_rs_producer_non_persistent[grid](
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
        barrier,
        workspace,
        fuse_scatter,
        local_world_size,
        world_size,
        **gemm_config.all_kwargs(),
    )


def update_triton_config(M, N, K, dtype: torch.dtype, world_size, local_world_size, config: triton.Config):
    """
        It's hard to autotune all parameters and record them all, especially when there are so many shapes and devices and dtypes.
        So we just use a simple heuristic rule to update the config.

        这里的启发式规则会比较 estimated GEMM time（估计 GEMM 时间）和 estimated reduce-scatter time
        （估计 reduce-scatter 时间）。如果通信更可能成为瓶颈，就减小 `GROUP_SIZE_M`，
        让 rank 所拥有的 M segment 更早完成、更早暴露给通信侧。
    """
    gemm_time_ms = estimate_gemm_sol_time_ms(M, N, K, dtype)
    rs_time_ms = estimate_reduce_scatter_time_ms(M * N * dtype.itemsize, world_size, local_world_size,
                                                 get_intranode_max_speed_gbps(), get_nic_gbps_per_gpu())
    BLOCK_SIZE_M = config.kwargs["BLOCK_SIZE_M"]
    GROUP_SIZE_M = config.kwargs["GROUP_SIZE_M"]
    if gemm_time_ms < rs_time_ms:
        # comm efficiency first
        M_per_rank = M // world_size
        tiled_m_per_rank = (M_per_rank + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
        if tiled_m_per_rank < GROUP_SIZE_M:
            # don't use too large GROUP_SIZE_M
            # TODO(houqi.1993) maybe we should take care into a wave
            config.kwargs["GROUP_SIZE_M"] = tiled_m_per_rank

    return config


def gemm_rs_op(A: torch.Tensor, B: torch.Tensor, ctx: GEMMReduceScatterTensorParallelContext,
               gemm_config: triton.Config, persistent: bool = True, fuse_scatter: bool = False):
    """GEMM + reduce-scatter 重叠执行的主调度入口。

    时间线可以概括为：
    1. GEMM 把部分结果写入 `gemm_out`
    2. GEMM 在 `scatter_signal` 中标记各个 rank 拥有的 M segment 已 ready
    3. reduce-scatter 在 `rs_stream` 上消费这些已 ready 的 segment

    单机说明：
    - `fuse_scatter=False` 表示“GEMM 先产出到本地对称缓冲区，再由 RS 消费”
    - `fuse_scatter=True` 表示“GEMM 直接写入 scatter 后的对称布局”，此时只剩最后的本地归约
    """
    if fuse_scatter:
        assert ctx.rs_ctx.nnodes == 1, "`fuse_scatter` does not support multi node`"
    world_size = ctx.rs_ctx.world_size
    local_world_size = ctx.rs_ctx.local_world_size
    rs_stream = ctx.rs_stream
    output_dtype = ctx.output_dtype
    num_gemm_sms = ctx.num_gemm_sms

    M, local_K = A.shape
    _, N = B.shape
    assert B.shape == (local_K, ctx.rs_ctx.N)

    assert M % world_size == 0
    M_per_rank = M // world_size
    current_stream = torch.cuda.current_stream()
    rs_stream.wait_stream(current_stream)

    output = torch.empty((M_per_rank, N), dtype=output_dtype, device=A.device)
    workspace = torch.zeros((world_size, ), dtype=torch.int32, device=A.device)
    gemm_out = ctx.get_gemm_out_buf(A)
    # `scatter_signal[segment] == 1` 表示 GEMM 已经完成了属于该 rank-owned segment
    # 的全部 tile，因此通信侧可以开始消费这个 segment。
    scatter_signal = ctx.rs_ctx.scatter_signal_buf

    # TMA descriptors require a global memory allocation
    def alloc_fn(size: int, alignment: int, stream: Optional[int]):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)

    if persistent:
        gemm_rs_producer_persistent(A, B, gemm_out, scatter_signal, workspace, world_size, local_world_size,
                                    fuse_scatter, num_gemm_sms, gemm_config)
    else:
        gemm_config = update_triton_config(M, N, local_K, A.dtype, world_size, local_world_size, gemm_config)
        gemm_rs_producer_non_persistent(A, B, gemm_out, scatter_signal, workspace, world_size, local_world_size,
                                        fuse_scatter, gemm_config)

    if not fuse_scatter:
        with torch.cuda.stream(rs_stream):
            # reduce-scatter 在 side stream 上运行，并等待 GEMM 产出的 segment-ready flag。
            reduce_scatter_2d_op(gemm_out, ctx.rs_ctx, output)
        current_stream.wait_stream(rs_stream)
    else:
        # fused 模式下，scatter 后的布局已经由 GEMM 直接生成，因此这里只需要做本地归约。
        # 这种模式只适用于单机运行。
        nvshmem_barrier_all_on_stream(current_stream)
        ring_reduce(gemm_out, output, ctx.rs_ctx.local_rank, local_world_size)
        nvshmem_barrier_all_on_stream(current_stream)
    return output


def key_fn(A, B, ctx: GEMMReduceScatterTensorParallelContext, *args, **kwargs):
    """autotune（自动调优）的缓存键。"""
    return (triton_dist.tune.to_hashable(A), triton_dist.tune.to_hashable(B), ctx.rs_ctx.world_size,
            ctx.rs_ctx.local_world_size)


def prune_fn(config, A, B, *args, **kwargs):
    """在实际 benchmark（基准测试）之前，先剪掉明显不合适的 autotune 候选项。"""
    itemsize = A.itemsize
    gemm_config = config["gemm_config"].all_kwargs()
    num_stages = gemm_config["num_stages"]
    BLOCK_SIZE_M = gemm_config["BLOCK_SIZE_M"]
    BLOCK_SIZE_N = gemm_config["BLOCK_SIZE_N"]
    BLOCK_SIZE_K = gemm_config["BLOCK_SIZE_K"]
    shared_memory = (itemsize * BLOCK_SIZE_M * BLOCK_SIZE_K + itemsize * BLOCK_SIZE_N * BLOCK_SIZE_K) * num_stages
    M, K = A.shape
    _, N = B.shape
    persistent = config["persistent"]
    fuse_scatter = config["fuse_scatter"]
    tiled_m = (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    tiled_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    tiles = tiled_m * tiled_n
    num_sms = torch.cuda.get_device_properties(0).multi_processor_count
    ntiles_per_sm = tiles // num_sms
    if torch.cuda.get_device_capability()[0] >= 9:
        # TODO(houqi.1993) hardcode here
        if ntiles_per_sm < 4:
            if persistent:
                return False
        if ntiles_per_sm > 10:
            if fuse_scatter or not persistent:
                return False
    return shared_memory < get_device_max_shared_memory_size(0)


def get_gemm_rs_config_space():
    """枚举 GEMM+RS 的 autotune 候选配置空间。"""
    config_space = []
    if torch.cuda.get_device_capability()[0] >= 9:
        config_space += [{"gemm_config": c, "fuse_scatter": fuse_scatter, "persistent": True}
                         for c in get_config_space(True)
                         for fuse_scatter in [True, False]]

    config_space += [{"gemm_config": c, "fuse_scatter": fuse_scatter, "persistent": False}
                     for c in get_config_space(False)
                     for fuse_scatter in [True, False]]
    return config_space


@triton_dist.tune.autotune(
    config_space=get_gemm_rs_config_space(),
    key_fn=key_fn,
    prune_fn=prune_fn,
)
def gemm_rs(A: torch.Tensor, B: torch.Tensor, ctx: GEMMReduceScatterTensorParallelContext, gemm_config: triton.Config,
            persistent=True, fuse_scatter=False):
    """GEMM Reduce-Scatter for Multi-Node

    computes local GEMM (A x B) to generate partial results, followed by `reduce_scatter` to produce c

    该实现通过让 GEMM 发布 per-segment readiness signal（逐段就绪信号），而不是等待整个
    `[M, N]` 输出矩阵完成后再启动通信，从而实现这两个阶段的重叠。

    Args:
        A (torch.Tensor<bfloat16/float16>): A matrix. shape: [M, local_K]
        B (torch.Tensor<bfloat16/float16>): B matrix. shape: [local_K, N]
        ctx(GEMMReduceScatterTensorParallelContext): context

    Returns:
        c (torch.Tensor<bfloat16/float16>): C matrix. shape: [M // world_size, N]
    """
    return gemm_rs_op(A, B, ctx, gemm_config, persistent, fuse_scatter)
