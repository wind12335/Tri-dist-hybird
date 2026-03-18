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
"""NOTE: allgather.py is for high-throughput. while low_latency_allgather.py is for low-latency.

这个文件实现的是 GEMM/reduce-scatter 重叠所依赖的 reduce-scatter 后端。

执行模型：
- producer kernel（生产者内核，例如 GEMM）先把部分 C tile 写入 symmetric buffer（对称缓冲区）
- 每个 segment（分段）对应的 signal buffer（信号缓冲区）告诉通信侧哪些 M 切片已经 ready（就绪）
- reduce-scatter 一旦发现某个切片 ready，就立刻开始消费，因此通信可以和后续 GEMM tile 的计算重叠，
  而不是等整个矩阵都算完再开始

结合你当前实验环境的拓扑说明：
- 单机 4 卡/8 卡 A100 NVLink 对应 `world_size == local_world_size` 且 `nnodes == 1`
- 此时最关键的是 intra-node scatter（节点内散播）+ local reduction（本地归约）这条路径
- 文件中保留 multi-node（多节点）分支，是为了说明完整设计；同一套 API 也会复用这些逻辑
"""

import dataclasses
from typing import List, Optional

import torch
import triton
import triton_dist
import triton.language as tl
from cuda import cudart
import nvshmem.bindings.nvshmem as pynvshmem
from triton_dist.kernels.nvidia.common_ops import _set_signal_cuda, _wait_eq_cuda
from triton_dist.language.extra import libshmem_device

import triton_dist.language as dl
from triton_dist.kernels.nvidia.common_ops import (barrier_on_this_grid, BarrierAllContext)
from triton_dist.utils import (CUDA_CHECK, NVSHMEM_SIGNAL_DTYPE, has_fullmesh_nvlink, has_tma,
                               launch_cooperative_grid_options, nvshmem_barrier_all_on_stream, nvshmem_create_tensors,
                               nvshmem_free_tensor_sync)
from triton_dist.language.extra.language_extra import tid, __syncthreads, ld, st, atomic_add


@dataclasses.dataclass
class ReduceScatter2DContext:
    """分层 reduce-scatter（hierarchical reduce-scatter）所需的运行时状态与缓冲区。

    对单机 tensor parallel（张量并行）来说，最重要的字段有：
    - `scatter_bufs`：节点内 scatter 阶段使用的 symmetric staging buffer（对称暂存缓冲区）
    - `signal_bufs`：由 producer 写入、由 reduce-scatter 消费的 readiness flag（就绪标志）
    - `reduction_stream`：用于执行 local reduction 的 side stream（侧边流），从而和 producer 重叠

    `rs_per_node_bufs`、`p2p_bufs` 这类更偏 multi-node 的缓冲区，即使单机也会分配，
    但当 `nnodes == 1` 时通常不会真正走到。
    """
    max_M: int
    N: int
    rank: int
    world_size: int
    local_world_size: int
    dtype: torch.dtype

    # comm buffer
    scatter_bufs: List[torch.Tensor]
    rs_per_node_bufs: List[torch.Tensor]
    p2p_bufs: List[torch.Tensor]

    # barrier bufs
    signal_bufs: List[torch.Tensor]  # need reset: signal_buf =  scatter_signal | rs_per_node_signal

    # intra-node barrier
    barrier: BarrierAllContext

    # stream
    reduction_stream: torch.cuda.Stream

    # sms
    num_sync_sms: int
    num_p2p_sms: int
    num_reduction_sms: int

    # preprocess to reduce cpu overhead
    # comm barriers
    scatter_signal_bufs: List[torch.Tensor] = dataclasses.field(init=False)
    rs_per_node_signal_bufs: List[torch.Tensor] = dataclasses.field(init=False)

    local_rank: int = dataclasses.field(init=False)
    node_id: int = dataclasses.field(init=False)
    nnodes: int = dataclasses.field(init=False)

    scatter_signal_buf_list_for_each_node: List[torch.Tensor] = dataclasses.field(init=False)

    def __post_init__(self):
        """预先计算拓扑元数据，并把打包的 signal buffer 拆成逻辑子区间。

        每个 `signal_buf` 的布局是：
        - `[0:world_size)`：scatter readiness flag（scatter 就绪标志）
        - `[world_size:2*world_size)`：per-node reduction readiness flag（每节点归约就绪标志）
        """
        self.local_rank = self.rank % self.local_world_size
        self.node_id = self.rank // self.local_world_size
        assert self.world_size % self.local_world_size == 0
        assert self.max_M % self.world_size == 0
        assert len(self.signal_bufs) == self.local_world_size
        self.nnodes = self.world_size // self.local_world_size
        self.scatter_signal_buf_list_for_each_node = []
        for buf in self.signal_bufs:
            assert buf.shape[0] >= 2 * self.world_size

        self.scatter_signal_bufs = [buf[:self.world_size] for buf in self.signal_bufs]
        self.rs_per_node_signal_bufs = [buf[self.world_size:self.world_size * 2] for buf in self.signal_bufs]

        for node_id in range(self.nnodes):
            # 预先缓存每个 node 对应的 scatter signal 视图，避免在热路径里重复切片。
            self.scatter_signal_buf_list_for_each_node.append(
                self.scatter_signal_bufs[self.local_rank][node_id * self.local_world_size:(node_id + 1) *
                                                          self.local_world_size])

    def reset_barriers(self):
        # 只重置本地 rank 的对称视图；其他 rank 通过 NVSHMEM 看到的是同一块底层存储。
        self.signal_bufs[self.local_rank].fill_(0)

    def get_scatter_bufs_and_signal_for_each_node(self, input, node_id):
        """返回某个 node 大小的 M 切片对应的 buffer 和 signal。"""
        M = input.shape[0]
        M_per_rank = M // self.world_size
        M_per_node = M_per_rank * self.local_world_size
        M_start = node_id * M_per_node
        M_end = M_start + M_per_node
        scatter_bufs_intra_node = [self.scatter_bufs[i][M_start:M_end] for i in range(self.local_world_size)]
        return scatter_bufs_intra_node, self.scatter_signal_buf_list_for_each_node[node_id]

    @property
    def rs_per_node_buf(self) -> torch.Tensor:
        return self.rs_per_node_bufs[self.local_rank]

    @property
    def rs_per_node_signal_buf(self) -> torch.Tensor:
        return self.rs_per_node_signal_bufs[self.local_rank]

    @property
    def p2p_buf(self) -> torch.Tensor:
        return self.p2p_bufs[self.local_rank]

    @property
    def num_rs_sms(self) -> int:
        if self.nnodes > 1:
            return self.num_sync_sms + self.num_p2p_sms + self.num_reduction_sms
        else:
            # for intra node rs, no need sm.
            return 0

    @property
    def scatter_signal_buf(self) -> torch.Tensor:
        return self.scatter_signal_bufs[self.local_rank]

    def finalize(self):
        """释放该 context 持有的全部 symmetric tensor（对称张量）。"""
        nvshmem_free_tensor_sync(self.scatter_bufs[self.local_rank])
        nvshmem_free_tensor_sync(self.rs_per_node_bufs[self.local_rank])
        nvshmem_free_tensor_sync(self.p2p_bufs[self.local_rank])
        nvshmem_free_tensor_sync(self.signal_bufs[self.local_rank])
        # 1.28 修复：手动释放 barrier 中的资源
        if self.barrier is not None:
            self.barrier.finalize()


def create_reduce_scater_2d_ctx(max_M, N, rank, world_size, local_world_size, dtype,
                                num_reduction_sms=15) -> ReduceScatter2DContext:
    """
        创建分层 reduce-scatter 所需的全部 symmetric buffer（对称缓冲区）。

        各个 buffer 的角色：
        - `scatter_bufs`：节点内 scatter 阶段的目标缓冲区
        - `rs_per_node_bufs`：每个 node 大小切片做完 local reduction 后的结果缓冲区
        - `p2p_bufs`：same-local-rank（相同本地 rank）跨节点交换时使用的暂存/输出缓冲区
        - `signal_bufs`：和 GEMM 重叠时使用的打包 readiness flag（就绪标志）

        for num_reduction_sms: tunable param, 16 are enough for H800
            For H800, we overlap local reduce and inter-node p2p with intra-node scatter.
            The reduction kernel bandwidth is not a bottleneck if it exceeds 450GB, so only a few SMs are needed.
            For machines with higher intra_node bandwidth(e.g. H100), we may need to increase the number of SMs or redesign overlapping.

        对你当前的单机 A100 配置：
        - `num_reduction_sms` 控制本地 reduce 阶段可使用的 SM 数量
        - inter-node 缓冲区虽然会存在，但由于 `nnodes == 1`，多节点阶段会被跳过
    """
    assert world_size % local_world_size == 0
    assert max_M % world_size == 0

    scatter_bufs = nvshmem_create_tensors((max_M, N), dtype, rank, local_world_size)
    rs_per_node_bufs = nvshmem_create_tensors((max_M // local_world_size, N), dtype, rank, local_world_size)
    p2p_bufs = nvshmem_create_tensors((max_M // local_world_size, N), dtype, rank, local_world_size)

    # signal_buf: scatter_signal | rs_per_node_signal
    num_signal_bufs = 2
    signal_bufs = nvshmem_create_tensors((world_size * num_signal_bufs, ), NVSHMEM_SIGNAL_DTYPE, rank, local_world_size)

    nvshmem_barrier_all_on_stream(torch.cuda.current_stream())

    reduction_stream: torch.cuda.Stream = torch.cuda.Stream(priority=-1)

    num_sync_sms = 0
    num_p2p_sms = 1
    ctx = ReduceScatter2DContext(max_M=max_M, N=N, rank=rank, world_size=world_size, local_world_size=local_world_size,
                                 dtype=dtype, scatter_bufs=scatter_bufs, rs_per_node_bufs=rs_per_node_bufs,
                                 p2p_bufs=p2p_bufs, signal_bufs=signal_bufs, barrier=BarrierAllContext(True),
                                 reduction_stream=reduction_stream, num_sync_sms=num_sync_sms, num_p2p_sms=num_p2p_sms,
                                 num_reduction_sms=num_reduction_sms)
    return ctx


@triton.jit
def add_continuous_kernel(lhs_ptr, rhs_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
    """扁平连续内存上的逐元素加法，用于本地归约步骤。"""
    pid = tl.program_id(axis=0)
    n_blocks = tl.cdiv(N, BLOCK_SIZE)
    num_pid = tl.num_programs(axis=0)
    for n in range(pid, n_blocks, num_pid):
        offs = n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < N
        lhs = tl.load(lhs_ptr + offs, mask)
        rhs = tl.load(rhs_ptr + offs, mask)
        tl.store(out_ptr + offs, lhs + rhs, mask)


@triton.jit
def copy_continuous_kernel(src_ptr, dst_ptr, N, BLOCK_SIZE: tl.constexpr):
    """扁平连续内存上的拷贝，用于 ring-push 路径。"""
    pid = tl.program_id(axis=0)
    n_blocks = tl.cdiv(N, BLOCK_SIZE)
    num_pid = tl.num_programs(axis=0)
    for n in range(pid, n_blocks, num_pid):
        offs = n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < N
        val = tl.load(src_ptr + offs, mask)
        tl.store(dst_ptr + offs, val, mask)


def add_continuous(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    out: Optional[torch.Tensor],
    num_ctas=16,
    num_warps=32,
):
    """在连续 tensor 上启动扁平加法 kernel。"""
    assert lhs.dtype == rhs.dtype and lhs.numel() == rhs.numel()
    if out is None:
        out = torch.empty_like(lhs)
    block_size = num_warps * 32 * 16 // lhs.itemsize
    add_continuous_kernel[(num_ctas, )](  # local memory bw is very high. use many blocks
        lhs,
        rhs,
        out,
        out.numel(),
        block_size,
    )
    return out


def copy_continous(
    src: torch.Tensor,
    dst: torch.Tensor,
    num_ctas=8,
    num_warps=32,
):
    """在连续 tensor 上启动扁平拷贝 kernel。"""
    assert src.dtype == dst.dtype and src.numel() == dst.numel()
    block_size = num_warps * 32 * 16 // src.itemsize
    copy_continuous_kernel[(num_ctas, )](  # local memory bw is very high. use many blocks
        src,
        dst,
        src.numel(),
        block_size,
    )


def reduce_scatter_ring_push_1d_intra_node_ce(
    rank,
    num_ranks,
    input_tensor: torch.Tensor,
    input_flag: torch.Tensor,
    symm_reduce_tensors: List[torch.Tensor],
    symm_reduce_flags: List[torch.Tensor],
    output: Optional[torch.Tensor] = None,
):
    """使用 stream wait（流等待）实现的、较易阅读的节点内 ring reduce-scatter。

    每个 stage（阶段）做的事：
    - 先确定本轮 ring hop（环上传递）要处理哪个 segment
    - 等待 GEMM 或其他 producer 把这个 segment 标记为 ready
    - stage 0 只转发本地贡献，不做加法
    - 后续 stage 先等待上一跳的 partial sum（部分和），再把本地贡献加进去并继续转发
    - 最后一跳直接把结果写入 `output`，不再落到中间 symmetric buffer

    对单机 tensor parallel 学习来说，这个函数最能直接体现算法结构，虽然实际还存在更快的版本。
    """
    assert input_tensor.is_contiguous()
    assert len(symm_reduce_tensors) == num_ranks
    assert len(symm_reduce_flags) == num_ranks
    (M, _) = input_tensor.shape
    M_per_rank = M // num_ranks
    if output is not None:
        assert (output.dtype == input_tensor.dtype and output.is_contiguous() and output.is_cuda
                and output.shape == (M_per_rank, _))

    to_rank = (rank - 1 + num_ranks) % num_ranks
    for stage in range(num_ranks):
        segment = (rank + stage + 1) % num_ranks
        M_start = segment * M_per_rank
        M_end = M_start + M_per_rank
        src = input_tensor[M_start:M_end]
        dst = symm_reduce_tensors[to_rank][M_start:M_end]
        # 这是 segment 级别重叠的入口：某个 M 切片一旦 ready，通信就可以立刻开始。
        _wait_eq_cuda(input_flag[segment], 1)
        if stage != 0:
            # 从 stage 1 开始，上一跳 ring hop 必须已经在当前 rank 的对称视图中写好了
            # 这个 segment 的 partial sum。
            _wait_eq_cuda(symm_reduce_flags[rank][segment], 1)
            buffer = symm_reduce_tensors[rank][M_start:M_end]
            cur_out = output if output is not None and stage == num_ranks - 1 else buffer
            add_continuous(src, buffer, cur_out)  # directly reduce to output
        if stage == num_ranks - 1:
            break
        if stage == 0:
            # 第一跳只负责转发本地贡献。
            dst.copy_(src)
        else:
            dst.copy_(buffer)
        # 向 ring 上的下一个 rank 发布“这个 partial sum 已可消费”的就绪信号。
        _set_signal_cuda(symm_reduce_flags[to_rank][segment], 1)
    return output


@triton.jit(do_not_specialize=["rank", "num_ranks"])
def reduce_scatter_ring_push_1d_intra_node_kernel(
    rank,
    num_ranks,
    input_ptr,
    symm_input_flag_ptr,
    symm_reduce_ptr,
    symm_reduce_flag_ptr,
    counter,  # use this to sync many grids
    output_ptr,
    elems_per_rank,
    BLOCK_SIZE: tl.constexpr,
    use_cooperative: tl.constexpr,
):
    """同一套 ring-push 算法的 CTA 并行实现。

    `counter` 是按 segment 统计的 CTA 完成计数器。只有最后一个完成该 segment 的 CTA
    才能发布 ready flag（就绪标志），这样可以避免下游 rank 读到未写完整的数据。
    """
    to_rank = (rank - 1 + num_ranks) % num_ranks
    peer_reduce_ptr = tl.multiple_of(dl.symm_at(symm_reduce_ptr, to_rank), 16)
    peer_symm_reduce_flag_ptr = dl.symm_at(symm_reduce_flag_ptr, to_rank)
    thread_idx = tid(0)
    pid = tl.program_id(0)
    num_pid = tl.num_programs(0)

    for stage in range(num_ranks):
        segment = (rank + stage + 1) % num_ranks
        src_ptr = input_ptr + segment * elems_per_rank
        dst_ptr = peer_reduce_ptr + segment * elems_per_rank

        # 等待 producer 把当前 segment 写完。
        if thread_idx == 0:
            while ld(symm_input_flag_ptr + segment, semantic="acquire", scope="sys") != 1:
                pass
        __syncthreads()

        if stage == 0:
            copy_continuous_kernel(src_ptr, dst_ptr, elems_per_rank, BLOCK_SIZE)
        else:
            # 等待上一跳 ring hop 为当前 segment 生成 partial sum。
            if thread_idx == 0:
                while ld(symm_reduce_flag_ptr + segment, semantic="acquire", scope="sys") != 1:
                    pass
            __syncthreads()

            reduce_buffer_ptr = symm_reduce_ptr + elems_per_rank * segment
            add_continuous_kernel(src_ptr, reduce_buffer_ptr, output_ptr if stage == num_ranks - 1 else dst_ptr,
                                  elems_per_rank, BLOCK_SIZE)  # directly reduce to output

        # 必须等所有 CTA 都处理完当前 segment，才能发布 ready flag。
        if thread_idx == 0:
            val = atomic_add(counter + segment, 1, semantic="release", scope="gpu")
            if val == num_pid - 1:
                st(peer_symm_reduce_flag_ptr + segment, 1, semantic="release", scope="sys")
                st(counter + segment, 0)
        __syncthreads()

    if pid == 0:
        libshmem_device.barrier_all_block()


def reduce_scatter_ring_push_1d_intra_node_sm(
    rank,
    num_ranks,
    input_tensor: torch.Tensor,
    input_flag: torch.Tensor,
    symm_reduce_tensor: torch.Tensor,
    symm_reduce_flag: torch.Tensor,
    grid_barrier: torch.Tensor,
    output: Optional[torch.Tensor] = None,
    num_sms=0,
):
    """CTA 并行节点内 ring reduce-scatter kernel 的启动包装函数。"""
    M, _ = input_tensor.shape
    M_per_rank = M // num_ranks
    output = output if output is not None else torch.empty(
        (M_per_rank, _), dtype=input_tensor.dtype, device=input_tensor.device)
    num_warps = 32
    block_size = 32 * num_warps * 16 // input_tensor.dtype.itemsize
    if num_sms == 0:
        num_sms = triton.cdiv(input_tensor.numel(), block_size)
    reduce_scatter_ring_push_1d_intra_node_kernel[(num_sms, )](
        rank,
        num_ranks,
        input_tensor,
        input_flag,
        symm_reduce_tensor,
        symm_reduce_flag,
        grid_barrier,
        output,
        input_tensor.numel() // num_ranks,
        BLOCK_SIZE=block_size,  # each thread copy a uint4
        num_warps=num_warps,
        use_cooperative=True,
        **launch_cooperative_grid_options(),
    )
    return output


@triton.jit(do_not_specialize=["rank", "num_ranks"])
def reduce_scatter_ring_push_1d_intra_node_rma_kernel(
    rank,
    num_ranks,
    symm_input_ptr,
    symm_input_flag_ptr,
    symm_reduce_ptr,
    symm_reduce_flag_ptr,
    grid_barrier_ptr,  # use this to sync many grids
    output_ptr,
    elems_per_rank,
    BLOCK_SIZE: tl.constexpr,
    use_cooperative: tl.constexpr,
):
    """ why this kernel, what's the difference with reduce_scatter_ring_push_1d_intra_node_kernel?

    for some PCI-e machines, we find that NCCL use NIC to communicate cross NUMA nodes. so we follow this design.

    with rma, the kernel is a little different:
    1. we have to implicit do ADD and save result to buffer. then putmem_rma to remote

    这条路径主要面向 PCIe/NUMA 风格环境。在你当前的单机 NVLink A100 机器上，
    它通常不是重点研究路径。
    """
    to_rank = (rank - 1 + num_ranks) % num_ranks
    thread_idx = tid(0)
    pid = tl.program_id(0)

    ITEM_SIZE = tl.constexpr(symm_input_ptr.dtype.primitive_bitwidth) // 8
    NUMA_WORLD_SIZE = 4
    use_rma = rank % NUMA_WORLD_SIZE == 0  # TODO(houqi.1993) maybe numa_world_size

    if not use_rma:
        return reduce_scatter_ring_push_1d_intra_node_kernel(rank, num_ranks, symm_input_ptr, symm_input_flag_ptr,
                                                             symm_reduce_ptr, symm_reduce_flag_ptr, grid_barrier_ptr,
                                                             output_ptr, elems_per_rank, BLOCK_SIZE=BLOCK_SIZE,
                                                             use_cooperative=use_cooperative)

    for stage in range(num_ranks):
        segment = (rank + stage + 1) % num_ranks
        src_ptr = symm_input_ptr + segment * elems_per_rank
        dst_ptr = symm_reduce_ptr + segment * elems_per_rank

        # wait by many CTA's is OK
        if thread_idx == 0:
            while ld(symm_input_flag_ptr + segment, semantic="acquire", scope="gpu") != 1:
                pass
        __syncthreads()

        if stage != 0:
            # wait for reduce ready
            if thread_idx == 0:
                while ld(symm_reduce_flag_ptr + segment, semantic="acquire", scope="sys") != 1:
                    pass
            __syncthreads()

            add_continuous_kernel(src_ptr, dst_ptr, output_ptr if stage == num_ranks - 1 else dst_ptr, elems_per_rank,
                                  BLOCK_SIZE)  # directly reduce to output
            barrier_on_this_grid(grid_barrier_ptr, use_cooperative)

        if stage != num_ranks - 1 and pid == 0:
            # set flag only after all CTAs done memcpy/reduce
            libshmem_device.putmem_signal_nbi_block(dst_ptr, dst_ptr if stage != 0 else src_ptr,
                                                    elems_per_rank * ITEM_SIZE, symm_reduce_flag_ptr, 1,
                                                    libshmem_device.NVSHMEM_SIGNAL_SET, to_rank)

    if pid == 0:
        libshmem_device.barrier_all_block()


def reduce_scatter_ring_push_1d_intra_node_sm_rma(
    rank,
    num_ranks,
    input_tensor: torch.Tensor,
    input_flag: torch.Tensor,
    symm_reduce_tensor: torch.Tensor,
    symm_reduce_flag: torch.Tensor,
    grid_barrier: torch.Tensor,
    output: Optional[torch.Tensor] = None,
    num_sms=1,
):
    """RMA 版本的包装函数。

    在你当前的 NVLink 环境下，这个 helper 通常不是性能关键路径。
    """
    M, _ = input_tensor.shape
    M_per_rank = M // num_ranks
    output = output if output is not None else torch.empty(
        (M_per_rank, _), dtype=input_tensor.dtype, device=input_tensor.device)
    num_warps = 32
    reduce_scatter_ring_push_1d_intra_node_kernel[(num_sms, )](
        rank, num_ranks, input_tensor, input_flag, symm_reduce_tensor, symm_reduce_flag, grid_barrier, output,
        input_tensor.numel() // num_ranks,
        BLOCK_SIZE=32 * num_warps * 16 // input_tensor.dtype.itemsize,  # each thread copy a uint4
        num_warps=num_warps, use_cooperative=True, **launch_cooperative_grid_options())
    return output


################### triton kernel ###################
@triton_dist.jit
def kernel_inter_node_p2p_for_same_local_rank(offset, local_world_size, M_per_rank, N, input,  # [M, N]
                                              output,  # [M, N]
                                              ):
    """在不同 node 之间交换一个 same-local-rank shard（相同本地 rank 的分片）。

    只有 `nnodes > 1` 时才会使用。你当前的单机场景下不会启动这个 kernel。
    """
    rank = dl.rank()
    world_size = dl.num_ranks()
    node_id = rank // local_world_size
    nnodes = world_size // local_world_size
    local_rank = rank % local_world_size
    nelem_per_rank = M_per_rank * N

    remote_node_id = (offset + 1 + node_id) % nnodes
    remote_rank = local_rank + remote_node_id * local_world_size
    elem_size = tl.constexpr(input.dtype.element_ty.primitive_bitwidth) // 8
    libshmem_device.putmem_block(
        output + node_id * nelem_per_rank,
        input + remote_node_id * nelem_per_rank,
        nelem_per_rank * elem_size,
        remote_rank,
    )


def reduce_scatter_for_each_node_ring(input: torch.Tensor, ctx: ReduceScatter2DContext,
                                      output: Optional[torch.Tensor] = None):
    """当 full-mesh NVLink 不可用时采用的分层 fallback 路径。

    在这条路径里，即使是节点内工作也用 ring-push 算法来表达。对一台正常的单机 NVLink
    机器来说，通常更值得关注的是 `reduce_scatter_for_each_node` 那条路径。
    """
    world_size = ctx.world_size
    local_world_size = ctx.local_world_size
    local_rank = ctx.local_rank
    (M, N) = input.shape
    M_per_rank = M // world_size
    M_per_node = M_per_rank * local_world_size
    nnodes = ctx.nnodes
    node_id = ctx.node_id
    p2p_buf = ctx.p2p_buf

    for n in range(0, nnodes):
        cur_node_id = (node_id + n + 1) % nnodes
        M_start = cur_node_id * M_per_node
        M_end = M_start + M_per_node
        scatter_bufs_intra_node, scatter_signal_buf_intra_node = ctx.get_scatter_bufs_and_signal_for_each_node(
            input, cur_node_id)

        scatter_buf = reduce_scatter_ring_push_1d_intra_node_ce(
            local_rank,
            local_world_size,
            input[M_start:M_end],
            scatter_signal_buf_intra_node,
            scatter_bufs_intra_node,
            [
                x[cur_node_id * local_world_size:(cur_node_id + 1) * local_world_size]
                for x in ctx.rs_per_node_signal_bufs
            ],
            output,
        )

        # 已完成归约的 shard 在 node 间继续交换；当 `nnodes == 1` 时会跳过。
        if nnodes > 1:
            if n == nnodes - 1:
                M_start = M_per_rank * node_id
                M_end = M_start + M_per_rank
                p2p_buf[M_start:M_end].copy_(scatter_buf)
            else:
                peer_node_id = (node_id + n + 1) % nnodes
                peer_rank = local_rank + peer_node_id * local_world_size
                nbytes_per_rank = M_per_rank * input.dtype.itemsize * N
                M_start = M_per_rank * node_id
                M_end = M_start + M_per_rank

                pynvshmem.putmem_on_stream(
                    p2p_buf[M_start:M_end].data_ptr(),
                    scatter_buf.data_ptr(),
                    nbytes_per_rank,
                    peer_rank,
                )
                nvshmem_barrier_all_on_stream()

    if nnodes == 1:
        return scatter_buf
    return p2p_buf[:M_per_rank * nnodes]


def intra_node_scatter(input_intra_node, scatter_bufs_intra_node: List[torch.Tensor],
                       scatter_signal_buf_intra_node: torch.Tensor, local_rank, overlap_with_gemm=True):
    """把一个 node 大小的 M 切片 scatter 到本地各个 GPU。

    这里最关键的重叠机制，是等待 `scatter_signal_buf_intra_node[remote_local_rank]`：
    每个远端 stripe（条带）只有在 producer 完成后才会被拷贝。也正因为如此，通信可以在
    后续 GEMM tile 仍在计算时就提前开始。
    """
    M, N = input_intra_node.shape
    local_world_size = len(scatter_bufs_intra_node)
    M_per_rank = M // local_world_size
    """
        use flattern pointer and driver api to reduce the overhead of slice, plus the offset is equal to tensor slice op:
            `signal_base_ptr + nbytes_per_scatter_signal * remote_local_rank`: `scatter_signal_buf_intra_node[remote_local_rank].data_ptr()`
            `scatter_bufs_intra_node[remote_local_rank].data_ptr() + remote_offset`: `scatter_bufs_intra_node[remote_local_rank][local_rank * M_per_rank:(local_rank + 1) * M_per_rank, :]`
            `local_buf_base_ptr + remote_local_rank * nbytes_per_rank`: `input_intra_node[remote_local_rank * M_per_rank:(remote_local_rank + 1) * M_per_rank, :]`
    """
    nbytes_per_rank = M_per_rank * N * input_intra_node.dtype.itemsize
    local_buf_base_ptr = input_intra_node.data_ptr()
    remote_offset = local_rank * nbytes_per_rank
    stream = torch.cuda.current_stream()
    for i in range(0, local_world_size):
        # 按 ring 顺序访问本地 peer。每次拷贝的区域都对应 reduce-scatter 之后应归属
        # `remote_local_rank` 的那条 stripe。
        remote_local_rank = (local_rank + i + 1) % local_world_size
        if overlap_with_gemm:
            _wait_eq_cuda(scatter_signal_buf_intra_node[remote_local_rank], 1, stream)
        remote_buf_ptr = scatter_bufs_intra_node[remote_local_rank].data_ptr() + remote_offset
        local_buf_ptr = local_buf_base_ptr + remote_local_rank * nbytes_per_rank
        (err, ) = cudart.cudaMemcpyAsync(
            remote_buf_ptr,
            local_buf_ptr,
            nbytes_per_rank,
            cudart.cudaMemcpyKind.cudaMemcpyDefault,
            stream.cuda_stream,
        )
        CUDA_CHECK(err)


def reduce_scatter_for_each_node(input: torch.Tensor, ctx: ReduceScatter2DContext,
                                 output: Optional[torch.Tensor] = None):
    """面向 full-mesh NVLink 系统的高性能分层 reduce-scatter 路径。

    对每个 node 大小的切片，执行流程是：
    1. 先把各 rank 拥有的 stripe scatter 到本地各 GPU
    2. 再在 `reduction_stream` 上对本地堆叠结果做 reduce
    3. 如果是多节点，还要在 node 间交换 same-local-rank shard

    对你当前的单机环境，只会执行前两步，并且最终结果可以直接写到 `output`。
    """
    world_size = ctx.world_size
    local_world_size = ctx.local_world_size
    local_rank = ctx.local_rank
    reduction_stream = ctx.reduction_stream
    num_reduction_sms = ctx.num_reduction_sms
    M, N = input.shape
    M_per_rank = M // world_size
    M_per_node = M_per_rank * local_world_size
    nnodes = ctx.nnodes
    node_id = ctx.node_id
    rs_per_node_buf = ctx.rs_per_node_buf
    p2p_buf = ctx.p2p_buf

    stream = torch.cuda.current_stream()
    for n in range(0, nnodes):
        cur_node_id = (node_id + n + 1) % nnodes
        input_intra_node = input[cur_node_id * M_per_node:(cur_node_id + 1) * M_per_node]
        scatter_bufs_intra_node, scatter_signal_buf_intra_node = ctx.get_scatter_bufs_and_signal_for_each_node(
            input, cur_node_id)
        intra_node_scatter(input_intra_node, scatter_bufs_intra_node, scatter_signal_buf_intra_node, local_rank)

        # local reduction 被放到 side stream 上执行，这样主 stream 可以继续推进 producer。
        # 这是实现重叠的关键部分之一。
        rs_buf_cur_node = rs_per_node_buf[M_per_rank * cur_node_id:(cur_node_id + 1) * M_per_rank]
        nvshmem_barrier_all_on_stream(stream)
        reduction_stream.wait_stream(stream)
        with torch.cuda.stream(reduction_stream):
            reduce_out_buf = output if nnodes == 1 else rs_buf_cur_node
            ring_reduce(scatter_bufs_intra_node[local_rank], reduce_out_buf, local_rank, local_world_size,
                        num_sms=-1 if n == nnodes - 1 else num_reduction_sms)

            # 仅多节点需要：把 same-local-rank shard 发送到目标 node。
            if nnodes > 1:
                if n == nnodes - 1:
                    p2p_buf[M_per_rank * node_id:M_per_rank * (node_id + 1)].copy_(
                        rs_per_node_buf[M_per_rank * node_id:M_per_rank * (node_id + 1)])
                else:
                    grid = lambda META: (ctx.num_p2p_sms, )
                    kernel_inter_node_p2p_for_same_local_rank[grid](
                        n,
                        local_world_size,
                        M_per_rank,
                        N,
                        rs_per_node_buf,
                        p2p_buf,
                        num_warps=16,
                    )

    stream.wait_stream(reduction_stream)
    if nnodes == 1:
        return output
    return p2p_buf[:M_per_rank * nnodes]


@triton.jit(do_not_specialize=["begin_idx"])
def kernel_ring_reduce_non_tma(
    in_ptr,  # c of shape [NUM_SPLITS, elems_per_rank]
    out_ptr,  # out = sum(c, axis=0) of shape [elems_per_rank]
    elems_per_rank,
    begin_idx,  # reduce in order (begin_idx + i) % NUM_SPLITS for i in [0, NUM_SPLITS - 1]
    NUM_SPLITS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """在不使用 TMA 的情况下，把 `NUM_SPLITS` 个连续 shard 归约成一个 shard。"""
    num_blocks = tl.cdiv(elems_per_rank, BLOCK_SIZE)
    pid = tl.program_id(0)
    npid = tl.num_programs(0)
    for n in range(pid, num_blocks, npid):
        segment = (begin_idx + 1) % NUM_SPLITS
        c_offs = elems_per_rank * segment + BLOCK_SIZE * n + tl.arange(0, BLOCK_SIZE)
        mask = c_offs < elems_per_rank * (segment + 1)
        accum = tl.load(in_ptr + c_offs, mask=mask)
        for i in range(1, NUM_SPLITS):
            segment = (i + begin_idx + 1) % NUM_SPLITS
            c_offs = elems_per_rank * segment + BLOCK_SIZE * n + tl.arange(0, BLOCK_SIZE)
            data = tl.load(in_ptr + c_offs, mask=mask)
            accum += data

        out_offs = BLOCK_SIZE * n + tl.arange(0, BLOCK_SIZE)
        tl.store(out_ptr + out_offs, accum, mask=mask)


@triton.jit
def kernel_ring_reduce_tma(
    c_ptr,  # [M, N]
    out_ptr,  # [M_per_split, N]
    # shape of matrix
    M_per_rank,
    N,
    begin_idx,
    num_splits: tl.constexpr,
    # reduce tile shape
    BLOCK_SIZE_M: tl.constexpr = 256,
    BLOCK_SIZE_N: tl.constexpr = 64,
):
    """基于 TMA 的本地归约，处理堆叠后的 `[num_splits, M_per_rank, N]` 布局。"""
    c_desc = tl.make_tensor_descriptor(
        c_ptr,
        shape=[M_per_rank * num_splits, N],
        strides=[N, 1],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
    )
    output_desc = tl.make_tensor_descriptor(
        out_ptr,
        shape=[M_per_rank, N],
        strides=[N, 1],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
    )

    pid = tl.program_id(axis=0)
    num_pid = tl.num_programs(axis=0)
    num_tiles_m = tl.cdiv(M_per_rank, BLOCK_SIZE_M)
    num_tiles_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_tiles_m * num_tiles_n
    for tile_id in range(pid, total_tiles, num_pid):
        tile_id_m = tile_id // num_tiles_n
        tile_id_n = tile_id % num_tiles_n
        # accum = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=out_ptr.dtype.element_ty)
        cur_rank = (begin_idx + 1) % num_splits
        accum = c_desc.load([tile_id_m * BLOCK_SIZE_M + cur_rank * M_per_rank, tile_id_n * BLOCK_SIZE_N])
        for i in range(1, num_splits):
            cur_rank = (i + begin_idx + 1) % num_splits
            data = c_desc.load([tile_id_m * BLOCK_SIZE_M + cur_rank * M_per_rank, tile_id_n * BLOCK_SIZE_N])
            accum += data

        output_desc.store([tile_id_m * BLOCK_SIZE_M, tile_id_n * BLOCK_SIZE_N], accum)


def ring_reduce_non_tma(
    input: torch.Tensor,  # [M_per_node, N]
    output: torch.Tensor,  # [M_per_rank, N]
    begin_idx,
    num_splits,
    num_sms=16,
):
    """非 TMA 本地归约 kernel 的 host 端包装函数。"""
    total_M, N = input.shape
    M_per_split = total_M // num_splits
    assert output.shape[0] == M_per_split and total_M % num_splits == 0, output.shape
    num_warps = 32
    kernel_ring_reduce_non_tma[(num_sms, )](
        input,
        output,
        M_per_split * N,
        begin_idx,
        num_splits,
        BLOCK_SIZE=32 * num_warps * 8 * 4,
        num_warps=num_warps,
    )
    return output


def ring_reduce_tma(
    input: torch.Tensor,  # [M_per_node, N]
    output: torch.Tensor,  # [M_per_rank, N]
    begin_idx,
    num_splits,
    num_sms=-1,
):
    """TMA 本地归约 kernel 的 host 端包装函数。"""
    total_M, N = input.shape
    M_per_split = total_M // num_splits
    assert output.shape[0] == M_per_split and total_M % num_splits == 0

    def alloc_fn(size, alignment, stream):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)

    if num_sms == -1:
        grid = lambda META: (triton.cdiv(M_per_split, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]), )
        kernel_ring_reduce_tma[grid](
            input,
            output,
            M_per_split,
            N,
            begin_idx,
            num_splits,
            BLOCK_SIZE_M=256,
            BLOCK_SIZE_N=64,
            num_warps=4,
        )
    else:
        grid = lambda META: (min(
            triton.cdiv(M_per_split, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]), num_sms), )
        kernel_ring_reduce_tma[grid](
            input,
            output,
            M_per_split,
            N,
            begin_idx,
            num_splits,
            BLOCK_SIZE_M=256,
            BLOCK_SIZE_N=128,
            num_warps=8,
        )

    return output


def ring_reduce(
    input,  # [M_per_node, N]
    output,  # [M_per_rank, N]
    begin_idx,
    num_splits,
    num_sms=-1,
):
    """根据硬件能力选择本地归约实现。"""
    if has_tma():
        return ring_reduce_tma(input, output, begin_idx, num_splits, num_sms)
    else:
        return ring_reduce_non_tma(input, output, begin_idx, num_splits, 16 if num_sms == -1 else num_sms)


def reduce_scatter_multi_node(input: torch.Tensor, ctx: ReduceScatter2DContext, output: Optional[torch.Tensor] = None):
    """
    A hierarchical reduce-scatter implementation that overlaps the intra-node scatter
    with the local reduce and the inter-node p2p(after reduce). It also provides a rank-wise
    signal and supports overlap with gemm.

    单机说明：
    - 当 `ctx.nnodes == 1` 时，`reduce_scatter_for_each_node(...)` 返回的每节点结果
      就已经是最终答案
    - 下面最后那一步跨节点 `ring_reduce(...)` 会被跳过
    """
    M, N = input.shape
    M_per_rank = M // ctx.world_size

    current_stream = torch.cuda.current_stream()
    ctx.reduction_stream.wait_stream(current_stream)

    # 单机时，节点内阶段就可以直接把结果写入最终输出张量。
    out_each_node = output if ctx.nnodes == 1 else None
    if not has_fullmesh_nvlink():
        rs_result_per_node = reduce_scatter_for_each_node_ring(input, ctx, out_each_node)
    else:
        rs_result_per_node = reduce_scatter_for_each_node(input, ctx, out_each_node)

    if ctx.nnodes == 1:
        return rs_result_per_node

    # 仅多节点需要：把每个 node 提供的一个 shard 再归约成当前 rank 的最终结果。
    nvshmem_barrier_all_on_stream(current_stream)
    if output is None:
        output = torch.empty((M_per_rank, N), dtype=input.dtype, device=input.device)
    ring_reduce(rs_result_per_node, output, ctx.node_id, ctx.nnodes)
    return output


def reduce_scatter_2d_op(input: torch.Tensor, ctx: ReduceScatter2DContext, output: Optional[torch.Tensor] = None):
    """供 GEMM 重叠逻辑调用的公开 reduce-scatter 入口。"""
    M, N = input.shape
    assert input.dtype == ctx.dtype
    assert ctx.max_M >= M and ctx.N == N
    assert M % ctx.world_size == 0

    # 在消费各 segment 的 ready signal 之前，先确保全局状态干净一致。
    nvshmem_barrier_all_on_stream(torch.cuda.current_stream())
    output = reduce_scatter_multi_node(input, ctx, output)
    # 这个 context 会被重复复用，因此返回前必须清空所有 segment 级别的 flag。
    ctx.reset_barriers()
    return output
