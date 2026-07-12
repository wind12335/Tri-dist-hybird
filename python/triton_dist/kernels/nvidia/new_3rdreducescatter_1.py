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
from cuda import cudart

from triton_dist.kernels.nvidia.common_ops import _set_signal_cuda, _wait_eq_cuda
from triton_dist.kernels.nvidia.reduce_scatter import (ReduceScatter2DContext, add_continuous, copy_continous,
                                                       create_reduce_scater_2d_ctx, reduce_scatter_2d_op)
from triton_dist.utils import (CUDA_CHECK, NVSHMEM_SIGNAL_DTYPE, has_fullmesh_nvlink, nvshmem_barrier_all_on_stream,
                               nvshmem_create_tensors, nvshmem_free_tensor_sync)


def _expected_arrival_order(local_rank: int, local_world_size: int) -> List[int]:
    """Return the source order that most closely matches the upstream scatter ring.

    In the original intra-node scatter loop, source rank `s` visits destination
    ranks in the order `(s + 1), (s + 2), ..., s`. For a fixed destination rank
    `d`, the earliest expected arrivals therefore come from
    `(d - 1), (d - 2), ..., d`.
    """
    return [((local_rank - 1 - step) % local_world_size) for step in range(local_world_size)]


@dataclasses.dataclass
class New3rdReduceScatterContext:
    """Single-node nonfused reduce-scatter with a third helper stream.

    This path keeps the upstream nonfused producer and scatter layout, but adds
    destination-local arrival flags so that a low-SM helper stream can start
    incremental local reduction as soon as contributions arrive.

    For destination rank `d`:
    - `arrival_flag_bufs[d][s] == 1` means source rank `s` has finished copying
      its contribution for destination shard `d` into `scatter_bufs[d]`.
    """

    base_ctx: ReduceScatter2DContext
    arrival_flag_bufs: List[torch.Tensor]
    copy_num_ctas: int = 4
    reduce_num_ctas: int = 4

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
    def scatter_signal_buf(self) -> torch.Tensor:
        return self.base_ctx.scatter_signal_buf

    @property
    def arrival_flag_buf(self) -> torch.Tensor:
        return self.arrival_flag_bufs[self.local_rank]

    def reset_runtime_state(self) -> None:
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
    copy_num_ctas: int = 4,
    reduce_num_ctas: int = 4,
) -> New3rdReduceScatterContext:
    """Create the context for the third-stream incremental reduce-scatter path."""
    base_ctx = create_reduce_scater_2d_ctx(max_M, N, rank, world_size, local_world_size, dtype)
    arrival_flag_bufs = nvshmem_create_tensors((local_world_size, ), NVSHMEM_SIGNAL_DTYPE, rank, local_world_size)
    arrival_flag_bufs[rank % local_world_size].zero_()
    nvshmem_barrier_all_on_stream(torch.cuda.current_stream())
    return New3rdReduceScatterContext(
        base_ctx=base_ctx,
        arrival_flag_bufs=arrival_flag_bufs,
        copy_num_ctas=copy_num_ctas,
        reduce_num_ctas=reduce_num_ctas,
    )


def _launch_incremental_local_reduce(
    local_scatter: torch.Tensor,
    ctx: New3rdReduceScatterContext,
    output: torch.Tensor,
) -> None:
    """Reduce one destination-local shard contribution by contribution.

    This helper runs on the third stream. It waits on destination-local arrival
    flags and accumulates into `output` as soon as each source contribution is
    visible, instead of deferring all work to a final one-shot `ring_reduce`.
    """
    reduction_stream = ctx.reduction_stream
    reduction_stream.wait_stream(torch.cuda.current_stream())

    M_total, N = local_scatter.shape
    M_per_rank = M_total // ctx.local_world_size
    order = _expected_arrival_order(ctx.local_rank, ctx.local_world_size)

    with torch.cuda.stream(reduction_stream):
        for stage, src_local_rank in enumerate(order):
            ready_scalar = ctx.arrival_flag_buf[src_local_rank:src_local_rank + 1]
            _wait_eq_cuda(ready_scalar, 1, reduction_stream)
            src_slice = local_scatter[src_local_rank * M_per_rank:(src_local_rank + 1) * M_per_rank]
            if stage == 0:
                copy_continous(src_slice, output, num_ctas=ctx.copy_num_ctas)
            else:
                add_continuous(output, src_slice, output, num_ctas=ctx.reduce_num_ctas)


def intra_node_3rd_scatter_reduce(
    input_intra_node: torch.Tensor,
    ctx: New3rdReduceScatterContext,
    output: torch.Tensor,
) -> None:
    """Scatter contributions first, then incrementally reduce on a helper stream.

    Stream roles:
    - current stream: node-local scatter copy
    - `ctx.reduction_stream`: destination-local incremental reduction
    """
    local_rank = ctx.local_rank
    local_world_size = ctx.local_world_size
    scatter_stream = torch.cuda.current_stream()

    scatter_bufs_intra_node, scatter_signal_buf_intra_node = ctx.base_ctx.get_scatter_bufs_and_signal_for_each_node(
        input_intra_node, ctx.node_id)
    local_scatter = scatter_bufs_intra_node[local_rank]
    _launch_incremental_local_reduce(local_scatter, ctx, output)

    M, N = input_intra_node.shape
    M_per_rank = M // local_world_size
    nbytes_per_rank = M_per_rank * N * input_intra_node.dtype.itemsize
    local_buf_base_ptr = input_intra_node.data_ptr()
    remote_offset = local_rank * nbytes_per_rank

    for step in range(local_world_size):
        dest_local_rank = (local_rank + step + 1) % local_world_size
        _wait_eq_cuda(scatter_signal_buf_intra_node[dest_local_rank], 1, scatter_stream)

        remote_buf_ptr = scatter_bufs_intra_node[dest_local_rank].data_ptr() + remote_offset
        local_buf_ptr = local_buf_base_ptr + dest_local_rank * nbytes_per_rank
        (err, ) = cudart.cudaMemcpyAsync(
            remote_buf_ptr,
            local_buf_ptr,
            nbytes_per_rank,
            cudart.cudaMemcpyKind.cudaMemcpyDefault,
            scatter_stream.cuda_stream,
        )
        CUDA_CHECK(err)

        remote_ready = ctx.arrival_flag_bufs[dest_local_rank][local_rank:local_rank + 1]
        _set_signal_cuda(remote_ready, 1, scatter_stream)


def new_3rd_reduce_scatter_2d_op(
    input: torch.Tensor,
    ctx: New3rdReduceScatterContext,
    output: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Incremental nonfused reduce-scatter driven by destination-local arrivals.

    Unsupported topologies fall back to the upstream implementation.
    """
    if ctx.nnodes != 1 or not has_fullmesh_nvlink():
        return reduce_scatter_2d_op(input, ctx.base_ctx, output)

    M, N = input.shape
    if output is None:
        output = torch.empty((M // ctx.world_size, N), dtype=input.dtype, device=input.device)

    intra_node_3rd_scatter_reduce(input, ctx, output)
    torch.cuda.current_stream().wait_stream(ctx.reduction_stream)
    return output
