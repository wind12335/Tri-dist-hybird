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

from triton_dist.kernels.nvidia.common_ops import _wait_eq_cuda
from triton_dist.kernels.nvidia.reduce_scatter import (ReduceScatter2DContext, create_reduce_scater_2d_ctx,
                                                       ring_reduce)
from triton_dist.utils import (has_fullmesh_nvlink, nvshmem_barrier_all_on_stream, nvshmem_create_tensors,
                               nvshmem_free_tensor_sync)


@dataclasses.dataclass
class FuseOverlapReduceScatterContext:
    """Single-node fused-scatter + early-local-reduce context.

    The base reduce-scatter context is reused for stream ownership and buffer
    lifecycle. The only new runtime state is a symmetric ready buffer:

    - `fused_ready_bufs[src_local_rank][dst_local_rank] == 1`
      means source rank `src_local_rank` has finished producing the contribution
      for destination/output rank `dst_local_rank`.
    """

    base_ctx: ReduceScatter2DContext
    fused_ready_bufs: List[torch.Tensor]

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
    def ready_buf(self) -> torch.Tensor:
        return self.fused_ready_bufs[self.local_rank]

    def reset_runtime_state(self) -> None:
        # Each rank only needs to clear its own symmetric ready vector.
        self.ready_buf.zero_()
        self.base_ctx.reset_barriers()

    def wait_ready_for_local_output(self, stream: Optional[torch.cuda.Stream] = None) -> None:
        """Wait until every local peer has marked its contribution on this rank."""
        stream = stream or torch.cuda.current_stream()
        for src_local_rank in range(self.local_world_size):
            ready_scalar = self.ready_buf[src_local_rank:src_local_rank + 1]
            _wait_eq_cuda(ready_scalar, 1, stream)

    def finalize(self) -> None:
        nvshmem_free_tensor_sync(self.ready_buf)
        self.base_ctx.finalize()


def create_fuse_overlap_reducescatter_2d_ctx(
    max_M: int,
    N: int,
    rank: int,
    world_size: int,
    local_world_size: int,
    dtype: torch.dtype,
) -> FuseOverlapReduceScatterContext:
    base_ctx = create_reduce_scater_2d_ctx(max_M, N, rank, world_size, local_world_size, dtype)
    fused_ready_bufs = nvshmem_create_tensors((local_world_size, ), torch.int32, rank, local_world_size)
    fused_ready_bufs[rank % local_world_size].zero_()
    nvshmem_barrier_all_on_stream(torch.cuda.current_stream())
    return FuseOverlapReduceScatterContext(base_ctx=base_ctx, fused_ready_bufs=fused_ready_bufs)


def fuse_overlap_reduce_scatter_2d_op(
    input: torch.Tensor,
    ctx: FuseOverlapReduceScatterContext,
    output: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Reduce the fused-scatter layout as soon as all peer contributions are ready.

    Layout assumption:
    `input` is shaped `[local_world_size * M_per_rank, N]`, where slice
    `[src_local_rank * M_per_rank : (src_local_rank + 1) * M_per_rank]`
    stores the contribution written by source rank `src_local_rank`.
    """
    if ctx.nnodes != 1:
        raise NotImplementedError("fuse_overlap_reduce_scatter only supports single-node runs")
    if not has_fullmesh_nvlink():
        raise NotImplementedError("fuse_overlap_reduce_scatter currently expects full-mesh NVLink")

    M_total, N = input.shape
    if M_total % ctx.local_world_size != 0:
        raise ValueError(f"input rows must be divisible by local_world_size, got shape={tuple(input.shape)}")

    M_per_rank = M_total // ctx.local_world_size
    if output is None:
        output = torch.empty((M_per_rank, N), dtype=input.dtype, device=input.device)

    current_stream = torch.cuda.current_stream()
    reduction_stream = ctx.reduction_stream
    reduction_stream.wait_stream(current_stream)
    with torch.cuda.stream(reduction_stream):
        ctx.wait_ready_for_local_output(reduction_stream)
        ring_reduce(input, output, ctx.local_rank, ctx.local_world_size)
    current_stream.wait_stream(reduction_stream)

    # This context is expected to be reused across benchmark iterations.
    ctx.reset_runtime_state()
    return output
