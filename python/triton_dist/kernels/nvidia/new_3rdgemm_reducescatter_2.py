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
import triton_dist
import triton_dist.tune

from triton_dist.kernels.nvidia.gemm import get_config_space
from triton_dist.kernels.nvidia.gemm_reduce_scatter import (gemm_rs_producer_non_persistent,
                                                            gemm_rs_producer_persistent, update_triton_config)
from triton_dist.kernels.nvidia.new_3rdreducescatter import (New3rdReduceScatterContext,
                                                             create_new_3rd_reducescatter_2d_ctx,
                                                             new_3rd_reduce_scatter_2d_op)
from triton_dist.utils import (get_device_max_shared_memory_size, has_fullmesh_nvlink, nvshmem_barrier_all_on_stream,
                               nvshmem_create_tensors, nvshmem_free_tensor_sync)


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
    copy_num_ctas: int = 4,
    reduce_num_ctas: int = 4,
) -> New3rdGEMMReduceScatterTensorParallelContext:
    rs_ctx = create_new_3rd_reducescatter_2d_ctx(
        max_M,
        N,
        rank,
        world_size,
        local_world_size,
        output_dtype,
        copy_num_ctas=copy_num_ctas,
        reduce_num_ctas=reduce_num_ctas,
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
    # Keep persistent/non-persistent as a runtime dimension instead of a config
    # field, otherwise benchmark-side manual autotune can accidentally pass the
    # same semantic knob twice.
    config_space = []
    if torch.cuda.get_device_capability()[0] >= 9:
        config_space += [{"gemm_config": c} for c in get_config_space(True)]
    config_space += [{"gemm_config": c} for c in get_config_space(False)]
    return config_space


def new_3rd_gemm_rs_op(
    A: torch.Tensor,
    B: torch.Tensor,
    ctx: New3rdGEMMReduceScatterTensorParallelContext,
    gemm_config: triton.Config,
    persistent: bool = True,
) -> torch.Tensor:
    """Run upstream nonfused GEMM producer plus the new third-stream RS path."""
    if ctx.rs_ctx.nnodes != 1:
        raise NotImplementedError("new_3rd_gemm_rs currently focuses on single-node only")
    if not has_fullmesh_nvlink():
        raise NotImplementedError("new_3rd_gemm_rs currently expects full-mesh NVLink")

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
    workspace = torch.zeros((world_size, ), dtype=torch.int32, device=A.device)
    gemm_out = ctx.get_gemm_out_buf(A)
    scatter_signal = ctx.rs_ctx.scatter_signal_buf
    ctx.rs_ctx.reset_runtime_state()

    if persistent:
        gemm_rs_producer_persistent(
            A,
            B,
            gemm_out,
            scatter_signal,
            workspace,
            world_size,
            local_world_size,
            False,
            ctx.num_gemm_sms,
            gemm_config,
        )
    else:
        gemm_config = update_triton_config(M, N, local_K, A.dtype, world_size, local_world_size, gemm_config)
        gemm_rs_producer_non_persistent(
            A,
            B,
            gemm_out,
            scatter_signal,
            workspace,
            world_size,
            local_world_size,
            False,
            gemm_config,
        )

    with torch.cuda.stream(ctx.rs_stream):
        new_3rd_reduce_scatter_2d_op(gemm_out, ctx.rs_ctx, output)
    current_stream.wait_stream(ctx.rs_stream)
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
