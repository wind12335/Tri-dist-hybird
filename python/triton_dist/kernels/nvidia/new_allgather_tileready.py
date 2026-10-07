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

from typing import Optional

import torch
import triton

from triton_dist.kernels.nvidia.allgather import (AllGatherMethod, cp_engine_producer_all_gather_intra_node,
                                                  get_auto_all_gather_method)
from triton_dist.kernels.nvidia.allgather_gemm import copy_and_barrier_all_intra_node_kernel
from triton_dist.kernels.nvidia.common_ops import _set_signal_cuda, _wait_eq_cuda
from triton_dist.utils import launch_cooperative_grid_options


def _default_copy_sms(copy_sms: int) -> int:
    num_sms = torch.cuda.get_device_properties("cuda").multi_processor_count
    if copy_sms > 0:
        return min(copy_sms, num_sms)
    return max(1, num_sms // 4)


def launch_new_allgather_intra_node(
    local_tensor: torch.Tensor,
    ctx,
    *,
    debug: bool = False,
    use_cooperative: bool = False,
    copy_sms: int = 0,
    all_gather_method: AllGatherMethod = AllGatherMethod.Auto,
    use_row_tile_barrier: bool = False,
    row_tile_barrier_buffers: Optional[list[torch.Tensor]] = None,
    num_tile_chunks: int = 0,
    tile_rows_per_chunk: int = 256,
) -> torch.cuda.Stream:
    if getattr(ctx, "is_multinode", False):
        raise NotImplementedError("launch_new_allgather_intra_node is for intra-node only")

    current_stream = torch.cuda.current_stream()

    method = all_gather_method
    if method == AllGatherMethod.Auto:
        method = get_auto_all_gather_method(ctx.num_ranks, ctx.num_local_ranks)

    if use_row_tile_barrier:
        if method != AllGatherMethod.All2All_IntraNode:
            raise ValueError("use_row_tile_barrier currently only supports All2All_IntraNode")
        if row_tile_barrier_buffers is None:
            raise ValueError("row_tile_barrier_buffers must be provided when use_row_tile_barrier=True")
        if num_tile_chunks <= 0:
            raise ValueError(f"num_tile_chunks must be > 0, got {num_tile_chunks}")
        if tile_rows_per_chunk <= 0:
            raise ValueError(f"tile_rows_per_chunk must be > 0, got {tile_rows_per_chunk}")

        # Initialize the ready state before the AG stream is allowed to submit
        # copies or readiness signals.  The remote consumer stream already
        # follows the caller's current stream; placing the clear here makes
        # that initialization an explicit predecessor of both paths.
        tile_barrier = row_tile_barrier_buffers[ctx.local_rank]
        tile_barrier.zero_()

    ctx.ag_intranode_stream.wait_stream(current_stream)

    with torch.cuda.stream(ctx.ag_intranode_stream):
        M_per_rank, K = local_tensor.shape
        cp_block_m, cp_block_n = 128, 256
        total_tiles = triton.cdiv(M_per_rank, cp_block_m) * triton.cdiv(K, cp_block_n)
        grid = (min(total_tiles, _default_copy_sms(copy_sms)), )

        additional_options = {}
        if use_cooperative:
            additional_options.update(launch_cooperative_grid_options())

        copy_and_barrier_all_intra_node_kernel[grid](
            ctx.local_rank,
            ctx.rank,
            ctx.num_ranks,
            local_tensor,
            ctx.symm_workspace,
            ctx.symm_barrier,
            ctx.symm_comm_buf,
            M_per_rank,
            K,
            local_tensor.stride(0),
            local_tensor.stride(1),
            ctx.symm_workspace.stride(0),
            ctx.symm_workspace.stride(1),
            ctx.phase,
            cp_block_m,
            cp_block_n,
            use_cooperative,
            **additional_options,
        )
        ctx.phase += 2

        if not use_row_tile_barrier:
            cp_engine_producer_all_gather_intra_node(
                ctx.rank,
                ctx.num_ranks,
                local_tensor,
                ctx.symm_workspaces,
                ctx.symm_barriers,
                ctx.ag_intranode_stream,
                all_gather_method=method,
                debug=debug,
            )
        else:
            # Row-tile barrier mode:
            # - Still uses full-mesh pull (P2P memcpy), but exposes "ready"
            #   at a finer granularity (row chunks) to the consumer GEMM.
            #
            # Implementation detail:
            # - We only need *local* ready flags (consumer waits on this rank).
            # - Because full-mesh pull writes into the local workspace, we can
            #   set a local chunk barrier right after each chunk copy_.
            # Chunk-major schedule: improves overlap because early row chunks
            # become ready quickly across all src_ranks.
            local_ws = ctx.symm_workspace
            rank_orders = [(ctx.rank + i) % ctx.num_ranks for i in range(ctx.num_ranks)]
            with torch.cuda.stream(ctx.ag_intranode_stream):
                for chunk_id in range(num_tile_chunks):
                    chunk_row_beg = chunk_id * tile_rows_per_chunk
                    chunk_row_end = min(chunk_row_beg + tile_rows_per_chunk, M_per_rank)
                    if chunk_row_beg >= chunk_row_end:
                        continue
                    for src_rank in rank_orders:
                        if src_rank == ctx.rank:
                            continue
                        src_ws = ctx.symm_workspaces[src_rank]
                        row_beg = src_rank * M_per_rank + chunk_row_beg
                        row_end = src_rank * M_per_rank + chunk_row_end
                        dst = local_ws[row_beg:row_end, :]
                        src = src_ws[row_beg:row_end, :]
                        dst.copy_(src)
                        _set_signal_cuda(tile_barrier[src_rank * num_tile_chunks + chunk_id], 1, ctx.ag_intranode_stream)

                # Preserve legacy rank-level readiness for any tiles that span
                # multiple ranks (or for fallbacks).
                for src_rank in range(ctx.num_ranks):
                    if src_rank == ctx.rank:
                        continue
                    _set_signal_cuda(ctx.symm_barrier[src_rank], 1, ctx.ag_intranode_stream)

    return ctx.ag_intranode_stream


def wait_rank_ready(
    ctx,
    peer_rank: int,
    *,
    value: int = 1,
    stream: Optional[torch.cuda.Stream] = None,
):
    if peer_rank < 0 or peer_rank >= ctx.num_ranks:
        raise ValueError(f"peer_rank out of range: {peer_rank}")
    _wait_eq_cuda(ctx.symm_barrier[peer_rank], value, stream or torch.cuda.current_stream())
