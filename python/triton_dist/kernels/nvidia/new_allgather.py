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
from triton_dist.kernels.nvidia.common_ops import _wait_eq_cuda
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
) -> torch.cuda.Stream:
    if getattr(ctx, "is_multinode", False):
        raise NotImplementedError("launch_new_allgather_intra_node is for intra-node only")

    current_stream = torch.cuda.current_stream()
    ctx.ag_intranode_stream.wait_stream(current_stream)

    method = all_gather_method
    if method == AllGatherMethod.Auto:
        method = get_auto_all_gather_method(ctx.num_ranks, ctx.num_local_ranks)

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

