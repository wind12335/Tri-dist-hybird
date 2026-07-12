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
import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import torch

from triton_dist.kernels.nvidia.allgather import cp_engine_producer_all_gather_intra_node
from triton_dist.kernels.nvidia.allgather_gemm import local_copy_and_barrier_all


def _clamp(v, lo, hi):
    return max(lo, min(v, hi))


def _round_up(v, align):
    if align <= 1:
        return v
    return ((v + align - 1) // align) * align


def estimate_copy_and_barrier_ms(M_per_rank: int, K: int, dtype: torch.dtype, num_ranks: int,
                                 intra_node_bw_gbps: float) -> float:
    # Rough model: one local shard replicated to peers; used only for chunk size selection.
    nbytes = M_per_rank * K * dtype.itemsize * max(num_ranks - 1, 1)
    return nbytes / (intra_node_bw_gbps * 1e9) * 1e3


def estimate_gemm_ms(M: int, N: int, K: int, gemm_tflops: float) -> float:
    if gemm_tflops <= 0:
        return 0.0
    flops = 2.0 * M * N * K
    return flops / (gemm_tflops * 1e12) * 1e3


@dataclass
class ChunkKSchedule:
    chunks: List[Tuple[int, int]]
    prefetch_chunk_id: int
    remaining_chunk_order: List[int]

    @property
    def prefetch_chunk(self) -> Tuple[int, int]:
        return self.chunks[self.prefetch_chunk_id]

    @property
    def num_chunks(self) -> int:
        return len(self.chunks)

    def ordered_chunks(self) -> Sequence[Tuple[int, int]]:
        order = [self.prefetch_chunk_id] + self.remaining_chunk_order
        return [self.chunks[i] for i in order]


def build_dynamic_k_schedule(
    K: int,
    rank: int,
    num_ranks: int,
    M_per_rank: int,
    N_per_rank: int,
    dtype: torch.dtype,
    k_alignment: int = 64,
    min_prefetch_k: int = 128,
    max_k_chunks: int = 8,
    overlap_target: float = 0.85,
    intra_node_bw_gbps: float = 250.0,
    gemm_tflops: float = 120.0,
    reorder_policy: str = "rank_swizzle",
) -> ChunkKSchedule:
    if K <= 0:
        raise ValueError("K must be > 0")

    min_prefetch_k = _round_up(min_prefetch_k, k_alignment)
    copy_ms = estimate_copy_and_barrier_ms(M_per_rank, K, dtype, num_ranks, intra_node_bw_gbps)
    target_prefetch_ms = copy_ms * overlap_target
    ideal_prefetch_k = int(target_prefetch_ms * gemm_tflops * 1e9 / max(2.0 * M_per_rank * N_per_rank, 1.0))
    prefetch_k = _round_up(max(min_prefetch_k, ideal_prefetch_k), k_alignment)
    prefetch_k = _clamp(prefetch_k, min_prefetch_k, K)

    chunks: List[Tuple[int, int]] = []
    chunks.append((0, prefetch_k))
    tail = K - prefetch_k
    if tail > 0:
        tail_chunk_count = _clamp(math.ceil(tail / max(prefetch_k, 1)), 1, max(max_k_chunks - 1, 1))
        tail_chunk_size = _round_up(math.ceil(tail / tail_chunk_count), k_alignment)
        cur = prefetch_k
        while cur < K:
            nxt = min(K, cur + tail_chunk_size)
            chunks.append((cur, nxt))
            cur = nxt

    prefetch_chunk_id = 0
    remaining_ids = [i for i in range(len(chunks)) if i != prefetch_chunk_id]
    policy = (reorder_policy or "rank_swizzle").strip().lower().replace("-", "_")
    alias = {
        "inorder": "in_order",
        "largest": "largest_first",
        "swizzle": "rank_swizzle",
    }
    policy = alias.get(policy, policy)
    if policy == "rank_swizzle":
        if len(remaining_ids) > 1:
            shift = rank % len(remaining_ids)
            remaining_ids = remaining_ids[shift:] + remaining_ids[:shift]
    elif policy == "largest_first":
        remaining_ids = sorted(remaining_ids, key=lambda i: chunks[i][1] - chunks[i][0], reverse=True)
    elif policy == "in_order":
        pass
    else:
        raise ValueError(f"Unsupported reorder policy: {reorder_policy}")

    return ChunkKSchedule(
        chunks=chunks,
        prefetch_chunk_id=prefetch_chunk_id,
        remaining_chunk_order=remaining_ids,
    )


def launch_chunked_allgather_intra_node(
    local_tensor: torch.Tensor,
    ctx,
    *,
    debug: bool = False,
    use_cooperative: bool = True,
) -> torch.cuda.Stream:
    if ctx.is_multinode:
        raise NotImplementedError("chunked all-gather path is intra-node only in this implementation")

    current_stream = torch.cuda.current_stream()
    ctx.ag_intranode_stream.wait_stream(current_stream)
    with torch.cuda.stream(ctx.ag_intranode_stream):
        M_per_rank, K = local_tensor.shape
        local_copy_and_barrier_all(
            ctx.local_rank,
            ctx.rank,
            ctx.num_ranks,
            local_tensor,
            ctx.symm_workspace,
            ctx.symm_comm_buf,
            ctx.symm_barrier,
            M_per_rank,
            K,
            ctx.phase,
            is_internode=False,
            use_cooperative=use_cooperative,
        )
        ctx.phase += 2
        cp_engine_producer_all_gather_intra_node(
            ctx.rank,
            ctx.num_ranks,
            local_tensor,
            ctx.symm_workspaces,
            ctx.symm_barriers,
            ctx.ag_intranode_stream,
            all_gather_method=ctx.all_gather_method,
            debug=debug,
        )
    return ctx.ag_intranode_stream
