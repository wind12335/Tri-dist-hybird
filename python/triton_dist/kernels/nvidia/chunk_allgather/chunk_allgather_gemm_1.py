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
from dataclasses import dataclass
from typing import Optional

import torch

from triton_dist.kernels.nvidia.allgather_gemm import AllGatherGEMMTensorParallelContext, create_ag_gemm_context
from triton_dist.kernels.nvidia.chunk_allgather import (ChunkKSchedule, build_dynamic_k_schedule,
                                                        launch_chunked_allgather_intra_node)


@dataclass
class ChunkAllGatherGEMMContext:
    base_ctx: AllGatherGEMMTensorParallelContext

    # K-slice policy
    k_alignment: int = 64
    min_prefetch_k: int = 128
    max_k_chunks: int = 8
    overlap_target: float = 0.85
    target_intra_bw_gbps: float = 250.0
    target_gemm_tflops: float = 120.0
    reorder_policy: str = "rank_swizzle"

    def finalize(self):
        self.base_ctx.finalize()


def create_chunk_ag_gemm_context(
    max_M: int,
    N: int,
    K: int,
    dtype: torch.dtype,
    rank: int,
    num_ranks: int,
    num_local_ranks: int = 8,
    ag_intranode_stream: Optional[torch.cuda.Stream] = None,
    ag_internode_stream: Optional[torch.cuda.Stream] = None,
    k_alignment: int = 64,
    min_prefetch_k: int = 128,
    max_k_chunks: int = 8,
    overlap_target: float = 0.85,
    target_intra_bw_gbps: float = 250.0,
    target_gemm_tflops: float = 120.0,
    reorder_policy: str = "rank_swizzle",
) -> ChunkAllGatherGEMMContext:
    base_ctx = create_ag_gemm_context(
        max_M,
        N,
        K,
        dtype,
        rank,
        num_ranks,
        num_local_ranks=num_local_ranks,
        ag_intranode_stream=ag_intranode_stream,
        ag_internode_stream=ag_internode_stream,
    )
    return ChunkAllGatherGEMMContext(
        base_ctx=base_ctx,
        k_alignment=k_alignment,
        min_prefetch_k=min_prefetch_k,
        max_k_chunks=max_k_chunks,
        overlap_target=overlap_target,
        target_intra_bw_gbps=target_intra_bw_gbps,
        target_gemm_tflops=target_gemm_tflops,
        reorder_policy=reorder_policy,
    )


def _accumulate_chunk(c: torch.Tensor, a: torch.Tensor, b: torch.Tensor, ks: int, ke: int):
    if ke <= ks:
        return
    c += torch.matmul(a[:, ks:ke], b[ks:ke, :])


def chunk_ag_gemm(
    A: torch.Tensor,  # [M_per_rank, K]
    B: torch.Tensor,  # [K, N_per_rank]
    ctx: ChunkAllGatherGEMMContext,
    schedule: Optional[ChunkKSchedule] = None,
    debug: bool = False,
    use_cooperative: bool = True,
) -> torch.Tensor:
    base = ctx.base_ctx
    if base.is_multinode:
        raise NotImplementedError("chunk_ag_gemm currently focuses on intra-node only")

    M_per_rank, K = A.shape
    if B.shape != (K, base.N_per_rank):
        raise ValueError(f"B should be [{K}, {base.N_per_rank}], but got {list(B.shape)}")
    if M_per_rank * base.num_ranks > base.max_M:
        raise ValueError(f"M exceeds context max_M: M={M_per_rank * base.num_ranks}, max_M={base.max_M}")
    if A.dtype != B.dtype or A.dtype != base.dtype:
        raise ValueError(f"dtype mismatch: A={A.dtype}, B={B.dtype}, ctx={base.dtype}")

    M = M_per_rank * base.num_ranks
    N_per_rank = base.N_per_rank
    C = torch.zeros((M, N_per_rank), dtype=A.dtype, device=A.device)

    if schedule is None:
        schedule = build_dynamic_k_schedule(
            K=K,
            rank=base.rank,
            num_ranks=base.num_ranks,
            M_per_rank=M_per_rank,
            N_per_rank=N_per_rank,
            dtype=A.dtype,
            k_alignment=ctx.k_alignment,
            min_prefetch_k=ctx.min_prefetch_k,
            max_k_chunks=ctx.max_k_chunks,
            overlap_target=ctx.overlap_target,
            intra_node_bw_gbps=ctx.target_intra_bw_gbps,
            gemm_tflops=ctx.target_gemm_tflops,
            reorder_policy=ctx.reorder_policy,
        )

    # 1) Launch copy+barrier+all-gather on comm stream.
    ag_stream = launch_chunked_allgather_intra_node(A, base, debug=debug, use_cooperative=use_cooperative)

    # 2) Local prefetch chunk compute while communication stream is active.
    prefetch_ks, prefetch_ke = schedule.prefetch_chunk
    local_m_start = base.rank * M_per_rank
    local_m_end = local_m_start + M_per_rank
    _accumulate_chunk(
        C[local_m_start:local_m_end, :],
        A,
        B,
        prefetch_ks,
        prefetch_ke,
    )

    # 3) Wait AG complete, then consume gathered A.
    current_stream = torch.cuda.current_stream()
    current_stream.wait_stream(ag_stream)
    A_full = base.symm_workspace[:M, :K]

    # 4) Finish prefetch chunk on remote rows only (local rows already done).
    if prefetch_ke > prefetch_ks:
        if local_m_start > 0:
            _accumulate_chunk(C[:local_m_start, :], A_full[:local_m_start, :], B, prefetch_ks, prefetch_ke)
        if local_m_end < M:
            _accumulate_chunk(C[local_m_end:, :], A_full[local_m_end:, :], B, prefetch_ks, prefetch_ke)

    # 5) Compute remaining K chunks over all rows. Order can be swizzled by policy.
    for chunk_id in schedule.remaining_chunk_order:
        ks, ke = schedule.chunks[chunk_id]
        _accumulate_chunk(C, A_full, B, ks, ke)

    return C
