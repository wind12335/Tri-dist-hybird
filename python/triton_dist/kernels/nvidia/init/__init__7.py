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
from .allgather import get_auto_all_gather_method, AllGatherMethod, cp_engine_producer_all_gather_intra_node, cp_engine_producer_all_gather_inter_node
from .allgather_gemm import ag_gemm, create_ag_gemm_context, gemm_persistent, gemm_non_persistent
from .low_latency_allgather import (fast_allgather, create_fast_allgather_context, _forward_pull_kernel,
                                    _forward_push_2d_kernel, _forward_push_3d_kernel, _forward_push_2d_ll_kernel,
                                    _forward_push_2d_ll_multimem_kernel, _forward_push_numa_2d_ll_kernel,
                                    _forward_push_numa_2d_kernel, _forward_push_numa_2d_ll_multinode_kernel)
from .allgather_group_gemm import (create_ag_group_gemm_context, ag_group_gemm)
from .flash_decode import (gqa_fwd_batch_decode_persistent, kernel_gqa_fwd_batch_decode_split_kv_persistent,
                           gqa_fwd_batch_decode_persistent_aot, gqa_fwd_batch_decode, gqa_fwd_batch_decode_aot,
                           gqa_fwd_batch_decode_intra_rank_aot, get_triton_combine_kv_algo_info,
                           gqa_fwd_batch_decode_intra_rank, kernel_inter_rank_gqa_fwd_batch_decode_combine_kv)
from .gemm_reduce_scatter import create_gemm_rs_context, gemm_rs
from .low_latency_all_to_all import create_all_to_all_context, fast_all_to_all, all_to_all_post_process
from .moe_reduce_rs import create_moe_rs_context
from .sp_ag_attention_intra_node import fused_sp_ag_attn_intra_node, create_sp_ag_attention_context_intra_node
from .sp_ag_attention_inter_node import fused_sp_ag_attn_inter_node, create_sp_ag_attention_context_inter_node
from .gemm_allreduce import create_gemm_ar_context, low_latency_gemm_allreduce_op, create_ll_gemm_ar_context, gemm_allreduce_op
from .all_to_all_single_2d import create_all_to_all_single_2d_context, all_to_all_single_2d
from .all_to_all_single_gemm import create_all_to_all_single_gemm_context, all_to_all_single_gemm
from .gdn import chunk_gated_delta_rule_fwd
from .low_latency_all_to_all_v2 import create_ep_ll_a2a_ctx, dispatch_kernel_v2, combine_kernel_v2
from .ulysses_sp_dispatch import create_ulysses_sp_pre_attn_comm_context
from .new_allgather import launch_new_allgather_intra_node, wait_rank_ready
from .new_allgather_gemm import create_new_ag_gemm_context, new_ag_gemm

try:
    from .new_reducescatter import create_new_reducescatter_2d_ctx, new_reduce_scatter_2d_op
    from .new_gemm_reducescatter import create_new_gemm_rs_context, new_gemm_rs
except ModuleNotFoundError:
    pass

try:
    from .new_3rdreducescatter import create_new_3rd_reducescatter_2d_ctx, new_3rd_reduce_scatter_2d_op
    from .new_3rdgemm_reducescatter import create_new_3rd_gemm_rs_context, new_3rd_gemm_rs
except ModuleNotFoundError:
    pass

try:
    from .new_3rd_v2_windowed_panel_rs import create_new_3rd_v2_windowed_panel_rs_context, new_3rd_v2_windowed_panel_rs_op
    from .new_3rd_v2_windowed_panel_rsgemm import (create_new_3rd_v2_windowed_panel_gemm_rs_context,
                                                   new_3rd_v2_windowed_panel_gemm_rs)
except ModuleNotFoundError:
    pass

try:
    from .fuse_overlap_reducescatter import create_fuse_overlap_reducescatter_2d_ctx, fuse_overlap_reduce_scatter_2d_op
    from .fuse_overlap_gemm_reducescatter import create_fuse_overlap_gemm_rs_context, fuse_overlap_gemm_rs
except ModuleNotFoundError:
    pass

__all__ = [
    "_forward_pull_kernel",
    "_forward_push_2d_kernel",
    "_forward_push_3d_kernel",
    "_forward_push_2d_ll_kernel",
    "_forward_push_2d_ll_multimem_kernel",
    "_forward_push_numa_2d_kernel",
    "_forward_push_numa_2d_ll_kernel",
    "_forward_push_numa_2d_ll_multinode_kernel",
    "ag_gemm",
    "ag_group_gemm",
    "all_to_all_post_process",
    "create_ag_gemm_context",
    "create_ag_group_gemm_context",
    "create_all_to_all_context",
    "create_fast_allgather_context",
    "create_gemm_rs_context",
    "create_moe_rs_context",
    "fast_all_to_all",
    "fast_allgather",
    "get_auto_all_gather_method",
    "AllGatherMethod",
    "cp_engine_producer_all_gather_intra_node",
    "cp_engine_producer_all_gather_inter_node",
    "gemm_rs",
    "gemm_persistent",
    "gemm_non_persistent",
    "get_triton_combine_kv_algo_info",
    "gqa_fwd_batch_decode_aot",
    "gqa_fwd_batch_decode_intra_rank_aot",
    "gqa_fwd_batch_decode_intra_rank",
    "gqa_fwd_batch_decode_persistent_aot",
    "gqa_fwd_batch_decode_persistent",
    "gqa_fwd_batch_decode",
    "kernel_gqa_fwd_batch_decode_split_kv_persistent",
    "kernel_inter_rank_gqa_fwd_batch_decode_combine_kv",
    "fused_sp_ag_attn_intra_node",
    "create_sp_ag_attention_context_intra_node",
    "fused_sp_ag_attn_inter_node",
    "create_sp_ag_attention_context_inter_node",
    "create_gemm_ar_context",
    "low_latency_gemm_allreduce_op",
    "create_ll_gemm_ar_context",
    "gemm_allreduce_op",
    "create_all_to_all_single_2d_context",
    "all_to_all_single_2d",
    "create_all_to_all_single_gemm_context",
    "all_to_all_single_gemm",
    "chunk_gated_delta_rule_fwd",
    "create_ep_ll_a2a_ctx",
    "dispatch_kernel_v2",
    "combine_kernel_v2",
    "create_ulysses_sp_pre_attn_comm_context",
    "ChunkKSchedule",
    "build_dynamic_k_schedule",
    "launch_chunked_allgather_intra_node",
    "create_chunk_ag_gemm_context",
    "chunk_ag_gemm",
    "launch_new_allgather_intra_node",
    "wait_rank_ready",
    "create_new_ag_gemm_context",
    "new_ag_gemm",
    "create_new_reducescatter_2d_ctx",
    "new_reduce_scatter_2d_op",
    "create_new_gemm_rs_context",
    "new_gemm_rs",
    "create_new_3rd_reducescatter_2d_ctx",
    "new_3rd_reduce_scatter_2d_op",
    "create_new_3rd_gemm_rs_context",
    "new_3rd_gemm_rs",
    "create_new_3rd_v2_windowed_panel_rs_context",
    "new_3rd_v2_windowed_panel_rs_op",
    "create_new_3rd_v2_windowed_panel_gemm_rs_context",
    "new_3rd_v2_windowed_panel_gemm_rs",
    "create_fuse_overlap_reducescatter_2d_ctx",
    "fuse_overlap_reduce_scatter_2d_op",
    "create_fuse_overlap_gemm_rs_context",
    "fuse_overlap_gemm_rs",
]
