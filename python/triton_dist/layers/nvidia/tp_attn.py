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

import torch
from torch import nn
import torch.distributed
import inspect
import math

from triton_dist.kernels.allreduce import AllReduceMethod
from triton_dist.kernels.nvidia.allgather_gemm import AllGatherGEMMTensorParallelContext, get_auto_all_gather_method, ag_gemm
from triton_dist.kernels.nvidia import create_gemm_rs_context, gemm_rs
from triton_dist.kernels.nvidia.new_allgather_gemm import create_new_ag_gemm_context, new_ag_gemm
from triton_dist.kernels.nvidia.new_3rd_v5_frontier_windowed_panel_rsgemm import (
    create_new_3rd_v5_frontier_windowed_panel_gemm_rs_context,
    new_3rd_v5_frontier_windowed_panel_gemm_rs,
)
from triton_dist.kernels.nvidia.new_windowed_panel_gemm_allreduce import (
    create_frontier_windowed_panel_gemm_ar_context,
    frontier_windowed_panel_gemm_allreduce,
)
from triton_dist.utils import nvshmem_barrier_all_on_stream
from triton_dist.kernels.nvidia.allreduce import (create_allreduce_ctx, all_reduce)
from triton_dist.layers.nvidia import GemmARLayer

try:
    import flashinfer
    _HAS_FLASHINFER = True
except ImportError:
    flashinfer = None
    _HAS_FLASHINFER = False

try:
    from flash_attn_interface import flash_attn_with_kvcache
    msg = "Using flash_attn_interface, which is faster for sm90."
except ImportError:
    try:
        from flash_attn import flash_attn_with_kvcache
        msg = "Using flash_attn, which is much slower than flash_attn_interface for sm90"
        _HAS_FLASH_ATTN = True
    except ImportError:
        flash_attn_with_kvcache = None
        msg = "flash_attn not found, falling back to PyTorch attention."
        _HAS_FLASH_ATTN = False
else:
    _HAS_FLASH_ATTN = True
print(msg)


def shard_local(tensor: torch.Tensor, world_size: int, dim: int, local_rank: int):
    tensor_dim = tensor.shape[dim]
    tensor_slice = tensor_dim // world_size
    if tensor_dim % world_size != 0:
        raise ValueError(f"Tensor dimension {tensor_dim} is not divisible by world size {world_size}.")
    if local_rank < 0 or local_rank >= world_size:
        raise ValueError(f"Local rank {local_rank} is out of bounds for world size {world_size}.")
    if dim < 0 or dim >= tensor.dim():
        raise ValueError(f"Dimension {dim} is out of bounds for tensor with {tensor.dim()} dimensions.")
    if tensor_slice == 0:
        raise ValueError(f"Tensor slice size is zero for tensor dimension {tensor_dim} and world size {world_size}.")
    return tensor.split(tensor_slice, dim=dim)[local_rank].contiguous()


def layer_norm(
    hidden_states: torch.Tensor,
    eps: float,
    w: torch.Tensor,
):
    """Applies RMS normalization with a flashinfer fast path and a PyTorch fallback."""
    if _HAS_FLASHINFER:
        return flashinfer.norm.rmsnorm(hidden_states.view(-1, hidden_states.size(-1)), w, eps).view_as(hidden_states)

    x = hidden_states.to(torch.float32)
    variance = x.square().mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    return (x.to(hidden_states.dtype) * w).view_as(hidden_states)


def _set_cos_sin_cache(inv_freq: torch.Tensor, max_length: int):
    """Precomputes cosine and sine cache for rotary position embeddings."""
    t = torch.arange(max_length, device="cuda", dtype=inv_freq.dtype)
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos_sin_cache = torch.cat((emb.cos()[:, :64], emb.sin()[:, :64]), dim=-1)
    return cos_sin_cache


class TP_Attn:
    """
    Tensor Parallel Attention.
    QKV Projection: Column Parallelism on weights (sharded over head dimension).
    Output Projection: Row Parallelism on weights.
    """

    def __init__(self, rank=0, world_size=8, group=None):
        # TODO does not support multiple node
        self.rank = rank
        self.world_size = world_size
        self.group = group
        self.head_dim = 128
        self.wqkv = None
        self.wo = None
        self.ag_ctx = None
        self.new_ag_ctx = None
        self.rs_ctx = None
        self.new_rs_ctx = None
        self.ar_ctx = None
        self.gemm_ar_ctx = None
        self.new_gemm_ar_ctx = None

    def _init_parameters(self, self_attn: nn.Module, verbose=False):
        self.q_size = self_attn.q_proj.weight.shape[0] // self.world_size
        self.kv_size = self_attn.k_proj.weight.shape[0] // self.world_size
        wq = shard_local(self_attn.q_proj.weight.detach(), self.world_size, 0, self.rank)
        wk = shard_local(self_attn.k_proj.weight.detach(), self.world_size, 0, self.rank)
        wv = shard_local(self_attn.v_proj.weight.detach(), self.world_size, 0, self.rank)
        self.wqkv: torch.Tensor = torch.cat((wq, wk, wv), dim=0).to("cuda", non_blocking=True)  # [qkv_dim, hidden_size]
        self.wo = shard_local(self_attn.o_proj.weight.detach(), self.world_size, 1,
                              self.rank).to("cuda", non_blocking=True)

        self.ag_N_per_rank = self.wqkv.shape[0]
        self.K = self.wqkv.shape[1]
        self.dtype = self.wqkv.dtype

        if hasattr(self_attn, "q_norm"):
            self.q_norm_eps = self_attn.q_norm.variance_epsilon
            self.q_norm_w = self_attn.q_norm.weight.detach().to("cuda", non_blocking=True)
        if hasattr(self_attn, "k_norm"):
            self.k_norm_eps = self_attn.k_norm.variance_epsilon
            self.k_norm_w = self_attn.k_norm.weight.detach().to("cuda", non_blocking=True)

        # bias
        if self_attn.q_proj.bias is not None:
            bq = shard_local(self_attn.q_proj.bias.detach(), self.world_size, 0, self.rank)
            bk = shard_local(self_attn.k_proj.bias.detach(), self.world_size, 0, self.rank)
            bv = shard_local(self_attn.v_proj.bias.detach(), self.world_size, 0, self.rank)
            self.bqkv = torch.cat((bq, bk, bv), dim=0).to("cuda", non_blocking=True)  # [qkv_dim]

        if verbose:
            print(f"[RANK {self.rank}] Attn initialized with parameters: qkv ({self.wqkv.shape}, o ({self.wo.shape}))")

    def _init_ctx(self, max_M, ag_intranode_stream: torch.cuda.Stream | None,
                  ag_internode_stream: torch.cuda.Stream | None):
        self.ag_ctx = AllGatherGEMMTensorParallelContext(
            N_per_rank=self.ag_N_per_rank, K=self.K, dtype=self.dtype, rank=self.rank, num_ranks=self.world_size,
            num_local_ranks=self.world_size, max_M=max_M, ag_intranode_stream=ag_intranode_stream,
            ag_internode_stream=ag_internode_stream,
            all_gather_method=get_auto_all_gather_method(self.world_size, self.world_size))
        self.rs_ctx = create_gemm_rs_context(
            max_M=max_M,
            N=self.K,
            rank=self.rank,
            world_size=self.world_size,
            local_world_size=self.world_size,
            output_dtype=self.dtype,
            rs_stream=ag_intranode_stream,
        )
        nvshmem_barrier_all_on_stream(torch.cuda.current_stream())
        torch.cuda.synchronize()

    def _init_new_ag_ctx(self,
                         max_M,
                         ag_intranode_stream: torch.cuda.Stream | None = None,
                         ag_internode_stream: torch.cuda.Stream | None = None,
                         **ag_kwargs):
        supported = inspect.signature(create_new_ag_gemm_context).parameters
        ag_kwargs = {k: v for k, v in ag_kwargs.items() if k in supported}
        self.new_ag_ctx = create_new_ag_gemm_context(
            max_M=max_M,
            N=self.ag_N_per_rank * self.world_size,
            K=self.K,
            dtype=self.dtype,
            rank=self.rank,
            num_ranks=self.world_size,
            num_local_ranks=self.world_size,
            ag_intranode_stream=ag_intranode_stream,
            ag_internode_stream=ag_internode_stream,
            **ag_kwargs,
        )
        nvshmem_barrier_all_on_stream(torch.cuda.current_stream())
        torch.cuda.synchronize()

    def _init_new_rs_ctx(self, max_M, **rs_kwargs):
        supported = inspect.signature(create_new_3rd_v5_frontier_windowed_panel_gemm_rs_context).parameters
        rs_kwargs = {k: v for k, v in rs_kwargs.items() if k in supported}
        self.new_rs_ctx = create_new_3rd_v5_frontier_windowed_panel_gemm_rs_context(
            max_M=max_M,
            N=self.K,
            rank=self.rank,
            world_size=self.world_size,
            local_world_size=self.world_size,
            output_dtype=self.dtype,
            **rs_kwargs,
        )
        nvshmem_barrier_all_on_stream(torch.cuda.current_stream())
        torch.cuda.synchronize()

    def _init_AR_ctx(self, max_M, method: AllReduceMethod, dtype=torch.bfloat16):
        self.ar_method = method
        self.ar_ctx = create_allreduce_ctx(
            workspace_nbytes=max_M * self.K * dtype.itemsize, rank=self.rank, world_size=self.world_size,
            local_world_size=self.world_size,  # TODO(houqi.1993) does not support multiple nodes now.
        )

    def finalize(self):
        if self.ag_ctx:
            self.ag_ctx.finalize()
        if self.new_ag_ctx:
            self.new_ag_ctx.finalize()
        if self.rs_ctx:
            self.rs_ctx.finalize()
        if self.new_rs_ctx:
            self.new_rs_ctx.finalize()
        if self.ar_ctx:
            self.ar_ctx.finalize()
        if self.gemm_ar_ctx:
            self.gemm_ar_ctx.finalize()
        if self.new_gemm_ar_ctx:
            self.new_gemm_ar_ctx.finalize()

    def _apply_qkv_bias(self, qkv: torch.Tensor, bsz: int, q_len: int):
        if hasattr(self, 'bqkv'):
            qkv = qkv + self.bqkv.view(1, 1, -1).expand(bsz, q_len, -1)
        return qkv

    def _run_attention_core(self, qkv: torch.Tensor, bsz: int, q_len: int, position_ids, cos_sin_cache, kv_cache,
                            layer_idx: int):
        qkv = self._apply_qkv_bias(qkv, bsz, q_len)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        v = v.view(bsz, q_len, -1, self.head_dim)

        if hasattr(self, 'q_norm_eps'):
            q = layer_norm(q.contiguous().view(bsz, q_len, -1, self.head_dim), self.q_norm_eps,
                           self.q_norm_w).view(bsz, q_len, -1)
        if hasattr(self, 'k_norm_eps'):
            k = layer_norm(k.contiguous().view(bsz, q_len, -1, self.head_dim), self.k_norm_eps,
                           self.k_norm_w).view(bsz, q_len, -1)
        q, k = self.apply_rotary_pos_emb(q, k, position_ids, cos_sin_cache)
        k_cache, v_cache, kv_offset = kv_cache.update_kv_cache(k, v, layer_idx)
        return self._attention_with_kvcache(q, k_cache, v_cache, k, v, kv_offset, causal=True)

    def _attention_with_kvcache(self, q, k_cache, v_cache, k, v, kv_offset, causal=True):
        if _HAS_FLASH_ATTN:
            return flash_attn_with_kvcache(q=q,
                                           k_cache=k_cache,
                                           v_cache=v_cache,
                                           k=k,
                                           v=v,
                                           cache_seqlens=kv_offset,
                                           causal=causal)
        return self._torch_attention_with_kvcache(q, k_cache, v_cache, k, v, kv_offset, causal=causal)

    def _torch_attention_with_kvcache(self, q, k_cache, v_cache, k, v, kv_offset, causal=True):
        bsz, q_len, q_heads, head_dim = q.shape
        kv_heads = k.shape[2]
        scale = 1.0 / math.sqrt(head_dim)
        outputs = []

        for batch_idx in range(bsz):
            start = int(kv_offset[batch_idx].item())
            end = start + q_len
            k_cache[batch_idx, start:end].copy_(k[batch_idx])
            v_cache[batch_idx, start:end].copy_(v[batch_idx])

            k_all = k_cache[batch_idx, :end]
            v_all = v_cache[batch_idx, :end]
            if q_heads != kv_heads:
                if q_heads % kv_heads != 0:
                    raise ValueError(f"q_heads ({q_heads}) must be divisible by kv_heads ({kv_heads}).")
                repeat = q_heads // kv_heads
                k_all = k_all.repeat_interleave(repeat, dim=1)
                v_all = v_all.repeat_interleave(repeat, dim=1)

            q_b = q[batch_idx].permute(1, 0, 2).to(torch.float32)
            k_b = k_all.permute(1, 0, 2).to(torch.float32)
            v_b = v_all.permute(1, 0, 2).to(torch.float32)

            scores = torch.matmul(q_b, k_b.transpose(-2, -1)) * scale
            if causal:
                key_positions = torch.arange(end, device=q.device)
                query_positions = start + torch.arange(q_len, device=q.device)
                causal_mask = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
                scores = scores.masked_fill(~causal_mask.unsqueeze(0), float("-inf"))

            probs = torch.softmax(scores, dim=-1)
            out = torch.matmul(probs.to(v_b.dtype), v_b).permute(1, 0, 2)
            outputs.append(out.to(q.dtype))

        return torch.stack(outputs, dim=0)

    @torch.inference_mode()
    def apply_rotary_pos_emb(self, q: torch.Tensor, k: torch.Tensor, position_ids: torch.Tensor,
                             cos_sin_cache: torch.Tensor):
        """Applies Rotary Position Embedding inplace."""
        bsz, seq, _ = q.shape
        if _HAS_FLASHINFER:
            if cos_sin_cache.dtype != torch.float32:
                cos_sin_cache = cos_sin_cache.to(torch.float32)
            flashinfer.apply_rope_with_cos_sin_cache_inplace(position_ids.contiguous(), q.view(bsz * seq, -1),
                                                             k.view(bsz * seq, -1), self.head_dim, cos_sin_cache,
                                                             True),
            q = q.view(bsz, seq, -1, self.head_dim)
            k = k.view(bsz, seq, -1, self.head_dim)
            return q, k

        q = q.view(bsz, seq, -1, self.head_dim)
        k = k.view(bsz, seq, -1, self.head_dim)
        cache = cos_sin_cache[position_ids].to(torch.float32)
        half = self.head_dim // 2
        cos = cache[..., :half].unsqueeze(2)
        sin = cache[..., half:].unsqueeze(2)

        q1, q2 = q[..., :half].to(torch.float32), q[..., half:].to(torch.float32)
        k1, k2 = k[..., :half].to(torch.float32), k[..., half:].to(torch.float32)
        q = torch.cat((q1 * cos - q2 * sin, q2 * cos + q1 * sin), dim=-1).to(self.wqkv.dtype)
        k = torch.cat((k1 * cos - k2 * sin, k2 * cos + k1 * sin), dim=-1).to(self.wqkv.dtype)
        return q, k

    @torch.inference_mode()
    def torch_fwd(self, x, position_ids, cos_sin_cache, kv_cache, layer_idx: int):
        """
        Reference PyTorch forward pass for attention with Tensor Parallelism.
        Activations related to head dimensions are sharded. Final output is AllReduced.
        x: input tensor, shape [batch_size, q_len, hidden_size_in] (replicated on each rank)
        """
        bsz, q_len, _ = x.size()
        qkv = torch.nn.functional.linear(x, self.wqkv)
        if hasattr(self, 'bqkv'):
            qkv = qkv + self.bqkv.view(1, 1, -1).expand(bsz, q_len, -1)

        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        v = v.view(bsz, q_len, -1, self.head_dim)

        # qk norm
        if hasattr(self, 'q_norm_eps'):
            q = layer_norm(q.contiguous().view(bsz, q_len, -1, self.head_dim), self.q_norm_eps,
                           self.q_norm_w).view(bsz, q_len, -1)
        if hasattr(self, 'k_norm_eps'):
            k = layer_norm(k.contiguous().view(bsz, q_len, -1, self.head_dim), self.k_norm_eps,
                           self.k_norm_w).view(bsz, q_len, -1)
        # RoPE
        q, k = self.apply_rotary_pos_emb(q, k, position_ids, cos_sin_cache)
        k_cache, v_cache, kv_offset = kv_cache.update_kv_cache(k, v, layer_idx)

        # FlashAttn
        out = self._attention_with_kvcache(q, k_cache, v_cache, k, v, kv_offset, causal=True)

        out = torch.nn.functional.linear(out.view(bsz, q_len, -1), self.wo)
        if self.world_size > 1:
            torch.distributed.all_reduce(out, torch.distributed.ReduceOp.SUM, group=self.group)
        return out

    @torch.inference_mode()
    def dist_triton_fwd(self, x, position_ids, cos_sin_cache, kv_cache, layer_idx: int, autotune=True):
        """
        triton_dist forward pass.
        Input x is batch-sharded. Output is also batch-sharded.
        x: input tensor, shape [batch_size_per_rank, q_len, hidden_size_in]
        """
        return self.dist_triton_select_ag_rs_fwd(
            x,
            position_ids,
            cos_sin_cache,
            kv_cache,
            layer_idx,
            ag_impl="old",
            rs_impl="old",
            autotune=autotune,
        )

    @torch.inference_mode()
    def dist_triton_new_qkv_ag_gemm(self, x, autotune=False):
        assert self.new_ag_ctx is not None, "New AllGather-GEMM context is not initialized."
        bsz, q_len, d = x.size()
        qkv = new_ag_gemm(x.view(-1, d), self.wqkv.T, ctx=self.new_ag_ctx, autotune=autotune)
        return qkv.view(bsz * self.world_size, q_len, -1)

    @torch.inference_mode()
    def dist_triton_new_o_gemm_rs(self, x, autotune=False):
        assert self.new_rs_ctx is not None, "New GEMM-ReduceScatter context is not initialized."
        return new_3rd_v5_frontier_windowed_panel_gemm_rs(
            x,
            self.wo.T,
            self.new_rs_ctx,
            persistent=False,
            autotune=autotune,
        )

    @torch.inference_mode()
    def dist_triton_select_ag_rs_fwd(self,
                                     x,
                                     position_ids,
                                     cos_sin_cache,
                                     kv_cache,
                                     layer_idx: int,
                                     ag_impl="old",
                                     rs_impl="old",
                                     autotune=True):
        bsz, q_len, d = x.size()
        if ag_impl == "old":
            assert self.ag_ctx is not None, "AllGather-GEMM context is not initialized."
            qkv = ag_gemm(x.view(-1, d), self.wqkv.T, ctx=self.ag_ctx, autotune=autotune)
            qkv = qkv.view(bsz * self.world_size, q_len, -1)
        elif ag_impl == "new":
            qkv = self.dist_triton_new_qkv_ag_gemm(x, autotune=autotune)
        else:
            raise ValueError(f"Unsupported ag_impl: {ag_impl}")

        total_bsz = bsz * self.world_size
        out = self._run_attention_core(qkv, total_bsz, q_len, position_ids, cos_sin_cache, kv_cache, layer_idx)
        out = out.view(total_bsz * q_len, -1)

        if rs_impl == "old":
            assert self.rs_ctx is not None, "GEMM-ReduceScatter context is not initialized."
            out = gemm_rs(out, self.wo.T, self.rs_ctx, autotune=autotune)
        elif rs_impl == "new":
            out = self.dist_triton_new_o_gemm_rs(out, autotune=autotune)
        else:
            raise ValueError(f"Unsupported rs_impl: {rs_impl}")
        return out.view(bsz, q_len, -1)

    @torch.inference_mode()
    def dist_triton_AR_fwd(self, x, position_ids, cos_sin_cache, kv_cache, layer_idx: int):
        """
        triton_dist AR forward pass for attention with Tensor Parallelism.
        Activations related to head dimensions are sharded. Final output is AllReduced.
        x: input tensor, shape [batch_size, q_len, hidden_size_in] (replicated on each rank)
        """
        bsz, q_len, _ = x.size()
        qkv = torch.nn.functional.linear(x, self.wqkv)
        if hasattr(self, 'bqkv'):
            qkv = qkv + self.bqkv.view(1, 1, -1).expand(bsz, q_len, -1)

        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        v = v.view(bsz, q_len, -1, self.head_dim)

        # qk norm
        if hasattr(self, 'q_norm_eps'):
            q = layer_norm(q.contiguous().view(bsz, q_len, -1, self.head_dim), self.q_norm_eps,
                           self.q_norm_w).view(bsz, q_len, -1)
        if hasattr(self, 'k_norm_eps'):
            k = layer_norm(k.contiguous().view(bsz, q_len, -1, self.head_dim), self.k_norm_eps,
                           self.k_norm_w).view(bsz, q_len, -1)
        # RoPE
        q, k = self.apply_rotary_pos_emb(q, k, position_ids, cos_sin_cache)
        k_cache, v_cache, kv_offset = kv_cache.update_kv_cache(k, v, layer_idx)

        # FlashAttn
        out = self._attention_with_kvcache(q, k_cache, v_cache, k, v, kv_offset, causal=True)

        out = torch.nn.functional.linear(out.view(bsz, q_len, -1), self.wo).view(bsz * q_len, -1)
        if self.world_size > 1:
            out_allreduce = torch.empty_like(out)
            out = all_reduce(x=out.contiguous(), output=out_allreduce, method=self.ar_method, ctx=self.ar_ctx)
        return out.view(bsz, q_len, -1)

    def _init_gemm_ar_ctx(self, max_M, dtype=torch.bfloat16):
        N = self.wo.shape[0]
        K = self.wo.shape[1]
        self.gemm_ar_ctx = GemmARLayer(self.group, max_M, N, K, dtype, dtype, self.world_size, persistent=True,
                                       use_ll_kernel=max_M <= 256, copy_to_local=False,
                                       NUM_COMM_SMS=16 if max_M <= 256 else 4)

    def _init_new_gemm_ar_ctx(self, max_M, dtype=torch.bfloat16, **ar_kwargs):
        N = self.wo.shape[0]
        supported = inspect.signature(create_frontier_windowed_panel_gemm_ar_context).parameters
        ar_kwargs = {k: v for k, v in ar_kwargs.items() if k in supported}
        self.new_gemm_ar_ctx = create_frontier_windowed_panel_gemm_ar_context(
            max_M=max_M,
            N=N,
            rank=self.rank,
            world_size=self.world_size,
            local_world_size=self.world_size,
            output_dtype=dtype,
            **ar_kwargs,
        )
        nvshmem_barrier_all_on_stream(torch.cuda.current_stream())
        torch.cuda.synchronize()

    @torch.inference_mode()
    def dist_triton_gemm_ar_fwd(self, x, position_ids, cos_sin_cache, kv_cache, layer_idx: int):
        return self.dist_triton_select_gemm_ar_fwd(
            x,
            position_ids,
            cos_sin_cache,
            kv_cache,
            layer_idx,
            impl="old",
            autotune=False,
        )

    @torch.inference_mode()
    def dist_triton_new_gemm_ar_fwd(self, x, position_ids, cos_sin_cache, kv_cache, layer_idx: int, autotune=False):
        return self.dist_triton_select_gemm_ar_fwd(
            x,
            position_ids,
            cos_sin_cache,
            kv_cache,
            layer_idx,
            impl="new",
            autotune=autotune,
        )

    @torch.inference_mode()
    def dist_triton_select_gemm_ar_fwd(self,
                                       x,
                                       position_ids,
                                       cos_sin_cache,
                                       kv_cache,
                                       layer_idx: int,
                                       impl="old",
                                       autotune=False):
        bsz, q_len, _ = x.size()
        qkv = torch.nn.functional.linear(x, self.wqkv)
        out = self._run_attention_core(qkv, bsz, q_len, position_ids, cos_sin_cache, kv_cache, layer_idx)
        out = out.view(bsz * q_len, -1)
        if impl == "old":
            assert self.gemm_ar_ctx is not None, "GemmAR context is not initialized."
            out = self.gemm_ar_ctx.forward(out, self.wo)
        elif impl == "new":
            assert self.new_gemm_ar_ctx is not None, "New GEMM-AllReduce context is not initialized."
            out = frontier_windowed_panel_gemm_allreduce(
                out,
                self.wo.T,
                self.new_gemm_ar_ctx,
                drain=True,
                autotune=autotune,
            )
        else:
            raise ValueError(f"Unsupported GEMM-AR impl: {impl}")
        return out.view(bsz, q_len, -1)

    def fwd(self, x: torch.Tensor, position_ids: torch.Tensor, cos_sin_cache: torch.Tensor, kv_cache, layer_idx: int):
        raise NotImplementedError("Please use torch_fwd or dist_triton_fwd instead.")
