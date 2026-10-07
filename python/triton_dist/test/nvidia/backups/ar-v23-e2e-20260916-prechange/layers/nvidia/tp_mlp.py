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


class TP_MLP:
    """
    Tensor Parallel MLP.
    This MLP uses a common TP strategy:
    1. First linear layer (gate/up) weights are column-parallel.
    2. Second linear layer (down) weights are row-parallel.
    """

    def __init__(self, rank=0, world_size=8, group=None):
        self.rank = rank
        self.world_size = world_size
        self.group = group
        self.act_fn = None
        self.gate_up_proj = None
        self.down_proj = None
        self.ag_ctx = None
        self.new_ag_ctx = None
        self.rs_ctx = None
        self.new_rs_ctx = None
        self.ar_ctx = None
        self.gemm_ar_ctx = None
        self.new_gemm_ar_ctx = None

    def _init_parameters(self, mlp: nn.Module, verbose=False):
        """
        Initializes and shards MLP parameters for Tensor Parallelism.
        mlp: A standard nn.Module MLP (e.g., from HuggingFace Transformers).
             Expected to have mlp.gate_proj, mlp.up_proj, mlp.down_proj.
        """
        gate_proj: torch.Tensor = shard_local(mlp.gate_proj.weight.detach(), self.world_size, 0, self.rank)
        up_proj: torch.Tensor = shard_local(mlp.up_proj.weight.detach(), self.world_size, 0, self.rank)
        self.gate_up_proj: torch.Tensor = torch.cat(
            (gate_proj, up_proj), dim=0).to("cuda", non_blocking=True)  # [MLP_size * 2 // world_size, hidden_size]
        self.down_proj: torch.Tensor = shard_local(mlp.down_proj.weight.detach(), self.world_size, 1, self.rank).to(
            "cuda", non_blocking=True)  # [hidden_size, MLP_size // world_size]

        self.act_fn = mlp.act_fn
        self.ag_N_per_rank = self.gate_up_proj.shape[0]
        self.K = self.gate_up_proj.shape[1]
        self.dtype = self.gate_up_proj.dtype

        assert mlp.gate_proj.bias is None, "We do not support bias for now."

        if verbose:
            print(
                f"[RANK {self.rank}] MLP initialized with parameters: gate_up_proj shape: {self.gate_up_proj.shape}, down_proj shape: {self.down_proj.shape}"
            )

    def _init_ctx(self, max_M, ag_intranode_stream: torch.cuda.Stream | None = None,
                  ag_internode_stream: torch.cuda.Stream | None = None):
        """Initializes contexts for triton_dist AllGather-GEMM and GEMM-ReduceScatter operations."""
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

    @staticmethod
    def _flatten_input(x: torch.Tensor):
        if len(x.size()) == 3:
            bsz, seq, d = x.size()
            return x.view(-1, d), True, (bsz, seq)
        return x, False, None

    @staticmethod
    def _restore_output(x: torch.Tensor, is_3d_input: bool, shape):
        if is_3d_input:
            bsz, seq = shape
            return x.view(bsz, seq, -1)
        return x

    @torch.inference_mode()
    def torch_fwd(self, x):
        '''
        Reference PyTorch forward pass using Tensor Parallelism.
        Final output is AllReduced.
        x: input tensor, shape [batch_size * seq_len, hidden_size] or [batch_size, seq_len, hidden_size]
        '''
        out_fused = torch.nn.functional.linear(x, self.gate_up_proj)
        wg, w1 = torch.chunk(out_fused, 2, dim=-1)
        out = self.act_fn(wg) * w1
        out = torch.nn.functional.linear(out, self.down_proj)
        if self.world_size > 1:
            torch.distributed.all_reduce(out, torch.distributed.ReduceOp.SUM, group=self.group)
        return out

    @torch.inference_mode()
    def dist_triton_fwd(self, x: torch.Tensor, autotune=True):
        """
        triton_dist forward pass for TP.
        This version uses ag_gemm and gemm_rs.
        x: input tensor, shape [batch_size, seq_len, hidden_size] or [batch_size * seq_len, hidden_size]
        """
        return self.dist_triton_select_ag_rs_fwd(x, ag_impl="old", rs_impl="old", autotune=autotune)

    @torch.inference_mode()
    def dist_triton_new_ag_gemm(self, x: torch.Tensor, autotune=False):
        assert self.new_ag_ctx is not None, "New AllGather-GEMM context is not initialized."
        return new_ag_gemm(x, self.gate_up_proj.T, ctx=self.new_ag_ctx, autotune=autotune)

    @torch.inference_mode()
    def dist_triton_new_gemm_rs(self, x: torch.Tensor, autotune=False):
        assert self.new_rs_ctx is not None, "New GEMM-ReduceScatter context is not initialized."
        return new_3rd_v5_frontier_windowed_panel_gemm_rs(
            x,
            self.down_proj.T,
            self.new_rs_ctx,
            persistent=False,
            autotune=autotune,
        )

    @torch.inference_mode()
    def dist_triton_select_ag_rs_fwd(self, x: torch.Tensor, ag_impl="old", rs_impl="old", autotune=True):
        x, is_3d_input, shape = self._flatten_input(x)
        if ag_impl == "old":
            assert self.ag_ctx is not None, "AllGather-GEMM context is not initialized."
            out_fused = ag_gemm(x, self.gate_up_proj.T, ctx=self.ag_ctx, autotune=autotune)
        elif ag_impl == "new":
            out_fused = self.dist_triton_new_ag_gemm(x, autotune=autotune)
        else:
            raise ValueError(f"Unsupported ag_impl: {ag_impl}")

        wg, w1 = torch.chunk(out_fused, 2, dim=-1)
        out = self.act_fn(wg) * w1

        if rs_impl == "old":
            assert self.rs_ctx is not None, "GEMM-ReduceScatter context is not initialized."
            out = gemm_rs(out, self.down_proj.T, self.rs_ctx, autotune=autotune)
        elif rs_impl == "new":
            out = self.dist_triton_new_gemm_rs(out, autotune=autotune)
        else:
            raise ValueError(f"Unsupported rs_impl: {rs_impl}")

        return self._restore_output(out, is_3d_input, shape)

    def _init_AR_ctx(self, max_M, method: AllReduceMethod, dtype=torch.bfloat16):
        self.ar_method = method
        N = self.down_proj.shape[0]
        self.ar_ctx = create_allreduce_ctx(
            workspace_nbytes=max_M * N * dtype.itemsize, rank=self.rank, world_size=self.world_size,
            local_world_size=self.world_size,  # TODO(houqi.1993) does not support multiple nodes now.
        )

    @torch.inference_mode()
    def dist_triton_AR_fwd(self, x: torch.Tensor):
        """
        triton_dist AR forward pass for TP.
        This version uses gemm + gemm + AllReduce
        x: input tensor, shape [batch_size, seq_len, hidden_size] or [batch_size * seq_len, hidden_size]
        """
        out_fused = torch.nn.functional.linear(x, self.gate_up_proj)
        wg, w1 = torch.chunk(out_fused, 2, dim=-1)
        out = self.act_fn(wg) * w1
        out = torch.nn.functional.linear(out, self.down_proj).view_as(x)
        if self.world_size > 1:
            out_ar = torch.empty_like(out)
            assert self.ar_ctx is not None, "AllReduce context is not initialized."
            out = all_reduce(out.contiguous(), output=out_ar, method=self.ar_method, ctx=self.ar_ctx)
        return out.view_as(x)

    @torch.inference_mode()
    def fwd(self, x: torch.Tensor):
        raise NotImplementedError("Please use torch_fwd or dist_triton_fwd instead.")

    def _init_gemm_ar_ctx(self, max_M, dtype=torch.bfloat16):
        N = self.down_proj.shape[0]
        K = self.down_proj.shape[1]
        self.gemm_ar_ctx = GemmARLayer(self.group, max_M, N, K, dtype, dtype, self.world_size, persistent=True,
                                       use_ll_kernel=max_M <= 256, copy_to_local=False,
                                       NUM_COMM_SMS=16 if max_M <= 256 else 4)

    def _init_new_gemm_ar_ctx(self, max_M, dtype=torch.bfloat16, **ar_kwargs):
        N = self.down_proj.shape[0]
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
    def dist_triton_gemm_ar_fwd(self, x: torch.Tensor):
        return self.dist_triton_select_gemm_ar_fwd(x, impl="old")

    @torch.inference_mode()
    def dist_triton_new_gemm_ar_fwd(self, x: torch.Tensor, autotune=False):
        return self.dist_triton_select_gemm_ar_fwd(x, impl="new", autotune=autotune)

    @torch.inference_mode()
    def dist_triton_select_gemm_ar_fwd(self, x: torch.Tensor, impl="old", autotune=False):
        """
        Triton Dist forward pass using GEMM-AllReduce.
        This version uses gemm_ar.
        x: input tensor, shape [batch_size * seq_len, hidden_size]
        """
        x, is_3d_input, shape = self._flatten_input(x)
        out_fused = torch.nn.functional.linear(x, self.gate_up_proj)
        wg, w1 = torch.chunk(out_fused, 2, dim=-1)
        out = self.act_fn(wg) * w1
        if impl == "old":
            assert self.gemm_ar_ctx is not None, "GemmAR context is not initialized."
            out = self.gemm_ar_ctx.forward(out, self.down_proj)
        elif impl == "new":
            assert self.new_gemm_ar_ctx is not None, "New GEMM-AllReduce context is not initialized."
            out = frontier_windowed_panel_gemm_allreduce(
                out,
                self.down_proj.T,
                self.new_gemm_ar_ctx,
                drain=True,
                autotune=autotune,
            )
        else:
            raise ValueError(f"Unsupported GEMM-AR impl: {impl}")
        return self._restore_output(out, is_3d_input, shape)

    @torch.inference_mode()
    def torch_ag_gemm(self, x: torch.Tensor):
        """
        Reference PyTorch forward pass using AllGather-GEMM.
        """
        M_per_rank, K = x.shape
        M = M_per_rank * self.world_size
        ag_buffer = torch.empty([M, K], dtype=x.dtype, device="cuda")
        # ag
        torch.distributed.all_gather_into_tensor(ag_buffer, x, group=self.group)
        # gemm
        return torch.matmul(ag_buffer, self.gate_up_proj.T)

    @torch.inference_mode()
    def dist_triton_ag_gemm(self, x: torch.Tensor, autotune=False):
        """
        Triton Dist forward pass using AllGather-GEMM.
        This version uses ag_gemm.
        x: input tensor, shape [batch_size * seq_len, hidden_size]
        """
        assert self.ag_ctx is not None
        return ag_gemm(x, self.gate_up_proj.T, ctx=self.ag_ctx, autotune=autotune)

    @torch.inference_mode()
    def torch_gemm_rs(self, x: torch.Tensor):
        """
        Reference PyTorch forward pass using GEMM-ReduceScatter.
        """
        # x: [M, K]
        M, K = x.shape
        rs_buffer = torch.empty([M // self.world_size, self.down_proj.shape[0]], dtype=x.dtype, device="cuda")
        # gemm
        gemm_out = torch.matmul(x, self.down_proj.T)
        torch.distributed.reduce_scatter_tensor(rs_buffer, gemm_out, group=self.group)
        return rs_buffer

    @torch.inference_mode()
    def dist_triton_gemm_rs(self, x: torch.Tensor, autotune=True):
        """
        Triton Dist forward pass using GEMM-ReduceScatter.
        This version uses gemm_rs.
        x: input tensor, shape [batch_size * seq_len, hidden_size]
        """
        assert self.rs_ctx is not None
        return gemm_rs(x, self.down_proj.T, self.rs_ctx, autotune=autotune)
