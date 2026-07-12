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
from triton_dist.kernels.nvidia.gemm_reduce_scatter import update_triton_config
from triton_dist.kernels.nvidia.new_3rd_v2_windowed_panel_rs import (
    New3rdV2WindowedPanelRSContext,
    create_new_3rd_v2_windowed_panel_rs_context,
    new_3rd_v2_windowed_panel_rs_op,
)
from triton_dist.kernels.nvidia.new_3rdgemm_reducescatter import gemm_rs_producer_non_persistent_chunked
from triton_dist.utils import get_device_max_shared_memory_size, has_fullmesh_nvlink, nvshmem_barrier_all_on_stream


@dataclasses.dataclass
class New3rdV2WindowedPanelGEMMRSContext:
    rs_ctx: New3rdV2WindowedPanelRSContext
    output_dtype: torch.dtype
    gemm_out: torch.Tensor
    num_gemm_sms: int

    def finalize(self) -> None:
        self.rs_ctx.finalize()

    def get_gemm_out_buf(self, input: torch.Tensor) -> torch.Tensor:
        return self.gemm_out[:input.shape[0]]


def _band_col_range(ctx: New3rdV2WindowedPanelRSContext, band_id: int) -> tuple[int, int]:
    col_start = band_id * ctx.max_band_cols
    col_end = min(col_start + ctx.max_band_cols, ctx.N)
    return col_start, col_end


def _band_signal_slice(ctx: New3rdV2WindowedPanelRSContext, band_id: int) -> torch.Tensor:
    band_span = ctx.world_size * ctx.num_chunks
    start = band_id * band_span
    end = start + band_span
    return ctx.chunk_signal[start:end]


def launch_v2_panelized_gemm_producer(
    A: torch.Tensor,
    B: torch.Tensor,
    gemm_out: torch.Tensor,
    ctx: New3rdV2WindowedPanelGEMMRSContext,
    workspace: torch.Tensor,
    signal_value: int,
    gemm_config: triton.Config,
) -> None:
    world_size = ctx.rs_ctx.world_size
    local_world_size = ctx.rs_ctx.local_world_size
    M, local_K = A.shape

    if ctx.rs_ctx.n_bands == 1:
        tuned_config = update_triton_config(M, B.shape[1], local_K, A.dtype, world_size, local_world_size, gemm_config)
        if ctx.rs_ctx.chunk_rows < tuned_config.kwargs["BLOCK_SIZE_M"]:
            raise ValueError(
                "new_3rd_v2_windowed_panel_gemm_rs requires chunk_rows >= BLOCK_SIZE_M, "
                f"but got chunk_rows={ctx.rs_ctx.chunk_rows}, BLOCK_SIZE_M={tuned_config.kwargs['BLOCK_SIZE_M']}"
            )
        workspace.zero_()
        gemm_rs_producer_non_persistent_chunked(
            A,
            B,
            gemm_out,
            ctx.rs_ctx.chunk_signal,
            workspace,
            world_size,
            local_world_size,
            ctx.rs_ctx.num_chunks,
            ctx.rs_ctx.chunk_rows,
            signal_value,
            tuned_config,
        )
        return

    for band_id in range(ctx.rs_ctx.n_bands):
        col_start, col_end = _band_col_range(ctx.rs_ctx, band_id)
        if col_end <= col_start:
            continue
        band_B = B[:, col_start:col_end]
        band_out = gemm_out[:, col_start:col_end]
        band_signal = _band_signal_slice(ctx.rs_ctx, band_id)
        tuned_config = update_triton_config(M, band_B.shape[1], local_K, A.dtype, world_size, local_world_size,
                                            gemm_config)
        if ctx.rs_ctx.chunk_rows < tuned_config.kwargs["BLOCK_SIZE_M"]:
            raise ValueError(
                "new_3rd_v2_windowed_panel_gemm_rs requires chunk_rows >= BLOCK_SIZE_M, "
                f"but got chunk_rows={ctx.rs_ctx.chunk_rows}, BLOCK_SIZE_M={tuned_config.kwargs['BLOCK_SIZE_M']}"
            )
        workspace.zero_()
        gemm_rs_producer_non_persistent_chunked(
            A,
            band_B,
            band_out,
            band_signal,
            workspace,
            world_size,
            local_world_size,
            ctx.rs_ctx.num_chunks,
            ctx.rs_ctx.chunk_rows,
            signal_value,
            tuned_config,
        )


def create_new_3rd_v2_windowed_panel_gemm_rs_context(
    max_M: int,
    N: int,
    rank: int,
    world_size: int,
    local_world_size: int,
    output_dtype: torch.dtype,
    *,
    chunk_rows: int = 0,
    target_chunks_per_rank: int = 2,
    min_chunk_rows: int = 512,
    active_chunk_window: int = 2,
    comm_lanes: int = 2,
    n_bands: int = 1,
    steady_sms: int = 6,
    tail_sms: int = 12,
    stage_slots: int = 4,
    tail_chunk_window: int = 1,
    local_seed_direct: bool = True,
) -> New3rdV2WindowedPanelGEMMRSContext:
    rs_ctx = create_new_3rd_v2_windowed_panel_rs_context(
        max_M,
        N,
        rank,
        world_size,
        local_world_size,
        output_dtype,
        chunk_rows=chunk_rows,
        target_chunks_per_rank=target_chunks_per_rank,
        min_chunk_rows=min_chunk_rows,
        active_chunk_window=active_chunk_window,
        comm_lanes=comm_lanes,
        n_bands=n_bands,
        steady_sms=steady_sms,
        tail_sms=tail_sms,
        stage_slots=stage_slots,
        tail_chunk_window=tail_chunk_window,
        local_seed_direct=local_seed_direct,
    )
    num_sms = torch.cuda.get_device_properties("cuda").multi_processor_count
    gemm_out = torch.empty((max_M, N), dtype=output_dtype, device="cuda")
    ctx = New3rdV2WindowedPanelGEMMRSContext(
        rs_ctx=rs_ctx,
        output_dtype=output_dtype,
        gemm_out=gemm_out,
        num_gemm_sms=num_sms,
    )
    nvshmem_barrier_all_on_stream(torch.cuda.current_stream())
    return ctx


def new_3rd_v2_windowed_panel_key_fn(A, B, ctx: New3rdV2WindowedPanelGEMMRSContext, *args, **kwargs):
    return (
        triton_dist.tune.to_hashable(A),
        triton_dist.tune.to_hashable(B),
        ctx.rs_ctx.world_size,
        ctx.rs_ctx.local_world_size,
        ctx.rs_ctx.chunk_rows,
        ctx.rs_ctx.active_chunk_window,
        ctx.rs_ctx.n_bands,
        ctx.rs_ctx.stage_slots,
        len(ctx.rs_ctx.comm_streams),
        kwargs.get("persistent", True),
    )


def new_3rd_v2_windowed_panel_prune_fn(config, A, B, *args, **kwargs):
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


def get_new_3rd_v2_windowed_panel_gemm_rs_config_space():
    return [{"gemm_config": config} for config in get_config_space(False)]


def new_3rd_v2_windowed_panel_gemm_rs_op(
    A: torch.Tensor,
    B: torch.Tensor,
    ctx: New3rdV2WindowedPanelGEMMRSContext,
    gemm_config: triton.Config,
    persistent: bool = True,
) -> torch.Tensor:
    if ctx.rs_ctx.nnodes != 1:
        raise NotImplementedError("new_3rd_v2_windowed_panel_gemm_rs currently supports single-node only")
    if not has_fullmesh_nvlink():
        raise NotImplementedError("new_3rd_v2_windowed_panel_gemm_rs currently expects full-mesh NVLink")
    if persistent:
        raise NotImplementedError("new_3rd_v2_windowed_panel_gemm_rs currently supports only --no-persistent")

    world_size = ctx.rs_ctx.world_size
    local_world_size = ctx.rs_ctx.local_world_size
    M, local_K = A.shape
    _, N = B.shape
    assert B.shape == (local_K, ctx.rs_ctx.N), f"B should be of shape [{local_K}, {ctx.rs_ctx.N}]"
    assert M % world_size == 0, "M must be divisible by world_size"

    output = torch.empty((M // world_size, N), dtype=ctx.output_dtype, device=A.device)
    workspace = torch.zeros((world_size * ctx.rs_ctx.num_chunks,), dtype=torch.int32, device=A.device)
    gemm_out = ctx.get_gemm_out_buf(A)

    num_runtime_chunks = triton.cdiv(M // world_size, ctx.rs_ctx.chunk_rows)
    signal_value = ctx.rs_ctx.begin_round(num_runtime_chunks)
    launch_v2_panelized_gemm_producer(A, B, gemm_out, ctx, workspace, signal_value, gemm_config)
    return new_3rd_v2_windowed_panel_rs_op(gemm_out, ctx.rs_ctx, output, prepare_round=False)


@triton_dist.tune.autotune(
    config_space=get_new_3rd_v2_windowed_panel_gemm_rs_config_space(),
    key_fn=new_3rd_v2_windowed_panel_key_fn,
    prune_fn=new_3rd_v2_windowed_panel_prune_fn,
)
def new_3rd_v2_windowed_panel_gemm_rs(
    A: torch.Tensor,
    B: torch.Tensor,
    ctx: New3rdV2WindowedPanelGEMMRSContext,
    gemm_config: triton.Config,
    persistent: bool = True,
):
    return new_3rd_v2_windowed_panel_gemm_rs_op(A, B, ctx, gemm_config, persistent)


__all__ = [
    "New3rdV2WindowedPanelGEMMRSContext",
    "create_new_3rd_v2_windowed_panel_gemm_rs_context",
    "launch_v2_panelized_gemm_producer",
    "new_3rd_v2_windowed_panel_gemm_rs",
    "new_3rd_v2_windowed_panel_gemm_rs_op",
]
