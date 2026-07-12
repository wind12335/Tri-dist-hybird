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

import argparse
import os
from functools import partial

import nvshmem.core
import torch
import torch.distributed
import triton

from triton_dist.layers.nvidia.tp_mlp import TP_MLP
from triton_dist.models.utils import init_model_cpu
from triton_dist.profiler_utils import group_profile, perf_func
from triton_dist.test.utils import assert_allclose
from triton_dist.utils import initialize_distributed, dist_print, nvshmem_barrier_all_on_stream


THRESHOLD_MAP = {
    torch.float16: 1e-2,
    torch.bfloat16: 2e-2,
    torch.float8_e4m3fn: 2e-2,
    torch.float8_e5m2: 2e-2,
    torch.int8: 0,
    torch.int32: 0,
}

DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}

AG_RS_MODES = {
    "ag_rs_old": ("old", "old"),
    "ag_new_rs_old": ("new", "old"),
    "ag_old_rs_new": ("old", "new"),
    "ag_rs_new": ("new", "new"),
}

GEMM_AR_MODES = {
    "gemm_ar_old": "old",
    "gemm_ar_new": "new",
}


def rand_tensor(shape: list[int], dtype: torch.dtype):
    if dtype in [torch.int32, torch.int8]:
        return torch.randint(-127, 128, shape, dtype=dtype).cuda()
    return torch.rand(shape, dtype=dtype).cuda() / 10


def make_cuda_graph(mempool, func):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(30):
            func()
        s.synchronize()
    torch.cuda.current_stream().wait_stream(s)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, pool=mempool):
        func()
    return graph


def run_benchmark(test_name: str, torch_func, triton_func, args: argparse.Namespace, group, rank: int, world_size: int):
    mempool = torch.cuda.graph_pool_handle()
    torch_graph = make_cuda_graph(mempool, torch_func)
    triton_dist_graph = make_cuda_graph(mempool, triton_func)

    with group_profile(f"tp_mlp_innov_{test_name}", args.profile, group=group):
        torch.cuda.synchronize()
        _, torch_perf = perf_func(torch_graph.replay, iters=args.iters, warmup_iters=args.warmup)
        nvshmem_barrier_all_on_stream()
        torch.cuda.synchronize()

        torch.cuda.synchronize()
        _, dist_triton_perf = perf_func(triton_dist_graph.replay, iters=args.iters, warmup_iters=args.warmup)
        nvshmem_barrier_all_on_stream()
        torch.cuda.synchronize()

    dist_print(
        f"TP MLP innov {test_name} #{rank} torch {torch_perf:0.3f} ms/iter",
        f"dist-triton {dist_triton_perf:0.3f} ms/iter",
        f"speedup {torch_perf / dist_triton_perf:0.3f}x",
        need_sync=True,
        allowed_ranks=list(range(world_size)),
    )

    del torch_graph, triton_dist_graph, mempool
    torch.cuda.empty_cache()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--M", default=4096, type=int, help="M dimension of the input tensor")
    parser.add_argument("--model", default="Qwen/Qwen3-32B", type=str, help="HuggingFace model name")
    parser.add_argument("--warmup", default=20, type=int, help="warmup iterations")
    parser.add_argument("--iters", default=100, type=int, help="perf iterations")
    parser.add_argument("--dtype", default="bfloat16", type=str, help="data type", choices=list(DTYPE_MAP.keys()))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--profile", default=False, action="store_true", help="dump torch.profiler.profile")
    parser.add_argument("--autotune", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--mode",
                        type=str,
                        default="ag_rs_new",
                        choices=list(AG_RS_MODES.keys()) + list(GEMM_AR_MODES.keys()))

    parser.add_argument("--ag_copy_sms", type=int, default=0)
    parser.add_argument("--ag_enable_tile_ready", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--ag_tile_rows_per_chunk", type=int, default=0)
    parser.add_argument("--ag_min_m_per_rank_for_tile_ready", type=int, default=4096)
    parser.add_argument("--ag_target_chunks_per_rank", type=int, default=2)
    parser.add_argument("--ag_min_tile_rows_per_chunk", type=int, default=1024)

    parser.add_argument("--rs_chunk_rows", type=int, default=0)
    parser.add_argument("--rs_target_chunks_per_rank", type=int, default=2)
    parser.add_argument("--rs_min_chunk_rows", type=int, default=512)
    parser.add_argument("--rs_active_chunk_window", type=int, default=4)
    parser.add_argument("--rs_comm_lanes", type=int, default=2)
    parser.add_argument("--rs_n_bands", type=int, default=1)
    parser.add_argument("--rs_frontier_chunks", type=int, default=1)
    parser.add_argument("--rs_steady_sms", type=int, default=6)
    parser.add_argument("--rs_tail_sms", type=int, default=12)
    parser.add_argument("--rs_stage_slots", type=int, default=4)
    parser.add_argument("--rs_tail_chunk_window", type=int, default=1)
    parser.add_argument("--rs_local_seed_direct", default=True, action=argparse.BooleanOptionalAction)

    parser.add_argument("--ar_chunk_rows", type=int, default=0)
    parser.add_argument("--ar_target_chunks", type=int, default=4)
    parser.add_argument("--ar_min_chunk_rows", type=int, default=512)
    parser.add_argument("--ar_active_chunk_window", type=int, default=2)
    parser.add_argument("--ar_n_bands", type=int, default=1)
    parser.add_argument("--ar_frontier_chunks", type=int, default=1)
    parser.add_argument("--ar_stage_slots", type=int, default=4)
    parser.add_argument("--ar_num_comm_sms", type=int, default=16)
    parser.add_argument("--ar_comm_lanes", type=int, default=2)
    return parser.parse_args()


def build_ag_kwargs(args):
    return {
        "copy_sms": args.ag_copy_sms,
        "enable_row_tile_barrier": args.ag_enable_tile_ready,
        "tile_rows_per_chunk": args.ag_tile_rows_per_chunk,
        "min_m_per_rank_for_tile_ready": args.ag_min_m_per_rank_for_tile_ready,
        "target_chunks_per_rank": args.ag_target_chunks_per_rank,
        "min_tile_rows_per_chunk": args.ag_min_tile_rows_per_chunk,
    }


def build_rs_kwargs(args):
    return {
        "chunk_rows": args.rs_chunk_rows,
        "target_chunks_per_rank": args.rs_target_chunks_per_rank,
        "min_chunk_rows": args.rs_min_chunk_rows,
        "active_chunk_window": args.rs_active_chunk_window,
        "comm_lanes": args.rs_comm_lanes,
        "n_bands": args.rs_n_bands,
        "frontier_chunks": args.rs_frontier_chunks,
        "steady_sms": args.rs_steady_sms,
        "tail_sms": args.rs_tail_sms,
        "stage_slots": args.rs_stage_slots,
        "tail_chunk_window": args.rs_tail_chunk_window,
        "local_seed_direct": args.rs_local_seed_direct,
    }


def build_ar_kwargs(args):
    return {
        "chunk_rows": args.ar_chunk_rows,
        "target_chunks": args.ar_target_chunks,
        "min_chunk_rows": args.ar_min_chunk_rows,
        "active_chunk_window": args.ar_active_chunk_window,
        "n_bands": args.ar_n_bands,
        "frontier_chunks": args.ar_frontier_chunks,
        "stage_slots": args.ar_stage_slots,
        "num_comm_sms": args.ar_num_comm_sms,
        "comm_lanes": args.ar_comm_lanes,
    }


if __name__ == "__main__":
    args = parse_args()

    RANK = int(os.environ.get("RANK", 0))
    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
    TP_GROUP = initialize_distributed()

    DTYPE = DTYPE_MAP[args.dtype]
    ATOL = THRESHOLD_MAP[DTYPE]
    RTOL = THRESHOLD_MAP[DTYPE]
    torch.manual_seed(args.seed)

    hf_model = init_model_cpu(model_name=args.model, dtype=DTYPE)
    hf_mlp = hf_model.model.layers[0].mlp.eval().cuda()

    mlp = TP_MLP(rank=RANK, world_size=WORLD_SIZE, group=TP_GROUP)
    mlp._init_parameters(hf_mlp, verbose=True)
    x = rand_tensor([args.M, hf_mlp.gate_proj.weight.shape[1]], dtype=DTYPE)

    with torch.inference_mode():
        golden = hf_mlp(x)

    torch_out = mlp.torch_fwd(x)
    assert_allclose(torch_out, golden, atol=ATOL, rtol=RTOL)

    if args.mode in AG_RS_MODES:
        ag_impl, rs_impl = AG_RS_MODES[args.mode]
        assert args.M % WORLD_SIZE == 0
        M_per_rank = args.M // WORLD_SIZE
        x_triton = x.split(M_per_rank, dim=0)[RANK].contiguous()

        def alloc_fn(size: int, alignment: int, stream):
            return torch.empty(size, device="cuda", dtype=torch.int8)

        triton.set_allocator(alloc_fn)

        ag_intranode_stream = torch.cuda.Stream(priority=-1)
        ag_internode_stream = torch.cuda.Stream()
        if ag_impl == "old" or rs_impl == "old":
            mlp._init_ctx(max_M=args.M,
                          ag_intranode_stream=ag_intranode_stream,
                          ag_internode_stream=ag_internode_stream)
        if ag_impl == "new":
            mlp._init_new_ag_ctx(max_M=args.M,
                                 ag_intranode_stream=ag_intranode_stream,
                                 ag_internode_stream=ag_internode_stream,
                                 **build_ag_kwargs(args))
        if rs_impl == "new":
            mlp._init_new_rs_ctx(max_M=args.M, **build_rs_kwargs(args))

        triton_func = partial(mlp.dist_triton_select_ag_rs_fwd,
                              x_triton,
                              ag_impl=ag_impl,
                              rs_impl=rs_impl,
                              autotune=args.autotune)
        out_triton = triton_func()
        out_golden = golden.split(M_per_rank, dim=0)[RANK].contiguous()
        assert_allclose(out_triton, out_golden, atol=ATOL, rtol=RTOL)
        run_benchmark(args.mode, partial(mlp.torch_fwd, x), triton_func, args, TP_GROUP, RANK, WORLD_SIZE)
    else:
        impl = GEMM_AR_MODES[args.mode]
        if impl == "old":
            mlp._init_gemm_ar_ctx(max_M=args.M, dtype=DTYPE)
        else:
            mlp._init_new_gemm_ar_ctx(max_M=args.M, dtype=DTYPE, **build_ar_kwargs(args))

        triton_func = partial(mlp.dist_triton_select_gemm_ar_fwd, x, impl=impl, autotune=args.autotune)
        out_triton = triton_func()
        assert_allclose(out_triton, golden, atol=ATOL, rtol=RTOL)
        run_benchmark(args.mode, partial(mlp.torch_fwd, x), triton_func, args, TP_GROUP, RANK, WORLD_SIZE)

    mlp.finalize()
    nvshmem.core.finalize()
    torch.distributed.destroy_process_group(TP_GROUP)
