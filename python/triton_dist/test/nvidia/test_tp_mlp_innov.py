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
from triton_dist.profiler_utils import group_profile
from triton_dist.test.utils import assert_allclose
from triton_dist.utils import initialize_distributed, dist_print

from tp_ag_rs_innov_common import run_pair, write_ranked_json


THRESHOLD_MAP = {
    torch.float16: 1e-2,
    torch.bfloat16: 2.5e-1,
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


def run_benchmark(test_name: str, torch_func, triton_func, args: argparse.Namespace, group, rank: int, world_size: int):
    with group_profile(f"tp_mlp_innov_{test_name}", args.profile, group=group):
        performance, graph_state = run_pair(
            torch_func,
            triton_func,
            group=group,
            warmup=args.warmup,
            iters=args.iters,
            use_cuda_graph=args.cuda_graph,
            graph_warmup=args.graph_warmup,
            synchronize_each_iter=args.synchronize_each_iter,
        )

    dist_print(
        f"TP MLP innov {test_name} #{rank} torch {performance['torch_local_ms']:0.3f} ms/iter",
        f"dist-triton {performance['selected_local_ms']:0.3f} ms/iter",
        f"local speedup {performance['local_speedup']:0.3f}x",
        f"rank-max speedup {performance['rank_max_speedup']:0.3f}x",
        need_sync=True,
        allowed_ranks=list(range(world_size)),
    )
    if args.cuda_graph and not performance["cuda_graph_used"]:
        dist_print("CUDA Graph capture was unavailable; used eager timing.", need_sync=True, allowed_ranks=[0])
    del graph_state
    torch.cuda.empty_cache()
    return performance


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--M", default=4096, type=int, help="M dimension of the input tensor")
    parser.add_argument("--model", default="Qwen/Qwen3-32B", type=str, help="HuggingFace model name")
    parser.add_argument("--warmup", default=20, type=int, help="warmup iterations")
    parser.add_argument("--iters", default=100, type=int, help="perf iterations")
    parser.add_argument("--dtype", default="bfloat16", type=str, help="data type", choices=list(DTYPE_MAP.keys()))
    parser.add_argument("--atol", type=float, default=None,
                        help="Override the dtype-specific correctness absolute tolerance.")
    parser.add_argument("--rtol", type=float, default=None,
                        help="Override the dtype-specific correctness relative tolerance.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--profile", default=False, action="store_true", help="dump torch.profiler.profile")
    parser.add_argument("--check", default=False, action="store_true",
                        help="Run correctness check and exit without performance timing.")
    parser.add_argument("--autotune", default=False, action=argparse.BooleanOptionalAction,
                        help="Disabled by default to match the stable operator-level AG/RS benchmarks.")
    parser.add_argument("--cuda_graph", default=False, action=argparse.BooleanOptionalAction,
                        help="Use CUDA Graph replay; synchronized eager timing is the safe default.")
    parser.add_argument("--graph_warmup", type=int, default=3)
    parser.add_argument("--synchronize_each_iter", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--stability_repeats", type=int, default=0)
    parser.add_argument("--result", type=str, default=None,
                        help="JSON file or directory for per-rank and rank-0 summary results.")
    parser.add_argument("--repeat_id", type=int, default=0)
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


def error_metrics(actual: torch.Tensor, expected: torch.Tensor):
    if actual.shape != expected.shape:
        raise AssertionError(f"Shape mismatch: actual={tuple(actual.shape)}, expected={tuple(expected.shape)}")
    diff = (actual.float() - expected.float()).abs()
    return {
        "max_abs_diff": float(diff.max().item()),
        "max_rel_diff": float((diff / expected.float().abs().clamp_min(1e-12)).max().item()),
        "finite": bool(torch.isfinite(actual).all() and torch.isfinite(expected).all()),
        "shape": list(actual.shape),
    }


def validate_output(actual: torch.Tensor, expected: torch.Tensor, *, atol: float, rtol: float):
    metrics = error_metrics(actual, expected)
    close = torch.isclose(actual.float(), expected.float(), atol=atol, rtol=rtol)
    metrics["passed"] = bool(close.all())
    metrics["mismatched"] = int((~close).sum().item())
    metrics["elements"] = close.numel()
    metrics["mismatch_fraction"] = metrics["mismatched"] / max(metrics["elements"], 1)
    if not metrics["passed"]:
        raise AssertionError(f"MLP output mismatch: {metrics}, atol={atol}, rtol={rtol}")
    return metrics


def run_stability(mlp, x, args, group, rank, world_size, dtype, atol, rtol, *, ag_impl=None, rs_impl=None, ar_impl=None):
    records = []
    for repeat in range(args.stability_repeats):
        repeat_x = (x.float() + (repeat + 1) * 1e-4).to(dtype)
        expected = mlp.torch_fwd(repeat_x)
        if ag_impl is not None:
            repeat_local = repeat_x.split(args.M // world_size, dim=0)[rank].contiguous()
            actual = mlp.dist_triton_select_ag_rs_fwd(
                repeat_local,
                ag_impl=ag_impl,
                rs_impl=rs_impl,
                autotune=args.autotune,
            )
            expected = expected.split(args.M // world_size, dim=0)[rank].contiguous()
        else:
            actual = mlp.dist_triton_select_gemm_ar_fwd(repeat_x, impl=ar_impl, autotune=args.autotune)
        metrics = validate_output(actual, expected, atol=atol, rtol=rtol)
        rank_max_abs = torch.tensor(metrics["max_abs_diff"], dtype=torch.float64, device="cuda")
        torch.distributed.all_reduce(rank_max_abs, op=torch.distributed.ReduceOp.MAX, group=group)
        record = {"repeat": repeat, **metrics, "rank_max_abs_diff": float(rank_max_abs.item())}
        records.append(record)
        dist_print(
            f"[MLP stability] repeat={repeat + 1}/{args.stability_repeats}, "
            f"rank_max_abs_diff={record['rank_max_abs_diff']:.8g}",
            need_sync=True,
            allowed_ranks=[0],
        )
    return records


if __name__ == "__main__":
    args = parse_args()

    if args.M <= 0 or args.iters <= 0 or args.warmup < 0 or args.graph_warmup < 0:
        raise ValueError("M and iters must be positive; warmup values must be non-negative.")
    if args.stability_repeats < 0:
        raise ValueError("--stability_repeats must be non-negative.")

    RANK = int(os.environ.get("RANK", 0))
    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
    TP_GROUP = initialize_distributed()

    DTYPE = DTYPE_MAP[args.dtype]
    default_tol = THRESHOLD_MAP[DTYPE]
    ATOL = default_tol if args.atol is None else args.atol
    RTOL = default_tol if args.rtol is None else args.rtol
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

    performance = None
    stability = []
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
        correctness = validate_output(out_triton, out_golden, atol=ATOL, rtol=RTOL)
        stability = run_stability(
            mlp,
            x,
            args,
            TP_GROUP,
            RANK,
            WORLD_SIZE,
            DTYPE,
            ATOL,
            RTOL,
            ag_impl=ag_impl,
            rs_impl=rs_impl,
        )
        if args.check:
            dist_print(f"CORRECTNESS CHECK PASSED: mlp_{args.mode}", need_sync=True, allowed_ranks=[0])
        else:
            performance = run_benchmark(
                args.mode,
                partial(mlp.torch_fwd, x),
                triton_func,
                args,
                TP_GROUP,
                RANK,
                WORLD_SIZE,
            )
    else:
        impl = GEMM_AR_MODES[args.mode]
        if impl == "old":
            mlp._init_gemm_ar_ctx(max_M=args.M, dtype=DTYPE)
        else:
            mlp._init_new_gemm_ar_ctx(max_M=args.M, dtype=DTYPE, **build_ar_kwargs(args))

        triton_func = partial(mlp.dist_triton_select_gemm_ar_fwd, x, impl=impl, autotune=args.autotune)
        out_triton = triton_func()
        correctness = validate_output(out_triton, golden, atol=ATOL, rtol=RTOL)
        stability = run_stability(
            mlp,
            x,
            args,
            TP_GROUP,
            RANK,
            WORLD_SIZE,
            DTYPE,
            ATOL,
            RTOL,
            ar_impl=impl,
        )
        if args.check:
            dist_print(f"CORRECTNESS CHECK PASSED: mlp_{args.mode}", need_sync=True, allowed_ranks=[0])
        else:
            performance = run_benchmark(
                args.mode,
                partial(mlp.torch_fwd, x),
                triton_func,
                args,
                TP_GROUP,
                RANK,
                WORLD_SIZE,
            )

    payload = {
        "script": "test_tp_mlp_innov.py",
        "model": args.model,
        "M": args.M,
        "dtype": args.dtype,
        "mode": args.mode,
        "rank": RANK,
        "world_size": WORLD_SIZE,
        "repeat_id": args.repeat_id,
        "args": vars(args),
        "correctness": correctness,
        "stability": stability,
        "performance": performance,
    }
    write_ranked_json(
        args.result,
        payload,
        group=TP_GROUP,
        rank=RANK,
        world_size=WORLD_SIZE,
        stem=f"mlp_{args.mode}_repeat_{args.repeat_id}",
    )

    mlp.finalize()
    nvshmem.core.finalize()
    torch.distributed.destroy_process_group(TP_GROUP)
