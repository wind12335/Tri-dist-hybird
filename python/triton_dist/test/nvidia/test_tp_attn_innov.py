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

from triton_dist.layers.nvidia.tp_attn import TP_Attn, _set_cos_sin_cache
from triton_dist.models.kv_cache import KV_Cache
from triton_dist.models.utils import init_model_cpu
from triton_dist.profiler_utils import group_profile
from triton_dist.utils import initialize_distributed, dist_print

from tp_ag_rs_innov_common import run_pair, write_ranked_json
# torchrun --nproc_per_node=4 python/triton_dist/test/nvidia/test_tp_attn_innov.py \
#     --model Qwen/Qwen2-72B \
#     --bsz 64 \
#     --seq_len 128 \
#     --run_type prefill \
#     --mode ag_rs_new \
#     --rs_chunk_rows 512 \
#     --rs_active_chunk_window 4 \
#     --rs_stage_slots 4 \
#     --rs_steady_sms 8 \
#     --rs_tail_sms 20 \
#     --rs_comm_lanes 2 \
#     --rs_n_bands 2 \
#     --rs_frontier_chunks 2 \
#     --rs_local_seed_direct

THRESHOLD_MAP = {
    torch.float16: 1e-2,
    torch.bfloat16: 1.25e-1,
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bsz", default=128, type=int, help="Batch size")
    parser.add_argument("--seq_len", default=128, type=int, help="Sequence length for prefill")
    parser.add_argument("--model", default="Qwen/Qwen3-32B", type=str, help="HuggingFace model name")
    parser.add_argument("--warmup", default=20, type=int, help="Warmup iterations")
    parser.add_argument("--iters", default=100, type=int, help="Performance iterations")
    parser.add_argument("--dtype", default="bfloat16", type=str, choices=list(DTYPE_MAP.keys()))
    parser.add_argument("--atol", type=float, default=None,
                        help="Override the dtype-specific correctness absolute tolerance.")
    parser.add_argument("--rtol", type=float, default=None,
                        help="Override the dtype-specific correctness relative tolerance.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--profile", default=False, action="store_true", help="Enable torch.profiler")
    parser.add_argument("--check", default=False, action="store_true",
                        help="Run correctness check and exit without performance timing.")
    parser.add_argument("--run_type", default="prefill", type=str, choices=["prefill", "decode"])
    parser.add_argument("--mode",
                        type=str,
                        default="ag_rs_new",
                        choices=list(AG_RS_MODES.keys()) + list(GEMM_AR_MODES.keys()))
    parser.add_argument("--autotune", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--cuda_graph", default=False, action=argparse.BooleanOptionalAction,
                        help="Use CUDA Graph replay; synchronized eager timing is the safe default.")
    parser.add_argument("--graph_warmup", type=int, default=3)
    parser.add_argument("--synchronize_each_iter", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--stability_repeats", type=int, default=0)
    parser.add_argument("--result", type=str, default=None,
                        help="JSON file or directory for per-rank and rank-0 summary results.")
    parser.add_argument("--repeat_id", type=int, default=0)

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


def rand_tensor(shape: list[int], dtype: torch.dtype):
    return torch.rand(shape, dtype=dtype).cuda() / 10


def prepare_kv_cache(kv_cache: KV_Cache, args: argparse.Namespace, seed_delta: int = 0):
    if args.run_type == "prefill":
        kv_cache.kv_offset.zero_()
        return
    torch.manual_seed(args.seed + 1001 + seed_delta)
    torch.cuda.manual_seed(args.seed + 1001 + seed_delta)
    kv_cache.kv_offset.fill_(args.seq_len)
    kv_cache.rand_fill_kv_cache(args.seq_len)


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
        raise AssertionError(f"Attention output mismatch: {metrics}, atol={atol}, rtol={rtol}")
    return metrics


def run_benchmark(
    test_name: str,
    torch_func,
    triton_func,
    args: argparse.Namespace,
    group,
    rank: int,
    world_size: int,
    before_each,
):
    with group_profile(f"tp_attn_innov_{test_name}", args.profile, group=group):
        performance, graph_state = run_pair(
            torch_func,
            triton_func,
            group=group,
            warmup=args.warmup,
            iters=args.iters,
            use_cuda_graph=args.cuda_graph,
            graph_warmup=args.graph_warmup,
            synchronize_each_iter=args.synchronize_each_iter,
            before_torch_each=before_each,
            before_selected_each=before_each,
        )

    dist_print(f"torch {test_name} #{rank}", performance["torch_local_ms"],
               need_sync=True, allowed_ranks=list(range(world_size)))
    dist_print(f"dist-triton {test_name} #{rank}", performance["selected_local_ms"],
               f"local={performance['local_speedup']:.2f}x rank-max={performance['rank_max_speedup']:.2f}x",
               need_sync=True, allowed_ranks=list(range(world_size)))
    if args.cuda_graph and not performance["cuda_graph_used"]:
        dist_print("CUDA Graph capture was unavailable; used eager timing.", need_sync=True, allowed_ranks=[0])
    del graph_state
    torch.cuda.empty_cache()
    return performance


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


def run_attention_test(attn: TP_Attn, cos_sin_cache, kv_cache: KV_Cache, args: argparse.Namespace, rank: int,
                       world_size: int, tp_group, dtype: torch.dtype, atol: float, rtol: float):
    if args.run_type == "prefill":
        seq_len = args.seq_len
        position_ids = torch.arange(0, seq_len, dtype=torch.int64, device="cuda").unsqueeze(0).expand(args.bsz, -1)
    else:
        seq_len = 1
        position_ids = torch.arange(args.seq_len, args.seq_len + 1, dtype=torch.int64,
                                    device="cuda").unsqueeze(0).expand(args.bsz, -1)

    x = rand_tensor([args.bsz, seq_len, attn.wqkv.shape[1]], dtype=dtype)
    M = args.bsz * seq_len
    bsz_per_rank = args.bsz // world_size

    torch_func = partial(attn.torch_fwd, x, position_ids, cos_sin_cache, kv_cache, layer_idx=0)
    prepare_kv_cache(kv_cache, args)
    golden_output = torch_func()

    if args.mode in AG_RS_MODES:
        ag_impl, rs_impl = AG_RS_MODES[args.mode]
        assert args.bsz % world_size == 0, f"Batch size {args.bsz} must be divisible by world size {world_size}."
        dist_x = x.split(bsz_per_rank, dim=0)[rank].contiguous()

        ag_intranode_stream = torch.cuda.Stream(priority=-1)
        ag_internode_stream = torch.cuda.Stream()
        if ag_impl == "old" or rs_impl == "old":
            attn._init_ctx(max_M=M, ag_intranode_stream=ag_intranode_stream, ag_internode_stream=ag_internode_stream)
        if ag_impl == "new":
            attn._init_new_ag_ctx(max_M=M,
                                  ag_intranode_stream=ag_intranode_stream,
                                  ag_internode_stream=ag_internode_stream,
                                  **build_ag_kwargs(args))
        if rs_impl == "new":
            attn._init_new_rs_ctx(max_M=M, **build_rs_kwargs(args))

        triton_func = partial(attn.dist_triton_select_ag_rs_fwd,
                              dist_x,
                              position_ids,
                              cos_sin_cache,
                              kv_cache,
                              0,
                              ag_impl=ag_impl,
                              rs_impl=rs_impl,
                              autotune=args.autotune)
        golden_for_assert = golden_output.split(bsz_per_rank, dim=0)[rank].contiguous()
        test_name = f"attn_{args.run_type}_{args.mode}"
    else:
        impl = GEMM_AR_MODES[args.mode]
        if impl == "old":
            attn._init_gemm_ar_ctx(M, dtype)
        else:
            attn._init_new_gemm_ar_ctx(M, dtype, **build_ar_kwargs(args))

        triton_func = partial(attn.dist_triton_select_gemm_ar_fwd,
                              x,
                              position_ids,
                              cos_sin_cache,
                              kv_cache,
                              0,
                              impl=impl,
                              autotune=args.autotune)
        golden_for_assert = golden_output
        test_name = f"attn_{args.run_type}_{args.mode}"

    prepare_kv_cache(kv_cache, args)
    triton_output = triton_func()
    correctness = validate_output(triton_output, golden_for_assert, atol=atol, rtol=rtol)

    stability = []
    for repeat in range(args.stability_repeats):
        repeat_x = (x.float() + (repeat + 1) * 1e-4).to(dtype)
        repeat_torch_func = partial(attn.torch_fwd, repeat_x, position_ids, cos_sin_cache, kv_cache, layer_idx=0)
        prepare_kv_cache(kv_cache, args, repeat)
        expected = repeat_torch_func()
        if args.mode in AG_RS_MODES:
            repeat_dist_x = repeat_x.split(bsz_per_rank, dim=0)[rank].contiguous()
            repeat_triton_func = partial(
                attn.dist_triton_select_ag_rs_fwd,
                repeat_dist_x,
                position_ids,
                cos_sin_cache,
                kv_cache,
                0,
                ag_impl=ag_impl,
                rs_impl=rs_impl,
                autotune=args.autotune,
            )
            expected = expected.split(bsz_per_rank, dim=0)[rank].contiguous()
        else:
            repeat_triton_func = partial(
                attn.dist_triton_select_gemm_ar_fwd,
                repeat_x,
                position_ids,
                cos_sin_cache,
                kv_cache,
                0,
                impl=impl,
                autotune=args.autotune,
            )
        prepare_kv_cache(kv_cache, args, repeat)
        actual = repeat_triton_func()
        metrics = validate_output(actual, expected, atol=atol, rtol=rtol)
        rank_max_abs = torch.tensor(metrics["max_abs_diff"], dtype=torch.float64, device="cuda")
        torch.distributed.all_reduce(rank_max_abs, op=torch.distributed.ReduceOp.MAX, group=tp_group)
        record = {"repeat": repeat, **metrics, "rank_max_abs_diff": float(rank_max_abs.item())}
        stability.append(record)
        dist_print(
            f"[Attention stability] repeat={repeat + 1}/{args.stability_repeats}, "
            f"rank_max_abs_diff={record['rank_max_abs_diff']:.8g}",
            need_sync=True,
            allowed_ranks=[0],
        )

    performance = None
    if args.check:
        dist_print(f"CORRECTNESS CHECK PASSED: {test_name}", need_sync=True, allowed_ranks=[0])
    else:
        performance = run_benchmark(
            test_name,
            torch_func,
            triton_func,
            args,
            tp_group,
            rank,
            world_size,
            partial(prepare_kv_cache, kv_cache, args),
        )
    return {
        "test_name": test_name,
        "correctness": correctness,
        "stability": stability,
        "performance": performance,
    }


if __name__ == "__main__":
    args = parse_args()
    if args.bsz <= 0 or args.seq_len <= 0 or args.iters <= 0:
        raise ValueError("--bsz, --seq_len, and --iters must be positive.")
    if args.warmup < 0 or args.graph_warmup < 0 or args.stability_repeats < 0:
        raise ValueError("Warmup and stability counts must be non-negative.")
    RANK = int(os.environ.get("RANK", 0))
    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
    if args.mode in AG_RS_MODES and args.bsz % WORLD_SIZE != 0:
        raise ValueError(f"--bsz must be divisible by world size {WORLD_SIZE} for AG/RS modes.")
    TP_GROUP = initialize_distributed()
    torch.manual_seed(args.seed)

    DTYPE = DTYPE_MAP[args.dtype]
    default_tol = THRESHOLD_MAP.get(DTYPE, 1e-2)
    ATOL = default_tol if args.atol is None else args.atol
    RTOL = default_tol if args.rtol is None else args.rtol

    hf_model = init_model_cpu(model_name=args.model, dtype=DTYPE)
    hf_attn = hf_model.model.layers[0].self_attn.eval().cuda()

    attn = TP_Attn(rank=RANK, world_size=WORLD_SIZE, group=TP_GROUP)
    attn._init_parameters(hf_attn, verbose=True)

    cos_sin_cache = _set_cos_sin_cache(hf_model.model.rotary_emb.inv_freq.cuda(), max_length=args.seq_len + 128)
    kv_cache = KV_Cache(
        num_layers=1,
        batch_size=args.bsz,
        max_length=args.seq_len + 128,
        kv_heads=hf_attn.config.num_key_value_heads,
        head_dim=hf_attn.head_dim,
        dtype=DTYPE,
        world_size=WORLD_SIZE,
    )

    dist_print(f"\n===== Running {args.run_type.capitalize()} Innov Test (Mode: {args.mode}) =====")
    result = run_attention_test(attn=attn,
                                cos_sin_cache=cos_sin_cache,
                                kv_cache=kv_cache,
                                args=args,
                                rank=RANK,
                                world_size=WORLD_SIZE,
                                tp_group=TP_GROUP,
                                dtype=DTYPE,
                                atol=ATOL,
                                rtol=RTOL)

    payload = {
        "script": "test_tp_attn_innov.py",
        "model": args.model,
        "bsz": args.bsz,
        "seq_len": args.seq_len,
        "run_type": args.run_type,
        "dtype": args.dtype,
        "mode": args.mode,
        "rank": RANK,
        "world_size": WORLD_SIZE,
        "repeat_id": args.repeat_id,
        "args": vars(args),
        **result,
    }
    write_ranked_json(
        args.result,
        payload,
        group=TP_GROUP,
        rank=RANK,
        world_size=WORLD_SIZE,
        stem=f"attn_{args.run_type}_{args.mode}_repeat_{args.repeat_id}",
    )

    attn.finalize()
    nvshmem.core.finalize()
    torch.distributed.destroy_process_group()
