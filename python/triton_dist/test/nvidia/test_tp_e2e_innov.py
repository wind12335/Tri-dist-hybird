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
import gc
import os

import torch
from functools import partial

from triton_dist.models.kv_cache import KV_Cache
from triton_dist.models.utils import seed_everything
from triton_dist.profiler_utils import group_profile, perf_func
from triton_dist.test.nvidia.synthetic_tp_dense import (
    SyntheticDenseLLM,
    format_synthetic_spec,
    get_synthetic_preset_names,
    resolve_synthetic_spec,
)
from triton_dist.utils import finalize_distributed, initialize_distributed, dist_print, nvshmem_barrier_all_on_stream

# torchrun --nproc_per_node=4 python/triton_dist/test/nvidia/test_tp_e2e_innov.py \
#     --synthetic \
#     --synthetic_config Qwen2-72B \
#     --synthetic_num_layers 2 \
#     --bsz 4 \
#     --seq_len 128 \
#     --run_type prefill \
#     --mode ag_rs_new \
#     --check

RANK = int(os.environ.get("RANK", 0))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))

THRESHOLD_MAP = {
    torch.float16: 1e-2,
    torch.bfloat16: 2e-2,
}

DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}

AG_RS_MODES = {
    "ag_rs_old": ("old", "old", "triton_dist_ag_rs_old"),
    "ag_new_rs_old": ("new", "old", "triton_dist_ag_new_rs_old"),
    "ag_old_rs_new": ("old", "new", "triton_dist_ag_old_rs_new"),
    "ag_rs_new": ("new", "new", "triton_dist_ag_rs_new"),
}

GEMM_AR_MODES = {
    "gemm_ar_old": ("old", "triton_dist_gemm_ar_old"),
    "gemm_ar_new": ("new", "triton_dist_gemm_ar_new"),
}


def validate_runtime_args(args):
    if args.bsz <= 0:
        raise ValueError(f"--bsz must be positive, got {args.bsz}.")

    if args.mode in AG_RS_MODES:
        if args.bsz < WORLD_SIZE:
            raise ValueError(f"--mode {args.mode} requires --bsz >= WORLD_SIZE ({WORLD_SIZE}), got {args.bsz}.")
        if args.bsz % WORLD_SIZE != 0:
            raise ValueError(
                f"--mode {args.mode} requires --bsz divisible by WORLD_SIZE ({WORLD_SIZE}), got {args.bsz}.")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bsz", default=128, type=int, help="Batch size")
    parser.add_argument("--seq_len", default=128, type=int, help="Sequence length for prefill")
    parser.add_argument("--model", default="Qwen/Qwen3-32B", type=str, help="HuggingFace model name")
    parser.add_argument("--warmup", default=10, type=int, help="Warmup iterations")
    parser.add_argument("--iters", default=20, type=int, help="Performance iterations")
    parser.add_argument("--dtype", default="bfloat16", type=str, choices=list(DTYPE_MAP.keys()))
    parser.add_argument("--atol", type=float, default=None, help="Override correctness absolute tolerance.")
    parser.add_argument("--rtol", type=float, default=None, help="Override correctness relative tolerance.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--profile", default=False, action="store_true", help="Enable torch.profiler")
    parser.add_argument("--check", default=False, action="store_true", help="Run correctness check and exit")
    parser.add_argument("--run_type", default="prefill", type=str, choices=["prefill", "decode"])
    parser.add_argument("--mode",
                        default="ag_rs_new",
                        type=str,
                        choices=list(AG_RS_MODES.keys()) + list(GEMM_AR_MODES.keys()))
    parser.add_argument("--autotune", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--synthetic",
                        default=False,
                        action="store_true",
                        help="Use a lightweight synthetic TP dense model instead of loading HuggingFace weights.")
    parser.add_argument("--synthetic_config",
                        type=str,
                        default=None,
                        choices=get_synthetic_preset_names(),
                        help="Preset synthetic model shape. Can be overridden by the synthetic_* flags below.")
    parser.add_argument("--synthetic_hidden_size", type=int, default=None)
    parser.add_argument("--synthetic_intermediate_size", type=int, default=None)
    parser.add_argument("--synthetic_num_layers", type=int, default=None)
    parser.add_argument("--synthetic_num_heads", type=int, default=None)
    parser.add_argument("--synthetic_num_kv_heads", type=int, default=None)
    parser.add_argument("--synthetic_head_dim", type=int, default=None)
    parser.add_argument("--synthetic_vocab_size", type=int, default=4096)
    parser.add_argument("--synthetic_max_length", type=int, default=0)
    parser.add_argument("--synthetic_rope_theta", type=float, default=None)
    parser.add_argument("--synthetic_norm_eps", type=float, default=1e-5)
    parser.add_argument("--synthetic_init_std", type=float, default=0.02)
    parser.add_argument("--synthetic_share_weights",
                        default=True,
                        action=argparse.BooleanOptionalAction,
                        help="Reuse one physical TP layer across all virtual transformer layers to save memory.")
    parser.add_argument("--synthetic_tie_word_embeddings",
                        default=True,
                        action=argparse.BooleanOptionalAction,
                        help="Tie embedding and LM head weights in synthetic mode to reduce memory.")

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


def check_allclose(out: torch.Tensor, golden: torch.Tensor, atol=1e-3, rtol=1e-3, mode_name=""):
    assert out.shape == golden.shape, f"Shape mismatch for {mode_name}: {out.shape} vs {golden.shape}"
    if torch.allclose(out, golden, atol=atol, rtol=rtol):
        dist_print(f"[RANK {RANK}] Correctness check passed for {mode_name}.", need_sync=True, allowed_ranks=[0])
    else:
        max_diff = torch.max(torch.abs(out - golden))
        dist_print(f"[RANK {RANK}] Max difference for {mode_name}: {max_diff.item()} (atol={atol}, rtol={rtol})")
        raise AssertionError(f"[RANK {RANK}] Output mismatch for {mode_name}.")


def make_cuda_graph(mempool, func):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(30):
            func()
    torch.cuda.current_stream().wait_stream(s)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, pool=mempool):
        func()
    return graph


def build_perf_runners(model, args, torch_func, triton_func):
    """
    Prefer CUDA Graph replay for lower-overhead timing, but fall back to eager
    execution when the current forward path is not graph-capturable.
    """
    mempool = torch.cuda.graph_pool_handle()
    try:
        model.set_fwd(mode='torch')
        torch_graph = make_cuda_graph(mempool, torch_func)
        model.set_fwd(mode=get_triton_mode(args))
        triton_graph = make_cuda_graph(mempool, triton_func)
        return torch_graph.replay, triton_graph.replay, (torch_graph, triton_graph, mempool), True
    except RuntimeError as e:
        # The PyTorch attention fallback uses host-side .item() on kv_offset, which
        # is illegal during CUDA graph capture. In that case, benchmark both paths
        # eagerly so the comparison remains fair.
        dist_print(f"CUDA Graph capture unavailable, falling back to eager timing: {e}",
                   need_sync=True,
                   allowed_ranks=[0])
        torch.cuda.synchronize()
        gc.collect()
        return torch_func, triton_func, (None, None, None), False


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


def get_triton_mode(args):
    if args.mode in AG_RS_MODES:
        return AG_RS_MODES[args.mode][2]
    return GEMM_AR_MODES[args.mode][1]


def build_model(args, dtype: torch.dtype, tp_group):
    if args.synthetic:
        max_length = args.synthetic_max_length if args.synthetic_max_length > 0 else args.seq_len + 128
        spec = resolve_synthetic_spec(
            name=args.synthetic_config,
            hidden_size=args.synthetic_hidden_size,
            intermediate_size=args.synthetic_intermediate_size,
            num_layers=args.synthetic_num_layers,
            num_heads=args.synthetic_num_heads,
            num_key_value_heads=args.synthetic_num_kv_heads,
            head_dim=args.synthetic_head_dim,
            max_length=max_length,
            vocab_size=args.synthetic_vocab_size,
            rope_theta=args.synthetic_rope_theta,
            norm_eps=args.synthetic_norm_eps,
            init_std=args.synthetic_init_std,
            tie_word_embeddings=args.synthetic_tie_word_embeddings,
            share_layer_weights=args.synthetic_share_weights,
            rank=RANK,
            world_size=WORLD_SIZE,
            dtype=dtype,
        )
        dist_print(f"Using synthetic TP model: {format_synthetic_spec(spec)}", need_sync=True, allowed_ranks=[0])
        return SyntheticDenseLLM(spec, tp_group)

    from triton_dist.models.config import ModelConfig
    from triton_dist.models import AutoLLM

    max_length = args.seq_len + 128
    model_config = ModelConfig(model_name=args.model,
                               max_length=max_length,
                               dtype=dtype,
                               rank=RANK,
                               world_size=WORLD_SIZE,
                               local_only=True)
    model = AutoLLM.from_pretrained(model_config, tp_group)
    if model.model_type != 'dense':
        raise NotImplementedError("test_tp_e2e_innov.py currently supports dense models only.")
    return model


def make_kv_cache(model, args, dtype: torch.dtype):
    return KV_Cache(num_layers=model.num_layers,
                    kv_heads=model.num_key_value_heads,
                    head_dim=model.head_dim,
                    batch_size=args.bsz,
                    dtype=dtype,
                    max_length=model.max_length,
                    world_size=WORLD_SIZE)


def make_input_ids(batch_size: int, seq_len: int, vocab_size: int):
    return torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long, device="cuda")


def init_model_for_mode(model, args, max_M: int):
    if args.mode in AG_RS_MODES:
        ag_impl, rs_impl, triton_mode = AG_RS_MODES[args.mode]
        kwargs = {}
        kwargs.update(build_ag_kwargs(args))
        kwargs.update(build_rs_kwargs(args))
        model.init_triton_dist_ablation_ctx(max_M=max_M, ag_impl=ag_impl, rs_impl=rs_impl, **kwargs)
        model.set_fwd(mode=triton_mode)
    else:
        impl, triton_mode = GEMM_AR_MODES[args.mode]
        model.init_triton_dist_gemm_ar_ablation_ctx(max_M=max_M, impl=impl, **build_ar_kwargs(args))
        model.set_fwd(mode=triton_mode)


def run_hf_baseline(model_name, input_ids, position_ids, dtype):
    from triton_dist.models.utils import init_model_cpu

    dist_print("Running HuggingFace baseline to get golden result...")
    hf_model = init_model_cpu(model_name=model_name, dtype=dtype).cuda()
    with torch.inference_mode():
        golden = hf_model.forward(input_ids=input_ids, position_ids=position_ids).logits.float()
    golden = golden[:, -1:, :].contiguous()
    del hf_model
    gc.collect()
    torch.cuda.empty_cache()
    dist_print("Finished HuggingFace baseline and freed memory.")
    return golden


def run_correctness_check(model, golden, input_ids, position_ids, kv_cache, args, atol, rtol):
    dist_print("\n--- Running Correctness Checks ---")
    model.set_fwd(mode='torch')
    logits_torch = model.inference(input_ids=input_ids, position_ids=position_ids, kv_cache=kv_cache)
    check_allclose(logits_torch.softmax(dim=-1, dtype=torch.float32),
                   golden.softmax(dim=-1, dtype=torch.float32),
                   atol=atol,
                   rtol=rtol,
                   mode_name="torch")

    max_M = args.bsz * args.seq_len if args.run_type == "prefill" else args.bsz
    init_model_for_mode(model, args, max_M=max_M)
    mode_name = args.mode
    dist_input = input_ids
    golden_check = golden
    if args.mode in AG_RS_MODES:
        dist_input = input_ids.split(args.bsz // WORLD_SIZE, dim=0)[RANK].contiguous()
        golden_check = golden.split(args.bsz // WORLD_SIZE, dim=0)[RANK].contiguous()

    logits_triton = model.inference(input_ids=dist_input, position_ids=position_ids, kv_cache=kv_cache)
    check_allclose(logits_triton.softmax(dim=-1, dtype=torch.float32),
                   golden_check.softmax(dim=-1, dtype=torch.float32),
                   atol=atol,
                   rtol=rtol,
                   mode_name=mode_name)


def run_performance_test(model, run_type, kv_cache, args, tp_group):
    dist_print(f"\n--- Running Performance Test: {run_type.capitalize()} (Mode: {args.mode}) ---")
    vocab_size = getattr(model, "vocab_size", 1000)

    if run_type == 'prefill':
        seq_len, max_M = args.seq_len, args.bsz * args.seq_len
        input_ids = make_input_ids(args.bsz, seq_len, vocab_size)
        position_ids = torch.arange(0, seq_len, dtype=torch.int64, device="cuda").unsqueeze(0).expand(args.bsz, -1)
        kv_cache.kv_offset.fill_(0)
    else:
        seq_len, max_M = 1, args.bsz
        input_ids = make_input_ids(args.bsz, seq_len, vocab_size)
        position_ids = torch.arange(args.seq_len, args.seq_len + 1, dtype=torch.int64,
                                    device="cuda").unsqueeze(0).expand(args.bsz, -1)
        kv_cache.kv_offset.fill_(args.seq_len)

    init_model_for_mode(model, args, max_M=max_M)

    torch_func = partial(model.inference, input_ids, position_ids, kv_cache, True)
    triton_func_input = input_ids.split(args.bsz //
                                        WORLD_SIZE, dim=0)[RANK].contiguous() if args.mode in AG_RS_MODES else input_ids
    triton_func = partial(model.inference, triton_func_input, position_ids, kv_cache, True)

    torch_runner, triton_runner, graph_state, used_cuda_graph = build_perf_runners(model, args, torch_func, triton_func)
    torch_graph, triton_dist_graph, mempool = graph_state

    with group_profile(f"e2e_innov_{run_type}", args.profile, group=tp_group):
        model.set_fwd(mode='torch')
        _, torch_perf = perf_func(torch_runner, iters=args.iters, warmup_iters=args.warmup)
        nvshmem_barrier_all_on_stream(torch.cuda.current_stream())

        model.set_fwd(mode=get_triton_mode(args))
        _, dist_triton_perf = perf_func(triton_runner, iters=args.iters, warmup_iters=args.warmup)
        nvshmem_barrier_all_on_stream(torch.cuda.current_stream())

    dist_print(f"torch {run_type} #{RANK}", torch_perf, need_sync=True, allowed_ranks=list(range(WORLD_SIZE)))
    dist_print(f"dist-triton-{args.mode} {run_type} #{RANK}",
               dist_triton_perf,
               f"{torch_perf / dist_triton_perf:.2f}x",
               need_sync=True,
               allowed_ranks=list(range(WORLD_SIZE)))

    if used_cuda_graph:
        del torch_graph, triton_dist_graph
    if mempool is not None:
        del mempool


if __name__ == "__main__":
    args = parse_args()
    validate_runtime_args(args)
    LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(LOCAL_RANK)
    TP_GROUP = initialize_distributed()

    DTYPE = DTYPE_MAP[args.dtype]
    default_tol = THRESHOLD_MAP.get(DTYPE, 1e-2)
    ATOL = args.atol if args.atol is not None else default_tol
    RTOL = args.rtol if args.rtol is not None else default_tol
    seed_everything(args.seed)

    if args.check and args.synthetic:
        model = build_model(args, DTYPE, TP_GROUP)
        input_ids = make_input_ids(args.bsz, args.seq_len, model.vocab_size)
        position_ids = torch.arange(0, args.seq_len, dtype=torch.long, device="cuda").unsqueeze(0).repeat(args.bsz, 1)
        golden_kv_cache = make_kv_cache(model, args, DTYPE)
        model.set_fwd(mode='torch')
        golden = model.inference(input_ids=input_ids, position_ids=position_ids, kv_cache=golden_kv_cache).float()
        kv_cache = make_kv_cache(model, args, DTYPE)
        run_correctness_check(model, golden, input_ids, position_ids, kv_cache, args, ATOL, RTOL)
    elif args.check:
        input_ids = torch.randint(10, 1000, (args.bsz, args.seq_len), dtype=torch.long, device="cuda")
        position_ids = torch.arange(0, args.seq_len, dtype=torch.long, device="cuda").unsqueeze(0).repeat(args.bsz, 1)
        golden = run_hf_baseline(args.model, input_ids, position_ids, DTYPE)
        model = build_model(args, DTYPE, TP_GROUP)
        kv_cache = make_kv_cache(model, args, DTYPE)
        run_correctness_check(model, golden, input_ids, position_ids, kv_cache, args, ATOL, RTOL)
    else:
        model = build_model(args, DTYPE, TP_GROUP)
        kv_cache = make_kv_cache(model, args, DTYPE)
        run_performance_test(model, args.run_type, kv_cache, args, TP_GROUP)

    model.finalize()
    finalize_distributed()
