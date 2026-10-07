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
import json
import os
from pathlib import Path

import torch
from functools import partial

from triton_dist.models.kv_cache import KV_Cache
from triton_dist.models.utils import seed_everything
from triton_dist.profiler_utils import group_profile
from triton_dist.utils import finalize_distributed, initialize_distributed, dist_print

from tp_ag_rs_innov_common import run_pair

# torchrun --nproc_per_node=2 python/triton_dist/test/nvidia/test_tp_e2e_innov_real.py \
#       --model Qwen/Qwen2-72B \
#       --bsz 64 \
#       --seq_len 128 \
#       --run_type prefill \
#       --mode ag_rs_new \
#       --rs_chunk_rows 512 \
#       --rs_active_chunk_window 4 \
#       --rs_stage_slots 4 \
#       --rs_steady_sms 8 \
#       --rs_tail_sms 20 \
#       --rs_comm_lanes 2 \
#       --rs_n_bands 2 \
#       --rs_frontier_chunks 2 \
#       --rs_local_seed_direct

#   如果你要做正确性检查，就加上 --check，必要时再加：

#   --atol 0.12 --rtol 0.12

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
    if args.seq_len <= 0:
        raise ValueError(f"--seq_len must be positive, got {args.seq_len}.")
    if args.warmup < 0 or args.graph_warmup < 0:
        raise ValueError("--warmup and --graph_warmup must be non-negative.")
    if args.iters <= 0:
        raise ValueError(f"--iters must be positive, got {args.iters}.")
    if args.num_layers < 0 or args.stability_repeats < 0:
        raise ValueError("--num_layers and --stability_repeats must be non-negative.")

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
    parser.add_argument("--model", default="Qwen/Qwen3-32B", type=str, help="Local HuggingFace model name/path")
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
    parser.add_argument("--autotune", default=False, action=argparse.BooleanOptionalAction,
                        help="Disabled by default to match the stable operator-level AG/RS benchmarks.")
    parser.add_argument("--cuda_graph", default=False, action=argparse.BooleanOptionalAction,
                        help="Use CUDA Graph replay; synchronized eager timing is the safe default.")
    parser.add_argument("--graph_warmup", type=int, default=3,
                        help="Warmup invocations before CUDA Graph capture (replaces the old fixed 30 forwards).")
    parser.add_argument("--synchronize_each_iter", default=True, action=argparse.BooleanOptionalAction,
                        help="Drain and align all ranks around every measured model invocation.")
    parser.add_argument("--include_lm_head", default=False, action=argparse.BooleanOptionalAction,
                        help="Include the final vocabulary projection in performance timing.")
    parser.add_argument("--num_layers", type=int, default=0,
                        help="Number of leading Transformer layers to run; 0 uses the full model.")
    parser.add_argument("--stability_repeats", type=int, default=0,
                        help="Run repeated correctness checks with changing token IDs before timing.")
    parser.add_argument("--result_dir", type=str, default=None,
                        help="Directory for structured JSON results. Disabled when omitted.")
    parser.add_argument("--repeat_id", type=int, default=0,
                        help="Independent repeat identifier written to structured results.")

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

    link_override_names = (
        "ag_copy_sms",
        "ag_tile_rows_per_chunk",
        "ag_min_m_per_rank_for_tile_ready",
        "ag_target_chunks_per_rank",
        "ag_min_tile_rows_per_chunk",
        "rs_chunk_rows",
        "rs_target_chunks_per_rank",
        "rs_min_chunk_rows",
        "rs_active_chunk_window",
        "rs_comm_lanes",
        "rs_n_bands",
        "rs_frontier_chunks",
        "rs_steady_sms",
        "rs_tail_sms",
        "rs_stage_slots",
        "rs_tail_chunk_window",
    )
    for link in ("attn", "mlp"):
        for name in link_override_names:
            parser.add_argument(f"--{link}_{name}", type=int, default=None,
                                help=f"Override --{name} for the {link} link only.")
    return parser.parse_args()


def check_allclose(out: torch.Tensor, golden: torch.Tensor, atol=1e-3, rtol=1e-3, mode_name=""):
    assert out.shape == golden.shape, f"Shape mismatch for {mode_name}: {out.shape} vs {golden.shape}"
    out_float = out.float()
    golden_float = golden.float()
    abs_diff = (out_float - golden_float).abs()
    max_abs_diff = abs_diff.max().item()
    max_rel_diff = (abs_diff / golden_float.abs().clamp_min(1e-12)).max().item()
    passed = bool(torch.allclose(out, golden, atol=atol, rtol=rtol))
    if passed:
        dist_print(f"[RANK {RANK}] Correctness check passed for {mode_name}.", need_sync=True, allowed_ranks=[0])
    else:
        dist_print(f"[RANK {RANK}] Max difference for {mode_name}: {max_abs_diff} (atol={atol}, rtol={rtol})")
        raise AssertionError(f"[RANK {RANK}] Output mismatch for {mode_name}.")
    return {
        "mode": mode_name,
        "passed": passed,
        "atol": atol,
        "rtol": rtol,
        "max_abs_diff": max_abs_diff,
        "max_rel_diff": max_rel_diff,
        "shape": list(out.shape),
    }


def get_triton_mode(args):
    if args.mode in AG_RS_MODES:
        return AG_RS_MODES[args.mode][2]
    return GEMM_AR_MODES[args.mode][1]


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


def build_link_ag_rs_kwargs(args, link: str):
    kwargs = {**build_ag_kwargs(args), **build_rs_kwargs(args)}
    for name in (
        "ag_copy_sms",
        "ag_tile_rows_per_chunk",
        "ag_min_m_per_rank_for_tile_ready",
        "ag_target_chunks_per_rank",
        "ag_min_tile_rows_per_chunk",
        "rs_chunk_rows",
        "rs_target_chunks_per_rank",
        "rs_min_chunk_rows",
        "rs_active_chunk_window",
        "rs_comm_lanes",
        "rs_n_bands",
        "rs_frontier_chunks",
        "rs_steady_sms",
        "rs_tail_sms",
        "rs_stage_slots",
        "rs_tail_chunk_window",
    ):
        override = getattr(args, f"{link}_{name}")
        if override is None:
            continue
        if override < 0 or (override == 0 and name not in {"ag_copy_sms", "ag_tile_rows_per_chunk", "rs_chunk_rows"}):
            raise ValueError(f"--{link}_{name} has invalid value {override}.")
        kwargs[name.removeprefix("ag_").removeprefix("rs_")] = override
    return kwargs


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


def build_model(args, dtype: torch.dtype, tp_group):
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
        raise NotImplementedError("test_tp_e2e_innov_real.py currently supports dense models only.")
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


def get_vocab_size(model):
    if hasattr(model, "vocab_size"):
        return model.vocab_size
    if hasattr(model, "config") and hasattr(model.config, "vocab_size"):
        return model.config.vocab_size
    return model.embed_tokens.shape[0]


def make_eval_inputs(model, args):
    if args.run_type == "prefill":
        seq_len = args.seq_len
        input_ids = make_input_ids(args.bsz, seq_len, get_vocab_size(model))
        position_ids = torch.arange(0, seq_len, dtype=torch.long, device="cuda").unsqueeze(0).expand(args.bsz, -1)
    else:
        input_ids = make_input_ids(args.bsz, 1, get_vocab_size(model))
        position_ids = torch.full((args.bsz, 1), args.seq_len, dtype=torch.long, device="cuda")
    return input_ids, position_ids


def reset_kv_cache(kv_cache, args):
    if args.run_type == "prefill":
        kv_cache.kv_offset.zero_()
        return

    torch.manual_seed(args.seed + 1001)
    torch.cuda.manual_seed(args.seed + 1001)
    kv_cache.kv_offset.fill_(args.seq_len)
    kv_cache.rand_fill_kv_cache(args.seq_len)


def init_model_for_mode(model, args, max_M: int):
    if args.mode in AG_RS_MODES:
        ag_impl, rs_impl, triton_mode = AG_RS_MODES[args.mode]
        model.init_triton_dist_ablation_ctx(
            max_M=max_M,
            ag_impl=ag_impl,
            rs_impl=rs_impl,
            attn_kwargs=build_link_ag_rs_kwargs(args, "attn"),
            mlp_kwargs=build_link_ag_rs_kwargs(args, "mlp"),
        )
        model.set_fwd(mode=triton_mode)
    else:
        impl, triton_mode = GEMM_AR_MODES[args.mode]
        model.init_triton_dist_gemm_ar_ablation_ctx(max_M=max_M, impl=impl, **build_ar_kwargs(args))
        model.set_fwd(mode=triton_mode)


def selected_input(input_ids, args):
    if args.mode not in AG_RS_MODES:
        return input_ids
    return input_ids.split(args.bsz // WORLD_SIZE, dim=0)[RANK].contiguous()


def run_correctness_check(model, golden, input_ids, position_ids, kv_cache, args, atol, rtol, num_layers):
    dist_print("\n--- Running Correctness Checks ---")
    reset_kv_cache(kv_cache, args)
    model.set_fwd(mode="torch")
    logits_torch = model.inference(
        input_ids=input_ids,
        position_ids=position_ids,
        kv_cache=kv_cache,
        num_layers=num_layers,
    )
    torch_metrics = check_allclose(logits_torch.softmax(dim=-1, dtype=torch.float32),
                                   golden.softmax(dim=-1, dtype=torch.float32),
                                   atol=atol,
                                   rtol=rtol,
                                   mode_name="torch")

    max_M = args.bsz * args.seq_len if args.run_type == "prefill" else args.bsz
    init_model_for_mode(model, args, max_M=max_M)
    dist_input = selected_input(input_ids, args)
    golden_check = golden
    if args.mode in AG_RS_MODES:
        golden_check = golden.split(args.bsz // WORLD_SIZE, dim=0)[RANK].contiguous()

    reset_kv_cache(kv_cache, args)
    logits_triton = model.inference(
        input_ids=dist_input,
        position_ids=position_ids,
        kv_cache=kv_cache,
        num_layers=num_layers,
    )
    triton_metrics = check_allclose(logits_triton.softmax(dim=-1, dtype=torch.float32),
                                    golden_check.softmax(dim=-1, dtype=torch.float32),
                                    atol=atol,
                                    rtol=rtol,
                                    mode_name=args.mode)
    return {"torch": torch_metrics, "selected": triton_metrics}


def run_stability_check(model, input_ids, position_ids, kv_cache, args, tp_group, atol, rtol, num_layers):
    if args.stability_repeats <= 0:
        return []

    records = []
    vocab_size = get_vocab_size(model)
    for repeat in range(args.stability_repeats):
        repeat_ids = torch.remainder(input_ids + repeat + 1, vocab_size)
        reset_kv_cache(kv_cache, args)
        model.set_fwd(mode="torch")
        expected = model.inference(
            repeat_ids,
            position_ids,
            kv_cache,
            False,
            num_layers=num_layers,
        ).softmax(dim=-1, dtype=torch.float32)

        repeat_dist_ids = selected_input(repeat_ids, args)
        if args.mode in AG_RS_MODES:
            expected = expected.split(args.bsz // WORLD_SIZE, dim=0)[RANK].contiguous()
        reset_kv_cache(kv_cache, args)
        model.set_fwd(mode=get_triton_mode(args))
        actual = model.inference(
            repeat_dist_ids,
            position_ids,
            kv_cache,
            False,
            num_layers=num_layers,
        ).softmax(dim=-1, dtype=torch.float32)

        metrics = check_allclose(actual, expected, atol=atol, rtol=rtol, mode_name=f"stability-{repeat}")
        max_abs = torch.tensor(metrics["max_abs_diff"], dtype=torch.float64, device="cuda")
        torch.distributed.all_reduce(max_abs, op=torch.distributed.ReduceOp.MAX, group=tp_group)
        record = {"repeat": repeat, "rank_max_abs_diff": float(max_abs.item())}
        records.append(record)
        dist_print(
            f"[stability] repeat={repeat + 1}/{args.stability_repeats}, "
            f"layers={num_layers}, rank_max_abs_diff={record['rank_max_abs_diff']:.8g}",
            need_sync=True,
            allowed_ranks=[0],
        )
    return records


def run_performance_test(model, run_type, kv_cache, args, tp_group, atol, rtol, num_layers):
    dist_print(f"\n--- Running Performance Test: {run_type.capitalize()} (Mode: {args.mode}) ---")
    vocab_size = get_vocab_size(model)

    if run_type == "prefill":
        max_M = args.bsz * args.seq_len
    else:
        max_M = args.bsz
    input_ids, position_ids = make_eval_inputs(model, args)
    reset_kv_cache(kv_cache, args)

    init_model_for_mode(model, args, max_M=max_M)
    stability = run_stability_check(
        model,
        input_ids,
        position_ids,
        kv_cache,
        args,
        tp_group,
        atol,
        rtol,
        num_layers,
    )

    torch_func = partial(
        model.inference,
        input_ids,
        position_ids,
        kv_cache,
        not args.include_lm_head,
        num_layers=num_layers,
    )
    triton_func = partial(
        model.inference,
        selected_input(input_ids, args),
        position_ids,
        kv_cache,
        not args.include_lm_head,
        num_layers=num_layers,
    )

    with group_profile(f"e2e_innov_real_{run_type}", args.profile, group=tp_group):
        performance, graph_state = run_pair(
            torch_func,
            triton_func,
            group=tp_group,
            warmup=args.warmup,
            iters=args.iters,
            use_cuda_graph=args.cuda_graph,
            graph_warmup=args.graph_warmup,
            synchronize_each_iter=args.synchronize_each_iter,
            before_torch_mode=partial(model.set_fwd, mode="torch"),
            before_selected_mode=partial(model.set_fwd, mode=get_triton_mode(args)),
            before_torch_each=partial(reset_kv_cache, kv_cache, args),
            before_selected_each=partial(reset_kv_cache, kv_cache, args),
        )

    if args.cuda_graph and not performance["cuda_graph_used"]:
        dist_print("CUDA Graph capture was unavailable; used eager timing.", need_sync=True, allowed_ranks=[0])
    torch_perf = performance["torch_local_ms"]
    dist_triton_perf = performance["selected_local_ms"]
    dist_print(f"torch {run_type} #{RANK}", torch_perf, need_sync=True, allowed_ranks=list(range(WORLD_SIZE)))
    dist_print(f"dist-triton-{args.mode} {run_type} #{RANK}",
               dist_triton_perf,
               f"{torch_perf / dist_triton_perf:.2f}x",
               need_sync=True,
               allowed_ranks=list(range(WORLD_SIZE)))

    rank_result = {
        "rank": RANK,
        "torch_ms": float(torch_perf),
        "dist_triton_ms": float(dist_triton_perf),
        "speedup": float(torch_perf / dist_triton_perf),
        "include_lm_head": bool(args.include_lm_head),
        "num_layers": num_layers,
        "stability": stability,
        "performance": performance,
        "ag_rs": {
            "attention": build_link_ag_rs_kwargs(args, "attn") if args.mode in AG_RS_MODES else None,
            "mlp": build_link_ag_rs_kwargs(args, "mlp") if args.mode in AG_RS_MODES else None,
        },
    }
    del graph_state
    return rank_result


def gather_and_write_results(args, tp_group, local_result):
    if args.result_dir is None:
        return

    output_dir = Path(args.result_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    gathered = [None for _ in range(WORLD_SIZE)]
    torch.distributed.all_gather_object(gathered, local_result, group=tp_group)

    metadata = {
        "script": "test_tp_e2e_innov_real.py",
        "model": args.model,
        "platform": torch.cuda.get_device_name(torch.cuda.current_device()),
        "world_size": WORLD_SIZE,
        "rank": RANK,
        "local_rank": int(os.environ.get("LOCAL_RANK", 0)),
        "dtype": args.dtype,
        "run_type": args.run_type,
        "mode": args.mode,
        "repeat_id": args.repeat_id,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "args": vars(args),
    }
    prefix = f"{args.mode}_{args.run_type}_repeat_{args.repeat_id}"
    rank_path = output_dir / f"{prefix}_rank_{RANK}.json"
    rank_path.write_text(json.dumps({**metadata, "result": local_result}, indent=2), encoding="utf-8")

    if RANK == 0:
        summary = {**metadata, "ranks": gathered}
        timing_results = [x for x in gathered if isinstance(x, dict) and "torch_ms" in x]
        if timing_results:
            torch_values = [x["torch_ms"] for x in timing_results]
            triton_values = [x["dist_triton_ms"] for x in timing_results]
            summary["rank_max"] = {
                "torch_ms": max(torch_values),
                "dist_triton_ms": max(triton_values),
                "speedup": max(torch_values) / max(triton_values),
            }
            summary["rank_mean"] = {
                "torch_ms": sum(torch_values) / len(torch_values),
                "dist_triton_ms": sum(triton_values) / len(triton_values),
                "speedup": sum(x["speedup"] for x in timing_results) / len(timing_results),
            }
        (output_dir / f"{prefix}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    args = parse_args()
    validate_runtime_args(args)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    tp_group = initialize_distributed()

    dtype = DTYPE_MAP[args.dtype]
    default_tol = THRESHOLD_MAP.get(dtype, 1e-2)
    atol = args.atol if args.atol is not None else default_tol
    rtol = args.rtol if args.rtol is not None else default_tol
    seed_everything(args.seed)

    model = None
    try:
        model = build_model(args, dtype, tp_group)
        num_layers = model.num_layers if args.num_layers == 0 else args.num_layers
        if not 1 <= num_layers <= model.num_layers:
            raise ValueError(f"--num_layers must be in [1, {model.num_layers}] or 0, got {args.num_layers}.")
        if args.check:
            input_ids, position_ids = make_eval_inputs(model, args)
            reference_cache = make_kv_cache(model, args, dtype)
            reset_kv_cache(reference_cache, args)
            model.set_fwd(mode="torch")
            golden = model.inference(
                input_ids=input_ids,
                position_ids=position_ids,
                kv_cache=reference_cache,
                num_layers=num_layers,
            ).detach()
            kv_cache = make_kv_cache(model, args, dtype)
            correctness = run_correctness_check(
                model,
                golden,
                input_ids,
                position_ids,
                kv_cache,
                args,
                atol,
                rtol,
                num_layers,
            )
            stability = run_stability_check(
                model,
                input_ids,
                position_ids,
                kv_cache,
                args,
                tp_group,
                atol,
                rtol,
                num_layers,
            )
            gather_and_write_results(
                args,
                tp_group,
                {"correctness": correctness, "stability": stability, "num_layers": num_layers},
            )
        else:
            kv_cache = make_kv_cache(model, args, dtype)
            result = run_performance_test(model, args.run_type, kv_cache, args, tp_group, atol, rtol, num_layers)
            gather_and_write_results(args, tp_group, result)
    finally:
        if model is not None:
            model.finalize()
        finalize_distributed()
