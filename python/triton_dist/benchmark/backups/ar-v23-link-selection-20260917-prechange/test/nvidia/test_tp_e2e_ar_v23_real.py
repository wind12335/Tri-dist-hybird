"""Real-model TP end-to-end experiment using v2.3 GEMM-AllReduce in attention and MLP."""

from __future__ import annotations

import argparse
import os
from functools import partial

import torch

from triton_dist.models.kv_cache import KV_Cache
from triton_dist.models.utils import seed_everything
from triton_dist.utils import dist_print, finalize_distributed, initialize_distributed

from tp_gemm_ar_v23_common import (
    add_ar_v23_args,
    build_ar_v23_kwargs,
    correctness_metrics,
    run_eager_pair,
    run_single_pair,
    validate_ar_v23_args,
    write_result,
)


DTYPE_MAP = {"bfloat16": torch.bfloat16, "float16": torch.float16}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bsz", type=int, default=64)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--run_type", choices=("prefill", "decode"), default="prefill")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-32B",
                        help="Local HuggingFace model name/path.")
    parser.add_argument("--dtype", choices=DTYPE_MAP, default="bfloat16")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--check", default=False, action="store_true")
    parser.add_argument("--single_shot", default=False, action="store_true",
                        help="Time one Torch and one v2.3 invocation, then exit.")
    parser.add_argument("--skip_correctness", default=False, action="store_true",
                        help="Skip the LM-head probability check; useful for a timing-only run after correctness passed.")
    parser.add_argument("--include_lm_head", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--num_layers", type=int, default=0,
                        help="Number of leading Transformer layers to run; 0 uses the full model.")
    parser.add_argument("--stability_repeats", type=int, default=0,
                        help="Run this many synchronized v2.3 forwards with changing token IDs before timing.")
    parser.add_argument("--atol", type=float, default=None)
    parser.add_argument("--rtol", type=float, default=None)
    parser.add_argument("--result", type=str, default=None)
    add_ar_v23_args(parser)
    for link in ("attn", "mlp"):
        parser.add_argument(f"--{link}_ar_chunk_rows", type=int, default=None)
        parser.add_argument(f"--{link}_ar_active_chunk_window", type=int, default=None)
        parser.add_argument(f"--{link}_ar_stage_slots", type=int, default=None)
        parser.add_argument(f"--{link}_ar_num_comm_sms", type=int, default=None)
        parser.add_argument(f"--{link}_ar_comm_lanes", type=int, default=None)
    return parser.parse_args()


def build_link_ar_v23_kwargs(args: argparse.Namespace, link: str) -> dict:
    kwargs = build_ar_v23_kwargs(args)
    for name in ("chunk_rows", "active_chunk_window", "stage_slots", "num_comm_sms", "comm_lanes"):
        override = getattr(args, f"{link}_ar_{name}")
        if override is not None:
            if override <= 0:
                raise ValueError(f"--{link}_ar_{name} must be positive, got {override}.")
            kwargs[name] = override
            if name == "chunk_rows":
                kwargs["stripe_rows"] = override
    return kwargs


def build_model(args: argparse.Namespace, dtype: torch.dtype, group):
    from triton_dist.models import AutoLLM
    from triton_dist.models.config import ModelConfig

    config = ModelConfig(
        model_name=args.model,
        max_length=args.seq_len + 128,
        dtype=dtype,
        rank=int(os.environ.get("RANK", 0)),
        world_size=int(os.environ.get("WORLD_SIZE", 1)),
        local_only=True,
    )
    model = AutoLLM.from_pretrained(config, group)
    if model.model_type != "dense":
        raise NotImplementedError("AR-v23 end-to-end testing currently supports dense models only.")
    return model


def make_kv_cache(model, args: argparse.Namespace, dtype: torch.dtype, world_size: int) -> KV_Cache:
    return KV_Cache(
        num_layers=model.num_layers,
        kv_heads=model.num_key_value_heads,
        head_dim=model.head_dim,
        batch_size=args.bsz,
        dtype=dtype,
        max_length=model.max_length,
        world_size=world_size,
    )


def prepare_kv_cache(kv_cache: KV_Cache, args: argparse.Namespace) -> None:
    if args.run_type == "prefill":
        kv_cache.kv_offset.zero_()
    else:
        torch.manual_seed(args.seed + 1001)
        torch.cuda.manual_seed(args.seed + 1001)
        kv_cache.kv_offset.fill_(args.seq_len)
        kv_cache.rand_fill_kv_cache(args.seq_len)


def make_inputs(model, args: argparse.Namespace) -> tuple[torch.Tensor, torch.Tensor]:
    q_len = args.seq_len if args.run_type == "prefill" else 1
    vocab_size = getattr(model, "vocab_size", model.embed_tokens.shape[0])
    input_ids = torch.randint(0, vocab_size, (args.bsz, q_len), dtype=torch.long, device="cuda")
    position_start = 0 if args.run_type == "prefill" else args.seq_len
    position_ids = torch.arange(position_start, position_start + q_len, dtype=torch.long, device="cuda")
    return input_ids, position_ids.unsqueeze(0).expand(args.bsz, -1)


def run_v23_stability(
    model,
    input_ids: torch.Tensor,
    position_ids: torch.Tensor,
    kv_cache: KV_Cache,
    args: argparse.Namespace,
    *,
    group,
    rank: int,
    num_layers: int,
) -> list[dict[str, float]]:
    model.set_fwd(mode="triton_dist_gemm_ar_v23")
    vocab_size = getattr(model, "vocab_size", model.embed_tokens.shape[0])
    records = []
    for repeat in range(args.stability_repeats):
        repeat_ids = torch.remainder(input_ids + repeat + 1, vocab_size)
        prepare_kv_cache(kv_cache, args)
        output = model.inference(
            repeat_ids,
            position_ids,
            kv_cache,
            not args.include_lm_head,
            num_layers=num_layers,
        )
        torch.cuda.synchronize()
        checksum = output.float().mean()
        checksum_min = checksum.clone()
        checksum_max = checksum.clone()
        torch.distributed.all_reduce(checksum_min, op=torch.distributed.ReduceOp.MIN, group=group)
        torch.distributed.all_reduce(checksum_max, op=torch.distributed.ReduceOp.MAX, group=group)
        torch.distributed.barrier(group=group, device_ids=[torch.cuda.current_device()])
        record = {
            "repeat": float(repeat),
            "checksum_min": float(checksum_min.item()),
            "checksum_max": float(checksum_max.item()),
        }
        records.append(record)
        if rank == 0:
            print(
                f"[stability] repeat={repeat + 1}/{args.stability_repeats}, "
                f"layers={num_layers}, checksum_range=[{record['checksum_min']:.8f}, "
                f"{record['checksum_max']:.8f}]",
                flush=True,
            )
    return records


def main() -> None:
    args = parse_args()
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    group = None
    model = None
    try:
        group = initialize_distributed()
        validate_ar_v23_args(args, world_size)
        if args.bsz <= 0 or args.seq_len <= 0:
            raise ValueError("--bsz and --seq_len must be positive.")
        dtype = DTYPE_MAP[args.dtype]
        atol = args.atol if args.atol is not None else (1.2e-1 if dtype == torch.bfloat16 else 2e-2)
        rtol = args.rtol if args.rtol is not None else atol
        seed_everything(args.seed)

        model = build_model(args, dtype, group)
        if model.head_dim != 128:
            raise ValueError(
                f"The current TP_Attn implementation requires head_dim=128, but {args.model} uses "
                f"head_dim={model.head_dim}. Choose a target model such as Llama3-70B or Qwen2-72B."
            )
        kv_cache = make_kv_cache(model, args, dtype, world_size)
        input_ids, position_ids = make_inputs(model, args)
        max_M = args.bsz * args.seq_len if args.run_type == "prefill" else args.bsz
        num_layers = model.num_layers if args.num_layers == 0 else args.num_layers
        if not 1 <= num_layers <= model.num_layers:
            raise ValueError(f"--num_layers must be in [1, {model.num_layers}] or 0, got {args.num_layers}.")
        attn_ar_kwargs = build_link_ar_v23_kwargs(args, "attn")
        mlp_ar_kwargs = build_link_ar_v23_kwargs(args, "mlp")
        model.init_triton_dist_gemm_ar_ablation_ctx(
            max_M=max_M,
            impl="v23",
            attn_kwargs=attn_ar_kwargs,
            mlp_kwargs=mlp_ar_kwargs,
        )

        torch_func = partial(
            model.inference,
            input_ids,
            position_ids,
            kv_cache,
            not args.include_lm_head,
            num_layers=num_layers,
        )
        v23_func = partial(
            model.inference,
            input_ids,
            position_ids,
            kv_cache,
            not args.include_lm_head,
            num_layers=num_layers,
        )

        correctness = None
        if not args.skip_correctness:
            # Match the established real-model correctness protocol: validate
            # FP32 output probabilities with LM-head included. Raw BF16 hidden
            # states accumulate reduction-order rounding differences across all
            # transformer layers and are not the paper's end-to-end criterion.
            correctness_func = partial(
                model.inference,
                input_ids,
                position_ids,
                kv_cache,
                False,
                num_layers=num_layers,
            )
            prepare_kv_cache(kv_cache, args)
            model.set_fwd(mode="torch")
            expected = correctness_func().softmax(dim=-1, dtype=torch.float32)
            prepare_kv_cache(kv_cache, args)
            model.set_fwd(mode="triton_dist_gemm_ar_v23")
            actual = correctness_func().softmax(dim=-1, dtype=torch.float32)
            correctness = correctness_metrics(actual, expected, atol=atol, rtol=rtol)
            dist_print(f"TP end-to-end AR-v23 correctness passed: {correctness}", need_sync=True, allowed_ranks=[0])

        stability = None
        if args.stability_repeats > 0:
            stability = run_v23_stability(
                model,
                input_ids,
                position_ids,
                kv_cache,
                args,
                group=group,
                rank=rank,
                num_layers=num_layers,
            )

        result = {
            "scope": "model_forward_logits" if args.include_lm_head else "transformer_body_excluding_lm_head",
            "model": args.model,
            "run_type": args.run_type,
            "bsz": args.bsz,
            "seq_len": args.seq_len,
            "max_M": max_M,
            "dtype": args.dtype,
            "world_size": world_size,
            "rank": rank,
            "num_layers": num_layers,
            "include_lm_head": args.include_lm_head,
            "ar": {"attention": attn_ar_kwargs, "mlp": mlp_ar_kwargs},
            "correctness": correctness,
            "stability": stability,
        }
        if args.single_shot:
            prepare_kv_cache(kv_cache, args)
            result["performance"] = run_single_pair(
                torch_func,
                v23_func,
                group=group,
                before_torch=partial(model.set_fwd, mode="torch"),
                before_v23=partial(model.set_fwd, mode="triton_dist_gemm_ar_v23"),
            )
            dist_print(f"TP end-to-end AR-v23 result: {result['performance']}", need_sync=True, allowed_ranks=[0])
        elif not args.check:
            prepare_kv_cache(kv_cache, args)
            result["performance"] = run_eager_pair(
                torch_func,
                v23_func,
                group=group,
                warmup=args.warmup,
                iters=args.iters,
                before_torch=partial(model.set_fwd, mode="torch"),
                before_v23=partial(model.set_fwd, mode="triton_dist_gemm_ar_v23"),
                synchronize_each_iter=True,
            )
            dist_print(f"TP end-to-end AR-v23 result: {result['performance']}", need_sync=True, allowed_ranks=[0])
        write_result(args.result, result, rank=rank)
    finally:
        if model is not None:
            model.finalize()
        if group is not None:
            finalize_distributed()
        elif torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
