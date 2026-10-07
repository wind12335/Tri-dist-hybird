"""Four-GPU TP-attention experiment for the v2.3 recursive-doubling GEMM-AllReduce path."""

from __future__ import annotations

import argparse
import os
from functools import partial

import nvshmem.core
import torch
import torch.distributed

from triton_dist.layers.nvidia.tp_attn import TP_Attn, _set_cos_sin_cache
from triton_dist.models.kv_cache import KV_Cache
from triton_dist.models.utils import init_model_cpu
from triton_dist.test.utils import assert_allclose
from triton_dist.utils import dist_print, initialize_distributed

from tp_gemm_ar_v23_common import (
    add_ar_v23_args,
    build_ar_v23_kwargs,
    correctness_metrics,
    run_eager_pair,
    validate_ar_v23_args,
    write_result,
)


DTYPE_MAP = {"bfloat16": torch.bfloat16, "float16": torch.float16}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bsz", type=int, default=64)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--run_type", choices=("prefill", "decode"), default="prefill")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-32B")
    parser.add_argument("--dtype", choices=DTYPE_MAP, default="bfloat16")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--check", default=False, action="store_true")
    parser.add_argument("--atol", type=float, default=None)
    parser.add_argument("--rtol", type=float, default=None)
    parser.add_argument("--result", type=str, default=None)
    add_ar_v23_args(parser)
    return parser.parse_args()


def prepare_kv_cache(kv_cache: KV_Cache, args: argparse.Namespace) -> None:
    if args.run_type == "prefill":
        kv_cache.kv_offset.zero_()
    else:
        torch.manual_seed(args.seed + 1001)
        torch.cuda.manual_seed(args.seed + 1001)
        kv_cache.kv_offset.fill_(args.seq_len)
        kv_cache.rand_fill_kv_cache(args.seq_len)


def main() -> None:
    args = parse_args()
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    group = None
    attn = None
    try:
        group = initialize_distributed()
        validate_ar_v23_args(args, world_size)
        if args.bsz <= 0 or args.seq_len <= 0:
            raise ValueError("--bsz and --seq_len must be positive.")
        dtype = DTYPE_MAP[args.dtype]
        atol = args.atol if args.atol is not None else (6e-2 if dtype == torch.bfloat16 else 1e-2)
        rtol = args.rtol if args.rtol is not None else atol
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

        hf_model = init_model_cpu(model_name=args.model, dtype=dtype)
        hf_attn = hf_model.model.layers[0].self_attn.eval().cuda()
        if hf_attn.head_dim != 128:
            raise ValueError(
                f"The current TP_Attn implementation requires head_dim=128, but {args.model} uses "
                f"head_dim={hf_attn.head_dim}. Choose a target model such as Llama3-70B or Qwen2-72B."
            )
        attn = TP_Attn(rank=rank, world_size=world_size, group=group)
        attn._init_parameters(hf_attn, verbose=rank == 0)
        cos_sin_cache = _set_cos_sin_cache(hf_model.model.rotary_emb.inv_freq.cuda(), max_length=args.seq_len + 128)
        kv_cache = KV_Cache(
            num_layers=1,
            batch_size=args.bsz,
            max_length=args.seq_len + 128,
            kv_heads=hf_attn.config.num_key_value_heads,
            head_dim=hf_attn.head_dim,
            dtype=dtype,
            world_size=world_size,
        )
        del hf_model, hf_attn

        q_len = args.seq_len if args.run_type == "prefill" else 1
        position_start = 0 if args.run_type == "prefill" else args.seq_len
        position_ids = torch.arange(position_start, position_start + q_len, dtype=torch.long, device="cuda")
        position_ids = position_ids.unsqueeze(0).expand(args.bsz, -1)
        x = torch.rand((args.bsz, q_len, attn.K), dtype=dtype, device="cuda") / 10
        M = args.bsz * q_len

        attn._init_new_gemm_ar_v23_ctx(max_M=M, dtype=dtype, **build_ar_v23_kwargs(args))
        torch_func = partial(attn.torch_fwd, x, position_ids, cos_sin_cache, kv_cache, layer_idx=0)
        v23_func = partial(
            attn.dist_triton_select_gemm_ar_fwd,
            x,
            position_ids,
            cos_sin_cache,
            kv_cache,
            0,
            impl="v23",
            autotune=args.autotune,
        )

        prepare_kv_cache(kv_cache, args)
        expected = torch_func()
        prepare_kv_cache(kv_cache, args)
        actual = v23_func()
        assert_allclose(actual, expected, atol=atol, rtol=rtol)
        correctness = correctness_metrics(actual, expected, atol=atol, rtol=rtol)
        dist_print(f"TP-attention AR-v23 correctness passed: {correctness}", need_sync=True, allowed_ranks=[0])

        result = {
            "scope": "tp_attention",
            "model": args.model,
            "run_type": args.run_type,
            "bsz": args.bsz,
            "seq_len": args.seq_len,
            "M": M,
            "dtype": args.dtype,
            "world_size": world_size,
            "rank": rank,
            "ar": build_ar_v23_kwargs(args),
            "correctness": correctness,
        }
        if not args.check:
            prepare_kv_cache(kv_cache, args)
            result["performance"] = run_eager_pair(
                torch_func, v23_func, group=group, warmup=args.warmup, iters=args.iters)
            dist_print(f"TP-attention AR-v23 result: {result['performance']}", need_sync=True, allowed_ranks=[0])
        write_result(args.result, result, rank=rank)
    finally:
        if attn is not None:
            attn.finalize()
        if group is not None:
            nvshmem.core.finalize()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
