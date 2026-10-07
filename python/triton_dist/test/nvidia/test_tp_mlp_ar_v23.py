"""Four-GPU TP-MLP experiment for the v2.3 recursive-doubling GEMM-AllReduce path."""

from __future__ import annotations

import argparse
import os
from functools import partial

import nvshmem.core
import torch
import torch.distributed

from triton_dist.layers.nvidia.tp_mlp import TP_MLP
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
    parser.add_argument("--M", type=int, default=8192, help="Flattened token count.")
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


def rand_tensor(shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    return torch.rand(shape, dtype=dtype, device="cuda") / 10


def main() -> None:
    args = parse_args()
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    group = None
    mlp = None
    try:
        group = initialize_distributed()
        validate_ar_v23_args(args, world_size)
        if args.M <= 0:
            raise ValueError(f"--M must be positive, got {args.M}.")
        dtype = DTYPE_MAP[args.dtype]
        atol = args.atol if args.atol is not None else (6e-2 if dtype == torch.bfloat16 else 1e-2)
        rtol = args.rtol if args.rtol is not None else atol
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

        hf_model = init_model_cpu(model_name=args.model, dtype=dtype)
        hf_mlp = hf_model.model.layers[0].mlp.eval().cuda()
        mlp = TP_MLP(rank=rank, world_size=world_size, group=group)
        mlp._init_parameters(hf_mlp, verbose=rank == 0)
        del hf_model, hf_mlp

        x = rand_tensor((args.M, mlp.K), dtype)
        mlp._init_new_gemm_ar_v23_ctx(max_M=args.M, dtype=dtype, **build_ar_v23_kwargs(args))
        torch_func = partial(mlp.torch_fwd, x)
        v23_func = partial(mlp.dist_triton_select_gemm_ar_fwd, x, impl="v23", autotune=args.autotune)

        expected = torch_func()
        actual = v23_func()
        assert_allclose(actual, expected, atol=atol, rtol=rtol)
        correctness = correctness_metrics(actual, expected, atol=atol, rtol=rtol)
        dist_print(f"TP-MLP AR-v23 correctness passed: {correctness}", need_sync=True, allowed_ranks=[0])

        result = {
            "scope": "tp_mlp",
            "model": args.model,
            "M": args.M,
            "dtype": args.dtype,
            "world_size": world_size,
            "rank": rank,
            "ar": build_ar_v23_kwargs(args),
            "correctness": correctness,
        }
        if not args.check:
            result["performance"] = run_eager_pair(
                torch_func, v23_func, group=group, warmup=args.warmup, iters=args.iters)
            dist_print(f"TP-MLP AR-v23 result: {result['performance']}", need_sync=True, allowed_ranks=[0])
        write_result(args.result, result, rank=rank)
    finally:
        if mlp is not None:
            mlp.finalize()
        if group is not None:
            nvshmem.core.finalize()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
