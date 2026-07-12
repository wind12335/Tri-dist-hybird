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
import importlib
import importlib.util
import inspect
import os
import sys
from pathlib import Path

import torch
import torch.distributed

from triton_dist.profiler_utils import group_profile, perf_func
from triton_dist.test.utils import assert_allclose
from triton_dist.kernels.nvidia import (ag_gemm, create_ag_gemm_context, create_new_ag_gemm_context, new_ag_gemm)
from triton_dist.utils import (dist_print, finalize_distributed, initialize_distributed, nvshmem_barrier_all_on_stream,
                               wait_until_max_gpu_clock_or_warning, rand_tensor)

FLUX = None
FLUX_IMPORT_ERROR = None


def _is_usable_flux_module(module) -> bool:
    required_attrs = ("init_flux_shm", "AGKernel", "AllGatherOption")
    return module is not None and all(hasattr(module, attr) for attr in required_attrs)


def _load_flux_from_local_source(repo_root: Path):
    flux_init_py = repo_root / "flux" / "python" / "flux" / "__init__.py"
    if not flux_init_py.exists():
        return None, FileNotFoundError(f"FLUX source package not found: {flux_init_py}")

    previous_flux = sys.modules.pop("flux", None)
    spec = importlib.util.spec_from_file_location(
        "flux",
        flux_init_py,
        submodule_search_locations=[str(flux_init_py.parent)],
    )
    if spec is None or spec.loader is None:
        if previous_flux is not None:
            sys.modules["flux"] = previous_flux
        return None, ImportError(f"Failed to create import spec for {flux_init_py}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["flux"] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        sys.modules.pop("flux", None)
        if previous_flux is not None:
            sys.modules["flux"] = previous_flux
        return None, exc
    return module, None


def _try_import_flux():
    repo_root = Path(__file__).resolve().parents[3]
    first_error = None

    try:
        flux = importlib.import_module("flux")
        if _is_usable_flux_module(flux):
            return flux, None
        first_error = AttributeError(
            "Imported module `flux` is not the official FLUX package "
            f"(file={getattr(flux, '__file__', None)}, path={getattr(flux, '__path__', None)})"
        )
    except Exception as exc:
        first_error = exc

    flux, second_error = _load_flux_from_local_source(repo_root)
    if flux is not None and _is_usable_flux_module(flux):
        return flux, None

    return None, second_error if second_error is not None else first_error


def _get_flux_ring_mode(flux_module, ring_mode: str):
    if ring_mode == "auto":
        return None
    ring_mode_map = {
        "all2all": flux_module.AGRingMode.All2All,
        "ring1d": flux_module.AGRingMode.Ring1D,
        "ring2d": flux_module.AGRingMode.Ring2D,
    }
    return ring_mode_map[ring_mode]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--M", type=int, default=8192)
    parser.add_argument("--N", type=int, required=True)
    parser.add_argument("--K", type=int, required=True)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--warmup_iters", type=int, default=5)
    parser.add_argument("--autotune", action="store_true", default=False)
    parser.add_argument("--profile", action="store_true", default=False)
    parser.add_argument("--dump_csv", action="store_true", default=False)
    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    parser.add_argument("--trans_b", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--cooperative_copy", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--copy_sms", type=int, default=0, help="<=0 means auto(about 1/4 SMs for copy)")
    parser.add_argument("--enable_tile_ready", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--tile_rows_per_chunk",
                        type=int,
                        default=0,
                        help=">0 uses a fixed super-chunk row size; 0 enables heuristic selection")
    parser.add_argument("--min_m_per_rank_for_tile_ready",
                        type=int,
                        default=4096,
                        help="Disable chunk-ready when M_per_rank is smaller than this threshold")
    parser.add_argument("--target_chunks_per_rank",
                        type=int,
                        default=2,
                        help="Heuristic target number of ready chunks per rank when tile_rows_per_chunk=0")
    parser.add_argument("--min_tile_rows_per_chunk",
                        type=int,
                        default=1024,
                        help="Minimum super-chunk rows used by the heuristic")
    parser.add_argument("--enable_flux", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--flux_ring_mode",
                        default="auto",
                        choices=["auto", "all2all", "ring1d", "ring2d"],
                        help="FLUX all-gather ring mode")
    parser.add_argument("--flux_use_cuda_core_local",
                        default=None,
                        action=argparse.BooleanOptionalAction,
                        help="Override FLUX local-copy backend")
    parser.add_argument("--flux_use_cuda_core_ag",
                        default=None,
                        action=argparse.BooleanOptionalAction,
                        help="Override FLUX all-gather backend")
    parser.add_argument("--flux_use_pdl",
                        default=False,
                        action=argparse.BooleanOptionalAction,
                        help="Enable FLUX Programmatic Dependent Launch")
    args = parser.parse_args()
    return args


def torch_ag_gemm(
    pg: torch.distributed.ProcessGroup,
    A: torch.Tensor,
    B: torch.Tensor,
):
    M_per_rank, K = A.shape
    A_full = torch.empty([M_per_rank * pg.size(), K], dtype=A.dtype, device=A.device)
    torch.distributed.all_gather_into_tensor(A_full, A, pg)
    ag_gemm_output = torch.matmul(A_full, B)
    return ag_gemm_output


def make_data(M, N, K, dtype: torch.dtype, trans_b, tp_group: torch.distributed.ProcessGroup):
    rank = tp_group.rank()
    num_ranks = tp_group.size()
    M_per_rank = M // num_ranks
    N_per_rank = N // num_ranks
    scale = (rank + 1) * 0.01

    current_device = torch.cuda.current_device()
    A = rand_tensor([M_per_rank, K], dtype=dtype, device=current_device) * scale
    if trans_b:
        B = (rand_tensor([N_per_rank, K], dtype=dtype, device=current_device) * scale).T.contiguous()
    else:
        B = (rand_tensor([K, N_per_rank], dtype=dtype, device=current_device) * scale).contiguous()

    return A, B


def perf_test(M: int, N: int, K: int, pg: torch.distributed.ProcessGroup):
    rank = pg.rank()
    world_size = pg.size()
    n_per_rank = N // world_size
    base_ctx = None
    new_ctx = None
    flux_kernel = None
    flux_output = None
    flux_option = None
    flux_duration_ms = None

    if rank == 0:
        print(f"test shape: M {M}, N {N}, K {K}")

    assert M % world_size == 0
    assert N % world_size == 0

    A, B = make_data(M, N, K, dtype, args.trans_b, pg)

    def _torch_func():
        return torch_ag_gemm(pg, A, B)

    def _sync_all():
        nvshmem_barrier_all_on_stream(torch.cuda.current_stream())
        torch.cuda.synchronize()
        torch.distributed.barrier(pg)

    def _check_close(ref: torch.Tensor, out: torch.Tensor, name: str):
        try:
            assert_allclose(ref, out, atol=1e-3, rtol=1e-3)
            return True, ""
        except Exception as exc:
            return False, f"{name} check failed on rank {rank}: {exc}"

    try:
        # Create both contexts once for this single shape.
        base_ctx = create_ag_gemm_context(M, N, K, dtype, rank, world_size, LOCAL_WORLD_SIZE)
        new_ctx_kwargs = dict(
            max_M=M,
            N=N,
            K=K,
            dtype=dtype,
            rank=rank,
            num_ranks=world_size,
            num_local_ranks=LOCAL_WORLD_SIZE,
            copy_sms=args.copy_sms,
        )
        supported_new_ctx_kwargs = inspect.signature(create_new_ag_gemm_context).parameters
        optional_new_ctx_kwargs = {
            "enable_row_tile_barrier": args.enable_tile_ready,
            "tile_rows_per_chunk": args.tile_rows_per_chunk,
            "min_m_per_rank_for_tile_ready": args.min_m_per_rank_for_tile_ready,
            "target_chunks_per_rank": args.target_chunks_per_rank,
            "min_tile_rows_per_chunk": args.min_tile_rows_per_chunk,
        }
        for key, value in optional_new_ctx_kwargs.items():
            if key in supported_new_ctx_kwargs:
                new_ctx_kwargs[key] = value

        new_ctx = create_new_ag_gemm_context(
            **new_ctx_kwargs,
        )
        if rank == 0:
            if hasattr(new_ctx, "enable_row_tile_barrier"):
                print(
                    "tile-ready config: "
                    f"enabled={new_ctx.enable_row_tile_barrier}, "
                    f"tile_rows_per_chunk={new_ctx.tile_rows_per_chunk}, "
                    f"num_tile_chunks={new_ctx.num_tile_chunks}"
                )
            else:
                print("tile-ready config: kernel does not expose row-tile-ready fields; using compatible context path")

        if FLUX is not None and args.enable_flux:
            flux_output = torch.empty((M, n_per_rank), dtype=dtype, device=A.device)
            flux_option = FLUX.AllGatherOption()
            flux_option.mode = _get_flux_ring_mode(FLUX, args.flux_ring_mode)
            if args.flux_use_cuda_core_local is not None:
                flux_option.use_cuda_core_local = args.flux_use_cuda_core_local
            if args.flux_use_cuda_core_ag is not None:
                flux_option.use_cuda_core_ag = args.flux_use_cuda_core_ag
            flux_weight = B.contiguous() if args.trans_b else B.T.contiguous()
            flux_kernel = FLUX.AGKernel(
                pg,
                max(1, world_size // LOCAL_WORLD_SIZE),
                M,
                n_per_rank,
                K,
                dtype,
                output_dtype=dtype,
                use_pdl=args.flux_use_pdl,
            )
            if rank == 0:
                print(f"flux config: global_N={N}, local_N={n_per_rank}, transpose_weight={args.trans_b}")

        def _base_triton_func():
            return ag_gemm(A, B, ctx=base_ctx, autotune=args.autotune, debug=args.debug)

        def _new_triton_func():
            return new_ag_gemm(
                A,
                B,
                ctx=new_ctx,
                autotune=args.autotune,
                debug=args.debug,
                use_cooperative=args.cooperative_copy,
            )

        def _flux_func():
            flux_kernel.forward(
                A,
                flux_weight,
                bias=None,
                output=flux_output,
                input_scale=None,
                weight_scale=None,
                output_scale=None,
                fast_accum=False,
                gathered_input=None,
                transpose_weight=args.trans_b,
                all_gather_option=flux_option,
            )
            return flux_output

        # Warmup with fresh inputs.
        for _ in range(5):
            A, B = make_data(M, N, K, dtype, args.trans_b, pg)
            if flux_kernel is not None:
                flux_weight = B.contiguous() if args.trans_b else B.T.contiguous()
            _sync_all()
            C_base = _base_triton_func()
            _sync_all()
            C_new = _new_triton_func()
            if flux_kernel is not None:
                _sync_all()
                C_flux = _flux_func()
                _sync_all()

        C_golden = _torch_func()
        ok_base, err_base = _check_close(C_golden, C_base, "base_triton")
        ok_new, err_new = _check_close(C_golden, C_new, "new_triton")
        ok_flux, err_flux = (True, "")
        if flux_kernel is not None:
            _sync_all()
            ok_flux, err_flux = _check_close(C_golden, C_flux, "flux")

        local_failed = 0 if (ok_base and ok_new and ok_flux) else 1
        failed_tensor = torch.tensor([local_failed], device=A.device, dtype=torch.int32)
        torch.distributed.all_reduce(failed_tensor, op=torch.distributed.ReduceOp.MAX, group=pg)
        if local_failed:
            for msg in (err_base, err_new, err_flux):
                if msg:
                    dist_print(msg, need_sync=True, allowed_ranks=[rank])
        if failed_tensor.item() != 0:
            raise RuntimeError("Correctness check failed; see rank-local messages above.")

        # IMPORTANT:
        # Use a single profiling context for all implementations.
        # If we open multiple `group_profile` blocks with the same name, the
        # second one will overwrite the merged trace and you may only see the
        # last section in Perfetto.
        with group_profile(f"new_ag_gemm_perf_m_{M}_n_{N}_k_{K}_{os.environ['TORCHELASTIC_RUN_ID']}", args.profile,
                           group=TP_GROUP):
            with torch.profiler.record_function("base/ag_gemm"):
                perf_func(_base_triton_func, iters=args.iters, warmup_iters=args.warmup_iters)
            with torch.profiler.record_function("new/new_ag_gemm"):
                perf_func(_new_triton_func, iters=args.iters, warmup_iters=args.warmup_iters)
            with torch.profiler.record_function("torch/serial_allgather_gemm"):
                perf_func(_torch_func, iters=args.iters, warmup_iters=args.warmup_iters)
            if flux_kernel is not None:
                with torch.profiler.record_function("flux/ag_gemm"):
                    perf_func(_flux_func, iters=args.iters, warmup_iters=args.warmup_iters)

        wait_until_max_gpu_clock_or_warning(torch.cuda.current_device())
        _, base_triton_duration_ms = perf_func(_base_triton_func, iters=args.iters, warmup_iters=args.warmup_iters)
        wait_until_max_gpu_clock_or_warning(torch.cuda.current_device())
        _, new_triton_duration_ms = perf_func(_new_triton_func, iters=args.iters, warmup_iters=args.warmup_iters)
        if flux_kernel is not None:
            wait_until_max_gpu_clock_or_warning(torch.cuda.current_device())
            _, flux_duration_ms = perf_func(_flux_func, iters=args.iters, warmup_iters=args.warmup_iters)
    finally:
        if new_ctx is not None:
            new_ctx.finalize()
        if base_ctx is not None:
            base_ctx.finalize()

    wait_until_max_gpu_clock_or_warning(torch.cuda.current_device())
    _, torch_duration_ms = perf_func(_torch_func, iters=args.iters, warmup_iters=args.warmup_iters)

    msg = (
        f"Rank {rank} latency (ms): "
        f"base_triton total={base_triton_duration_ms:.2f}, "
        f"new_triton total={new_triton_duration_ms:.2f}, "
        f"torch total={torch_duration_ms:.2f}"
    )
    if flux_duration_ms is not None:
        msg += f", flux total={flux_duration_ms:.2f}"
    msg += (
        f", new_speedup {torch_duration_ms / new_triton_duration_ms:.2f}, "
        f"new_vs_base {base_triton_duration_ms / new_triton_duration_ms:.2f}"
    )
    if flux_duration_ms is not None:
        msg += (
            f", flux_speedup {torch_duration_ms / flux_duration_ms:.2f}, "
            f"flux_vs_base {base_triton_duration_ms / flux_duration_ms:.2f}, "
            f"flux_vs_new {new_triton_duration_ms / flux_duration_ms:.2f}"
        )

    dist_print(msg, need_sync=True, allowed_ranks=list(range(world_size)))

    return base_triton_duration_ms, new_triton_duration_ms, torch_duration_ms, flux_duration_ms


if __name__ == "__main__":
    args = parse_args()

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    TP_GROUP = initialize_distributed()
    LOCAL_WORLD_SIZE = int(os.environ.get("LOCAL_WORLD_SIZE", TP_GROUP.size()))
    if args.enable_flux:
        FLUX, FLUX_IMPORT_ERROR = _try_import_flux()
        if FLUX is None:
            if TP_GROUP.rank() == 0:
                print(f"[warn] FLUX import failed, skipping FLUX benchmark: {FLUX_IMPORT_ERROR}")
        else:
            try:
                FLUX.init_flux_shm(TP_GROUP)
                torch.cuda.synchronize()
                torch.distributed.barrier(TP_GROUP)
            except Exception as exc:
                if TP_GROUP.rank() == 0:
                    print(f"[warn] FLUX shm init failed, skipping FLUX benchmark: {exc}")
                FLUX = None
                FLUX_IMPORT_ERROR = exc

    base_ms, new_ms, torch_ms, flux_ms = perf_test(args.M, args.N, args.K, TP_GROUP)

    if args.dump_csv and TP_GROUP.rank() == 0:
        if not os.path.exists("csv"):
            os.makedirs("csv")
        csv_file = Path("csv") / f"perf_new_ag_gemm_{TP_GROUP.size()}_ranks.csv"

        with open(csv_file, "w") as fout:
            print(
                ",".join(
                    map(
                        str,
                        [
                            "Model",
                            "M",
                            "N",
                            "K",
                            "dist-triton ag gemm latency (ms)",
                            "new dist-triton ag gemm latency (ms)",
                            "torch ag gemm latency (ms)",
                            "flux ag gemm latency (ms)",
                            "new speed up",
                            "new vs base",
                            "flux speed up",
                            "flux vs base",
                            "flux vs new",
                        ],
                    )),
                file=fout,
            )
            csv_values = [
                base_ms,
                new_ms,
                torch_ms,
                flux_ms if flux_ms is not None else float("nan"),
                torch_ms / new_ms,
                base_ms / new_ms,
                torch_ms / flux_ms if flux_ms is not None else float("nan"),
                base_ms / flux_ms if flux_ms is not None else float("nan"),
                new_ms / flux_ms if flux_ms is not None else float("nan"),
            ]
            print(
                ",".join(["custom"] + list(map(
                    "{:d}".format,
                    [
                        args.M,
                        args.N,
                        args.K,
                    ],
                )) + list(
                    map(
                        "{:02f}".format,
                        csv_values,
                    ))),
                file=fout,
                flush=True,
            )
        print(f"csv file is dumped into {csv_file}")

    finalize_distributed()
