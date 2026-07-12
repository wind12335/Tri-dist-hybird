################################################################################
#
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software,
# subject to the following conditions:
#
################################################################################

from __future__ import annotations

import importlib.util
from pathlib import Path


BASE_PATH = Path(__file__).resolve().parent / "bench_ag_ready_granularity_sweep.py"


def load_base_module():
    spec = importlib.util.spec_from_file_location("bench_ag_ready_granularity_sweep_base", BASE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load base sweep module from {BASE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def patched_build_cmd(base_module, args, point):
    cmd = [
        args.torchrun_bin,
        "--nproc_per_node",
        str(args.nproc_per_node),
        str(base_module.AG_SCRIPT),
        "--M",
        str(args.M),
        "--N",
        str(args.N),
        "--K",
        str(args.K),
        "--iters",
        str(args.iters),
        "--warmup_iters",
        str(args.warmup_iters),
        "--dtype",
        args.dtype,
        "--target_chunks_per_rank",
        str(args.target_chunks_per_rank),
        "--min_tile_rows_per_chunk",
        str(args.min_tile_rows_per_chunk),
        "--min_m_per_rank_for_tile_ready",
        str(args.min_m_per_rank_for_tile_ready),
        "--copy_sms",
        str(args.copy_sms),
        "--tile_rows_per_chunk",
        str(point.tile_rows_per_chunk),
        "--dump_csv",
    ]
    if args.autotune:
        cmd.append("--autotune")
    cmd.append("--trans_b" if args.trans_b else "--no-trans_b")
    cmd.append("--enable_tile_ready" if point.enable_tile_ready else "--no-enable_tile_ready")
    cmd.append("--cooperative_copy" if args.cooperative_copy else "--no-cooperative_copy")
    if args.profile:
        cmd.append("--profile")
    return cmd


def main():
    base_module = load_base_module()
    base_module.build_cmd = lambda args, point: patched_build_cmd(base_module, args, point)
    base_module.main()


if __name__ == "__main__":
    main()
