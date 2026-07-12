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

from triton_dist.kernels.nvidia.new_allgather_tileready import (
    launch_new_allgather_intra_node as tileready_launch_new_allgather_intra_node,
)


BASE_PATH = Path(__file__).resolve().parent / "bench_new_allgather_gemm_8rank.py"


def load_base_module():
    spec = importlib.util.spec_from_file_location("bench_new_allgather_gemm_8rank_base", BASE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load base benchmark module from {BASE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    base = load_base_module()

    # Patch the AG kernel module to use the tile-ready launcher whose signature
    # matches the arguments emitted by new_allgather_gemm_hurastic.new_ag_gemm.
    import triton_dist.kernels.nvidia.new_allgather_gemm_hurastic as hurastic

    hurastic.launch_new_allgather_intra_node = tileready_launch_new_allgather_intra_node

    base.args = base.parse_args()
    base.dtype = {"float16": base.torch.float16, "bfloat16": base.torch.bfloat16}[base.args.dtype]
    base.TP_GROUP = base.initialize_distributed()
    base.LOCAL_WORLD_SIZE = int(base.os.environ.get("LOCAL_WORLD_SIZE", base.TP_GROUP.size()))
    metrics = base.perf_test(base.args.M, base.args.N, base.args.K, base.TP_GROUP)

    if base.args.dump_csv and base.TP_GROUP.rank() == 0:
        if not base.os.path.exists("csv"):
            base.os.makedirs("csv")
        csv_file = base.Path("csv") / f"perf_new_ag_gemm_{base.TP_GROUP.size()}_ranks.csv"

        with open(csv_file, "w", encoding="utf-8") as fout:
            print(
                ",".join(
                    [
                        "Model",
                        "M",
                        "N",
                        "K",
                        "dist-triton ag gemm latency (ms)",
                        "new dist-triton ag gemm latency (ms)",
                        "torch ag gemm latency (ms)",
                        "new speed up",
                        "new vs base",
                        "enable_row_tile_barrier",
                        "tile_rows_per_chunk",
                        "num_tile_chunks",
                        "tile_barrier_present",
                        "first_ready_ts_ms",
                        "first_consumer_ts_ms",
                        "last_completion_ts_ms",
                        "consumer_wait_ms",
                        "consumer_ts_is_proxy",
                    ]
                ),
                file=fout,
            )
            print(
                ",".join(
                    ["custom", str(base.args.M), str(base.args.N), str(base.args.K)]
                    + [
                        f"{metrics['base_triton_duration_ms']:.6f}",
                        f"{metrics['new_triton_duration_ms']:.6f}",
                        f"{metrics['torch_duration_ms']:.6f}",
                        f"{metrics['new_speedup']:.6f}",
                        f"{metrics['new_vs_base']:.6f}",
                        str(metrics["enable_row_tile_barrier"]),
                        str(metrics["tile_rows_per_chunk"]),
                        str(metrics["num_tile_chunks"]),
                        str(metrics["tile_barrier_present"]),
                        f"{metrics['first_ready_ts_ms']:.6f}",
                        f"{metrics['first_consumer_ts_ms']:.6f}",
                        f"{metrics['last_completion_ts_ms']:.6f}",
                        f"{metrics['consumer_wait_ms']:.6f}",
                        str(metrics["consumer_ts_is_proxy"]),
                    ]
                ),
                file=fout,
                flush=True,
            )
        print(f"csv file is dumped into {csv_file}")

    base.finalize_distributed()


if __name__ == "__main__":
    main()
