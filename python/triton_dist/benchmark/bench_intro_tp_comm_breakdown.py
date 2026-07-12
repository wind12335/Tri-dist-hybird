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
import csv
import os
from pathlib import Path

import torch
import torch.distributed

from triton_dist.test.utils import LAYER_CONFIGS
from triton_dist.utils import (
    finalize_distributed,
    initialize_distributed,
    rand_tensor,
    wait_until_max_gpu_clock_or_warning,
)


PATTERN_LABELS = {
    "ag_gemm": "AllGather + GEMM",
    "gemm_rs": "GEMM + ReduceScatter",
    "gemm_ar": "GEMM + AllReduce",
}

DEFAULT_MODELS = [
    "LLaMA-7B",
    "LLaMA-3.1-8B",
    "LLaMA-3.1-70B",
    "Qwen2-72B",
    "GPT-3-175B",
    "LLaMA-3.1-405B",
]

MODEL_PLOT_LABELS = {
    "LLaMA-7B": "LLaMA-7B",
    "LLaMA-3.1-8B": "LLaMA-3.1-8B",
    "LLaMA-3.1-70B": "LLaMA-3.1-70B",
    "Qwen2-72B": "Qwen2-72B",
    "GPT-3-175B": "GPT-3-175B",
    "LLaMA-3.1-405B": "LLaMA-3.1-405B",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Measure serial torch+NCCL communication/computation breakdown for "
            "representative tensor-parallel operator chains and optionally plot "
            "the Introduction motivation figure."
        )
    )
    parser.add_argument("--M", type=int, default=8192)
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--warmup_iters", type=int, default=5)
    parser.add_argument(
        "--models",
        "--model",
        dest="models",
        type=str,
        default=",".join(DEFAULT_MODELS),
        help=(
            "Comma-separated model names from triton_dist.test.utils.LAYER_CONFIGS. "
            "You can pass any subset, e.g. "
            "'Qwen2-72B,GPT-3-175B,LLaMA-3.1-405B'."
        ),
    )
    parser.add_argument(
        "--patterns",
        type=str,
        default="ag_gemm,gemm_rs,gemm_ar",
        help="Comma-separated patterns from {ag_gemm, gemm_rs, gemm_ar}",
    )
    parser.add_argument("--dump_csv", action="store_true", default=False)
    parser.add_argument(
        "--csv_path",
        type=str,
        default="",
        help="Optional output CSV path. Defaults to csv/intro_tp_comm_breakdown_*.csv",
    )
    parser.add_argument("--plot", action="store_true", default=False)
    parser.add_argument(
        "--plot_dir",
        type=str,
        default="figures",
        help="Directory for combined and per-pattern figures",
    )
    parser.add_argument(
        "--plot_formats",
        type=str,
        default="png,pdf",
        help="Comma-separated formats for figures, e.g. png,pdf",
    )
    parser.add_argument(
        "--skip_separate_plots",
        action="store_true",
        default=False,
        help="Only save the combined 3-panel figure",
    )
    return parser.parse_args()


def sync_all(pg: torch.distributed.ProcessGroup):
    torch.cuda.synchronize()
    torch.distributed.barrier(pg, device_ids=[torch.cuda.current_device()])


def perf_func_lockstep(func, pg: torch.distributed.ProcessGroup, iters: int, warmup_iters: int):
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    stop_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    output = None
    for n in range(iters + warmup_iters):
        sync_all(pg)
        if n >= warmup_iters:
            start_events[n - warmup_iters].record()
        output = func()
        if n >= warmup_iters:
            stop_events[n - warmup_iters].record()
        sync_all(pg)

    duration_ms = 0.0
    for i in range(iters):
        stop_events[i].synchronize()
        duration_ms += start_events[i].elapsed_time(stop_events[i])
    return output, duration_ms / max(iters, 1)


def aggregate_metric(local_value: float, pg: torch.distributed.ProcessGroup) -> dict[str, float]:
    device = torch.cuda.current_device()
    value = torch.tensor([local_value], device=device, dtype=torch.float64)
    sum_value = value.clone()
    max_value = value.clone()
    min_value = value.clone()
    torch.distributed.all_reduce(sum_value, op=torch.distributed.ReduceOp.SUM, group=pg)
    torch.distributed.all_reduce(max_value, op=torch.distributed.ReduceOp.MAX, group=pg)
    torch.distributed.all_reduce(min_value, op=torch.distributed.ReduceOp.MIN, group=pg)
    world_size = pg.size()
    return {
        "mean": float(sum_value.item() / max(world_size, 1)),
        "max": float(max_value.item()),
        "min": float(min_value.item()),
    }


def make_ag_data(M: int, N: int, K: int, dtype: torch.dtype, pg: torch.distributed.ProcessGroup):
    rank = pg.rank()
    world_size = pg.size()
    assert M % world_size == 0
    assert N % world_size == 0
    m_per_rank = M // world_size
    n_per_rank = N // world_size
    scale = (rank + 1) * 0.01
    device = torch.cuda.current_device()

    a = rand_tensor([m_per_rank, K], dtype=dtype, device=device) * scale
    b = (rand_tensor([n_per_rank, K], dtype=dtype, device=device) * scale).T.contiguous()
    return a, b


def torch_ag_gemm(pg: torch.distributed.ProcessGroup, a_shard: torch.Tensor, b_local: torch.Tensor):
    m_per_rank, k = a_shard.shape
    a_full = torch.empty([m_per_rank * pg.size(), k], dtype=a_shard.dtype, device=a_shard.device)
    torch.distributed.all_gather_into_tensor(a_full, a_shard, group=pg)
    return torch.matmul(a_full, b_local)


def make_rs_data(M: int, N: int, K: int, dtype: torch.dtype, pg: torch.distributed.ProcessGroup):
    rank = pg.rank()
    world_size = pg.size()
    assert K % world_size == 0
    k_per_rank = K // world_size
    scale = (rank + 1) * 0.01
    device = torch.cuda.current_device()

    a = rand_tensor([M, k_per_rank], dtype=dtype, device=device) * scale
    b = (rand_tensor([N, k_per_rank], dtype=dtype, device=device) * scale).T.contiguous()
    return a, b


def torch_gemm_rs(pg: torch.distributed.ProcessGroup, a_local: torch.Tensor, b_local: torch.Tensor):
    m, _ = a_local.shape
    _, n = b_local.shape
    partial = torch.matmul(a_local, b_local)
    output = torch.empty((m // pg.size(), n), dtype=partial.dtype, device=a_local.device)
    torch.distributed.reduce_scatter_tensor(output, partial, group=pg)
    return output


def make_ar_data(M: int, N: int, K: int, dtype: torch.dtype, pg: torch.distributed.ProcessGroup):
    rank = pg.rank()
    world_size = pg.size()
    assert K % world_size == 0
    local_k = K // world_size
    scale = 0.01 * (rank + 1)
    device = torch.cuda.current_device()

    a = rand_tensor((M, local_k), dtype=dtype, device=device) * scale
    weight = rand_tensor((N, local_k), dtype=dtype, device=device) * scale
    return a, weight


def torch_gemm_ar(pg: torch.distributed.ProcessGroup, a_local: torch.Tensor, weight_local: torch.Tensor):
    output = torch.matmul(a_local, weight_local.T)
    torch.distributed.all_reduce(output, group=pg)
    return output


def measure_ag_gemm(M: int, N: int, K: int, dtype: torch.dtype, pg: torch.distributed.ProcessGroup, iters: int, warmup_iters: int):
    a_shard, b_local = make_ag_data(M, N, K, dtype, pg)
    m_per_rank, k = a_shard.shape
    a_full = torch.empty([m_per_rank * pg.size(), k], dtype=a_shard.dtype, device=a_shard.device)
    torch.distributed.all_gather_into_tensor(a_full, a_shard, group=pg)
    sync_all(pg)

    def _serial_total():
        return torch_ag_gemm(pg, a_shard, b_local)

    def _compute_only():
        return torch.matmul(a_full, b_local)

    def _comm_only():
        torch.distributed.all_gather_into_tensor(a_full, a_shard, group=pg)
        return a_full

    wait_until_max_gpu_clock_or_warning(torch.cuda.current_device())
    _, total_ms = perf_func_lockstep(_serial_total, pg=pg, iters=iters, warmup_iters=warmup_iters)
    wait_until_max_gpu_clock_or_warning(torch.cuda.current_device())
    _, compute_ms = perf_func_lockstep(_compute_only, pg=pg, iters=iters, warmup_iters=warmup_iters)
    wait_until_max_gpu_clock_or_warning(torch.cuda.current_device())
    _, comm_ms = perf_func_lockstep(_comm_only, pg=pg, iters=iters, warmup_iters=warmup_iters)
    return total_ms, compute_ms, comm_ms


def measure_gemm_rs(M: int, N: int, K: int, dtype: torch.dtype, pg: torch.distributed.ProcessGroup, iters: int, warmup_iters: int):
    a_local, b_local = make_rs_data(M, N, K, dtype, pg)
    partial = torch.matmul(a_local, b_local)
    rs_output = torch.empty((M // pg.size(), N), dtype=dtype, device=a_local.device)
    sync_all(pg)

    def _serial_total():
        return torch_gemm_rs(pg, a_local, b_local)

    def _compute_only():
        return torch.matmul(a_local, b_local)

    def _comm_only():
        torch.distributed.reduce_scatter_tensor(rs_output, partial, group=pg)
        return rs_output

    wait_until_max_gpu_clock_or_warning(torch.cuda.current_device())
    _, total_ms = perf_func_lockstep(_serial_total, pg=pg, iters=iters, warmup_iters=warmup_iters)
    wait_until_max_gpu_clock_or_warning(torch.cuda.current_device())
    _, compute_ms = perf_func_lockstep(_compute_only, pg=pg, iters=iters, warmup_iters=warmup_iters)
    wait_until_max_gpu_clock_or_warning(torch.cuda.current_device())
    _, comm_ms = perf_func_lockstep(_comm_only, pg=pg, iters=iters, warmup_iters=warmup_iters)
    return total_ms, compute_ms, comm_ms


def measure_gemm_ar(M: int, N: int, K: int, dtype: torch.dtype, pg: torch.distributed.ProcessGroup, iters: int, warmup_iters: int):
    a_local, weight_local = make_ar_data(M, N, K, dtype, pg)
    partial = torch.matmul(a_local, weight_local.T)
    sync_all(pg)

    def _serial_total():
        return torch_gemm_ar(pg, a_local, weight_local)

    def _compute_only():
        return torch.matmul(a_local, weight_local.T)

    def _comm_only():
        output = partial.clone()
        torch.distributed.all_reduce(output, group=pg)
        return output

    wait_until_max_gpu_clock_or_warning(torch.cuda.current_device())
    _, total_ms = perf_func_lockstep(_serial_total, pg=pg, iters=iters, warmup_iters=warmup_iters)
    wait_until_max_gpu_clock_or_warning(torch.cuda.current_device())
    _, compute_ms = perf_func_lockstep(_compute_only, pg=pg, iters=iters, warmup_iters=warmup_iters)
    wait_until_max_gpu_clock_or_warning(torch.cuda.current_device())
    _, comm_ms = perf_func_lockstep(_comm_only, pg=pg, iters=iters, warmup_iters=warmup_iters)
    return total_ms, compute_ms, comm_ms


def measure_pattern(pattern: str, M: int, N: int, K: int, dtype: torch.dtype, pg: torch.distributed.ProcessGroup, iters: int, warmup_iters: int):
    if pattern == "ag_gemm":
        return measure_ag_gemm(M, N, K, dtype, pg, iters, warmup_iters)
    if pattern == "gemm_rs":
        return measure_gemm_rs(M, N, K, dtype, pg, iters, warmup_iters)
    if pattern == "gemm_ar":
        return measure_gemm_ar(M, N, K, dtype, pg, iters, warmup_iters)
    raise ValueError(f"Unsupported pattern: {pattern}")


def build_rows(results: list[dict]) -> list[list[str]]:
    rows = []
    for item in results:
        rows.append([
            item["pattern"],
            item["pattern_label"],
            item["model"],
            str(item["M"]),
            str(item["N"]),
            str(item["K"]),
            f"{item['serial_total_ms_mean']:.6f}",
            f"{item['serial_total_ms_max']:.6f}",
            f"{item['compute_only_ms_mean']:.6f}",
            f"{item['compute_only_ms_max']:.6f}",
            f"{item['comm_only_ms_mean']:.6f}",
            f"{item['comm_only_ms_max']:.6f}",
            f"{item['component_compute_share']:.6f}",
            f"{item['component_comm_share']:.6f}",
            f"{item['serial_compute_share']:.6f}",
            f"{item['serial_comm_share']:.6f}",
        ])
    return rows


def write_csv(csv_path: Path, results: list[dict]):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as fout:
        writer = csv.writer(fout)
        writer.writerow([
            "pattern",
            "pattern_label",
            "model",
            "M",
            "N",
            "K",
            "serial_total_ms_mean",
            "serial_total_ms_max",
            "compute_only_ms_mean",
            "compute_only_ms_max",
            "comm_only_ms_mean",
            "comm_only_ms_max",
            "component_compute_share",
            "component_comm_share",
            "serial_compute_share",
            "serial_comm_share",
        ])
        writer.writerows(build_rows(results))


def plot_results(results: list[dict], plot_dir: Path, plot_formats: list[str], skip_separate_plots: bool):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import PercentFormatter
        import numpy as np
    except Exception as exc:
        print(f"[warn] matplotlib is unavailable, skipping plot generation: {exc}", flush=True)
        return

    plot_dir.mkdir(parents=True, exist_ok=True)
    patterns = list(dict.fromkeys(item["pattern"] for item in results))
    model_order = [item["model"] for item in results if item["pattern"] == patterns[0]]
    model_order = list(dict.fromkeys(model_order))
    num_models = len(model_order)
    color_compute = "#4C78A8"
    color_comm = "#F58518"

    def _subset(pattern: str) -> list[dict]:
        lookup = {(item["pattern"], item["model"]): item for item in results}
        return [lookup[(pattern, model)] for model in model_order]

    def _draw(ax, pattern: str, subset: list[dict], *, show_ylabel: bool):
        y = np.arange(len(subset))
        compute = np.array([100.0 * item["component_compute_share"] for item in subset])
        comm = np.array([100.0 * item["component_comm_share"] for item in subset])
        bar_height = 0.48
        ax.barh(y, compute, height=bar_height, color=color_compute, edgecolor="white", linewidth=0.8, label="Computation")
        ax.barh(
            y,
            comm,
            height=bar_height,
            left=compute,
            color=color_comm,
            edgecolor="white",
            linewidth=0.8,
            label="Communication",
        )
        ax.set_yticks(y)
        ax.set_yticklabels([MODEL_PLOT_LABELS.get(item["model"], item["model"]) for item in subset], fontsize=10)
        ax.invert_yaxis()
        ax.set_xlim(0, 100)
        ax.xaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_title(PATTERN_LABELS[pattern], fontsize=12, pad=8)
        if show_ylabel:
            ax.set_ylabel("Model", fontsize=11)
        else:
            ax.tick_params(axis="y", left=False, labelleft=False)
        ax.set_xlabel("Component Share", fontsize=11)
        ax.grid(axis="x", linestyle="--", alpha=0.3)
        ax.tick_params(axis="x", labelsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig_width = 4.75 * len(patterns)
    fig_height = max(3.9, 1.55 + 0.72 * num_models)
    fig, axes = plt.subplots(1, len(patterns), figsize=(fig_width, fig_height), sharey=True, constrained_layout=False)
    if len(patterns) == 1:
        axes = [axes]
    for idx, (ax, pattern) in enumerate(zip(axes, patterns)):
        _draw(ax, pattern, _subset(pattern), show_ylabel=(idx == 0))
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=color_compute),
        plt.Rectangle((0, 0), 1, 1, color=color_comm),
    ]
    fig.subplots_adjust(left=0.08, right=0.992, bottom=0.18, top=0.81, wspace=0.14)
    fig.legend(
        handles,
        ["Computation", "Communication"],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=2,
        frameon=False,
        fontsize=11,
        handlelength=1.6,
        columnspacing=1.8,
    )

    for fmt in plot_formats:
        fig.savefig(plot_dir / f"intro_tp_comm_breakdown_combined.{fmt}", dpi=220, bbox_inches="tight")
    plt.close(fig)

    if skip_separate_plots:
        return

    for pattern in patterns:
        subset = _subset(pattern)
        fig_single_height = max(3.9, 1.7 + 0.75 * len(subset))
        fig_single, ax_single = plt.subplots(1, 1, figsize=(6.2, fig_single_height), constrained_layout=False)
        _draw(ax_single, pattern, subset, show_ylabel=True)
        handles = [
            plt.Rectangle((0, 0), 1, 1, color=color_compute),
            plt.Rectangle((0, 0), 1, 1, color=color_comm),
        ]
        fig_single.subplots_adjust(left=0.18, right=0.98, bottom=0.18, top=0.82)
        ax_single.legend(
            handles,
            ["Computation", "Communication"],
            loc="upper center",
            bbox_to_anchor=(0.5, 1.06),
            ncol=2,
            frameon=False,
            fontsize=10,
        )
        for fmt in plot_formats:
            fig_single.savefig(plot_dir / f"intro_tp_comm_breakdown_{pattern}.{fmt}", dpi=220, bbox_inches="tight")
        plt.close(fig_single)


if __name__ == "__main__":
    args = parse_args()
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    pg = initialize_distributed(initialize_shmem=False)
    rank = pg.rank()

    model_names = [name.strip() for name in args.models.split(",") if name.strip()]
    pattern_names = [name.strip() for name in args.patterns.split(",") if name.strip()]
    invalid_models = [name for name in model_names if name not in LAYER_CONFIGS]
    invalid_patterns = [name for name in pattern_names if name not in PATTERN_LABELS]
    if invalid_models:
        raise ValueError(f"Unknown model(s): {invalid_models}. Available: {sorted(LAYER_CONFIGS.keys())}")
    if invalid_patterns:
        raise ValueError(f"Unknown pattern(s): {invalid_patterns}. Available: {sorted(PATTERN_LABELS.keys())}")

    results: list[dict] = []
    try:
        for pattern in pattern_names:
            for model_name in model_names:
                config = LAYER_CONFIGS[model_name]
                if rank == 0:
                    print(
                        f"[intro-breakdown] measuring pattern={pattern} ({PATTERN_LABELS[pattern]}), "
                        f"model={model_name}, M={args.M}, N={config['N']}, K={config['K']}",
                        flush=True,
                    )

                total_ms_local, compute_ms_local, comm_ms_local = measure_pattern(
                    pattern,
                    args.M,
                    config["N"],
                    config["K"],
                    dtype,
                    pg,
                    args.iters,
                    args.warmup_iters,
                )

                total_ms_stats = aggregate_metric(total_ms_local, pg)
                compute_ms_stats = aggregate_metric(compute_ms_local, pg)
                comm_ms_stats = aggregate_metric(comm_ms_local, pg)

                component_den = max(compute_ms_stats["mean"] + comm_ms_stats["mean"], 1e-9)
                serial_den = max(total_ms_stats["mean"], 1e-9)
                record = {
                    "pattern": pattern,
                    "pattern_label": PATTERN_LABELS[pattern],
                    "model": model_name,
                    "M": args.M,
                    "N": config["N"],
                    "K": config["K"],
                    "serial_total_ms_mean": total_ms_stats["mean"],
                    "serial_total_ms_max": total_ms_stats["max"],
                    "compute_only_ms_mean": compute_ms_stats["mean"],
                    "compute_only_ms_max": compute_ms_stats["max"],
                    "comm_only_ms_mean": comm_ms_stats["mean"],
                    "comm_only_ms_max": comm_ms_stats["max"],
                    "component_compute_share": compute_ms_stats["mean"] / component_den,
                    "component_comm_share": comm_ms_stats["mean"] / component_den,
                    "serial_compute_share": compute_ms_stats["mean"] / serial_den,
                    "serial_comm_share": comm_ms_stats["mean"] / serial_den,
                }
                results.append(record)

                if rank == 0:
                    print(
                        f"[intro-breakdown] result pattern={pattern}, model={model_name}: "
                        f"serial_total={record['serial_total_ms_mean']:.4f} ms, "
                        f"compute_only={record['compute_only_ms_mean']:.4f} ms, "
                        f"comm_only={record['comm_only_ms_mean']:.4f} ms, "
                        f"component_comm_share={100.0 * record['component_comm_share']:.2f}%, "
                        f"serial_comm_share={100.0 * record['serial_comm_share']:.2f}%",
                        flush=True,
                    )

        if rank == 0:
            csv_path = Path(args.csv_path) if args.csv_path else (
                Path("csv") / f"intro_tp_comm_breakdown_{pg.size()}ranks_M{args.M}_{args.dtype}.csv"
            )
            if args.dump_csv or args.plot:
                write_csv(csv_path, results)
                print(f"[intro-breakdown] csv written to {csv_path}", flush=True)
            if args.plot:
                plot_formats = [fmt.strip() for fmt in args.plot_formats.split(",") if fmt.strip()]
                plot_results(results, Path(args.plot_dir), plot_formats, args.skip_separate_plots)
                print(f"[intro-breakdown] figures written to {Path(args.plot_dir)}", flush=True)
    finally:
        finalize_distributed()
