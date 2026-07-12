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
import math
from collections import defaultdict
from pathlib import Path
# python python/triton_dist/benchmark/plot_nvshmem_feasibility.py   --input_csv csv/nvshmem_baseline_rs_fixedM819216384.csv   --output_dir plots/nvshmem_baseline_rs_fixedM819216384_v5   --formats png,pdf   --skip_heatmap   --skip_frontier   --skip_memory

STATUS_ORDER = [
    "success",
    "nvshmem_oom",
    "shape_mismatch",
    "timeout",
    "crash",
    "missing_json",
    "error",
]

STATUS_COLORS = {
    "success": "#2ca02c",
    "nvshmem_oom": "#d62728",
    "shape_mismatch": "#ff7f0e",
    "timeout": "#7f7f7f",
    "crash": "#9467bd",
    "missing_json": "#8c564b",
    "error": "#1f77b4",
    "unknown": "#bcbd22",
    "missing": "#ffffff",
}

STATUS_LABELS = {
    "success": "OK",
    "nvshmem_oom": "OOM",
    "shape_mismatch": "SHP",
    "timeout": "TO",
    "crash": "CR",
    "missing_json": "MISS",
    "error": "ERR",
    "unknown": "UNK",
}

IMPL_LABELS = {
    "baseline_rs": "Baseline GEMM-ReduceScatter",
    "new_rs_v5": "Windowed Frontier GEMM-ReduceScatter",
    "baseline_ar": "Baseline GEMM-AllReduce",
    "new_ar_v23": "Compact Windowed GEMM-AllReduce v23",
}

IMPL_LINE_COLORS = {
    "baseline_rs": "#d62728",
    "new_rs_v5": "#1f77b4",
    "baseline_ar": "#8c564b",
    "new_ar_v23": "#17becf",
}

K_MARKERS = ["o", "s", "^", "D", "P", "X", "v", "<", ">", "*", "h", "8"]


def parse_args():
    parser = argparse.ArgumentParser(description="Plot NVSHMEM feasibility CSV outputs.")
    parser.add_argument("--input_csv", required=True, type=str)
    parser.add_argument("--output_dir", default="", type=str)
    parser.add_argument("--formats", default="png,pdf", type=str)
    parser.add_argument("--skip_heatmap", action="store_true", default=False)
    parser.add_argument("--skip_frontier", action="store_true", default=False)
    parser.add_argument("--skip_memory", action="store_true", default=False)
    parser.add_argument("--heatmap_no_annotate", action="store_true", default=False)
    return parser.parse_args()


def safe_int(value):
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(float(text))
    except Exception:
        return None


def safe_float(value):
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except Exception:
        return None


def load_rows(input_csv: Path) -> list[dict]:
    rows = []
    with open(input_csv, "r", encoding="utf-8-sig", newline="") as fin:
        reader = csv.DictReader(fin)
        for row in reader:
            parsed = dict(row)
            parsed["M"] = safe_int(row.get("M"))
            parsed["N"] = safe_int(row.get("N"))
            parsed["K"] = safe_int(row.get("K"))
            parsed["world_size"] = safe_int(row.get("world_size"))
            parsed["local_world_size"] = safe_int(row.get("local_world_size"))
            parsed["dtype"] = (row.get("dtype") or "").strip()
            parsed["estimated_scatter_buf_bytes"] = safe_float(row.get("estimated_scatter_buf_bytes"))
            parsed["estimated_gemm_out_buf_bytes"] = safe_float(row.get("estimated_gemm_out_buf_bytes"))
            parsed["estimated_rs_per_node_buf_bytes"] = safe_float(row.get("estimated_rs_per_node_buf_bytes"))
            parsed["estimated_p2p_buf_bytes"] = safe_float(row.get("estimated_p2p_buf_bytes"))
            parsed["actual_scatter_buf_bytes"] = safe_float(row.get("actual_scatter_buf_bytes"))
            parsed["actual_gemm_out_buf_bytes"] = safe_float(row.get("actual_gemm_out_buf_bytes"))
            parsed["actual_rs_per_node_buf_bytes"] = safe_float(row.get("actual_rs_per_node_buf_bytes"))
            parsed["actual_p2p_buf_bytes"] = safe_float(row.get("actual_p2p_buf_bytes"))
            parsed["estimated_symm_data_bytes"] = safe_float(row.get("estimated_symm_data_bytes"))
            parsed["estimated_symm_total_bytes"] = safe_float(row.get("estimated_symm_total_bytes"))
            parsed["actual_symm_data_bytes"] = safe_float(row.get("actual_symm_data_bytes"))
            parsed["actual_symm_total_bytes"] = safe_float(row.get("actual_symm_total_bytes"))
            parsed["effective_compaction_ratio"] = safe_float(row.get("effective_compaction_ratio"))
            parsed["estimated_compaction_ratio"] = safe_float(row.get("estimated_compaction_ratio"))
            parsed["status"] = (row.get("status") or "unknown").strip() or "unknown"
            rows.append(parsed)
    return rows


def unique_sorted(values):
    return sorted({x for x in values if x is not None})


def impl_label(impl: str) -> str:
    return IMPL_LABELS.get(impl, impl)


def bytes_per_elem_from_dtype(dtype_name: str) -> int | None:
    mapping = {
        "float16": 2,
        "bfloat16": 2,
        "float32": 4,
        "int32": 4,
        "int64": 8,
    }
    return mapping.get(dtype_name)


def status_rank(status: str) -> int:
    if status in STATUS_ORDER:
        return STATUS_ORDER.index(status)
    return len(STATUS_ORDER)


def pick_row_for_cell(rows: list[dict]) -> dict | None:
    if not rows:
        return None
    return sorted(rows, key=lambda item: status_rank(item.get("status", "unknown")))[0]


def plot_heatmaps(rows: list[dict], output_dir: Path, formats: list[str], annotate: bool):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.colors import ListedColormap
        from matplotlib.patches import Patch
    except Exception as exc:
        print(f"[warn] matplotlib is unavailable, skipping feasibility heatmaps: {exc}", flush=True)
        return

    by_k = defaultdict(list)
    for row in rows:
        by_k[row["K"]].append(row)

    status_categories = STATUS_ORDER + ["unknown"]
    cmap = ListedColormap([STATUS_COLORS[key] for key in status_categories] + [STATUS_COLORS["missing"]])
    status_to_code = {status: idx for idx, status in enumerate(status_categories)}
    missing_code = len(status_categories)

    legend_handles = [Patch(facecolor=STATUS_COLORS[key], edgecolor="#444444", label=key) for key in status_categories]
    legend_handles.append(Patch(facecolor=STATUS_COLORS["missing"], edgecolor="#444444", label="missing"))

    for k_value, subset in sorted(by_k.items(), key=lambda item: (item[0] is None, item[0])):
        impls = unique_sorted(row.get("impl") for row in subset)
        if not impls:
            continue
        cols = min(2, max(1, len(impls)))
        rows_n = math.ceil(len(impls) / cols)
        fig, axes = plt.subplots(rows_n, cols, figsize=(5.2 * cols, 4.4 * rows_n), constrained_layout=False)
        if not isinstance(axes, (list, tuple, np.ndarray)):
            axes = [axes]
        else:
            axes = list(np.ravel(axes))

        for ax in axes[len(impls):]:
            ax.axis("off")

        for ax, impl in zip(axes, impls):
            impl_rows = [row for row in subset if row.get("impl") == impl]
            Ms = unique_sorted(row.get("M") for row in impl_rows)
            Ns = unique_sorted(row.get("N") for row in impl_rows)
            matrix = np.full((len(Ms), len(Ns)), missing_code, dtype=np.int32)
            labels = [["" for _ in Ns] for _ in Ms]

            cell_rows = defaultdict(list)
            for row in impl_rows:
                cell_rows[(row["M"], row["N"])].append(row)

            for mi, m_value in enumerate(Ms):
                for ni, n_value in enumerate(Ns):
                    row = pick_row_for_cell(cell_rows.get((m_value, n_value), []))
                    if row is None:
                        continue
                    status = row.get("status", "unknown")
                    matrix[mi, ni] = status_to_code.get(status, status_to_code["unknown"])
                    labels[mi][ni] = STATUS_LABELS.get(status, "UNK")

            ax.imshow(matrix, cmap=cmap, aspect="auto", origin="lower", vmin=0, vmax=missing_code)
            ax.set_title(impl_label(impl), fontsize=11)
            ax.set_xlabel("N")
            ax.set_ylabel("M")
            ax.set_xticks(range(len(Ns)))
            ax.set_xticklabels([str(x) for x in Ns], rotation=35, ha="right")
            ax.set_yticks(range(len(Ms)))
            ax.set_yticklabels([str(x) for x in Ms])
            ax.set_xticks([x - 0.5 for x in range(1, len(Ns))], minor=True)
            ax.set_yticks([y - 0.5 for y in range(1, len(Ms))], minor=True)
            ax.grid(which="minor", color="#d9d9d9", linewidth=0.8)
            ax.tick_params(which="minor", bottom=False, left=False)

            if annotate and len(Ms) * len(Ns) <= 64:
                for mi, _ in enumerate(Ms):
                    for ni, _ in enumerate(Ns):
                        code = matrix[mi, ni]
                        if code == missing_code:
                            continue
                        text_color = "white" if labels[mi][ni] in ("OOM", "CR", "ERR") else "black"
                        ax.text(ni, mi, labels[mi][ni], ha="center", va="center", fontsize=8.5, color=text_color)

        title = "NVSHMEM Feasibility Status Heatmap"
        if k_value is not None:
            title += f" (K={k_value})"
        fig.suptitle(title, fontsize=13)
        fig.subplots_adjust(left=0.08, right=0.995, top=0.80, bottom=0.18, wspace=0.18, hspace=0.28)
        fig.legend(handles=legend_handles,
                   loc="upper center",
                   bbox_to_anchor=(0.5, 0.93),
                   ncol=min(len(legend_handles), 3),
                   frameon=False,
                   fontsize=10)

        suffix = f"_K{k_value}" if k_value is not None else ""
        for fmt in formats:
            fig.savefig(output_dir / f"nvshmem_feasibility_heatmap{suffix}.{fmt}", dpi=220, bbox_inches="tight")
        plt.close(fig)


def plot_frontiers(rows: list[dict], output_dir: Path, formats: list[str]):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except Exception as exc:
        print(f"[warn] matplotlib is unavailable, skipping feasibility frontier plots: {exc}", flush=True)
        return

    by_k = defaultdict(list)
    for row in rows:
        by_k[row["K"]].append(row)

    status_handles = [
        Line2D([0], [0], color="#333333", marker="o", linestyle="none", markerfacecolor=STATUS_COLORS["success"], label="success"),
        Line2D([0], [0], color="#333333", marker="x", linestyle="none", label="no success at this N"),
    ]

    for k_value, subset in sorted(by_k.items(), key=lambda item: (item[0] is None, item[0])):
        impls = unique_sorted(row.get("impl") for row in subset)
        all_ns = unique_sorted(row.get("N") for row in subset)
        if len(all_ns) <= 1:
            continue

        fig, ax = plt.subplots(1, 1, figsize=(7.2, 4.8), constrained_layout=False)
        plotted_any = False

        for impl in impls:
            impl_rows = [row for row in subset if row.get("impl") == impl]
            xs = []
            ys = []
            no_success_x = []
            for n_value in all_ns:
                candidates = [row for row in impl_rows if row.get("N") == n_value]
                success_ms = sorted(row["M"] for row in candidates if row.get("status") == "success" and row.get("M") is not None)
                if success_ms:
                    xs.append(n_value)
                    ys.append(success_ms[-1])
                else:
                    no_success_x.append(n_value)
            if xs:
                plotted_any = True
                ax.plot(xs,
                        ys,
                        color=IMPL_LINE_COLORS.get(impl, "#333333"),
                        marker="o",
                        linewidth=2.1,
                        markersize=5.5,
                        label=impl_label(impl))
            if no_success_x:
                ax.scatter(no_success_x,
                           [0] * len(no_success_x),
                           color=IMPL_LINE_COLORS.get(impl, "#333333"),
                           marker="x",
                           s=48,
                           alpha=0.9)

        if not plotted_any:
            plt.close(fig)
            continue

        ax.set_title(f"Maximum Successful M vs N (K={k_value})" if k_value is not None else "Maximum Successful M vs N")
        ax.set_xlabel("N")
        ax.set_ylabel("Max Successful M")
        ax.grid(alpha=0.28, linestyle="--", linewidth=0.8)
        ax.legend(loc="upper left", frameon=False, fontsize=10)
        fig.subplots_adjust(left=0.12, right=0.98, top=0.88, bottom=0.16)

        suffix = f"_K{k_value}" if k_value is not None else ""
        for fmt in formats:
            fig.savefig(output_dir / f"nvshmem_feasibility_frontier{suffix}.{fmt}", dpi=220, bbox_inches="tight")
        plt.close(fig)


def plot_memory_curves(rows: list[dict], output_dir: Path, formats: list[str]):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except Exception as exc:
        print(f"[warn] matplotlib is unavailable, skipping feasibility memory plots: {exc}", flush=True)
        return

    by_kn = defaultdict(list)
    for row in rows:
        by_kn[(row["K"], row["N"])].append(row)

    status_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=STATUS_COLORS["success"], markeredgecolor="#333333",
               markersize=7, label="success"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=STATUS_COLORS["nvshmem_oom"], markeredgecolor="#333333",
               markersize=7, label="nvshmem_oom"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=STATUS_COLORS["timeout"], markeredgecolor="#333333",
               markersize=7, label="timeout/crash/error"),
    ]

    for (k_value, n_value), subset in sorted(by_kn.items(), key=lambda item: ((item[0][0] is None), item[0][0], (item[0][1] is None), item[0][1])):
        Ms = unique_sorted(row.get("M") for row in subset)
        if len(Ms) <= 1:
            continue

        impls = unique_sorted(row.get("impl") for row in subset)
        fig, ax = plt.subplots(1, 1, figsize=(7.6, 4.9), constrained_layout=False)
        plotted_any = False

        for impl in impls:
            impl_rows = sorted([row for row in subset if row.get("impl") == impl and row.get("M") is not None],
                               key=lambda row: row["M"])
            xs = [row["M"] for row in impl_rows if row.get("estimated_symm_total_bytes") is not None]
            ys = [row["estimated_symm_total_bytes"] / (1024**3) for row in impl_rows if row.get("estimated_symm_total_bytes") is not None]
            if xs and ys:
                plotted_any = True
                ax.plot(xs,
                        ys,
                        color=IMPL_LINE_COLORS.get(impl, "#333333"),
                        linewidth=2.0,
                        marker="o",
                        markersize=4.5,
                        alpha=0.95,
                        label=impl_label(impl))

            for row in impl_rows:
                if row.get("estimated_symm_total_bytes") is None or row.get("M") is None:
                    continue
                status = row.get("status", "unknown")
                if status == "success":
                    status_key = "success"
                elif status == "nvshmem_oom":
                    status_key = "nvshmem_oom"
                else:
                    status_key = "timeout"
                ax.scatter(row["M"],
                           row["estimated_symm_total_bytes"] / (1024**3),
                           s=58,
                           facecolors=STATUS_COLORS.get(status_key, STATUS_COLORS["unknown"]),
                           edgecolors="#222222",
                           linewidths=0.75,
                           zorder=4)
                if row.get("actual_symm_total_bytes") is not None:
                    ax.scatter(row["M"],
                               row["actual_symm_total_bytes"] / (1024**3),
                               s=68,
                               facecolors="none",
                               edgecolors=IMPL_LINE_COLORS.get(impl, "#333333"),
                               linewidths=1.2,
                               zorder=5)

        if not plotted_any:
            plt.close(fig)
            continue

        title = "Estimated Symmetric Memory vs M"
        meta = []
        if k_value is not None:
            meta.append(f"K={k_value}")
        if n_value is not None:
            meta.append(f"N={n_value}")
        if meta:
            title += " (" + ", ".join(meta) + ")"
        ax.set_title(title)
        ax.set_xlabel("M")
        ax.set_ylabel("Estimated symmetric memory (GiB)")
        ax.grid(alpha=0.28, linestyle="--", linewidth=0.8)

        handles, labels = ax.get_legend_handles_labels()
        fig.subplots_adjust(left=0.12, right=0.98, top=0.86, bottom=0.18)
        legend_impl = ax.legend(handles, labels, loc="upper left", frameon=False, fontsize=10)
        ax.add_artist(legend_impl)
        ax.legend(handles=status_handles,
                  loc="upper right",
                  frameon=False,
                  fontsize=9,
                  title="Marker Status")

        suffix_parts = []
        if k_value is not None:
            suffix_parts.append(f"K{k_value}")
        if n_value is not None:
            suffix_parts.append(f"N{n_value}")
        suffix = "_" + "_".join(suffix_parts) if suffix_parts else ""
        for fmt in formats:
            fig.savefig(output_dir / f"nvshmem_feasibility_memory{suffix}.{fmt}", dpi=220, bbox_inches="tight")
        plt.close(fig)


def plot_fixed_m_nk_heatmaps(rows: list[dict], output_dir: Path, formats: list[str], annotate: bool):
    all_ms = unique_sorted(row.get("M") for row in rows)
    all_ks = unique_sorted(row.get("K") for row in rows)
    if len(all_ms) > 1 and len(all_ks) > 1:
        return plot_mn_heatmaps_by_k(rows, output_dir, formats, annotate)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.colors import ListedColormap
        from matplotlib.patches import Patch
    except Exception as exc:
        print(f"[warn] matplotlib is unavailable, skipping fixed-M N-K heatmaps: {exc}", flush=True)
        return

    grouped = defaultdict(list)
    for row in rows:
        if row.get("M") is None or row.get("N") is None or row.get("K") is None:
            continue
        grouped[(row.get("impl"), row.get("M"))].append(row)

    status_categories = STATUS_ORDER + ["unknown"]
    cmap = ListedColormap([STATUS_COLORS[key] for key in status_categories] + [STATUS_COLORS["missing"]])
    status_to_code = {status: idx for idx, status in enumerate(status_categories)}
    missing_code = len(status_categories)
    for (impl, m_value), subset in sorted(grouped.items(), key=lambda item: ((item[0][0] or ""), item[0][1])):
        Ns = unique_sorted(row.get("N") for row in subset)
        Ks = unique_sorted(row.get("K") for row in subset)
        if len(Ns) <= 1 or len(Ks) <= 1:
            continue

        fig, ax = plt.subplots(1, 1, figsize=(1.25 * max(len(Ns), 5), 1.0 * max(len(Ks), 4) + 1.8), constrained_layout=False)
        matrix = np.full((len(Ks), len(Ns)), missing_code, dtype=np.int32)
        labels = [["" for _ in Ns] for _ in Ks]

        cell_rows = defaultdict(list)
        for row in subset:
            cell_rows[(row["K"], row["N"])].append(row)

        for ki, k_value in enumerate(Ks):
            for ni, n_value in enumerate(Ns):
                row = pick_row_for_cell(cell_rows.get((k_value, n_value), []))
                if row is None:
                    continue
                status = row.get("status", "unknown")
                matrix[ki, ni] = status_to_code.get(status, status_to_code["unknown"])
                labels[ki][ni] = STATUS_LABELS.get(status, "UNK")

        ax.imshow(matrix, cmap=cmap, aspect="auto", origin="lower", vmin=0, vmax=missing_code)
        ax.set_title(f"{impl_label(impl)} Feasibility Heatmap (M={m_value})")
        ax.set_xlabel("N")
        ax.set_ylabel("K")
        ax.set_xticks(range(len(Ns)))
        ax.set_xticklabels([str(x) for x in Ns], rotation=30, ha="right")
        ax.set_yticks(range(len(Ks)))
        ax.set_yticklabels([str(x) for x in Ks])
        ax.set_xticks([x - 0.5 for x in range(1, len(Ns))], minor=True)
        ax.set_yticks([y - 0.5 for y in range(1, len(Ks))], minor=True)
        ax.grid(which="minor", color="#d9d9d9", linewidth=0.8)
        ax.tick_params(which="minor", bottom=False, left=False)

        if annotate and len(Ns) * len(Ks) <= 64:
            for ki, _ in enumerate(Ks):
                for ni, _ in enumerate(Ns):
                    code = matrix[ki, ni]
                    if code == missing_code:
                        continue
                    text_color = "white" if labels[ki][ni] in ("OOM", "CR", "ERR") else "black"
                    ax.text(ni, ki, labels[ki][ni], ha="center", va="center", fontsize=8.5, color=text_color)

        present_statuses = []
        for status in status_categories:
            code = status_to_code.get(status)
            if code is None:
                continue
            if (matrix == code).any():
                present_statuses.append(status)
        legend_handles = [Patch(facecolor=STATUS_COLORS[key], edgecolor="#444444", label=key) for key in present_statuses]
        if (matrix == missing_code).any():
            legend_handles.append(Patch(facecolor=STATUS_COLORS["missing"], edgecolor="#444444", label="missing"))

        fig.subplots_adjust(left=0.11, right=0.99, top=0.80, bottom=0.20)
        if legend_handles:
            fig.legend(handles=legend_handles,
                       loc="upper center",
                       bbox_to_anchor=(0.5, 0.94),
                       ncol=min(len(legend_handles), 4),
                       frameon=False,
                       fontsize=9.5)

        safe_impl = (impl or "unknown").replace(" ", "_")
        for fmt in formats:
            fig.savefig(output_dir / f"nvshmem_feasibility_heatmap_NK_{safe_impl}_M{m_value}.{fmt}",
                        dpi=220,
                        bbox_inches="tight")
        plt.close(fig)


def plot_mn_heatmaps_by_k(rows: list[dict], output_dir: Path, formats: list[str], annotate: bool):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.colors import ListedColormap
        from matplotlib.patches import Patch
    except Exception as exc:
        print(f"[warn] matplotlib is unavailable, skipping K-faceted M-N heatmaps: {exc}", flush=True)
        return

    grouped = defaultdict(list)
    for row in rows:
        if row.get("M") is None or row.get("N") is None or row.get("K") is None:
            continue
        grouped[row.get("impl")].append(row)

    status_categories = STATUS_ORDER + ["unknown"]
    cmap = ListedColormap([STATUS_COLORS[key] for key in status_categories] + [STATUS_COLORS["missing"]])
    status_to_code = {status: idx for idx, status in enumerate(status_categories)}
    missing_code = len(status_categories)

    for impl, subset in sorted(grouped.items(), key=lambda item: item[0] or ""):
        Ks = unique_sorted(row.get("K") for row in subset)
        Ms = unique_sorted(row.get("M") for row in subset)
        Ns = unique_sorted(row.get("N") for row in subset)
        if len(Ks) <= 1 or len(Ms) <= 1 or len(Ns) <= 1:
            continue

        cols = min(2, max(1, len(Ks)))
        rows_n = math.ceil(len(Ks) / cols)
        fig, axes = plt.subplots(rows_n, cols, figsize=(5.3 * cols, 4.2 * rows_n), constrained_layout=False)
        if not isinstance(axes, (list, tuple, np.ndarray)):
            axes = [axes]
        else:
            axes = list(np.ravel(axes))

        present_statuses = set()
        used_missing = False
        cell_rows = defaultdict(list)
        for row in subset:
            cell_rows[(row["K"], row["M"], row["N"])].append(row)

        for panel_idx, (ax, k_value) in enumerate(zip(axes, Ks)):
            matrix = np.full((len(Ms), len(Ns)), missing_code, dtype=np.int32)
            labels = [["" for _ in Ns] for _ in Ms]

            for mi, m_value in enumerate(Ms):
                for ni, n_value in enumerate(Ns):
                    row = pick_row_for_cell(cell_rows.get((k_value, m_value, n_value), []))
                    if row is None:
                        used_missing = True
                        continue
                    status = row.get("status", "unknown")
                    matrix[mi, ni] = status_to_code.get(status, status_to_code["unknown"])
                    labels[mi][ni] = STATUS_LABELS.get(status, "UNK")
                    present_statuses.add(status)

            ax.imshow(matrix, cmap=cmap, aspect="auto", origin="lower", vmin=0, vmax=missing_code)
            ax.set_title(f"K={k_value}", fontsize=11)
            ax.set_xlabel("N", labelpad=6)
            if panel_idx % cols == 0:
                ax.set_ylabel("M", labelpad=12)
            else:
                ax.set_ylabel("")
            ax.set_xticks(range(len(Ns)))
            ax.set_xticklabels([str(x) for x in Ns], rotation=30, ha="right")
            ax.set_yticks(range(len(Ms)))
            ax.set_yticklabels([str(x) for x in Ms])
            ax.set_xticks([x - 0.5 for x in range(1, len(Ns))], minor=True)
            ax.set_yticks([y - 0.5 for y in range(1, len(Ms))], minor=True)
            ax.grid(which="minor", color="#d9d9d9", linewidth=0.8)
            ax.tick_params(which="minor", bottom=False, left=False)

            if annotate and len(Ms) * len(Ns) <= 64:
                for mi, _ in enumerate(Ms):
                    for ni, _ in enumerate(Ns):
                        code = matrix[mi, ni]
                        if code == missing_code:
                            continue
                        text_color = "white" if labels[mi][ni] in ("OOM", "CR", "ERR") else "black"
                        ax.text(ni, mi, labels[mi][ni], ha="center", va="center", fontsize=8.5, color=text_color)

        for ax in axes[len(Ks):]:
            ax.axis("off")

        ordered_present_statuses = [status for status in status_categories if status in present_statuses]
        legend_handles = [Patch(facecolor=STATUS_COLORS[key], edgecolor="#444444", label=key) for key in ordered_present_statuses]
        if used_missing:
            legend_handles.append(Patch(facecolor=STATUS_COLORS["missing"], edgecolor="#444444", label="missing"))

        fig.suptitle(f"{impl_label(impl)} Feasibility Heatmaps", fontsize=13)
        fig.subplots_adjust(left=0.10, right=0.99, top=0.85, bottom=0.12, wspace=0.16, hspace=0.42)
        if legend_handles:
            fig.legend(handles=legend_handles,
                       loc="upper center",
                       bbox_to_anchor=(0.5, 0.94),
                       ncol=min(len(legend_handles), 4),
                       frameon=False,
                       fontsize=10)

        safe_impl = (impl or "unknown").replace(" ", "_")
        for fmt in formats:
            fig.savefig(output_dir / f"nvshmem_feasibility_heatmap_MN_by_K_{safe_impl}.{fmt}",
                        dpi=220,
                        bbox_inches="tight")
        plt.close(fig)


def plot_fixed_m_n_bytes(rows: list[dict], output_dir: Path, formats: list[str]):
    all_ms = unique_sorted(row.get("M") for row in rows)
    all_ks = unique_sorted(row.get("K") for row in rows)
    if len(all_ms) > 1 and len(all_ks) > 1:
        return plot_n_curves_by_m(rows, output_dir, formats)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except Exception as exc:
        print(f"[warn] matplotlib is unavailable, skipping fixed-M bytes-vs-N plots: {exc}", flush=True)
        return

    grouped = defaultdict(list)
    for row in rows:
        if row.get("M") is None or row.get("N") is None or row.get("K") is None:
            continue
        if row.get("estimated_symm_data_bytes") is None and row.get("actual_symm_data_bytes") is None:
            continue
        grouped[(row.get("impl"), row.get("M"))].append(row)

    for (impl, m_value), subset in sorted(grouped.items(), key=lambda item: ((item[0][0] or ""), item[0][1])):
        Ns = unique_sorted(row.get("N") for row in subset)
        Ks = unique_sorted(row.get("K") for row in subset)
        if len(Ns) <= 1:
            continue

        ordered_cases = sorted(
            [row for row in subset if row.get("N") is not None and row.get("K") is not None and row.get("estimated_symm_data_bytes") is not None],
            key=lambda item: (item["N"], item["K"]),
        )
        if len(ordered_cases) <= 1:
            continue

        fig_width = max(11.0, 0.58 * len(ordered_cases))
        fig, ax = plt.subplots(1, 1, figsize=(fig_width, 5.3), constrained_layout=False)

        x_positions = list(range(len(ordered_cases)))
        theory_gib = [row["estimated_symm_data_bytes"] / (1024**3) for row in ordered_cases]
        ax.plot(x_positions,
                theory_gib,
                color="#222222",
                linestyle="--",
                linewidth=2.0,
                label="Theory / estimated symm data bytes")

        for x_pos, row in zip(x_positions, ordered_cases):
            status = row.get("status", "unknown")
            if status == "success":
                y_value = row["actual_symm_data_bytes"] / (1024**3) if row.get("actual_symm_data_bytes") is not None else row[
                    "estimated_symm_data_bytes"] / (1024**3)
                marker = "o"
                facecolor = STATUS_COLORS["success"]
            elif status == "nvshmem_oom":
                y_value = row["estimated_symm_data_bytes"] / (1024**3)
                marker = "X"
                facecolor = STATUS_COLORS["nvshmem_oom"]
            else:
                y_value = row["estimated_symm_data_bytes"] / (1024**3)
                marker = "D"
                facecolor = STATUS_COLORS["timeout"]
            ax.scatter(x_pos,
                       y_value,
                       s=95,
                       marker=marker,
                       facecolors=facecolor,
                       edgecolors="#222222",
                       linewidths=0.9,
                       zorder=4)

        n_group_boundaries = []
        for idx in range(len(ordered_cases) - 1):
            if ordered_cases[idx]["N"] != ordered_cases[idx + 1]["N"]:
                n_group_boundaries.append(idx + 0.5)
        for boundary in n_group_boundaries:
            ax.axvline(boundary, color="#cfcfcf", linestyle=":", linewidth=1.0, zorder=1)

        status_handles = []
        status_style = {
            "success": ("o", STATUS_COLORS["success"], "success"),
            "nvshmem_oom": ("X", STATUS_COLORS["nvshmem_oom"], "nvshmem_oom"),
            "other": ("D", STATUS_COLORS["timeout"], "other status"),
        }
        present_statuses = []
        if any(row.get("status") == "success" for row in ordered_cases):
            present_statuses.append("success")
        if any(row.get("status") == "nvshmem_oom" for row in ordered_cases):
            present_statuses.append("nvshmem_oom")
        if any(row.get("status") not in ("success", "nvshmem_oom") for row in ordered_cases):
            present_statuses.append("other")
        for status_name in present_statuses:
            marker, color, label = status_style[status_name]
            status_handles.append(
                Line2D([0], [0], marker=marker, color="none", markerfacecolor=color, markeredgecolor="#222222",
                       markersize=8, label=label)
            )

        ax.set_title(f"{impl_label(impl)} Symmetric Data Capacity vs Ordered (N, K) Cases (M={m_value})")
        ax.set_xlabel("Ordered (N, K) cases")
        ax.set_ylabel("Symmetric data capacity (GiB)")
        ax.grid(alpha=0.28, linestyle="--", linewidth=0.8)
        ax.set_xticks(x_positions)
        ax.set_xticklabels([f"{row['N']}\n{row['K']}" for row in ordered_cases], rotation=0, ha="center", fontsize=8.5)
        ax.margins(x=0.015)

        line_handle = [Line2D([0], [0], color="#222222", linestyle="--", linewidth=2.0,
                              label="Theory / estimated symm data bytes")]
        ax.legend(handles=line_handle + status_handles, loc="upper left", frameon=False, fontsize=9.5)

        fig.subplots_adjust(left=0.10, right=0.99, top=0.88, bottom=0.23)
        safe_impl = (impl or "unknown").replace(" ", "_")
        for fmt in formats:
            fig.savefig(output_dir / f"nvshmem_feasibility_bytes_vs_N_{safe_impl}_M{m_value}.{fmt}",
                        dpi=220,
                        bbox_inches="tight")
        plt.close(fig)


def plot_n_curves_by_m(rows: list[dict], output_dir: Path, formats: list[str]):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except Exception as exc:
        print(f"[warn] matplotlib is unavailable, skipping N-curves-by-M plots: {exc}", flush=True)
        return

    grouped = defaultdict(list)
    for row in rows:
        if row.get("M") is None or row.get("N") is None or row.get("K") is None:
            continue
        if row.get("estimated_symm_data_bytes") is None and row.get("actual_symm_data_bytes") is None:
            continue
        grouped[row.get("impl")].append(row)

    for impl, subset in sorted(grouped.items(), key=lambda item: item[0] or ""):
        Ms = unique_sorted(row.get("M") for row in subset)
        Ns = unique_sorted(row.get("N") for row in subset)
        Ks = unique_sorted(row.get("K") for row in subset)
        if len(Ms) <= 1 or len(Ns) <= 1:
            continue

        fig, ax = plt.subplots(1, 1, figsize=(8.8, 5.3), constrained_layout=False)
        x_positions = list(range(len(Ns)))
        line_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#8c564b", "#17becf", "#d62728"]
        color_by_m = {m_value: line_colors[idx % len(line_colors)] for idx, m_value in enumerate(Ms)}

        present_statuses = set()
        plotted_any = False

        for m_value in Ms:
            m_rows = [row for row in subset if row.get("M") == m_value]
            y_values = []
            point_specs = []
            valid_line = True
            for x_pos, n_value in zip(x_positions, Ns):
                candidates = [row for row in m_rows if row.get("N") == n_value]
                if not candidates:
                    valid_line = False
                    break

                row_for_est = pick_row_for_cell([item for item in candidates if item.get("estimated_symm_data_bytes") is not None])
                if row_for_est is None:
                    valid_line = False
                    break
                y_est = row_for_est["estimated_symm_data_bytes"] / (1024**3)
                y_values.append(y_est)

                statuses = {item.get("status", "unknown") for item in candidates}
                if statuses == {"success"}:
                    point_status = "success"
                    success_actuals = [
                        item.get("actual_symm_data_bytes") / (1024**3)
                        for item in candidates
                        if item.get("actual_symm_data_bytes") is not None
                    ]
                    point_y = success_actuals[0] if success_actuals else y_est
                elif statuses == {"nvshmem_oom"}:
                    point_status = "nvshmem_oom"
                    point_y = y_est
                else:
                    point_status = "mixed"
                    success_actuals = [
                        item.get("actual_symm_data_bytes") / (1024**3)
                        for item in candidates
                        if item.get("status") == "success" and item.get("actual_symm_data_bytes") is not None
                    ]
                    point_y = success_actuals[0] if success_actuals else y_est
                point_specs.append((x_pos, point_y, point_status))
                present_statuses.add(point_status)

            if not valid_line:
                continue

            plotted_any = True
            color = color_by_m[m_value]
            ax.plot(x_positions,
                    y_values,
                    color=color,
                    linewidth=2.2,
                    marker=None,
                    label=f"M={m_value}")

            for x_pos, point_y, point_status in point_specs:
                if point_status == "success":
                    marker = "o"
                    facecolor = STATUS_COLORS["success"]
                elif point_status == "nvshmem_oom":
                    marker = "X"
                    facecolor = STATUS_COLORS["nvshmem_oom"]
                else:
                    marker = "D"
                    facecolor = STATUS_COLORS["timeout"]
                ax.scatter(x_pos,
                           point_y,
                           s=90,
                           marker=marker,
                           facecolors=facecolor,
                           edgecolors="#222222",
                           linewidths=0.8,
                           zorder=4)

        if not plotted_any:
            plt.close(fig)
            continue

        if len(Ks) == 1:
            title_suffix = f"K={Ks[0]}"
        else:
            title_suffix = f"aggregated over K ({', '.join(str(x) for x in Ks)})"

        status_handles = []
        status_style = {
            "success": ("o", STATUS_COLORS["success"], "all tested K: success"),
            "nvshmem_oom": ("X", STATUS_COLORS["nvshmem_oom"], "all tested K: OOM"),
            "mixed": ("D", STATUS_COLORS["timeout"], "mixed K outcomes"),
        }
        for status_name in ["success", "nvshmem_oom", "mixed"]:
            if status_name in present_statuses:
                marker, color, label = status_style[status_name]
                status_handles.append(
                    Line2D([0], [0], marker=marker, color="none", markerfacecolor=color, markeredgecolor="#222222",
                           markersize=8, label=label)
                )

        ax.set_title(f"{impl_label(impl)} Symmetric Data Capacity vs N ({title_suffix})")
        ax.set_xlabel("N")
        ax.set_ylabel("Symmetric data capacity (GiB)")
        ax.grid(alpha=0.28, linestyle="--", linewidth=0.8)
        ax.set_xticks(x_positions)
        ax.set_xticklabels([str(x) for x in Ns], rotation=30, ha="right")

        handles, labels = ax.get_legend_handles_labels()
        legend_lines = ax.legend(handles, labels, loc="upper left", frameon=False, fontsize=9.5, title="Line by M")
        ax.add_artist(legend_lines)
        if status_handles:
            ax.legend(handles=status_handles,
                      loc="upper left",
                      bbox_to_anchor=(0.31, 1.0),
                      frameon=False,
                      fontsize=9.2,
                      title="Point status")

        fig.subplots_adjust(left=0.11, right=0.98, top=0.88, bottom=0.18)
        safe_impl = (impl or "unknown").replace(" ", "_")
        for fmt in formats:
            fig.savefig(output_dir / f"nvshmem_feasibility_curves_N_by_M_{safe_impl}.{fmt}",
                        dpi=220,
                        bbox_inches="tight")
        plt.close(fig)


def plot_from_csv(input_csv: Path,
                  output_dir: Path,
                  formats: list[str],
                  *,
                  skip_heatmap: bool = False,
                  skip_frontier: bool = False,
                  skip_memory: bool = False,
                  heatmap_annotate: bool = True):
    rows = load_rows(input_csv)
    if not rows:
        print(f"[warn] no rows found in {input_csv}, skipping plots", flush=True)
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    if not skip_heatmap:
        plot_heatmaps(rows, output_dir, formats, annotate=heatmap_annotate)
    if not skip_frontier:
        plot_frontiers(rows, output_dir, formats)
    if not skip_memory:
        plot_memory_curves(rows, output_dir, formats)
    plot_fixed_m_nk_heatmaps(rows, output_dir, formats, annotate=heatmap_annotate)
    plot_fixed_m_n_bytes(rows, output_dir, formats)


def main():
    args = parse_args()
    input_csv = Path(args.input_csv)
    output_dir = Path(args.output_dir) if args.output_dir else input_csv.with_suffix("")
    formats = [item.strip() for item in args.formats.split(",") if item.strip()]
    plot_from_csv(
        input_csv,
        output_dir,
        formats,
        skip_heatmap=args.skip_heatmap,
        skip_frontier=args.skip_frontier,
        skip_memory=args.skip_memory,
        heatmap_annotate=not args.heatmap_no_annotate,
    )
    print(f"[plot] plots written to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
