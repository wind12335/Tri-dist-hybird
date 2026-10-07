#!/usr/bin/env python3
"""Plot recorded TP-prefill speedups for Attention, MLP, and E2E prefill.

Each source text file contains one historical speedup record per model,
platform, and execution variant. The A100 record set contains the Torch
no-overlap reference, Triton-distributed, two one-sided AG/RS substitutions,
the combined AG/RS path, and the AR path. K100_AI records the Torch reference,
the combined AG/RS path, and the AR path. The script keeps the platform panels separate so
their platform-local normalizations are never read as absolute comparisons.

Example:
    python benchmark/plot_e2e_prefill_speedup.py
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


DEFAULT_INPUT_DIR = Path("/root/做的实验")
DEFAULT_OUTPUT = Path("benchmark/evaluation_figures/e2e_prefill_speedup")

WORKLOAD_ORDER = ("Attention", "MLP", "E2E")
MODEL_ORDER = ("Llama3-70B", "Qwen2-72B")
PLATFORM_ORDER = ("A100", "K100_AI")
PLATFORM_LABELS = {
    "A100": "A100 (4 GPUs)",
    "K100_AI": "K100_AI (4 GPUs)",
}
VARIANT_ORDER = (
    "non_overlap",
    "triton_dist",
    "ag_old_rs_new",
    "ag_new_rs_old",
    "ag_rs_new",
    "gemm_ar_v23",
)
VARIANT_LABELS = {
    "non_overlap": "Torch no-overlap",
    "triton_dist": "Triton-Dist",
    "ag_old_rs_new": "Baseline AG / Proposed RS",
    "ag_new_rs_old": "Proposed AG / Baseline RS",
    "ag_rs_new": "Proposed AG / Proposed RS",
    "gemm_ar_v23": "Ours AR",
}
VARIANT_COLORS = {
    "non_overlap": "#7A7A7A",
    "triton_dist": "#4E79A7",
    "ag_old_rs_new": "#F18F01",
    "ag_new_rs_old": "#76B7B2",
    "ag_rs_new": "#C73E1D",
    "gemm_ar_v23": "#9467BD",
}
EXPECTED_VARIANTS = {
    "A100": VARIANT_ORDER,
    "K100_AI": ("non_overlap", "ag_rs_new", "gemm_ar_v23"),
}


@dataclass(frozen=True)
class SpeedupRecord:
    workload: str
    platform: str
    model: str
    variant: str
    speedup: float
    source: str


def normalize_model(header: str) -> str:
    lower = header.casefold()
    if "llama3-70b" in lower or "llama3" in lower:
        return "Llama3-70B"
    if "qwen2-72b" in lower or "qweb2-72b" in lower or "qwen2" in lower:
        return "Qwen2-72B"
    raise ValueError(f"Cannot identify model from header: {header!r}")


def normalize_platform(header: str) -> str:
    lower = header.casefold()
    if "dcu" in lower or "k_100ai" in lower or "k100_ai" in lower:
        return "K100_AI"
    # Historical A100 records omit the platform prefix; DCU records carry it.
    return "A100"


def normalize_variant(token: str) -> str:
    normalized = token.casefold().replace("-", "_")
    aliases = {
        "non_overlap": "non_overlap",
        "triton_dist": "triton_dist",
        "ag_baseline_rs_new": "ag_old_rs_new",
        "ag_basline_rs_new": "ag_old_rs_new",
        "ag_old_rs_new": "ag_old_rs_new",
        "ag_new_rs_baseline": "ag_new_rs_old",
        "ag_new_rs_old": "ag_new_rs_old",
        "ag_rs_new": "ag_rs_new",
        "gemm_ar_v23": "gemm_ar_v23",
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        raise ValueError(f"Unsupported E2E variant label: {token!r}") from exc


def parse_speedup_file(path: Path, workload: str) -> list[SpeedupRecord]:
    records: list[SpeedupRecord] = []
    current_header: str | None = None
    speedup_pattern = re.compile(
        r"^\s*([A-Za-z][A-Za-z0-9_-]*)\s+([0-9]+(?:\.[0-9]+)?)\s*[xX]\s*$"
    )

    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#"):
            current_header = line[1:].strip()
            continue

        match = speedup_pattern.match(line)
        if match is None:
            raise ValueError(f"Cannot parse E2E record at {path}:{line_number}: {line!r}")
        if current_header is None:
            raise ValueError(f"Speedup appears before a model header at {path}:{line_number}")

        records.append(
            SpeedupRecord(
                workload=workload,
                platform=normalize_platform(current_header),
                model=normalize_model(current_header),
                variant=normalize_variant(match.group(1)),
                speedup=float(match.group(2)),
                source=str(path),
            )
        )

    if not records:
        raise ValueError(f"No E2E speedup records found in {path}")
    return records


def validate_records(records: list[SpeedupRecord]) -> None:
    expected = {
        (workload, platform, model, variant)
        for workload in WORKLOAD_ORDER
        for platform in PLATFORM_ORDER
        for model in MODEL_ORDER
        for variant in EXPECTED_VARIANTS[platform]
    }
    actual = {(record.workload, record.platform, record.model, record.variant) for record in records}
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    duplicate_keys = [
        key
        for key in sorted(actual)
        if sum(
            record.workload == key[0]
            and record.platform == key[1]
            and record.model == key[2]
            and record.variant == key[3]
            for record in records
        )
        > 1
    ]
    if missing:
        raise ValueError(f"Missing E2E workload/platform/model/variant records: {missing}")
    if unexpected:
        raise ValueError(f"Unexpected E2E workload/platform/model/variant records: {unexpected}")
    if duplicate_keys:
        raise ValueError(f"Duplicate E2E records found: {duplicate_keys}")
    if len(records) != len(expected):
        raise ValueError(f"Expected {len(expected)} E2E records, found {len(records)}")


def write_csv(records: list[SpeedupRecord], output_base: Path) -> Path:
    csv_path = output_base.with_suffix(".csv")
    ordering = {
        (workload, platform, model, variant): (
            WORKLOAD_ORDER.index(workload),
            PLATFORM_ORDER.index(platform),
            MODEL_ORDER.index(model),
            VARIANT_ORDER.index(variant),
        )
        for workload in WORKLOAD_ORDER
        for platform in PLATFORM_ORDER
        for model in MODEL_ORDER
        for variant in EXPECTED_VARIANTS[platform]
    }
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["workload", "platform", "model", "variant", "variant_label", "speedup", "source"])
        for record in sorted(records, key=lambda item: ordering[(item.workload, item.platform, item.model, item.variant)]):
            writer.writerow(
                [
                    record.workload,
                    record.platform,
                    record.model,
                    record.variant,
                    VARIANT_LABELS[record.variant],
                    f"{record.speedup:.6f}",
                    record.source,
                ]
            )
    return csv_path


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "STIXGeneral"],
            "mathtext.fontset": "stix",
            "axes.linewidth": 0.85,
            "axes.unicode_minus": False,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "savefig.dpi": 450,
            "savefig.bbox": "tight",
        }
    )


def draw_platform_panel(
    axis: plt.Axes,
    lookup: dict[tuple[str, str, str, str], float],
    workload: str,
    platform: str,
    y_min: float,
    y_max: float,
) -> None:
    variants = EXPECTED_VARIANTS[platform]
    x = np.arange(len(MODEL_ORDER), dtype=float)
    width = 0.14 if platform == "A100" else 0.20
    offsets = (np.arange(len(variants), dtype=float) - (len(variants) - 1) / 2.0) * width
    label_entries: dict[int, list[tuple[plt.Rectangle, float, int]]] = {
        model_index: [] for model_index in range(len(MODEL_ORDER))
    }

    for variant_index, (offset, variant) in enumerate(zip(offsets, variants)):
        values = [lookup[(workload, platform, model, variant)] for model in MODEL_ORDER]
        bars = axis.bar(
            x + offset,
            values,
            width=width * 0.92,
            color=VARIANT_COLORS[variant],
            edgecolor="white",
            linewidth=0.65,
            zorder=3,
        )
        for model_index, (bar, value) in enumerate(zip(bars, values)):
            label_entries[model_index].append((bar, value, variant_index))

    for entries in label_entries.values():
        placed_y: list[float] = []
        for bar, value, variant_index in sorted(entries, key=lambda item: item[1]):
            label_y = value + 0.010
            if placed_y:
                label_y = max(label_y, placed_y[-1] + 0.045)
            placed_y.append(label_y)
            label_x = bar.get_x() + bar.get_width() / 2.0
            if platform == "A100":
                label_x += (variant_index - (len(variants) - 1) / 2.0) * 0.012
            axis.text(
                label_x,
                label_y,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=6.6 if platform == "A100" else 7.2,
                color="#334155",
            )

    axis.axhline(1.0, color="#475569", linestyle="--", linewidth=0.8, zorder=2)
    axis.set_title(f"{workload}: {PLATFORM_LABELS[platform]}", fontsize=9.4, pad=6)
    axis.set_xticks(x)
    axis.set_xticklabels(MODEL_ORDER, fontsize=7.6)
    axis.set_ylim(y_min, y_max)
    axis.set_yticks(np.arange(1.0, y_max + 0.001, 0.1))
    axis.grid(axis="y", color="#D7DDE5", linewidth=0.65, alpha=0.9, zorder=0)
    axis.tick_params(axis="y", labelsize=7.5, width=0.75, length=3)
    axis.tick_params(axis="x", width=0.75, length=3, pad=3)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def draw_plot(records: list[SpeedupRecord], output_base: Path) -> None:
    configure_style()
    lookup = {
        (record.workload, record.platform, record.model, record.variant): record.speedup
        for record in records
    }
    all_values = [record.speedup for record in records]
    y_min = min(0.95, min(all_values) - 0.025)
    y_max = max(1.35, max(all_values) + 0.06)

    fig, axes = plt.subplots(
        len(WORKLOAD_ORDER),
        len(PLATFORM_ORDER),
        figsize=(8.25, 7.0),
        sharey=True,
        gridspec_kw={"wspace": 0.20, "hspace": 0.50},
    )
    for row_index, workload in enumerate(WORKLOAD_ORDER):
        for column_index, platform in enumerate(PLATFORM_ORDER):
            axis = axes[row_index, column_index]
            draw_platform_panel(axis, lookup, workload, platform, y_min, y_max)
            if column_index == 0:
                axis.set_ylabel("Speedup vs. Torch baseline (x)", fontsize=8.4)
            else:
                axis.tick_params(axis="y", labelleft=False)

    legend_handles = [
        Patch(facecolor=VARIANT_COLORS[variant], edgecolor="white", label=VARIANT_LABELS[variant])
        for variant in VARIANT_ORDER
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.005),
        ncol=len(legend_handles),
        fontsize=7.8,
        frameon=False,
        columnspacing=0.95,
        handlelength=1.3,
    )
    fig.text(
        0.5,
        0.010,
        "Each platform is normalized to its own Torch no-overlap record; no cross-platform absolute comparison is implied.",
        ha="center",
        fontsize=7.6,
        color="#475569",
    )
    fig.subplots_adjust(left=0.095, right=0.99, top=0.93, bottom=0.075)

    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".png"), dpi=450)
    fig.savefig(output_base.with_suffix(".pdf"))
    fig.savefig(output_base.with_suffix(".svg"))
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_files = [
        ("Attention", args.input_dir / "e2e-attn.txt"),
        ("MLP", args.input_dir / "e2e-mlp.txt"),
        ("E2E", args.input_dir / "e2e.txt"),
    ]

    records: list[SpeedupRecord] = []
    for workload, path in input_files:
        if not path.exists():
            raise FileNotFoundError(path)
        records.extend(parse_speedup_file(path, workload))

    validate_records(records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    csv_path = write_csv(records, args.output)
    draw_plot(records, args.output)
    print(f"Wrote {csv_path}")
    for suffix in (".png", ".pdf", ".svg"):
        print(f"Wrote {args.output.with_suffix(suffix)}")


if __name__ == "__main__":
    main()
