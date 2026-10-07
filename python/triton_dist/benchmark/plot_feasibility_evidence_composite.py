#!/usr/bin/env python3
"""Plot RS/AR speedup together with symmetric-heap use.

Speedup ranges are parsed directly from the author's ``rs-gemm.txt`` and
``ar-gemm.txt`` records. Symmetric-memory values come from the structured
CSV derived from the large-heap measurement note. Bars use the historical
Figure-9 palette; line markers are drawn above the bars on a secondary axis.
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
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


METHODS = ("Triton-Dist", "Ours")
COLORS = {"Triton-Dist": "#4E79A7", "Ours": "#F28E2B"}
HEADER_RE = re.compile(r"^#\s*(\d+)\s+(\d+)\s+(\d+)\s*$")
RANGE_RE = re.compile(r"最低\s*([0-9.]+)\s*最高\s*([0-9.]+)")


@dataclass(frozen=True)
class SpeedupRange:
    chain: str
    shape: tuple[int, int, int]
    method: str
    low: float
    high: float


@dataclass(frozen=True)
class MemoryRecord:
    chain: str
    shape: tuple[int, int, int]
    triton_dist_gib_per_pe: float
    ours_reserved_gib_per_pe: float
    heap_gib_per_pe: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rs-log", type=Path, default=Path("/root/做的实验/rs-gemm.txt"))
    parser.add_argument("--ar-log", type=Path, default=Path("/root/做的实验/ar-gemm.txt"))
    parser.add_argument(
        "--memory-csv",
        type=Path,
        default=Path("benchmark/figure9_symmetric_memory_20260917.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark/evaluation_figures/feasibility_evidence_upper"),
    )
    return parser.parse_args()


def parse_range(line: str) -> tuple[float, float] | None:
    if "failed" in line.casefold() or "无法处理" in line:
        return None
    match = RANGE_RE.search(line)
    if match is None:
        raise ValueError(f"Cannot parse speedup range: {line!r}")
    return float(match.group(1)), float(match.group(2))


def parse_speedup_file(path: Path, chain: str) -> list[SpeedupRange]:
    records: list[SpeedupRange] = []
    current_shape: tuple[int, int, int] | None = None
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        header = HEADER_RE.match(line)
        if header:
            current_shape = tuple(int(header.group(index)) for index in range(1, 4))
            continue
        if current_shape is None:
            continue
        normalized = line.casefold()
        if normalized.startswith("triton-dist") or normalized.startswith("tri-dist"):
            method = "Triton-Dist"
        elif line.startswith("我的创新"):
            method = "Ours"
        else:
            continue
        value_range = parse_range(line)
        if value_range is None:
            raise ValueError(
                f"Figure 9 requires completed {method} measurements; found failure at {path}:{line_number}"
            )
        records.append(SpeedupRange(chain, current_shape, method, *value_range))
    return records


def parse_memory_csv(path: Path) -> list[MemoryRecord]:
    records: list[MemoryRecord] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            records.append(
                MemoryRecord(
                    chain=row["chain"],
                    shape=(int(row["M"]), int(row["N"]), int(row["K"])),
                    triton_dist_gib_per_pe=float(row["triton_dist_gib_per_pe"]),
                    ours_reserved_gib_per_pe=float(row["ours_reserved_gib_per_pe"]),
                    heap_gib_per_pe=float(row["heap_gib_per_pe"]),
                )
            )
    return records


def shape_label(shape: tuple[int, int, int]) -> str:
    m, n, k = shape
    m_label = f"{m // 1024}K" if m % 1024 == 0 else str(m)
    return f"{m_label}×{n}\n{k}"


def validate(
    speedups: list[SpeedupRange], memories: list[MemoryRecord]
) -> tuple[
    dict[tuple[str, tuple[int, int, int], str], SpeedupRange],
    dict[tuple[str, tuple[int, int, int]], MemoryRecord],
]:
    speedup_lookup = {(item.chain, item.shape, item.method): item for item in speedups}
    memory_lookup = {(item.chain, item.shape): item for item in memories}
    for chain in ("RS", "AR"):
        shapes = [item.shape for item in memories if item.chain == chain]
        if len(shapes) != 4 or len(set(shapes)) != 4:
            raise ValueError(f"Expected four unique {chain} memory rows, found {shapes}")
        for shape in shapes:
            for method in METHODS:
                if (chain, shape, method) not in speedup_lookup:
                    raise ValueError(f"Missing speedup for {(chain, shape, method)}")
            memory = memory_lookup[(chain, shape)]
            if memory.heap_gib_per_pe <= 0:
                raise ValueError(f"Invalid heap size for {(chain, shape)}")
    return speedup_lookup, memory_lookup


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "STIXGeneral"],
            "mathtext.fontset": "stix",
            "font.size": 5.8,
            "axes.linewidth": 0.55,
            "axes.unicode_minus": False,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "savefig.dpi": 600,
            "savefig.bbox": None,
        }
    )


def draw_panel(
    axis: plt.Axes,
    chain: str,
    shapes: list[tuple[int, int, int]],
    speedup_lookup: dict[tuple[str, tuple[int, int, int], str], SpeedupRange],
    memory_lookup: dict[tuple[str, tuple[int, int, int]], MemoryRecord],
) -> None:
    x = np.arange(len(shapes), dtype=float)
    width = 0.29
    offsets = {"Triton-Dist": -0.17, "Ours": 0.17}

    for method in METHODS:
        lows = np.array([speedup_lookup[(chain, shape, method)].low for shape in shapes])
        highs = np.array([speedup_lookup[(chain, shape, method)].high for shape in shapes])
        midpoints = (lows + highs) / 2.0
        positions = x + offsets[method]
        axis.bar(
            positions,
            midpoints,
            width=width,
            color=COLORS[method],
            edgecolor="black",
            linewidth=0.5,
            zorder=3,
        )
        axis.errorbar(
            positions,
            midpoints,
            yerr=np.vstack((midpoints - lows, highs - midpoints)),
            fmt="none",
            ecolor="black",
            elinewidth=0.6,
            capsize=1.7,
            capthick=0.6,
            zorder=4,
        )

    axis.axhline(1.0, color="#666666", linestyle="--", linewidth=0.55, zorder=2)
    axis.set_ylim(0.88, 1.31)
    axis.set_yticks([0.9, 1.0, 1.1, 1.2, 1.3])
    axis.set_xlim(-0.55, len(shapes) - 0.45)
    axis.set_xticks(x)
    axis.set_xticklabels([shape_label(shape) for shape in shapes], fontsize=5.2, linespacing=0.92)
    axis.tick_params(axis="x", length=2.0, width=0.55, pad=2.0)
    axis.tick_params(axis="y", labelsize=5.4, length=2.0, width=0.55, pad=1.5)
    axis.grid(axis="y", color="#D8D8D8", linewidth=0.4, alpha=0.85, zorder=0)
    axis.set_ylabel("Speedup (×)", fontsize=6.1, labelpad=2.2)
    axis.set_title(f"({'a' if chain == 'RS' else 'b'}) GEMM--{chain}", fontsize=6.6, fontweight="bold", pad=2.4)

    memory_axis = axis.twinx()
    triton_pct = [
        100.0
        * memory_lookup[(chain, shape)].triton_dist_gib_per_pe
        / memory_lookup[(chain, shape)].heap_gib_per_pe
        for shape in shapes
    ]
    ours_pct = [
        100.0
        * memory_lookup[(chain, shape)].ours_reserved_gib_per_pe
        / memory_lookup[(chain, shape)].heap_gib_per_pe
        for shape in shapes
    ]
    memory_axis.plot(
        x,
        triton_pct,
        color=COLORS["Triton-Dist"],
        marker="o",
        markersize=3.2,
        markerfacecolor="white",
        markeredgewidth=0.75,
        linewidth=1.0,
        zorder=7,
    )
    memory_axis.plot(
        x,
        ours_pct,
        color=COLORS["Ours"],
        marker="s",
        markersize=3.0,
        markerfacecolor="white",
        markeredgewidth=0.75,
        linewidth=1.0,
        zorder=7,
    )
    memory_axis.set_ylim(0.0, 65.0)
    memory_axis.set_yticks([0, 20, 40, 60])
    memory_axis.tick_params(axis="y", labelsize=5.2, length=2.0, width=0.55, pad=1.5)
    memory_axis.set_ylabel("Allocation / heap (%)", fontsize=5.9, labelpad=2.2)
    memory_axis.spines["top"].set_visible(False)
    axis.spines["top"].set_visible(False)


def write_derived_csv(
    output_base: Path,
    speedup_lookup: dict[tuple[str, tuple[int, int, int], str], SpeedupRange],
    memory_lookup: dict[tuple[str, tuple[int, int, int]], MemoryRecord],
) -> Path:
    output_path = output_base.with_name(output_base.name + "_data").with_suffix(".csv")
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "chain",
                "M",
                "N",
                "K",
                "method",
                "speedup_low",
                "speedup_high",
                "memory_gib_per_pe",
                "heap_percent",
            ]
        )
        for chain in ("RS", "AR"):
            for shape in [item.shape for item in memory_lookup.values() if item.chain == chain]:
                memory = memory_lookup[(chain, shape)]
                for method in METHODS:
                    speedup = speedup_lookup[(chain, shape, method)]
                    memory_gib = (
                        memory.triton_dist_gib_per_pe
                        if method == "Triton-Dist"
                        else memory.ours_reserved_gib_per_pe
                    )
                    writer.writerow(
                        [
                            chain,
                            *shape,
                            method,
                            f"{speedup.low:.6f}",
                            f"{speedup.high:.6f}",
                            f"{memory_gib:.6f}",
                            f"{100.0 * memory_gib / memory.heap_gib_per_pe:.6f}",
                        ]
                    )
    return output_path


def main() -> None:
    args = parse_args()
    for path in (args.rs_log, args.ar_log, args.memory_csv):
        if not path.is_file():
            raise SystemExit(f"Missing input: {path}")

    speedups = parse_speedup_file(args.rs_log, "RS") + parse_speedup_file(args.ar_log, "AR")
    memories = parse_memory_csv(args.memory_csv)
    speedup_lookup, memory_lookup = validate(speedups, memories)

    configure_style()
    fig, axes = plt.subplots(2, 1, figsize=(3.26, 2.91), sharex=False, constrained_layout=False)
    for axis, chain in zip(axes, ("RS", "AR")):
        shapes = [item.shape for item in memories if item.chain == chain]
        draw_panel(axis, chain, shapes, speedup_lookup, memory_lookup)

    legend_handles = [
        Patch(
            facecolor=COLORS["Triton-Dist"],
            edgecolor="black",
            linewidth=0.5,
            label="Triton-Dist speedup",
        ),
        Patch(facecolor=COLORS["Ours"], edgecolor="black", linewidth=0.5, label="Ours speedup"),
        Line2D(
            [0],
            [0],
            color=COLORS["Triton-Dist"],
            marker="o",
            markerfacecolor="white",
            linewidth=1.0,
            label="Triton-Dist allocation",
        ),
        Line2D(
            [0],
            [0],
            color=COLORS["Ours"],
            marker="s",
            markerfacecolor="white",
            linewidth=1.0,
            label="Ours reserved envelope",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=2,
        fontsize=5.2,
        frameon=False,
        columnspacing=0.75,
        handlelength=1.35,
        handletextpad=0.35,
    )
    fig.subplots_adjust(left=0.13, right=0.875, top=0.865, bottom=0.10, hspace=0.55)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    derived_csv = write_derived_csv(args.output, speedup_lookup, memory_lookup)
    fig.savefig(args.output.with_suffix(".png"), dpi=600, bbox_inches=None, pad_inches=0)
    fig.savefig(args.output.with_suffix(".pdf"), bbox_inches=None, pad_inches=0)
    fig.savefig(args.output.with_suffix(".svg"), bbox_inches=None, pad_inches=0)
    plt.close(fig)

    print(f"Wrote {derived_csv}")
    for suffix in (".png", ".pdf", ".svg"):
        print(f"Wrote {args.output.with_suffix(suffix)}")


if __name__ == "__main__":
    main()
