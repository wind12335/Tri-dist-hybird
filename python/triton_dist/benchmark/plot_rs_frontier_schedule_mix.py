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

"""
Standalone RS schedule-analysis plotter.

This script does not modify or depend on the RS benchmark CSV format. Instead,
it provides a standalone auxiliary figure for the paper's
"wake-up work vs sustain-flow work" argument.

Two analysis modes are supported:

- panel_policy:
    A schedule-level proxy that models how producer micro-work is distributed
    across panel contributions. This is the recommended mode for the paper's
    auxiliary subplot because it directly exposes how uniform spreading touches
    more non-wake-up work before the first consumer activation.

- kernel_tile:
    A closer emulation of the existing low-level tile order in the current
    implementation. This is useful for sanity checks, but the separation can be
    weak for some parameter settings and is not the recommended main paper plot.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


WAKEUP_COLOR = "#2E86AB"
NON_WAKEUP_COLOR = "#F18F01"
ANNOTATION_COLOR = "#4E5B6A"


@dataclass(frozen=True)
class PanelId:
    destination_rank: int
    chunk_id: int
    band_id: int

    def to_label(self) -> str:
        return f"d{self.destination_rank}.c{self.chunk_id}.b{self.band_id}"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Analyze RS producer scheduling without modifying the benchmark. "
            "The script compares a uniform/average producer policy with a "
            "frontier-first policy and plots the pre-start panel mix."
        )
    )
    parser.add_argument("--world_size", type=int, default=4)
    parser.add_argument("--M", type=int, default=32768)
    parser.add_argument("--N", type=int, default=28672)
    parser.add_argument("--chunk_rows", type=int, default=1024)
    parser.add_argument("--n_bands", type=int, default=2)
    parser.add_argument("--frontier_chunks", type=int, default=1)
    parser.add_argument("--block_size_m", type=int, default=128)
    parser.add_argument("--block_size_n", type=int, default=256)
    parser.add_argument("--group_size_m", type=int, default=4)
    parser.add_argument(
        "--analysis_mode",
        type=str,
        default="panel_policy",
        choices=["panel_policy", "kernel_tile"],
        help=(
            "panel_policy: schedule-level proxy intended for the wake-up vs sustain-flow auxiliary plot; "
            "kernel_tile: closer emulation of the existing low-level tile order."
        ),
    )
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--formats", type=str, default="png,svg")
    parser.add_argument("--dpi", type=int, default=450)
    parser.add_argument("--title", type=str, default="RS pre-start panel mix")
    parser.add_argument(
        "--subtitle",
        type=str,
        default="Producer-touched panel contributions before first consumer start (schedule-level proxy)",
    )
    return parser.parse_args()


def cdiv(x: int, y: int) -> int:
    return (x + y - 1) // y


def swizzle_2d(tile_id: int, num_pid_m: int, num_pid_n: int, group_size_m: int) -> tuple[int, int]:
    num_pid_in_group = group_size_m * num_pid_n
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * group_size_m
    group_size_m_eff = min(num_pid_m - first_pid_m, group_size_m)
    pid_m = first_pid_m + (tile_id % group_size_m_eff)
    pid_n = (tile_id % num_pid_in_group) // group_size_m_eff
    return pid_m, pid_n


def band_info(pid_n: int, N: int, n_bands: int, block_size_n: int) -> tuple[int | None, int | None]:
    max_band_cols = cdiv(N, n_bands)
    num_pid_n_per_band = cdiv(max_band_cols, block_size_n)
    band_id = pid_n // num_pid_n_per_band
    if band_id >= n_bands:
        return None, None
    local_pid_n = pid_n - band_id * num_pid_n_per_band
    band_col_start = band_id * max_band_cols
    band_cols = max(0, min(max_band_cols, N - band_col_start))
    num_pid_n_in_band = cdiv(band_cols, block_size_n)
    if local_pid_n >= num_pid_n_in_band:
        return None, None
    return band_id, num_pid_n_in_band


def panel_required_tiles(
    world_size: int,
    M: int,
    N: int,
    chunk_rows: int,
    n_bands: int,
    block_size_m: int,
    block_size_n: int,
) -> dict[PanelId, int]:
    if M % world_size != 0:
        raise ValueError(f"M must be divisible by world_size for RS analysis, got M={M}, world_size={world_size}")

    m_per_rank = M // world_size
    num_chunks = cdiv(m_per_rank, chunk_rows)
    max_band_cols = cdiv(N, n_bands)

    required: dict[PanelId, int] = {}
    for dst in range(world_size):
        segment_row_start = dst * m_per_rank
        segment_row_end = segment_row_start + m_per_rank
        for chunk_id in range(num_chunks):
            row_start = segment_row_start + chunk_id * chunk_rows
            row_end = min(row_start + chunk_rows, segment_row_end)
            if row_start >= row_end:
                continue
            tile_m_first = row_start // block_size_m
            tile_m_last = (row_end - 1) // block_size_m
            num_tile_m = tile_m_last - tile_m_first + 1
            for band_id in range(n_bands):
                col_start = band_id * max_band_cols
                col_end = min(col_start + max_band_cols, N)
                if col_start >= col_end:
                    continue
                num_pid_n_in_band = cdiv(col_end - col_start, block_size_n)
                required[PanelId(dst, chunk_id, band_id)] = num_tile_m * num_pid_n_in_band
    return required


def panels_touched_by_tile(
    global_pid_m: int,
    pid_n: int,
    world_size: int,
    M: int,
    N: int,
    chunk_rows: int,
    n_bands: int,
    block_size_m: int,
    block_size_n: int,
) -> list[PanelId]:
    if M % world_size != 0:
        raise ValueError(f"M must be divisible by world_size for RS analysis, got M={M}, world_size={world_size}")

    band_id, _ = band_info(pid_n, N, n_bands, block_size_n)
    if band_id is None:
        return []

    m_per_rank = M // world_size
    tile_row_start = global_pid_m * block_size_m
    tile_row_end = min((global_pid_m + 1) * block_size_m, M) - 1
    segment_start = tile_row_start // m_per_rank
    segment_end = tile_row_end // m_per_rank

    panels: list[PanelId] = []
    for segment in range(segment_start, min(segment_end, world_size - 1) + 1):
        seg_global_row_start = segment * m_per_rank
        seg_global_row_end = seg_global_row_start + m_per_rank - 1
        seg_local_start = max(tile_row_start, seg_global_row_start) - seg_global_row_start
        seg_local_end = min(tile_row_end, seg_global_row_end) - seg_global_row_start
        chunk_start = seg_local_start // chunk_rows
        chunk_end = seg_local_end // chunk_rows
        for chunk_id in range(chunk_start, chunk_end + 1):
            panels.append(PanelId(segment, chunk_id, band_id))
    return panels


def generate_uniform_order(
    M: int,
    N: int,
    block_size_m: int,
    block_size_n: int,
    group_size_m: int,
) -> list[tuple[int, int]]:
    num_pid_m = cdiv(M, block_size_m)
    num_pid_n = cdiv(N, block_size_n)
    order: list[tuple[int, int]] = []
    for tile_id in range(num_pid_m * num_pid_n):
        pid_m, pid_n = swizzle_2d(tile_id, num_pid_m, num_pid_n, group_size_m)
        order.append((pid_m, pid_n))
    return order


def generate_frontier_order(
    rank: int,
    world_size: int,
    M: int,
    N: int,
    chunk_rows: int,
    frontier_chunks: int,
    block_size_m: int,
    block_size_n: int,
    group_size_m: int,
) -> list[tuple[int, int]]:
    if M % world_size != 0:
        raise ValueError(f"M must be divisible by world_size for RS analysis, got M={M}, world_size={world_size}")

    m_per_rank = M // world_size
    tiles_per_segment = cdiv(m_per_rank, block_size_m)
    frontier_chunks = max(0, min(frontier_chunks, cdiv(m_per_rank, chunk_rows)))
    frontier_rows = min(chunk_rows * frontier_chunks, m_per_rank)
    frontier_tiles_per_segment = min(cdiv(frontier_rows, block_size_m), tiles_per_segment)
    tail_tiles_per_segment = tiles_per_segment - frontier_tiles_per_segment
    num_pid_n = cdiv(N, block_size_n)

    order: list[tuple[int, int]] = []
    if frontier_tiles_per_segment > 0:
        phase_total_pid_m = frontier_tiles_per_segment * world_size
        for tile_id in range(phase_total_pid_m * num_pid_n):
            pid_m_phase, pid_n = swizzle_2d(tile_id, phase_total_pid_m, num_pid_n, group_size_m)
            segment_id = pid_m_phase // frontier_tiles_per_segment
            local_tile_in_segment = pid_m_phase % frontier_tiles_per_segment
            global_pid_m = segment_id * tiles_per_segment + local_tile_in_segment
            order.append((global_pid_m, pid_n))

    if tail_tiles_per_segment > 0:
        phase_total_pid_m = tail_tiles_per_segment * world_size
        tail_phase_tile_offset = ((rank + 1) % world_size) * tail_tiles_per_segment
        for tile_id in range(phase_total_pid_m * num_pid_n):
            pid_m_phase, pid_n = swizzle_2d(tile_id, phase_total_pid_m, num_pid_n, group_size_m)
            tail_space_pid_m = (pid_m_phase + tail_phase_tile_offset) % phase_total_pid_m
            segment_id = tail_space_pid_m // tail_tiles_per_segment
            local_tile_in_segment = tail_space_pid_m % tail_tiles_per_segment
            global_pid_m = segment_id * tiles_per_segment + frontier_tiles_per_segment + local_tile_in_segment
            order.append((global_pid_m, pid_n))
    return order


def classify_policy_kernel_tile(
    policy: str,
    world_size: int,
    M: int,
    N: int,
    chunk_rows: int,
    n_bands: int,
    frontier_chunks: int,
    block_size_m: int,
    block_size_n: int,
    group_size_m: int,
) -> dict[str, object]:
    if policy not in {"uniform", "frontier_first"}:
        raise ValueError(f"unknown policy: {policy}")

    panel_required = panel_required_tiles(
        world_size=world_size,
        M=M,
        N=N,
        chunk_rows=chunk_rows,
        n_bands=n_bands,
        block_size_m=block_size_m,
        block_size_n=block_size_n,
    )
    panels = sorted(panel_required.keys(), key=lambda p: (p.destination_rank, p.chunk_id, p.band_id))

    per_rank_orders: list[list[tuple[int, int]]] = []
    for rank in range(world_size):
        if policy == "uniform":
            per_rank_orders.append(generate_uniform_order(M, N, block_size_m, block_size_n, group_size_m))
        else:
            per_rank_orders.append(
                generate_frontier_order(
                    rank=rank,
                    world_size=world_size,
                    M=M,
                    N=N,
                    chunk_rows=chunk_rows,
                    frontier_chunks=frontier_chunks,
                    block_size_m=block_size_m,
                    block_size_n=block_size_n,
                    group_size_m=group_size_m,
                )
            )

    source_panel_tile_counts = [defaultdict(int) for _ in range(world_size)]
    source_panel_ready = [set() for _ in range(world_size)]
    panel_ready_sources = {panel: 0 for panel in panels}
    prestart_contrib_touch_set: set[tuple[int, PanelId]] = set()
    prestart_tile_work: dict[tuple[int, PanelId], int] = defaultdict(int)

    first_consumer_start_step = None
    wakeup_panels: list[PanelId] = []
    total_steps = len(per_rank_orders[0])

    for step in range(total_steps):
        newly_all_source_ready: list[PanelId] = []
        for rank in range(world_size):
            global_pid_m, pid_n = per_rank_orders[rank][step]
            touched_panels = panels_touched_by_tile(
                global_pid_m=global_pid_m,
                pid_n=pid_n,
                world_size=world_size,
                M=M,
                N=N,
                chunk_rows=chunk_rows,
                n_bands=n_bands,
                block_size_m=block_size_m,
                block_size_n=block_size_n,
            )
            for panel in touched_panels:
                prestart_contrib_touch_set.add((rank, panel))
                prestart_tile_work[(rank, panel)] += 1
                source_panel_tile_counts[rank][panel] += 1
                if (
                    panel not in source_panel_ready[rank]
                    and source_panel_tile_counts[rank][panel] == panel_required[panel]
                ):
                    source_panel_ready[rank].add(panel)
                    panel_ready_sources[panel] += 1
                    if panel_ready_sources[panel] == world_size:
                        newly_all_source_ready.append(panel)

        if newly_all_source_ready:
            first_consumer_start_step = step + 1
            wakeup_panels = sorted(
                set(newly_all_source_ready),
                key=lambda p: (p.destination_rank, p.chunk_id, p.band_id),
            )
            break

    if first_consumer_start_step is None:
        raise RuntimeError(f"{policy} never produced an all-source-ready panel; check the schedule simulation.")

    return finalize_result(
        policy=policy,
        world_size=world_size,
        M=M,
        N=N,
        chunk_rows=chunk_rows,
        n_bands=n_bands,
        frontier_chunks=frontier_chunks,
        block_size_m=block_size_m,
        block_size_n=block_size_n,
        group_size_m=group_size_m,
        first_consumer_start_step=first_consumer_start_step,
        wakeup_panels=wakeup_panels,
        prestart_tile_work=prestart_tile_work,
        prestart_contrib_touch_set=prestart_contrib_touch_set,
    )


def next_round_robin_panel(panel_list: list[PanelId], remaining: dict[PanelId, int], cursor: int) -> tuple[PanelId, int]:
    num_panels = len(panel_list)
    for offset in range(num_panels):
        idx = (cursor + offset) % num_panels
        panel = panel_list[idx]
        if remaining[panel] > 0:
            return panel, (idx + 1) % num_panels
    raise RuntimeError("no remaining panel found in round-robin scheduler")


def next_frontier_panel(panel_list: list[PanelId], remaining: dict[PanelId, int], cursor: int) -> tuple[PanelId, int]:
    idx = cursor
    while idx < len(panel_list) and remaining[panel_list[idx]] == 0:
        idx += 1
    if idx >= len(panel_list):
        raise RuntimeError("no remaining panel found in frontier-first scheduler")
    return panel_list[idx], idx


def classify_policy_panel_policy(
    policy: str,
    world_size: int,
    M: int,
    N: int,
    chunk_rows: int,
    n_bands: int,
    frontier_chunks: int,
    block_size_m: int,
    block_size_n: int,
    group_size_m: int,
) -> dict[str, object]:
    del group_size_m
    if policy not in {"uniform", "frontier_first"}:
        raise ValueError(f"unknown policy: {policy}")
    if M % world_size != 0:
        raise ValueError(f"M must be divisible by world_size for RS analysis, got M={M}, world_size={world_size}")

    panel_required = panel_required_tiles(
        world_size=world_size,
        M=M,
        N=N,
        chunk_rows=chunk_rows,
        n_bands=n_bands,
        block_size_m=block_size_m,
        block_size_n=block_size_n,
    )
    all_panels = sorted(panel_required.keys(), key=lambda p: (p.destination_rank, p.chunk_id, p.band_id))
    frontier_panels = [panel for panel in all_panels if panel.chunk_id < frontier_chunks]
    tail_panels = [panel for panel in all_panels if panel.chunk_id >= frontier_chunks]
    if not frontier_panels:
        frontier_panels = list(all_panels)
        tail_panels = []

    source_remaining = [dict(panel_required) for _ in range(world_size)]
    source_panel_ready = [set() for _ in range(world_size)]
    panel_ready_sources = {panel: 0 for panel in all_panels}
    prestart_contrib_touch_set: set[tuple[int, PanelId]] = set()
    prestart_tile_work: dict[tuple[int, PanelId], int] = defaultdict(int)

    rr_cursors = [0 for _ in range(world_size)]
    frontier_cursors = [0 for _ in range(world_size)]
    phase = ["frontier" for _ in range(world_size)]

    first_consumer_start_step = None
    wakeup_panels: list[PanelId] = []
    max_steps = sum(panel_required.values())

    for step in range(max_steps):
        newly_all_source_ready: list[PanelId] = []
        for rank in range(world_size):
            if policy == "uniform":
                panel, next_cursor = next_round_robin_panel(all_panels, source_remaining[rank], rr_cursors[rank])
                rr_cursors[rank] = next_cursor
            else:
                active_panels = frontier_panels if phase[rank] == "frontier" else tail_panels
                if not active_panels:
                    active_panels = all_panels
                try:
                    panel, next_cursor = next_frontier_panel(active_panels, source_remaining[rank], frontier_cursors[rank])
                except RuntimeError:
                    if phase[rank] == "frontier":
                        phase[rank] = "tail"
                        frontier_cursors[rank] = 0
                        active_panels = tail_panels if tail_panels else all_panels
                        panel, next_cursor = next_frontier_panel(active_panels, source_remaining[rank], frontier_cursors[rank])
                    else:
                        raise
                frontier_cursors[rank] = next_cursor

            prestart_contrib_touch_set.add((rank, panel))
            prestart_tile_work[(rank, panel)] += 1
            source_remaining[rank][panel] -= 1
            if source_remaining[rank][panel] == 0 and panel not in source_panel_ready[rank]:
                source_panel_ready[rank].add(panel)
                panel_ready_sources[panel] += 1
                if panel_ready_sources[panel] == world_size:
                    newly_all_source_ready.append(panel)
                if policy == "frontier_first" and phase[rank] == "frontier":
                    frontier_cursors[rank] += 1

        if newly_all_source_ready:
            first_consumer_start_step = step + 1
            wakeup_panels = sorted(
                set(newly_all_source_ready),
                key=lambda p: (p.destination_rank, p.chunk_id, p.band_id),
            )
            break

    if first_consumer_start_step is None:
        raise RuntimeError(f"{policy} never produced an all-source-ready panel in panel-policy analysis mode.")

    return finalize_result(
        policy=policy,
        world_size=world_size,
        M=M,
        N=N,
        chunk_rows=chunk_rows,
        n_bands=n_bands,
        frontier_chunks=frontier_chunks,
        block_size_m=block_size_m,
        block_size_n=block_size_n,
        group_size_m=-1,
        first_consumer_start_step=first_consumer_start_step,
        wakeup_panels=wakeup_panels,
        prestart_tile_work=prestart_tile_work,
        prestart_contrib_touch_set=prestart_contrib_touch_set,
    )


def finalize_result(
    policy: str,
    world_size: int,
    M: int,
    N: int,
    chunk_rows: int,
    n_bands: int,
    frontier_chunks: int,
    block_size_m: int,
    block_size_n: int,
    group_size_m: int,
    first_consumer_start_step: int,
    wakeup_panels: list[PanelId],
    prestart_tile_work: dict[tuple[int, PanelId], int],
    prestart_contrib_touch_set: set[tuple[int, PanelId]],
) -> dict[str, object]:
    wakeup_panel_set = set(wakeup_panels)
    wakeup_contribs = 0
    non_wakeup_contribs = 0
    wakeup_tile_updates = 0
    non_wakeup_tile_updates = 0
    for contrib, work in prestart_tile_work.items():
        _, panel = contrib
        if panel in wakeup_panel_set:
            wakeup_contribs += 1
            wakeup_tile_updates += work
        else:
            non_wakeup_contribs += 1
            non_wakeup_tile_updates += work

    m_per_rank = M // world_size
    num_chunks = cdiv(m_per_rank, chunk_rows)
    return {
        "policy": policy,
        "world_size": world_size,
        "M": M,
        "N": N,
        "m_per_rank": m_per_rank,
        "chunk_rows": chunk_rows,
        "num_chunks": num_chunks,
        "n_bands": n_bands,
        "frontier_chunks": frontier_chunks,
        "block_size_m": block_size_m,
        "block_size_n": block_size_n,
        "group_size_m": group_size_m,
        "first_consumer_start_step": first_consumer_start_step,
        "first_panel_ready_step": first_consumer_start_step,
        "wakeup_panel_count": len(wakeup_panels),
        "wakeup_panels": "|".join(panel.to_label() for panel in wakeup_panels),
        "distinct_panel_contribs_touched_prestart": len(prestart_contrib_touch_set),
        "wakeup_panel_contribs_touched_prestart": wakeup_contribs,
        "non_wakeup_panel_contribs_touched_prestart": non_wakeup_contribs,
        "tile_updates_prestart": sum(prestart_tile_work.values()),
        "wakeup_tile_updates_prestart": wakeup_tile_updates,
        "non_wakeup_tile_updates_prestart": non_wakeup_tile_updates,
    }


def setup_style(plt):
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.7,
            "legend.frameon": False,
            "savefig.bbox": "tight",
        }
    )


def plot_results(
    rows: list[dict[str, object]],
    output_dir: Path,
    formats: list[str],
    dpi: int,
    title: str,
    subtitle: str,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError(f"matplotlib is required for plotting: {exc}") from exc

    setup_style(plt)
    label_map = {
        "uniform": "Uniform",
        "frontier_first": "Frontier-first",
    }
    labels = [label_map[str(row["policy"])] for row in rows]
    wakeup_vals = [int(row["wakeup_panel_contribs_touched_prestart"]) for row in rows]
    non_wakeup_vals = [int(row["non_wakeup_panel_contribs_touched_prestart"]) for row in rows]
    y_positions = list(range(len(rows)))

    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    bar_height = 0.46
    ax.barh(y_positions, wakeup_vals, color=WAKEUP_COLOR, height=bar_height, label="Wake-up contributions")
    ax.barh(
        y_positions,
        non_wakeup_vals,
        left=wakeup_vals,
        color=NON_WAKEUP_COLOR,
        height=bar_height,
        label="Non-wake-up contributions",
    )

    for idx, row in enumerate(rows):
        total = int(row["distinct_panel_contribs_touched_prestart"])
        first_step = int(row["first_consumer_start_step"])
        wakeup_panel_count = int(row["wakeup_panel_count"])
        ax.text(
            total + max(1.0, total * 0.025),
            idx,
            f"step={first_step}\nwake-up panels={wakeup_panel_count}",
            va="center",
            ha="left",
            fontsize=8.5,
            color=ANNOTATION_COLOR,
        )

    ax.set_yticks(y_positions, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Distinct producer-touched panel contributions before first consumer start")
    ax.set_title(title, pad=24)
    ax.text(0.0, 1.09, subtitle, transform=ax.transAxes, fontsize=8.5, color=ANNOTATION_COLOR, va="bottom")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)

    max_total = max(int(row["distinct_panel_contribs_touched_prestart"]) for row in rows)
    ax.set_xlim(0, max_total * 1.48 if max_total > 0 else 1.0)
    fig.subplots_adjust(top=0.78, bottom=0.24, left=0.16, right=0.96)

    fig_name = "rs_frontier_prestart_panel_mix"
    output_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fmt = fmt.strip().lower()
        if not fmt:
            continue
        fig.savefig(output_dir / f"{fig_name}.{fmt}", dpi=dpi if fmt == "png" else None)
    plt.close(fig)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "policy",
        "world_size",
        "M",
        "N",
        "m_per_rank",
        "chunk_rows",
        "num_chunks",
        "n_bands",
        "frontier_chunks",
        "block_size_m",
        "block_size_n",
        "group_size_m",
        "first_consumer_start_step",
        "first_panel_ready_step",
        "wakeup_panel_count",
        "wakeup_panels",
        "distinct_panel_contribs_touched_prestart",
        "wakeup_panel_contribs_touched_prestart",
        "non_wakeup_panel_contribs_touched_prestart",
        "tile_updates_prestart",
        "wakeup_tile_updates_prestart",
        "non_wakeup_tile_updates_prestart",
    ]
    with open(path, "w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def default_output_dir() -> Path:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(__file__).resolve().parent / "rs_frontier_schedule_results" / run_id


def main():
    args = parse_args()
    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir()
    classifier = classify_policy_panel_policy if args.analysis_mode == "panel_policy" else classify_policy_kernel_tile
    rows = [
        classifier(
            policy="uniform",
            world_size=args.world_size,
            M=args.M,
            N=args.N,
            chunk_rows=args.chunk_rows,
            n_bands=args.n_bands,
            frontier_chunks=args.frontier_chunks,
            block_size_m=args.block_size_m,
            block_size_n=args.block_size_n,
            group_size_m=args.group_size_m,
        ),
        classifier(
            policy="frontier_first",
            world_size=args.world_size,
            M=args.M,
            N=args.N,
            chunk_rows=args.chunk_rows,
            n_bands=args.n_bands,
            frontier_chunks=args.frontier_chunks,
            block_size_m=args.block_size_m,
            block_size_n=args.block_size_n,
            group_size_m=args.group_size_m,
        ),
    ]

    csv_path = output_dir / "rs_frontier_schedule_summary.csv"
    write_csv(csv_path, rows)
    plot_results(
        rows=rows,
        output_dir=output_dir,
        formats=[fmt.strip() for fmt in args.formats.split(",")],
        dpi=args.dpi,
        title=args.title,
        subtitle=args.subtitle,
    )

    print(f"[rs-frontier-schedule] summary csv: {csv_path}")
    print(f"[rs-frontier-schedule] figure dir: {output_dir}")
    print(f"[rs-frontier-schedule] analysis_mode: {args.analysis_mode}")
    for row in rows:
        print(
            "[rs-frontier-schedule] "
            f"policy={row['policy']}, "
            f"first_consumer_start_step={row['first_consumer_start_step']}, "
            f"prestart_contribs={row['distinct_panel_contribs_touched_prestart']}, "
            f"wakeup_contribs={row['wakeup_panel_contribs_touched_prestart']}, "
            f"non_wakeup_contribs={row['non_wakeup_panel_contribs_touched_prestart']}, "
            f"wakeup_panels={row['wakeup_panels']}"
        )


if __name__ == "__main__":
    main()
