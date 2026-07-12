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
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
PKG_ROOT = Path(__file__).resolve().parents[1]
BENCH_DIR = Path(__file__).resolve().parent

AG_SCRIPT = BENCH_DIR / "bench_new_allgather_gemm_8rank.py"
RS_SCRIPT = BENCH_DIR / "bench_3rdv5_frontier_windowed_panel_gemmrs.py"
AR_SCRIPT = BENCH_DIR / "bench_new_windowed_panel_gemm_allreduce.py"

CHILD_CSV_DIR = PKG_ROOT / "csv"
OUTPUT_ROOT = BENCH_DIR / "figure3_2_results"

AG_CHILD_CSV = "perf_new_ag_gemm_{nproc}_ranks.csv"
RS_CHILD_CSV = "perf_3rdv5_frontier_windowed_panel_gemm_rs_{nproc}_ranks.csv"
AR_CHILD_CSV = "perf_new_windowed_panel_gemm_allreduce_{nproc}_ranks.csv"

AG_TILE_RE = re.compile(
    r"tile-ready config:\s+enabled=(?P<enabled>True|False),\s+"
    r"tile_rows_per_chunk=(?P<tile_rows_per_chunk>\d+),\s+"
    r"num_tile_chunks=(?P<num_tile_chunks>\d+)"
)
AR_BASELINE_SKIP_RE = re.compile(r"baseline skipped: .*allocation failed", re.IGNORECASE)
AR_NEW_SKIP_RE = re.compile(r"new kernel skipped: .*allocation failed", re.IGNORECASE)


def parse_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        text = str(value).strip()
        if not text or text.lower() == "nan":
            return None
        return float(text)
    except (TypeError, ValueError):
        return None


def normalize_success(driver_status: object) -> int:
    return int(str(driver_status) == "ok")


def semantic_columns(family: str, row: dict[str, object]) -> dict[str, object]:
    if family == "ag":
        return {
            "figure_row": 1,
            "observed_bottleneck": "coarse ready unit",
            "runtime_symptom": "consumer starts too late",
            "design_principle": "Fine-Grained Ready Units",
            "expected_effect": "earlier consumer activation",
        }
    if family == "rs":
        return {
            "figure_row": 2,
            "observed_bottleneck": "wrong producer order",
            "runtime_symptom": "critical frontier served too late",
            "design_principle": "Bifurcated Overlap Orchestration",
            "expected_effect": "shorter producer-to-consumer critical path",
        }
    runtime_symptom = "allocation failed / shallow pipeline"
    if parse_float(row.get("new_internal_overlap_ratio")) is not None:
        runtime_symptom = "frontier delayed plus resource-bounded overlap"
    return {
        "figure_row": 3,
        "observed_bottleneck": "staging footprint blow-up",
        "runtime_symptom": runtime_symptom,
        "design_principle": "Active Windowed Symmetric Staging",
        "expected_effect": "bounded physical footprint and wider feasible region",
    }


def enrich_row(family: str, row: dict[str, object]) -> dict[str, object]:
    enriched = dict(row)
    enriched.update(semantic_columns(family, enriched))
    enriched["success"] = normalize_success(enriched.get("driver_status"))
    if family == "ag":
        new_ms = parse_float(enriched.get("new dist-triton ag gemm latency (ms)"))
        base_ms = parse_float(enriched.get("dist-triton ag gemm latency (ms)"))
        torch_ms = parse_float(enriched.get("torch ag gemm latency (ms)"))
        if new_ms is not None:
            enriched["new_triton_total_ms"] = new_ms
        if base_ms is not None:
            enriched["base_triton_total_ms"] = base_ms
        if torch_ms is not None:
            enriched["torch_total_ms"] = torch_ms
        first_ready = parse_float(enriched.get("first_ready_ts_ms"))
        first_consumer = parse_float(enriched.get("first_consumer_ts_ms"))
        completion = parse_float(enriched.get("last_completion_ts_ms"))
        if first_ready is not None and completion is not None:
            enriched["ready_to_completion_tail_ms"] = completion - first_ready
        if first_ready is not None and first_consumer is not None:
            enriched["consumer_activation_gap_ms"] = first_consumer - first_ready
    elif family == "rs":
        v2_total = parse_float(enriched.get("v2_total_ms"))
        old_total = parse_float(enriched.get("old_total_ms"))
        torch_total = parse_float(enriched.get("torch_total_ms"))
        if v2_total is not None:
            enriched["new_total_latency"] = v2_total
        if old_total is not None:
            enriched["baseline_total_latency"] = old_total
        if torch_total is not None:
            enriched["torch_total_latency"] = torch_total
        enriched["internal_overlap_ratio"] = enriched.get("v2_internal_overlap_ratio")
        enriched["overlap_ratio_vs_torch_serial"] = enriched.get("v2_internal_overlap_ratio")
    elif family == "ar":
        baseline_status = str(enriched.get("baseline_status", ""))
        new_status = str(enriched.get("new_status", ""))
        enriched["baseline_failed"] = int(baseline_status not in {"ok", "disabled"})
        enriched["new_failed"] = int(new_status != "ok")
        if not enriched.get("skipped_reason") and baseline_status not in {"ok", "disabled", ""}:
            enriched["failure_reason"] = baseline_status
        else:
            enriched["failure_reason"] = enriched.get("skipped_reason", "")
        enriched["internal_overlap_ratio"] = enriched.get("new_internal_overlap_ratio")
        enriched["overlap_ratio_vs_torch_serial"] = enriched.get("new_overlap_ratio_vs_torch_serial")
    return enriched


@dataclass(frozen=True)
class ShapeSpec:
    tag: str
    M: int
    N: int
    K: int


@dataclass(frozen=True)
class VariantSpec:
    tag: str
    params: dict[str, object]
    note: str = ""


def default_ag_shapes() -> list[ShapeSpec]:
    # These presets are only protocol defaults for Figure 3-2. Replace them if
    # your platform shows a different comm/compute balance.
    return [
        ShapeSpec("comm_bound", 4096, 28672, 8192),
        ShapeSpec("balanced", 8192, 28672, 8192),
        ShapeSpec("compute_bound", 16384, 28672, 16384),
    ]


def default_rs_shapes() -> list[ShapeSpec]:
    return [
        ShapeSpec("balanced", 8192, 29568, 8192),
        ShapeSpec("wide_n", 8192, 49152, 12288),
    ]


def default_ar_shapes() -> list[ShapeSpec]:
    return [
        ShapeSpec("balanced", 16384, 49152, 12288),
        ShapeSpec("resource_pressure", 16384, 53248, 16384),
    ]


def default_ag_variants() -> list[VariantSpec]:
    return [
        VariantSpec(
            "rank_ready",
            {
                "enable_tile_ready": False,
                "tile_rows_per_chunk": 0,
                "target_chunks_per_rank": 2,
                "min_tile_rows_per_chunk": 1024,
            },
            "Disable row-chunk-ready to approximate coarse rank-ready release.",
        ),
        VariantSpec(
            "tile_ready_heuristic",
            {
                "enable_tile_ready": True,
                "tile_rows_per_chunk": 0,
                "target_chunks_per_rank": 2,
                "min_tile_rows_per_chunk": 1024,
            },
            "Use the benchmark heuristic to expose row-chunk-ready release.",
        ),
        VariantSpec(
            "tile_ready_fixed1024",
            {
                "enable_tile_ready": True,
                "tile_rows_per_chunk": 1024,
                "target_chunks_per_rank": 2,
                "min_tile_rows_per_chunk": 1024,
            },
            "Use a fixed 1024-row ready chunk for direct Figure 3-2 comparison.",
        ),
    ]


def default_rs_variants() -> list[VariantSpec]:
    return [
        VariantSpec(
            "window1_frontier1",
            {
                "chunk_rows": 512,
                "target_chunks_per_rank": 2,
                "min_chunk_rows": 512,
                "active_chunk_window": 1,
                "stage_slots": 2,
                "steady_sms": 8,
                "tail_sms": 20,
                "tail_chunk_window": 1,
                "comm_lanes": 2,
                "n_bands": 2,
                "frontier_chunks": 1,
                "local_seed_direct": True,
            },
            "Minimal active window for shallow frontier service.",
        ),
        VariantSpec(
            "window2_frontier1",
            {
                "chunk_rows": 512,
                "target_chunks_per_rank": 2,
                "min_chunk_rows": 512,
                "active_chunk_window": 2,
                "stage_slots": 4,
                "steady_sms": 8,
                "tail_sms": 20,
                "tail_chunk_window": 1,
                "comm_lanes": 2,
                "n_bands": 2,
                "frontier_chunks": 1,
                "local_seed_direct": True,
            },
            "Moderate window with a single frontier chunk.",
        ),
        VariantSpec(
            "window2_frontier2",
            {
                "chunk_rows": 512,
                "target_chunks_per_rank": 2,
                "min_chunk_rows": 512,
                "active_chunk_window": 2,
                "stage_slots": 4,
                "steady_sms": 8,
                "tail_sms": 20,
                "tail_chunk_window": 1,
                "comm_lanes": 2,
                "n_bands": 2,
                "frontier_chunks": 2,
                "local_seed_direct": True,
            },
            "Increase frontier emphasis without deepening the active window.",
        ),
        VariantSpec(
            "window4_frontier1",
            {
                "chunk_rows": 512,
                "target_chunks_per_rank": 2,
                "min_chunk_rows": 512,
                "active_chunk_window": 4,
                "stage_slots": 8,
                "steady_sms": 8,
                "tail_sms": 20,
                "tail_chunk_window": 1,
                "comm_lanes": 2,
                "n_bands": 2,
                "frontier_chunks": 1,
                "local_seed_direct": True,
            },
            "Deep window with weak frontier emphasis.",
        ),
        VariantSpec(
            "window4_frontier2",
            {
                "chunk_rows": 512,
                "target_chunks_per_rank": 2,
                "min_chunk_rows": 512,
                "active_chunk_window": 4,
                "stage_slots": 8,
                "steady_sms": 8,
                "tail_sms": 20,
                "tail_chunk_window": 1,
                "comm_lanes": 2,
                "n_bands": 2,
                "frontier_chunks": 2,
                "local_seed_direct": True,
            },
            "Deep window with stronger frontier-first scheduling.",
        ),
    ]


def default_ar_variants() -> list[VariantSpec]:
    return [
        VariantSpec(
            "window2_frontier1_stage4_stripe256",
            {
                "chunk_rows": 1024,
                "stripe_rows": 256,
                "target_chunks": 4,
                "min_chunk_rows": 512,
                "active_chunk_window": 2,
                "n_bands": 2,
                "frontier_chunks": 1,
                "stage_slots": 4,
                "num_comm_sms": 24,
                "comm_lanes": 2,
                "streaming_depth": 1,
            },
            "Conservative active window and staging footprint.",
        ),
        VariantSpec(
            "window2_frontier2_stage8_stripe256",
            {
                "chunk_rows": 1024,
                "stripe_rows": 256,
                "target_chunks": 4,
                "min_chunk_rows": 512,
                "active_chunk_window": 2,
                "n_bands": 2,
                "frontier_chunks": 2,
                "stage_slots": 8,
                "num_comm_sms": 24,
                "comm_lanes": 2,
                "streaming_depth": 1,
            },
            "Balanced frontier window and staging depth.",
        ),
        VariantSpec(
            "window4_frontier2_stage8_stripe256",
            {
                "chunk_rows": 1024,
                "stripe_rows": 256,
                "target_chunks": 4,
                "min_chunk_rows": 512,
                "active_chunk_window": 4,
                "n_bands": 2,
                "frontier_chunks": 2,
                "stage_slots": 8,
                "num_comm_sms": 24,
                "comm_lanes": 2,
                "streaming_depth": 1,
            },
            "Increase active-window depth while keeping staging constant.",
        ),
        VariantSpec(
            "window4_frontier2_stage16_stripe256",
            {
                "chunk_rows": 1024,
                "stripe_rows": 256,
                "target_chunks": 4,
                "min_chunk_rows": 512,
                "active_chunk_window": 4,
                "n_bands": 2,
                "frontier_chunks": 2,
                "stage_slots": 16,
                "num_comm_sms": 24,
                "comm_lanes": 2,
                "streaming_depth": 1,
            },
            "Push stage slots to probe the feasibility boundary.",
        ),
        VariantSpec(
            "window2_frontier2_stage8_stripe128",
            {
                "chunk_rows": 1024,
                "stripe_rows": 128,
                "target_chunks": 4,
                "min_chunk_rows": 512,
                "active_chunk_window": 2,
                "n_bands": 2,
                "frontier_chunks": 2,
                "stage_slots": 8,
                "num_comm_sms": 24,
                "comm_lanes": 2,
                "streaming_depth": 1,
            },
            "Tighter stripe packing to test compact staging.",
        ),
        VariantSpec(
            "window2_frontier2_stage8_stripe512",
            {
                "chunk_rows": 1024,
                "stripe_rows": 512,
                "target_chunks": 4,
                "min_chunk_rows": 512,
                "active_chunk_window": 2,
                "n_bands": 2,
                "frontier_chunks": 2,
                "stage_slots": 8,
                "num_comm_sms": 24,
                "comm_lanes": 2,
                "streaming_depth": 1,
            },
            "Wider stripe granularity to increase physical staging pressure.",
        ),
    ]


def parse_bool(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise ValueError(f"invalid boolean value: {value}")


def parse_shape_specs(value: str | None, defaults: list[ShapeSpec]) -> list[ShapeSpec]:
    if value is None:
        return defaults
    value = value.strip()
    if not value:
        return defaults

    shapes: list[ShapeSpec] = []
    for item in value.split(";"):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            tag, dims = item.split(":", 1)
            shape_tag = tag.strip()
        else:
            shape_tag = f"shape{len(shapes)}"
            dims = item
        parts = [part.strip() for part in dims.split(",") if part.strip()]
        if len(parts) != 3:
            raise ValueError(f"invalid shape spec: {item}")
        M, N, K = (int(part) for part in parts)
        shapes.append(ShapeSpec(shape_tag, M, N, K))
    return shapes


def append_bool_flag(cmd: list[str], flag: str, value: bool) -> None:
    cmd.append(flag if value else f"--no-{flag[2:]}")


def child_csv_path(family: str, nproc: int) -> Path:
    if family == "ag":
        return CHILD_CSV_DIR / AG_CHILD_CSV.format(nproc=nproc)
    if family == "rs":
        return CHILD_CSV_DIR / RS_CHILD_CSV.format(nproc=nproc)
    if family == "ar":
        return CHILD_CSV_DIR / AR_CHILD_CSV.format(nproc=nproc)
    raise ValueError(f"unsupported family: {family}")


def build_ag_cmd(args, shape: ShapeSpec, variant: VariantSpec) -> list[str]:
    params = variant.params
    cmd = [
        args.torchrun_bin,
        "--nproc_per_node",
        str(args.nproc_per_node),
        str(AG_SCRIPT),
        "--M",
        str(shape.M),
        "--N",
        str(shape.N),
        "--K",
        str(shape.K),
        "--iters",
        str(args.iters),
        "--warmup_iters",
        str(args.warmup_iters),
        "--dtype",
        args.dtype,
        "--target_chunks_per_rank",
        str(params["target_chunks_per_rank"]),
        "--min_tile_rows_per_chunk",
        str(params["min_tile_rows_per_chunk"]),
        "--tile_rows_per_chunk",
        str(params["tile_rows_per_chunk"]),
        "--dump_csv",
    ]
    append_bool_flag(cmd, "--autotune", args.autotune)
    append_bool_flag(cmd, "--trans_b", args.trans_b)
    append_bool_flag(cmd, "--enable_tile_ready", bool(params["enable_tile_ready"]))
    if args.profile:
        cmd.append("--profile")
    return cmd


def build_rs_cmd(args, shape: ShapeSpec, variant: VariantSpec) -> list[str]:
    params = variant.params
    cmd = [
        args.torchrun_bin,
        "--nproc_per_node",
        str(args.nproc_per_node),
        str(RS_SCRIPT),
        "--M",
        str(shape.M),
        "--N",
        str(shape.N),
        "--K",
        str(shape.K),
        "--mode",
        "all",
        "--iters",
        str(args.iters),
        "--warmup_iters",
        str(args.warmup_iters),
        "--dtype",
        args.dtype,
        "--chunk_rows",
        str(params["chunk_rows"]),
        "--target_chunks_per_rank",
        str(params["target_chunks_per_rank"]),
        "--min_chunk_rows",
        str(params["min_chunk_rows"]),
        "--active_chunk_window",
        str(params["active_chunk_window"]),
        "--stage_slots",
        str(params["stage_slots"]),
        "--steady_sms",
        str(params["steady_sms"]),
        "--tail_sms",
        str(params["tail_sms"]),
        "--tail_chunk_window",
        str(params["tail_chunk_window"]),
        "--comm_lanes",
        str(params["comm_lanes"]),
        "--n_bands",
        str(params["n_bands"]),
        "--frontier_chunks",
        str(params["frontier_chunks"]),
        "--dump_csv",
    ]
    append_bool_flag(cmd, "--autotune", args.autotune)
    append_bool_flag(cmd, "--trans_b", args.trans_b)
    append_bool_flag(cmd, "--local_seed_direct", bool(params["local_seed_direct"]))
    if args.profile:
        cmd.append("--profile")
    return cmd


def build_ar_cmd(args, shape: ShapeSpec, variant: VariantSpec) -> list[str]:
    params = variant.params
    cmd = [
        args.torchrun_bin,
        "--nproc_per_node",
        str(args.nproc_per_node),
        str(AR_SCRIPT),
        "--M",
        str(shape.M),
        "--N",
        str(shape.N),
        "--K",
        str(shape.K),
        "--iters",
        str(args.iters),
        "--warmup_iters",
        str(args.warmup_iters),
        "--dtype",
        args.dtype,
        "--chunk_rows",
        str(params["chunk_rows"]),
        "--stripe_rows",
        str(params["stripe_rows"]),
        "--target_chunks",
        str(params["target_chunks"]),
        "--min_chunk_rows",
        str(params["min_chunk_rows"]),
        "--active_chunk_window",
        str(params["active_chunk_window"]),
        "--n_bands",
        str(params["n_bands"]),
        "--frontier_chunks",
        str(params["frontier_chunks"]),
        "--stage_slots",
        str(params["stage_slots"]),
        "--num_comm_sms",
        str(params["num_comm_sms"]),
        "--comm_lanes",
        str(params["comm_lanes"]),
        "--streaming_depth",
        str(params["streaming_depth"]),
        "--baseline_num_comm_sms",
        str(args.baseline_num_comm_sms),
        "--dump_csv",
    ]
    append_bool_flag(cmd, "--autotune", args.autotune)
    append_bool_flag(cmd, "--run_baseline", args.run_baseline)
    append_bool_flag(cmd, "--check", args.check)
    if args.profile:
        cmd.append("--profile")
    return cmd


def command_for(family: str, args, shape: ShapeSpec, variant: VariantSpec) -> list[str]:
    if family == "ag":
        return build_ag_cmd(args, shape, variant)
    if family == "rs":
        return build_rs_cmd(args, shape, variant)
    if family == "ar":
        return build_ar_cmd(args, shape, variant)
    raise ValueError(f"unsupported family: {family}")


def prepare_child_csv(csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if csv_path.exists():
        csv_path.unlink()


def child_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    python_root = str(REPO_ROOT / "python")
    old_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = python_root if not old_pythonpath else f"{python_root}:{old_pythonpath}"
    return env


def run_child(cmd: list[str], log_path: Path, timeout_sec: float) -> tuple[int, str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(PKG_ROOT),
            env=child_env(),
            text=True,
            capture_output=True,
            timeout=timeout_sec if timeout_sec > 0 else None,
        )
        merged = proc.stdout + ("\n" + proc.stderr if proc.stderr else "")
        log_path.write_text(merged, encoding="utf-8")
        return proc.returncode, merged
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        merged = stdout + ("\n" + stderr if stderr else "")
        merged += f"\n[driver] timeout after {timeout_sec:.1f}s\n"
        log_path.write_text(merged, encoding="utf-8")
        return 124, merged


def read_single_row_csv(csv_path: Path) -> dict[str, str] | None:
    if not csv_path.exists():
        return None
    with open(csv_path, "r", encoding="utf-8") as fin:
        rows = list(csv.DictReader(fin))
    if not rows:
        return None
    return rows[0]


def archive_child_csv(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    return str(dst)


def extract_ag_tile_config(output: str) -> dict[str, object]:
    match = AG_TILE_RE.search(output)
    if not match:
        return {}
    return {
        "observed_enable_tile_ready": parse_bool(match.group("enabled")),
        "observed_tile_rows_per_chunk": int(match.group("tile_rows_per_chunk")),
        "observed_num_tile_chunks": int(match.group("num_tile_chunks")),
    }


def summarize_status(family: str, returncode: int, csv_row: dict[str, str] | None, output: str) -> dict[str, object]:
    status: dict[str, object] = {
        "returncode": returncode,
        "driver_status": "ok" if returncode == 0 else "child_failed",
    }
    if returncode != 0:
        return status
    if csv_row is None:
        status["driver_status"] = "missing_child_csv"
        return status

    if family == "ar":
        if AR_NEW_SKIP_RE.search(output):
            status["new_status"] = "oom"
            status["driver_status"] = "new_skipped"
            status["skipped_reason"] = "nvshmem_oom"
        else:
            status["new_status"] = "ok"
        if AR_BASELINE_SKIP_RE.search(output):
            status["baseline_status"] = "oom"
        elif "baseline_status" in csv_row:
            status["baseline_status"] = csv_row["baseline_status"]
        if csv_row.get("baseline_status") == "skipped":
            status["driver_status"] = "new_skipped"
            status["skipped_reason"] = "nvshmem_oom"
    return status


def write_summary_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    preferred = [
        "run_id",
        "family",
        "shape_tag",
        "variant_tag",
        "variant_note",
        "M",
        "N",
        "K",
        "driver_status",
        "returncode",
        "log_path",
        "raw_child_csv_path",
        "command",
    ]
    fieldnames: list[str] = []
    seen = set()
    for name in preferred:
        if any(name in row for row in rows):
            fieldnames.append(name)
            seen.add(name)
    for row in rows:
        for key in row:
            if key in seen:
                continue
            fieldnames.append(key)
            seen.add(key)

    with open(path, "w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_family(
    family: str,
    args,
    shapes: list[ShapeSpec],
    variants: list[VariantSpec],
    run_id: str,
    output_dir: Path,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for shape in shapes:
        for variant in variants:
            cmd = command_for(family, args, shape, variant)
            run_tag = f"{family}_{shape.tag}_{variant.tag}"
            log_path = output_dir / "logs" / family / f"{run_tag}.log"
            raw_csv_path = output_dir / "raw_child_csv" / family / f"{run_tag}.csv"
            child_csv = child_csv_path(family, args.nproc_per_node)
            prepare_child_csv(child_csv)

            if args.dry_run:
                row = {
                    "run_id": run_id,
                    "family": family,
                    "shape_tag": shape.tag,
                    "variant_tag": variant.tag,
                    "variant_note": variant.note,
                    "M": shape.M,
                    "N": shape.N,
                    "K": shape.K,
                    "driver_status": "dry_run",
                    "returncode": 0,
                    "log_path": str(log_path),
                    "raw_child_csv_path": str(raw_csv_path),
                    "command": " ".join(cmd),
                }
                row.update(variant.params)
                row = enrich_row(family, row)
                rows.append(row)
                print(f"[dry-run][{family}] {' '.join(cmd)}", flush=True)
                continue

            print(f"[run][{family}] shape={shape.tag} variant={variant.tag}", flush=True)
            returncode, output = run_child(cmd, log_path=log_path, timeout_sec=args.timeout_sec)
            csv_row = read_single_row_csv(child_csv)

            row: dict[str, object] = {
                "run_id": run_id,
                "family": family,
                "shape_tag": shape.tag,
                "variant_tag": variant.tag,
                "variant_note": variant.note,
                "M": shape.M,
                "N": shape.N,
                "K": shape.K,
                "log_path": str(log_path),
                "raw_child_csv_path": "",
                "command": " ".join(cmd),
            }
            row.update(variant.params)
            row.update(summarize_status(family, returncode, csv_row, output))

            if csv_row is not None:
                row.update(csv_row)
                row["raw_child_csv_path"] = archive_child_csv(child_csv, raw_csv_path)

            if family == "ag":
                row.update(extract_ag_tile_config(output))

            row = enrich_row(family, row)

            if returncode != 0 and args.fail_fast:
                rows.append(row)
                raise RuntimeError(f"{family} child benchmark failed: {' '.join(cmd)}")

            rows.append(row)
    return rows


def parse_args():
    parser = argparse.ArgumentParser(
        description="Unified Figure 3-2 experiment driver for AG ready, RS frontier, and AR windowed staging studies."
    )
    parser.add_argument("--family", default="all", choices=["all", "ag", "rs", "ar"])
    parser.add_argument("--nproc_per_node", type=int, default=4)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--warmup_iters", type=int, default=5)
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16"])
    parser.add_argument("--autotune", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--profile", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--trans_b", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--run_baseline", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--check", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--baseline_num_comm_sms", type=int, default=16)
    parser.add_argument("--timeout_sec", type=float, default=0.0)
    parser.add_argument("--fail_fast", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--dry_run", "--dry-run", dest="dry_run", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--torchrun_bin", default="torchrun")
    parser.add_argument("--output_root", default=str(OUTPUT_ROOT))
    parser.add_argument("--ag_shapes", default="")
    parser.add_argument("--rs_shapes", default="")
    parser.add_argument("--ar_shapes", default="")
    return parser.parse_args()


def main():
    args = parse_args()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    ag_shapes = parse_shape_specs(args.ag_shapes, default_ag_shapes())
    rs_shapes = parse_shape_specs(args.rs_shapes, default_rs_shapes())
    ar_shapes = parse_shape_specs(args.ar_shapes, default_ar_shapes())

    all_rows: dict[str, list[dict[str, object]]] = {}
    if args.family in {"all", "ag"}:
        all_rows["ag"] = run_family("ag", args, ag_shapes, default_ag_variants(), run_id, output_dir)
    if args.family in {"all", "rs"}:
        all_rows["rs"] = run_family("rs", args, rs_shapes, default_rs_variants(), run_id, output_dir)
    if args.family in {"all", "ar"}:
        all_rows["ar"] = run_family("ar", args, ar_shapes, default_ar_variants(), run_id, output_dir)

    for family, rows in all_rows.items():
        summary_path = output_dir / f"figure3_2_{family}_summary.csv"
        write_summary_csv(summary_path, rows)
        print(f"[summary] {family}: {summary_path}", flush=True)

    manifest_path = output_dir / "README.txt"
    manifest_path.write_text(
        "\n".join(
            [
                f"run_id={run_id}",
                f"family={args.family}",
                f"nproc_per_node={args.nproc_per_node}",
                f"iters={args.iters}",
                f"warmup_iters={args.warmup_iters}",
                f"dtype={args.dtype}",
                f"autotune={args.autotune}",
                f"profile={args.profile}",
                f"dry_run={args.dry_run}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[manifest] {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
