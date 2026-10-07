#!/usr/bin/env python3
"""Merge per-shape ``ag_ready_aggregated.csv`` files into one combined CSV.

Each shape sweep (``bench_ag_ready_granularity_sweep_eval.py``) writes its own
``ag_ready_aggregated.csv``: one row per ready-granularity cell. This script
concatenates the newest aggregated CSV under each given shape/run directory into
a single combined CSV that ``plot_ag_ready_ablation.py`` can consume directly.
The sweep's per-shape CSVs do not carry ``shape_tag`` / ``M`` / ``N`` / ``K``
(shape identity lives only in the result directory name), so the script injects
those columns from the directory name before writing the combined CSV. This lets
``plot_ag_ready_ablation.py`` group rows by shape. It also pre-computes a
``speedup`` column (= ``base_median_ms / new_median_ms``) so the plot reads the
speedup straight off the CSV.

Usage::

  python benchmark/merge_ag_ready_shapes.py \
      --shape_dirs benchmark/ag_ready_granularity_eval_results/16384x29568x8192 \
                   benchmark/ag_ready_granularity_eval_results/28672x28672x8192 \
      --output_csv benchmark/ag_ready_granularity_eval_results/ag_ready_combined.csv

  python benchmark/plot_ag_ready_ablation.py \
      --input_csv benchmark/ag_ready_granularity_eval_results/ag_ready_combined.csv \
      --output_dir overleaf_methodology_zh/figures

Use glob patterns in --shape_dirs if you prefer (the script also accepts a raw
path to a run directory, and picks the newest aggregated CSV inside it).
"""

from __future__ import annotations

import argparse
import csv
import glob
import re
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--shape_dirs", nargs="+", required=True,
                   help="shape/run dirs (or glob patterns) holding ag_ready_aggregated.csv")
    p.add_argument("--output_csv", default="",
                   help="combined CSV path (default: <parent-of-first>/ag_ready_combined.csv)")
    return p.parse_args()


def resolve_dirs(patterns: list[str]) -> list[Path]:
    dirs: list[Path] = []
    seen: set[str] = set()
    for pattern in patterns:
        if any(ch in pattern for ch in "*?["):
            matches = sorted(glob.glob(pattern))
        else:
            matches = [pattern]
        for m in matches:
            p = Path(m)
            if not p.is_dir():
                print(f"[warn] not a directory, skipping: {p}", file=sys.stderr)
                continue
            key = str(p.resolve())
            if key not in seen:
                seen.add(key)
                dirs.append(p)
    return dirs


def newest_aggregated(shape_dir: Path) -> Path | None:
    matches = sorted(shape_dir.rglob("ag_ready_aggregated.csv"),
                     key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def safe_float(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in ("nan", "none", ""):
        return None
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def shape_identity_from_csv_path(csv_path: Path) -> tuple[str, str | None, str | None, str | None]:
    """Derive (shape_tag, M, N, K) from the result dir hierarchy.

    The sweep writes ``<output_root>/<shape_tag>/<run_id>/ag_ready_aggregated.csv``,
    so the shape tag is the grandparent directory name (``"16384x29568x8192"``).
    When the name is exactly ``MxNxK`` we also parse M/N/K.
    """
    shape_tag = csv_path.parent.parent.name if csv_path.parent.parent else ""
    m = n = k = None
    mt = re.fullmatch(r"(\d+)x(\d+)x(\d+)", shape_tag)
    if mt:
        m, n, k = mt.groups()
    return shape_tag, m, n, k


def main() -> None:
    args = parse_args()
    dirs = resolve_dirs(args.shape_dirs)
    if not dirs:
        raise SystemExit("no shape dirs resolved")

    csvs: list[Path] = []
    for d in dirs:
        found = newest_aggregated(d)
        if found is None:
            print(f"[warn] no ag_ready_aggregated.csv under {d}; skipping", file=sys.stderr)
            continue
        csvs.append(found)

    if not csvs:
        raise SystemExit("no aggregated CSVs found")
    if len(csvs) != len(dirs):
        print(f"[warn] found {len(csvs)}/{len(dirs)} aggregated CSVs", file=sys.stderr)

    # Union of columns so every row keeps its fields.
    cols: list[str] = []
    rows: list[dict] = []
    for c in csvs:
        with c.open("r", encoding="utf-8-sig", newline="") as fin:
            reader = csv.DictReader(fin)
            for col in reader.fieldnames or []:
                if col not in cols:
                    cols.append(col)
            source_rows = [dict(row) for row in reader]
        # The aggregated CSVs do NOT carry shape_tag/M/N/K (shape identity lives
        # only in the directory name, e.g. "16384x29568x8192"). Inject it so the
        # plot can group by shape.
        shape_tag, m, n, k = shape_identity_from_csv_path(c)
        for row in source_rows:
            if not (row.get("shape_tag") or "").strip():
                row["shape_tag"] = shape_tag
            if not (row.get("M") or "").strip() and m is not None:
                row["M"] = m
            if not (row.get("N") or "").strip() and n is not None:
                row["N"] = n
            if not (row.get("K") or "").strip() and k is not None:
                row["K"] = k
            # Pre-compute the speedup (base / new, median convention) so the plot
            # reads it straight off the CSV instead of recomputing at render time.
            new_med = safe_float(row.get("new_median_ms"))
            base_med = safe_float(row.get("base_median_ms"))
            if new_med is not None and base_med is not None and new_med > 0:
                row["speedup"] = f"{base_med / new_med:.6f}"
        for col in ("shape_tag", "M", "N", "K"):
            if col not in cols:
                cols.append(col)
        if "speedup" not in cols:
            cols.append("speedup")
        rows.extend(source_rows)
        print(f"[merge] {c} ({len(source_rows)} rows, shape={shape_tag})")

    out = Path(args.output_csv) if args.output_csv else csvs[0].parent.parent / "ag_ready_combined.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    shapes = sorted({(r.get("shape_tag"), r.get("M"), r.get("N"), r.get("K")) for r in rows})
    print(f"\n[done] wrote {out} with {len(rows)} rows across {len(shapes)} shapes:")
    for tag, m, n, k in shapes:
        print(f"  - {tag}  M={m} N={n} K={k}")


if __name__ == "__main__":
    main()
