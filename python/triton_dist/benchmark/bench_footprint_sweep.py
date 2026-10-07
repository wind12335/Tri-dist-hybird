# Footprint sweep for the active-window figure:
#   fixed N,K; sweep M; compare baseline_rs (no reuse) vs new_rs_v5 (windowed).
# Each point runs bench_nvshmem_feasibility_case.py under torchrun and captures
# the rank-0 [probe-json] payload (status + actual symmetric bytes).
#
# Requires: source scripts/setenv.sh, LD_PRELOAD libstdc++, NVSHMEM_SYMMETRIC_SIZE.

import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

JSON_PREFIX = "[probe-json]"


def run_point(impl, M, N, K, dtype, nproc, chunk_rows, window, env_extra):
    cmd = [
        "torchrun", f"--nproc_per_node={nproc}",
        str(Path(__file__).parent / "bench_nvshmem_feasibility_case.py"),
        "--impl", impl, "--M", str(M), "--N", str(N), "--K", str(K),
        "--dtype", dtype,
    ]
    if impl == "new_rs_v5":
        cmd += ["--rs_chunk_rows", str(chunk_rows),
                "--rs_active_chunk_window", str(window)]
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env_extra,
                          timeout=1800)
    payload = None
    for line in (proc.stdout + "\n" + proc.stderr).splitlines():
        if JSON_PREFIX in line:
            try:
                payload = json.loads(line.split(JSON_PREFIX, 1)[1])
            except json.JSONDecodeError:
                continue
    row = {
        "impl": impl, "M": M, "N": N, "K": K, "dtype": dtype,
        "chunk_rows": chunk_rows if impl == "new_rs_v5" else "",
        "active_window": window if impl == "new_rs_v5" else "",
        "returncode": proc.returncode,
    }
    if payload is None:
        row.update(status="missing_json", actual_symm_total_bytes="",
                   reason=(proc.stderr or proc.stdout)[-300:].replace("\n", " "))
    else:
        row.update(status=payload.get("status", "unknown"),
                   actual_symm_total_bytes=payload.get("actual_symm_total_bytes",
                                                       payload.get("estimated_symm_total_bytes", "")),
                   reason=str(payload.get("reason", ""))[:300])
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--impls", default="baseline_rs,new_rs_v5")
    ap.add_argument("--M_values", default="8192,16384,32768,65536,131072")
    ap.add_argument("--N", type=int, default=53248)
    ap.add_argument("--K", type=int, default=16384)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--nproc", type=int, default=4)
    ap.add_argument("--chunk_rows", type=int, default=512)
    ap.add_argument("--active_window", type=int, default=4)
    ap.add_argument("--output_csv", required=True)
    args = ap.parse_args()

    env_extra = dict(__import__("os").environ)

    out = Path(args.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["impl", "M", "N", "K", "dtype", "chunk_rows", "active_window",
              "status", "actual_symm_total_bytes", "reason", "returncode"]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for impl in [x.strip() for x in args.impls.split(",") if x.strip()]:
            for M in [int(x) for x in args.M_values.split(",") if x.strip()]:
                row = run_point(impl, M, args.N, args.K, args.dtype,
                                args.nproc, args.chunk_rows, args.active_window,
                                env_extra)
                w.writerow(row)
                f.flush()
                print(f"[sweep] {impl} M={M}: status={row['status']} "
                      f"symm={row['actual_symm_total_bytes']}", flush=True)
    print(f"[sweep] done -> {out}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
