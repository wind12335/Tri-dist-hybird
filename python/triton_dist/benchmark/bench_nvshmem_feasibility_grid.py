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
import itertools
import json
import subprocess
import sys
from pathlib import Path

# python python/triton_dist/benchmark/bench_nvshmem_feasibility_grid.py \
#   --nproc_per_node 4 \
#   --M_values 4096,8192,16384 \
#   --N_values 14336,28672,29568,49152,53248 \
#   --K_values 4096,8192,12288,16384 \
#   --impls baseline_rs \
#   --dtype bfloat16 \
#   --timeout_s 240 \
#   --output_csv csv/nvshmem_baseline_rs_fixedM4096819216384.csv

JSON_PREFIX = "[probe-json]"
CASE_SCRIPT = Path(__file__).with_name("bench_nvshmem_feasibility_case.py")
PLOT_SCRIPT = Path(__file__).with_name("plot_nvshmem_feasibility.py")
REPO_ROOT = Path(__file__).resolve().parents[3]


def try_parse_probe_json(text: str) -> dict | None:
    decoder = json.JSONDecoder()
    for line in reversed(text.splitlines()):
        if JSON_PREFIX not in line:
            continue
        payload_text = line.split(JSON_PREFIX, 1)[1].lstrip()
        for candidate in [payload_text, payload_text[payload_text.find("{"):] if "{" in payload_text else ""]:
            if not candidate:
                continue
            try:
                parsed, _ = decoder.raw_decode(candidate)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                continue
    return None


def parse_int_list(raw: str) -> list[int]:
    values = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        values.append(int(item))
    if not values:
        raise ValueError(f"empty integer list: {raw!r}")
    return values


def parse_str_list(raw: str) -> list[str]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError(f"empty string list: {raw!r}")
    return values


def parse_args():
    parser = argparse.ArgumentParser(description="Grid driver for NVSHMEM feasibility probing.")
    parser.add_argument("--nproc_per_node", type=int, default=8)
    parser.add_argument("--M_values", type=str, required=True)
    parser.add_argument("--N_values", type=str, required=True)
    parser.add_argument("--K_values", type=str, default="8192")
    parser.add_argument("--impls", type=str, default="baseline_rs,new_rs_v5,baseline_ar,new_ar_v23")
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16"])
    parser.add_argument("--timeout_s", type=int, default=240)
    parser.add_argument("--output_csv", type=str, default="")
    parser.add_argument("--plot", action="store_true", default=False)
    parser.add_argument("--plot_dir", type=str, default="")
    parser.add_argument("--plot_formats", type=str, default="png,pdf")
    return parser.parse_known_args()


def run_case(nproc_per_node: int,
             timeout_s: int,
             impl: str,
             M: int,
             N: int,
             K: int,
             dtype: str,
             passthrough_args: list[str]) -> tuple[dict, str]:
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        f"--nproc_per_node={nproc_per_node}",
        str(CASE_SCRIPT),
        "--impl",
        impl,
        "--M",
        str(M),
        "--N",
        str(N),
        "--K",
        str(K),
        "--dtype",
        dtype,
        *passthrough_args,
    ]
    try:
        proc = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        payload = {
            "impl": impl,
            "M": M,
            "N": N,
            "K": K,
            "dtype": dtype,
            "status": "timeout",
            "reason": f"timeout after {timeout_s}s",
        }
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        return payload, f"{stdout}\n{stderr}".strip()

    combined = "\n".join([proc.stdout, proc.stderr]).strip()
    parsed = try_parse_probe_json(combined)

    if parsed is None:
        parsed = {
            "impl": impl,
            "M": M,
            "N": N,
            "K": K,
            "dtype": dtype,
            "status": "crash" if proc.returncode != 0 else "missing_json",
            "reason": f"returncode={proc.returncode}",
        }
        tail_lines = [line for line in combined.splitlines() if line.strip()]
        if tail_lines:
            parsed["reason"] = tail_lines[-1]
    return parsed, combined


def main():
    args, passthrough_args = parse_args()
    M_values = parse_int_list(args.M_values)
    N_values = parse_int_list(args.N_values)
    K_values = parse_int_list(args.K_values)
    impls = parse_str_list(args.impls)

    rows = []
    total = len(M_values) * len(N_values) * len(K_values) * len(impls)
    idx = 0
    for M, N, K, impl in itertools.product(M_values, N_values, K_values, impls):
        idx += 1
        print(f"[grid] case {idx}/{total}: impl={impl}, M={M}, N={N}, K={K}", flush=True)
        payload, combined_output = run_case(
            nproc_per_node=args.nproc_per_node,
            timeout_s=args.timeout_s,
            impl=impl,
            M=M,
            N=N,
            K=K,
            dtype=args.dtype,
            passthrough_args=passthrough_args,
        )
        payload["raw_output_tail"] = "\n".join(combined_output.splitlines()[-20:]) if combined_output else ""
        rows.append(payload)
        print(f"[grid] -> status={payload.get('status')}", flush=True)

    output_csv = Path(args.output_csv) if args.output_csv else (
        REPO_ROOT / "csv" / f"nvshmem_feasibility_grid_{args.nproc_per_node}_ranks.csv"
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(output_csv, "w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"[grid] csv file is dumped into {output_csv}", flush=True)

    by_impl = {}
    for row in rows:
        impl = row["impl"]
        status = row.get("status", "unknown")
        by_impl.setdefault(impl, {})
        by_impl[impl][status] = by_impl[impl].get(status, 0) + 1

    print("[grid] summary:", flush=True)
    for impl in impls:
        counts = by_impl.get(impl, {})
        ordered = ", ".join(f"{status}={counts[status]}" for status in sorted(counts))
        print(f"  {impl}: {ordered}", flush=True)

    if args.plot:
        plot_dir = Path(args.plot_dir) if args.plot_dir else output_csv.with_suffix("")
        plot_cmd = [
            sys.executable,
            str(PLOT_SCRIPT),
            "--input_csv",
            str(output_csv),
            "--output_dir",
            str(plot_dir),
            "--formats",
            args.plot_formats,
        ]
        print(f"[grid] generating plots into {plot_dir}", flush=True)
        proc = subprocess.run(plot_cmd, cwd=REPO_ROOT, text=True, check=False)
        if proc.returncode != 0:
            print(f"[grid] plot generation failed with returncode={proc.returncode}", flush=True)


if __name__ == "__main__":
    main()
