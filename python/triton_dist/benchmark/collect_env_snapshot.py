"""Collect a machine/environment snapshot for controlled-rerun provenance.

Writes one JSON plus raw text dumps (nvidia-smi, topo, lscpu, git state) into
--output_dir. Intended to run once on the target GPU machine BEFORE any
benchmark of a controlled batch, so every result directory can reference the
same snapshot. Read-only: never modifies code, data, or git state.

Usage (rank-0 only, after `source scripts/setenv.sh`):
    python python/triton_dist/benchmark/collect_env_snapshot.py \
        --output_dir .../e2e_controlled_reruns/<batch>/env_snapshot \
        --batch_id 20260901_4xa100_m3_r2
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def _run(
    cmd: list[str],
    timeout: int = 60,
    max_stdout_bytes: int | None = 20000,
    max_stderr_bytes: int = 4000,
) -> dict:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        stdout = p.stdout if max_stdout_bytes is None else p.stdout[-max_stdout_bytes:]
        return {"returncode": p.returncode, "stdout": stdout, "stderr": p.stderr[-max_stderr_bytes:]}
    except Exception as e:  # noqa: BLE001 - snapshot must never abort the batch
        return {"returncode": -1, "stdout": "", "stderr": f"{type(e).__name__}: {e}"}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relevant_environment() -> dict[str, str]:
    exact_names = {
        "CONDA_PREFIX",
        "CUDA_HOME",
        "CUDA_PATH",
        "CUDA_VISIBLE_DEVICES",
        "LD_LIBRARY_PATH",
        "LD_PRELOAD",
        "PATH",
        "PYTHONPATH",
    }
    prefixes = ("CUDA_", "NCCL_", "NVSHMEM_", "TORCH_", "TRITON_")
    result = {key: os.environ.get(key, "<unset>") for key in sorted(exact_names)}
    result.update({
        key: value
        for key, value in sorted(os.environ.items())
        if key not in exact_names and key.startswith(prefixes)
    })
    return result


def _package_versions(distributions: tuple[str, ...]) -> dict[str, str]:
    versions: dict[str, str] = {}
    for distribution in distributions:
        result = _run([sys.executable, "-m", "pip", "show", distribution])
        if result["returncode"] != 0:
            continue
        for line in result["stdout"].splitlines():
            if line.startswith("Version:"):
                versions[distribution] = line.partition(":")[2].strip()
                break
    return versions


def _git_state() -> dict:
    def g(*args: str) -> str:
        r = _run(["git", "-C", str(REPO_ROOT), *args])
        return r["stdout"].strip() if r["returncode"] == 0 else f"<git {args[0]} failed: {r['stderr'][:200]}>"

    diff = _run(
        ["git", "-C", str(REPO_ROOT), "diff", "--no-ext-diff", "--binary"],
        max_stdout_bytes=None,
    )
    return {
        "commit": g("rev-parse", "HEAD"),
        "branch": g("rev-parse", "--abbrev-ref", "HEAD"),
        "describe": g("describe", "--always", "--dirty"),
        "status_short": g("status", "--short")[:20000],
        "tracked_diff_sha256": hashlib.sha256(diff["stdout"].encode()).hexdigest(),
        "tracked_diff_bytes": len(diff["stdout"].encode()),
        "tracked_diff_returncode": diff["returncode"],
    }


def _python_stack() -> dict:
    info: dict = {"python": sys.version.split()[0], "executable": sys.executable}
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_runtime"] = torch.version.cuda
        info["nccl"] = torch.cuda.nccl.version() if torch.cuda.is_available() else None
        info["cudnn"] = torch.backends.cudnn.version() if torch.cuda.is_available() else None
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            info["gpu_name"] = props.name
            info["gpu_count"] = torch.cuda.device_count()
            info["gpu_total_mem_gib"] = round(props.total_memory / 1024**3, 2)
    except Exception as e:  # noqa: BLE001
        info["torch_error"] = f"{type(e).__name__}: {e}"
    try:
        import triton

        info["triton"] = triton.__version__
    except Exception as e:  # noqa: BLE001
        info["triton_error"] = f"{type(e).__name__}: {e}"
    try:
        import triton_dist

        info["triton_dist"] = getattr(triton_dist, "__version__", "unknown")
    except Exception as e:  # noqa: BLE001
        info["triton_dist_error"] = f"{type(e).__name__}: {e}"
    info["pip_distribution_versions"] = _package_versions(
        ("torch", "triton", "nvidia-nvshmem-cu12", "nvshmem4py-cu12", "triton-dist")
    )
    try:
        p = subprocess.run(
            [sys.executable, "-c", "import nvidia.nvshmem, pathlib; print(pathlib.Path(nvidia.nvshmem.__path__[0]))"],
            capture_output=True, text=True, timeout=60)
        info["nvshmem_wheel_path"] = p.stdout.strip() if p.returncode == 0 else None
    except Exception:  # noqa: BLE001
        info["nvshmem_wheel_path"] = None
    ld = _run(["ldconfig", "-p"])
    for line in ld["stdout"].splitlines():
        if "libnvshmem_host" in line:
            info["libnvshmem_host"] = line.strip()
            break
    return info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", type=Path, required=True)
    ap.add_argument("--batch_id", type=str, required=True)
    args = ap.parse_args()

    out = args.output_dir
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"refusing to overwrite nonempty snapshot directory: {out}")
    out.mkdir(parents=True, exist_ok=True)

    raw = {
        "nvidia_smi": _run(["nvidia-smi", "-q"], max_stdout_bytes=None),
        "nvidia_smi_topo": _run(["nvidia-smi", "topo", "-m"]),
        "nvidia_smi_short": _run(["nvidia-smi"]),
        "lscpu": _run(["lscpu"]),
        "meminfo": _run(["cat", "/proc/meminfo"]),
        "os_release": _run(["cat", "/etc/os-release"]),
        "uname": _run(["uname", "-a"]),
        "nvcc": _run(["nvcc", "--version"]) if shutil.which("nvcc") else {"returncode": -1, "stdout": "", "stderr": "nvcc not on PATH"},
        "ibstat": _run(["ibstat"]) if shutil.which("ibstat") else {"returncode": -1, "stdout": "", "stderr": "ibstat not on PATH"},
        "pip_selected": _run([sys.executable, "-m", "pip", "show", "torch", "triton", "nvidia-nvshmem-cu12", "nvshmem4py-cu12", "triton-dist"]),
    }
    for name, r in raw.items():
        (out / f"{name}.txt").write_text(r["stdout"] + "\n--- stderr ---\n" + r["stderr"], encoding="utf-8")

    snapshot = {
        "batch_id": args.batch_id,
        "captured_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "hostname": platform.node(),
        "os": {"system": platform.system(), "release": platform.release(), "machine": platform.machine()},
        "kernel_cmdline": _run(["cat", "/proc/cmdline"])["stdout"].strip(),
        "invocation": [sys.executable, *sys.argv],
        "collector": {"path": str(Path(__file__).resolve()), "sha256": _sha256_file(Path(__file__).resolve())},
        "environment": _relevant_environment(),
        "python_stack": _python_stack(),
        "git": _git_state(),
        "raw_files": sorted(p.name for p in out.glob("*.txt")),
    }
    (out / "env_snapshot.json").write_text(json.dumps(snapshot, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"[env-snapshot] wrote {out / 'env_snapshot.json'}")
    print(json.dumps({k: snapshot[k] for k in ("batch_id", "captured_at", "hostname")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
