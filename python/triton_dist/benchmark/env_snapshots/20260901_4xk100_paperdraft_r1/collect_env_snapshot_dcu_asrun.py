"""Collect a DCU (Hygon / DTK) machine + environment snapshot.

Companion of ``collect_env_snapshot.py`` (NVIDIA side). Same discipline:
read-only, no benchmark, no repo mutation. Run ONCE, in the DCU python
environment, before any experiment. Writes a structured
``env_snapshot_dcu.json`` plus one raw ``*.txt`` per probed command into
``--output_dir``. Every probe fails gracefully and records stderr, so the
script keeps working across DTK versions with different tool names/paths.

This version was written against a real Hygon DTK box (not guessed), so the
probes point at the paths that actually exist on DTK:
  * SMI:      hy-smi (/opt/hyhal/bin), rocm-smi / rocminfo (/opt/dtk/bin)
  * topology: ``--showtopo`` (``hy-smi topo`` is NOT a valid subcommand)
  * DTK ver:  /opt/dtk/.info/{rocm_version,version-dev,version-libs,version-utils}
  * RCCL:     /opt/dtk/lib/librccl.so*  (NOT in ldconfig cache on this box)
  * DUSHMEM:  /opt/dtk/dushmem/lib/libdushmem_host.so*  (version = soname tail)
Versions are parsed straight from the .so sonames / ``strings`` output, not
left blank.

NOTE: git / repository state is intentionally NOT captured (per instruction).

Usage (rank-0 only, in the DCU python env):
    python collect_env_snapshot_dcu.py \
        --output_dir .../env_snapshot_dcu --batch_id <batch_id>
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

# DTK install root (follow the /opt/dtk symlink to the concrete version dir).
DTK_ROOT = Path(os.environ.get("ROCM_PATH") or os.environ.get("DTK_ROOT") or "/opt/dtk")

# Directories DTK actually ships its runtime libraries in. These are NOT in the
# default ldconfig cache, so we scan them directly instead of trusting ldconfig.
DCU_LIB_DIRS = [
    DTK_ROOT / "lib",
    DTK_ROOT / "dushmem" / "lib",
    DTK_ROOT / "rccl" / "lib",
    DTK_ROOT / "hip" / "lib",
]


def _run(cmd: list[str], timeout: int = 60, env: dict | None = None) -> dict:
    """Run a command, capture output, never raise."""
    if not shutil.which(cmd[0]) and not Path(cmd[0]).exists():
        return {"cmd": " ".join(cmd), "returncode": -1, "stdout": "", "stderr": f"{cmd[0]} not on PATH"}
    try:
        run_env = {**os.environ, **env} if env else None
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=run_env)
        return {"cmd": " ".join(cmd), "returncode": p.returncode, "stdout": p.stdout[-40000:], "stderr": p.stderr[-4000:]}
    except Exception as e:  # noqa: BLE001
        return {"cmd": " ".join(cmd), "returncode": -1, "stdout": "", "stderr": f"{type(e).__name__}: {e}"}


# ---- SMI / topology / hardware probes -------------------------------------
# hy-smi lives in /opt/hyhal/bin; rocm-smi/rocminfo in /opt/dtk/bin. Topology is
# `--showtopo` on both (the bare `topo` subcommand is rejected). We keep several
# hy-smi views because each reports a different, non-overlapping slice.
SMI_CANDIDATES = [
    ("hy_smi", ["hy-smi"]),
    ("hy_smi_showtopo", ["hy-smi", "--showtopo"]),
    ("hy_smi_showhw", ["hy-smi", "--showhw"]),
    ("hy_smi_driverversion", ["hy-smi", "--showdriverversion"]),
    ("hy_smi_showvbios", ["hy-smi", "--showvbios"]),
    ("hy_smi_showfwinfo", ["hy-smi", "--showfwinfo"]),
    ("hy_smi_allinfo", ["hy-smi", "-a"]),
    ("rocm_smi", ["rocm-smi"]),
    ("rocm_smi_showtopo", ["rocm-smi", "--showtopo"]),
    ("rocm_smi_showid", ["rocm-smi", "--showid"]),
    ("rocminfo", ["rocminfo"]),
    ("lspci_display", ["bash", "-c", "lspci -nn | grep -Ei 'display|processing|co-processor|3d'"]),
]

# HIP / compiler toolchain versions.
TOOLCHAIN_CANDIDATES = [
    ("hipconfig", ["hipconfig", "--full"]),
    ("hipcc_version", ["hipcc", "--version"]),
    ("dcc_version", ["dcc", "--version"]),
    ("mpirun_version", ["mpirun", "--version"]),
]


def _dtk_versions() -> dict:
    """Read the concrete DTK version. Real markers live in <dtk>/.info/."""
    info: dict = {}
    try:
        info["dtk_root"] = str(DTK_ROOT.resolve())
    except Exception:  # noqa: BLE001
        info["dtk_root"] = str(DTK_ROOT)
    for fname in ("rocm_version", "version-dev", "version-libs", "version-utils"):
        f = DTK_ROOT / ".info" / fname
        try:
            if f.is_file():
                info[fname] = f.read_text(encoding="utf-8", errors="replace").strip()
        except Exception as e:  # noqa: BLE001
            info[fname] = f"<read failed: {e}>"
    if not any(k for k in info if k != "dtk_root"):
        info["note"] = "no version markers under <dtk>/.info"
    return info


def _resolve_soname(so: Path) -> str:
    """Follow a .so symlink chain to its concrete file, returning the tail
    version (e.g. libdushmem_host.so.3.2.5 -> '3.2.5')."""
    try:
        target = so.resolve()
        m = re.search(r"\.so\.([0-9][0-9.]*)$", target.name)
        return m.group(1) if m else target.name
    except Exception:  # noqa: BLE001
        return "<unresolved>"


def _scan_dcu_libraries() -> dict:
    """Find rccl / dushmem / hipblas(lt) / miopen shared objects by scanning the
    DTK lib dirs directly. ldconfig does NOT index these on a DTK box."""
    patterns = {
        "librccl": "librccl.so*",
        "libdushmem_host": "libdushmem_host.so*",
        "libhipblas": "libhipblas.so*",
        "libhipblaslt": "libhipblaslt.so*",
        "libMIOpen": "libMIOpen*.so*",
    }
    found: dict = {}
    for tag, pat in patterns.items():
        hits: list[str] = []
        for d in DCU_LIB_DIRS:
            if not d.is_dir():
                continue
            for p in sorted(d.glob(pat)):
                # Prefer the real file (skip pure symlinks in the listing detail,
                # but still record where the soname resolves to).
                hits.append(str(p))
        if hits:
            # Pick a representative .so to resolve the version from.
            rep = next((Path(h) for h in hits if Path(h).name.endswith(tuple(f".so.{i}" for i in range(1, 9))) or ".so." in Path(h).name), Path(hits[0]))
            found[tag] = {"paths": hits, "version": _resolve_soname(rep)}
        else:
            found[tag] = {"paths": [], "version": None, "note": "not found under DTK lib dirs"}
    return found


def _rccl_version_string() -> str | None:
    """Pull the human 'RCCL version : X.Y.Z' line out of librccl via strings."""
    for d in (DTK_ROOT / "lib", DTK_ROOT / "rccl" / "lib"):
        for so in sorted(d.glob("librccl.so.*")) if d.is_dir() else []:
            if so.is_file() and not so.is_symlink():
                r = _run(["bash", "-c", f"strings {so} | grep -Ei 'RCCL version' | head -1"])
                m = re.search(r"([0-9]+\.[0-9]+\.[0-9]+)", r["stdout"])
                if m:
                    return m.group(1)
    return None


def _dushmem_details() -> dict:
    """DUSHMEM bootstrap/transport plugins present + the info tool path."""
    d = DTK_ROOT / "dushmem" / "lib"
    plugins = sorted(p.name for p in d.glob("dushmem_*.so.*")) if d.is_dir() else []
    info_tool = DTK_ROOT / "dushmem" / "bin" / "dushmem-info"
    return {
        "lib_dir": str(d),
        "device_lib": str(d / "libdushmem_device.a") if (d / "libdushmem_device.a").exists() else None,
        "bootstrap_transport_plugins": plugins,
        "info_tool": str(info_tool) if info_tool.exists() else None,
    }


def _relevant_environment() -> dict[str, str]:
    keys = ("DTK", "ROCM", "ROCR", "HIP", "HSA", "CUDA_VISIBLE", "DCU", "DUSHMEM",
            "NVSHMEM", "RCCL", "NCCL", "HYGO", "GPU_", "PATH", "LD_LIBRARY_PATH")
    return {k: v for k, v in sorted(os.environ.items()) if any(k2 in k.upper() for k2 in keys)}


def _python_stack() -> dict:
    info: dict = {"python": sys.version.split()[0], "executable": sys.executable}
    try:
        import torch

        info["torch"] = torch.__version__
        info["torch_cuda"] = torch.version.cuda  # None on HIP builds
        info["torch_hip"] = torch.version.hip
        try:
            info["cuda_available"] = torch.cuda.is_available()
            info["device_count"] = torch.cuda.device_count()
            devs = []
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                devs.append({
                    "index": i,
                    "name": props.name,
                    "total_mem_gib": round(props.total_memory / 1024**3, 2),
                    "gcn_arch": getattr(props, "gcnArchName", None),
                    "multi_processor_count": getattr(props, "multi_processor_count", None),
                })
            info["devices"] = devs
        except Exception as e:  # noqa: BLE001
            info["device_error"] = f"{type(e).__name__}: {e}"
    except Exception as e:  # noqa: BLE001
        info["torch_error"] = f"{type(e).__name__}: {e}"
    try:
        import triton

        info["triton"] = triton.__version__
    except Exception as e:  # noqa: BLE001
        info["triton_error"] = f"{type(e).__name__}: {e}"

    r = _run([sys.executable, "-m", "pip", "list"])
    if r["returncode"] == 0:
        wanted = ("torch", "triton", "rccl", "dushmem", "pytorch-triton",
                  "numpy", "hip", "flash", "apex", "transformer")
        sel = [ln for ln in r["stdout"].splitlines()
               if any(w in ln.lower() for w in wanted)]
        info["pip_selected"] = "\n".join(sel)
    else:
        info["pip_error"] = r["stderr"][:2000]
    return info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", type=Path, required=True)
    ap.add_argument("--batch_id", type=str, required=True)
    args = ap.parse_args()

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    # Raw command dumps (one txt each), for full-fidelity evidence.
    raw = {name: _run(cmd) for name, cmd in SMI_CANDIDATES}
    raw.update({name: _run(cmd) for name, cmd in TOOLCHAIN_CANDIDATES})
    raw.update({
        "lscpu": _run(["lscpu"]),
        "meminfo": _run(["cat", "/proc/meminfo"]),
        "os_release": _run(["cat", "/etc/os-release"]),
        "uname": _run(["uname", "-a"]),
        "ibstat": _run(["ibstat"]),
    })
    for name, r in raw.items():
        (out / f"{name}.txt").write_text(
            f"$ {r.get('cmd', name)}\n[returncode={r['returncode']}]\n\n"
            + r["stdout"] + "\n--- stderr ---\n" + r["stderr"],
            encoding="utf-8",
        )

    # Structured summary (what the paper platform row is filled from).
    dcu_libs = _scan_dcu_libraries()
    # librccl soname is only ...so.1.0 (SO version, not the RCCL release), so the
    # authoritative release number comes from the embedded "RCCL version" string.
    rccl_str = _rccl_version_string()
    if rccl_str:
        dcu_libs.setdefault("librccl", {})["soname_version"] = dcu_libs["librccl"].get("version")
        dcu_libs["librccl"]["version"] = rccl_str
        dcu_libs["librccl"]["version_string"] = rccl_str

    snapshot = {
        "batch_id": args.batch_id,
        "captured_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "hostname": platform.node(),
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "platform": platform.platform(),
        },
        "dtk": _dtk_versions(),
        "dcu_libraries": dcu_libs,
        "dushmem": _dushmem_details(),
        "environment_vars": _relevant_environment(),
        "python_stack": _python_stack(),
        "raw_files": sorted(p.name for p in out.glob("*.txt")),
    }
    dst = out / "env_snapshot_dcu.json"
    dst.write_text(json.dumps(snapshot, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"[env-snapshot-dcu] wrote {dst}")
    print(json.dumps(
        {k: snapshot[k] for k in ("batch_id", "captured_at", "hostname")},
        ensure_ascii=False,
    ))


if __name__ == "__main__":
    main()
