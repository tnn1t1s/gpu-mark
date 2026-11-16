"""Hardware and software inventory of the host and its GPUs.

Reads nvidia-smi, /proc and the Python runtime. Needs no credential and
makes no network call.
"""

from __future__ import annotations

import os
import platform
import socket
import subprocess

GPU_FIELDS = [
    ("index", "index"),
    ("name", "name"),
    ("uuid", "uuid"),
    ("serial", "serial"),
    ("pci_bus_id", "pci.bus_id"),
    ("vbios", "vbios_version"),
    ("memory_total_mib", "memory.total"),
    ("ecc_mode", "ecc.mode.current"),
    ("mig_mode", "mig.mode.current"),
    ("power_limit_w", "power.limit"),
    ("power_default_limit_w", "power.default_limit"),
    ("power_max_limit_w", "power.max_limit"),
    ("clocks_max_sm_mhz", "clocks.max.sm"),
    ("clocks_max_mem_mhz", "clocks.max.mem"),
    ("clocks_app_sm_mhz", "clocks.applications.graphics"),
    ("pcie_gen_current", "pcie.link.gen.current"),
    ("pcie_gen_max", "pcie.link.gen.max"),
    ("pcie_width_current", "pcie.link.width.current"),
    ("pcie_width_max", "pcie.link.width.max"),
    ("persistence_mode", "persistence_mode"),
    ("compute_mode", "compute_mode"),
    ("temp_c_at_start", "temperature.gpu"),
    ("util_pct_at_start", "utilization.gpu"),
    ("memory_used_mib_at_start", "memory.used"),
]


def parse_gpu_csv(text: str) -> list[dict]:
    gpus = []
    for line in text.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != len(GPU_FIELDS):
            continue
        gpus.append({k: (None if v in ("[N/A]", "N/A", "[Not Supported]") else v)
                     for (k, _), v in zip(GPU_FIELDS, parts)})
    return gpus


def _run(args: list[str]) -> str:
    return subprocess.run(args, capture_output=True, text=True, timeout=30, check=True).stdout


def nvidia_smi_gpus() -> list[dict]:
    out = _run(["nvidia-smi", "--query-gpu=" + ",".join(q for _, q in GPU_FIELDS),
                "--format=csv,noheader,nounits"])
    return parse_gpu_csv(out)


def nvidia_smi_driver() -> dict:
    out = _run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]).strip().splitlines()
    header = _run(["nvidia-smi"]).splitlines()
    cuda_driver = None
    for line in header[:5]:
        if "CUDA Version" in line:
            cuda_driver = line.split("CUDA Version:")[1].split("|")[0].strip()
    return {"driver_version": out[0].strip() if out else None, "cuda_driver_api_version": cuda_driver}


def compute_processes() -> list[dict]:
    """Other processes using a GPU when the run starts (concurrent load)."""
    out = _run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                "--format=csv,noheader,nounits"])
    procs = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 4:
            procs.append({"gpu_uuid": parts[0], "pid": parts[1], "process_name": parts[2], "used_memory_mib": parts[3]})
    return procs


def cpu_model() -> str | None:
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or None


def mem_total_bytes() -> int | None:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


def torch_versions() -> dict:
    import torch

    return {
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "cublas_note": "cuBLAS is the version bundled with torch_cuda",
    }


def host_inventory(provider: str) -> dict:
    return {
        "provider": provider,
        "hostname": socket.gethostname(),
        "os": platform.platform(),
        "kernel": platform.release(),
        "python": platform.python_version(),
        "cpu_model": cpu_model(),
        "cpu_count_logical": os.cpu_count(),
        "cpu_affinity_count": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "mem_total_bytes": mem_total_bytes(),
    }
