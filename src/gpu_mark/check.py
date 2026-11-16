"""`gpu-mark check`: inventory and health checks without any benchmark.

Needs only nvidia-smi, so it runs in seconds on a machine before vLLM or torch
is installed.
"""

from __future__ import annotations

import os
import uuid as uuidlib
from datetime import datetime, timezone

from . import __version__, health, inventory

SCHEMA_VERSION = "1"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run(provider: str, rental: dict) -> dict:
    gpus = inventory.nvidia_smi_gpus()
    procs = inventory.compute_processes()
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "check",
        "tool": {"name": "gpu-mark", "version": __version__},
        "run": {"run_id": str(uuidlib.uuid4()), "started_at": _now(), "provider": provider,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")},
        "rental": rental,
        "inventory": {
            "host": inventory.host_inventory(provider),
            "driver": inventory.nvidia_smi_driver(),
            "gpus": gpus,
            "compute_processes_at_start": procs,
        },
        "health": health.snapshot(gpus, procs),
    }


def format_summary(doc: dict) -> str:
    drv = doc["inventory"]["driver"]
    lines = [f"gpu-mark {doc['tool']['version']} check  provider={doc['run']['provider']}  "
             f"driver {drv['driver_version']}"]
    for g in doc["inventory"]["gpus"]:
        lines.append(f"GPU {g['index']}  {g['name']}  {g['uuid']}")
    missing = doc["rental"]["not_given"]
    if missing:
        lines.append(f"rental fields not given: {', '.join(missing)}")
    lines.append(health.format_findings(doc["health"]))
    return "\n".join(lines)
