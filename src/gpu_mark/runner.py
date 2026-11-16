"""Runs the per-GPU pass and the all-GPU pass and assembles the result document."""

from __future__ import annotations

import datetime as dt
import os
import threading
import time
import uuid as uuidlib

import torch

from . import __version__, bench, inventory, metrics, telemetry
from .params import NOT_IMPLEMENTED, MethodParams

SCHEMA_VERSION = "1"


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


class Phases:
    """Wall-clock start and end of each benchmark phase, for joining telemetry."""

    def __init__(self):
        self.items: list[dict] = []

    def run(self, name: str, gpus: list[int], fn):
        start = time.time()
        try:
            return fn()
        finally:
            self.items.append({"name": name, "gpus": gpus, "start": start, "end": time.time()})


def _precisions(device: torch.device) -> tuple[list[str], dict]:
    precisions = ["fp32", "fp16", "bf16"]
    skipped = {}
    if bench.fp8_supported(device):
        precisions.append("fp8")
    else:
        cap = torch.cuda.get_device_capability(device) if device.type == "cuda" else None
        skipped["matmul_fp8_tflops"] = (
            f"not supported: compute capability {cap} is below {bench.FP8_MIN_CAPABILITY}"
            if cap else "not supported on this device type"
        )
    return precisions, skipped


def _matmul_series(p: MethodParams, precision: str, device: torch.device) -> list[float]:
    n = p.value("matmul_n")
    mm = bench.Matmul(precision, n, device, seed=0)
    try:
        return [
            metrics.tflops(n, p.value("matmul_timed_iters"),
                           mm.run(p.value("matmul_warmup_iters"), p.value("matmul_timed_iters")))
            for _ in range(p.value("repetitions"))
        ]
    finally:
        mm.free()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def _d2d_series(p: MethodParams, device: torch.device) -> dict:
    nbytes, iters = p.value("d2d_copy_bytes"), p.value("d2d_timed_iters")
    cp = bench.DeviceCopy(nbytes, device)
    rw, once = [], []
    try:
        for _ in range(p.value("repetitions")):
            bw = metrics.d2d_bandwidths(nbytes, iters, cp.run(p.value("d2d_warmup_iters"), iters), p.value("gb_unit"))
            rw.append(bw["read_plus_write"])
            once.append(bw["each_byte_once"])
    finally:
        cp.free()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return {"d2d_copy_read_plus_write_gbps": rw, "d2d_copy_each_byte_once_gbps": once}


def _host_series(p: MethodParams, device: torch.device) -> dict:
    nbytes, iters = p.value("host_copy_bytes"), p.value("host_timed_iters")
    hc = bench.HostCopy(nbytes, device)
    out = {"h2d_pinned_gbps": [], "d2h_pinned_gbps": []}
    try:
        for _ in range(p.value("repetitions")):
            for direction in ("h2d", "d2h"):
                s = hc.run(direction, p.value("host_warmup_iters"), iters)
                out[f"{direction}_pinned_gbps"].append(metrics.bandwidth_gbps(nbytes, iters, s, p.value("gb_unit")))
    finally:
        hc.free()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return out


UNITS = {
    "matmul_fp32_tflops": "TFLOPS",
    "matmul_fp16_tflops": "TFLOPS",
    "matmul_bf16_tflops": "TFLOPS",
    "matmul_fp8_tflops": "TFLOPS",
    "d2d_copy_read_plus_write_gbps": "GB/s (bytes read + bytes written)",
    "d2d_copy_each_byte_once_gbps": "GB/s (copy size counted once)",
    "h2d_pinned_gbps": "GB/s",
    "d2h_pinned_gbps": "GB/s",
}


def _metric(name: str, samples: list[float]) -> dict:
    return {"unit": UNITS[name], **metrics.summarize(samples)}


def per_gpu_pass(p: MethodParams, device: torch.device, phases: Phases, idx: int) -> dict:
    precisions, skipped = _precisions(device)
    out: dict = {"metrics": {}, "skipped": skipped}
    for prec in precisions:
        name = f"matmul_{prec}_tflops"
        out["metrics"][name] = _metric(name, phases.run(f"gpu{idx}:{name}", [idx],
                                                         lambda: _matmul_series(p, prec, device)))
    for name, s in phases.run(f"gpu{idx}:d2d_copy", [idx], lambda: _d2d_series(p, device)).items():
        out["metrics"][name] = _metric(name, s)
    for name, s in phases.run(f"gpu{idx}:host_copy", [idx], lambda: _host_series(p, device)).items():
        out["metrics"][name] = _metric(name, s)
    return out


def _concurrent(devices: list[torch.device], work) -> list:
    """Run work(device) on every device at once, one thread each; return results in device order."""
    barrier = threading.Barrier(len(devices))
    results: list = [None] * len(devices)
    errors: list = [None] * len(devices)

    def target(i, d):
        try:
            with torch.cuda.device(d):
                results[i] = work(d, barrier)
        except BaseException as e:  # re-raised below on the main thread
            errors[i] = e
            barrier.abort()

    threads = [threading.Thread(target=target, args=(i, d)) for i, d in enumerate(devices)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for e in errors:
        if e is not None:
            raise e
    return results


def all_gpu_pass(p: MethodParams, devices: list[torch.device], phases: Phases) -> dict:
    """Same matmul and copy on every GPU at the same time.

    Each GPU times its own batch with its own CUDA events. The aggregate for
    a repetition is the sum of the per-GPU throughputs; the batches start
    together at a barrier, so the sum is the machine's throughput under shared
    power, cooling and host load.
    """
    idxs = [d.index for d in devices]
    precisions, skipped = _precisions(devices[0])
    out: dict = {"gpus": idxs, "metrics": {}, "per_gpu_medians": {}, "skipped": skipped}
    n = p.value("matmul_n")
    reps = p.value("repetitions")

    for prec in precisions:
        name = f"matmul_{prec}_tflops"

        def work(d, barrier, prec=prec):
            mm = bench.Matmul(prec, n, d, seed=0)
            series = []
            for _ in range(reps):
                mm.run(p.value("matmul_warmup_iters"), 0)
                barrier.wait()
                s = mm.run(0, p.value("matmul_timed_iters"))
                series.append(metrics.tflops(n, p.value("matmul_timed_iters"), s))
            mm.free()
            torch.cuda.empty_cache()
            return series

        per = phases.run(f"all:{name}", idxs, lambda: _concurrent(devices, work))
        out["metrics"][name] = _metric(name, [sum(r) for r in zip(*per)])
        out["per_gpu_medians"][name] = [metrics.summarize(s)["median"] for s in per]

    nbytes, iters = p.value("d2d_copy_bytes"), p.value("d2d_timed_iters")

    def copy_work(d, barrier):
        cp = bench.DeviceCopy(nbytes, d)
        series = []
        for _ in range(reps):
            cp.run(p.value("d2d_warmup_iters"), 0)
            barrier.wait()
            series.append(metrics.d2d_bandwidths(nbytes, iters, cp.run(0, iters), p.value("gb_unit")))
        cp.free()
        torch.cuda.empty_cache()
        return series

    per = phases.run("all:d2d_copy", idxs, lambda: _concurrent(devices, copy_work))
    for key, name in (("read_plus_write", "d2d_copy_read_plus_write_gbps"),
                      ("each_byte_once", "d2d_copy_each_byte_once_gbps")):
        out["metrics"][name] = _metric(name, [sum(x[key] for x in rep) for rep in zip(*per)])
        out["per_gpu_medians"][name] = [metrics.summarize([x[key] for x in s])["median"] for s in per]
    return out


def _match_smi_index(torch_indices: list[int], gpus_smi: list[dict]) -> dict:
    """Map each torch device index to its nvidia-smi row.

    Matches by UUID when torch exposes it. CUDA_VISIBLE_DEVICES can renumber
    devices, so an index match is used only when that variable is unset.
    """
    by_uuid = {g["uuid"]: g for g in gpus_smi}
    out = {}
    for i in torch_indices:
        props = torch.cuda.get_device_properties(i)
        u = getattr(props, "uuid", None)
        key = f"GPU-{u}" if u is not None else None
        if key in by_uuid:
            out[i] = {"index": int(by_uuid[key]["index"]), "uuid": key, "match": "uuid"}
        elif os.environ.get("CUDA_VISIBLE_DEVICES") is None and i < len(gpus_smi):
            out[i] = {"index": int(gpus_smi[i]["index"]), "uuid": gpus_smi[i]["uuid"], "match": "pci_order"}
        else:
            raise RuntimeError(f"cannot match torch device {i} to an nvidia-smi GPU "
                               f"(torch uuid {u!r}, CUDA_VISIBLE_DEVICES set)")
    return out


def run(provider: str, p: MethodParams, gpu_indices: list[int] | None = None) -> dict:
    if not provider:
        raise ValueError("provider is required")
    order = os.environ.get("CUDA_DEVICE_ORDER")
    if order != "PCI_BUS_ID":
        # nvidia-smi numbers GPUs by PCI bus; CUDA must match or UUIDs attach to the wrong GPU.
        raise RuntimeError(f"CUDA_DEVICE_ORDER must be PCI_BUS_ID before torch initialises CUDA, got {order!r}")
    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is False: no usable CUDA device")

    started = time.time()
    doc: dict = {
        "schema_version": SCHEMA_VERSION,
        "tool": {"name": "gpu-mark", "version": __version__},
        "run": {"run_id": str(uuidlib.uuid4()), "started_at": _now(), "provider": provider,
                "cuda_device_order": order},
        "method": {"parameters": p.as_dict(), "fp32_math_mode": bench.configure_fp32(p.value("fp32_tf32_allowed"))},
        "not_implemented": NOT_IMPLEMENTED,
        "errors": [],
    }

    gpus_smi = inventory.nvidia_smi_gpus()
    doc["inventory"] = {
        "host": inventory.host_inventory(provider),
        "driver": inventory.nvidia_smi_driver(),
        "software": inventory.torch_versions(),
        "gpus": gpus_smi,
        "compute_processes_at_start": inventory.compute_processes(),
    }

    count = torch.cuda.device_count()
    indices = gpu_indices if gpu_indices is not None else list(range(count))
    devices = [torch.device("cuda", i) for i in indices]
    torch_names = {i: torch.cuda.get_device_name(i) for i in indices}
    caps = {i: list(torch.cuda.get_device_capability(i)) for i in indices}
    smi_index = _match_smi_index(indices, gpus_smi)
    doc["run"]["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES")

    phases = Phases()
    with telemetry.Sampler(p.value("telemetry_interval_ms")) as sampler:
        per_gpu = []
        for d in devices:
            smi = smi_index[d.index]
            entry = {"index": d.index, "nvidia_smi_index": smi["index"], "uuid": smi["uuid"],
                     "match": smi["match"], "torch_name": torch_names[d.index],
                     "compute_capability": caps[d.index]}
            with torch.cuda.device(d):
                entry.update(per_gpu_pass(p, d, phases, d.index))
            per_gpu.append(entry)

        if len(devices) >= p.value("all_gpu_pass_min_gpus"):
            all_gpu = {"status": "ran", **all_gpu_pass(p, devices, phases)}
        else:
            all_gpu = {"status": "skipped",
                       "reason": f"{len(devices)} GPU(s) below all_gpu_pass_min_gpus={p.value('all_gpu_pass_min_gpus')}"}

    doc["results"] = {"per_gpu": per_gpu, "all_gpu": all_gpu}
    doc["telemetry"] = {
        "source": "nvidia-smi",
        "interval_ms": p.value("telemetry_interval_ms"),
        "phases": [
            {**ph, "summary": {str(i): telemetry.summarize_phase(
                sampler.samples, smi_index[i]["index"], ph["start"], ph["end"]) for i in ph["gpus"]}}
            for ph in phases.items
        ],
        "samples": sampler.samples,
        "sampler_errors": sampler.errors,
    }
    ended = time.time()
    doc["run"]["ended_at"] = _now()
    doc["run"]["duration_s"] = round(ended - started, 3)
    return doc
