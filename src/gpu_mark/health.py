"""Checks a rented GPU can fail before any benchmark runs.

Each check compares one value nvidia-smi or the kernel log reports against a
condition, and returns a finding with status ``ok``, ``flag``, ``unavailable``
(the value could not be read; the reason says why) or ``not_checked`` (the
value is recorded but this check does not judge it; the reason says why).
A flag is a reason to look at the machine before using it. gpu-mark never
refuses to run on a flagged GPU.

Every condition below is an invariant of a healthy, whole, unshared GPU, not a
tuned threshold: zero is the only acceptable count of uncorrected memory errors.
"""

from __future__ import annotations

import subprocess

# nvidia-smi fields read only for health. Kept apart from the inventory query,
# so a driver that lacks one of them costs the health checks, not the run.
HEALTH_FIELDS = [
    ("index", "index"),
    ("uuid", "uuid"),
    ("ecc_uncorrected_volatile", "ecc.errors.uncorrected.volatile.total"),
    ("ecc_uncorrected_aggregate", "ecc.errors.uncorrected.aggregate.total"),
    ("ecc_corrected_volatile", "ecc.errors.corrected.volatile.total"),
    ("retired_pages_pending", "retired_pages.pending"),
    ("remapped_rows_pending", "remapped_rows.pending"),
    ("remapped_rows_failure", "remapped_rows.failure"),
    ("throttle_reasons_at_start", "clocks_throttle_reasons.active"),
]

# Throttle reason bits that mean hardware protection, not policy, from NVML's
# nvmlClocksThrottleReasons. Seen on an idle GPU they point at a cooling or
# power-delivery fault. 0x1 (GPU idle) and 0x4 (software power cap) are normal.
HW_THROTTLE_BITS = {
    0x8: "hardware slowdown",
    0x40: "hardware thermal slowdown",
    0x80: "hardware power brake slowdown",
}

_NA = ("[N/A]", "N/A", "[Not Supported]", "")


def _run(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(args, capture_output=True, text=True, timeout=30)


def parse_health_csv(text: str) -> list[dict]:
    rows = []
    for line in text.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != len(HEALTH_FIELDS):
            continue
        rows.append({k: (None if v in _NA else v) for (k, _), v in zip(HEALTH_FIELDS, parts)})
    return rows


def read_health() -> dict:
    """Health fields per GPU, or the reason they could not be read."""
    r = _run(["nvidia-smi", "--query-gpu=" + ",".join(q for _, q in HEALTH_FIELDS),
              "--format=csv,noheader,nounits"])
    if r.returncode != 0:
        return {"rows": [], "error": f"nvidia-smi exited {r.returncode}: {(r.stderr or r.stdout).strip()[:300]}"}
    return {"rows": parse_health_csv(r.stdout), "error": None}


def read_xid_lines() -> dict:
    """NVIDIA Xid lines from the kernel log since boot.

    A container usually cannot read the kernel log; the result then says so
    instead of reporting zero Xid events.
    """
    try:
        r = _run(["dmesg"])
    except FileNotFoundError:
        return {"lines": None, "error": "no dmesg command on this host"}
    if r.returncode != 0:
        return {"lines": None, "error": f"dmesg exited {r.returncode}: {r.stderr.strip()[:200]}"}
    return {"lines": [ln for ln in r.stdout.splitlines() if "NVRM: Xid" in ln], "error": None}


def _finding(check: str, uuid: str | None, status: str, observed, expected: str, reason: str) -> dict:
    return {"check": check, "gpu_uuid": uuid, "status": status,
            "observed": observed, "expected": expected, "reason": reason}


def _int(v):
    return int(v) if v is not None else None


def check_gpu(gpu: dict, health_row: dict | None, procs: list[dict]) -> list[dict]:
    """Findings for one GPU from its inventory entry, health row and compute processes."""
    u = gpu.get("uuid")
    out = []

    # Link generation and width both drop at idle to save power: a DGX Spark
    # (GB10, driver 580.173.02) reported gen 1 of 1 and width x1 of x16 while
    # idle. Judging either needs a reading taken during a host transfer.
    for check, cur, mx in (("pcie_gen", "pcie_gen_current", "pcie_gen_max"),
                           ("pcie_width", "pcie_width_current", "pcie_width_max")):
        out.append(_finding(check, u, "not_checked", {"current": gpu.get(cur), "max": gpu.get(mx)},
                            "current == max during a host transfer",
                            "the link drops speed and width at idle; judging it needs a reading taken "
                            "during a transfer"))

    pl, pd = gpu.get("power_limit_w"), gpu.get("power_default_limit_w")
    if pl is None or pd is None:
        out.append(_finding("power_limit", u, "unavailable", {"limit_w": pl, "default_w": pd}, "limit >= default",
                            "nvidia-smi reported no power limit"))
    else:
        ok = float(pl) >= float(pd)
        out.append(_finding("power_limit", u, "ok" if ok else "flag", {"limit_w": float(pl), "default_w": float(pd)},
                            "limit >= default", "a lowered power limit caps clocks under load"))

    ecc = gpu.get("ecc_mode")
    if ecc is None:
        out.append(_finding("ecc_mode", u, "unavailable", None, "Enabled", "this GPU reports no ECC mode"))
    else:
        out.append(_finding("ecc_mode", u, "ok" if ecc == "Enabled" else "flag", ecc, "Enabled",
                            "with ECC off, memory errors go undetected"))

    mig = gpu.get("mig_mode")
    out.append(_finding("mig_mode", u, "flag" if mig == "Enabled" else "ok", mig, "Disabled or not supported",
                        "a MIG instance is a slice of the GPU; its results don't describe the whole card"))

    mine = [p for p in procs if p.get("gpu_uuid") == u]
    out.append(_finding("other_compute_processes", u, "flag" if mine else "ok",
                        [{"pid": p["pid"], "process_name": p["process_name"]} for p in mine], "none",
                        "another process on the GPU shares its compute and memory during the run"))

    if health_row is None:
        out.append(_finding("memory_errors", u, "unavailable", None, "no uncorrected errors, no pending or failed "
                            "row remaps or page retirements", "the health query returned no row for this GPU"))
        out.append(_finding("hw_throttle_at_start", u, "unavailable", None, "no hardware throttle reason",
                            "the health query returned no row for this GPU"))
        return out

    counts = {k: _int(health_row.get(k)) for k in
              ("ecc_uncorrected_volatile", "ecc_uncorrected_aggregate", "retired_pages_pending")}
    remap_pending = _int(health_row.get("remapped_rows_pending"))
    remap_failure = health_row.get("remapped_rows_failure")
    observed = {**counts, "ecc_corrected_volatile": _int(health_row.get("ecc_corrected_volatile")),
                "remapped_rows_pending": remap_pending, "remapped_rows_failure": remap_failure}
    bad = [k for k, v in counts.items() if v]
    if remap_pending:
        bad.append("remapped_rows_pending")
    if remap_failure not in (None, "No", "0"):
        bad.append("remapped_rows_failure")
    readable = [v for v in (*counts.values(), remap_pending) if v is not None]
    if bad:
        status = "flag"
    elif readable:
        status = "ok"
    else:
        status = "unavailable"
    out.append(_finding("memory_errors", u, status, observed,
                        "no uncorrected errors, no pending or failed row remaps or page retirements",
                        "uncorrected errors and failed remaps mean damaged GPU memory; pending ones clear only "
                        f"after a GPU reset{'; nonzero: ' + ', '.join(bad) if bad else ''}"))

    tr = health_row.get("throttle_reasons_at_start")
    if tr is None:
        out.append(_finding("hw_throttle_at_start", u, "unavailable", None, "no hardware throttle reason",
                            "nvidia-smi reported no throttle reasons"))
    else:
        bits = int(tr, 16)
        hit = [name for bit, name in HW_THROTTLE_BITS.items() if bits & bit]
        out.append(_finding("hw_throttle_at_start", u, "flag" if hit else "ok", {"bits": tr, "hardware": hit},
                            "no hardware throttle reason",
                            "hardware throttling before any load points at cooling or power delivery"))
    return out


def check_xid(xid: dict) -> dict:
    if xid["lines"] is None:
        return _finding("xid_errors", None, "unavailable", None, "no Xid lines since boot",
                        f"the kernel log is not readable here ({xid['error']})")
    return _finding("xid_errors", None, "flag" if xid["lines"] else "ok", xid["lines"][-20:],
                    "no Xid lines since boot", "Xid lines are the driver's record of GPU faults")


def evaluate(gpus: list[dict], procs: list[dict], health: dict, xid: dict) -> dict:
    by_uuid = {r.get("uuid"): r for r in health["rows"]}
    findings = []
    for g in gpus:
        findings.extend(check_gpu(g, by_uuid.get(g.get("uuid")), procs))
    findings.append(check_xid(xid))
    return {
        "flagged": sorted({f["check"] for f in findings if f["status"] == "flag"}),
        "findings": findings,
        "raw": {"health_rows": health["rows"], "health_error": health["error"], "xid_error": xid["error"]},
    }


def snapshot(gpus: list[dict], procs: list[dict]) -> dict:
    """Read the health fields and the kernel log, then evaluate every check."""
    return evaluate(gpus, procs, read_health(), read_xid_lines())


def format_findings(h: dict) -> str:
    lines = [f"health: {'flagged ' + ', '.join(h['flagged']) if h['flagged'] else 'no flags'}"]
    for f in h["findings"]:
        if f["status"] != "ok":
            who = f" {f['gpu_uuid']}" if f["gpu_uuid"] else ""
            lines.append(f"  {f['status']:12s} {f['check']}{who}: observed {f['observed']!r}; "
                         f"expected {f['expected']}. {f['reason']}")
    return "\n".join(lines)
