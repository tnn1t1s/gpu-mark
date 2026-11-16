"""GPU telemetry sampled with nvidia-smi while the benchmark runs.

nvidia-smi ships with the NVIDIA driver, so this needs no Python binding.
Samples carry a host wall-clock timestamp; the runner records phase start
and end times on the same clock so samples can be attributed to phases.
"""

from __future__ import annotations

import subprocess
import threading
import time

FIELDS = [
    ("index", "index", int),
    ("power_w", "power.draw", float),
    ("power_limit_w", "power.limit", float),
    ("temp_c", "temperature.gpu", float),
    ("sm_clock_mhz", "clocks.sm", float),
    ("mem_clock_mhz", "clocks.mem", float),
    ("util_pct", "utilization.gpu", float),
    ("throttle_reasons", "clocks_throttle_reasons.active", str),
]


def _num(conv, raw: str):
    raw = raw.strip()
    if raw in ("[N/A]", "N/A", "[Not Supported]", ""):
        return None
    return conv(raw)


def parse_line(line: str) -> dict | None:
    """Parse one CSV line from `nvidia-smi --format=csv,noheader,nounits`."""
    parts = [p.strip() for p in line.split(",")]
    if len(parts) != len(FIELDS):
        return None
    try:
        return {name: _num(conv, raw) for (name, _, conv), raw in zip(FIELDS, parts)}
    except ValueError:
        return None


def query_args() -> list[str]:
    return [
        "nvidia-smi",
        "--query-gpu=" + ",".join(q for _, q, _ in FIELDS),
        "--format=csv,noheader,nounits",
    ]


class Sampler:
    """Polls nvidia-smi on a background thread every `interval_ms`."""

    def __init__(self, interval_ms: int):
        self.interval_s = interval_ms / 1000.0
        self.samples: list[dict] = []
        self.errors: list[str] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self):
        args = query_args()
        while not self._stop.is_set():
            t = time.time()
            try:
                out = subprocess.run(args, capture_output=True, text=True, timeout=5, check=True).stdout
                for line in out.strip().splitlines():
                    row = parse_line(line)
                    if row is not None:
                        row["t"] = t
                        self.samples.append(row)
            except Exception as e:  # recorded, not swallowed: written to the result file
                if len(self.errors) < 20:
                    self.errors.append(f"{type(e).__name__}: {e}")
            self._stop.wait(max(0.0, self.interval_s - (time.time() - t)))

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=10)


def summarize_phase(samples: list[dict], gpu_index: int, start: float, end: float) -> dict:
    """Summary of one GPU's samples taken between start and end (host epoch seconds)."""
    if not isinstance(gpu_index, int):
        # Samples carry an int index; a str index would silently match nothing.
        raise TypeError(f"gpu_index must be int, got {type(gpu_index).__name__}")
    rows = [s for s in samples if s["index"] == gpu_index and start <= s["t"] <= end]

    def vals(k):
        return [r[k] for r in rows if r.get(k) is not None]

    def stat(k, fn):
        v = vals(k)
        return fn(v) if v else None

    throttles = sorted({r["throttle_reasons"] for r in rows if r.get("throttle_reasons")})
    return {
        "n_samples": len(rows),
        "power_w_mean": stat("power_w", lambda v: sum(v) / len(v)),
        "power_w_max": stat("power_w", max),
        "power_limit_w": stat("power_limit_w", max),
        "temp_c_max": stat("temp_c", max),
        "sm_clock_mhz_min": stat("sm_clock_mhz", min),
        "sm_clock_mhz_max": stat("sm_clock_mhz", max),
        "mem_clock_mhz_min": stat("mem_clock_mhz", min),
        "throttle_reasons_seen": throttles,
    }
