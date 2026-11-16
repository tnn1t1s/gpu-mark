"""Pure arithmetic: FLOP counts, bandwidth and summary statistics.

Nothing here imports torch, so the tests can check every formula without a GPU.
"""

from __future__ import annotations

import statistics


def matmul_flops(n: int) -> int:
    """FLOPs in one n x n by n x n matrix multiply: n^3 multiply-adds, 2 FLOPs each."""
    return 2 * n**3


def tflops(n: int, iters: int, elapsed_s: float) -> float:
    if elapsed_s <= 0:
        raise ValueError(f"elapsed time must be positive, got {elapsed_s}")
    return matmul_flops(n) * iters / elapsed_s / 1e12


def bandwidth_gbps(bytes_moved: int, iters: int, elapsed_s: float, gb: int) -> float:
    """Bytes moved per second in units of `gb` bytes."""
    if elapsed_s <= 0:
        raise ValueError(f"elapsed time must be positive, got {elapsed_s}")
    return bytes_moved * iters / elapsed_s / gb


def d2d_bandwidths(copy_bytes: int, iters: int, elapsed_s: float, gb: int) -> dict:
    """Both conventions for an on-device copy.

    read_plus_write counts every byte twice (read from the source, written to
    the destination); this is the convention the published method appears to
    use. each_byte_once counts the copy size once.
    """
    once = bandwidth_gbps(copy_bytes, iters, elapsed_s, gb)
    return {"read_plus_write": 2 * once, "each_byte_once": once}


def summarize(samples: list[float]) -> dict:
    if not samples:
        raise ValueError("no samples to summarize")
    return {
        "samples": list(samples),
        "median": statistics.median(samples),
        "min": min(samples),
        "max": max(samples),
        "n": len(samples),
    }
