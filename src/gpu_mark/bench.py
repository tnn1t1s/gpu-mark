"""Timed kernels: matrix multiply per precision, on-device copy, host transfers.

Each function returns elapsed seconds for one timed batch. On a CUDA device
the time comes from CUDA events on the current stream; on CPU (used only by
the tests) it comes from time.perf_counter.
"""

from __future__ import annotations

import time
from contextlib import contextmanager

import torch

FP8_MIN_CAPABILITY = (8, 9)  # Ada Lovelace; Hopper is (9, 0)

DTYPES = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def configure_fp32(tf32_allowed: bool) -> dict:
    """Set the FP32 math mode process-wide and return the effective settings."""
    torch.backends.cuda.matmul.allow_tf32 = tf32_allowed
    torch.backends.cudnn.allow_tf32 = tf32_allowed
    torch.set_float32_matmul_precision("high" if tf32_allowed else "highest")
    return {
        "torch.backends.cuda.matmul.allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "torch.backends.cudnn.allow_tf32": torch.backends.cudnn.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
    }


def fp8_supported(device: torch.device) -> bool:
    if device.type != "cuda":
        return False
    if not hasattr(torch, "float8_e4m3fn") or not hasattr(torch, "_scaled_mm"):
        return False
    return torch.cuda.get_device_capability(device) >= FP8_MIN_CAPABILITY


def _sync(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@contextmanager
def _timed(device: torch.device, out: list):
    """Append the elapsed seconds of the enclosed work to `out`."""
    if device.type == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        yield
        end.record()
        end.synchronize()
        out.append(start.elapsed_time(end) / 1000.0)
    else:
        t0 = time.perf_counter()
        yield
        out.append(time.perf_counter() - t0)


class Matmul:
    """Operands and a launch function for one precision on one device."""

    def __init__(self, precision: str, n: int, device: torch.device, seed: int = 0, fp8_out_dtype=torch.bfloat16):
        self.precision = precision
        self.n = n
        self.device = device
        g = torch.Generator(device="cpu").manual_seed(seed)
        if precision == "fp8":
            a = torch.randn(n, n, generator=g).to(device=device, dtype=torch.float8_e4m3fn)
            # _scaled_mm needs the second operand in column-major layout.
            b = torch.randn(n, n, generator=g).to(device=device, dtype=torch.float8_e4m3fn).t()
            self.scale = torch.ones((), device=device, dtype=torch.float32)
            self.a, self.b = a, b
            self.out_dtype = fp8_out_dtype
            self.c = None
        else:
            dtype = DTYPES[precision]
            self.a = torch.randn(n, n, generator=g).to(device=device, dtype=dtype)
            self.b = torch.randn(n, n, generator=g).to(device=device, dtype=dtype)
            self.c = torch.empty(n, n, device=device, dtype=dtype)

    def launch(self):
        if self.precision == "fp8":
            r = torch._scaled_mm(self.a, self.b, scale_a=self.scale, scale_b=self.scale, out_dtype=self.out_dtype)
            return r[0] if isinstance(r, tuple) else r  # torch < 2.4 returns (out, amax)
        return torch.matmul(self.a, self.b, out=self.c)

    def run(self, warmup: int, iters: int) -> float:
        for _ in range(warmup):
            self.launch()
        _sync(self.device)
        out: list[float] = []
        with _timed(self.device, out):
            for _ in range(iters):
                self.launch()
        return out[0]

    def free(self):
        self.a = self.b = self.c = None


class DeviceCopy:
    def __init__(self, nbytes: int, device: torch.device):
        self.device = device
        self.src = torch.empty(nbytes, dtype=torch.uint8, device=device).random_(0, 256)
        self.dst = torch.empty_like(self.src)

    def run(self, warmup: int, iters: int) -> float:
        for _ in range(warmup):
            self.dst.copy_(self.src)
        _sync(self.device)
        out: list[float] = []
        with _timed(self.device, out):
            for _ in range(iters):
                self.dst.copy_(self.src)
        return out[0]

    def free(self):
        self.src = self.dst = None


class HostCopy:
    """Pinned host buffer and device buffer for host-to-device and device-to-host transfers."""

    def __init__(self, nbytes: int, device: torch.device):
        self.device = device
        pin = device.type == "cuda"
        self.host = torch.empty(nbytes, dtype=torch.uint8, pin_memory=pin)
        self.dev = torch.empty(nbytes, dtype=torch.uint8, device=device)

    def run(self, direction: str, warmup: int, iters: int) -> float:
        if direction == "h2d":
            src, dst = self.host, self.dev
        elif direction == "d2h":
            src, dst = self.dev, self.host
        else:
            raise ValueError(f"direction must be h2d or d2h, got {direction!r}")
        for _ in range(warmup):
            dst.copy_(src, non_blocking=True)
        _sync(self.device)
        out: list[float] = []
        with _timed(self.device, out):
            for _ in range(iters):
                dst.copy_(src, non_blocking=True)
        return out[0]

    def free(self):
        self.host = self.dev = None
