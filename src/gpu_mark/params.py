"""Method parameters.

Every value that changes what the benchmark measures lives here, with its
unit, its classification and the reason for the default. The effective
values are written into every result file under ``method.parameters`` so a
result can be compared against another method without reading the source.

Classification follows the README: ``source`` means the value comes from the
published method we reproduce (Slynko et al., GPGPU '26); ``default`` means
we chose it and it is open to revision.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields

GIB = 1 << 30


@dataclass(frozen=True)
class Param:
    value: object
    unit: str
    kind: str  # "source" | "default"
    reason: str

    def as_dict(self) -> dict:
        return {"value": self.value, "unit": self.unit, "kind": self.kind, "reason": self.reason}


def _p(value, unit, kind, reason):
    return field(default_factory=lambda: Param(value, unit, kind, reason))


@dataclass(frozen=True)
class MethodParams:
    matmul_n: Param = _p(
        16384, "elements per side", "source",
        "Square matrix size of the published FP16 test. Applied to every precision; "
        "the published method does not state the size for FP32, BF16 or FP8.",
    )
    matmul_warmup_iters: Param = _p(
        5, "iterations", "default",
        "Untimed calls before each timed batch. The first calls select a cuBLAS "
        "kernel and allocate workspace, and the GPU clock ramps from idle.",
    )
    matmul_timed_iters: Param = _p(
        10, "iterations", "default",
        "Back-to-back calls inside one timed batch. At A100 FP32 speed one call takes "
        "about 0.45 s, so a batch lasts about 4.5 s; tensor-core batches last under 0.5 s.",
    )
    matmul_input: Param = _p(
        "normal(0,1), seed 0", "distribution", "default",
        "Tensor-core power draw depends on operand values; zeros would understate load. "
        "A fixed seed makes the operands identical across runs.",
    )
    fp32_tf32_allowed: Param = _p(
        False, "bool", "default",
        "FP32 is measured on CUDA cores. TF32 is disabled for matmul and cuDNN, and "
        "float32 matmul precision is set to 'highest'.",
    )
    fp8_out_dtype: Param = _p(
        "bfloat16", "dtype", "default",
        "FP8 matmul output type for torch._scaled_mm with unit scales. Runs only on "
        "compute capability 8.9 or higher.",
    )
    d2d_copy_bytes: Param = _p(
        GIB, "bytes", "source",
        "Size of the device-to-device copy. The source says 1 GB; we read that as 2^30 bytes.",
    )
    d2d_warmup_iters: Param = _p(5, "iterations", "default", "Untimed copies before each timed batch.")
    d2d_timed_iters: Param = _p(
        20, "iterations", "default",
        "Copies inside one timed batch, about 20 ms of work at 2 TB/s.",
    )
    host_copy_bytes: Param = _p(
        GIB, "bytes", "default",
        "Size of each host-to-device and device-to-host transfer, from pinned host memory. "
        "Same size as the on-device copy.",
    )
    host_warmup_iters: Param = _p(2, "iterations", "default", "Untimed transfers before each timed batch.")
    host_timed_iters: Param = _p(
        5, "iterations", "default",
        "Transfers inside one timed batch, about 0.2 s on PCIe 4.0 x16.",
    )
    repetitions: Param = _p(
        5, "repetitions", "default",
        "Timed batches per metric per GPU. The published method uses devices with three or "
        "more samples for run-to-run analysis; five gives a median that one outlier cannot move.",
    )
    timing: Param = _p(
        "cuda_events", "method", "default",
        "Elapsed time between two CUDA events recorded on the measuring stream around the "
        "timed batch. Excludes Python launch overhead before the first kernel.",
    )
    statistic: Param = _p(
        "median, min, max across repetitions; each repetition is the mean over its timed batch",
        "statistic", "default",
        "The median resists one slow repetition; min and max show the spread.",
    )
    gb_unit: Param = _p(
        1_000_000_000, "bytes per GB", "default",
        "Bandwidth is reported in decimal GB/s, the unit of GPU memory specifications.",
    )
    clocks_locked: Param = _p(
        False, "bool", "default",
        "Clocks run as the host configured them. Locking needs root and changes the "
        "machine, and the point is to measure the machine as rented.",
    )
    telemetry_interval_ms: Param = _p(
        500, "ms", "default",
        "nvidia-smi sampling period during the run. Shorter periods add CPU load and "
        "the driver averages power over a window anyway.",
    )
    all_gpu_pass_min_gpus: Param = _p(
        2, "GPUs", "default",
        "The all-GPU pass runs only with this many GPUs or more. With one GPU it would "
        "repeat the per-GPU pass.",
    )

    def as_dict(self) -> dict:
        return {f.name: getattr(self, f.name).as_dict() for f in fields(self)}

    def value(self, name: str):
        return getattr(self, name).value

    def with_values(self, **overrides) -> "MethodParams":
        """Return a copy with some values replaced; kind becomes 'override'."""
        kw = {}
        for name, v in overrides.items():
            if not hasattr(self, name):
                raise KeyError(f"unknown parameter {name!r}")
            old: Param = getattr(self, name)
            kw[name] = Param(v, old.unit, "override", f"set on the command line; default was {old.value!r}")
        return MethodParams(**{f.name: kw.get(f.name, getattr(self, f.name)) for f in fields(self)})


# Metrics deliberately left out of v1, written into every result file.
NOT_IMPLEMENTED = [
    {"name": "l2_bandwidth", "reason": "No PyTorch-level kernel isolates L2 cleanly; a copy sized to fit in L2 "
     "also measures the L2 partition split and kernel launch rate. Needs a custom kernel."},
    {"name": "gpu_to_gpu_bandwidth", "reason": "Peer-to-peer copy between GPUs is out of scope for v1."},
    {"name": "nccl_allreduce_broadcast", "reason": "NCCL collectives are out of scope for v1."},
    {"name": "cluster_network", "reason": "Multi-node network test is out of scope for v1."},
    {"name": "finetuning", "reason": "Fine-tuning throughput is out of scope. LLM inference is a separate "
     "command, `gpu-mark inference`, with its own result file."},
    {"name": "health_audit", "reason": "Memory test, DCGM diagnostics and XID scan are out of scope for v1."},
    {"name": "thermal_check", "reason": "The separate heat-and-cool check is unspecified in public sources."},
    {"name": "peer_tier_grading", "reason": "Grading needs a same-model population of results; none exists yet."},
]
