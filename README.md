# gpu-mark

gpu-mark measures what one machine's NVIDIA GPUs deliver on a fixed set of
kernels: dense matrix multiply throughput at each precision, on-device memory
copy bandwidth, and host-to-device and device-to-host transfer bandwidth. It
times each GPU on its own, then all of them at once, records the hardware and
software it ran on, samples power, temperature and clocks while it runs, and
writes one JSON file per run.

The test selection follows the single-GPU suite described in Slynko et al.,
"Did You Win the GPU Cloud Lottery? Benchmarking from TFLOPS to Tokens/$"
(GPGPU '26, [doi:10.1145/3818671.3818674](https://doi.org/10.1145/3818671.3818674)),
which is the source of the 16,384 × 16,384 matrix size and the 1 GB copy size.
The paper leaves warmup, iteration count, timing method and FP32 math mode
unstated, so gpu-mark picks its own values for those and writes every choice
into the result file. gpu-mark numbers compare cleanly with each other;
comparing them with another tool's numbers means holding the parameter table
below against that tool's method.

## Running it

gpu-mark needs Linux, an NVIDIA driver with `nvidia-smi` on the path, and a
CUDA build of PyTorch 2.1 or newer. It uses the PyTorch already installed on
the host, because a GPU image usually ships a torch build matched to its CUDA
libraries.

```sh
pip install --no-deps git+https://github.com/tnn1t1s/gpu-mark
gpu-mark run --provider <who supplies the machine> --out result.json
```

Without installing, from a checkout:

```sh
PYTHONPATH=src python3 -m gpu_mark.cli run --provider <provider> --out result.json
```

`--provider` is required and is recorded as given; gpu-mark doesn't try to
detect the provider. `--gpus 0,2` limits the run to some devices.
`--set NAME=VALUE` overrides a method parameter, and the result file then marks
that parameter as an override and keeps the default it replaced.
`gpu-mark params` prints every parameter with its unit and reason.
`--json-stdout` also prints the result document between `GPU_MARK_JSON_BEGIN`
and `GPU_MARK_JSON_END` lines, for job runners that collect stdout but not
files.

The run makes no network call and reads no credential. Most of its time goes
to the FP32 matrix multiply, which runs on the CUDA cores at a small fraction
of tensor-core speed.

## What it measures

**Matrix multiply.** For each precision gpu-mark multiplies two square
matrices of side 16,384 with `torch.matmul` and reports TFLOPS, counting
2 × 16,384³ floating-point operations per multiply. Operands come from a
standard normal distribution with a fixed seed, because tensor-core power draw
depends on operand values and all-zero inputs would understate the load.

- FP32 runs with TF32 disabled for both cuBLAS and cuDNN and with
  `torch.set_float32_matmul_precision("highest")`, so it measures the CUDA
  cores. The effective flags are copied into `method.fp32_math_mode`.
- FP16 and BF16 run on the tensor cores.
- FP8 (`float8_e4m3fn`) runs through `torch._scaled_mm` with unit scales and
  BF16 output, and only on GPUs of compute capability 8.9 or higher (Ada
  Lovelace, Hopper and later). On older GPUs the result lists FP8 under
  `skipped` with the reason.

**On-device copy.** gpu-mark copies a 2³⁰-byte buffer to a second buffer on
the same GPU and reports the result two ways:

- `d2d_copy_read_plus_write_gbps` counts the bytes read plus the bytes
  written, so each copied byte counts twice. The GPGPU '26 paper's figures
  fit this convention: its best H200 figure is 4,213 GB/s, and counting each
  byte once would put the memory traffic at about 8.4 TB/s, above the
  H200's 4.8 TB/s specification.
- `d2d_copy_each_byte_once_gbps` counts the copy size once, half the first
  figure.

Both describe a copy; neither is a read-only or STREAM-style bandwidth.

**Host transfers.** A 2³⁰-byte pinned host buffer is copied to the GPU and
back, reported as `h2d_pinned_gbps` and `d2h_pinned_gbps`. These depend on
the PCIe generation and width, which the inventory records.

**All GPUs at once.** On a machine with two or more GPUs, gpu-mark repeats the
matrix multiplies and the on-device copy on every GPU concurrently, one thread
per GPU. Each repetition starts on all GPUs together at a barrier, each GPU
times its own batch, and the aggregate is the sum of the per-GPU results. Run
this way, the GPUs share the machine's power, cooling and host CPU, which the
one-at-a-time pass never exercises. The per-GPU medians from the concurrent
pass sit next to the sum. With one GPU the pass is skipped and the result
says so.

**Telemetry.** A background thread runs `nvidia-smi` every 500 ms and records
power draw, power limit, temperature, SM and memory clocks, utilisation and
the active clock-throttle reasons for every GPU. The runner records the start
and end of each benchmark phase on the same clock, and the result file holds a
per-phase summary (mean and maximum power, maximum temperature, SM clock
range, throttle reasons seen) along with the raw samples.

**Inventory.** For each GPU: model string, UUID, serial number, VBIOS, memory
size, ECC and MIG mode, current, default and maximum power limit, maximum and
application clocks, PCIe link generation and width, persistence and compute
mode, and the temperature, utilisation and memory use at start. For the
software: driver version, the CUDA version the driver supports, and the
PyTorch, CUDA and cuDNN versions torch was built with. For the host: CPU
model, logical CPU count, CPUs available to the process, RAM, kernel, OS and
hostname. Processes already using a GPU when the run starts go under
`compute_processes_at_start`, since they lower every figure.

## Method parameters

Every parameter below is written into `method.parameters` of each result with
its value, unit, kind and reason. Kind `source` means the value comes from the
GPGPU '26 paper; kind `default` means gpu-mark chose it.

| Parameter | Default | Kind | Why |
|---|---|---|---|
| `matmul_n` | 16,384 | source | Matrix side of the paper's FP16 test. gpu-mark uses it for every precision; the paper doesn't give the size for the others. |
| `matmul_warmup_iters` | 5 | default | The first calls select a cuBLAS kernel and allocate workspace while the clock ramps up from idle. These calls are not timed. |
| `matmul_timed_iters` | 10 | default | Ten calls per timed batch keep an A100 FP32 batch near 4.5 s and a tensor-core batch under half a second. |
| `matmul_input` | normal(0,1), seed 0 | default | Realistic operand values, identical across runs. |
| `fp32_tf32_allowed` | false | default | FP32 measures the CUDA cores. With TF32 allowed, an A100 FP32 multiply would run on the tensor cores at up to eight times the speed. |
| `fp8_out_dtype` | bfloat16 | default | Output type for the FP8 multiply. |
| `d2d_copy_bytes` | 2³⁰ | source | The paper's 1 GB, read as 2³⁰ bytes. |
| `d2d_warmup_iters` / `d2d_timed_iters` | 5 / 20 | default | A batch of 20 copies lasts about 20 ms at 2 TB/s, long enough that event timing resolution doesn't matter. |
| `host_copy_bytes` | 2³⁰ | default | Same size as the on-device copy. |
| `host_warmup_iters` / `host_timed_iters` | 2 / 5 | default | About 0.2 s per batch on PCIe 4.0 x16. |
| `repetitions` | 5 | default | Timed batches per metric per GPU. The paper's run-to-run analysis uses devices with three or more samples; with five, one slow batch can't move the median. |
| `timing` | CUDA events | default | Two events on the measuring stream bracket each timed batch, so Python overhead before the first launch is excluded. |
| `statistic` | median, min, max | default | Each repetition yields one figure, the mean over its batch. The summary reports the median across repetitions with the minimum and maximum, and keeps every sample. |
| `gb_unit` | 10⁹ bytes | default | Decimal GB/s, the unit of GPU memory specifications. |
| `clocks_locked` | false | default | Clocks run as the host set them. Locking them needs root and changes the machine under test. |
| `telemetry_interval_ms` | 500 | default | Sampling period for `nvidia-smi`. |
| `all_gpu_pass_min_gpus` | 2 | default | With fewer GPUs the concurrent pass would repeat the per-GPU pass. |

GPU numbering follows the PCI bus order that `nvidia-smi` uses. The command
line sets `CUDA_DEVICE_ORDER=PCI_BUS_ID` before CUDA starts, and the runner
refuses to run under any other order. Each torch device is matched to its
`nvidia-smi` row by UUID when torch exposes one, and by position otherwise;
the result records which.

## Result file

The JSON Schema is in
[`src/gpu_mark/result.schema.json`](src/gpu_mark/result.schema.json). The top
level holds:

| Key | Contents |
|---|---|
| `schema_version` | `"1"` |
| `tool` | name and version of gpu-mark |
| `run` | run id (random UUID), start and end time in UTC, duration in seconds, provider, CUDA device order, `CUDA_VISIBLE_DEVICES` |
| `method` | every parameter as `{value, unit, kind, reason}`, and the effective FP32 math flags |
| `inventory` | `host`, `driver`, `software`, `gpus` and `compute_processes_at_start`, as listed above |
| `results.per_gpu` | one entry per GPU: index, `nvidia_smi_index`, UUID, how the UUID was matched, torch device name, compute capability, `metrics` and `skipped` |
| `results.all_gpu` | `status` of `ran` or `skipped`; when it ran, the summed `metrics` and `per_gpu_medians` |
| `telemetry` | phases with start, end and per-GPU summary, raw samples, sampler errors |
| `not_implemented` | the tests gpu-mark leaves out, with reasons |
| `errors` | reserved for errors recorded during a run |

Each metric is `{unit, samples, median, min, max, n}`, where `samples` holds
one figure per repetition. Metric names are `matmul_{fp32,fp16,bf16,fp8}_tflops`,
`d2d_copy_read_plus_write_gbps`, `d2d_copy_each_byte_once_gbps`,
`h2d_pinned_gbps` and `d2h_pinned_gbps`.

The hostname, GPU serial numbers and UUIDs identify a machine. Strip them
before publishing a result if that matters to you.

## Not implemented

- L2 cache bandwidth. A copy small enough to stay in L2 also measures how the
  L2 is partitioned and how fast kernels launch, so a clean figure needs a
  purpose-written kernel rather than a PyTorch operation.
- GPU-to-GPU bandwidth, and NCCL all-reduce and broadcast, on multi-GPU
  machines.
- Network tests between machines.
- LLM inference and fine-tuning throughput.
- Health checks: GPU memory tests, DCGM diagnostics, XID log scans, and a
  separate sustained-load thermal check.
- Grading a GPU against other GPUs of the same model. That needs a population
  of results, which a single run can't provide.

## How this method can differ from SiliconMark

SiliconMark's public documentation and the GPGPU '26 paper fix the FP16 matrix
size and the copy size and say nothing else about timing. A gpu-mark figure
and a SiliconMark figure for the same GPU can differ because of any of these:

- warmup length, iteration count, and whether the reported figure is a mean,
  median or best batch;
- timing with CUDA events here, against an unpublished method there;
- matrix size and math mode for FP32, BF16 and FP8, which the paper doesn't
  state;
- whether the copy figure counts bytes once or twice (gpu-mark reports both);
- the PyTorch and cuBLAS versions, which change kernel selection;
- the GPU's thermal state when the run starts.

## Development

```sh
uv sync
uv run pytest
```

The tests need no GPU. They check the arithmetic, the parameter table, and
the telemetry and inventory parsers, then run the matrix multiply and copy
code on CPU at small sizes and validate the resulting document against the
schema.

## License

MIT. See [LICENSE](LICENSE).
