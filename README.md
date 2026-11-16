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

## Purpose

gpu-mark exists to reproduce published GPU benchmark results first, and to
publish its own measurements only after that. A reproduction runs the
published method as closely as the paper allows, lines the numbers up against
the paper's tables, and records every place where the method had to guess.
New benchmarks go into gpu-mark only once the reproduction they build on
holds. Each reproduction is tracked as a GitHub issue that lists the paper's
claims one by one and marks each as reproduced, contradicted or not testable,
with the reason.

Raw result files identify the machine they ran on, so they don't belong in
this repository. It holds the code, the method and aggregated tables.

## Running it

gpu-mark needs Linux, an NVIDIA driver with `nvidia-smi` on the path, and a
CUDA build of PyTorch 2.1 or newer. It uses the PyTorch already installed on
the host, because a GPU image usually ships a torch build matched to its CUDA
libraries.

```sh
pip install --no-deps git+https://github.com/tnn1t1s/gpu-mark
gpu-mark run --provider <who supplies the machine> --out result.json
```

On a host with no PyTorch installed, [uv](https://docs.astral.sh/uv/) can
supply one for the run without touching the system Python. The PyPI torch
wheel for Linux x86_64 bundles its CUDA libraries and needs several GB of
disk.

```sh
uv run --with torch --with git+https://github.com/tnn1t1s/gpu-mark \
  gpu-mark run --provider <provider> --out result.json
```

On a small container disk, set `UV_LINK_MODE=symlink` so uv links the
environment to its cache instead of copying the torch files a second time.

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

## LLM inference test

`gpu-mark inference` serves `nvidia/Llama-3.1-Nemotron-Nano-4B-v1.1` with vLLM
at its default settings in bf16, and drives it with `vllm bench serve`. It runs
the Chat workload (128 input and 128 output tokens) and the Summarization
workload (4,096 in, 512 out), each at 1, 25, 50 and 100 requests in flight. The
workloads and levels come from Slynko et al., GPGPU '26, section 2.2. Every
choice that paper leaves open is a parameter with its reason;
`gpu-mark inference-params` prints them. The result records output and total
tokens per second, time to first token, time per output token, the KV cache
size vLLM reports, prefix-cache hits and GPU telemetry for each level.

It needs one GPU with about 70 GiB of memory, `vllm` on the path, and about
30 GB of disk for vLLM, its kernels and the model weights. vLLM also needs a C
compiler and Python headers on the host, because Triton compiles a launcher
when the server starts. Without `nvcc` on the host, vLLM also needs
FlashInfer's precompiled kernels, which FlashInfer publishes on its own package
index rather than on PyPI:

```sh
uv run --python 3.12 --with vllm==0.33.0 \
  --index https://flashinfer.ai/whl --index https://flashinfer.ai/whl/cu130 \
  --with flashinfer-cubin==0.7.2.post1 \
  --with flashinfer-jit-cache==0.7.2.post1+cu130 \
  --with git+https://github.com/tnn1t1s/gpu-mark \
  gpu-mark inference --provider <provider> --out result.json
```

The kernel package versions must match the `flashinfer-python` version that
vLLM installs, and the `cu130` label must match torch's CUDA version.

## Example output

The summary printed at the end of a single-GPU run on an
NVIDIA A100-SXM4-80GB (UUID removed):

```
gpu-mark 0.1.0  provider=runpod  duration=56.9s
driver 595.91.07  torch 2.15.0+cu132  CUDA 13.2  cuDNN 92600

GPU 0  NVIDIA A100-SXM4-80GB
  metric                                 median        min        max  unit (n=5)
  matmul_fp32_tflops                       19.1       19.1       19.1  TFLOPS
  matmul_fp16_tflops                      260.6      259.9      263.0  TFLOPS
  matmul_bf16_tflops                      267.1      265.6      269.9  TFLOPS
  d2d_copy_read_plus_write_gbps          1730.9     1688.5     1742.3  GB/s (bytes read + bytes written)
  d2d_copy_each_byte_once_gbps            865.5      844.3      871.2  GB/s (copy size counted once)
  h2d_pinned_gbps                          18.4       18.3       18.6  GB/s
  d2h_pinned_gbps                          16.0       16.0       16.1  GB/s
  matmul_fp8_tflops                  skipped: not supported: compute capability (8, 0) is below (8, 9)

All-GPU pass skipped: 1 GPU(s) below all_gpu_pass_min_gpus=2
  telemetry gpu0:matmul_fp32_tflops      power mean 352 W (limit 400)  temp max 61 C  SM clock 1155-1410 MHz
  telemetry gpu0:matmul_fp16_tflops      power mean 202 W (limit 400)  temp max 58 C  SM clock 1155-1410 MHz
  telemetry gpu0:matmul_bf16_tflops      power mean 240 W (limit 400)  temp max 58 C  SM clock 1155-1410 MHz
  telemetry gpu0:host_copy               power mean 120 W (limit 400)  temp max 50 C  SM clock 1410-1410 MHz
```

The full result is a JSON document described under [Result file](#result-file).

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
  H200's 4.8 TB/s specification. An A100-SXM4-80GB points the same way: its
  read-plus-write figure falls inside the paper's A100 SXM4 80GB range of
  1,459 to 1,759 GB/s, while its each-byte-once figure falls far below it.
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
- Fine-tuning throughput.
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
