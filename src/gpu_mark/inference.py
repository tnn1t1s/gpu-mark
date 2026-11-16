"""LLM inference test: serve one model with vLLM on one GPU and measure
throughput at each workload and concurrency level.

The flow follows section 2.2 of Slynko et al., GPGPU '26: start a vLLM
server, then for each workload (input and output length) and each
concurrency level, send requests with at most that many in flight and record
the throughput. gpu-mark adds what the paper leaves out: a pinned model
revision, the load generator's settings, server startup timings, the KV
cache size and attention backend vLLM reports, and GPU telemetry per level.

vLLM is not a dependency of gpu-mark. The `vllm` command must be on PATH,
for example through `uv run --with vllm==<version>`.
"""

from __future__ import annotations

import datetime as dt
import importlib.metadata as md
import json
import os
import re
import shutil
import socket
import subprocess
import tempfile
import time
import urllib.request
import uuid as uuidlib

from . import __version__, inventory, telemetry
from .inference_params import InferenceParams, requests_for

SCHEMA_VERSION = "1"

# Server log lines kept as evidence: KV cache sizing, attention backend,
# kernel availability and CUDA graph capture. Kept verbatim, not interpreted.
LOG_EVIDENCE = re.compile(r"KV cache|[Bb]ackend|FlashInfer|CUDA graph|max_model_len|Maximum concurrency")
KV_GIB = re.compile(r"Available KV cache memory:\s*([0-9.]+)\s*GiB")

# Scalar fields copied from each `vllm bench serve` result file.
BENCH_FIELDS = [
    "completed", "failed", "duration", "total_input_tokens", "total_output_tokens",
    "request_throughput", "output_throughput", "total_token_throughput",
    "mean_ttft_ms", "median_ttft_ms", "p99_ttft_ms",
    "mean_tpot_ms", "median_tpot_ms", "p99_tpot_ms",
    "mean_itl_ms", "median_itl_ms", "p99_itl_ms",
    "mean_e2el_ms", "median_e2el_ms", "p99_e2el_ms",
    "max_concurrency",
]


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _version(dist: str) -> str | None:
    try:
        return md.version(dist)
    except md.PackageNotFoundError:
        return None


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _http(url: str, body: dict | None = None, timeout: float = 10.0) -> tuple[int, bytes]:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"} if data else {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, r.read()


def bench_args(vllm: str, p: InferenceParams, base_url: str, tokenizer: str, workload: dict,
               concurrency: int, result_dir: str, result_file: str) -> list[str]:
    return [
        vllm, "bench", "serve",
        "--backend", "vllm",
        "--base-url", base_url,
        "--endpoint", p.value("endpoint"),
        "--model", p.value("model"),
        "--tokenizer", tokenizer,
        "--dataset-name", "random",
        "--random-input-len", str(workload["input_tokens"]),
        "--random-output-len", str(workload["output_tokens"]),
        "--random-range-ratio", "0",
        *(["--ignore-eos"] if p.value("ignore_eos") else []),
        "--temperature", str(p.value("temperature")),
        "--num-prompts", str(requests_for(p, concurrency)),
        "--max-concurrency", str(concurrency),
        "--request-rate", "inf",
        "--num-warmups", str(p.value("warmup_requests")),
        "--seed", str(p.value("seed")),
        "--percentile-metrics", "ttft,tpot,itl,e2el",
        "--metric-percentiles", "50,99",
        "--disable-tqdm",
        "--save-result", "--result-dir", result_dir, "--result-filename", result_file,
    ]


def extract_bench(raw: dict) -> tuple[dict, list[str]]:
    """Copy BENCH_FIELDS from a bench result; report the names it lacks."""
    out = {k: raw[k] for k in BENCH_FIELDS if k in raw}
    missing = [k for k in BENCH_FIELDS if k not in raw]
    return out, missing


LOG_TAIL_LINES = 40  # last lines of the server log kept, so a startup failure keeps its final exception
LOG_ERROR_LINES = 40  # last ERROR lines kept; the engine process logs its root cause there


def log_evidence(log_text: str) -> dict:
    all_lines = log_text.splitlines()
    lines = [ln.strip() for ln in all_lines if LOG_EVIDENCE.search(ln)]
    m = KV_GIB.findall(log_text)
    # vLLM's engine process logs its root cause as ERROR lines well before the
    # API server's own traceback, which fills the tail; keep those lines too.
    errors = [ln.strip() for ln in all_lines
              if " ERROR " in ln and not ln.rstrip().endswith("^") and '  File "' not in ln]
    return {"kv_cache_gib": float(m[-1]) if m else None, "lines": lines[:60],
            "errors": errors[-LOG_ERROR_LINES:], "tail": all_lines[-LOG_TAIL_LINES:]}


class Server:
    """A `vllm serve` process on a loopback port, logging to a file."""

    def __init__(self, vllm: str, p: InferenceParams, model_path: str, log_path: str):
        self.port = _free_port()
        self.base_url = f"http://127.0.0.1:{self.port}"
        self.log_path = log_path
        self.args = [
            vllm, "serve", model_path,
            "--served-model-name", p.value("model"),
            "--dtype", p.value("dtype"),
            "--gpu-memory-utilization", str(p.value("gpu_memory_utilization")),
            "--seed", str(p.value("seed")),
            "--host", "127.0.0.1", "--port", str(self.port),
        ]
        self.proc: subprocess.Popen | None = None

    def start(self):
        self._log = open(self.log_path, "w")
        self.proc = subprocess.Popen(self.args, stdout=self._log, stderr=subprocess.STDOUT)

    def wait_ready(self, p: InferenceParams) -> dict:
        """Ready means one generation completed, not only that /health answered."""
        t0 = time.time()
        deadline = t0 + p.value("server_ready_timeout_s")
        health_s = None
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"vllm serve exited with code {self.proc.returncode} during startup; "
                                   f"see {self.log_path}")
            try:
                if health_s is None and _http(self.base_url + "/health", timeout=5)[0] == 200:
                    health_s = time.time() - t0
                if health_s is not None:
                    status, body = _http(self.base_url + "/v1/completions",
                                         {"model": p.value("model"), "prompt": "Hello", "max_tokens": 2,
                                          "temperature": 0}, timeout=120)
                    if status == 200 and json.loads(body).get("choices"):
                        return {"health_ok_s": round(health_s, 2),
                                "first_generation_s": round(time.time() - t0, 2)}
            except OSError:
                pass
            time.sleep(2)
        raise TimeoutError(f"vllm serve not ready after {p.value('server_ready_timeout_s')} s; see {self.log_path}")

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        if getattr(self, "_log", None):
            self._log.close()


def resolve_model(p: InferenceParams) -> str:
    """Local snapshot of the pinned revision, downloading it if needed."""
    from huggingface_hub import snapshot_download

    return snapshot_download(p.value("model"), revision=p.value("model_revision"))


def run(provider: str, p: InferenceParams, out_path: str, server_log: str | None = None) -> dict:
    if not provider:
        raise ValueError("provider is required")
    vllm = shutil.which("vllm")
    if vllm is None:
        raise RuntimeError("the `vllm` command is not on PATH; run under `uv run --with vllm==<version>`")
    order = os.environ.get("CUDA_DEVICE_ORDER")
    if order != "PCI_BUS_ID":
        raise RuntimeError(f"CUDA_DEVICE_ORDER must be PCI_BUS_ID, got {order!r}")

    started = time.time()
    work = tempfile.mkdtemp(prefix="gpu-mark-inference-")
    log_path = server_log or os.path.join(work, "vllm-serve.log")
    gpus = inventory.nvidia_smi_gpus()
    if len(gpus) != 1:
        # Multi-GPU hosts are out of scope: telemetry and inventory would need to
        # follow the one GPU vLLM picks.
        raise RuntimeError(f"the inference test needs exactly one GPU visible to nvidia-smi; found {len(gpus)}")
    doc: dict = {
        "schema_version": SCHEMA_VERSION,
        "kind": "inference",
        "tool": {"name": "gpu-mark", "version": __version__},
        "run": {"run_id": str(uuidlib.uuid4()), "started_at": _now(), "provider": provider,
                "cuda_device_order": order, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")},
        "method": {"parameters": p.as_dict()},
        "inventory": {
            "host": inventory.host_inventory(provider),
            "driver": inventory.nvidia_smi_driver(),
            "software": {**inventory.torch_versions(),
                         **{d: _version(d) for d in ("vllm", "transformers", "flashinfer-python",
                                                      "flashinfer-cubin", "triton", "huggingface-hub")}},
            "gpus": gpus,
            "compute_processes_at_start": inventory.compute_processes(),
        },
        "server": {},
        "results": {"levels": []},
        "errors": [],
    }

    def checkpoint():
        with open(out_path, "w") as f:
            json.dump(doc, f, indent=1)

    phases: list[dict] = []
    server = None
    with telemetry.Sampler(p.value("telemetry_interval_ms")) as sampler:
        try:
            t = time.time()
            model_path = resolve_model(p)
            doc["server"]["model_resolve_s"] = round(time.time() - t, 2)
            server = Server(vllm, p, model_path, log_path)
            doc["server"]["args"] = server.args[1:]
            server.start()
            t = time.time()
            doc["server"].update(server.wait_ready(p))
            phases.append({"name": "server_startup", "start": t, "end": time.time()})
            with open(log_path) as f:
                doc["server"]["log_evidence"] = log_evidence(f.read())
            checkpoint()

            for wname, wl in p.value("workloads").items():
                for c in p.value("concurrency_levels"):
                    name = f"{wname}_c{c}"
                    fname = f"{name}.json"
                    args = bench_args(vllm, p, server.base_url, model_path, wl, c, work, fname)
                    t = time.time()
                    proc = subprocess.run(args, capture_output=True, text=True)
                    end = time.time()
                    phases.append({"name": name, "start": t, "end": end})
                    level = {"name": name, "workload": wname, **wl, "concurrency": c,
                             "requests": requests_for(p, c), "wall_s": round(end - t, 2)}
                    if proc.returncode != 0:
                        doc["errors"].append({"level": name, "returncode": proc.returncode,
                                              "output_tail": (proc.stdout + proc.stderr)[-3000:]})
                    else:
                        with open(os.path.join(work, fname)) as f:
                            metrics, missing = extract_bench(json.load(f))
                        level["metrics"] = metrics
                        if missing:
                            doc["errors"].append({"level": name, "missing_bench_fields": missing})
                    doc["results"]["levels"].append(level)
                    checkpoint()
                    if server.proc.poll() is not None:
                        raise RuntimeError(f"vllm serve exited with code {server.proc.returncode} during {name}")
        except Exception as e:  # recorded in the result; the CLI exits non-zero
            doc["errors"].append({"fatal": f"{type(e).__name__}: {e}"})
        finally:
            if server is not None:
                server.stop()
                if os.path.exists(log_path):
                    with open(log_path) as f:
                        doc["server"]["log_evidence"] = log_evidence(f.read())
            doc["server"]["log_path"] = log_path

    gpu_index = gpus[0]["index"]
    doc["telemetry"] = {
        "source": "nvidia-smi",
        "interval_ms": p.value("telemetry_interval_ms"),
        "phases": [{**ph, "summary": telemetry.summarize_phase(sampler.samples, gpu_index, ph["start"], ph["end"])}
                   for ph in phases],
        "samples": sampler.samples,
        "sampler_errors": sampler.errors,
    }
    doc["run"]["ended_at"] = _now()
    doc["run"]["duration_s"] = round(time.time() - started, 3)
    checkpoint()
    return doc


def format_summary(doc: dict) -> str:
    sw = doc["inventory"]["software"]
    srv = doc["server"]
    ev = srv.get("log_evidence", {})
    lines = [
        f"gpu-mark {doc['tool']['version']} inference  provider={doc['run']['provider']}  "
        f"duration={doc['run']['duration_s']:.0f}s",
        f"vllm {sw.get('vllm')}  torch {sw.get('torch')}  CUDA {sw.get('torch_cuda')}  "
        f"flashinfer {sw.get('flashinfer-python')}",
        f"model {doc['method']['parameters']['model']['value']} @ "
        f"{str(doc['method']['parameters']['model_revision']['value'])[:12]}",
        f"server ready: health {srv.get('health_ok_s')} s, first generation {srv.get('first_generation_s')} s; "
        f"KV cache {ev.get('kv_cache_gib')} GiB",
        "",
        f"  {'level':20s} {'reqs':>5s} {'out tok/s':>10s} {'total tok/s':>12s} {'TTFT p50 ms':>12s} "
        f"{'TPOT p50 ms':>12s} {'failed':>6s}",
    ]
    for lv in doc["results"]["levels"]:
        m = lv.get("metrics")
        if m is None:
            lines.append(f"  {lv['name']:20s} {lv['requests']:5d}  failed, see errors")
            continue

        def f(k, w, nd=1):
            v = m.get(k)
            return f"{v:{w}.{nd}f}" if isinstance(v, (int, float)) else f"{'-':>{w}s}"
        lines.append(f"  {lv['name']:20s} {lv['requests']:5d} {f('output_throughput', 10)} "
                     f"{f('total_token_throughput', 12)} {f('median_ttft_ms', 12)} {f('median_tpot_ms', 12)} "
                     f"{m.get('failed', '-')!s:>6s}")
    for e in doc["errors"]:
        lines.append(f"error: {json.dumps(e)[:300]}")
    return "\n".join(lines)
