"""Method parameters for the LLM inference test.

Values marked ``source`` come from Slynko et al., GPGPU '26, section 2.2.
The paper does not publish its load generator, request count, warmup,
sampling settings or vLLM version; every value gpu-mark chose for those is
marked ``default`` and carries its reason. Effective values are written into
every inference result under ``method.parameters``.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

from .params import Param, _p


@dataclass(frozen=True)
class InferenceParams:
    model: Param = _p(
        "nvidia/Llama-3.1-Nemotron-Nano-4B-v1.1", "Hugging Face repo", "source",
        "The model the published inference test serves.",
    )
    model_revision: Param = _p(
        "d552708a9d575fa8d4a690b988fd870d65279f98", "git commit", "default",
        "The published method names the model but not a revision. Pinning the commit keeps "
        "weights and tokenizer identical across runs; this was the repo head on 2026-11-16.",
    )
    dtype: Param = _p("bfloat16", "dtype", "source", "The published test runs every experiment in bfloat16.")
    gpu_memory_utilization: Param = _p(
        0.9, "fraction of GPU memory", "default",
        "vLLM's own default. The paper reports about 58 GiB of KV cache next to 8.4 GiB of weights, "
        "which fits this fraction on an 80 GB GPU; that match is a hypothesis, checked by the KV cache "
        "size vLLM logs, which gpu-mark records.",
    )
    workloads: Param = _p(
        {"chat": {"input_tokens": 128, "output_tokens": 128},
         "summarization": {"input_tokens": 4096, "output_tokens": 512}},
        "tokens per request", "source",
        "The two synthetic workloads of the published test, as input and output sequence lengths.",
    )
    concurrency_levels: Param = _p(
        [1, 25, 50, 100], "simultaneous requests", "source",
        "The published test's concurrency levels: requests in flight at once, not batch size.",
    )
    load_generator: Param = _p(
        "vllm bench serve", "tool", "default",
        "The paper does not name its load generator. vLLM's own client is the closest standard choice: "
        "it ships with the engine and caps requests in flight with --max-concurrency.",
    )
    endpoint: Param = _p(
        "/v1/completions", "HTTP path", "default",
        "Plain completions, so no chat template changes the input length. The paper does not say "
        "which endpoint it used.",
    )
    dataset: Param = _p(
        "random, exact lengths (range ratio 0)", "dataset", "default",
        "The paper calls its workloads synthetic. Random token prompts with a range ratio of 0 make "
        "every request exactly the stated input length.",
    )
    ignore_eos: Param = _p(
        True, "bool", "default",
        "Every request generates exactly the stated output length instead of stopping early, so "
        "throughput figures compare across runs.",
    )
    temperature: Param = _p(
        0.0, "temperature", "default",
        "Greedy decoding. Sampling settings are unpublished; greedy removes one source of run-to-run noise.",
    )
    requests_per_slot: Param = _p(
        5, "requests per concurrent slot", "default",
        "Each level sends this many requests per slot, so every slot completes about five requests. "
        "The paper does not publish its request count.",
    )
    min_requests: Param = _p(
        50, "requests", "default",
        "Floor on requests per level, so low concurrency levels still give a median over 50 requests.",
    )
    warmup_requests: Param = _p(
        10, "requests", "default",
        "Untimed requests sent before each level's timed requests.",
    )
    seed: Param = _p(0, "seed", "default", "Seed for the random prompts and for the server.")
    server_ready_timeout_s: Param = _p(
        1200, "s", "default",
        "Time allowed for vLLM to load the weights, compile, capture CUDA graphs and complete one "
        "generation. Startup on an A100 was not yet measured when this was set.",
    )
    telemetry_interval_ms: Param = _p(
        500, "ms", "default", "nvidia-smi sampling period, the same as the micro-benchmarks.",
    )

    def as_dict(self) -> dict:
        return {f.name: getattr(self, f.name).as_dict() for f in fields(self)}

    def value(self, name: str):
        return getattr(self, name).value

    def with_values(self, **overrides) -> "InferenceParams":
        kw = {}
        for name, v in overrides.items():
            if not hasattr(self, name):
                raise KeyError(f"unknown parameter {name!r}")
            old: Param = getattr(self, name)
            kw[name] = Param(v, old.unit, "override", f"set on the command line; default was {old.value!r}")
        return InferenceParams(**{f.name: kw.get(f.name, getattr(self, f.name)) for f in fields(self)})


def requests_for(p: InferenceParams, concurrency: int) -> int:
    return max(p.value("min_requests"), p.value("requests_per_slot") * concurrency)
