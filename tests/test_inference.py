import json

import pytest

from gpu_mark import cli, inference
from gpu_mark.inference_params import InferenceParams, requests_for


def test_every_parameter_has_unit_kind_and_reason():
    for name, p in InferenceParams().as_dict().items():
        assert p["unit"], name
        assert p["kind"] in ("source", "default"), name
        assert len(p["reason"]) > 20, name
    json.dumps(InferenceParams().as_dict())


def test_source_values_match_published_method():
    p = InferenceParams()
    assert p.value("model") == "nvidia/Llama-3.1-Nemotron-Nano-4B-v1.1"
    assert p.value("dtype") == "bfloat16"
    assert p.value("concurrency_levels") == [1, 25, 50, 100]
    assert p.value("workloads") == {"chat": {"input_tokens": 128, "output_tokens": 128},
                                    "summarization": {"input_tokens": 4096, "output_tokens": 512}}


def test_request_count_rule():
    p = InferenceParams()
    assert [requests_for(p, c) for c in (1, 25, 50, 100)] == [50, 125, 250, 500]


def test_override_and_unknown():
    p = InferenceParams().with_values(concurrency_levels=[1, 25])
    assert p.as_dict()["concurrency_levels"]["kind"] == "override"
    with pytest.raises(KeyError):
        InferenceParams().with_values(nope=1)


def test_bench_args_carry_workload_and_concurrency():
    p = InferenceParams()
    a = inference.bench_args("vllm", p, "http://127.0.0.1:1", "/m", {"input_tokens": 4096, "output_tokens": 512},
                             50, "/r", "x.json")
    s = " ".join(a)
    for frag in ("--random-input-len 4096", "--random-output-len 512", "--max-concurrency 50",
                 "--num-prompts 250", "--random-range-ratio 0", "--ignore-eos", "--tokenizer /m",
                 "--model nvidia/Llama-3.1-Nemotron-Nano-4B-v1.1"):
        assert frag in s, frag


def test_extract_bench_reports_missing_fields():
    out, missing = inference.extract_bench({"output_throughput": 10.0, "completed": 5, "extra": 1})
    assert out == {"output_throughput": 10.0, "completed": 5}
    assert "total_token_throughput" in missing and "extra" not in out


def test_log_evidence_keeps_lines_and_kv_size():
    log = "INFO x\nINFO Available KV cache memory: 57.83 GiB\nINFO Using FLASH_ATTN backend\nnoise\n"
    ev = inference.log_evidence(log)
    assert ev["kv_cache_gib"] == 57.83
    assert len(ev["lines"]) == 2
    assert ev["tail"][-1] == "noise"


def test_inference_params_command(capsys):
    assert cli.main(["inference-params"]) == 0
    assert "concurrency_levels" in json.loads(capsys.readouterr().out)
