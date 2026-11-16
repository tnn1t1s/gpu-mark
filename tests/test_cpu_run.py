"""Exercise the kernels and the result schema on CPU with small sizes."""

import json
from importlib import resources

import pytest

torch = pytest.importorskip("torch")
jsonschema = pytest.importorskip("jsonschema")

from gpu_mark import bench, cli, health, runner  # noqa: E402
from gpu_mark.params import NOT_IMPLEMENTED, MethodParams  # noqa: E402

SMALL = dict(matmul_n=64, matmul_warmup_iters=1, matmul_timed_iters=2, d2d_copy_bytes=1 << 16,
             d2d_warmup_iters=1, d2d_timed_iters=2, host_copy_bytes=1 << 16, host_warmup_iters=1,
             host_timed_iters=2, repetitions=3)


def test_matmul_matches_reference_on_cpu():
    mm = bench.Matmul("fp32", 32, torch.device("cpu"))
    assert torch.allclose(mm.launch(), mm.a @ mm.b)
    assert mm.run(1, 2) > 0


def test_configure_fp32_disables_tf32():
    eff = bench.configure_fp32(False)
    assert eff["torch.backends.cuda.matmul.allow_tf32"] is False
    assert eff["float32_matmul_precision"] == "highest"


def test_fp8_unsupported_on_cpu():
    assert bench.fp8_supported(torch.device("cpu")) is False


def _document(per_gpu_entry, p):
    return {
        "schema_version": runner.SCHEMA_VERSION,
        "tool": {"name": "gpu-mark", "version": "0.0.0"},
        "run": {"run_id": "x", "started_at": "t0", "ended_at": "t1", "duration_s": 1.0,
                "provider": "test", "cuda_device_order": "PCI_BUS_ID", "cuda_visible_devices": None},
        "method": {"parameters": p.as_dict(), "fp32_math_mode": bench.configure_fp32(False)},
        "inventory": {
            "host": {"provider": "test", "hostname": "h", "cpu_model": "c", "mem_total_bytes": 1},
            "driver": {"driver_version": "1", "cuda_driver_api_version": "1"},
            "software": {"torch": torch.__version__, "torch_cuda": None, "cudnn": None},
            "gpus": [{"index": "0", "name": "n", "uuid": "GPU-x", "serial": None, "ecc_mode": "Enabled",
                      "mig_mode": "Disabled", "power_limit_w": "400"}],
            "compute_processes_at_start": [],
        },
        "results": {
            "per_gpu": [{"index": 0, "nvidia_smi_index": 0, "uuid": "GPU-x", "match": "pci_order",
                         "torch_name": "cpu", "compute_capability": [0, 0], **per_gpu_entry}],
            "all_gpu": {"status": "skipped", "reason": "1 GPU(s) below all_gpu_pass_min_gpus=2"},
        },
        "telemetry": {"source": "nvidia-smi", "interval_ms": 500, "samples": [], "sampler_errors": [],
                      "phases": [{**ph, "summary": {}} for ph in []]},
        "rental": None,
        "health": health.evaluate([{"uuid": "GPU-x", "ecc_mode": "Enabled", "mig_mode": "Disabled",
                                    "power_limit_w": "400", "power_default_limit_w": "400"}], [],
                                  {"rows": [], "error": "not read in a CPU test"},
                                  {"lines": None, "error": "not read in a CPU test"}),
        "not_implemented": NOT_IMPLEMENTED,
        "errors": [],
    }


def test_per_gpu_pass_on_cpu_produces_schema_valid_result():
    p = MethodParams().with_values(**SMALL)
    phases = runner.Phases()
    entry = runner.per_gpu_pass(p, torch.device("cpu"), phases, 0)
    assert set(entry["metrics"]) == {
        "matmul_fp32_tflops", "matmul_fp16_tflops", "matmul_bf16_tflops",
        "d2d_copy_read_plus_write_gbps", "d2d_copy_each_byte_once_gbps", "h2d_pinned_gbps", "d2h_pinned_gbps"}
    assert "matmul_fp8_tflops" in entry["skipped"]
    for m in entry["metrics"].values():
        assert m["n"] == 3 and m["min"] <= m["median"] <= m["max"]
    assert [ph["name"] for ph in phases.items][:1] == ["gpu0:matmul_fp32_tflops"]

    doc = json.loads(json.dumps(_document(entry, p)))
    schema = json.loads(resources.files("gpu_mark").joinpath("result.schema.json").read_text())
    jsonschema.validate(doc, schema)
    text = cli.format_summary(doc)
    assert "matmul_fp32_tflops" in text and "All-GPU pass skipped" in text


def test_schema_rejects_missing_provider():
    p = MethodParams().with_values(**SMALL)
    entry = runner.per_gpu_pass(p, torch.device("cpu"), runner.Phases(), 0)
    doc = json.loads(json.dumps(_document(entry, p)))
    doc["run"]["provider"] = ""
    schema = json.loads(resources.files("gpu_mark").joinpath("result.schema.json").read_text())
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(doc, schema)
