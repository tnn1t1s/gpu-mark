from gpu_mark import inventory, telemetry


def test_parse_telemetry_line():
    row = telemetry.parse_line("0, 312.45, 400.00, 61, 1410, 1593, 100, 0x0000000000000000")
    assert row == {"index": 0, "power_w": 312.45, "power_limit_w": 400.0, "temp_c": 61.0,
                   "sm_clock_mhz": 1410.0, "mem_clock_mhz": 1593.0, "util_pct": 100.0,
                   "throttle_reasons": "0x0000000000000000"}


def test_parse_telemetry_handles_na_and_garbage():
    row = telemetry.parse_line("1, [N/A], 400.00, 40, 210, 1593, 0, [Not Supported]")
    assert row["power_w"] is None and row["throttle_reasons"] is None
    assert telemetry.parse_line("not, csv") is None


def test_summarize_phase_filters_by_gpu_and_time():
    s = [
        {"t": 1.0, "index": 0, "power_w": 100.0, "power_limit_w": 400.0, "temp_c": 40.0,
         "sm_clock_mhz": 1400.0, "mem_clock_mhz": 1593.0, "throttle_reasons": "0x1"},
        {"t": 2.0, "index": 0, "power_w": 300.0, "power_limit_w": 400.0, "temp_c": 60.0,
         "sm_clock_mhz": 1200.0, "mem_clock_mhz": 1593.0, "throttle_reasons": "0x4"},
        {"t": 2.0, "index": 1, "power_w": 999.0, "power_limit_w": 400.0, "temp_c": 99.0,
         "sm_clock_mhz": 1.0, "mem_clock_mhz": 1.0, "throttle_reasons": "0x0"},
        {"t": 9.0, "index": 0, "power_w": 999.0, "power_limit_w": 400.0, "temp_c": 99.0,
         "sm_clock_mhz": 1.0, "mem_clock_mhz": 1.0, "throttle_reasons": "0x0"},
    ]
    out = telemetry.summarize_phase(s, 0, 0.5, 2.5)
    assert out["n_samples"] == 2
    assert out["power_w_mean"] == 200.0 and out["power_w_max"] == 300.0
    assert out["temp_c_max"] == 60.0 and out["sm_clock_mhz_min"] == 1200.0
    assert out["throttle_reasons_seen"] == ["0x1", "0x4"]
    assert telemetry.summarize_phase(s, 5, 0, 10)["power_w_mean"] is None


def test_parse_gpu_csv():
    n = len(inventory.GPU_FIELDS)
    vals = ["0", "NVIDIA A100-SXM4-80GB", "GPU-abc", "[N/A]"] + ["x"] * (n - 4)
    gpus = inventory.parse_gpu_csv(", ".join(vals) + "\nshort, line\n")
    assert len(gpus) == 1
    assert gpus[0]["name"] == "NVIDIA A100-SXM4-80GB" and gpus[0]["serial"] is None
