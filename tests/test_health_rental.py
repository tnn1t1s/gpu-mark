import argparse

import pytest

from gpu_mark import health, rental

UUID = "GPU-abc"


def _gpu(**kw):
    g = {"uuid": UUID, "pcie_gen_current": "1", "pcie_gen_max": "4", "pcie_width_current": "16",
         "pcie_width_max": "16", "power_limit_w": "400.00", "power_default_limit_w": "400.00",
         "ecc_mode": "Enabled", "mig_mode": "Disabled"}
    g.update(kw)
    return g


def _row(**kw):
    r = {"index": "0", "uuid": UUID, "ecc_uncorrected_volatile": "0", "ecc_uncorrected_aggregate": "0",
         "ecc_corrected_volatile": "3", "retired_pages_pending": None, "remapped_rows_pending": "0",
         "remapped_rows_failure": "No", "throttle_reasons_at_start": "0x0000000000000001"}
    r.update(kw)
    return r


def _by_check(findings):
    return {f["check"]: f for f in findings}


def test_healthy_gpu_has_no_flags():
    f = _by_check(health.check_gpu(_gpu(), _row(), []))
    assert {k for k, v in f.items() if v["status"] == "flag"} == set()
    # idle gen 1 of 4 is power management, not a fault
    assert f["pcie_gen"]["status"] == "not_checked"
    assert f["memory_errors"]["status"] == "ok"


@pytest.mark.parametrize("gpu_kw,row_kw,check", [
    ({"pcie_width_current": "8"}, {}, "pcie_width"),
    ({"power_limit_w": "300.00"}, {}, "power_limit"),
    ({"ecc_mode": "Disabled"}, {}, "ecc_mode"),
    ({"mig_mode": "Enabled"}, {}, "mig_mode"),
    ({}, {"ecc_uncorrected_aggregate": "2"}, "memory_errors"),
    ({}, {"remapped_rows_pending": "1"}, "memory_errors"),
    ({}, {"remapped_rows_failure": "Yes"}, "memory_errors"),
    ({}, {"throttle_reasons_at_start": "0x0000000000000041"}, "hw_throttle_at_start"),
])
def test_each_fault_flags_its_check(gpu_kw, row_kw, check):
    f = _by_check(health.check_gpu(_gpu(**gpu_kw), _row(**row_kw), []))
    assert f[check]["status"] == "flag"


def test_software_power_cap_at_start_is_not_a_hardware_throttle():
    f = _by_check(health.check_gpu(_gpu(), _row(throttle_reasons_at_start="0x0000000000000004"), []))
    assert f["hw_throttle_at_start"]["status"] == "ok"


def test_other_process_on_this_gpu_flags_only_this_gpu():
    procs = [{"gpu_uuid": UUID, "pid": "7", "process_name": "python", "used_memory_mib": "900"},
             {"gpu_uuid": "GPU-other", "pid": "8", "process_name": "x", "used_memory_mib": "1"}]
    f = _by_check(health.check_gpu(_gpu(), _row(), procs))
    assert f["other_compute_processes"]["status"] == "flag"
    assert f["other_compute_processes"]["observed"] == [{"pid": "7", "process_name": "python"}]


def test_missing_health_row_is_unavailable_not_ok():
    f = _by_check(health.check_gpu(_gpu(), None, []))
    assert f["memory_errors"]["status"] == "unavailable"
    assert f["hw_throttle_at_start"]["status"] == "unavailable"


def test_unreadable_kernel_log_is_unavailable_not_zero_xids():
    assert health.check_xid({"lines": None, "error": "dmesg exited 1"})["status"] == "unavailable"
    assert health.check_xid({"lines": [], "error": None})["status"] == "ok"
    assert health.check_xid({"lines": ["NVRM: Xid (PCI:0000:00:05): 79"], "error": None})["status"] == "flag"


def test_evaluate_matches_rows_by_uuid_and_lists_flags():
    out = health.evaluate([_gpu(ecc_mode="Disabled")], [], {"rows": [_row()], "error": None},
                          {"lines": [], "error": None})
    assert out["flagged"] == ["ecc_mode"]


def test_parse_health_csv():
    n = len(health.HEALTH_FIELDS)
    rows = health.parse_health_csv(", ".join(["0", UUID] + ["[N/A]"] * (n - 2)) + "\nshort\n")
    assert len(rows) == 1 and rows[0]["uuid"] == UUID and rows[0]["remapped_rows_pending"] is None


def _rental_args(argv):
    p = argparse.ArgumentParser()
    rental.add_arguments(p)
    return rental.from_args(p.parse_args(argv))


def test_rental_records_not_given_fields_as_null():
    r = _rental_args(["--region", "US-TX-3"])
    assert r["region"] == "US-TX-3" and r["price_usd_per_hour"] is None
    assert "price_usd_per_hour" in r["not_given"] and "region" not in r["not_given"]


def test_rental_price_needs_a_source():
    with pytest.raises(SystemExit):
        _rental_args(["--price-usd-per-hour", "1.19"])
    r = _rental_args(["--price-usd-per-hour", "1.19", "--price-source", "runpod console"])
    assert r["price_usd_per_hour"] == 1.19


def test_rental_start_needs_an_offset():
    with pytest.raises(SystemExit):
        _rental_args(["--rental-started-at", "2026-11-16T18:00:00"])
    assert _rental_args(["--rental-started-at", "2026-11-16T18:00:00+00:00"])["rental_started_at"]
