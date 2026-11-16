import json

import pytest

from gpu_mark.params import NOT_IMPLEMENTED, MethodParams


def test_every_parameter_has_unit_kind_and_reason():
    for name, p in MethodParams().as_dict().items():
        assert p["unit"], name
        assert p["kind"] in ("source", "default"), name
        assert len(p["reason"]) > 20, name
    json.dumps(MethodParams().as_dict())


def test_source_values_match_published_method():
    p = MethodParams()
    assert p.value("matmul_n") == 16384
    assert p.value("d2d_copy_bytes") == 1 << 30
    assert p.value("fp32_tf32_allowed") is False
    assert p.value("repetitions") >= 3


def test_override_records_kind_and_old_value():
    p = MethodParams().with_values(repetitions=7)
    d = p.as_dict()["repetitions"]
    assert d["value"] == 7 and d["kind"] == "override" and "5" in d["reason"]
    assert MethodParams().value("repetitions") == 5


def test_unknown_override_fails():
    with pytest.raises(KeyError):
        MethodParams().with_values(no_such_param=1)


def test_not_implemented_lists_l2_and_nccl():
    names = {x["name"] for x in NOT_IMPLEMENTED}
    assert {"l2_bandwidth", "nccl_allreduce_broadcast", "peer_tier_grading"} <= names
