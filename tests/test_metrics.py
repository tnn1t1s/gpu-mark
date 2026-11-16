import pytest

from gpu_mark import metrics


def test_matmul_flops_counts_two_per_multiply_add():
    assert metrics.matmul_flops(2) == 16
    assert metrics.matmul_flops(16384) == 2 * 16384**3


def test_tflops():
    # 2 * 1000^3 FLOPs, 10 iterations, 1 s -> 0.02 TFLOPS
    assert metrics.tflops(1000, 10, 1.0) == pytest.approx(0.02)


def test_bandwidth_decimal_gb():
    assert metrics.bandwidth_gbps(10**9, 4, 2.0, 10**9) == pytest.approx(2.0)


def test_d2d_read_plus_write_is_twice_each_byte_once():
    bw = metrics.d2d_bandwidths(1 << 30, 20, 0.02, 10**9)
    assert bw["read_plus_write"] == pytest.approx(2 * bw["each_byte_once"])
    assert bw["each_byte_once"] == pytest.approx((1 << 30) * 20 / 0.02 / 1e9)


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_nonpositive_time_fails(bad):
    with pytest.raises(ValueError):
        metrics.tflops(10, 1, bad)
    with pytest.raises(ValueError):
        metrics.bandwidth_gbps(10, 1, bad, 10**9)


def test_summarize():
    s = metrics.summarize([3.0, 1.0, 2.0, 10.0, 4.0])
    assert (s["median"], s["min"], s["max"], s["n"]) == (3.0, 1.0, 10.0, 5)
    with pytest.raises(ValueError):
        metrics.summarize([])
