import numpy as np
import pyBigWig
import pytest

from bigwig_qc.bigwig_io import degrade_binned_signal, read_binned_mean_signal


def test_read_binned_mean_signal_computes_exact_per_bin_mean(tmp_path):
    # chr1 length 250 with bin_size=100: bins are [0,100), [100,200), [200,250)
    # (last bin partial - only 50 real bases, padded with zeros to average over).
    path = tmp_path / "toy.bw"
    bw = pyBigWig.open(str(path), "w")
    bw.addHeader([("chr1", 250)])
    bw.addEntries(["chr1", "chr1"], [0, 150], ends=[100, 200], values=[4.0, 8.0])
    bw.close()

    bw = pyBigWig.open(str(path))
    result = read_binned_mean_signal(bw, "chr1", 250, bin_size=100)
    bw.close()

    assert result.shape == (3,)
    assert result[0] == pytest.approx(4.0)  # bin [0,100): all 4.0
    assert result[1] == pytest.approx(4.0)  # bin [100,200): half 0.0 (gap), half 8.0 -> mean 4.0
    assert result[2] == pytest.approx(0.0)  # bin [200,250): no data, zero-padded to 100


def test_degrade_binned_signal_down_sample_matches_expected_mean():
    rng = np.random.default_rng(0)
    mean_signal = np.full(10_000, 2.0)  # bin total = mean * bin_size = 200
    result = degrade_binned_signal(mean_signal, mode="down_sample", level=0.1, bin_size=100, rng=rng)
    # Expected bin total after thinning: 200 * 0.1 = 20 -> mean = 20 / 100 = 0.2
    assert result.mean() == pytest.approx(0.2, rel=0.1)


def test_degrade_binned_signal_inject_noise_spreads_into_zero_bins():
    rng = np.random.default_rng(0)
    mean_signal = np.concatenate([np.full(100, 10.0), np.zeros(900)])
    result = degrade_binned_signal(mean_signal, mode="inject_noise", level=0.5, bin_size=100, rng=rng)
    assert result[100:].mean() > 0


def test_degrade_binned_signal_rejects_unknown_mode():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        degrade_binned_signal(np.array([1.0]), mode="bogus", level=0.5, bin_size=100, rng=rng)
