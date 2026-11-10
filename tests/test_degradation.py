import numpy as np
import pytest

from bigwig_qc.degradation import inject_uniform_noise, thin_signal


def test_thin_signal_fraction_one_is_identity():
    rng = np.random.default_rng(0)
    values = np.array([0.0, 5.0, 100.0, 3.7])
    result = thin_signal(values, fraction=1.0, rng=rng)
    assert np.array_equal(result, np.round(values))


def test_thin_signal_fraction_zero_is_all_zero():
    rng = np.random.default_rng(0)
    values = np.array([10.0, 20.0, 30.0])
    result = thin_signal(values, fraction=0.0, rng=rng)
    assert np.all(result == 0.0)


def test_thin_signal_reduces_expected_total_on_average():
    rng = np.random.default_rng(0)
    values = np.full(10_000, 20.0)
    result = thin_signal(values, fraction=0.1, rng=rng)
    # E[Binomial(20, 0.1)] = 2 per bin; allow generous tolerance for sampling noise.
    assert 1.8 <= result.mean() <= 2.2


def test_thin_signal_rejects_invalid_fraction():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        thin_signal(np.array([1.0]), fraction=1.5, rng=rng)


def test_inject_uniform_noise_fraction_zero_is_identity():
    rng = np.random.default_rng(0)
    values = np.array([0.0, 5.0, 100.0])
    result = inject_uniform_noise(values, fraction=0.0, rng=rng)
    assert np.array_equal(result, values)


def test_inject_uniform_noise_conserves_total_signal_on_average():
    rng = np.random.default_rng(0)
    values = np.concatenate([np.full(5_000, 10.0), np.zeros(5_000)])
    result = inject_uniform_noise(values, fraction=0.5, rng=rng)
    # Signal mass is conserved in expectation (moved, not removed).
    assert result.sum() == pytest.approx(values.sum(), rel=0.05)


def test_inject_uniform_noise_spreads_signal_into_previously_empty_bins():
    rng = np.random.default_rng(0)
    values = np.concatenate([np.full(100, 1_000.0), np.zeros(9_900)])
    result = inject_uniform_noise(values, fraction=0.8, rng=rng)
    previously_empty = result[100:]
    assert previously_empty.mean() > 0


def test_inject_uniform_noise_rejects_invalid_fraction():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        inject_uniform_noise(np.array([1.0]), fraction=-0.1, rng=rng)
