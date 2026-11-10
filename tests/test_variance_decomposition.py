import pytest

from bigwig_qc.variance_decomposition import intraclass_correlation


def test_intraclass_correlation_equal_variances_gives_half():
    assert intraclass_correlation(100.0, 100.0) == pytest.approx(0.5)


def test_intraclass_correlation_zero_group_variance_gives_zero():
    assert intraclass_correlation(0.0, 100.0) == pytest.approx(0.0)


def test_intraclass_correlation_zero_residual_variance_gives_one():
    assert intraclass_correlation(100.0, 0.0) == pytest.approx(1.0)


def test_intraclass_correlation_matches_e7_real_result():
    # Regression check against the actual E7 result (results/e7_mixed_effects.tsv):
    # group_variance=1734236.7, residual_variance=1749718.5 -> ICC ~ 0.498.
    icc = intraclass_correlation(1734236.704, 1749718.4903)
    assert icc == pytest.approx(0.498, abs=0.001)
