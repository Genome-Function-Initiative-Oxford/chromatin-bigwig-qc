import numpy as np

from bigwig_qc.conformal import calibrate, empirical_coverage, predict_interval


def test_calibrate_returns_zero_for_perfect_predictions():
    y_calib = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    half_width = calibrate(y_calib, y_calib, alpha=0.1)
    assert half_width == 0.0


def test_calibrate_covers_the_largest_residual_at_alpha_zero_point_something():
    # With n=4 calibration points and alpha=0.2, ceil((4+1)*0.8)/4 = 1.0 -> the
    # 100th percentile (max) residual is used, guaranteeing full calibration-set coverage.
    y_calib = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred_calib = np.array([1.0, 2.0, 3.0, 10.0])  # one big residual of 6.0
    half_width = calibrate(y_calib, y_pred_calib, alpha=0.2)
    assert half_width == 6.0


def test_predict_interval_centers_on_point_prediction():
    y_pred = np.array([10.0, 20.0, 30.0])
    intervals = predict_interval(y_pred, half_width=5.0)
    assert list(intervals.point_prediction) == [10.0, 20.0, 30.0]
    assert list(intervals.lower) == [5.0, 15.0, 25.0]
    assert list(intervals.upper) == [15.0, 25.0, 35.0]


def test_empirical_coverage_counts_fraction_within_interval():
    y_true = np.array([10.0, 20.0, 100.0])  # third point is way outside its interval
    intervals = predict_interval(np.array([10.0, 20.0, 30.0]), half_width=2.0)
    coverage = empirical_coverage(y_true, intervals)
    assert coverage == 2 / 3


def test_calibrate_produces_interval_achieving_at_least_nominal_coverage_in_expectation():
    # Larger-sample sanity check: calibrate on one synthetic sample, verify
    # coverage on an independent, identically-distributed sample is close to
    # the 1-alpha target (exact equality isn't expected, just the right ballpark).
    rng = np.random.default_rng(0)
    y_calib = rng.normal(size=2000)
    y_pred_calib = np.zeros_like(y_calib)
    half_width = calibrate(y_calib, y_pred_calib, alpha=0.1)

    y_test = rng.normal(size=5000)
    y_pred_test = np.zeros_like(y_test)
    intervals = predict_interval(y_pred_test, half_width)
    coverage = empirical_coverage(y_test, intervals)
    assert 0.85 <= coverage <= 0.95
