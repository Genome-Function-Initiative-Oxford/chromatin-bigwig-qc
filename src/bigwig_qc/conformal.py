"""Split conformal prediction intervals for the quality-score regressor.

Standard split conformal regression (Lei et al. 2018-style): given a model
already fit on a disjoint training set, and a held-out calibration set, the
absolute-residual nonconformity score's empirical (1-alpha)(n+1)/n quantile
gives a margin that, added/subtracted around any new point prediction,
guarantees marginal coverage >= 1-alpha under exchangeability of calibration
and test points - no distributional assumptions on the residuals themselves.
"""
import numpy as np
import pandas as pd


def calibrate(y_calib: pd.Series, y_pred_calib: np.ndarray, alpha: float = 0.1) -> float:
    """Return the nonconformity-score quantile (interval half-width) for 1-alpha coverage."""
    residuals = np.abs(np.asarray(y_calib) - np.asarray(y_pred_calib))
    n = residuals.shape[0]
    # Finite-sample-corrected quantile level (Lei et al. 2018, eq. 2.1) - using
    # the uncorrected (1-alpha) quantile would UNDER-cover in finite samples.
    level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    return float(np.quantile(residuals, level, method="higher"))


def predict_interval(y_pred: np.ndarray, half_width: float) -> pd.DataFrame:
    """Return a DataFrame of (point, lower, upper) for every prediction in y_pred."""
    y_pred = np.asarray(y_pred)
    return pd.DataFrame(
        {
            "point_prediction": y_pred,
            "lower": y_pred - half_width,
            "upper": y_pred + half_width,
        }
    )


def empirical_coverage(y_true: pd.Series, intervals: pd.DataFrame) -> float:
    """Fraction of y_true that fall within [lower, upper] - the metric to validate against 1-alpha."""
    y_true = np.asarray(y_true)
    covered = (y_true >= intervals.lower.to_numpy()) & (y_true <= intervals.upper.to_numpy())
    return float(covered.mean())
