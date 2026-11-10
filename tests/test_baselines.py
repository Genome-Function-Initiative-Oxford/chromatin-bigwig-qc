import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression

from bigwig_qc.baselines import (
    BASELINE_SPECS,
    compare_models_to_reference,
    corrected_resampled_ttest,
    evaluate_baselines,
    evaluate_feature_ablations,
)
from bigwig_qc.features import FEATURE_COLUMNS, FEATURE_GROUPS

CHEAP_XGB_PARAMS = {"objective": "reg:squarederror", "n_estimators": 5, "max_depth": 2, "random_state": 0}

SYNTHETIC_FEATURE_GROUPS = {"group_a": ("feature_1",), "group_b": ("feature_2",)}
SYNTHETIC_FEATURE_COLUMNS = ["feature_1", "feature_2"]

TINY_BASELINE_SPECS = {
    "mean_predictor": (SYNTHETIC_FEATURE_COLUMNS, lambda: DummyRegressor(strategy="mean")),
    "linear_feature_1_only": (["feature_1"], lambda: LinearRegression()),
}


def _synthetic_data(n_per_collection=20, n_collections=4, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(n_collections):
        for i in range(n_per_collection):
            feature_1 = rng.normal()
            feature_2 = rng.normal()
            rows.append(
                {
                    "collection_name": f"collection_{c}",
                    "experiment_name": f"c{c}_e{i}",
                    "n_promoters": 100 + 10 * feature_1,
                    "n_ctcf": 50 + 5 * feature_2,
                    "feature_1": feature_1,
                    "feature_2": feature_2,
                }
            )
    return pd.DataFrame(rows)


def test_feature_groups_cover_every_feature_column_exactly_once():
    covered = [c for group in FEATURE_GROUPS.values() for c in group]
    assert sorted(covered) == sorted(FEATURE_COLUMNS)
    assert len(covered) == len(set(covered))


def test_evaluate_baselines_includes_every_model_and_xgboost():
    data = _synthetic_data()
    results = evaluate_baselines(
        data,
        CHEAP_XGB_PARAMS,
        baseline_specs=TINY_BASELINE_SPECS,
        xgboost_feature_columns=SYNTHETIC_FEATURE_COLUMNS,
        n_repeats=2,
    )
    assert set(results.model) == {"mean_predictor", "linear_feature_1_only", "xgboost"}
    assert set(results.repeat) == {0, 1}


def test_evaluate_feature_ablations_includes_full_model_and_each_group_dropped():
    data = _synthetic_data()
    results = evaluate_feature_ablations(
        data,
        CHEAP_XGB_PARAMS,
        feature_groups=SYNTHETIC_FEATURE_GROUPS,
        feature_columns=SYNTHETIC_FEATURE_COLUMNS,
        n_repeats=2,
    )
    assert set(results.ablation) == {"full_model", "without_group_a", "without_group_b"}


def test_corrected_resampled_ttest_identical_series_gives_zero_statistic():
    metrics = np.array([0.8, 0.85, 0.9, 0.82])
    result = corrected_resampled_ttest(metrics, metrics, n_train=400, n_test=100)
    assert result["mean_diff"] == pytest.approx(0.0)
    assert result["t_stat"] == pytest.approx(0.0)
    assert result["p_value"] == pytest.approx(1.0)


def test_corrected_resampled_ttest_detects_consistent_difference():
    a = np.array([0.9, 0.92, 0.88, 0.91, 0.89])
    b = np.array([0.5, 0.52, 0.48, 0.51, 0.49])
    result = corrected_resampled_ttest(a, b, n_train=400, n_test=100)
    assert result["mean_diff"] > 0
    assert result["p_value"] < 0.05


def test_compare_models_to_reference_one_row_per_non_reference_model():
    data = _synthetic_data()
    results = evaluate_baselines(
        data,
        CHEAP_XGB_PARAMS,
        baseline_specs=TINY_BASELINE_SPECS,
        xgboost_feature_columns=SYNTHETIC_FEATURE_COLUMNS,
        n_repeats=3,
    )
    comparison = compare_models_to_reference(results, reference_model="xgboost", metric="r2")
    assert set(comparison.model) == {"mean_predictor", "linear_feature_1_only"}
    assert all(col in comparison.columns for col in ("t_stat", "p_value", "mean_diff"))


def test_baseline_specs_includes_poisson_n_peaks_only():
    # todo.tex asked for "linear/Poisson on n_peaks only" - Poisson is the
    # statistically appropriate GLM for a count target, not just an
    # alternative phrasing of the linear baseline, so both must be present.
    assert "linear_n_peaks_only" in BASELINE_SPECS
    assert "poisson_n_peaks_only" in BASELINE_SPECS

    feature_columns, build_estimator = BASELINE_SPECS["poisson_n_peaks_only"]
    assert feature_columns == ["n_peaks"]

    rng = np.random.default_rng(0)
    n_peaks = rng.integers(100, 100_000, size=200).astype(float)
    counts = rng.poisson(lam=n_peaks / 50.0).astype(float)
    X = pd.DataFrame({"n_peaks": n_peaks})

    model = build_estimator()
    model.fit(X, counts)
    predictions = model.predict(X)
    assert (predictions >= 0).all()  # Poisson/log-link predictions are never negative
