"""Baseline models and feature-group ablations, compared against the XGBoost
quality-score model (E3).

Baselines, per todo.tex: mean predictor; linear **and** Poisson regression on
n_peaks only (the target, n_ctcf + n_promoters, is a count - Poisson is the
statistically appropriate GLM for it, not just an alternative phrasing of
"linear"); linear regression on percentage_coverage only; ElasticNet; Random
Forest; LightGBM; MLP - plus the tuned XGBoost model itself as the point of
comparison. Ablations: drop one bigwig_qc.features.FEATURE_GROUPS group at a
time, keeping the model fixed as XGBoost.

Every baseline/ablation is evaluated on exactly the same repeated
per-collection stratified folds (bigwig_qc.evaluation.generate_stratified_folds)
so comparisons are paired - required for corrected_resampled_ttest to be valid.

MACS3 features (todo.tex's third ablation, substituting MACS3 for LanceOtron
peak calls) are not implemented here: this repo has no MACS3 peak calls for
any track, so there is nothing to substitute in. See
experiments/e3_baselines_ablations/README.md.
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy import stats as scipy_stats
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, LinearRegression, PoissonRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .evaluation import compute_metrics, generate_stratified_folds
from .features import FEATURE_COLUMNS, FEATURE_GROUPS
from .model import RANDOM_STATE, regression_target

try:
    from lightgbm import LGBMRegressor

    _HAS_LIGHTGBM = True
except ImportError:  # pragma: no cover - exercised only when the optional dep is missing
    _HAS_LIGHTGBM = False


def _all_feature_columns() -> list:
    return list(FEATURE_COLUMNS)


# name -> (feature_columns, zero-arg estimator factory). A factory (not a
# shared instance) so every fold gets a freshly-initialised model.
BASELINE_SPECS = {
    "mean_predictor": (_all_feature_columns(), lambda: DummyRegressor(strategy="mean")),
    "linear_n_peaks_only": (["n_peaks"], lambda: LinearRegression()),
    "poisson_n_peaks_only": (["n_peaks"], lambda: make_pipeline(StandardScaler(), PoissonRegressor())),
    "linear_coverage_only": (["percentage_coverage"], lambda: LinearRegression()),
    "elastic_net": (_all_feature_columns(), lambda: make_pipeline(StandardScaler(), ElasticNet(random_state=RANDOM_STATE))),
    "random_forest": (_all_feature_columns(), lambda: RandomForestRegressor(random_state=RANDOM_STATE)),
    "mlp": (
        _all_feature_columns(),
        lambda: make_pipeline(StandardScaler(), MLPRegressor(random_state=RANDOM_STATE, max_iter=2000)),
    ),
}
if _HAS_LIGHTGBM:
    BASELINE_SPECS["lightgbm"] = (_all_feature_columns(), lambda: LGBMRegressor(random_state=RANDOM_STATE, verbosity=-1))


def evaluate_baselines(
    data: pd.DataFrame,
    xgboost_params: dict,
    baseline_specs: dict = None,
    xgboost_feature_columns: list = None,
    test_size: float = 0.2,
    n_repeats: int = 10,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Evaluate every baseline + the tuned XGBoost model on the same per-repeat folds.

    xgboost_feature_columns defaults to bigwig_qc.features.FEATURE_COLUMNS -
    overridable so this can be unit-tested against a synthetic feature set.

    Returns one row per (model, repeat) with R2/RMSE/MAE/Spearman rho, n_train, n_test.
    """
    baseline_specs = dict(baseline_specs) if baseline_specs is not None else dict(BASELINE_SPECS)
    xgboost_feature_columns = (
        list(xgboost_feature_columns) if xgboost_feature_columns is not None else _all_feature_columns()
    )
    y_all = regression_target(data)

    rows = []
    for repeat, train_mask, test_mask in generate_stratified_folds(data, test_size, n_repeats, random_state):
        y_train, y_test = y_all[train_mask], y_all[test_mask]
        n_train, n_test = int(train_mask.sum()), int(test_mask.sum())

        for model_name, (feature_columns, build_estimator) in baseline_specs.items():
            estimator = build_estimator()
            estimator.fit(data.loc[train_mask, feature_columns], y_train)
            y_pred = estimator.predict(data.loc[test_mask, feature_columns])
            rows.append(
                {"model": model_name, "repeat": repeat, "n_train": n_train, "n_test": n_test, **compute_metrics(y_test, y_pred)}
            )

        xgb_model = xgb.XGBRegressor(**xgboost_params)
        xgb_model.fit(data.loc[train_mask, xgboost_feature_columns], y_train)
        y_pred = xgb_model.predict(data.loc[test_mask, xgboost_feature_columns])
        rows.append(
            {"model": "xgboost", "repeat": repeat, "n_train": n_train, "n_test": n_test, **compute_metrics(y_test, y_pred)}
        )

    return pd.DataFrame(rows)


def evaluate_feature_ablations(
    data: pd.DataFrame,
    xgboost_params: dict,
    feature_groups: dict = None,
    feature_columns: list = None,
    test_size: float = 0.2,
    n_repeats: int = 10,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Drop one feature_groups group at a time, keeping the model fixed as
    XGBoost (same fixed hyperparameters as the full model - a cleaner read on
    each group's contribution than re-tuning per ablation).

    feature_columns defaults to bigwig_qc.features.FEATURE_COLUMNS (the "full
    model"'s feature set) - overridable so this can be unit-tested against a
    synthetic feature set without touching the real FEATURE_COLUMNS.

    Returns one row per (ablation, repeat), where ablation="full_model" is the
    no-group-dropped baseline and ablation="without_<group>" drops that group.
    """
    feature_groups = dict(feature_groups) if feature_groups is not None else dict(FEATURE_GROUPS)
    feature_columns = list(feature_columns) if feature_columns is not None else _all_feature_columns()
    y_all = regression_target(data)

    ablations = {"full_model": feature_columns}
    for group_name, group_columns in feature_groups.items():
        ablations[f"without_{group_name}"] = [c for c in feature_columns if c not in group_columns]

    rows = []
    for repeat, train_mask, test_mask in generate_stratified_folds(data, test_size, n_repeats, random_state):
        y_train, y_test = y_all[train_mask], y_all[test_mask]
        n_train, n_test = int(train_mask.sum()), int(test_mask.sum())

        for ablation_name, feature_columns in ablations.items():
            model = xgb.XGBRegressor(**xgboost_params)
            model.fit(data.loc[train_mask, feature_columns], y_train)
            y_pred = model.predict(data.loc[test_mask, feature_columns])
            rows.append(
                {
                    "ablation": ablation_name,
                    "repeat": repeat,
                    "n_train": n_train,
                    "n_test": n_test,
                    **compute_metrics(y_test, y_pred),
                }
            )

    return pd.DataFrame(rows)


def corrected_resampled_ttest(metric_a: np.ndarray, metric_b: np.ndarray, n_train: int, n_test: int) -> dict:
    """Nadeau & Bengio (2003) corrected resampled t-test for two repeated-CV metric series.

    Corrects the naive paired t-test's underestimated variance, which arises
    because repeated resampled train/test splits are not independent (unlike
    the IID assumption behind an ordinary paired t-test) - the standard
    correction for comparing models across repeated CV folds.
    """
    diffs = np.asarray(metric_a, dtype=float) - np.asarray(metric_b, dtype=float)
    n = len(diffs)
    mean_diff = diffs.mean()
    var_diff = diffs.var(ddof=1) if n > 1 else 0.0

    correction = (1.0 / n) + (n_test / n_train)
    denom = np.sqrt(correction * var_diff)

    if denom == 0:
        t_stat = np.inf if mean_diff > 0 else (-np.inf if mean_diff < 0 else 0.0)
    else:
        t_stat = mean_diff / denom

    df = n - 1
    p_value = 2 * (1 - scipy_stats.t.cdf(abs(t_stat), df)) if np.isfinite(t_stat) else 0.0
    return {"mean_diff": mean_diff, "t_stat": t_stat, "df": df, "p_value": p_value}


def compare_models_to_reference(
    results: pd.DataFrame, reference_model: str = "xgboost", metric: str = "r2", group_column: str = "model"
) -> pd.DataFrame:
    """Corrected resampled t-test of every other model/ablation against reference_model, on `metric`."""
    reference = results[results[group_column] == reference_model].sort_values("repeat")
    n_train = int(reference.n_train.iloc[0])
    n_test = int(reference.n_test.iloc[0])

    rows = []
    for name, group in results[results[group_column] != reference_model].groupby(group_column):
        group = group.sort_values("repeat")
        test_result = corrected_resampled_ttest(
            reference[metric].to_numpy(), group[metric].to_numpy(), n_train=n_train, n_test=n_test
        )
        rows.append({group_column: name, "metric": metric, "reference": reference_model, **test_result})
    return pd.DataFrame(rows)
