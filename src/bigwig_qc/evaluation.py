"""Generalisation evaluation for the quality-score model (E2).

Three complementary evaluation schemes:

  - leave_one_collection_out: train on 10 collections, predict the 11th, for
    each of the 11 collections in turn - a test of generalisation to an
    entirely unseen dataset/processing route (predicted vs. observed, in the
    held-out-collection setting). This conflates two
    things: whether the covariates predict genuine track-level quality, and
    whether the model can extrapolate across the batch/processing-depth
    differences between collections (see repeated collection analysis in
    results/e2_excl_pancreatic_pbmc - collections differ systematically in
    total_bigwig_sum, which barely overlaps between collections).
  - repeated_group_kfold: collections are repeatedly shuffled and split into
    n_splits folds (default 10 repeats x 5 folds) - same extrapolation
    question as leave_one_collection_out, at coarser/finer group sizes.
  - stratified_track_holdout: a few tracks are held out from *every*
    collection each repeat, rather than whole collections. The model has
    already seen each collection's processing/depth baseline during
    training, so this isolates genuine track-to-track quality signal within
    a shared processing context from the cross-collection extrapolation
    problem above - a fairer test of "do the covariates predict quality" on
    its own.

All three use FIXED XGBoost hyperparameters (e.g. from the already-trained
results/model.json) rather than re-running bigwig_qc.model's 3-stage grid
search inside every fold - nested-CV re-tuning per fold is comprehensively
more expensive, and is a separate, larger question (how sensitive is the
*tuning procedure* to the data split) from what E2 asks (how well does *the*
model generalise).
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy import stats
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .model import RANDOM_STATE, regression_target

METRIC_COLUMNS = ("r2", "rmse", "mae", "spearman_rho")

# Below this many tracks, per-collection R2 is fragile regardless of
# evaluation scheme: a narrow true-quality range (small denominator in R2)
# means one so-so prediction swings it hugely negative. Confirmed empirically
# - CAD (n=10), H1_hESCs (n=4) and super_PBMC (n=19) all stayed R2-negative
# even under stratified_track_holdout, where their own batch/collection was
# already represented in training (see experiments/e2_leave_one_collection_out
# /README.md). Report pooled/aggregate metrics for collections below this
# threshold rather than trusting their individual R2.
LOW_N_THRESHOLD = 25


def _fit_fixed_params(X_train: pd.DataFrame, y_train: pd.Series, xgb_params: dict) -> xgb.XGBRegressor:
    model = xgb.XGBRegressor(**xgb_params)
    model.fit(X_train, y_train)
    return model


def flag_low_n(metrics_df: pd.DataFrame, n_column: str, threshold: int = LOW_N_THRESHOLD) -> pd.DataFrame:
    """Add a `low_n` column marking rows whose sample count is below threshold."""
    metrics_df = metrics_df.copy()
    metrics_df["low_n"] = metrics_df[n_column] < threshold
    return metrics_df


def compute_metrics(y_true, y_pred) -> dict:
    return {
        "r2": r2_score(y_true, y_pred),
        # np.sqrt(...) rather than mean_squared_error(..., squared=False): the
        # squared= kwarg was removed in newer scikit-learn.
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mae": mean_absolute_error(y_true, y_pred),
        "spearman_rho": spearmanr(y_true, y_pred).correlation,
    }


def leave_one_collection_out(data: pd.DataFrame, feature_columns: list, xgb_params: dict) -> pd.DataFrame:
    """Train on all but one collection, predict it, for every collection in turn.

    Returns one row per track with its held-out prediction.
    """
    y = regression_target(data)
    X = data[feature_columns]

    rows = []
    for held_out in sorted(data.collection_name.unique()):
        test_mask = data.collection_name == held_out
        train_mask = ~test_mask

        model = _fit_fixed_params(X[train_mask], y[train_mask], xgb_params)
        y_pred = model.predict(X[test_mask])

        for experiment_name, true_value, predicted_value in zip(
            data.loc[test_mask, "experiment_name"], y[test_mask], y_pred
        ):
            rows.append(
                {
                    "collection_name": held_out,
                    "experiment_name": experiment_name,
                    "y_true": true_value,
                    "y_pred": predicted_value,
                }
            )
    return pd.DataFrame(rows)


def leave_one_collection_out_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Per-collection R2/RMSE/MAE/Spearman rho from leave_one_collection_out's output."""
    rows = []
    for collection_name, group in predictions.groupby("collection_name"):
        rows.append(
            {"collection_name": collection_name, "n_tracks": group.shape[0], **compute_metrics(group.y_true, group.y_pred)}
        )
    return flag_low_n(pd.DataFrame(rows), n_column="n_tracks")


def repeated_group_kfold(
    data: pd.DataFrame,
    feature_columns: list,
    xgb_params: dict,
    n_splits: int = 5,
    n_repeats: int = 10,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Repeated group k-fold CV, grouped by collection_name.

    Collections (not rows) are shuffled and split into n_splits folds each
    repeat, so every repeat uses a different random grouping of collections
    into folds - this is what makes it "repeated" (sklearn's GroupKFold is
    deterministic and wouldn't vary across repeats with so few groups).
    """
    y_all = regression_target(data)
    X_all = data[feature_columns]
    collections = np.array(sorted(data.collection_name.unique()))

    rng = np.random.default_rng(random_state)
    rows = []
    for repeat in range(n_repeats):
        shuffled = rng.permutation(collections)
        folds = np.array_split(shuffled, n_splits)

        for fold_index, held_out_collections in enumerate(folds):
            test_mask = data.collection_name.isin(held_out_collections)
            train_mask = ~test_mask

            model = _fit_fixed_params(X_all[train_mask], y_all[train_mask], xgb_params)
            y_pred = model.predict(X_all[test_mask])

            rows.append(
                {
                    "repeat": repeat,
                    "fold": fold_index,
                    "held_out_collections": ",".join(held_out_collections),
                    "n_test": int(test_mask.sum()),
                    **compute_metrics(y_all[test_mask], y_pred),
                }
            )
    return pd.DataFrame(rows)


def generate_stratified_folds(
    data: pd.DataFrame, test_size: float = 0.2, n_repeats: int = 10, random_state: int = RANDOM_STATE
):
    """Yield (repeat, train_mask, test_mask) for a repeated per-collection holdout.

    Every collection contributes tracks to both train and test in every
    repeat (unlike leave_one_collection_out). Factored out of
    stratified_track_holdout so other experiments (e.g. E3's baseline/ablation
    comparisons in bigwig_qc.baselines) can fit *multiple* models on exactly
    the same folds - required for a valid paired statistical test.
    """
    rng = np.random.default_rng(random_state)
    for repeat in range(n_repeats):
        test_indices = []
        for _, group in data.groupby("collection_name"):
            n = group.shape[0]
            n_test = max(1, round(n * test_size)) if n > 1 else 0
            shuffled = rng.permutation(group.index.to_numpy())
            test_indices.extend(shuffled[:n_test])

        test_mask = data.index.isin(test_indices)
        yield repeat, ~test_mask, test_mask


def stratified_track_holdout(
    data: pd.DataFrame,
    feature_columns: list,
    xgb_params: dict,
    test_size: float = 0.2,
    n_repeats: int = 10,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Repeated per-collection holdout: every collection contributes tracks to
    both train and test in every repeat (unlike leave_one_collection_out).

    Returns one row per (repeat, track) held-out prediction.
    """
    y_all = regression_target(data)
    X_all = data[feature_columns]

    rows = []
    for repeat, train_mask, test_mask in generate_stratified_folds(data, test_size, n_repeats, random_state):
        model = _fit_fixed_params(X_all[train_mask], y_all[train_mask], xgb_params)
        y_pred = model.predict(X_all[test_mask])

        for experiment_name, collection_name, true_value, predicted_value in zip(
            data.loc[test_mask, "experiment_name"],
            data.loc[test_mask, "collection_name"],
            y_all[test_mask],
            y_pred,
        ):
            rows.append(
                {
                    "repeat": repeat,
                    "collection_name": collection_name,
                    "experiment_name": experiment_name,
                    "y_true": true_value,
                    "y_pred": predicted_value,
                }
            )
    return pd.DataFrame(rows)


def stratified_track_holdout_overall_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    """One row of R2/RMSE/MAE/Spearman rho per repeat, pooling all collections' held-out tracks."""
    rows = []
    for repeat, group in predictions.groupby("repeat"):
        rows.append({"repeat": repeat, "n_test": group.shape[0], **compute_metrics(group.y_true, group.y_pred)})
    return pd.DataFrame(rows)


def stratified_track_holdout_per_collection_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    """R2/RMSE/MAE/Spearman rho per collection, pooling held-out tracks across all repeats.

    low_n is flagged on n_tracks (distinct experiment_name - the collection's
    actual size), not n_predictions (which inflates with n_repeats re-testing
    the same small set of tracks): sample-count fragility is a property of
    the underlying collection, not of how many times it was resampled.
    """
    rows = []
    for collection_name, group in predictions.groupby("collection_name"):
        rows.append(
            {
                "collection_name": collection_name,
                "n_predictions": group.shape[0],
                "n_tracks": group.experiment_name.nunique(),
                **compute_metrics(group.y_true, group.y_pred),
            }
        )
    return flag_low_n(pd.DataFrame(rows), n_column="n_tracks")


def summarise_with_ci(metrics_df: pd.DataFrame, metric_columns=METRIC_COLUMNS, ci: float = 0.95) -> pd.DataFrame:
    """Mean + ci%% CI (normal approximation via the standard error of the mean)."""
    z = stats.norm.ppf(0.5 + ci / 2)

    rows = []
    for column in metric_columns:
        values = metrics_df[column].dropna()
        mean = values.mean()
        sem = values.std(ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0.0
        rows.append({"metric": column, "mean": mean, "ci_low": mean - z * sem, "ci_high": mean + z * sem, "n": len(values)})
    return pd.DataFrame(rows)


def summarise_groups_with_ci(
    results: pd.DataFrame, group_column: str, metric_columns=METRIC_COLUMNS, ci: float = 0.95
) -> pd.DataFrame:
    """summarise_with_ci applied independently within each group (e.g. one row
    per model/ablation name x metric, each with its own mean/CI across its
    repeats) - the table format todo.tex asks for in E3 (R2/RMSE/MAE/rho with
    95% CIs for every model/ablation), not just a single pooled summary.
    """
    summaries = []
    for name, group in results.groupby(group_column):
        summary = summarise_with_ci(group, metric_columns=metric_columns, ci=ci)
        summary.insert(0, group_column, name)
        summaries.append(summary)
    return pd.concat(summaries, ignore_index=True)
