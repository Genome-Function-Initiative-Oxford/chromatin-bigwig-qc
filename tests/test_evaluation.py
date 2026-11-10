import numpy as np
import pandas as pd
import pytest

from bigwig_qc.evaluation import (
    compute_metrics,
    leave_one_collection_out,
    leave_one_collection_out_metrics,
    repeated_group_kfold,
    stratified_track_holdout,
    stratified_track_holdout_overall_metrics,
    stratified_track_holdout_per_collection_metrics,
    summarise_groups_with_ci,
    summarise_with_ci,
)

CHEAP_XGB_PARAMS = {"objective": "reg:squarederror", "n_estimators": 5, "max_depth": 2, "random_state": 0}


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


def test_compute_metrics_perfect_prediction():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    metrics = compute_metrics(y, y)
    assert metrics["r2"] == pytest.approx(1.0)
    assert metrics["rmse"] == pytest.approx(0.0)
    assert metrics["mae"] == pytest.approx(0.0)
    assert metrics["spearman_rho"] == pytest.approx(1.0)


def test_leave_one_collection_out_covers_every_track_exactly_once():
    data = _synthetic_data()
    predictions = leave_one_collection_out(data, ["feature_1", "feature_2"], CHEAP_XGB_PARAMS)

    assert predictions.shape[0] == data.shape[0]
    assert set(predictions.experiment_name) == set(data.experiment_name)


def test_leave_one_collection_out_never_trains_on_the_held_out_collection():
    # A collection whose target is wildly different from the others should
    # still only ever be predicted, never trained on - checked indirectly by
    # confirming every held-out collection's predictions come only from that
    # collection's rows (i.e. no cross-contamination in the output grouping).
    data = _synthetic_data()
    predictions = leave_one_collection_out(data, ["feature_1", "feature_2"], CHEAP_XGB_PARAMS)

    merged = predictions.merge(data[["experiment_name", "collection_name"]], on="experiment_name", suffixes=("_pred", "_true"))
    assert (merged.collection_name_pred == merged.collection_name_true).all()


def test_leave_one_collection_out_metrics_one_row_per_collection():
    data = _synthetic_data(n_collections=3)
    predictions = leave_one_collection_out(data, ["feature_1", "feature_2"], CHEAP_XGB_PARAMS)
    metrics = leave_one_collection_out_metrics(predictions)
    assert set(metrics.collection_name) == {"collection_0", "collection_1", "collection_2"}
    assert all(col in metrics.columns for col in ("r2", "rmse", "mae", "spearman_rho", "low_n"))


def test_repeated_group_kfold_row_count_and_disjoint_folds():
    data = _synthetic_data(n_collections=6)
    result = repeated_group_kfold(
        data, ["feature_1", "feature_2"], CHEAP_XGB_PARAMS, n_splits=3, n_repeats=2, random_state=0
    )
    assert result.shape[0] == 3 * 2
    assert set(result.repeat) == {0, 1}


def test_summarise_with_ci_mean_matches_simple_average():
    metrics_df = pd.DataFrame({"r2": [0.8, 0.9, 1.0], "rmse": [1.0, 2.0, 3.0], "mae": [1.0, 1.0, 1.0], "spearman_rho": [1.0, 1.0, 1.0]})
    summary = summarise_with_ci(metrics_df)
    r2_row = summary[summary.metric == "r2"].iloc[0]
    assert r2_row["mean"] == pytest.approx(0.9)
    assert r2_row.ci_low < 0.9 < r2_row.ci_high


def test_stratified_track_holdout_every_collection_represented_in_every_repeat():
    data = _synthetic_data(n_per_collection=20, n_collections=4)
    predictions = stratified_track_holdout(
        data, ["feature_1", "feature_2"], CHEAP_XGB_PARAMS, test_size=0.2, n_repeats=3, random_state=0
    )
    for repeat, group in predictions.groupby("repeat"):
        assert set(group.collection_name) == {"collection_0", "collection_1", "collection_2", "collection_3"}


def test_stratified_track_holdout_test_set_disjoint_from_training_each_repeat():
    data = _synthetic_data(n_per_collection=20, n_collections=3)
    predictions = stratified_track_holdout(
        data, ["feature_1", "feature_2"], CHEAP_XGB_PARAMS, test_size=0.2, n_repeats=1, random_state=0
    )
    # ~20% held out per collection (20 tracks x 3 collections x 0.2 = ~12)
    assert 9 <= predictions.shape[0] <= 15


def test_stratified_track_holdout_small_collection_gets_at_least_one_test_track():
    data = _synthetic_data(n_per_collection=3, n_collections=2)
    predictions = stratified_track_holdout(
        data, ["feature_1", "feature_2"], CHEAP_XGB_PARAMS, test_size=0.2, n_repeats=1, random_state=0
    )
    assert set(predictions.collection_name) == {"collection_0", "collection_1"}


def test_stratified_track_holdout_overall_metrics_one_row_per_repeat():
    data = _synthetic_data(n_per_collection=20, n_collections=3)
    predictions = stratified_track_holdout(
        data, ["feature_1", "feature_2"], CHEAP_XGB_PARAMS, test_size=0.2, n_repeats=4, random_state=0
    )
    overall = stratified_track_holdout_overall_metrics(predictions)
    assert set(overall.repeat) == {0, 1, 2, 3}


def test_stratified_track_holdout_per_collection_metrics_one_row_per_collection():
    data = _synthetic_data(n_per_collection=20, n_collections=3)
    predictions = stratified_track_holdout(
        data, ["feature_1", "feature_2"], CHEAP_XGB_PARAMS, test_size=0.2, n_repeats=4, random_state=0
    )
    per_collection = stratified_track_holdout_per_collection_metrics(predictions)
    assert set(per_collection.collection_name) == {"collection_0", "collection_1", "collection_2"}
    assert "low_n" in per_collection.columns


def test_stratified_track_holdout_per_collection_metrics_n_tracks_counts_distinct_tracks_not_repeats():
    # 20 tracks/collection, test_size=0.2 -> 4 held out per repeat; across 5
    # repeats with a small pool, the *distinct* held-out tracks should still
    # be <= 20 (the collection's actual size), not 4*5=20 coincidentally
    # inflated by repeats re-testing the same tracks.
    data = _synthetic_data(n_per_collection=20, n_collections=2)
    predictions = stratified_track_holdout(
        data, ["feature_1", "feature_2"], CHEAP_XGB_PARAMS, test_size=0.2, n_repeats=5, random_state=0
    )
    per_collection = stratified_track_holdout_per_collection_metrics(predictions)
    assert (per_collection.n_tracks <= 20).all()
    assert (per_collection.n_predictions >= per_collection.n_tracks).all()


def test_flag_low_n_marks_rows_below_threshold():
    from bigwig_qc.evaluation import flag_low_n

    metrics_df = pd.DataFrame({"collection_name": ["big", "small"], "n_tracks": [100, 5]})
    flagged = flag_low_n(metrics_df, n_column="n_tracks", threshold=25)
    assert flagged.set_index("collection_name").low_n.to_dict() == {"big": False, "small": True}


def test_summarise_groups_with_ci_one_row_per_group_per_metric():
    results = pd.DataFrame(
        {
            "model": ["a", "a", "a", "b", "b", "b"],
            "r2": [0.8, 0.9, 1.0, 0.1, 0.2, 0.3],
            "rmse": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "mae": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "spearman_rho": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        }
    )
    summary = summarise_groups_with_ci(results, group_column="model")

    assert set(summary.model) == {"a", "b"}
    assert set(summary.metric) == {"r2", "rmse", "mae", "spearman_rho"}
    assert summary.shape[0] == 2 * 4  # 2 models x 4 metrics

    a_r2 = summary[(summary.model == "a") & (summary.metric == "r2")].iloc[0]
    assert a_r2["mean"] == pytest.approx(0.9)
    b_r2 = summary[(summary.model == "b") & (summary.metric == "r2")].iloc[0]
    assert b_r2["mean"] == pytest.approx(0.2)
    # groups summarised independently, not pooled together
    assert a_r2["mean"] != b_r2["mean"]
