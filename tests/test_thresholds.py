import pandas as pd
import pytest

from bigwig_qc.thresholds import classify, derive_calibrated_thresholds, quantile_classify, quantile_thresholds


def _synthetic_degradation_results():
    rows = []
    # Undegraded (clean) reference: scores clustered high, 9000-9400.
    for i, score in enumerate([9000, 9100, 9200, 9300, 9400]):
        rows.append({"degradation_mode": "down_sample", "level": 1.0, "score": score, "seed": i})
    # down_sample's heaviest level (smallest value kept) - clearly degraded, low scores.
    for i, score in enumerate([1000, 1100, 1200, 1300, 1400]):
        rows.append({"degradation_mode": "down_sample", "level": 0.01, "score": score, "seed": i})
    # inject_noise's heaviest level (LARGEST value injected) - also clearly degraded.
    for i, score in enumerate([900, 1000, 1100, 1200, 1300]):
        rows.append({"degradation_mode": "inject_noise", "level": 0.8, "score": score, "seed": i})
    # inject_noise's lightest level - should NOT be mistaken for "heaviest".
    for i, score in enumerate([8000, 8100, 8200]):
        rows.append({"degradation_mode": "inject_noise", "level": 0.1, "score": score, "seed": i})
    return pd.DataFrame(rows)


def test_heaviest_inject_noise_level_is_the_largest_value_not_the_smallest():
    # Regression test for a real bug caught during development: down_sample's
    # "heaviest" level is its smallest value (fewest reads kept), but
    # inject_noise's "heaviest" level is its LARGEST value (most noise
    # injected) - using .min() for both would wrongly treat inject_noise's
    # *lightest* degradation as its reference "low quality" population.
    data = _synthetic_degradation_results()
    thresholds = derive_calibrated_thresholds(data, percentile=50.0)
    # With percentile=50, low_medium is the median of the pooled "heaviest"
    # scores. If inject_noise's heaviest were wrongly taken as level=0.1
    # (scores ~8000-8200), the pooled low_medium would be pulled far higher
    # than if level=0.8 (scores ~900-1300) is used correctly.
    assert thresholds["low_medium"] < 2000  # correct: pooled from 1000-1400 and 900-1300


def test_derive_calibrated_thresholds_separates_clean_from_degraded():
    data = _synthetic_degradation_results()
    thresholds = derive_calibrated_thresholds(data, percentile=5.0)
    assert thresholds["low_medium"] < thresholds["medium_high"]
    # Clean reference (9000-9400) and degraded reference (900-1400) are far
    # apart: medium_high (low tail of "clean") should land near 9000-9400,
    # low_medium (high tail of "degraded") near 900-1400 - and there should
    # be a wide gap between the two bands given how separated the clusters are.
    assert 900 <= thresholds["low_medium"] <= 1400
    assert 9000 <= thresholds["medium_high"] <= 9400
    assert thresholds["medium_high"] - thresholds["low_medium"] > 5000


def test_classify_labels_scores_into_three_bands():
    # pd.cut's default bins are right-inclusive: a score exactly AT a
    # threshold falls into the band below it, not above.
    thresholds = {"low_medium": 2000, "medium_high": 8000}
    scores = pd.Series([500, 2000, 5000, 8000, 9000])
    labels = classify(scores, thresholds)
    assert list(labels) == ["low", "low", "medium", "medium", "high"]


def test_classify_handles_collapsed_thresholds_without_crashing():
    # Regression test: when the two reference populations' tails coincide,
    # derive_calibrated_thresholds collapses low_medium == medium_high (see
    # its own docstring) - classify() must degrade to a 2-band low/high split
    # rather than raising pandas' "bin edges must be unique" error (hit for
    # real on E1's actual data at percentile=5, not just a hypothetical).
    thresholds = {"low_medium": 5000, "medium_high": 5000}
    scores = pd.Series([1000, 5000, 9000])
    labels = classify(scores, thresholds)
    assert list(labels) == ["low", "low", "high"]
    assert set(labels.cat.categories) == {"low", "high"}


def test_quantile_thresholds_returns_n_minus_1_equal_count_cuts():
    reference = pd.Series(range(1, 101))  # 1..100, uniform
    cuts = quantile_thresholds(reference, labels=["bad", "decent", "good"])
    assert len(cuts) == 2
    # Tertiles of 1..100 land near 33.67 and 67.33.
    assert 33 < cuts[0] < 35
    assert 66 < cuts[1] < 68


def test_quantile_thresholds_five_levels():
    reference = pd.Series(range(1, 101))
    cuts = quantile_thresholds(reference, labels=["bad", "poor", "decent", "good", "excellent"])
    assert len(cuts) == 4


def test_quantile_classify_against_a_different_larger_reference():
    # The reference distribution is 1..1000 (so tertiles are ~333/~667);
    # scores being classified are a small, separate sample - classification
    # must use the REFERENCE's quantiles, not re-derive quantiles from the
    # small `scores` sample itself (which would be circular).
    reference = pd.Series(range(1, 1001))
    scores = pd.Series([10, 500, 900])
    labels = quantile_classify(scores, reference, labels=["bad", "decent", "good"])
    assert list(labels) == ["bad", "decent", "good"]


def test_quantile_classify_five_levels_matches_manual_bins():
    reference = pd.Series(range(1, 101))
    scores = pd.Series([5, 25, 50, 75, 95])
    labels = quantile_classify(scores, reference, labels=["bad", "poor", "decent", "good", "excellent"])
    assert list(labels) == ["bad", "poor", "decent", "good", "excellent"]
