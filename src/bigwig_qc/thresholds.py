"""Derive high/medium/low quality-score thresholds from E1's degradation
series, rather than from visual inspection of the score distribution's
quantiles (the original CIBCB approach).

Method: E1's degradation grid gives two known-label reference populations by
construction - tracks deliberately left undegraded (down_sample, level=1.0)
are "clearly high quality", and tracks degraded at the heaviest level of
either mode are "clearly low quality". The medium/high boundary is set at
the low percentile of the clean reference (a track scoring below this looks
worse than all but `percentile`% of known-good tracks); the low/medium
boundary is set at the high percentile of the degraded reference (a track
scoring above this looks better than all but `percentile`% of known-bad
tracks). Tracks between the two boundaries are "medium".
"""
import pandas as pd

UNDEGRADED_MODE = "down_sample"
UNDEGRADED_LEVEL = 1.0

# Direction of "heaviest degradation" in each mode's `level` column:
# down_sample's level is the FRACTION OF READS KEPT (1.0=none removed, so the
# heaviest level is the smallest value); inject_noise's level is the fraction
# of signal injected as noise (so the heaviest level is the LARGEST value) -
# these are opposite directions, not interchangeable.
_HEAVIEST_LEVEL_AGG = {"down_sample": "min", "inject_noise": "max"}


def _heaviest_degraded_scores(degradation_results: pd.DataFrame) -> pd.Series:
    """Pooled scores at the heaviest level of EACH mode (not just one), since
    "clearly low quality" should be defined by the worst case across both
    degradation types, not just whichever mode happens to have more levels.
    """
    pooled = []
    for mode, group in degradation_results.groupby("degradation_mode"):
        agg = _HEAVIEST_LEVEL_AGG[mode]
        heaviest_level = group.level.min() if agg == "min" else group.level.max()
        pooled.append(group[group.level == heaviest_level].score)
    return pd.concat(pooled, ignore_index=True)


def derive_calibrated_thresholds(degradation_results: pd.DataFrame, percentile: float = 5.0) -> dict:
    """Return {"low_medium": ..., "medium_high": ...} score thresholds.

    percentile: the tail percentile used on each reference distribution (5 =
    a 90%-confidence-style separation between the two known populations).
    """
    clean = degradation_results[
        (degradation_results.degradation_mode == UNDEGRADED_MODE)
        & (degradation_results.level == UNDEGRADED_LEVEL)
    ].score
    degraded = _heaviest_degraded_scores(degradation_results)

    medium_high = float(clean.quantile(percentile / 100.0))
    low_medium = float(degraded.quantile(1 - percentile / 100.0))

    if low_medium > medium_high:
        # The two reference populations' tails overlap at this percentile -
        # collapse to a single boundary (their midpoint) rather than return
        # an inverted, nonsensical [low_medium > medium_high] pair.
        midpoint = (low_medium + medium_high) / 2
        low_medium = medium_high = midpoint

    return {"low_medium": low_medium, "medium_high": medium_high}


def quantile_thresholds(reference_scores: pd.Series, labels: list) -> list:
    """Interior cut points for an N-level, equal-COUNT quantile split of
    `reference_scores` (CIBCB's original quality-banding approach - equal
    population counts per band, not equal-width slices of the score range,
    which would give wildly unequal band sizes on a skewed distribution).
    Returns len(labels)-1 boundaries.
    """
    n = len(labels)
    return reference_scores.quantile([i / n for i in range(1, n)]).tolist()


def quantile_classify(scores: pd.Series, reference_scores: pd.Series, labels: list) -> pd.Series:
    """Label each value in `scores` by which equal-count quantile band of
    `reference_scores` it falls into.

    `reference_scores` is typically a different, larger, already-understood
    dataset than `scores` (e.g. classifying a handful of new external tracks
    against the main training distribution's quantiles) - computing the
    quantiles from `scores` itself would be circular when `scores` is small
    (any sample trivially splits into equal-sized bands, which says nothing
    about absolute quality).
    """
    cuts = quantile_thresholds(reference_scores, labels)
    bounds = [-float("inf")] + cuts + [float("inf")]
    return pd.cut(scores, bins=bounds, labels=labels)


def classify(scores: pd.Series, thresholds: dict) -> pd.Series:
    """Label each score as "low"/"medium"/"high" per `thresholds`.

    If the two reference populations' tails coincided (derive_calibrated_
    thresholds collapsed both boundaries to one value), there is no
    meaningful "medium" band - scores are split "low"/"high" at that single
    boundary instead of raising on a degenerate (duplicate-edge) 3-bin cut.
    """
    if thresholds["low_medium"] == thresholds["medium_high"]:
        return pd.cut(
            scores,
            bins=[-float("inf"), thresholds["low_medium"], float("inf")],
            labels=["low", "high"],
        )
    return pd.cut(
        scores,
        bins=[-float("inf"), thresholds["low_medium"], thresholds["medium_high"], float("inf")],
        labels=["low", "medium", "high"],
    )
