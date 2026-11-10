"""Per-bigWig QC covariates: the cheap, bigWig-level regression inputs.

These do not require the constant promoter/CTCF reference sets - that is the
*target* side of the pipeline, computed in targets.py instead.
"""
import json
from pathlib import Path

import pandas as pd
import pyBigWig

from . import config

FEATURE_COLUMNS = (
    "mean_peak_score",
    "median_peak_length",
    "mean_shape_score",
    "mean_peak_length",
    "mean_enrichment_score",
    "median_enrichment_score",
    "lower_quartile_enrichment_score",
    "upper_quartile_enrichment_score",
    "percentage_coverage",
    "percentage_coverage_0_5",
    "percentage_coverage_0_95",
    "n_peaks",
    "n_peaks_0_5",
    "n_peaks_0_95",
    "total_bigwig_sum",
)

# Groups of FEATURE_COLUMNS by what they measure, for E3's feature-group
# ablations (experiments/e3_baselines_ablations). Covers every column in
# FEATURE_COLUMNS exactly once.
FEATURE_GROUPS = {
    "peak_score": ("mean_peak_score", "median_peak_length"),
    "shape": ("mean_shape_score", "mean_peak_length"),
    "enrichment": (
        "mean_enrichment_score",
        "median_enrichment_score",
        "lower_quartile_enrichment_score",
        "upper_quartile_enrichment_score",
    ),
    "coverage": ("percentage_coverage", "percentage_coverage_0_5", "percentage_coverage_0_95"),
    "peak_count": ("n_peaks", "n_peaks_0_5", "n_peaks_0_95"),
    "depth": ("total_bigwig_sum",),
}


def compute_bigwig_features_from_handle(bw, peak_calls: pd.DataFrame) -> dict:
    """Core computation, taking an already-open bigWig handle (testable without a .bw file)."""
    peak_calls = peak_calls.copy()

    total_signal = sum(bw.stats(chrom, type="sum", exact=True)[0] for chrom in set(peak_calls.chrom))
    peak_calls["bigwig_sum"] = [
        bw.stats(row.chrom, row.start, row.end, type="sum", exact=True)[0]
        for row in peak_calls.itertuples()
    ]
    peak_calls["length"] = peak_calls.end - peak_calls.start

    high_score = {t: peak_calls.overall_peak_score >= t for t in config.PEAK_SCORE_THRESHOLDS}

    return {
        "mean_peak_score": peak_calls.overall_peak_score.mean(),
        # kept as in the original pipeline: computes the score median, not the
        # peak length, despite the name - preserved for numerical parity.
        "median_peak_length": peak_calls.overall_peak_score.median(),
        "mean_shape_score": peak_calls.shape_score.mean(),
        "mean_peak_length": peak_calls.length.mean(),
        "mean_enrichment_score": peak_calls.enrichment_score.mean(),
        "median_enrichment_score": peak_calls.enrichment_score.median(),
        "lower_quartile_enrichment_score": peak_calls.enrichment_score.quantile(0.25),
        "upper_quartile_enrichment_score": peak_calls.enrichment_score.quantile(0.75),
        "percentage_coverage": peak_calls.bigwig_sum.sum() / total_signal,
        "percentage_coverage_0_5": peak_calls[high_score[0.5]].bigwig_sum.sum() / total_signal,
        "percentage_coverage_0_95": peak_calls[high_score[0.95]].bigwig_sum.sum() / total_signal,
        "n_peaks": peak_calls.shape[0],
        "n_peaks_0_5": peak_calls[high_score[0.5]].shape[0],
        "n_peaks_0_95": peak_calls[high_score[0.95]].shape[0],
        "total_bigwig_sum": round(total_signal / 1000),
    }


def compute_bigwig_features(bigwig_path: Path, peak_calls_path: Path) -> dict:
    bw = pyBigWig.open(str(bigwig_path))
    peak_calls = pd.read_csv(peak_calls_path, sep="\t")
    return compute_bigwig_features_from_handle(bw, peak_calls)


def save_features_json(features: dict, output_path: Path) -> None:
    with open(output_path, "w") as fh:
        json.dump(features, fh, indent=4)
