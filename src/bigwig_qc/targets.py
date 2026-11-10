"""Regression targets: recovery of constant promoters/CTCF sites.

For each experiment's peak calls, count how many peaks intersect the constant
promoter / constant CTCF reference trees (references.py), overall and at two
peak-score thresholds. This is the "ground truth" quality signal that the
covariates in features.py are used to predict.
"""
import os
from glob import glob
from pathlib import Path

import pandas as pd
from intervaltree import IntervalTree

from . import config


def _overlaps(peak_calls: pd.DataFrame, tree_by_chrom: dict) -> pd.Series:
    return peak_calls.apply(
        lambda row: len(tree_by_chrom.get(row.chrom, IntervalTree())[row.start:row.end]) > 0,
        axis=1,
    )


def summarise_experiment(peak_calls: pd.DataFrame, promoter_tree: dict, ctcf_tree: dict) -> dict:
    """Core computation on an already-loaded peak-calls DataFrame (testable)."""
    peak_calls = peak_calls.copy()
    peak_calls["promoter"] = _overlaps(peak_calls, promoter_tree)
    peak_calls["ctcf"] = _overlaps(peak_calls, ctcf_tree)

    high_score = {t: peak_calls.overall_peak_score >= t for t in config.PEAK_SCORE_THRESHOLDS}

    return {
        "n_promoters": peak_calls[peak_calls.promoter].shape[0],
        "n_promoters_0_5": peak_calls[peak_calls.promoter & high_score[0.5]].shape[0],
        "n_promoters_0_95": peak_calls[peak_calls.promoter & high_score[0.95]].shape[0],
        "n_ctcf": peak_calls[peak_calls.ctcf].shape[0],
        "n_ctcf_0_5": peak_calls[peak_calls.ctcf & high_score[0.5]].shape[0],
        "n_ctcf_0_95": peak_calls[peak_calls.ctcf & high_score[0.95]].shape[0],
    }


def summarise_experiment_file(peak_calls_path: Path, promoter_tree: dict, ctcf_tree: dict) -> dict:
    peak_calls = pd.read_csv(peak_calls_path, sep="\t")
    return summarise_experiment(peak_calls, promoter_tree, ctcf_tree)


def summarise_collection(directory_path: Path, promoter_tree: dict, ctcf_tree: dict) -> pd.DataFrame:
    """Summarise every *_L-tron.bed peak-call file in a collection's directory."""
    directory_path = Path(directory_path)
    files = sorted(glob(os.path.join(str(directory_path), f"*{config.PEAK_CALL_SUFFIX}")))
    collection_name = directory_path.name

    rows = []
    for file in files:
        experiment_name = os.path.basename(file).replace(config.PEAK_CALL_SUFFIX, "")
        summary = summarise_experiment_file(Path(file), promoter_tree, ctcf_tree)
        rows.append({"experiment_name": experiment_name, "collection_name": collection_name, **summary})

    return pd.DataFrame(rows)
