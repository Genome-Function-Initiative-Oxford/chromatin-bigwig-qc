"""Join per-experiment covariates (features.py) and targets (targets.py) into
one modelling dataframe, keyed by (collection_name, experiment_name).
"""
import json
import os
from glob import glob
from pathlib import Path

import pandas as pd

from . import config


def load_feature_jsons(summary_dir: Path) -> pd.DataFrame:
    """Load one row per `<collection>.<experiment>.json` file in summary_dir."""
    rows = []
    for filepath in sorted(glob(os.path.join(str(summary_dir), "*.json"))):
        collection_name, experiment_name = os.path.basename(filepath).split(".", 2)[:2]
        with open(filepath) as fh:
            row = json.load(fh)
        row["collection_name"] = collection_name
        row["experiment_name"] = experiment_name
        rows.append(row)
    return pd.DataFrame(rows)


def load_target_tsvs(targets_dir: Path) -> pd.DataFrame:
    """Load and concatenate every `<collection>.tsv` file in targets_dir."""
    files = sorted(glob(os.path.join(str(targets_dir), "*.tsv")))
    return pd.concat([pd.read_csv(f, sep="\t") for f in files], ignore_index=True)


def build_model_dataset(summary_dir: Path, targets_dir: Path) -> pd.DataFrame:
    features = load_feature_jsons(summary_dir)
    targets = load_target_tsvs(targets_dir)
    merged = features.merge(targets, on=["experiment_name", "collection_name"])

    # Restrict to config.COLLECTION_NAMES rather than relying on summary_dir/
    # targets_dir only ever containing the right files - e.g. pancreatic_pbmc
    # is excluded here (not deleted from data/interim/) because E2's
    # leave-one-collection-out run showed it has experimental problems
    # (near-chance rank correlation when held out; a sub-group of its tracks
    # has mean_peak_score far below the rest of the dataset).
    return merged[merged.collection_name.isin(config.COLLECTION_NAMES)].reset_index(drop=True)
