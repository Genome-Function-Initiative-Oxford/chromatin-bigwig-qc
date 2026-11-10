#!/usr/bin/env python
"""CLI: run E2 (leave-one-collection-out CV, repeated group k-fold CV, and the
within-collection stratified track holdout).

Uses the fixed hyperparameters already found by scripts/train_model.py's grid
search (loaded from the saved model) rather than re-tuning in every fold - see
bigwig_qc.evaluation's module docstring for why.
"""
import argparse
from pathlib import Path

import pandas as pd

from bigwig_qc.evaluation import (
    leave_one_collection_out,
    leave_one_collection_out_metrics,
    repeated_group_kfold,
    stratified_track_holdout,
    stratified_track_holdout_overall_metrics,
    stratified_track_holdout_per_collection_metrics,
    summarise_with_ci,
)
from bigwig_qc.features import FEATURE_COLUMNS
from bigwig_qc.model import clean_hyperparams, load_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path, help="Parquet from scripts/build_dataset.py")
    parser.add_argument(
        "--model", required=True, type=Path, help="Trained model.json - supplies the fixed XGBoost hyperparameters"
    )
    parser.add_argument("--n-splits", type=int, default=5, help="For repeated_group_kfold")
    parser.add_argument("--n-repeats", type=int, default=10, help="For repeated_group_kfold and stratified_track_holdout")
    parser.add_argument("--test-size", type=float, default=0.2, help="For stratified_track_holdout")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    data = pd.read_parquet(args.dataset)
    feature_columns = list(FEATURE_COLUMNS)
    xgb_params = clean_hyperparams(load_model(args.model).get_params())

    args.output_dir.mkdir(parents=True, exist_ok=True)

    loco_predictions = leave_one_collection_out(data, feature_columns, xgb_params)
    loco_metrics = leave_one_collection_out_metrics(loco_predictions)
    loco_predictions.to_csv(args.output_dir / "loco_predictions.tsv", sep="\t", index=False)
    loco_metrics.to_csv(args.output_dir / "loco_metrics_per_collection.tsv", sep="\t", index=False)
    print("Leave-one-collection-out metrics (per held-out collection):")
    print(loco_metrics.to_string(index=False))

    kfold_metrics = repeated_group_kfold(
        data, feature_columns, xgb_params, n_splits=args.n_splits, n_repeats=args.n_repeats
    )
    kfold_metrics.to_csv(args.output_dir / "repeated_kfold_metrics.tsv", sep="\t", index=False)

    summary = summarise_with_ci(kfold_metrics)
    summary.to_csv(args.output_dir / "repeated_kfold_summary_ci.tsv", sep="\t", index=False)
    print(f"\nRepeated group {args.n_splits}-fold x{args.n_repeats} summary (95% CI):")
    print(summary.to_string(index=False))

    stratified_predictions = stratified_track_holdout(
        data, feature_columns, xgb_params, test_size=args.test_size, n_repeats=args.n_repeats
    )
    stratified_predictions.to_csv(args.output_dir / "stratified_holdout_predictions.tsv", sep="\t", index=False)

    stratified_overall = stratified_track_holdout_overall_metrics(stratified_predictions)
    stratified_overall.to_csv(args.output_dir / "stratified_holdout_overall_metrics.tsv", sep="\t", index=False)

    stratified_summary = summarise_with_ci(stratified_overall)
    stratified_summary.to_csv(args.output_dir / "stratified_holdout_summary_ci.tsv", sep="\t", index=False)
    print(f"\nStratified within-collection holdout ({args.test_size:.0%} per collection x{args.n_repeats}) summary (95% CI):")
    print(stratified_summary.to_string(index=False))

    stratified_per_collection = stratified_track_holdout_per_collection_metrics(stratified_predictions)
    stratified_per_collection.to_csv(args.output_dir / "stratified_holdout_metrics_per_collection.tsv", sep="\t", index=False)
    print("\nStratified within-collection holdout metrics (per collection, pooled across repeats):")
    print(stratified_per_collection.to_string(index=False))


if __name__ == "__main__":
    main()
