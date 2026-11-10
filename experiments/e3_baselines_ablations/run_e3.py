#!/usr/bin/env python
"""CLI: run E3 (baselines, feature-group ablations, and corrected resampled
t-tests against the tuned XGBoost model).

Uses the fixed hyperparameters already found by scripts/train_model.py's grid
search for the XGBoost side of every comparison - see bigwig_qc.evaluation's
module docstring (shared with E2) for why.
"""
import argparse
from pathlib import Path

import pandas as pd

from bigwig_qc.baselines import compare_models_to_reference, evaluate_baselines, evaluate_feature_ablations
from bigwig_qc.evaluation import summarise_groups_with_ci
from bigwig_qc.model import clean_hyperparams, load_model

METRIC_COLUMNS = ["r2", "rmse", "mae", "spearman_rho"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path, help="Parquet from scripts/build_dataset.py")
    parser.add_argument(
        "--model", required=True, type=Path, help="Trained model.json - supplies XGBoost's fixed hyperparameters"
    )
    parser.add_argument("--n-repeats", type=int, default=10)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    data = pd.read_parquet(args.dataset)
    xgb_params = clean_hyperparams(load_model(args.model).get_params())
    args.output_dir.mkdir(parents=True, exist_ok=True)

    baseline_results = evaluate_baselines(data, xgb_params, test_size=args.test_size, n_repeats=args.n_repeats)
    baseline_results.to_csv(args.output_dir / "baseline_results.tsv", sep="\t", index=False)

    baseline_summary = baseline_results.groupby("model")[METRIC_COLUMNS].agg(["mean", "std"])
    baseline_summary.to_csv(args.output_dir / "baseline_summary.tsv", sep="\t")
    print("Baseline comparison (mean/std across repeats):")
    print(baseline_summary.to_string())

    baseline_summary_ci = summarise_groups_with_ci(baseline_results, group_column="model")
    baseline_summary_ci.to_csv(args.output_dir / "baseline_summary_ci.tsv", sep="\t", index=False)
    print("\nBaseline comparison (95% CI across repeats, per todo.tex's Table 2 ask):")
    print(baseline_summary_ci.to_string(index=False))

    baseline_ttests = compare_models_to_reference(baseline_results, reference_model="xgboost", metric="r2")
    baseline_ttests.to_csv(args.output_dir / "baseline_ttests_vs_xgboost.tsv", sep="\t", index=False)
    print("\nCorrected resampled t-test vs. xgboost (R2):")
    print(baseline_ttests.to_string(index=False))

    ablation_results = evaluate_feature_ablations(data, xgb_params, test_size=args.test_size, n_repeats=args.n_repeats)
    ablation_results.to_csv(args.output_dir / "ablation_results.tsv", sep="\t", index=False)

    ablation_summary = ablation_results.groupby("ablation")[METRIC_COLUMNS].agg(["mean", "std"])
    ablation_summary.to_csv(args.output_dir / "ablation_summary.tsv", sep="\t")
    print("\nFeature-group ablations (mean/std across repeats):")
    print(ablation_summary.to_string())

    ablation_summary_ci = summarise_groups_with_ci(ablation_results, group_column="ablation")
    ablation_summary_ci.to_csv(args.output_dir / "ablation_summary_ci.tsv", sep="\t", index=False)
    print("\nFeature-group ablations (95% CI across repeats, per todo.tex's Table 2 ask):")
    print(ablation_summary_ci.to_string(index=False))

    ablation_ttests = compare_models_to_reference(
        ablation_results, reference_model="full_model", metric="r2", group_column="ablation"
    )
    ablation_ttests.to_csv(args.output_dir / "ablation_ttests_vs_full_model.tsv", sep="\t", index=False)
    print("\nCorrected resampled t-test vs. full_model (R2):")
    print(ablation_ttests.to_string(index=False))


if __name__ == "__main__":
    main()
