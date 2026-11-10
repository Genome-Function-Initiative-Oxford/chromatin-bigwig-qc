#!/usr/bin/env python
"""CLI: split conformal prediction intervals for the quality score (E6).

A proper 3-way split (train/calibrate/test) is used so that calibration-set
residuals (used to size the interval) and test-set coverage (used to
validate it) are never the same data - reusing one held-out set for both
would overstate how well-calibrated the intervals actually are. The
regressor itself keeps the already-tuned hyperparameters
(results/model_excl_pancreatic_pbmc.json) rather than re-running the 3-stage
grid search, since this experiment is about quantifying uncertainty around
an already-established model, not re-deriving it.
"""
import argparse
from pathlib import Path

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split

from bigwig_qc.conformal import calibrate, empirical_coverage, predict_interval
from bigwig_qc.features import FEATURE_COLUMNS
from bigwig_qc.model import RANDOM_STATE, clean_hyperparams, load_model, regression_target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path, help="Parquet from build_dataset.py")
    parser.add_argument("--model", required=True, type=Path, help="Trained model.json - supplies fixed hyperparameters")
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.1, 0.2], help="1-alpha is the target coverage")
    parser.add_argument("--train-size", type=float, default=0.6)
    parser.add_argument("--calib-size", type=float, default=0.2)  # remainder (0.2 here) goes to test
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    data = pd.read_parquet(args.dataset)
    feature_columns = list(FEATURE_COLUMNS)
    X, y = data[feature_columns], regression_target(data)

    X_train, X_rest, y_train, y_rest = train_test_split(X, y, train_size=args.train_size, random_state=RANDOM_STATE)
    calib_fraction_of_rest = args.calib_size / (1 - args.train_size)
    X_calib, X_test, y_calib, y_test = train_test_split(
        X_rest, y_rest, train_size=calib_fraction_of_rest, random_state=RANDOM_STATE
    )
    print(f"train={X_train.shape[0]} calib={X_calib.shape[0]} test={X_test.shape[0]}")

    xgb_params = clean_hyperparams(load_model(args.model).get_params())
    model = xgb.XGBRegressor(**xgb_params)
    model.fit(X_train, y_train)

    y_pred_calib = model.predict(X_calib)
    y_pred_test = model.predict(X_test)

    rows = []
    for alpha in args.alphas:
        half_width = calibrate(y_calib, y_pred_calib, alpha=alpha)
        intervals = predict_interval(y_pred_test, half_width)
        coverage = empirical_coverage(y_test, intervals)
        rows.append(
            {
                "alpha": alpha,
                "nominal_coverage": 1 - alpha,
                "empirical_coverage": coverage,
                "half_width": half_width,
                "mean_interval_width": 2 * half_width,
            }
        )
        print(
            f"alpha={alpha}: nominal={1 - alpha:.2f} empirical={coverage:.3f} "
            f"half_width={half_width:.1f} (interval width={2 * half_width:.1f})"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output_dir / "conformal_calibration.tsv", sep="\t", index=False)

    # Save the test-set intervals at the first alpha for downstream plotting.
    half_width = calibrate(y_calib, y_pred_calib, alpha=args.alphas[0])
    test_intervals = predict_interval(y_pred_test, half_width)
    test_intervals["y_true"] = y_test.to_numpy()
    test_intervals.to_csv(args.output_dir / "conformal_test_intervals.tsv", sep="\t", index=False)
    print(f"Saved {args.output_dir / 'conformal_calibration.tsv'} and conformal_test_intervals.tsv")


if __name__ == "__main__":
    main()
