#!/usr/bin/env python
"""CLI: train the XGBoost quality-score model and save it + a SHAP summary plot."""
import argparse
from pathlib import Path

import pandas as pd

from bigwig_qc import explain
from bigwig_qc import model as model_module
from bigwig_qc.features import FEATURE_COLUMNS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path, help="Parquet from build_dataset.py")
    parser.add_argument("--model-output", required=True, type=Path)
    parser.add_argument("--shap-output-stub", required=True, type=Path, help="Written as <stub>.pdf and <stub>.png")
    args = parser.parse_args()

    data = pd.read_parquet(args.dataset)
    feature_columns = list(FEATURE_COLUMNS)

    result = model_module.train(data, feature_columns)
    print(f"Test R^2={result.test_r2:.4f} MSE={result.test_mse:.2f} params={result.best_params}")

    model_module.save_model(result.model, args.model_output)
    print(f"Saved model to {args.model_output}")

    shap_values = explain.compute_shap_values(result.model, data[feature_columns])
    explain.save_summary_plot(shap_values, data[feature_columns], args.shap_output_stub)
    print(f"Saved SHAP summary plot to {args.shap_output_stub}.{{pdf,png}}")


if __name__ == "__main__":
    main()
