#!/usr/bin/env python
"""CLI: SHAP dependence and interaction-value analysis (E6), extending the
summary plot already produced by scripts/train_model.py. Computed on the
same full dataset/model as that summary plot, for direct comparability.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from bigwig_qc import explain
from bigwig_qc.features import FEATURE_COLUMNS
from bigwig_qc.model import load_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path, help="Parquet from build_dataset.py")
    parser.add_argument("--model", required=True, type=Path, help="Trained model.json")
    parser.add_argument("--n-top-features", type=int, default=3, help="Number of top-|SHAP| features to plot")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    data = pd.read_parquet(args.dataset)
    feature_columns = list(FEATURE_COLUMNS)
    X = data[feature_columns]

    model = load_model(args.model)
    shap_values = explain.compute_shap_values(model, X)
    interaction_values = explain.compute_shap_interaction_values(model, X)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    mean_abs_shap = pd.Series(np.abs(shap_values).mean(axis=0), index=feature_columns).sort_values(ascending=False)
    top_features = mean_abs_shap.head(args.n_top_features).index.tolist()
    print("Top features by mean |SHAP|:")
    print(mean_abs_shap.head(args.n_top_features).to_string())

    for feature in top_features:
        stub = args.output_dir / f"dependence_{feature}"
        explain.save_dependence_plot(shap_values, X, feature, stub)
        print(f"Saved {stub}.{{pdf,png}}")

    top_pair = explain.top_interacting_pair(interaction_values, feature_columns)
    print(f"Top interacting feature pair: {top_pair}")
    pair_stub = args.output_dir / f"interaction_{top_pair[0]}_x_{top_pair[1]}"
    explain.save_dependence_plot(shap_values, X, top_pair[0], pair_stub, interaction_index=top_pair[1])
    print(f"Saved {pair_stub}.{{pdf,png}}")

    summary = {
        "top_features_by_mean_abs_shap": mean_abs_shap.head(args.n_top_features).to_dict(),
        "top_interacting_pair": list(top_pair),
    }
    summary_path = args.output_dir / "shap_dependence_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Saved {summary_path}")


if __name__ == "__main__":
    main()
