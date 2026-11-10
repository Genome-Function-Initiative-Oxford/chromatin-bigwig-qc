#!/usr/bin/env python
"""CLI: derive calibrated high/medium/low quality-score thresholds from E1's
degradation series (bigwig_qc.thresholds), rather than from visual
inspection of the score distribution's quantiles, and apply them to the
full 490-track dataset.
"""
import argparse
from pathlib import Path

import pandas as pd

from bigwig_qc.features import FEATURE_COLUMNS
from bigwig_qc.model import load_model
from bigwig_qc.thresholds import classify, derive_calibrated_thresholds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--degradation-results", required=True, type=Path, help="results/e1_degradation_results.tsv")
    parser.add_argument("--dataset", required=True, type=Path, help="Parquet from build_dataset.py")
    parser.add_argument("--model", required=True, type=Path, help="Trained model.json")
    parser.add_argument("--percentile", type=float, default=5.0)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    degradation_results = pd.read_csv(args.degradation_results, sep="\t")
    thresholds = derive_calibrated_thresholds(degradation_results, percentile=args.percentile)
    print(f"Calibrated thresholds (from E1, percentile={args.percentile}): {thresholds}")

    data = pd.read_parquet(args.dataset)
    model = load_model(args.model)
    data["score"] = model.predict(data[list(FEATURE_COLUMNS)])
    data["quality_band"] = classify(data.score, thresholds)

    print("\nBand counts, calibrated thresholds (this method):")
    print(data.quality_band.value_counts().reindex(["low", "medium", "high"]).to_string())

    quantile_bands = pd.qcut(data.score, q=[0, 1 / 3, 2 / 3, 1], labels=["low", "medium", "high"])
    print("\nBand counts, a simpler quantile approach (tertile quantiles of the score distribution, for comparison):")
    print(quantile_bands.value_counts().reindex(["low", "medium", "high"]).to_string())

    agreement = (data.quality_band.astype(str) == quantile_bands.astype(str)).mean()
    print(f"\nAgreement between the two methods: {agreement:.1%}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    data[["collection_name", "experiment_name", "score", "quality_band"]].assign(
        quantile_band=quantile_bands
    ).to_csv(args.output, sep="\t", index=False)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
