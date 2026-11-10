#!/usr/bin/env python
"""CLI: join TSS enrichment + FRiP against the learned quality score and
compute concordance correlations (E4).
"""
import argparse
from pathlib import Path

import pandas as pd
from scipy.stats import pearsonr, spearmanr

from bigwig_qc.features import FEATURE_COLUMNS
from bigwig_qc.model import load_model

ESTABLISHED_METRICS = ("tss_enrichment", "percentage_coverage")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path, help="Parquet from scripts/build_dataset.py")
    parser.add_argument("--tss-enrichment", required=True, type=Path, help="Output of compute_tss_enrichment.py")
    parser.add_argument("--model", required=True, type=Path, help="Trained model.json")
    parser.add_argument("--output", required=True, type=Path, help="Merged table with quality_score + metrics")
    args = parser.parse_args()

    data = pd.read_parquet(args.dataset)
    tss_enrichment = pd.read_csv(args.tss_enrichment, sep="\t")
    merged = data.merge(tss_enrichment, on=["collection_name", "experiment_name"])
    print(f"Merged {merged.shape[0]} / {data.shape[0]} dataset rows with {tss_enrichment.shape[0]} TSS-enrichment rows")

    model = load_model(args.model)
    merged["quality_score"] = model.predict(merged[list(FEATURE_COLUMNS)])

    for metric in ESTABLISHED_METRICS:
        rho, p_rho = spearmanr(merged.quality_score, merged[metric])
        r, p_r = pearsonr(merged.quality_score, merged[metric])
        print(f"{metric}: Spearman rho={rho:.3f} (p={p_rho:.2e}), Pearson r={r:.3f} (p={p_r:.2e})")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output, sep="\t", index=False)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
