#!/usr/bin/env python
"""CLI: join E1's learned quality scores with E5's open-chromatin
power/false-positive-rate results and test whether the score predicts
downstream statistical reliability - E5's original research question.

E1 (experiments/e1_degradation_benchmark) and E5
(experiments/e5_downstream_utility) both run over the exact same
(collection_name, experiment_name, degradation_mode, level, seed) grid
(E1's manifest.tsv), so the join is exact - no feature recomputation needed,
E1's result rows already carry the model's "score" column directly.
"""
import argparse
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

JOIN_KEY = ["collection_name", "experiment_name", "degradation_mode", "level", "seed"]
TARGET_METRICS = ("power", "false_positive_rate")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e1-results", required=True, type=Path, help="E1's aggregate_results.py output")
    parser.add_argument("--e5-results", required=True, type=Path, help="E5's aggregate_results.py output")
    parser.add_argument("--output", required=True, type=Path, help="Merged table with score + power + false_positive_rate")
    args = parser.parse_args()

    e1 = pd.read_csv(args.e1_results, sep="\t")
    e5 = pd.read_csv(args.e5_results, sep="\t")
    merged = e1.merge(e5[[*JOIN_KEY, *TARGET_METRICS]], on=JOIN_KEY, how="inner")
    print(f"Joined {merged.shape[0]} rows (E1: {e1.shape[0]}, E5: {e5.shape[0]})")

    print("\nOverall (pooled across both degradation modes):")
    for metric in TARGET_METRICS:
        rho, p = spearmanr(merged.score, merged[metric])
        print(f"  score vs. {metric}: Spearman rho={rho:.3f} (p={p:.2e})")

    print("\nPer degradation_mode (the two modes have different score/power/FPR dose-response shapes - see READMEs):")
    for mode, group in merged.groupby("degradation_mode"):
        print(f"  {mode} (n={group.shape[0]}):")
        for metric in TARGET_METRICS:
            rho, p = spearmanr(group.score, group[metric])
            print(f"    score vs. {metric}: Spearman rho={rho:.3f} (p={p:.2e})")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output, sep="\t", index=False)
    print(f"\nSaved {args.output}")


if __name__ == "__main__":
    main()
