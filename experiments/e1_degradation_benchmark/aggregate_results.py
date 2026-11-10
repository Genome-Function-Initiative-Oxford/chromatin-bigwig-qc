#!/usr/bin/env python
"""CLI: concatenate per-job E1 result TSVs (run_one_degradation.py output) into one table."""
import argparse
from glob import glob
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True, type=Path, help="Directory of per-job result TSVs")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    files = sorted(glob(str(args.results_dir / "*.tsv")))
    if not files:
        raise FileNotFoundError(f"No per-job result files found in {args.results_dir}")

    combined = pd.concat([pd.read_csv(f, sep="\t") for f in files], ignore_index=True)
    combined.to_csv(args.output, sep="\t", index=False)
    print(f"Aggregated {len(files)} job results ({combined.shape[0]} rows) -> {args.output}")


if __name__ == "__main__":
    main()
