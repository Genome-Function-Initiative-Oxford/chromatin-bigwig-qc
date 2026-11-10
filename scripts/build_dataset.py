#!/usr/bin/env python
"""CLI: join feature JSONs + target TSVs into one modelling dataset (parquet)."""
import argparse
from pathlib import Path

from bigwig_qc.dataset import build_model_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-dir", required=True, type=Path, help="Directory of feature JSONs (run_features.py output)")
    parser.add_argument("--targets-dir", required=True, type=Path, help="Directory of target TSVs (run_targets.py output)")
    parser.add_argument("--output", required=True, type=Path, help="Output .parquet path")
    args = parser.parse_args()

    data = build_model_dataset(args.summary_dir, args.targets_dir)
    data.to_parquet(args.output, index=False)
    print(f"Saved {data.shape[0]} rows to {args.output}")


if __name__ == "__main__":
    main()
