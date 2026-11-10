#!/usr/bin/env python
"""CLI: compute QC covariates for one bigWig + its peak calls.

Replaces the parent repo's process_bigwig.py with the same interface.
"""
import argparse
from pathlib import Path

from bigwig_qc.features import compute_bigwig_features, save_features_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-bigwig", required=True, type=Path, help="Path to the input bigWig file.")
    parser.add_argument("--input-peak-calls", required=True, type=Path, help="Path to the input peak calls file.")
    parser.add_argument("--output", required=True, type=Path, help="Path to the output JSON file.")
    args = parser.parse_args()

    features = compute_bigwig_features(args.input_bigwig, args.input_peak_calls)
    save_features_json(features, args.output)
    print(f"Summary saved to {args.output}")


if __name__ == "__main__":
    main()
