#!/usr/bin/env python
"""CLI: summarise constant promoter/CTCF recovery for one collection's peak calls.

Replaces the parent repo's process_targets.py with the same interface.
"""
import argparse
from pathlib import Path

from bigwig_qc import references, targets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Path to directory of peak calls")
    parser.add_argument("--output", required=True, type=Path, help="tsv with summary counts")
    args = parser.parse_args()

    genes = references.load_protein_coding_genes()
    promoter_tree = references.build_constant_promoter_tree(genes)
    ctcf_tree = references.build_constant_ctcf_tree()

    summary = targets.summarise_collection(args.input, promoter_tree, ctcf_tree)
    summary.to_csv(args.output, sep="\t", index=False)
    print(f"Saved {summary.shape[0]} rows to {args.output}")


if __name__ == "__main__":
    main()
