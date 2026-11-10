#!/usr/bin/env python
"""CLI: download the bigWig files listed in manifest.tsv (E8)."""
import argparse
import urllib.request
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest, sep="\t")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for i, row in enumerate(manifest.itertuples(), start=1):
        destination = args.output_dir / f"{row.experiment_accession}.bigWig"
        if destination.exists() and destination.stat().st_size > 1_000_000:
            print(f"[{i}/{manifest.shape[0]}] {destination.name} already present ({destination.stat().st_size} bytes), skipping")
            continue
        print(f"[{i}/{manifest.shape[0]}] downloading {row.bigwig_url} -> {destination}")
        urllib.request.urlretrieve(row.bigwig_url, destination)
        size = destination.stat().st_size
        print(f"    {size} bytes")
        if size < 1_000_000:
            raise RuntimeError(f"{destination} is suspiciously small ({size} bytes) - download likely failed")

    print(f"Done: {manifest.shape[0]} files in {args.output_dir}")


if __name__ == "__main__":
    main()
