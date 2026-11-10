#!/usr/bin/env python
"""CLI: compute TSS enrichment (an established, independent QC metric) for
every track in the dataset.

See src/bigwig_qc/tss_enrichment.py for the method. Needs each track's real
bigWig file (looked up via config.BIGWIG_COLLECTION_ROOT), not just the
already-computed feature JSONs/targets TSVs - this is new data, not a
refactor of the existing pipeline. ~9s/track at the default --n-tss-sample on
this cluster - run in the background or via SLURM for the full dataset.
"""
import argparse
from pathlib import Path

import pandas as pd
import pyBigWig

from bigwig_qc import config, references
from bigwig_qc.tss_enrichment import DEFAULT_N_TSS_SAMPLE, compute_tss_enrichment, sample_tss_positions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path, help="Parquet from scripts/build_dataset.py (gives the track list)")
    parser.add_argument("--n-tss-sample", type=int, default=DEFAULT_N_TSS_SAMPLE)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    data = pd.read_parquet(args.dataset)

    genes = references.load_protein_coding_genes()
    tss_positions = sample_tss_positions(genes, n_genes=args.n_tss_sample, random_state=args.random_state)
    print(f"Sampled {tss_positions.shape[0]} TSS across {tss_positions.chrom.nunique()} chromosomes")

    rows = []
    for i, track in enumerate(data.itertuples(), start=1):
        bigwig_path = config.BIGWIG_COLLECTION_ROOT / track.collection_name / "bigwigs" / f"{track.experiment_name}.bw"
        bw = pyBigWig.open(str(bigwig_path))
        score = compute_tss_enrichment(bw, tss_positions)
        bw.close()
        rows.append({"collection_name": track.collection_name, "experiment_name": track.experiment_name, "tss_enrichment": score})
        print(f"[{i}/{data.shape[0]}] {track.collection_name}.{track.experiment_name}: tss_enrichment={score:.3f}")

    pd.DataFrame(rows).to_csv(args.output, sep="\t", index=False)
    print(f"Saved {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
