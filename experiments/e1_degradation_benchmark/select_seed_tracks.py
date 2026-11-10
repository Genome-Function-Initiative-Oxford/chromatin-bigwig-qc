#!/usr/bin/env python
"""CLI: select the seed tracks for the E1 controlled-degradation benchmark.

Ranking logic lives in bigwig_qc.seed_selection; this just wires it to the
existing feature/target directories and writes the chosen tracks to a TSV
consumed by run_degradation_benchmark.py.
"""
import argparse
from pathlib import Path

from bigwig_qc import config
from bigwig_qc.dataset import build_model_dataset
from bigwig_qc.seed_selection import select_seed_tracks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-dir", required=True, type=Path, help="Directory of feature JSONs")
    parser.add_argument("--targets-dir", required=True, type=Path, help="Directory of target TSVs")
    parser.add_argument("--n-tracks", type=int, default=config.N_SEED_TRACKS)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    data = build_model_dataset(args.summary_dir, args.targets_dir)
    seed_tracks = select_seed_tracks(data, args.n_tracks)

    columns = ["collection_name", "experiment_name", "recovery_score", "n_promoters", "n_ctcf", "n_peaks"]
    seed_tracks[columns].to_csv(args.output, sep="\t", index=False)
    print(
        f"Selected {seed_tracks.shape[0]} seed tracks spanning "
        f"{seed_tracks.collection_name.nunique()} collections -> {args.output}"
    )


if __name__ == "__main__":
    main()
