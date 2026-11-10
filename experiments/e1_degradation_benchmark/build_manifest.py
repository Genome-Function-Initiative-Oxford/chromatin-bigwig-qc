#!/usr/bin/env python
"""CLI: expand the seed-track list (select_seed_tracks.py output) into the full
E1 job manifest consumed by scripts/slurm/run_e1_degradation.sh.
"""
import argparse
from pathlib import Path

import pandas as pd

from bigwig_qc.manifest import build_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-tracks", required=True, type=Path, help="Output of select_seed_tracks.py")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    seed_tracks = pd.read_csv(args.seed_tracks, sep="\t")
    manifest = build_manifest(seed_tracks)
    manifest.to_csv(args.output, sep="\t", index=False)
    print(f"Wrote {manifest.shape[0]} jobs ({seed_tracks.shape[0]} tracks) to {args.output}")


if __name__ == "__main__":
    main()
