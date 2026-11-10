#!/usr/bin/env python
"""CLI: produce a synthetically degraded bigWig (down-sampled or noise-injected).

See src/bigwig_qc/degradation.py for the method and
experiments/e1_degradation_benchmark/ for how this fits into the controlled-
degradation benchmark (E1). The input bigWig's signal is rasterised onto a
fixed bin grid (--bin-size) before degradation, so the output is at that
resolution rather than the input's native resolution.
"""
import argparse
from pathlib import Path

from bigwig_qc import config
from bigwig_qc.bigwig_io import degrade_bigwig_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-bigwig", required=True, type=Path)
    parser.add_argument("--mode", required=True, choices=["down_sample", "inject_noise"])
    parser.add_argument(
        "--level", required=True, type=float,
        help="down_sample: fraction of depth kept. inject_noise: fraction of signal redistributed. Both in [0, 1].",
    )
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--bin-size", type=int, default=config.DEGRADATION_BIN_SIZE_BP)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    degrade_bigwig_file(args.input_bigwig, args.mode, args.level, args.seed, args.output, args.bin_size)
    print(f"Wrote degraded bigWig ({args.mode}, level={args.level}, seed={args.seed}) to {args.output}")


if __name__ == "__main__":
    main()
