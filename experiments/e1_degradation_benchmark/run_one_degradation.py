#!/usr/bin/env python
"""Run exactly one (track, mode, level, seed) E1 job end-to-end.

Degrades the seed track's bigWig, re-calls peaks with LanceOtron, recomputes
QC covariates, scores with the trained model, and writes a single-row result
TSV. Intended to be invoked once per manifest row (build_manifest.py), one per
SLURM array task under scripts/slurm/run_e1_degradation.sh.
"""
import argparse
import tempfile
from pathlib import Path

import pandas as pd

from bigwig_qc import config
from bigwig_qc.bigwig_io import degrade_bigwig_file
from bigwig_qc.features import FEATURE_COLUMNS, compute_bigwig_features
from bigwig_qc.lanceotron import call_peaks
from bigwig_qc.model import load_model


def run_one(collection_name, experiment_name, mode, level, seed, model_path, bin_size) -> dict:
    input_bigwig = config.BIGWIG_COLLECTION_ROOT / collection_name / "bigwigs" / f"{experiment_name}.bw"

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        degraded_bigwig = tmp / f"{experiment_name}.{mode}.{level}.{seed}.bw"
        degrade_bigwig_file(input_bigwig, mode, level, seed, degraded_bigwig, bin_size)

        peak_calls_path = call_peaks(degraded_bigwig, tmp / "peaks")
        features = compute_bigwig_features(degraded_bigwig, peak_calls_path)

    model = load_model(model_path)
    score = float(model.predict(pd.DataFrame([features])[list(FEATURE_COLUMNS)])[0])

    return {
        "collection_name": collection_name,
        "experiment_name": experiment_name,
        # "degradation_mode", not "mode": a DataFrame column named "mode"
        # collides with the reserved DataFrame.mode() method on attribute
        # access (df.mode is the method, not the column) - see test_manifest.py.
        "degradation_mode": mode,
        "level": level,
        "seed": seed,
        "score": score,
        **features,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection-name", required=True)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--mode", required=True, choices=["down_sample", "inject_noise"])
    parser.add_argument("--level", required=True, type=float)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--model", required=True, type=Path, help="Trained model.json from scripts/train_model.py")
    parser.add_argument("--bin-size", type=int, default=config.DEGRADATION_BIN_SIZE_BP)
    parser.add_argument("--output", required=True, type=Path, help="Single-row result TSV for this job")
    args = parser.parse_args()

    result = run_one(
        args.collection_name, args.experiment_name, args.mode, args.level, args.seed, args.model, args.bin_size
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([result]).to_csv(args.output, sep="\t", index=False)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
