#!/usr/bin/env python
"""Run exactly one ENCODE DNase-seq bigWig end-to-end (E8, DNase-seq half):
call peaks with LanceOtron, compute the same QC covariates used throughout
this pipeline, and score with the trained quality-score model. Intended to
be invoked once per manifest row, one per SLURM array task under
scripts/slurm/run_e8_dnase_peaks.sh.

No new reference data is needed here - DNase-seq is still human/hg38, so
this reuses the pipeline's existing (human-only) constant promoter/CTCF
reference sets and trained model unchanged.
"""
import argparse
from pathlib import Path

import pandas as pd

from bigwig_qc.features import FEATURE_COLUMNS, compute_bigwig_features
from bigwig_qc.lanceotron import call_peaks
from bigwig_qc.model import load_model


def run_one(bigwig_path, experiment_accession, audit_tier, model_path, peaks_dir) -> dict:
    peak_calls_path = call_peaks(bigwig_path, peaks_dir)
    features = compute_bigwig_features(bigwig_path, peak_calls_path)

    model = load_model(model_path)
    score = float(model.predict(pd.DataFrame([features])[list(FEATURE_COLUMNS)])[0])

    return {
        "experiment_accession": experiment_accession,
        "audit_tier": audit_tier,
        "score": score,
        **features,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bigwig", required=True, type=Path)
    parser.add_argument("--experiment-accession", required=True)
    parser.add_argument("--audit-tier", required=True, choices=["clean", "warning", "error"])
    parser.add_argument("--model", required=True, type=Path, help="Trained model.json from scripts/train_model.py")
    parser.add_argument("--peaks-dir", required=True, type=Path, help="Where LanceOtron writes its peak calls")
    parser.add_argument("--output", required=True, type=Path, help="Single-row result TSV for this job")
    args = parser.parse_args()

    result = run_one(args.bigwig, args.experiment_accession, args.audit_tier, args.model, args.peaks_dir)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([result]).to_csv(args.output, sep="\t", index=False)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
