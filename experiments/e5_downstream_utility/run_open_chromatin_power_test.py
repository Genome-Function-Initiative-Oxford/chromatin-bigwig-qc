#!/usr/bin/env python
"""CLI: run E5(A) - open-chromatin statistical power/FPR test across E1's
degradation grid.

For every (track, mode, level, seed) combination in E1's manifest.tsv,
degrade the track's real bigWig (bigwig_qc.degradation/bigwig_io - the same
code E1 uses, but no LanceOtron/GPU needed here), then test a fixed set of
true-positive (constant promoter/CTCF) and true-negative (background) loci
for open-chromatin significance (bigwig_qc.open_chromatin_test). Records
statistical power and false-positive rate per combination.

--row-start/--row-end allow chunked execution (e.g. one SLURM array task per
chunk), mirroring E1's one-task-per-manifest-row design but batched, since
this test is cheap enough per row to not need one task per row.
"""
import argparse
import tempfile
import time
from pathlib import Path

import pandas as pd
import pyBigWig

from bigwig_qc import config, references
from bigwig_qc.bigwig_io import chrom_sizes, degrade_bigwig_file
from bigwig_qc.open_chromatin_test import compute_power_and_fpr, sample_true_negative_loci, sample_true_positive_loci


def _timed(label, fn, *args, **kwargs):
    start = time.monotonic()
    result = fn(*args, **kwargs)
    print(f"[timing] {label}: {time.monotonic() - start:.1f}s", flush=True)
    return result


def build_loci(n_true_positive: int, n_true_negative: int, random_state: int):
    genes = _timed("load_protein_coding_genes", references.load_protein_coding_genes)
    promoter_tree = _timed("build_constant_promoter_tree", references.build_constant_promoter_tree, genes)
    ctcf_tree = _timed("build_constant_ctcf_tree", references.build_constant_ctcf_tree)

    true_positive_loci = _timed(
        "sample_true_positive_loci",
        sample_true_positive_loci,
        promoter_tree, ctcf_tree, n_loci=n_true_positive, random_state=random_state,
    )

    reference_bigwig_dir = config.BIGWIG_COLLECTION_ROOT / config.COLLECTION_NAMES[0] / "bigwigs"
    first_bw_path = next(reference_bigwig_dir.glob("*.bw"))
    bw = pyBigWig.open(str(first_bw_path))
    sizes = chrom_sizes(bw)
    bw.close()

    true_negative_loci = _timed(
        "sample_true_negative_loci",
        sample_true_negative_loci,
        sizes, exclude_trees=[promoter_tree, ctcf_tree], n_loci=n_true_negative, random_state=random_state,
    )
    return true_positive_loci, true_negative_loci


def run_one(collection_name, experiment_name, mode, level, seed, true_positive_loci, true_negative_loci, bin_size) -> dict:
    input_bigwig = config.BIGWIG_COLLECTION_ROOT / collection_name / "bigwigs" / f"{experiment_name}.bw"

    with tempfile.TemporaryDirectory() as tmp:
        degraded_bigwig = Path(tmp) / f"{experiment_name}.{mode}.{level}.{seed}.bw"
        _timed(f"degrade_bigwig_file({experiment_name})", degrade_bigwig_file, input_bigwig, mode, level, seed, degraded_bigwig, bin_size)

        bw = pyBigWig.open(str(degraded_bigwig))
        result = _timed(f"compute_power_and_fpr({experiment_name})", compute_power_and_fpr, bw, true_positive_loci, true_negative_loci)
        bw.close()

    return {
        "collection_name": collection_name,
        "experiment_name": experiment_name,
        "degradation_mode": mode,
        "level": level,
        "seed": seed,
        **result,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path, help="E1's manifest.tsv")
    parser.add_argument("--n-true-positive-loci", type=int, default=500)
    parser.add_argument("--n-true-negative-loci", type=int, default=500)
    parser.add_argument("--bin-size", type=int, default=config.DEGRADATION_BIN_SIZE_BP)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--row-start", type=int, default=0, help="For chunked/array execution")
    parser.add_argument("--row-end", type=int, default=None)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest, sep="\t")
    if args.row_end is not None:
        manifest = manifest.iloc[args.row_start : args.row_end]

    true_positive_loci, true_negative_loci = build_loci(
        args.n_true_positive_loci, args.n_true_negative_loci, args.random_state
    )
    print(f"true positives: {true_positive_loci.shape[0]}, true negatives: {true_negative_loci.shape[0]}")

    rows = []
    for i, row in enumerate(manifest.itertuples(), start=1):
        result = run_one(
            row.collection_name,
            row.experiment_name,
            row.degradation_mode,
            row.level,
            row.seed,
            true_positive_loci,
            true_negative_loci,
            args.bin_size,
        )
        rows.append(result)
        print(
            f"[{i}/{manifest.shape[0]}] {row.collection_name}.{row.experiment_name} "
            f"{row.degradation_mode}={row.level} seed={row.seed}: "
            f"power={result['power']:.3f} fpr={result['false_positive_rate']:.3f}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, sep="\t", index=False)
    print(f"Saved {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
