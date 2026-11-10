#!/usr/bin/env python
"""CLI: quantify how much of the learned quality score's variance is
explained by collection identity alone (E7's lightweight, data-only half).

This dataset has no independent, cross-comparable "processing route" or
standardised "cell_type" variable - `experiment_name` is a free-text,
per-collection label (472 distinct values across 490 tracks, i.e. almost
one per row), not a controlled vocabulary comparable across collections, so
it cannot serve as a meaningful random-effect grouping factor on its own.
The full `score ~ collection + processing + cell_type` model from todo.tex
is therefore not fittable from this data as-is; what IS fittable, and
directly answers "how much does collection identity alone explain", is a
random-intercept model `score ~ 1 + (1 | collection_name)`, reporting the
intraclass correlation (ICC) - the fraction of total score variance that is
between-collection rather than within-collection.
"""
import argparse
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf

from bigwig_qc.features import FEATURE_COLUMNS
from bigwig_qc.model import load_model
from bigwig_qc.variance_decomposition import intraclass_correlation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path, help="Parquet from build_dataset.py")
    parser.add_argument("--model", required=True, type=Path, help="Trained model.json")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    data = pd.read_parquet(args.dataset)
    model = load_model(args.model)
    data["score"] = model.predict(data[list(FEATURE_COLUMNS)])

    fit = smf.mixedlm("score ~ 1", data, groups=data["collection_name"]).fit()

    group_var = float(fit.cov_re.iloc[0, 0])
    residual_var = float(fit.scale)
    icc = intraclass_correlation(group_var, residual_var)

    print(fit.summary())
    print(f"\nBetween-collection variance: {group_var:.1f}")
    print(f"Within-collection (residual) variance: {residual_var:.1f}")
    print(f"Intraclass correlation (ICC): {icc:.3f}")
    print(f"-> {icc:.0%} of total score variance is explained by collection identity alone.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [{"group_variance": group_var, "residual_variance": residual_var, "icc": icc, "n_tracks": data.shape[0], "n_collections": data.collection_name.nunique()}]
    ).to_csv(args.output, sep="\t", index=False)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
