#!/usr/bin/env python
"""CLI: build a UCSC Track Hub for the 30 ENCODE DNase-seq tracks scored in
E8, so a human expert can visually inspect the actual signal and form an
independent judgement - not just compare numbers. Each track's label shows
ENCODE's own audit tier (clean/warning/error), this pipeline's own
quantile-based quality label, and the raw learned score, colour-coded by
ENCODE's tier, ordered tier-then-score so browsing top-to-bottom roughly
follows the expected quality gradient and any mismatch (e.g. a "warning"
track sitting among "clean" ones, or a track ENCODE calls clean that this
pipeline calls "bad") is easy to spot.

Quality-label thresholds are equal-count quantiles of the score
distribution on the 490-track ATAC-seq dataset this
model was actually trained/evaluated on (--reference-dataset/
--reference-model) - NOT re-derived from just these 30 DNase tracks, which
would be circular (any small sample trivially splits into equal-sized bands
and says nothing about absolute quality).

Track hub spec: https://genome.ucsc.edu/goldenPath/help/hgTrackHubHelp.html
"""
import argparse
import shutil
from pathlib import Path

import pandas as pd

from bigwig_qc.features import FEATURE_COLUMNS
from bigwig_qc.model import load_model
from bigwig_qc.thresholds import quantile_classify

TIER_COLOR = {"clean": "0,153,0", "warning": "230,159,0", "error": "204,0,0"}
TIER_ORDER = {"clean": 0, "warning": 1, "error": 2}
QUALITY_LABELS = ["bad", "poor", "decent", "good", "excellent"]


def _reference_scores(reference_dataset: Path, reference_model: Path) -> pd.Series:
    reference_data = pd.read_parquet(reference_dataset)
    model = load_model(reference_model)
    return pd.Series(model.predict(reference_data[list(FEATURE_COLUMNS)]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, type=Path, help="results/e8_dnase_results.tsv")
    parser.add_argument("--bigwig-dir", required=True, type=Path, help="data/raw/encode_dnase/ (source bigWigs)")
    parser.add_argument("--reference-dataset", required=True, type=Path, help="data/interim/model_dataset.parquet")
    parser.add_argument("--reference-model", required=True, type=Path, help="results/model_excl_pancreatic_pbmc.json")
    parser.add_argument("--output-dir", required=True, type=Path, help="Track hub destination directory")
    parser.add_argument(
        "--base-url", required=True,
        help="Public base URL the output-dir is served at (e.g. https://example.org/public/bw_qc_hub)",
    )
    parser.add_argument("--hub-name", default="bigwig_qc_paper")
    args = parser.parse_args()

    reference_scores = _reference_scores(args.reference_dataset, args.reference_model)
    cuts = reference_scores.quantile([i / len(QUALITY_LABELS) for i in range(1, len(QUALITY_LABELS))]).tolist()
    print(f"{'/'.join(QUALITY_LABELS)} thresholds (from the 490-track ATAC-seq reference): {[round(c) for c in cuts]}")

    data = pd.read_csv(args.results, sep="\t")
    data["quality_label"] = quantile_classify(data.score, reference_scores, QUALITY_LABELS)
    data = data.sort_values(
        by=["audit_tier", "score"],
        key=lambda col: col.map(TIER_ORDER) if col.name == "audit_tier" else col,
        ascending=[True, False],
    )

    bigwig_subdir = args.output_dir / "bigwigs"
    bigwig_subdir.mkdir(parents=True, exist_ok=True)

    hub_txt = f"""hub {args.hub_name}
shortLabel BigWig QC score vs ENCODE audit
longLabel Learned bigWig quality score (bad/poor/decent/good/excellent, quintiles of the 490-track ATAC-seq training distribution) vs. ENCODE's own audit-flag severity (clean/warning/error, by SPOT-score audit), on 30 ENCODE DNase-seq tracks - E8 transfer experiment
genomesFile genomes.txt
email simone.riva@imm.ox.ac.uk
"""
    (args.output_dir / "hub.txt").write_text(hub_txt)
    (args.output_dir / "genomes.txt").write_text("genome hg38\ntrackDb trackDb.txt\n")

    stanzas = []
    for priority, row in enumerate(data.itertuples(), start=1):
        source = args.bigwig_dir / f"{row.experiment_accession}.bigWig"
        destination = bigwig_subdir / f"{row.experiment_accession}.bigWig"
        if not destination.exists():
            shutil.copy2(source, destination)
        stanzas.append(
            f"track {row.experiment_accession}\n"
            f"bigDataUrl bigwigs/{row.experiment_accession}.bigWig\n"
            f"shortLabel {row.experiment_accession} {row.audit_tier}/{row.quality_label}\n"
            f"longLabel {row.experiment_accession} -- ENCODE audit: {row.audit_tier} -- this tool: {row.quality_label} (score {row.score:.0f})\n"
            f"type bigWig\n"
            f"visibility full\n"
            f"autoScale on\n"
            f"maxHeightPixels 40:40:40\n"
            f"color {TIER_COLOR[row.audit_tier]}\n"
            f"priority {priority}\n"
        )
        print(f"[{priority}/{data.shape[0]}] {row.experiment_accession} (ENCODE={row.audit_tier}, tool={row.quality_label}, score={row.score:.0f})")

    (args.output_dir / "trackDb.txt").write_text("\n".join(stanzas))
    print(f"\nHub written to {args.output_dir}")
    print(f"Load in UCSC via: {args.base_url}/hub.txt")


if __name__ == "__main__":
    main()
