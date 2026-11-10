#!/usr/bin/env python
"""CLI: add real ATAC-seq example tracks to the track hub built by
build_track_hub.py, stratified by this pipeline's quantile-based quality
label (same equal-count quantile thresholds of the 490-track training
distribution used for the DNase-seq tracks there). This directly lets a
human expert visually validate the core claim - do "good"/"excellent"
tracks actually look clean, and "bad" ones actually look noisy? - rather
than only checking the DNase-seq transfer result. Also serves as a live,
interactive replacement for the "visual inspection of representative
tracks" figure (CIBCB Fig. 8) the manuscript currently has no source
graphics for.
"""
import argparse
import shutil
from pathlib import Path

import pandas as pd

from bigwig_qc import config
from bigwig_qc.features import FEATURE_COLUMNS
from bigwig_qc.model import load_model
from bigwig_qc.thresholds import quantile_classify

LABEL_COLOR = {
    "bad": "204,0,0",
    "poor": "230,126,34",
    "decent": "153,153,0",
    "good": "102,178,0",
    "excellent": "0,102,51",
}
LABEL_ORDER = {"excellent": 0, "good": 1, "decent": 2, "poor": 3, "bad": 4}
QUALITY_LABELS = ["bad", "poor", "decent", "good", "excellent"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path, help="data/interim/model_dataset.parquet")
    parser.add_argument("--model", required=True, type=Path, help="results/model_excl_pancreatic_pbmc.json")
    parser.add_argument("--n-per-band", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--output-dir", required=True, type=Path, help="Same track hub directory as build_track_hub.py")
    parser.add_argument("--priority-offset", type=int, default=100, help="Keep these tracks ordered after the DNase set")
    args = parser.parse_args()

    data = pd.read_parquet(args.dataset)
    model = load_model(args.model)
    data["score"] = model.predict(data[list(FEATURE_COLUMNS)])
    # Reference distribution IS this same dataset here (not circular: with
    # n=490 and 5 equal-count bands, each band still has ~98 tracks - large
    # enough to be a meaningful population, unlike the 30-track DNase set).
    data["quality_label"] = quantile_classify(data.score, data.score, QUALITY_LABELS)
    cuts = data.score.quantile([i / len(QUALITY_LABELS) for i in range(1, len(QUALITY_LABELS))]).tolist()
    print(f"{'/'.join(QUALITY_LABELS)} thresholds: {[round(c) for c in cuts]}")

    bigwig_subdir = args.output_dir / "bigwigs_atac_reference"
    bigwig_subdir.mkdir(parents=True, exist_ok=True)

    selected = []
    for label, group in data.groupby("quality_label", observed=True):
        # One track per distinct collection first (diversity), then fill the
        # rest by score extremity within the band (most clearly "good"/"bad").
        group = group.sort_values("score", ascending=(label == "bad"))
        picked = group.drop_duplicates("collection_name").head(args.n_per_band)
        if picked.shape[0] < args.n_per_band:
            remaining = group[~group.index.isin(picked.index)]
            picked = pd.concat([picked, remaining.head(args.n_per_band - picked.shape[0])])
        selected.append(picked)
    selected = pd.concat(selected).sort_values(
        by="quality_label", key=lambda col: col.map(LABEL_ORDER)
    )

    stanzas = []
    for i, row in enumerate(selected.itertuples(), start=1):
        source = config.BIGWIG_COLLECTION_ROOT / row.collection_name / "bigwigs" / f"{row.experiment_name}.bw"
        if not source.exists():
            print(f"SKIP (bigwig not found): {source}")
            continue
        track_name = f"atac_{row.collection_name}_{row.experiment_name}".replace(" ", "_").replace(".", "_")
        destination = bigwig_subdir / f"{track_name}.bw"
        if not destination.exists():
            shutil.copy2(source, destination)
        priority = args.priority_offset + i
        stanzas.append(
            f"track {track_name}\n"
            f"bigDataUrl bigwigs_atac_reference/{track_name}.bw\n"
            f"shortLabel {row.collection_name}/{row.experiment_name} {row.quality_label}\n"
            f"longLabel ATAC-seq reference example -- {row.collection_name}/{row.experiment_name} -- this tool: {row.quality_label} (score {row.score:.0f})\n"
            f"type bigWig\n"
            f"visibility full\n"
            f"autoScale on\n"
            f"maxHeightPixels 40:40:40\n"
            f"color {LABEL_COLOR[row.quality_label]}\n"
            f"priority {priority}\n"
        )
        print(f"[{i}] {row.collection_name}/{row.experiment_name} ({row.quality_label}, score={row.score:.0f})")

    trackdb_path = args.output_dir / "trackDb.txt"
    existing = trackdb_path.read_text() if trackdb_path.exists() else ""
    trackdb_path.write_text(existing.rstrip("\n") + "\n\n" + "\n".join(stanzas))
    print(f"\nAppended {len(stanzas)} ATAC reference tracks to {trackdb_path}")


if __name__ == "__main__":
    main()
