#!/usr/bin/env python
"""CLI: plot the learned quality score's distribution by collection, a direct
reproduction (on the current 490-track, pancreatic_pbmc-excluded dataset) of
CIBCB's original Fig. 7 (score distribution by collection).
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from bigwig_qc.features import FEATURE_COLUMNS
from bigwig_qc.model import load_model

SERIES_BLUE = "#2a78d6"
SERIES_BLUE_DARK = "#184f95"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path, help="data/interim/model_dataset.parquet")
    parser.add_argument("--model", required=True, type=Path, help="results/model_excl_pancreatic_pbmc.json")
    parser.add_argument("--output-stub", required=True, type=Path, help="Written as <stub>.pdf and <stub>.png")
    args = parser.parse_args()

    data = pd.read_parquet(args.dataset)
    model = load_model(args.model)
    data["score"] = model.predict(data[list(FEATURE_COLUMNS)])

    order = data.groupby("collection_name").score.median().sort_values().index.tolist()
    boxes = [data.loc[data.collection_name == name, "score"].values for name in order]
    counts = [len(b) for b in boxes]

    fig, ax = plt.subplots(figsize=(8, 5.5), facecolor="white")
    y_pos = range(len(order))
    bp = ax.boxplot(
        boxes, positions=list(y_pos), vert=False, widths=0.6, patch_artist=True,
        medianprops=dict(color=SERIES_BLUE_DARK, linewidth=1.6),
        whiskerprops=dict(color=INK_MUTED), capprops=dict(color=INK_MUTED),
        flierprops=dict(marker="o", markersize=3, markerfacecolor=SERIES_BLUE, markeredgecolor="none", alpha=0.5),
    )
    for box in bp["boxes"]:
        box.set(facecolor=SERIES_BLUE, alpha=0.35, edgecolor=SERIES_BLUE_DARK)

    labels = [f"{name} (n={n})" for name, n in zip(order, counts)]
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(labels, fontsize=9, color=INK_PRIMARY)
    ax.set_xlabel("Learned quality score", color=INK_PRIMARY, fontsize=11)
    ax.set_title(
        "Learned quality score by collection (490 tracks, 10 collections)",
        color=INK_PRIMARY, fontsize=11,
    )
    ax.tick_params(colors=INK_MUTED)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(INK_MUTED)

    fig.tight_layout()
    fig.savefig(f"{args.output_stub}.pdf", format="pdf", bbox_inches="tight", dpi=300, facecolor="white")
    fig.savefig(f"{args.output_stub}.png", format="png", bbox_inches="tight", dpi=300, facecolor="white")
    print(f"Saved {args.output_stub}.{{pdf,png}}")


if __name__ == "__main__":
    main()
