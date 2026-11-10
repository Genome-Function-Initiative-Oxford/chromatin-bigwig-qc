#!/usr/bin/env python
"""CLI: plot leave-one-collection-out (LOCO) performance (E2) - predicted vs.
observed recovery on held-out collections, and per-collection R2/Spearman rho.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SERIES_BLUE = "#2a78d6"
SERIES_BLUE_DARK = "#184f95"
SERIES_RED = "#c23b3b"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"


def _style_axis(ax):
    ax.tick_params(colors=INK_MUTED)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(INK_MUTED)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path, help="loco_predictions.tsv")
    parser.add_argument("--metrics", required=True, type=Path, help="loco_metrics_per_collection.tsv")
    parser.add_argument("--output-stub", required=True, type=Path, help="Written as <stub>.pdf and <stub>.png")
    args = parser.parse_args()

    predictions = pd.read_csv(args.predictions, sep="\t")
    metrics = pd.read_csv(args.metrics, sep="\t").sort_values("r2")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor="white")

    ax = axes[0]
    _style_axis(ax)
    ax.scatter(predictions.y_pred, predictions.y_true, s=14, color=SERIES_BLUE, alpha=0.5, edgecolors="none")
    lims = [
        min(predictions.y_pred.min(), predictions.y_true.min()),
        max(predictions.y_pred.max(), predictions.y_true.max()),
    ]
    ax.plot(lims, lims, color=SERIES_BLUE_DARK, linewidth=1.5, linestyle="--")
    ax.set_xlabel("Predicted (n$_{constant CTCF}$ + n$_{constant promoters}$)", color=INK_PRIMARY, fontsize=10)
    ax.set_ylabel("Observed", color=INK_PRIMARY, fontsize=10)
    ax.set_title("Leave-one-collection-out: predicted vs. observed", color=INK_PRIMARY, fontsize=11)

    ax = axes[1]
    _style_axis(ax)
    colors = [SERIES_RED if low_n else SERIES_BLUE for low_n in metrics.low_n]
    y_pos = np.arange(metrics.shape[0])
    x_floor = -10
    clipped = metrics.r2.clip(lower=x_floor)
    ax.barh(y_pos, clipped, color=colors)
    for y, (value, clipped_value) in enumerate(zip(metrics.r2, clipped)):
        if value < x_floor:
            ax.text(x_floor + 0.3, y, f"{value:.0f}", va="center", ha="left", fontsize=8, color="white")
    ax.axvline(0, color=INK_MUTED, linewidth=1)
    ax.set_xlim(x_floor, 1.1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(metrics.collection_name, fontsize=9, color=INK_PRIMARY)
    ax.set_xlabel(f"R$^2$ (held out, clipped at {x_floor} - label shows true value)", color=INK_PRIMARY, fontsize=10)
    ax.set_title(
        "Per-collection R$^2$ (red = small collection, n$<$25 tracks)", color=INK_PRIMARY, fontsize=11,
    )

    fig.tight_layout()
    fig.savefig(f"{args.output_stub}.pdf", format="pdf", bbox_inches="tight", dpi=300, facecolor="white")
    fig.savefig(f"{args.output_stub}.png", format="png", bbox_inches="tight", dpi=300, facecolor="white")
    print(f"Saved {args.output_stub}.{{pdf,png}}")


if __name__ == "__main__":
    main()
