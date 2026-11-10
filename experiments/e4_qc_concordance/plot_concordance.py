#!/usr/bin/env python
"""CLI: plot the quality score against each established QC metric (E4).

Two panels: score vs. TSS enrichment (independent of the training target -
computed over all protein-coding genes, not the constant/housekeeping subset)
and score vs. FRiP (percentage_coverage, already one of the model's own
input features - a weaker independence claim, included for completeness).
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

SERIES_BLUE = "#2a78d6"
SERIES_BLUE_DARK = "#184f95"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"


def _scatter_panel(ax, x, y, xlabel, ylabel):
    rho, p_value = spearmanr(x, y)

    ax.scatter(x, y, s=14, color=SERIES_BLUE, alpha=0.55, edgecolors="none")

    slope, intercept = np.polyfit(x, y, 1)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, slope * x_line + intercept, color=SERIES_BLUE_DARK, linewidth=2)

    p_label = "p < 1e-300" if p_value == 0 else f"p = {p_value:.1e}"
    ax.text(
        0.04, 0.94, f"Spearman ρ = {rho:.2f}\n{p_label}",
        transform=ax.transAxes, va="top", ha="left", fontsize=10, color=INK_SECONDARY,
    )

    ax.set_xlabel(xlabel, color=INK_PRIMARY, fontsize=11)
    ax.set_ylabel(ylabel, color=INK_PRIMARY, fontsize=11)
    ax.tick_params(colors=INK_MUTED)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(INK_MUTED)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merged", required=True, type=Path, help="results/e4_merged_concordance.tsv")
    parser.add_argument("--output-stub", required=True, type=Path, help="Written as <stub>.pdf and <stub>.png")
    args = parser.parse_args()

    data = pd.read_csv(args.merged, sep="\t")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), facecolor="white")
    _scatter_panel(
        axes[0], data.quality_score, data.tss_enrichment,
        "Learned quality score", "TSS enrichment\n(all protein-coding genes, independent of target)",
    )
    _scatter_panel(
        axes[1], data.quality_score, data.percentage_coverage,
        "Learned quality score", "FRiP\n(percentage_coverage - a training feature)",
    )
    fig.tight_layout()

    fig.savefig(f"{args.output_stub}.pdf", format="pdf", bbox_inches="tight", dpi=300, facecolor="white")
    fig.savefig(f"{args.output_stub}.png", format="png", bbox_inches="tight", dpi=300, facecolor="white")
    print(f"Saved {args.output_stub}.{{pdf,png}}")


if __name__ == "__main__":
    main()
