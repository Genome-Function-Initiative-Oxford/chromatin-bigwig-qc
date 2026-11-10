#!/usr/bin/env python
"""CLI: plot the learned quality score against open-chromatin test power and
false-positive rate (E5), split by degradation mode - visualises why the
score strongly predicts power/FPR under down_sample but not under
inject_noise (see correlate_score_with_power_fpr.py for the underlying
Spearman correlations).
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import spearmanr

MODE_COLORS = {"down_sample": "#2a78d6", "inject_noise": "#c23b3b"}
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"


def _style_axis(ax):
    ax.tick_params(colors=INK_MUTED)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(INK_MUTED)


def _panel(ax, data, metric, ylabel):
    for mode, color in MODE_COLORS.items():
        subset = data[data.degradation_mode == mode]
        rho, p = spearmanr(subset.score, subset[metric])
        p_label = "p < 1e-300" if p == 0 else f"p = {p:.1e}"
        ax.scatter(
            subset.score, subset[metric], s=10, color=color, alpha=0.45, edgecolors="none",
            label=f"{mode} (ρ={rho:.2f}, {p_label})",
        )
    ax.set_xlabel("Learned quality score", color=INK_PRIMARY, fontsize=10)
    ax.set_ylabel(ylabel, color=INK_PRIMARY, fontsize=10)
    ax.legend(fontsize=8, frameon=False, loc="best")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merged", required=True, type=Path, help="results/e5_score_vs_power_fpr.tsv")
    parser.add_argument("--output-stub", required=True, type=Path, help="Written as <stub>.pdf and <stub>.png")
    args = parser.parse_args()

    data = pd.read_csv(args.merged, sep="\t")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), facecolor="white")
    _style_axis(axes[0])
    _panel(axes[0], data, "power", "Statistical power")
    _style_axis(axes[1])
    _panel(axes[1], data, "false_positive_rate", "False-positive rate")
    fig.suptitle(
        "E5: learned score vs. downstream test reliability, by degradation mode",
        color=INK_SECONDARY, fontsize=11,
    )
    fig.tight_layout()

    fig.savefig(f"{args.output_stub}.pdf", format="pdf", bbox_inches="tight", dpi=300, facecolor="white")
    fig.savefig(f"{args.output_stub}.png", format="png", bbox_inches="tight", dpi=300, facecolor="white")
    print(f"Saved {args.output_stub}.{{pdf,png}}")


if __name__ == "__main__":
    main()
