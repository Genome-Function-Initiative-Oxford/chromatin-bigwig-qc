#!/usr/bin/env python
"""CLI: plot the learned quality score against degradation severity (E1).

Two panels, down_sample and inject_noise, each one line per seed track
(thin, low-alpha) plus the across-track mean (solid). X-axis is framed as
"degradation severity" in both panels so more degradation always reads
left-to-right as worse: for down_sample this is 1 - level (fraction of reads
REMOVED, since level is the fraction KEPT - level=1.0 keeps everything, i.e.
zero severity); for inject_noise, level already increases with severity, so
it's used directly.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

SERIES_BLUE = "#2a78d6"
SERIES_BLUE_DARK = "#184f95"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"


def _dose_response_panel(ax, data, mode, severity_fn, xlabel):
    subset = data[data.degradation_mode == mode].copy()
    subset["severity"] = severity_fn(subset.level)

    for _, track in subset.groupby(["collection_name", "experiment_name"]):
        track_mean = track.groupby("severity")["score"].mean().sort_index()
        ax.plot(track_mean.index, track_mean.values, color=SERIES_BLUE, alpha=0.18, linewidth=1)

    overall_mean = subset.groupby("severity")["score"].mean().sort_index()
    ax.plot(overall_mean.index, overall_mean.values, color=SERIES_BLUE_DARK, linewidth=2.5, marker="o", markersize=5)

    ax.set_xlabel(xlabel, color=INK_PRIMARY, fontsize=11)
    ax.set_ylabel("Learned quality score", color=INK_PRIMARY, fontsize=11)
    ax.set_title(mode, color=INK_PRIMARY, fontsize=12)
    ax.tick_params(colors=INK_MUTED)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(INK_MUTED)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, type=Path, help="results/e1_degradation_results.tsv")
    parser.add_argument("--output-stub", required=True, type=Path, help="Written as <stub>.pdf and <stub>.png")
    args = parser.parse_args()

    data = pd.read_csv(args.results, sep="\t")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), facecolor="white")
    _dose_response_panel(
        axes[0], data, "down_sample", severity_fn=lambda level: 1 - level,
        xlabel="Fraction of reads removed\n(0 = undegraded, 0.99 = heaviest)",
    )
    _dose_response_panel(
        axes[1], data, "inject_noise", severity_fn=lambda level: level,
        xlabel="Noise injection level\n(0.1 = lightest, 0.8 = heaviest)",
    )
    fig.suptitle(
        "E1: learned quality score vs. degradation severity (30 seed tracks, thin lines; dark line = mean)",
        color=INK_SECONDARY, fontsize=11,
    )
    fig.tight_layout()

    fig.savefig(f"{args.output_stub}.pdf", format="pdf", bbox_inches="tight", dpi=300, facecolor="white")
    fig.savefig(f"{args.output_stub}.png", format="png", bbox_inches="tight", dpi=300, facecolor="white")
    print(f"Saved {args.output_stub}.{{pdf,png}}")


if __name__ == "__main__":
    main()
