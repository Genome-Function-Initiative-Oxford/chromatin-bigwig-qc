#!/usr/bin/env python
"""CLI: plot conformal prediction intervals on the held-out test set (E6) -
sorted by point prediction, with the true value marked covered/not covered.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

SERIES_BLUE = "#2a78d6"
SERIES_RED = "#c23b3b"
INK_PRIMARY = "#0b0b0b"
INK_MUTED = "#898781"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intervals", required=True, type=Path, help="conformal_test_intervals.tsv")
    parser.add_argument("--nominal-coverage", type=float, required=True, help="For the plot title/legend only")
    parser.add_argument("--output-stub", required=True, type=Path)
    args = parser.parse_args()

    data = pd.read_csv(args.intervals, sep="\t").sort_values("point_prediction").reset_index(drop=True)
    covered = (data.y_true >= data.lower) & (data.y_true <= data.upper)
    x = range(data.shape[0])

    fig, ax = plt.subplots(figsize=(10, 5), facecolor="white")
    ax.fill_between(x, data.lower, data.upper, color=SERIES_BLUE, alpha=0.15, label="Conformal interval")
    ax.plot(x, data.point_prediction, color=SERIES_BLUE, linewidth=1.5, label="Point prediction")
    ax.scatter(
        [i for i, c in zip(x, covered) if c], data.y_true[covered], s=10, color=INK_PRIMARY, alpha=0.6,
        label=f"Observed (covered, {covered.mean():.1%})",
    )
    ax.scatter(
        [i for i, c in zip(x, covered) if not c], data.y_true[~covered], s=18, color=SERIES_RED,
        label=f"Observed (NOT covered, {(~covered).mean():.1%})",
    )

    ax.set_xlabel("Test tracks, sorted by point prediction", color=INK_PRIMARY, fontsize=10)
    ax.set_ylabel("Constant CTCF + promoter recovery", color=INK_PRIMARY, fontsize=10)
    ax.set_title(
        f"E6: split conformal prediction intervals (nominal coverage={args.nominal_coverage:.0%})",
        color=INK_PRIMARY, fontsize=11,
    )
    ax.tick_params(colors=INK_MUTED)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(INK_MUTED)
    ax.legend(fontsize=8, frameon=False, loc="upper left")

    fig.tight_layout()
    fig.savefig(f"{args.output_stub}.pdf", format="pdf", bbox_inches="tight", dpi=300, facecolor="white")
    fig.savefig(f"{args.output_stub}.png", format="png", bbox_inches="tight", dpi=300, facecolor="white")
    print(f"Saved {args.output_stub}.{{pdf,png}}")


if __name__ == "__main__":
    main()
