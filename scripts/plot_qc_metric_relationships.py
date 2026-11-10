#!/usr/bin/env python
"""CLI: plot background-noise distribution and positional promoter/CTCF
recovery profiles as one merged multi-panel figure for this manuscript, plus
a correlation matrix between QC metrics.

Panels:
(a) Background noise (1 - percentage_coverage) distribution across all tracks.
(b)/(c) Mean bigWig signal in bins around constant-promoter TSS / constant-CTCF
    site midpoints, for a sample of tracks per quality tertile (equal-count
    quantile bands: bad/decent/good). Each track's
    profile is normalised by that track's own total signal before averaging,
    since raw depth otherwise swamps the comparison (E7/E8 found this same
    depth confound). Profiles are centred on the reference interval's
    midpoint, not strand-oriented to the biological TSS - a deliberate
    simplification for a descriptive figure, not a primary quantitative claim.
(d) Spearman correlation matrix of representative QC metrics (one per
    FEATURE_GROUPS group), background noise, and the regression target.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyBigWig
from scipy.stats import spearmanr

from bigwig_qc import config, references
from bigwig_qc.features import FEATURE_COLUMNS
from bigwig_qc.model import load_model
from bigwig_qc.thresholds import quantile_classify

SERIES_BLUE = "#2a78d6"
SERIES_BLUE_DARK = "#184f95"
TIER_COLORS = {"bad": "#c23b3b", "decent": "#c7960a", "good": "#1a8a4a"}
TIER_ORDER = ["bad", "decent", "good"]
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"

REPRESENTATIVE_FEATURES = [
    "mean_peak_score",
    "mean_shape_score",
    "mean_enrichment_score",
    "percentage_coverage",
    "n_peaks",
    "total_bigwig_sum",
]


def _style_axis(ax):
    ax.tick_params(colors=INK_MUTED)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(INK_MUTED)


def _sample_midpoints(tree, n_loci, rng):
    """Flatten a per-chrom IntervalTree dict to (chrom, midpoint) pairs, sample n_loci."""
    all_loci = [
        (chrom, (iv.begin + iv.end) / 2)
        for chrom, subtree in tree.items()
        for iv in subtree
    ]
    rng.shuffle(all_loci)
    return all_loci[:n_loci]


def _select_tracks_per_tier(data, n_per_tier, rng):
    data = data.copy()
    data["tier"] = quantile_classify(data.score, data.score, TIER_ORDER)
    selected = {}
    for tier in TIER_ORDER:
        group = data[data.tier == tier]
        # One track per distinct collection first (diversity), then fill by
        # random sample within the tier.
        picked = group.drop_duplicates("collection_name").sample(
            n=min(n_per_tier, group.collection_name.nunique()), random_state=rng.integers(1e9)
        )
        if picked.shape[0] < n_per_tier:
            remaining = group[~group.index.isin(picked.index)]
            extra = remaining.sample(
                n=min(n_per_tier - picked.shape[0], remaining.shape[0]), random_state=rng.integers(1e9)
            )
            picked = pd.concat([picked, extra])
        selected[tier] = picked
    return selected


def _track_profile(bigwig_path, loci, half_window, n_bins):
    bw = pyBigWig.open(str(bigwig_path))
    try:
        total_signal = sum(bw.stats(chrom, type="sum", exact=True)[0] or 0.0 for chrom in bw.chroms())
        if total_signal == 0:
            return None
        profiles = []
        for chrom, mid in loci:
            if chrom not in bw.chroms():
                continue
            chrom_len = bw.chroms()[chrom]
            start, end = int(mid - half_window), int(mid + half_window)
            if start < 0 or end > chrom_len:
                continue
            values = bw.stats(chrom, start, end, type="mean", nBins=n_bins, exact=True)
            values = [v if v is not None else 0.0 for v in values]
            profiles.append(values)
        if not profiles:
            return None
        mean_profile = np.nanmean(np.array(profiles), axis=0)
        return mean_profile / total_signal * 1e6  # signal per million total, a.u.
    finally:
        bw.close()


def _plot_profile_panel(ax, selected, loci, half_window, n_bins, title):
    x = np.linspace(-half_window, half_window, n_bins) / 1000.0  # kb
    for tier in TIER_ORDER:
        tier_profiles = []
        for row in selected[tier].itertuples():
            bigwig_path = config.BIGWIG_COLLECTION_ROOT / row.collection_name / "bigwigs" / f"{row.experiment_name}.bw"
            if not bigwig_path.exists():
                continue
            profile = _track_profile(bigwig_path, loci, half_window, n_bins)
            if profile is not None:
                tier_profiles.append(profile)
        if not tier_profiles:
            continue
        mean_profile = np.mean(np.array(tier_profiles), axis=0)
        ax.plot(x, mean_profile, color=TIER_COLORS[tier], linewidth=2, label=f"{tier} (n={len(tier_profiles)} tracks)")
    ax.set_xlabel("Distance from locus midpoint (kb)", color=INK_PRIMARY, fontsize=10)
    ax.set_ylabel("Mean signal\n(per million total, a.u.)", color=INK_PRIMARY, fontsize=10)
    ax.set_title(title, color=INK_PRIMARY, fontsize=11)
    ax.legend(fontsize=8, frameon=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path, help="data/interim/model_dataset.parquet")
    parser.add_argument("--model", required=True, type=Path, help="results/model_excl_pancreatic_pbmc.json")
    parser.add_argument("--output-stub", required=True, type=Path, help="Written as <stub>.pdf and <stub>.png")
    parser.add_argument("--n-per-tier", type=int, default=5)
    parser.add_argument("--n-loci", type=int, default=250)
    parser.add_argument("--half-window-bp", type=int, default=2000)
    parser.add_argument("--n-bins", type=int, default=41)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    data = pd.read_parquet(args.dataset)
    model = load_model(args.model)
    data["score"] = model.predict(data[list(FEATURE_COLUMNS)])
    data["noise"] = 1 - data.percentage_coverage
    data["target"] = data.n_promoters + data.n_ctcf

    print("Building reference trees (constant promoters, constant CTCF sites)...")
    genes = references.load_protein_coding_genes()
    promoter_tree = references.build_constant_promoter_tree(genes)
    ctcf_tree = references.build_constant_ctcf_tree()

    promoter_loci = _sample_midpoints(promoter_tree, args.n_loci, rng)
    ctcf_loci = _sample_midpoints(ctcf_tree, args.n_loci, rng)
    print(f"Sampled {len(promoter_loci)} promoter loci, {len(ctcf_loci)} CTCF loci.")

    selected = _select_tracks_per_tier(data, args.n_per_tier, rng)
    for tier in TIER_ORDER:
        print(f"Tier '{tier}': {selected[tier].shape[0]} tracks selected "
              f"({selected[tier].collection_name.nunique()} distinct collections).")

    fig, axes = plt.subplots(2, 2, figsize=(11, 9), facecolor="white")

    ax = axes[0][0]
    _style_axis(ax)
    ax.hist(data.noise, bins=30, color=SERIES_BLUE, alpha=0.75, edgecolor=SERIES_BLUE_DARK)
    ax.axvline(data.noise.median(), color=INK_PRIMARY, linestyle="--", linewidth=1.2)
    ax.text(
        0.97, 0.94, f"median = {data.noise.median():.2f}",
        transform=ax.transAxes, va="top", ha="right", fontsize=9, color=INK_SECONDARY,
    )
    ax.set_xlabel("Background noise\n(fraction of signal outside called peaks)", color=INK_PRIMARY, fontsize=10)
    ax.set_ylabel("Number of tracks", color=INK_PRIMARY, fontsize=10)
    ax.set_title("(a) Background noise distribution (490 tracks)", color=INK_PRIMARY, fontsize=11)

    print("Computing promoter recovery profiles (this reads real bigWigs - may take a few minutes)...")
    ax = axes[0][1]
    _style_axis(ax)
    _plot_profile_panel(ax, selected, promoter_loci, args.half_window_bp, args.n_bins, "(b) Constant-promoter recovery profile")

    print("Computing CTCF recovery profiles...")
    ax = axes[1][0]
    _style_axis(ax)
    _plot_profile_panel(ax, selected, ctcf_loci, args.half_window_bp, args.n_bins, "(c) Constant-CTCF recovery profile")

    ax = axes[1][1]
    corr_columns = REPRESENTATIVE_FEATURES + ["noise", "target", "score"]
    corr = data[corr_columns].corr(method="spearman")
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr_columns)))
    ax.set_yticks(range(len(corr_columns)))
    ax.set_xticklabels(corr_columns, rotation=45, ha="right", fontsize=8, color=INK_PRIMARY)
    ax.set_yticklabels(corr_columns, fontsize=8, color=INK_PRIMARY)
    for i in range(len(corr_columns)):
        for j in range(len(corr_columns)):
            ax.text(
                j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center",
                fontsize=7, color="white" if abs(corr.values[i, j]) > 0.6 else INK_PRIMARY,
            )
    ax.set_title("(d) Spearman correlation between QC metrics", color=INK_PRIMARY, fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(f"{args.output_stub}.pdf", format="pdf", bbox_inches="tight", dpi=300, facecolor="white")
    fig.savefig(f"{args.output_stub}.png", format="png", bbox_inches="tight", dpi=300, facecolor="white")
    print(f"Saved {args.output_stub}.{{pdf,png}}")


if __name__ == "__main__":
    main()
