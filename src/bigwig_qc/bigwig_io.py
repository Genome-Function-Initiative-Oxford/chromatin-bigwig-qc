"""Bin a bigWig's signal onto a fixed-width grid, degrade it, and write it back out.

Used by scripts/degrade_bigwig.py for the E1 controlled-degradation benchmark.
The thinning/noise-injection math itself lives in degradation.py and is agnostic
of bin size; this module handles the bigWig <-> binned-array conversion and the
mean <-> bin-total unit conversion that math needs.
"""
from pathlib import Path

import numpy as np
import pyBigWig

from . import config
from .degradation import inject_uniform_noise, thin_signal


def chrom_sizes(bw) -> dict:
    return {chrom: length for chrom, length in bw.chroms().items() if chrom in config.AUTOSOMES_AND_SEX_CHROMS}


def read_binned_mean_signal(
    bw, chrom: str, chrom_length: int, bin_size: int = config.DEGRADATION_BIN_SIZE_BP
) -> np.ndarray:
    """Per-bin mean coverage across `chrom`, as a dense array (0.0 where uncovered).

    Computed from bw.values() + vectorized numpy binning, not
    bw.stats(..., nBins=...). Two things were measured on a real bigWig during
    E1 debugging:
      - bw.stats(..., exact=True) forces per-base rescanning and took 6+
        CPU-minutes for chr1 alone without finishing.
      - bw.stats(..., exact=False) is fast (253s for chr1) but uses the
        bigWig's pre-built zoom-level summaries, which don't necessarily align
        with our exact 100bp grid - confirmed to measurably smooth across bin
        boundaries versus the true per-bin mean (correlation 0.96, not 1.0,
        against this function's values()-based computation over the same
        chromosome). It's a fast approximation, not an exact computation.
    Fetching raw values and binning ourselves is both exact (the literal mean
    over our exact bin boundaries, not an approximation of it) and ~9x faster
    than exact=False in that same comparison (~28s vs ~253s for chr1).
    Confirmed end-to-end on a real bigWig (degrade_bigwig_file, all
    chromosomes): 265.5s total, down from 2881.5s (48min) with the previous
    exact=False implementation - an ~11x speedup, not just a per-chromosome
    estimate.
    """
    values = np.nan_to_num(np.asarray(bw.values(chrom, 0, chrom_length), dtype=np.float64), nan=0.0)
    pad = (-len(values)) % bin_size
    if pad:
        values = np.concatenate([values, np.zeros(pad)])
    return values.reshape(-1, bin_size).mean(axis=1)


def degrade_binned_signal(
    mean_signal: np.ndarray, mode: str, level: float, bin_size: int, rng: np.random.Generator
) -> np.ndarray:
    """Thin or noise-inject a per-bin mean-signal array, in bin-total units, back to means."""
    bin_totals = mean_signal * bin_size
    if mode == "down_sample":
        degraded_totals = thin_signal(bin_totals, fraction=level, rng=rng)
    elif mode == "inject_noise":
        degraded_totals = inject_uniform_noise(bin_totals, fraction=level, rng=rng)
    else:
        raise ValueError(f"Unknown degradation mode: {mode!r} (expected 'down_sample' or 'inject_noise')")
    return degraded_totals / bin_size


def degrade_bigwig_file(
    input_path: Path,
    mode: str,
    level: float,
    seed: int,
    output_path: Path,
    bin_size: int = config.DEGRADATION_BIN_SIZE_BP,
) -> None:
    """End-to-end: read input_path, degrade every chromosome, write output_path."""
    rng = np.random.default_rng(seed)
    bw = pyBigWig.open(str(input_path))
    sizes = chrom_sizes(bw)

    binned_means = {}
    for chrom, length in sizes.items():
        mean_signal = read_binned_mean_signal(bw, chrom, length, bin_size)
        binned_means[chrom] = degrade_binned_signal(mean_signal, mode, level, bin_size, rng)
    bw.close()

    write_binned_bigwig(output_path, sizes, binned_means, bin_size)


def write_binned_bigwig(
    output_path: Path, sizes: dict, binned_means: dict, bin_size: int = config.DEGRADATION_BIN_SIZE_BP
) -> None:
    bw = pyBigWig.open(str(output_path), "w")
    bw.addHeader(sorted(sizes.items()))
    for chrom in sorted(binned_means):
        values = binned_means[chrom]
        length = sizes[chrom]
        starts = np.arange(len(values), dtype=np.int64) * bin_size
        ends = np.minimum(starts + bin_size, length)

        keep = values > 0
        if not keep.any():
            continue
        bw.addEntries(
            [chrom] * int(keep.sum()),
            starts[keep].astype(int).tolist(),
            ends=ends[keep].astype(int).tolist(),
            values=values[keep].astype(float).tolist(),
        )
    bw.close()
