"""Synthetic degradation of bigWig signal for the E1 controlled-degradation benchmark.

Raw reads are not available for every track, so degradation operates directly
on the bigWig signal instead of resampling reads - consistent with the rest of
this pipeline (and with LanceOtron itself, which is designed to work from
bigWig signal alone). Two operations are provided:

  - thin_signal: simulates down-sampling to a given read-depth fraction by
    treating each bin's (rounded) signal as a read count and drawing
    Binomial(count, fraction) - the same thinning used to subsample a Poisson
    process, so it reproduces both the depth reduction and the increased
    relative noise of true read subsampling.
  - inject_uniform_noise: simulates the "reads redistributed uniformly"
    background-noise injection by moving a fraction of the total signal mass
    out of its original bins and scattering it as Poisson-distributed counts
    spread evenly across every bin (including previously-empty ones).

See bigwig_io.py for applying these to a real .bw file on a fixed bin grid.
"""
import numpy as np


def thin_signal(values: np.ndarray, fraction: float, rng: np.random.Generator) -> np.ndarray:
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"fraction must be in [0, 1], got {fraction}")
    counts = np.clip(np.nan_to_num(values, nan=0.0), 0, None).round().astype(np.int64)
    return rng.binomial(counts, fraction).astype(np.float64)


def inject_uniform_noise(values: np.ndarray, fraction: float, rng: np.random.Generator) -> np.ndarray:
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"fraction must be in [0, 1], got {fraction}")
    values = np.clip(np.nan_to_num(values, nan=0.0), 0, None).astype(np.float64)
    if values.size == 0:
        return values

    total_signal = values.sum()
    moved_signal = total_signal * fraction
    kept = values * (1.0 - fraction)

    rate_per_bin = moved_signal / values.size
    scattered = rng.poisson(rate_per_bin, size=values.size).astype(np.float64)
    return kept + scattered
