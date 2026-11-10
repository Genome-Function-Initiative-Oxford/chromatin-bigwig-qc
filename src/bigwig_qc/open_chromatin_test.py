"""Open-chromatin statistical power test (E5-A): does using a low-quality
bigWig for a standard open-chromatin significance test give unreliable
results, and does that unreliability track the learned quality score?

Test: local-background Poisson enrichment - the same style of test MACS/
LanceOtron use internally - is the signal in a window significantly elevated
over its local flanking background? Applied at two kinds of loci:
  - true positives: constant promoter/CTCF sites (bigwig_qc.references) -
    known to be genuinely open in essentially every real track.
  - true negatives: random background loci, confirmed not to overlap any
    constant site - should essentially never call significant.

Statistical power = fraction of true positives correctly called significant.
False-positive rate = fraction of true negatives incorrectly called
significant. Run on a degraded bigWig (bigwig_qc.degradation), power should
fall and/or the false-positive rate should rise as degradation increases -
a sharper, more concrete version of "garbage in, garbage out" than the
quality score alone demonstrates.
"""
import numpy as np
import pandas as pd
from scipy.stats import poisson

TEST_WINDOW_BP = 200
BACKGROUND_FLANK_BP = 5000
ALPHA = 0.05
DEFAULT_N_LOCI = 500


def _interval_midpoints(tree_by_chrom: dict) -> pd.DataFrame:
    """Flatten a bigwig_qc.references-style {chrom: IntervalTree} into one
    (chrom, position) row per interval, position = interval midpoint."""
    rows = []
    for chrom, tree in tree_by_chrom.items():
        for interval in tree:
            rows.append({"chrom": chrom, "position": (interval.begin + interval.end) // 2})
    return pd.DataFrame(rows)


def sample_true_positive_loci(
    promoter_tree: dict, ctcf_tree: dict, n_loci: int = DEFAULT_N_LOCI, random_state: int = 0
) -> pd.DataFrame:
    """Random sample of constant promoter + CTCF site midpoints."""
    all_loci = pd.concat([_interval_midpoints(promoter_tree), _interval_midpoints(ctcf_tree)], ignore_index=True)
    return all_loci.sample(n=min(n_loci, all_loci.shape[0]), random_state=random_state).reset_index(drop=True)


def sample_true_negative_loci(
    chrom_sizes: dict,
    exclude_trees: list,
    n_loci: int = DEFAULT_N_LOCI,
    min_distance_bp: int = BACKGROUND_FLANK_BP,
    random_state: int = 0,
) -> pd.DataFrame:
    """Random genomic positions confirmed not to fall within min_distance_bp of any interval in exclude_trees."""
    rng = np.random.default_rng(random_state)
    chroms = list(chrom_sizes.keys())

    rows = []
    attempts, max_attempts = 0, n_loci * 200
    while len(rows) < n_loci and attempts < max_attempts and chroms:
        attempts += 1
        chrom = chroms[rng.integers(0, len(chroms))]
        length = chrom_sizes[chrom]
        lo = min_distance_bp
        hi = max(lo + 1, length - min_distance_bp)
        position = int(rng.integers(lo, hi))

        excluded = False
        for tree_by_chrom in exclude_trees:
            tree = tree_by_chrom.get(chrom)
            if tree and tree.overlap(position - min_distance_bp, position + min_distance_bp):
                excluded = True
                break
        if excluded:
            continue
        rows.append({"chrom": chrom, "position": position})

    return pd.DataFrame(rows)


def _background_mean(bw, chrom: str, start: int, end: int, chrom_length: int, flank_bp: int):
    """Width-weighted mean signal in the two flanking windows, excluding [start, end) itself."""
    left = (max(0, start - flank_bp), start)
    right = (end, min(chrom_length, end + flank_bp))

    total, count = 0.0, 0
    for s, e in (left, right):
        if e > s:
            mean = bw.stats(chrom, s, e, type="mean", exact=True)[0]
            if mean is not None:
                total += mean * (e - s)
                count += e - s
    return total / count if count else None


def poisson_enrichment_pvalue(
    bw, chrom: str, position: int, window_bp: int = TEST_WINDOW_BP, background_flank_bp: int = BACKGROUND_FLANK_BP
):
    """Upper-tail local-background Poisson p-value for the window around `position`."""
    chrom_length = bw.chroms(chrom)
    if chrom_length is None:
        return None

    half_window = window_bp // 2
    start, end = max(0, position - half_window), min(chrom_length, position + half_window)
    if end <= start:
        return None

    observed_mean = bw.stats(chrom, start, end, type="mean", exact=True)[0]
    if observed_mean is None:
        return None
    observed_total = observed_mean * (end - start)

    background_mean = _background_mean(bw, chrom, start, end, chrom_length, background_flank_bp)
    if background_mean is None:
        return None
    expected = background_mean * (end - start)

    if expected <= 0:
        return 0.0 if observed_total > 0 else 1.0
    return float(poisson.sf(observed_total - 1, mu=expected))


def call_significant(bw, loci: pd.DataFrame, alpha: float = ALPHA, **kwargs) -> np.ndarray:
    # NOTE: must check "is not None" explicitly, not `p_value or default` - a
    # legitimate p_value of 0.0 (maximally significant) is falsy in Python and
    # would silently be replaced by the "not tested" default otherwise.
    calls = []
    for chrom, position in zip(loci.chrom, loci.position):
        p_value = poisson_enrichment_pvalue(bw, chrom, position, **kwargs)
        calls.append(p_value is not None and p_value < alpha)
    return np.array(calls)


def compute_power_and_fpr(
    bw, true_positive_loci: pd.DataFrame, true_negative_loci: pd.DataFrame, alpha: float = ALPHA, **kwargs
) -> dict:
    tp_calls = call_significant(bw, true_positive_loci, alpha=alpha, **kwargs)
    fp_calls = call_significant(bw, true_negative_loci, alpha=alpha, **kwargs)
    return {
        "power": float(tp_calls.mean()) if len(tp_calls) else float("nan"),
        "false_positive_rate": float(fp_calls.mean()) if len(fp_calls) else float("nan"),
        "n_true_positive_loci": int(len(tp_calls)),
        "n_true_negative_loci": int(len(fp_calls)),
    }
