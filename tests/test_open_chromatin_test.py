import pandas as pd
import pytest
from intervaltree import IntervalTree

from bigwig_qc.open_chromatin_test import (
    _background_mean,
    call_significant,
    compute_power_and_fpr,
    poisson_enrichment_pvalue,
    sample_true_negative_loci,
    sample_true_positive_loci,
)


class FakeBigWig:
    """Minimal stand-in for pyBigWig.open(...): mean signal per queried region."""

    def __init__(self, chrom_sizes, region_means, default_mean=0.0):
        self._chrom_sizes = chrom_sizes
        self._region_means = region_means
        self._default_mean = default_mean

    def chroms(self, chrom):
        return self._chrom_sizes.get(chrom)

    def stats(self, chrom, start, end, type="mean", exact=True):
        return [self._region_means.get((chrom, start, end), self._default_mean)]


def test_sample_true_positive_loci_pools_promoters_and_ctcf():
    promoter_tree = {"chr1": IntervalTree()}
    promoter_tree["chr1"][1000:1100] = "TSS"
    ctcf_tree = {"chr1": IntervalTree()}
    ctcf_tree["chr1"][5000:5100] = 0.99

    sampled = sample_true_positive_loci(promoter_tree, ctcf_tree, n_loci=10, random_state=0)
    assert sampled.shape[0] == 2
    assert set(sampled.position) == {1050, 5050}


def test_sample_true_negative_loci_avoids_excluded_regions():
    exclude_tree = {"chr1": IntervalTree()}
    exclude_tree["chr1"][0:50_000] = "excluded"

    sampled = sample_true_negative_loci(
        chrom_sizes={"chr1": 100_000},
        exclude_trees=[exclude_tree],
        n_loci=20,
        min_distance_bp=1000,
        random_state=0,
    )
    assert (sampled.position > 51_000).all()


def test_background_mean_excludes_the_test_window_itself():
    bw = FakeBigWig(
        chrom_sizes={"chr1": 100_000},
        region_means={("chr1", 8000, 9000): 1.0, ("chr1", 11000, 12000): 3.0},
    )
    mean = _background_mean(bw, "chr1", start=9000, end=11000, chrom_length=100_000, flank_bp=1000)
    assert mean == pytest.approx(2.0)  # width-weighted average of 1.0 and 3.0 over equal widths


def test_poisson_enrichment_pvalue_significant_for_strong_enrichment():
    bw = FakeBigWig(
        chrom_sizes={"chr1": 100_000},
        region_means={
            ("chr1", 9900, 10100): 50.0,  # test window: strongly enriched
            ("chr1", 4900, 9900): 1.0,  # left background
            ("chr1", 10100, 15100): 1.0,  # right background
        },
    )
    p_value = poisson_enrichment_pvalue(bw, "chr1", position=10_000, window_bp=200, background_flank_bp=5000)
    assert p_value < 0.001


def test_poisson_enrichment_pvalue_not_significant_for_flat_signal():
    bw = FakeBigWig(
        chrom_sizes={"chr1": 100_000},
        region_means={
            ("chr1", 9900, 10100): 1.0,
            ("chr1", 4900, 9900): 1.0,
            ("chr1", 10100, 15100): 1.0,
        },
    )
    p_value = poisson_enrichment_pvalue(bw, "chr1", position=10_000, window_bp=200, background_flank_bp=5000)
    assert p_value > 0.05


def test_call_significant_handles_exact_zero_pvalue():
    # Regression test: p_value=0.0 must be treated as significant, not
    # silently replaced by a "not significant" default (0.0 is falsy).
    bw = FakeBigWig(
        chrom_sizes={"chr1": 100_000},
        region_means={
            ("chr1", 9900, 10100): 1000.0,
            ("chr1", 4900, 9900): 0.001,
            ("chr1", 10100, 15100): 0.001,
        },
    )
    loci = pd.DataFrame({"chrom": ["chr1"], "position": [10_000]})
    calls = call_significant(bw, loci, window_bp=200, background_flank_bp=5000)
    assert calls[0] == True  # noqa: E712 - explicit bool check reads clearer here


def test_compute_power_and_fpr_perfect_discrimination():
    bw = FakeBigWig(
        chrom_sizes={"chr1": 100_000},
        region_means={
            ("chr1", 9900, 10100): 50.0,
            ("chr1", 4900, 9900): 1.0,
            ("chr1", 10100, 15100): 1.0,
            ("chr1", 49900, 50100): 1.0,
            ("chr1", 44900, 49900): 1.0,
            ("chr1", 50100, 55100): 1.0,
        },
    )
    true_positives = pd.DataFrame({"chrom": ["chr1"], "position": [10_000]})
    true_negatives = pd.DataFrame({"chrom": ["chr1"], "position": [50_000]})

    result = compute_power_and_fpr(bw, true_positives, true_negatives, window_bp=200, background_flank_bp=5000)
    assert result["power"] == 1.0
    assert result["false_positive_rate"] == 0.0
