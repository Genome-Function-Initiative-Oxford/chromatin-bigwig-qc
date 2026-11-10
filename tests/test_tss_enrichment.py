import pandas as pd
import pytest

from bigwig_qc.tss_enrichment import compute_tss_enrichment, sample_tss_positions


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


def _genes(rows):
    # columns 0=chrom, 3=start, 4=end, 6=strand - matching references.load_protein_coding_genes's raw GTF columns.
    return pd.DataFrame(rows)


def test_sample_tss_positions_uses_strand_correct_tss():
    genes = _genes(
        [
            {0: "chr1", 3: 1000, 4: 2000, 6: "+"},
            {0: "chr1", 3: 5000, 4: 6000, 6: "-"},
        ]
    )
    sampled = sample_tss_positions(genes, n_genes=2, random_state=0)
    tss_by_chrom = dict(zip(sampled.chrom, sampled.tss))
    assert set(sampled.tss) == {1000, 6000}


def test_sample_tss_positions_respects_n_genes_cap():
    genes = _genes([{0: "chr1", 3: i, 4: i + 100, 6: "+"} for i in range(50)])
    sampled = sample_tss_positions(genes, n_genes=10, random_state=0)
    assert sampled.shape[0] == 10


def test_sample_tss_positions_is_reproducible():
    genes = _genes([{0: "chr1", 3: i, 4: i + 100, 6: "+"} for i in range(50)])
    a = sample_tss_positions(genes, n_genes=10, random_state=42)
    b = sample_tss_positions(genes, n_genes=10, random_state=42)
    assert list(a.tss) == list(b.tss)


def test_compute_tss_enrichment_ratio_of_high_tss_signal_to_low_background():
    tss_positions = pd.DataFrame({"chrom": ["chr1"], "tss": [10_000]})
    bw = FakeBigWig(
        chrom_sizes={"chr1": 100_000},
        region_means={
            ("chr1", 9000, 11000): 4.0,  # TSS window
            ("chr1", 8000, 8100): 1.0,  # left background
            ("chr1", 11900, 12000): 1.0,  # right background
        },
    )
    score = compute_tss_enrichment(bw, tss_positions)
    assert score == pytest.approx(4.0)


def test_compute_tss_enrichment_skips_unknown_chromosome():
    tss_positions = pd.DataFrame({"chrom": ["chrUnknown", "chr1"], "tss": [500, 10_000]})
    bw = FakeBigWig(
        chrom_sizes={"chr1": 100_000},
        region_means={
            ("chr1", 9000, 11000): 2.0,
            ("chr1", 8000, 8100): 1.0,
            ("chr1", 11900, 12000): 1.0,
        },
    )
    score = compute_tss_enrichment(bw, tss_positions)
    assert score == pytest.approx(2.0)
