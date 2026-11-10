import pandas as pd

from bigwig_qc.features import FEATURE_COLUMNS, compute_bigwig_features_from_handle


class FakeBigWig:
    """Minimal stand-in for pyBigWig.open(...), keyed on the (chrom, start, end) stats calls."""

    def __init__(self, chrom_sums, segment_sums):
        self._chrom_sums = chrom_sums
        self._segment_sums = segment_sums

    def stats(self, chrom, start=None, end=None, type="sum", exact=True):
        if start is None:
            return [self._chrom_sums[chrom]]
        return [self._segment_sums[(chrom, start, end)]]


def _peak_calls():
    return pd.DataFrame(
        {
            "chrom": ["chr1", "chr1"],
            "start": [0, 100],
            "end": [10, 110],
            "overall_peak_score": [0.97, 0.3],
            "shape_score": [0.5, 0.2],
            "enrichment_score": [0.8, 0.1],
        }
    )


def test_compute_bigwig_features_from_handle():
    bw = FakeBigWig(
        chrom_sums={"chr1": 1000.0},
        segment_sums={("chr1", 0, 10): 400.0, ("chr1", 100, 110): 100.0},
    )

    features = compute_bigwig_features_from_handle(bw, _peak_calls())

    assert set(features.keys()) == set(FEATURE_COLUMNS)
    assert features["n_peaks"] == 2
    assert features["n_peaks_0_5"] == 1
    assert features["n_peaks_0_95"] == 1
    assert features["percentage_coverage"] == (500.0 / 1000.0)
    assert features["percentage_coverage_0_95"] == (400.0 / 1000.0)
    assert features["mean_peak_length"] == 10.0
    assert features["total_bigwig_sum"] == 1  # round(1000 / 1000)
