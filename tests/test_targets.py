import pandas as pd
from intervaltree import IntervalTree

from bigwig_qc.targets import summarise_experiment


def _peak_calls():
    return pd.DataFrame(
        {
            "chrom": ["chr1", "chr1", "chr1"],
            "start": [100, 500, 900],
            "end": [200, 600, 1000],
            "overall_peak_score": [0.99, 0.4, 0.96],
        }
    )


def test_summarise_experiment_counts_promoter_and_ctcf_overlaps():
    promoter_tree = {"chr1": IntervalTree()}
    promoter_tree["chr1"][150:160] = "TSS"  # overlaps first peak (score 0.99)

    ctcf_tree = {"chr1": IntervalTree()}
    ctcf_tree["chr1"][950:960] = 0.99  # overlaps third peak (score 0.96)

    summary = summarise_experiment(_peak_calls(), promoter_tree, ctcf_tree)

    assert summary == {
        "n_promoters": 1,
        "n_promoters_0_5": 1,
        "n_promoters_0_95": 1,
        "n_ctcf": 1,
        "n_ctcf_0_5": 1,
        "n_ctcf_0_95": 1,
    }


def test_summarise_experiment_below_threshold_excluded_from_high_score_counts():
    promoter_tree = {"chr1": IntervalTree()}
    promoter_tree["chr1"][550:560] = "TSS"  # overlaps second peak (score 0.4)

    ctcf_tree = {"chr1": IntervalTree()}

    summary = summarise_experiment(_peak_calls(), promoter_tree, ctcf_tree)

    assert summary["n_promoters"] == 1
    assert summary["n_promoters_0_5"] == 0
    assert summary["n_promoters_0_95"] == 0
    assert summary["n_ctcf"] == 0
