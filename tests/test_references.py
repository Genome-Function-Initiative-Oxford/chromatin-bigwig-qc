import pandas as pd

from bigwig_qc.references import build_ctcf_tree_from_peaks, build_promoter_tree_from_genes


def _gene(chrom="chr1", start=1000, end=2000, strand="+", gene_id="GENE1"):
    return pd.DataFrame({0: [chrom], 3: [start], 4: [end], 6: [strand], "gene_id": [gene_id]})


def test_promoter_tree_plus_strand_windows_around_start():
    tree = build_promoter_tree_from_genes(_gene(strand="+"), gene_ids={"GENE1"}, window_bp=100)
    assert len(tree["chr1"][999:1001]) == 1
    assert len(tree["chr1"][1200:1300]) == 0


def test_promoter_tree_minus_strand_windows_around_end():
    tree = build_promoter_tree_from_genes(_gene(strand="-"), gene_ids={"GENE1"}, window_bp=100)
    assert len(tree["chr1"][1999:2001]) == 1
    assert len(tree["chr1"][500:600]) == 0


def test_promoter_tree_excludes_non_housekeeping_genes():
    tree = build_promoter_tree_from_genes(_gene(gene_id="OTHER"), gene_ids={"GENE1"}, window_bp=100)
    assert len(tree["chr1"]) == 0


def test_ctcf_tree_filters_by_recurrence():
    peaks = pd.DataFrame({0: ["chr1", "chr1"], 1: [100, 500], 2: [200, 600], 4: [0.99, 0.5]})
    tree = build_ctcf_tree_from_peaks(peaks, min_recurrence=0.99)
    assert len(tree["chr1"]) == 1
    hits = tree["chr1"][150]
    assert next(iter(hits)).data == 0.99
