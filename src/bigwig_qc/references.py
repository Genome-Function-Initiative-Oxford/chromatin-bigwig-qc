"""Reference genomic features used to define the quality-score targets.

A bigWig's recovery of two sets of cell-type-invariant elements - constant
housekeeping-gene promoters and constant CTCF sites - is treated as ground
truth for how good the underlying peak calls (and so the bigWig) are. See
targets.py for how these trees are turned into per-experiment recovery counts.

Each "load_*"/"build_constant_*" function does file I/O against config.py
defaults; the "build_*_from_*" functions are pure and unit-testable.
"""
from pathlib import Path

import pandas as pd
from intervaltree import IntervalTree

from . import config


def load_chrom_rename_map(path: Path = config.CHROM_RENAME_FILE) -> dict:
    """Map RefSeq accession names (e.g. NC_000001.11) to chr1-style names."""
    rename = {}
    with open(path) as fh:
        for line in fh:
            old, new = line.strip().split("\t")
            rename[old] = new
    return rename


def load_protein_coding_genes(
    gtf_path: Path = config.REFSEQ_GTF,
    chrom_rename_path: Path = config.CHROM_RENAME_FILE,
) -> pd.DataFrame:
    """Load protein_coding/miRNA/lnc_RNA gene records from a RefSeq GTF.

    Returns the raw GTF columns (0=chrom, 3=start, 4=end, 6=strand, 8=attributes)
    plus a parsed "gene_id" column, restricted to autosomes + chrX/chrY.
    """
    chrom_rename = load_chrom_rename_map(chrom_rename_path)

    genes = pd.read_csv(gtf_path, skiprows=5, header=None, sep="\t")
    genes = genes[genes[2] == "gene"]
    genes = genes[genes[8].str.contains("protein_coding|miRNA|lnc_RNA")]
    genes = genes.replace({0: chrom_rename})
    genes = genes[genes[0].isin(config.AUTOSOMES_AND_SEX_CHROMS)]
    genes[3] = genes[3].astype(int)
    genes[4] = genes[4].astype(int)
    genes["gene_id"] = (
        genes[8].str.split(";", expand=True)[0].str.split(" ", expand=True)[1].str.strip('"')
    )
    return genes


def _tss_interval(gene_row, window_bp: int) -> tuple:
    strand, start, end = gene_row[6], gene_row[3], gene_row[4]
    if strand == "+":
        return start - window_bp, start + 1
    return end - 1, end + window_bp


def build_promoter_tree_from_genes(
    genes: pd.DataFrame, gene_ids: set, window_bp: int = config.PROMOTER_WINDOW_BP
) -> dict:
    """Per-chromosome IntervalTree of TSS +/- window_bp, restricted to gene_ids."""
    trees = {}
    for chrom in set(genes[0]):
        trees[chrom] = IntervalTree()
        chrom_genes = genes[(genes[0] == chrom) & genes.gene_id.isin(gene_ids)]
        for _, gene in chrom_genes.iterrows():
            lo, hi = _tss_interval(gene, window_bp)
            trees[chrom][lo:hi] = "TSS"
    return trees


def build_constant_promoter_tree(
    genes: pd.DataFrame,
    housekeeping_xlsx: Path = config.HOUSEKEEPING_GENES_XLSX,
    window_bp: int = config.PROMOTER_WINDOW_BP,
) -> dict:
    """Constant-promoter tree for the housekeeping genes listed in the GTEx sheet.

    Source: https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1010295
    """
    housekeeping_genes = pd.read_excel(housekeeping_xlsx, sheet_name="GTEx")
    return build_promoter_tree_from_genes(genes, set(housekeeping_genes.Genes), window_bp)


def build_ctcf_tree_from_peaks(
    ctcf_peaks: pd.DataFrame, min_recurrence: float = config.CONSTANT_CTCF_MIN_RECURRENCE
) -> dict:
    """Per-chromosome IntervalTree of CTCF sites recurring in >= min_recurrence of samples.

    ctcf_peaks columns are positional: 0=chrom, 1=start, 2=end, 4=recurrence fraction.
    """
    recurrent = ctcf_peaks[ctcf_peaks[4] >= min_recurrence]

    trees = {}
    for chrom in set(recurrent[0]):
        trees[chrom] = IntervalTree()
        for _, region in recurrent[recurrent[0] == chrom].iterrows():
            trees[chrom][region[1]:region[2]] = region[4]
    return trees


def build_constant_ctcf_tree(
    ctcf_bed: Path = config.CONSTANT_CTCF_BED,
    min_recurrence: float = config.CONSTANT_CTCF_MIN_RECURRENCE,
) -> dict:
    ctcf_peaks = pd.read_csv(ctcf_bed, sep="\t", header=None)
    return build_ctcf_tree_from_peaks(ctcf_peaks, min_recurrence)
