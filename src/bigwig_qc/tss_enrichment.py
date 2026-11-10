"""TSS enrichment: an established, independent QC metric for E4 (concordance
with established QC metrics), computed directly from bigWig signal - no raw
reads needed - against a sample of ALL protein-coding gene TSS, not the
constant/housekeeping promoter subset used as the regression target. Staying
independent of the training target is the point: this is meant to validate
the learned score against something it was never fit to predict.

Follows the ENCODE ATAC-seq pipeline's TSS enrichment definition: mean signal
in a window around the TSS, normalised by mean signal in a flanking
background region further from the TSS. Queries bw.stats(..., type="mean",
exact=True) per site - the same per-region query pattern bigwig_qc.features
already uses per peak, so this scales the same way the rest of the pipeline
already does.
"""
import numpy as np
import pandas as pd

TSS_WINDOW_BP = 1000
BACKGROUND_OFFSET_BP = 1900
BACKGROUND_WIDTH_BP = 100
DEFAULT_N_TSS_SAMPLE = 2000


def sample_tss_positions(genes: pd.DataFrame, n_genes: int = DEFAULT_N_TSS_SAMPLE, random_state: int = 0) -> pd.DataFrame:
    """A reproducible random sample of (chrom, tss) pairs across all genes.

    Sampling (rather than using all ~20,000 protein-coding genes genome-wide)
    keeps per-track compute cost bounded.
    """
    sampled = genes.sample(n=min(n_genes, genes.shape[0]), random_state=random_state)
    tss = np.where(sampled[6] == "+", sampled[3], sampled[4])
    return pd.DataFrame({"chrom": sampled[0].to_numpy(), "tss": tss})


def compute_tss_enrichment(bw, tss_positions: pd.DataFrame) -> float:
    """TSS enrichment = mean signal near the TSS / mean signal in flanking background."""
    tss_values = []
    background_values = []

    for chrom, tss in zip(tss_positions.chrom, tss_positions.tss):
        chrom_length = bw.chroms(chrom)
        if chrom_length is None:
            continue

        tss_start, tss_end = max(0, tss - TSS_WINDOW_BP), min(chrom_length, tss + TSS_WINDOW_BP)
        if tss_end > tss_start:
            value = bw.stats(chrom, tss_start, tss_end, type="mean", exact=True)[0]
            if value is not None:
                tss_values.append(value)

        for sign in (-1, 1):
            start, end = sorted((tss + sign * BACKGROUND_OFFSET_BP, tss + sign * (BACKGROUND_OFFSET_BP + BACKGROUND_WIDTH_BP)))
            start, end = max(0, start), min(chrom_length, end)
            if end > start:
                value = bw.stats(chrom, start, end, type="mean", exact=True)[0]
                if value is not None:
                    background_values.append(value)

    tss_mean = float(np.mean(tss_values)) if tss_values else float("nan")
    background_mean = float(np.mean(background_values)) if background_values else float("nan")
    return tss_mean / background_mean if background_mean else float("nan")
