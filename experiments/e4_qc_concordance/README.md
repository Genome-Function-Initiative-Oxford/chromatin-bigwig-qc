# E4 - Concordance with established QC metrics

Status: **done** for the two metrics that are feasible without raw reads
(TSS enrichment, FRiP). NSC/RSC and fragment-length periodicity blocked (see
below). Genome-browser disagreement views not started.

**Paper section:** Materials and Methods / Comparison with established QC
metrics; Results / "Concordance with established QC metrics". New vs. CIBCB.

## Ask

- Compute, for every reprocessed track: TSS enrichment, FRiP (fraction of
  reads in peaks), fragment-length periodicity, NSC/RSC, and (for scATAC)
  fragments per cell.
- Figure: correlation of the learned quality score against each established
  metric.
- Genome-browser views of cases where the score and an established metric
  disagree.

## What's feasible without raw reads (same constraint as E1)

Raw reads/BAMs aren't available for every track (see E1). Of the four
established metrics asked for:

- **FRiP** - already available. `bigwig_qc.features.compute_bigwig_features`'s
  `percentage_coverage` *is* FRiP computed from bigWig signal rather than
  reads (fraction of total signal falling inside called peaks).
- **TSS enrichment** - genuinely computable from bigWig signal alone
  (`bigwig_qc.tss_enrichment`, new module). ENCODE ATAC-seq pipeline's
  standard definition: mean signal in a window around the TSS, normalised by
  mean signal in a flanking background region further away. Computed over a
  random sample of 2000 sites across **all** protein-coding gene TSS (not the
  constant/housekeeping promoter subset used as the regression target) - this
  independence from the training target is the point.
- **NSC/RSC** - blocked. Needs strand-resolved read coverage; bigWigs here
  are already strand-merged.
- **Fragment-length periodicity** - blocked. Needs paired-end insert-size
  distributions from raw reads/fragments.

## Results

Computed for all 490 tracks (`results/e4_tss_enrichment.tsv`, SLURM job
`6423556` on the `long` partition - ran overnight, no GPU needed). Joined
against the learned quality score (`model_excl_pancreatic_pbmc.json`'s
predictions) in `results/e4_merged_concordance.tsv`:

| Metric | Spearman rho | p-value | Pearson r |
|---|---|---|---|
| TSS enrichment | **0.86** | 5.1e-142 | 0.78 |
| FRiP (`percentage_coverage`) | 0.73 | 5.5e-82 | 0.62 |

Both strong, highly significant positive correlations - plotted in
`results/e4_concordance.pdf`/`.png` (score on x-axis, metric on y-axis, linear
fit overlaid).

**TSS enrichment is the more important result of the two.** It correlates
*more strongly* than FRiP despite being computed from an entirely independent
reference set (all protein-coding genes genome-wide) that the model never saw
during training - FRiP, by contrast, *is* one of the model's own 15 input
features (`percentage_coverage`), so its correlation with the score is
expected almost by construction and is a weaker form of validation (closer to
an internal-consistency check than independent concordance). TSS enrichment
correlating this strongly with a score that never used TSS-enrichment-style
features as direct training input is the real "does the score mean what we
think it means" result for this experiment.

One secondary observation worth a caption note: the FRiP panel shows a
"fan" shape at high quality scores - a subset of high-scoring tracks still
has comparatively low FRiP. Not investigated further here; worth a sentence
in the paper rather than over-interpreting from this analysis alone.

## Running it

```bash
# 1. Compute TSS enrichment for every track (CPU-only, ~8h - use the SLURM job)
sbatch scripts/slurm/run_e4_tss_enrichment.sh
# (sbatch must be invoked from the repo root - see scripts/slurm/run_e4_tss_enrichment.sh)

# 2. Join against the quality score and compute correlations
python experiments/e4_qc_concordance/compute_concordance.py \
    --dataset data/interim/model_dataset.parquet \
    --tss-enrichment results/e4_tss_enrichment.tsv \
    --model results/model_excl_pancreatic_pbmc.json \
    --output results/e4_merged_concordance.tsv

# 3. Plot
python experiments/e4_qc_concordance/plot_concordance.py \
    --merged results/e4_merged_concordance.tsv \
    --output-stub results/e4_concordance
```

## Still to do

- Genome-browser views of score/metric-disagreement cases - not started.
- NSC/RSC and fragment-length periodicity stay unimplemented (blocked by lack
  of raw reads) - flag explicitly in the paper rather than silently omitting.
