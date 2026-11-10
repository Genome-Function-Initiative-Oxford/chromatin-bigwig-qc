# E8 - Transfer to other assays and species (optional)

Status: **partial, by deliberate scope decision.** DNase-seq half done,
including a real negative-transfer finding and a live track hub for visual
inspection. Mouse ATAC-seq half explicitly out of scope.

**Paper section:** Results / "Transfer to other assays and species
(optional)", marked optional in the roadmap.

## Ask, and what changed

The original ask had two parts: (1) ENCODE DNase-seq tracks, using ENCODE's
own audit flags as an external quality label; (2) mouse ATAC-seq, to test
generalisation beyond human hg38.

**Mouse ATAC-seq is out of scope.** It needs three new mm10/mm39 reference
resources (RefSeq GTF, a housekeeping-gene list, and a curated "constant CTCF
sites" list) - the GTF and gene list are straightforward downloads, but the
constant-CTCF list is not: the human one came from a separate modelling
pipeline (a CNN-based classifier of constant vs. cell-type-specific CTCF
sites) that is not reproducible here without that original work. This is a
bigger, separate undertaking, not attempted.

**DNase-seq was done, and it produced a real, concrete transfer finding** -
not a messy null result, but a mechanistically-explained one.

## Method

30 ENCODE DNase-seq experiments (GRCh38, released) were sampled across three
tiers defined by ENCODE's own audit system, specifically audits tied to
"SPOT score" (Signal Portion Of Tags - ENCODE's analogue of FRiP/peak
enrichment, i.e. the audit category most directly comparable to what this
pipeline's own covariates measure), not generic/administrative audits:
10 "clean" (no audits at all), 10 "warning" (WARNING-level "low spot score"),
10 "error" (ERROR-level "extremely low spot score"). For each, one
"read-depth normalized signal" bigWig (GRCh38, released) was downloaded
directly from ENCODE (`build_manifest.py`, `download_bigwigs.py`;
`experiments/e8_transfer_assays_species/manifest.tsv`; ~16GB,
`data/raw/encode_dnase/`, gitignored).

Each track was re-peak-called with LanceOtron and scored with the existing
trained model exactly as for any other track (`run_one_dnase.py`,
`scripts/slurm/run_e8_dnase_peaks.sh`, a 30-task SLURM array, `%4` throttle,
reusing E1's confirmed LanceOtron/XLA environment fixes - no new ones
needed). No new reference data was required: DNase-seq is still human/hg38,
so the pipeline's existing constant promoter/CTCF reference sets and trained
model apply unchanged. 30/30 tasks completed with no errors.

## Results: the score does not transfer, and the mechanism is clear

A Kruskal-Wallis test across the three audit tiers on the overall score was
not significant ($H=5.275$, $p=0.072$, $n=10$/tier), and the direction was
wrong: mean score was `warning` (6320) > `error` (6048) > `clean` (5645) -
not the expected `clean` > `warning` > `error`. Breaking this down by
individual covariate (Spearman $\rho$ vs. worsening ENCODE audit severity,
coded clean=0/warning=1/error=2) explains why:

| Feature | $\rho$ | $p$ | Direction |
|---|---|---|---|
| `mean_peak_score` | $-0.783$ | $<0.0001$ | correctly decreases with worse quality |
| `percentage_coverage` | $-0.585$ | $0.0007$ | correctly decreases with worse quality |
| `n_peaks` | $+0.623$ | $0.0002$ | **increases** with worse quality |
| overall `score` | $+0.123$ | $0.52$ | net: wrong direction, not significant |

The covariates that measure peak **confidence** (`mean_peak_score`) and
**signal concentration** (`percentage_coverage`) both correctly track
ENCODE's independent audit severity, even on an assay the model never saw.
But the overall score is dominated by `n_peaks` - the single most important
feature per E6's SHAP analysis, with a steeply saturating positive
relationship up to ~100,000 peaks. On DNase-seq tracks ENCODE itself flags
as low-quality, LanceOtron calls **more** peaks, not fewer - plausibly
because diffuse, low-confidence signal gets over-segmented into many
spurious low-score peaks rather than suppressed. The score rewards that
volume regardless of confidence, cancelling out the two covariates that do
transfer correctly. This is a genuine, mechanistically-explained negative
transfer result: not "the score is noisy on DNase-seq" but "the score's
specific reliance on raw peak count is actively misled by an assay-specific
peak-calling behaviour difference."

## A second, larger confound: pipeline/normalisation scaling

`total_bigwig_sum` (one of the model's own input features) differs by
**orders of magnitude** between the two track sets, for reasons unrelated to
quality:

| | mean | std | min | max |
|---|---|---|---|---|
| DNase (ENCODE "read-depth normalized signal") | 149,821 | 1,709 | 145,137 | 151,003 |
| ATAC (this pipeline's training data) | 63,987,480 | 63,246,730 | 981,863 | 191,513,700 |

ENCODE's "read-depth normalized signal" convention scales every DNase-seq
track to almost exactly the same total (~150,000, essentially no variance),
while this pipeline's ATAC-seq bigWigs are not normalised to a common scale
at all. The DNase tracks' values sit **below the minimum** of the ATAC
training range - the model isn't scoring out-of-domain data on this feature,
it is extrapolating entirely outside anything it was ever trained on.
`percentage_coverage` differs more modestly (DNase mean 0.254 vs.\ ATAC mean
0.322) but is still not on obviously comparable footing.

**We checked whether this is specific to DNase-seq or a more general
cross-pipeline problem**, by looking at whether ENCODE's own ATAC-seq
releases (same assay as the training data, different source pipeline) could
isolate the two confounds. They cannot, for a structural reason: ENCODE's
ATAC-seq pipeline (463 released GRCh38 experiments checked) **never
produces a "read-depth normalized signal" bigWig at all** - only "signal
p-value" and "fold change over control", both statistical/ratio
transformations rather than a (depth-normalised) coverage track. Neither is
the kind of track this pipeline's features were designed to operate on.
ENCODE's ATAC-seq audit vocabulary is also different (e.g. "low FRiP score",
"moderate library complexity" rather than "spot score"), so even the audit-
tier sampling design would need to change. This means the DNase-seq finding
above cannot be cleanly separated into "a DNase-vs-ATAC assay difference"
vs. "an ENCODE-vs-our-pipeline processing difference" using ENCODE's
standard outputs - any external track pulled from a different source
inherits that source's own processing conventions, which is a structural
limitation of this kind of cross-source comparison in general, not a
DNase-specific quirk. Not pursued further given the scope of this
experiment.

## Two live track hubs, for visual inspection

A number alone cannot be the final word on whether a scoring method is
trustworthy - an expert looking directly at the signal is a stronger check
than any statistic. Two separate UCSC Track Hubs (hg38) are built, one per
script, and published as saved UCSC Genome Browser sessions so each can be
opened and inspected directly, with no manual hub-loading steps:

- `build_track_hub.py`: all 30 ENCODE DNase-seq tracks, coloured by
  ENCODE's own audit tier, labelled with both the ENCODE tier and this
  pipeline's own quantile-based quality label (see below). Open at:
  [ENCODE DNase-seq, all 30 tracks](https://genome.ucsc.edu/s/simone.riva/ENCODE%20DNase%2Dseq%2C%20all%2030%20tracks)
- `build_atac_reference_hub.py`: 15 real ATAC-seq tracks from the actual
  training data (3 per quality band), so the same visual check can be done
  on the pipeline's actual home turf, not just the DNase-seq transfer case.
  Open at:
  [ATAC-seq training examples, 15 tracks](https://genome.ucsc.edu/s/simone.riva/ATAC%2Dseq%20training%20examples%2C%2015%20tracks)

These also serve as a live, interactive alternative to a static
genome-browser screenshot for the "visual inspection of representative
tracks" section - an interactive hub lets a reader inspect any locus, not
just one fixed screenshot.

### Quality-label terminology

Labels are **equal-count quantiles** of the score distribution on the
490-track ATAC-seq reference dataset this model was trained/evaluated on
(`bigwig_qc.thresholds.quantile_classify`/`quantile_thresholds`) - i.e. each
band contains the same *number* of tracks, not an equal-width slice of the
score range (which would give wildly unequal band sizes on this skewed
distribution). This is deliberately **not** re-derived from the small
external sample being classified (circular - any 30 scores trivially split
into equal-sized bands and say nothing about absolute quality).

Two granularities are supported; the track hub currently uses the 5-level
one:

- 3-level (`bad`/`decent`/`good`): thresholds 5454 / 7105.
- 5-level (`bad`/`poor`/`decent`/`good`/`excellent`): thresholds 4108 / 5906
  / 6878 / 7393.

## Still to do

- Mouse ATAC-seq (out of scope, see above).
- A genuine isolation of the assay-type vs.\ pipeline/normalisation
  confounds would need either (a) raw ENCODE ATAC-seq reads reprocessed
  through this pipeline's own upstream processing, or (b) a non-ENCODE
  source of DNase-seq/ATAC-seq data sharing our pipeline's normalisation
  convention - both bigger undertakings than attempted here.
