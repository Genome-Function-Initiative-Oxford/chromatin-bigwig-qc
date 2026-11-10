# E5 - Downstream utility of quality-aware filtering

Status: **done**, including the score-vs-downstream-reliability correlation
against E1's completed run.

**Paper section:** Materials and Methods / Downstream utility; Results /
"Quality-aware filtering improves downstream learning". New vs. CIBCB.

## Ask, and why it was redefined

The original plan suggested cell-type classification from accessibility at a fixed
peak set as *an example* downstream task, with settings: all tracks;
score-filtered; randomly filtered to the same size; score-weighted.

Rejected after discussion: not how the user would actually validate the
score's usefulness in practice, and the first candidate collection
(`super_PBMC`) turned out to have exactly 1 track per cell type anyway -
"filtering" would just delete entire classes rather than clean up noisy
replicates, which doesn't match what quality-filtering is supposed to
demonstrate.

**Redefined as (A): does using a low-quality track for a standard
statistical test on open chromatin produce unreliable conclusions, and does
the learned quality score predict that unreliability?** This is a sharper,
more concrete version of "garbage in, garbage out" than classification
accuracy would show, and reuses E1's existing degradation machinery entirely
(no GPU/LanceOtron needed for this test - it operates directly on bigWig
signal).

(Also discussed, not built: (B) train a sequence-to-signal regression model
using tracks as targets, and show low-quality targets hurt the trained
model - directly matches the user's actual research (FireCrest/Enformer-style
work) but needs real model training, likely GPU, probably the FireCrest/HYPER
pipeline rather than anything built here. Bigger lift, not started.)

## Method

`bigwig_qc.open_chromatin_test`: a local-background Poisson enrichment test
(the same style of test MACS/LanceOtron use internally) - is the signal in a
window significantly elevated over its local flanking background? Applied at:

- **true positives**: constant promoter/CTCF site midpoints
  (`sample_true_positive_loci`) - known to be genuinely open in essentially
  every real track.
- **true negatives**: random background loci confirmed not to overlap any
  constant site (`sample_true_negative_loci`).

**Statistical power** = fraction of true positives correctly called
significant. **False-positive rate** = fraction of true negatives incorrectly
called significant. Computed on bigWigs degraded via E1's exact degradation
grid (`bigwig_qc.degradation`/`bigwig_io`, reusing `manifest.tsv`) - no
LanceOtron/GPU involved, pure CPU/bigWig-signal work.

`experiments/e5_downstream_utility/run_open_chromatin_power_test.py` ties it
together: for each manifest row, degrade the track, test both loci sets,
record power/FPR.

## A real bug found and fixed along the way

Developing and timing this experiment is what surfaced a serious performance
bug in `bigwig_qc.bigwig_io.read_binned_mean_signal` (used by E1 too):
`bw.stats(..., exact=False)` took 48 minutes per track genome-wide and was
confirmed to be a real approximation (not just slow) - it uses the bigWig's
zoom-level summaries, which don't align with our exact 100bp grid and
measurably smooth across bin boundaries (correlation 0.96, not 1.0, against
the true per-bin mean). Replaced with `bw.values()` + vectorized numpy
binning: exact AND ~11x faster (265-330s/track genome-wide, confirmed both in
isolation and at full pipeline scale). See `bigwig_io.py`'s docstring and
`scripts/slurm/run_e1_degradation.sh`'s comments for the full story - this
fix benefits E1 directly too.

One side effect worth noting: re-running the same (track, mode=down_sample,
level=1.0, seed) combination before and after this fix gave slightly
different `power` (0.88 -> 0.82, same `false_positive_rate`=0.16) - expected,
since the old binning was a real (if small) approximation; the post-fix
numbers should be treated as the trustworthy ones.

## Results: full-scale run (30 seed tracks, 900 rows, 2000+2000 loci each)

Mean +/- std power / false-positive rate, pooled across all 30 tracks, per
degradation mode and level:

| degradation_mode | level | power (mean) | power (std) | FPR (mean) | FPR (std) |
|---|---|---|---|---|---|
| down_sample | 0.01 (heaviest) | 0.859 | 0.089 | 0.108 | 0.065 |
| down_sample | 0.05 | 0.905 | 0.032 | 0.135 | 0.078 |
| down_sample | 0.10 | 0.911 | 0.031 | 0.144 | 0.079 |
| down_sample | 0.25 | 0.915 | 0.032 | 0.157 | 0.074 |
| down_sample | 0.50 | 0.917 | 0.033 | 0.167 | 0.070 |
| down_sample | 1.00 (undegraded baseline) | 0.918 | 0.033 | 0.177 | 0.068 |
| inject_noise | 0.10 (lightest) | 0.934 | 0.014 | 0.238 | 0.072 |
| inject_noise | 0.30 | 0.934 | 0.014 | 0.230 | 0.079 |
| inject_noise | 0.50 | 0.933 | 0.014 | 0.221 | 0.085 |
| inject_noise | 0.80 (heaviest) | 0.929 | 0.014 | 0.196 | 0.086 |

**Core finding: down-sampling and noise injection produce qualitatively
different failure modes, not just different magnitudes of the same one.**

- **down_sample**: power and FPR move *together* as degradation worsens - both
  drop (power 0.918 -> 0.859, FPR 0.177 -> 0.108 from undegraded to the
  heaviest level). The power loss is concentrated almost entirely at the most
  extreme level (0.01, i.e. 1% of reads kept) - levels 0.05-1.0 are all within
  ~0.012 of each other. Consistent with a uniformly more conservative test
  under fewer total counts: with less signal everywhere, the Poisson
  enrichment test calls fewer windows significant across the board, true
  positives and false positives alike. This is *not* the "unreliable, falsely
  confident conclusions" failure mode - it is the test becoming too
  conservative, if anything, under read-depth loss alone.
- **inject_noise**: power stays essentially flat and tight (0.929-0.934,
  std only ~0.014, far tighter than down_sample's ~0.03-0.09) across every
  tested noise strength - noise barely touches detection of real signal.
  **FPR, however, sits well above the undegraded baseline (0.177) at every
  noise level tested (0.196-0.238)**, including the lightest (0.10). This *is*
  the "garbage in, garbage out" failure mode `todo.tex`'s redefinition set out
  to demonstrate: injected noise contamination - not general degradation, and
  specifically not read-depth loss - is what inflates false-positive calls on
  a standard open-chromatin test, while leaving power (and therefore the
  illusion of a working track) intact.

**FPR calibration question, now resolved (not a small-sample artefact)**: the
undegraded baseline's FPR is 0.177 (std 0.068) across all 30 real tracks at
n=2000 true-negative loci each - a precise estimate, not pilot noise. The
Poisson local-background enrichment test as implemented here is genuinely
over-calling significance at nominal alpha=0.05, roughly 3.5x the nominal
rate. This is a real calibration limitation of the test/background-loci setup
(worth flagging explicitly in the manuscript as a caveat on absolute FPR
values), but it does **not** undermine the core comparative finding above -
the *relative* FPR inflation from noise injection (0.177 -> 0.196-0.238) is
the result that matters for the "does quality-filtering avoid unreliable
conclusions" question, and that comparison is unaffected by a shared,
systematic baseline miscalibration.

Earlier pilot numbers (1 track, 50+50 loci: power=0.82, FPR=0.16) and an
interim single-track check at 2000+2000 loci (power=0.855, FPR=0.091) are
both superseded by the full 30-track numbers above.

## Running it

```bash
python experiments/e5_downstream_utility/run_open_chromatin_power_test.py \
    --manifest experiments/e1_degradation_benchmark/manifest.tsv \
    --output results/e5_open_chromatin_power.tsv
```

Note: the CLI's own default is 500+500 loci, not 2000+2000 (a pre-existing
doc/code mismatch, left as-is) - pass `--n-true-positive-loci`/
`--n-true-negative-loci` explicitly for 2000+2000 (what the full-scale run
below actually uses). At ~4.5-5min/track for degradation (confirmed: 251-273s
per track across several runs; `compute_power_and_fpr` itself is negligible,
1.7s even at 2000+2000 loci), the full 900-row manifest takes ~70-75 CPU-hours
sequentially - `--row-start`/`--row-end` allow chunking.

**Full-scale run**: `scripts/slurm/run_e5_power_test.sh` - a 30-task SLURM
array (CHUNK_SIZE=30 rows/task, `--array=1-30%6`), `--partition=long` (CPU
only, no GPU/LanceOtron needed here unlike E1). Writes one TSV per chunk to
`results/e5_jobs/<task_id>.tsv`; aggregated with E1's generic
`experiments/e1_degradation_benchmark/aggregate_results.py --results-dir
results/e5_jobs --output results/e5_open_chromatin_power.tsv`. **Complete**:
30/30 chunks, 900/900 rows. One chunk (array index 18,
`catlas_fetal.Adrenal_Cortical`) hit the original `--time=03:00:00` limit at
27/30 rows (that track's degradation step ran ~345-360s/row vs. ~250-330s
elsewhere - a real per-track speed difference, not a bug; several other
chunks finished within ~10% of the same limit, confirming the original budget
had essentially no safety margin). Fixed by bumping to `--time=05:00:00` and
resubmitting just that one chunk (`sbatch --array=18
scripts/slurm/run_e5_power_test.sh`), which completed cleanly.

## Results: does the learned score predict downstream reliability?

`correlate_score_with_power_fpr.py` joins this experiment's power/FPR against
E1's learned `score`, on the shared `(collection_name, experiment_name,
degradation_mode, level, seed)` key (exact join, 900/900 rows matched - both
experiments run over the identical E1 manifest grid):

```bash
python experiments/e5_downstream_utility/correlate_score_with_power_fpr.py \
    --e1-results results/e1_degradation_results.tsv \
    --e5-results results/e5_open_chromatin_power.tsv \
    --output results/e5_score_vs_power_fpr.tsv
```

| | score vs. power (rho) | score vs. FPR (rho) |
|---|---|---|
| pooled (n=900) | 0.589 (p=4e-85) | 0.321 (p=4e-23) |
| down_sample only (n=540) | **0.803** (p=5e-123) | 0.411 (p=2e-23) |
| inject_noise only (n=360) | 0.083 (p=0.12, n.s.) | -0.101 (p=0.055, n.s.) |

**Important, more nuanced than the pooled number suggests**: the score
strongly predicts power *within* `down_sample` (rho=0.80) - unsurprising,
since both the score and read-depth-driven power/FPR respond to the same
underlying factor (how much signal survived down-sampling). But **within
`inject_noise` specifically - the degradation mode E5's own result above
identifies as the actual source of unreliable (FPR-inflating) conclusions -
the score has essentially no correlation with either power or FPR** (both
non-significant at alpha=0.05). The pooled correlation is real but is almost
entirely driven by the down_sample subset, not by the score discriminating
noise-contamination severity.

Taken together with E1's own finding (the score *does* decline monotonically
with `inject_noise` severity, see `experiments/e1_degradation_benchmark/
README.md`), this means: the score responds to noise contamination in
aggregate, but that response doesn't line up track-by-track with which
specific tracks will actually produce an inflated false-positive rate on a
downstream test. Worth stating as an honest limitation in the paper rather
than overclaiming the score as a general-purpose "will this track give
unreliable conclusions" predictor - it is a stronger predictor of degradation
severity than of downstream statistical reliability specifically for the
noise-contamination failure mode.

## Still to do

- Option (B) (sequence-to-signal target corruption) not started - bigger,
  separate undertaking, likely needs the FireCrest/HYPER pipeline.
- Consider investigating the Poisson test's FPR miscalibration itself (window/
  flank size, background-loci sampling) if the manuscript wants to report
  well-calibrated absolute FPR numbers rather than just the relative
  down_sample-vs-inject_noise comparison.
