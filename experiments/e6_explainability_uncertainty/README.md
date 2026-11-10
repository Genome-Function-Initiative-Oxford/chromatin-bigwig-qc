# E6 - Explainability and uncertainty

Status: **done.** All three asks implemented and run: SHAP dependence/
interaction plots, split conformal prediction intervals, and
degradation-calibrated quality-score thresholds. Results in `results/e6/`.

**Paper section:** Materials and Methods / Explainability and uncertainty;
Results / "Interpretation of the quality score". Partly reused from CIBCB
(SHAP summary plot), partly new.

## Ask (from `../../../paper_TODO/todo.tex`)

- SHAP dependence and interaction value plots (beyond the CIBCB summary plot).
- Conformal prediction intervals for the quality score.
- Calibrated high/medium/low quality thresholds derived from the E1
  degradation series, rather than from visual inspection of score quantiles.

## Method and results: SHAP dependence and interaction values

`bigwig_qc.explain` gained `compute_shap_interaction_values` (pairwise SHAP
interaction values via `TreeExplainer.shap_interaction_values`),
`save_dependence_plot`, and `top_interacting_pair` (the feature pair with the
largest mean |off-diagonal interaction| across all samples).
`run_shap_dependence.py` runs this on the same model + full 490-track
dataset as the existing SHAP summary plot (`results/model_excl_pancreatic_pbmc.json`).

Top features by mean |SHAP|: `n_peaks` (772), `mean_enrichment_score` (585),
`percentage_coverage` (213) - consistent with the summary plot. The
dependence plot for `n_peaks` (`results/e6/dependence_n_peaks.png`) shows a
clean saturating relationship: SHAP contribution rises steeply with peak
count up to ~100,000 peaks, then plateaus - additional peaks beyond that
point barely move the model's quality assessment further.

**Top interacting feature pair: `n_peaks` x `mean_enrichment_score`** -
exactly the two most individually important features also interact most
strongly with each other (`results/e6/interaction_n_peaks_x_mean_enrichment_score.png`),
suggesting the model partly learns a joint "are there many peaks AND are
they strongly enriched" condition rather than treating the two as
independent signals.

## Method and results: conformal prediction intervals

`bigwig_qc.conformal` implements standard split conformal regression (Lei et
al. 2018): a 3-way train (60%) / calibrate (20%) / test (20%) split - using
*different* data for sizing the interval (calibration) and validating its
coverage (test) is essential; reusing one held-out set for both would
overstate how well-calibrated the result looks. The regressor keeps the
already-tuned hyperparameters from `results/model_excl_pancreatic_pbmc.json`
rather than re-running the 3-stage grid search.

```bash
python experiments/e6_explainability_uncertainty/run_conformal_calibration.py \
    --dataset data/interim/model_dataset.parquet \
    --model results/model_excl_pancreatic_pbmc.json \
    --output-dir results/e6
```

| alpha | nominal coverage | empirical coverage | interval half-width |
|---|---|---|---|
| 0.1 | 90% | **89.8%** | 541.5 |
| 0.2 | 80% | **81.6%** | 379.2 |

Both nominal coverage levels are matched almost exactly (within 1-2
percentage points, on a 98-track test set) - strong evidence the intervals
are correctly calibrated, not just plausible-looking. `results/e6/
conformal_intervals.png` visualises this: sorted test-set point predictions
with their interval band, true values marked covered (black) vs. not covered
(red) - the uncovered points are scattered across the whole score range
rather than concentrated in one region, consistent with a valid marginal
(not just average-case) coverage guarantee.

## Method and results: calibrated quality thresholds from E1

`bigwig_qc.thresholds.derive_calibrated_thresholds` defines two reference
populations directly from E1's degradation grid: tracks left undegraded
(`down_sample`, level=1.0) as "clearly high quality", and tracks degraded at
the heaviest level of *either* mode (`down_sample` level=0.01 pooled with
`inject_noise` level=0.80) as "clearly low quality". The medium/high
boundary is the low-percentile tail of the clean reference; the low/medium
boundary is the high-percentile tail of the degraded reference.

**Result, and why it does not give a clean 3-band split**: at both
percentile=5 and percentile=10, the two boundaries collapse to a single
value (`classify` degrades gracefully to a 2-band low/high split rather than
raising on the resulting duplicate bin edge - a real case hit on this
data, not just a hypothetical, with a regression test). Inspecting the raw
scores explains why: collection identity drives score variation on a scale
comparable to, or larger than, the degradation effect itself. H1_hESCs'
*undegraded* seed tracks score only ~2900-3400, while MPAL's *99%
down-sampled* seed tracks still score ~5080-7590, and even `inject_noise`'s
*heaviest* (80%) level leaves several catlas/MPAL/super_PBMC tracks at
~5200-5900 - all comfortably inside or above the "clean" reference's own
lower half. The two reference populations are not cleanly separated in score
space; they interleave, because *which collection a track came from*
matters as much to the score as how much it was synthetically degraded.

This connects directly to E2's finding (`total_bigwig_sum` and related
features behave as a near-constant per-collection signature that transfers
poorly across collections) - here it shows up again, from a different angle,
as collection identity confounding a *global* quality threshold. The
resulting 2-band split (`results/e6/quality_bands_p5.tsv`) agrees with the
original CIBCB tertile-quantile approach only 59.4% of the time - the two
thresholding philosophies disagree substantially, not just at the margins.

**Honest conclusion, not papered over**: a single, global, absolute
quality-score threshold - whether quantile-based (CIBCB's original approach)
or degradation-calibrated (this attempt) - cannot cleanly separate
"genuinely good" from "genuinely degraded" tracks when collection identity
itself explains much of the score's variance. A collection-relative or
per-batch threshold (e.g.\ percentile within collection, or a threshold
derived after regressing out collection identity) is the natural next step,
not yet implemented.

```bash
python experiments/e6_explainability_uncertainty/derive_quality_thresholds.py \
    --degradation-results results/e1_degradation_results.tsv \
    --dataset data/interim/model_dataset.parquet \
    --model results/model_excl_pancreatic_pbmc.json \
    --percentile 5 \
    --output results/e6/quality_bands_p5.tsv
```

## Still to do

- Collection-relative (rather than global) calibrated thresholds, given the
  collapse finding above.
- SHAP dependence plots for additional features beyond the top 3 shown here,
  if useful for the manuscript.
