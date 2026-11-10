# E3 - Baselines, ablations and significance testing

Status: **done** (baselines + feature-group ablations + significance tests).
MACS3 ablation not implemented (see below). Results in `results/e3/`.

**Paper section:** Materials and Methods / Evaluation protocol; Results /
"Comparison with baselines and feature ablations". New vs. CIBCB.

## Method

Implemented in `bigwig_qc.baselines`, run by `run_e3.py`. Every baseline and
every ablation is evaluated on **exactly the same** repeated per-collection
stratified folds (`bigwig_qc.evaluation.generate_stratified_folds`, the same
scheme as E2's `stratified_track_holdout`) - required so the corrected
resampled t-test (Nadeau & Bengio, 2003) comparison is paired, not comparing
different random splits. XGBoost uses the fixed hyperparameters from the
trained model (`results/model_excl_pancreatic_pbmc.json`), same as E2.

- **Baselines**: mean predictor, linear **and Poisson** regression on `n_peaks`
  only (the target is a count, so Poisson is the statistically appropriate
  GLM here, not just an alternative phrasing of "linear" - both are included,
  per `todo.tex`'s "linear/Poisson" wording), linear regression on
  `percentage_coverage` only, ElasticNet, Random Forest, LightGBM, MLP - each
  compared against XGBoost (`bigwig_qc.baselines.BASELINE_SPECS`).
- **Feature-group ablations**: `bigwig_qc.features.FEATURE_GROUPS` partitions
  the 15 features into 6 groups (`peak_score`, `shape`, `enrichment`,
  `coverage`, `peak_count`, `depth`); each is dropped in turn, keeping XGBoost
  fixed, and compared against the full model.
- **MACS3 ablation** (todo.tex's third ablation, substituting MACS3 peak
  calls for LanceOtron's) is **not implemented** - this repo has no MACS3
  peak calls for any track, and generating them is a separate, large
  data-processing task (re-running MACS3 genome-wide on 490 bigWigs). Flagged
  here rather than silently skipped.

A real bug was hit and fixed during this work: a loaded model's
`get_params()` includes a stale `feature_types` list sized to its *original*
training column count, which breaks the moment it's reused to fit a new
model on a different number of columns (exactly what the ablations do). Fixed
with `bigwig_qc.model.clean_hyperparams`, with a regression test
(`tests/test_model.py`) that reproduces the exact failure.

## Results: baselines

| Model | R2 | Spearman rho | vs. XGBoost (R2) |
|---|---|---|---|
| xgboost | 0.973 | 0.974 | (reference) |
| random_forest | 0.969 | 0.971 | p=0.221 (not significant) |
| lightgbm | 0.969 | 0.971 | p=0.147 (not significant) |
| elastic_net | 0.896 | 0.951 | p=4.1e-7 |
| linear_n_peaks_only | 0.421 | 0.877 | p=1.5e-7 |
| poisson_n_peaks_only | 0.219 | 0.877 | p=5.8e-6 |
| linear_coverage_only | 0.360 | 0.705 | p=8.4e-9 |
| mlp | 0.186 | 0.780 | p=2.0e-4 |
| mean_predictor | -0.005 | NaN (constant output - not a bug, see note) | p~0 |

`poisson_n_peaks_only` and `linear_n_peaks_only` have **identical** Spearman
rho (0.877) - expected, since both are monotonic functions of `n_peaks`
alone, so rank order is preserved regardless of link function. Their R2
differs (0.219 vs 0.421) because R2 is sensitive to the actual predicted
values, not just rank - the log-link GLM fits this particular target worse in
raw-error terms here, despite being the more principled choice for count
data. Either way, both are far below XGBoost's 0.973 (p=5.8e-6), so the
"does the score add information beyond n_peaks alone" conclusion is
unaffected by which link function is used.

**Does the score add information beyond `n_peaks` alone?** Yes, clearly: the
`n_peaks`-only linear baseline reaches R2=0.42 vs. the full model's 0.97, and
XGBoost beats it with p=1.5e-7. Same for coverage alone (R2=0.36). This
directly confirms, rather than just asserts (as the CIBCB paper does - "these
[simple baselines] do not capture the multi-dimensional structure..."), that
the learned score is not just a restatement of any single covariate.

**XGBoost is not uniquely justified over other tree ensembles**: Random
Forest and LightGBM are statistically indistinguishable from XGBoost
(p=0.22, p=0.15) - both land at R2=0.969 vs. XGBoost's 0.973. What matters is
using a non-linear, multivariate tree-based model at all, not the specific
choice of XGBoost. Worth stating plainly rather than overclaiming XGBoost's
specific necessity.

MLP underperforms substantially (R2=0.19, high run-to-run variance,
std=0.22) with its current plain architecture/defaults - a baseline
weakness, not a deep finding; a more careful MLP (architecture search,
different scaling) would likely do better, but that wasn't attempted here.

`mean_predictor`'s `spearman_rho=NaN` is correct, not a bug: a constant
predictor's output has zero variance, so rank correlation with it is
undefined (`scipy.stats.spearmanr` raises `ConstantInputWarning` and returns
NaN) - exactly what should happen, confirming the baseline is behaving as
intended (always predicting the training mean).

## Results: feature-group ablations

| Ablation | R2 | delta vs. full | p-value |
|---|---|---|---|
| full_model | 0.9728 | - | - |
| without_peak_score | 0.9726 | -0.0002 | 0.598 (not significant) |
| without_shape | 0.9710 | -0.0018 | 0.258 (not significant) |
| without_enrichment | 0.9708 | -0.0020 | 0.054 (borderline) |
| without_coverage | 0.9693 | -0.0035 | 0.067 (not significant) |
| without_depth | 0.9693 | -0.0035 | **0.007** (significant) |
| without_peak_count | 0.9691 | -0.0037 | **0.043** (significant) |

All effect sizes are small (R2 only ranges 0.969-0.973 across every ablation)
- consistent with the CIBCB paper's own SHAP finding that no single feature
group is individually indispensable; the model relies on combinations. Still,
`depth` (`total_bigwig_sum`) and `peak_count` (`n_peaks`/`n_peaks_0_5`/
`n_peaks_0_95`) are the two groups whose removal causes a *statistically
significant* drop, within this (easier, within-collection) evaluation scheme.

**This connects directly to E2's finding and is worth reading together with
it, not in isolation**: E2 found that `total_bigwig_sum` (and the features
that scale with it) behaves like a near-constant per-collection signature
that actively *hurts* cross-collection generalisation (leave-one-collection-out).
Here, within a collection, those same features carry real, statistically
significant predictive signal. Both can be true at once: `depth`/`peak_count`
are genuinely informative of quality *within* a processing batch, but also
encode batch identity in a way that fails to transfer *across* batches. This
is exactly the follow-up E2's README flagged as not yet done: rerun
leave-one-collection-out with these groups ablated, to see whether sacrificing
some within-collection signal improves cross-collection transfer. Still open.

## Running it

```bash
python experiments/e3_baselines_ablations/run_e3.py \
    --dataset data/interim/model_dataset.parquet \
    --model results/model_excl_pancreatic_pbmc.json \
    --output-dir results/e3
```

Takes ~9 minutes (MLP and Random Forest dominate). Requires the optional
`lightgbm` dependency: `pip install -e ".[baselines]"`.

Outputs (gitignored): `baseline_results.tsv`, `baseline_summary.tsv`
(mean/std), `baseline_summary_ci.tsv` (95% CI per model/metric, via
`bigwig_qc.evaluation.summarise_groups_with_ci` - the table format `todo.tex`
actually asks for), `baseline_ttests_vs_xgboost.tsv`, and the `ablation_*`
equivalents of all four.

## Still to do

- Table formatting for the manuscript (this README's tables, and the
  `*_summary_ci.tsv` files, are working summaries, not the final Table 2
  layout).
- The MACS3 ablation (blocked, see above).
- The cross-collection version of the depth/peak-count ablation (see "connects
  directly to E2's finding" above) - rerun `leave_one_collection_out` with
  those feature groups dropped.
