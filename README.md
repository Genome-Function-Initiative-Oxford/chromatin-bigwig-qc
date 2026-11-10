# chromatin-bigwig-qc

[![PyPI version](https://img.shields.io/pypi/v/chromatin-bigwig-qc.svg)](https://pypi.org/project/chromatin-bigwig-qc/)

Learning quality scores for chromatin accessibility bigWig tracks.

This repository is the codebase for a Frontiers in Artificial Intelligence
submission ("Computational Intelligence for Bioinformatics, Computational
Biology, and Biomedical Informatics" research topic):

> A neural-network peak-caller's recovery of cell-type-invariant ("constant")
> promoters and CTCF sites is used as ground-truth quality signal for a
> chromatin-accessibility bigWig track. An XGBoost regressor is trained to
> predict that recovery from cheap bigWig/peak-call summary statistics, so a
> track's quality can be scored without the constant-site annotation at
> inference time.

## Status

Released: `.gitignore`, `LICENSE` (MIT), and `pyproject.toml` metadata are
all in place, and the full test suite passes (86/86) via `pytest tests/`.

## Repository layout

```
src/bigwig_qc/      installable package: the core, reusable pipeline code
scripts/            thin CLIs around the package (SLURM-ready)
scripts/slurm/       sbatch wrappers for running the pipeline over a full collection
notebooks/          exploratory notebooks (not the source of truth for the pipeline)
experiments/         one folder per new experiment (E1-E9), see below
tests/               unit tests for the package (no cluster data required)
data/                README documenting external data dependencies
```

## Environment

On the cluster, everything runs via the `python-cbrg` module:

```bash
module load python-cbrg
pip install -e .           # registers bigwig_qc as an importable package
pip install -e ".[dev]"    # + pytest, for running tests/
pip install -e ".[baselines]"  # + lightgbm, needed only for experiment E3
```

## Pipeline quickstart

The three-stage pipeline, implemented in `src/bigwig_qc/` and exposed as
CLIs in `scripts/`:

```bash
# 1. Per-bigWig QC covariates (bigwig_qc.features)
python scripts/run_features.py \
    --input-bigwig path/to/sample.bw \
    --input-peak-calls path/to/sample_L-tron.bed \
    --output data/interim/features/<collection>.<sample>.json

# 2. Constant promoter/CTCF recovery counts, i.e. the regression targets (bigwig_qc.references + bigwig_qc.targets)
python scripts/run_targets.py \
    --input path/to/collection/peak_calls_dir \
    --output data/interim/targets/<collection>.tsv

# 3. Join (1) and (2) into one modelling dataset (bigwig_qc.dataset)
python scripts/build_dataset.py \
    --summary-dir data/interim/features \
    --targets-dir data/interim/targets \
    --output data/interim/model_dataset.parquet

# 4. Train + explain the quality-score model (bigwig_qc.model, bigwig_qc.explain)
python scripts/train_model.py \
    --dataset data/interim/model_dataset.parquet \
    --model-output results/model.json \
    --shap-output-stub results/shap_summary_plot
```

For the full 11-collection sweep, use the SLURM wrappers instead:
`sbatch scripts/slurm/run_features.sh` then `sbatch scripts/slurm/run_targets.sh`.
They set `PYTHONPATH` to `src/` themselves, so no local `pip install` is
required to run them on the cluster.

`data/` and `results/` are gitignored - they hold regenerated pipeline
outputs, not source.

## Experiments roadmap

Each folder under `experiments/` corresponds to one new analysis required for
the Frontiers extension.

| Folder | Paper section | Summary |
|---|---|---|
| `e1_degradation_benchmark` | Controlled degradation benchmark | **Done.** Full run: 900/900 (track, mode, level, seed) combinations across 30 seed tracks. Four real bugs found and fixed during debugging (XLA/`ptxas` crash, a masked `libdevice` error, a `bigwig_qc.bigwig_io` binning performance+correctness bug - ~11x speedup, exact not approximate, and a LanceOtron output-filename truncation bug on multi-dot degraded filenames) - see `scripts/slurm/run_e1_degradation.sh`'s header comment for the full history. Score shows a clean, strong, monotonic dose-response to `inject_noise` but is largely flat under `down_sample` except at the most extreme level. See its README for the full results and figure. |
| `e2_leave_one_collection_out` | Evaluation protocol | **Done.** Leave-one-collection-out + repeated group k-fold + a within-collection stratified holdout (`bigwig_qc.evaluation`). Found `pancreatic_pbmc` has experimental problems (excluded from `config.COLLECTION_NAMES`); found cross-collection generalisation is unreliable for several small/differently-processed collections (batch/depth confound, not a covariate-quality failure - confirmed within-collection R2/rho ~0.97 once the batch is represented). See its README for full findings. |
| `e3_baselines_ablations` | Evaluation protocol | **Done.** Baselines (mean/linear/ElasticNet/RF/LightGBM/MLP) + feature-group ablations, with corrected resampled t-tests (`bigwig_qc.baselines`). XGBoost beats simple baselines (p<0.001) but is statistically indistinguishable from Random Forest/LightGBM; `depth`/`peak_count` feature groups are the only ones whose removal is significant - connects to E2's batch-confound finding. MACS3 ablation blocked (no MACS3 peak calls in this repo). See its README. |
| `e4_qc_concordance` | Comparison with established QC metrics | **Done** (for the 2 metrics feasible without raw reads). TSS enrichment (independent of the training target) correlates with the learned score at Spearman rho=0.86 (p=5e-142); FRiP (a training feature) at rho=0.73. NSC/RSC and fragment-length periodicity blocked (need raw reads). Genome-browser disagreement views not started. See its README. |
| `e5_downstream_utility` | Downstream utility | **Done.** Redefined away from an original cell-type-classification plan (rejected by the user - doesn't match what they'd actually use to validate this, and the first candidate collection had 1 track/cell-type anyway): tests whether low-quality tracks produce unreliable *statistical* conclusions (`bigwig_qc.open_chromatin_test` - local-background Poisson enrichment test, statistical power/false-positive rate at true-positive constant CTCF/promoter sites and true-negative background loci, across E1's degradation grid). Full 900-row run complete: `inject_noise` (not `down_sample`) inflates FPR substantially above baseline at every severity tested, while power stays flat - the real "garbage in, garbage out" failure mode. Joined against E1's scores: score strongly predicts power/FPR within `down_sample` (rho=0.80) but not within `inject_noise` (rho~0.08-0.10, n.s.) - an important caveat on how well the score predicts downstream reliability specifically. Baseline FPR (0.18) is also substantially above nominal alpha=0.05, a flagged limitation of the test's calibration. See its README for full results. |
| `e6_explainability_uncertainty` | Explainability and uncertainty | **Done.** SHAP dependence/interaction plots (`n_peaks` x `mean_enrichment_score` is the top-interacting pair); split conformal prediction intervals (89.8%/81.6% empirical coverage at 90%/80% nominal - well calibrated); degradation-calibrated quality thresholds from E1, which collapse to a 2-band (not 3-band) split because collection identity confounds a global threshold as much as degradation severity does - connects directly to E2's batch-confound finding, agrees with a simpler quantile-based tertile approach only 59% of the time. See its README. |
| `e7_processing_effects` | Processing effects | **Partial, by decision.** Reprocessing/renormalisation deliberately out of scope (most collections are bigwig-only; reprocessing a large published collection is often infeasible). Variance-decomposition (mixed-effects model, `score ~ 1 + (1\|collection)`) done: ICC=0.498 - collection identity alone explains ~half the score's total variance, quantifying what E2/E6 found qualitatively. See its README. |
| `e8_transfer_assays_species` | Transfer to other assays/species (optional) | **Partial, by decision.** Mouse ATAC-seq out of scope (needs a from-scratch constant-CTCF reference). DNase-seq done: 30 real ENCODE tracks scored, finding a genuine negative-transfer result - `mean_peak_score`/`percentage_coverage` correctly track ENCODE's own audit severity (Spearman rho=-0.78/-0.59) but the overall score fails because `n_peaks` moves the *wrong* direction on low-quality DNase tracks (rho=+0.62); also found a large cross-pipeline normalisation confound (ENCODE's depth-normalisation clamps `total_bigwig_sum` to ~150K vs. our data's 1M-191M range). Built a live UCSC track hub (30 DNase + 15 ATAC reference tracks, labelled by both ENCODE's tier and this pipeline's own quantile-based quality label) for direct visual inspection. See its README. |
| `e9_release_packaging` | Implementation and availability | **Done.** GitHub repository released (this repo); Zenodo DOI pending. |

See `data/README.md` for the external (cluster-only) data paths the pipeline
depends on.
