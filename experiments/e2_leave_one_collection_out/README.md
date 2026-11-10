# E2 - Generalisation across collections

Status: **implemented and run** against the current dataset (490 tracks, 10
collections, `pancreatic_pbmc` excluded - see below). Results in
`results/e2_excl_pancreatic_pbmc/`. Not yet turned into paper figures.

**Paper section:** Materials and Methods / Evaluation protocol; Results /
"The quality score generalises across data collections". New vs. CIBCB
(reuses CIBCB Fig. 5, predicted vs. observed, now with CIs).

## Data: `pancreatic_pbmc` excluded

Removed from `bigwig_qc.config.COLLECTION_NAMES` (502 -> 490 tracks). Known
experimental problems with this collection, corroborated here: held out, it
scored R2=-3.23, Spearman rho=0.14 (near-chance rank correlation) - far worse
than any other collection. Root cause: it mixes two sub-populations (pancreas/
islet samples and separate `pbmc_HemaCare-*` samples), and the HemaCare
tracks have `mean_peak_score` averaging 0.09 versus 0.52 across the rest of
the dataset (only 65/490 other tracks score that low) - consistent with
genuinely noisy peak calls, not a modelling artefact.

Retrained model (`results/model_excl_pancreatic_pbmc.json`) on the cleaned
490-track dataset: test R2=0.975 (up from 0.96 with `pancreatic_pbmc`
included). This is a *separate* file from `results/model.json` - the latter
is left untouched while E1's SLURM array job is still reading it.

## Method: three complementary evaluation schemes

Implemented in `bigwig_qc.evaluation`, run together by `run_e2.py`. All three
use the fixed hyperparameters from the trained model rather than re-running
`bigwig_qc.model`'s 3-stage grid search inside every fold (would be 61+ full
grid searches - a separate, much more expensive question about the *tuning
procedure's* sensitivity to the split, not what E2 asks).

1. **`leave_one_collection_out`** - train on 10 collections, predict the
   11th, for every collection in turn. Tests generalisation to an entirely
   unseen processing batch. Reuses/extends CIBCB Fig. 5.
2. **`repeated_group_kfold`** - collections repeatedly shuffled and split
   into 5 folds x 10 repeats, for confidence intervals on the aggregate
   metrics. Same question as (1), at a coarser group size (multiple
   collections held out per fold).
3. **`stratified_track_holdout`** - a few tracks (`test_size`, default 20%)
   are held out from *every* collection each repeat, instead of whole
   collections. The model has already seen each collection's processing
   baseline during training, so this isolates genuine track-to-track quality
   signal from the cross-collection extrapolation question in (1)/(2).

(1)/(2) and (3) answer different questions and are **both kept** - neither
supersedes the other. See "Why these are different findings" below.

## Results

### 1 + 2: cross-collection generalisation is poor for several collections

| Collection | n | R2 | Spearman rho | low_n |
|---|---|---|---|---|
| catlas_fetal | 110 | 0.95 | 0.98 | |
| catlas_adult | 111 | 0.86 | 0.96 | |
| calderon | 175 | 0.72 | 0.86 | |
| Days7_10_13_17 | 15 | 0.72 | 0.88 | flagged |
| immune_cell | 15 | 0.65 | 0.88 | flagged |
| ludwig2019 | 8 | 0.86 | 0.79 | flagged |
| super_PBMC | 19 | -1.69 | 0.43 | flagged |
| CAD | 10 | -1.95 | 0.20 | flagged |
| MPAL | 23 | -5.30 | 0.58 | flagged |
| H1_hESCs | 4 | -153 | -0.20 | flagged |

Repeated group 5-fold x10 (holding out ~2 collections at a time): R2=-0.21
[-1.04, 0.62], Spearman rho=0.84 [0.78, 0.90].

**Mechanism** (verified, not just hypothesised - see conversation history /
`data/interim/model_dataset.parquet` for the underlying numbers): predictions
for the poorly-generalising collections show a *systematic* bias (every track
under-predicted by a near-constant amount: CAD -301, MPAL -412, H1_hESCs
-824, super_PBMC -199) and *compressed* variance relative to the true range -
the classic signature of a tree ensemble extrapolating past its training
range. These four collections have `n_peaks` 2-5x higher than the rest of the
dataset's average, and `total_bigwig_sum` (~total sequencing depth) forms
near-disjoint per-collection bands (e.g. CAD 62-72M vs. MPAL 174-191M vs.
H1_hESCs 30-35M) - i.e. `total_bigwig_sum` behaves like a near-constant
per-batch signature, not a portable quality signal, so no amount of *other*
differently-shifted collections in training lets the model extrapolate to a
specific new one.

**This does not fully transfer between the "weird" collections either**:
training on MPAL alone (23 tracks) and testing on super_PBMC is *worse* than
full leave-one-collection-out (R2=-2.23 vs -1.69, Spearman rho=**-0.23** vs
+0.43 - the ranking flips to anti-correlated). So it's not simply "not enough
representative collections in training" - MPAL and super_PBMC are each
shifted in their own idiosyncratic way, and training on the full diverse
467-track set beats training on a smaller "similar" collection. This also
can't be fully separated from small-n training instability (23 points is a
weak regime for hyperparameters tuned on ~470 rows) without further probing.

### 3: within-collection, the covariates do track real quality

Pooled across all collections: R2=0.97, Spearman rho=0.97 [0.97, 0.98] -
matching the original CIBCB random-split number. Once a collection's
batch/processing baseline is represented in training, the model recovers
genuine track-to-track quality differences well.

Per-collection breakdown still shows CAD (R2=-0.22), H1_hESCs (R2=-6.56) and
super_PBMC (R2=-0.08) negative even in-sample - but all three are `low_n`
(4-19 distinct tracks): with so few points and a narrow true-quality range
(e.g. CAD's targets span only 7765-8431), R2 is fragile regardless of
evaluation scheme. Every one of the four problem collections' Spearman rho
improved under this scheme vs. leave-one-collection-out (e.g. MPAL 0.58 ->
0.71, H1_hESCs -0.20 -> 0.25), consistent with the batch-extrapolation
mechanism above, not a fundamental failure of the covariates.

## Why these are different, both-valid findings (not a contradiction)

- Leave-one-collection-out / repeated group k-fold answer: "can this model
  score a track from a completely new processing batch it's never seen
  anything like?" - **no, not reliably**, and that's a real, practical
  limitation worth stating plainly (motivates comprehensive training-data
  coverage across processing batches as a deployment requirement, not an
  optional nicety).
- Stratified track holdout answers: "once a batch is represented, do the
  covariates discriminate real quality within it?" - **yes**, R2/rho ~0.97.
- `low_n` (< `bigwig_qc.evaluation.LOW_N_THRESHOLD` = 25 tracks):
  per-collection R2 is not a reliable metric at this scale regardless of
  which scheme is used - report pooled/aggregate metrics for small
  collections rather than their individual R2. Flagged automatically in
  `leave_one_collection_out_metrics` and
  `stratified_track_holdout_per_collection_metrics`.

## Running it

```bash
python experiments/e2_leave_one_collection_out/run_e2.py \
    --dataset data/interim/model_dataset.parquet \
    --model results/model_excl_pancreatic_pbmc.json \
    --output-dir results/e2_excl_pancreatic_pbmc
```

Outputs (all gitignored, regenerate via the command above):
`loco_predictions.tsv`, `loco_metrics_per_collection.tsv`,
`repeated_kfold_metrics.tsv`, `repeated_kfold_summary_ci.tsv`,
`stratified_holdout_predictions.tsv`, `stratified_holdout_overall_metrics.tsv`,
`stratified_holdout_summary_ci.tsv`,
`stratified_holdout_metrics_per_collection.tsv`.

## Still to do

- Figures: per-collection performance (extends CIBCB Fig. 5) and predicted
  vs. observed with CIs - not yet built (no plotting code written for E2 yet).
- Once E1 finishes and `results/model.json` is no longer being read live,
  decide whether to promote `model_excl_pancreatic_pbmc.json` to be the
  "real" `results/model.json`, and whether E1's seed tracks (which included 2
  `pancreatic_pbmc` tracks) need regenerating.
- Possible follow-up ablation (not yet run): retrain without
  `total_bigwig_sum` (and other depth-scaling features like raw `n_peaks`)
  and rerun leave-one-collection-out, to test whether removing the
  depth-as-shortcut features closes the cross-collection generalisation gap -
  natural overlap with E3's feature-ablation work.
