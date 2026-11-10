# E7 - Processing effects

Status: **partial, by deliberate scope decision.** The data-only half
(variance decomposition) is done. The reprocessing/renormalisation half is
explicitly out of scope - see below.

**Paper section:** Materials and Methods / Processing effects; Results /
"Quality across collections and processing routes" (extends CIBCB Fig. 7).
New vs. CIBCB.

## Ask, and what changed

The original ask had two parts: (1) reprocess a CATlas subset with CATCH-UP,
or apply a common normalisation across collections, and re-score; (2) a
mixed-effects model, `score ~ collection + processing + cell_type`.

**Part (1) is out of scope, by decision, not by default.** In practice, most
chromatin accessibility collections are only available as already-processed
bigWigs - the raw reads needed for genuine reprocessing are frequently
unavailable, and even when they are, reprocessing a large published
collection from scratch is often impractical. Treating "control for
processing by reprocessing" as a hard requirement would make this kind of
analysis infeasible for most real-world bigWig collections - which is itself
part of the motivation for a QC approach that works directly on bigWig
signal rather than requiring raw-read access (see Discussion in the
manuscript). We did not attempt the "apply a common signal normalisation and
re-call peaks" alternative either, since it would need a new LanceOtron/GPU
SLURM run of similar scope to E1's, and the data-only part below already
answers the more fundamental question (how large is the collection effect)
without it.

**Part (2) could not be fit exactly as specified.** This dataset has no
independent, cross-comparable "processing route" variable, and
`experiment_name` (the closest thing to a "cell_type" field) is a free-text,
per-collection label - 472 distinct values across 490 tracks, i.e. close to
one per row - not a controlled vocabulary comparable across collections. It
cannot serve as a meaningful random-effect grouping factor on its own. What
*is* fittable, and directly answers "how much does collection identity
alone explain", is a random-intercept model, `score ~ 1 + (1 | collection_name)`,
reporting the intraclass correlation (ICC).

## Method and result

`bigwig_qc.variance_decomposition.intraclass_correlation` computes
`group_variance / (group_variance + residual_variance)` from a fitted
`statsmodels` `MixedLM`; `run_mixed_effects.py` fits the model on the
490-track dataset (the trained quality score, not the raw recovery target)
and reports it.

```bash
python experiments/e7_processing_effects/run_mixed_effects.py \
    --dataset data/interim/model_dataset.parquet \
    --model results/model_excl_pancreatic_pbmc.json \
    --output results/e7_mixed_effects.tsv
```

**ICC = 0.498** (10 collections, 490 tracks): essentially half of the
learned quality score's total variance is explained purely by which
collection a track came from, before accounting for any track-specific
signal at all. This is a single, clean, quantitative confirmation of what
E2 (leave-one-collection-out instability) and E6 (collapsed
degradation-calibrated thresholds) both found qualitatively from different
angles: collection identity is not a minor nuisance in this dataset, it is
responsible for roughly as much of the score's variation as everything else
combined.

Requires the optional `statsmodels` dependency:
`pip install -e ".[mixedlm]"` (already present in this environment).

## Still to do

- The reprocessing/renormalisation half, if ever revisited: would need a new
  SLURM GPU array job (renormalise a track subset, re-call peaks with
  LanceOtron, re-score) of similar scope to E1's.
- A true `processing` covariate (e.g. "downloaded as-is" vs.
  "CATCH-UP-reprocessed") is not currently recorded per-track in this
  repo's dataset; adding it would let the model separate collection
  identity from processing route specifically, rather than conflating them
  under the single `collection_name` grouping factor used here.
