# E1 - Controlled degradation benchmark

Status: **done.** Full run complete: 900/900 (track, mode, level, seed)
combinations across all 30 seed tracks, 11 collections. See "Results" below.

**Paper section:** Materials and Methods / Controlled degradation benchmark
(ground-truth validation); Results / "The quality score tracks controlled
degradation". New vs. CIBCB.

## Method

Raw reads aren't available for every track, so degradation operates directly
on the **bigWig signal** instead of resampling reads - consistent with the
rest of this pipeline (and with LanceOtron itself, designed to work from
bigWig signal alone, no BAM required).

1. **Select seed tracks** (`select_seed_tracks.py`): rank all 502 tracks by
   their already-measured quality (`n_promoters + n_ctcf`, the same quantity
   `bigwig_qc.model` is trained to predict), and take the top few per
   collection so the seed set spans all 11 collections (`bigwig_qc.seed_selection`).
2. **Expand to a job manifest** (`build_manifest.py`): every seed track x
   5 down-sampling fractions (50/25/10/5/1%) x 4 noise-injection fractions
   (10/30/50/80%) x 3 random seeds, plus a `down_sample, level=1.0` row per
   track as the "0% degradation" baseline (`bigwig_qc.manifest`).
3. **Degrade** (`bigwig_qc.degradation`, `bigwig_qc.bigwig_io`): the bigWig is
   rasterised onto a fixed 100 bp bin grid (`config.DEGRADATION_BIN_SIZE_BP`),
   then:
   - *down-sampling*: each bin's (rounded) signal is treated as a read count
     and thinned via `Binomial(count, fraction)` - the same thinning used to
     subsample a Poisson process, reproducing both the depth reduction and
     the increased relative noise of true read subsampling.
   - *noise injection*: a fraction of total signal mass is removed from its
     original bins and scattered back as Poisson-distributed counts spread
     uniformly across every bin, including previously-empty ones.
4. **Re-call peaks** (`bigwig_qc.lanceotron.call_peaks`): LanceOtron is run on
   the degraded bigWig (`lanceotron callPeaks ... --format Web -c 0`), in an
   isolated subprocess/module environment (LanceOtron needs its own
   Python/TensorFlow stack, different from the rest of this package - see
   "Running it").
5. **Re-score** (`bigwig_qc.features` + `bigwig_qc.model.load_model`): the
   same QC covariates used everywhere else in this pipeline are recomputed
   from the degraded bigWig + its new peak calls, and scored with the model
   trained by `scripts/train_model.py`.
6. **Aggregate and plot** (`aggregate_results.py`, `plot_dose_response.py`):
   one row per (track, mode, level, seed) job, score vs. degradation level -
   the dose-response figure.

Each (track, mode, level, seed) combination is processed end-to-end by
`run_one_degradation.py`, run as one SLURM array task
(`scripts/slurm/run_e1_degradation.sh`, `--array=1-900%4` on `gpu-ada`).

## Results

`results/e1_dose_response.{pdf,png}` (built by `plot_dose_response.py`):

![E1 dose-response](../../results/e1_dose_response.png)

Mean score by degradation mode/level, across all 30 tracks:

| degradation_mode | level | score (mean) | score (std) |
|---|---|---|---|
| down_sample | 0.01 (heaviest) | 4733 | 1500 |
| down_sample | 0.05 | 5518 | 1205 |
| down_sample | 0.10 | 5635 | 1289 |
| down_sample | 0.25 | 5653 | 1324 |
| down_sample | 0.50 | 5568 | 1311 |
| down_sample | 1.00 (undegraded baseline) | 5510 | 1379 |
| inject_noise | 0.10 (lightest) | 7264 | 500 |
| inject_noise | 0.30 | 6876 | 525 |
| inject_noise | 0.50 | 6346 | 577 |
| inject_noise | 0.80 (heaviest) | 4919 | 480 |

**`inject_noise`: a clean, strong, monotonic dose-response** - score falls
smoothly and substantially (7264 -> 4919) as injected-noise severity
increases across all four tested levels. The score tracks this degradation
mode very well.

**`down_sample`: largely flat until the most extreme level, then a cliff** -
scores across levels 0.05-1.0 all sit within ~140 points of each other
(5510-5653), then drop sharply (-785 points, ~14%) only at the heaviest level
(0.01, i.e. 99% of reads removed). The model is essentially insensitive to
moderate read-depth loss and only reacts once degradation becomes severe.

**Connects directly to E5's finding** (see
`experiments/e5_downstream_utility/README.md`): E5 found that `inject_noise`
- not `down_sample` - is the degradation mode that actually produces
unreliable (false-positive-inflating) downstream statistical conclusions,
while `down_sample` mostly costs detection power at its most extreme level.
The score's shape here lines up with that story at face value (strong,
smooth response to noise; weak, threshold-only response to down-sampling).
**However**, a direct join of this experiment's scores against E5's power/FPR
(`experiments/e5_downstream_utility/correlate_score_with_power_fpr.py`) shows
this alignment is more qualified than it first appears: within `inject_noise`
specifically, score does *not* significantly correlate with either power or
FPR (rho=0.08, rho=-0.10, both n.s.) - the score reacts to noise severity in
aggregate, but that reaction doesn't track which specific noisy tracks will
actually produce inflated false-positive rates. See E5's README for the full
breakdown; this is an important caveat on how strongly the score should be
claimed to predict downstream statistical reliability.

## Known limitations (state these in the Methods text)

- **Bin resolution**: degradation operates at 100 bp resolution, coarser than
  the input bigWig's native resolution (though on the same order as
  LanceOtron's own default 400 bp candidate-peak window). The `level=1.0`
  down-sample baseline controls for this - it's rebinned and re-peak-called
  exactly like every degraded point, so the dose-response curve measures the
  effect of degradation, not of rebinning.
- **Signal units**: thinning assumes bin values are approximately proportional
  to read counts. If the source bigWigs are strongly normalised (e.g. RPGC),
  this is an approximation of true read-depth subsampling rather than an
  exact reproduction of it.
- **LanceOtron version**: re-scored with whatever `lanceotron` module version
  is loaded at run time, which may differ from the version originally used to
  produce `peaks_unfiltered/` in the parent repo - this is why the pipeline
  always re-calls peaks on the `level=1.0` baseline too, rather than reusing
  the old peak calls, so every point in the dose-response curve is internally
  consistent.

## Running it

LanceOtron's module (`module load lanceotron`) pulls in its own Python,
cuda/cuDNN and tensorflow2-gpu - a separate environment from the
`python-cbrg` + `pip install -e .` environment the rest of this package uses.
`bigwig_qc.lanceotron.call_peaks` isolates this automatically (clean subshell,
`module purge` + `module load lanceotron`, `PYTHONPATH` unset) so
`run_one_degradation.py` only needs `python-cbrg` loaded itself.

**LanceOtron needs a GPU** - `scripts/slurm/run_e1_degradation.sh` runs on
`gpu-ada`. Getting this running end-to-end surfaced four real bugs (ptxas/XLA
crash, a masked libdevice error, a binning performance+correctness bug, and a
LanceOtron output-filename truncation bug on this experiment's multi-dot
degraded filenames) - all fixed, see the script's header comment for the full
history. Do not attempt to smoke-test `call_peaks` on a CPU-only node.

```bash
module load python-cbrg
pip install -e ".[dev]"   # from the repo root

# 1. Pick seed tracks
python experiments/e1_degradation_benchmark/select_seed_tracks.py \
    --summary-dir data/interim/features --targets-dir data/interim/targets \
    --output experiments/e1_degradation_benchmark/seed_tracks.tsv

# 2. Expand to the full job manifest
python experiments/e1_degradation_benchmark/build_manifest.py \
    --seed-tracks experiments/e1_degradation_benchmark/seed_tracks.tsv \
    --output experiments/e1_degradation_benchmark/manifest.tsv

# 3. Submit the array job (--array=1-900%4 is already set in the script,
#    matching the manifest's 900 rows; update it if the seed-track list
#    changes). sbatch must be invoked from the repo root.
sbatch scripts/slurm/run_e1_degradation.sh

# 4. Once all array tasks finish, aggregate and plot
python experiments/e1_degradation_benchmark/aggregate_results.py \
    --results-dir results/e1_jobs --output results/e1_degradation_results.tsv
python experiments/e1_degradation_benchmark/plot_dose_response.py \
    --results results/e1_degradation_results.tsv \
    --output-stub results/e1_dose_response
```

Confirmed end-to-end: 900/900 array tasks succeeded, no gaps against
`manifest.tsv`, ~5min/task (throttled at 4 concurrent GPU tasks on `gpu-ada`
via the `%4` in `--array`), ~15h total wall-clock.

`results/e1_jobs/`, `results/e1_degradation_results.tsv`, and
`results/e1_dose_response.{pdf,png}` are gitignored (regenerated pipeline
output, not source).
