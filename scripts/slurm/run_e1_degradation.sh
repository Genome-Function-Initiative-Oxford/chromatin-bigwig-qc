#!/bin/bash
#SBATCH --partition=gpu-ada
#SBATCH --gpus=1
#SBATCH --job-name=e1_degradation
#SBATCH --mem=24G
#SBATCH --mail-user=<simone.riva@imm.ox.ac.uk>
#SBATCH --time=00:30:00
#SBATCH --output=%A_%a_%x.out
#SBATCH --error=%A_%a_%x.err
#SBATCH --array=1-900%4
# --array matches manifest.tsv's current row count (30 seed tracks x 30
# mode/level/seed combinations = 900; regenerate via build_manifest.py and
# update this if the seed-track list changes). The %4 throttle caps
# concurrently RUNNING tasks at 4 (all 900 still queue) - keeps this job from
# hogging every GPU on gpu-ada at once; raise/drop it depending on how busy
# the partition is.
# --gpus=1 (not --gres) per FireCrest/HYPER's GPU jobs.
#
# FOUR bugs found during debugging (reproduced on gpu-turing/TITAN RTX AND
# gpu-ada/L40S - cluster/environment-wide, not partition-specific):
#   1. FIXED AND CONFIRMED: TensorFlow's XLA JIT crashed with SIGABRT on this
#      environment's bundled ptxas (12.3.103): "ptxas 12.3.103 has a bug that
#      we think can affect XLA." TF_XLA_FLAGS=--tf_xla_auto_jit=0 alone did
#      NOT fix it (confirmed: identical crash - the bug is a hardcoded
#      ptxas-version check inside XLA itself). XLA_FLAGS=
#      --xla_gpu_autotune_level=0 (in bigwig_qc.lanceotron.call_peaks) DID fix
#      it - confirmed, crash gone entirely.
#   2. FIXED AND CONFIRMED (smoke test job 6558835): with (1) fixed, hit a
#      second, previously masked error: "libdevice not found at
#      ./libdevice.10.bc" - this cluster's CUDA 12.3 (what tensorflow2-gpu/
#      2.16.1 + ptxas 12.3.103 target) has no standalone install and no pip
#      nvidia-cuda-nvcc wheel. Pointed --xla_gpu_cuda_data_dir at
#      /package/cuda/11.8 instead (newest standalone CUDA actually present,
#      confirmed to contain nvvm/libdevice/libdevice.10.bc) - confirmed by
#      job 6558835 running TensorFlow inference cleanly through every window,
#      no ptxas/libdevice error in the .err log at all.
#   3. FIXED AND CONFIRMED: bigwig_qc.bigwig_io.read_binned_mean_signal used
#      bw.stats(..., nBins=...) (~2.5M bins for chr1 alone) - exact=True
#      forces per-base rescanning (6+ min for chr1 alone, killed before
#      finishing); exact=False is fast but uses the bigWig's zoom-level
#      summaries, which don't align with our exact bin grid and measurably
#      smooth across bin boundaries (correlation 0.96, not 1.0, vs the true
#      per-bin mean). Replaced with bw.values() + vectorized numpy binning:
#      exact (not an approximation) AND faster. Confirmed end-to-end on a real
#      bigWig, all chromosomes: 265.5s, down from 2881.5s (48min) - ~11x.
#   4. FIXED AND CONFIRMED (same smoke test, surfaced once (1)+(2) stopped
#      masking it): LanceOtron's own output-filename logic
#      (lanceotron/modules.py's find_and_score_peaks) truncates the input
#      bigWig's basename at the FIRST "." - fine for this pipeline's normal
#      single-dot bigwig names, but this benchmark's degraded filenames
#      ("<track>.<mode>.<level>.<seed>.bw") have multiple dots, so LanceOtron
#      wrote e.g. "Mast_L-tron.bed" instead of the full-stem name our glob
#      expected, and bigwig_qc.lanceotron.call_peaks raised FileNotFoundError
#      even though LanceOtron had succeeded. Fixed by predicting LanceOtron's
#      own naming convention directly (split on first ".", not Path.stem)
#      instead of using the full stem.
# --time=00:30:00: smoke test job 6558950 (--array=1-1), run after all four
# bugs above were fixed, completed the full pipeline (degrade -> call peaks
# -> compute features -> score) in ~5min15s end-to-end (CBB profiling start
# 16:10:15 -> result TSV written 16:15:29 on 2026-11-02) - consistent with the
# previously-confirmed ~4.5min/track degradation step plus a modest
# LanceOtron/feature/scoring overhead. 30min gives a healthy ~6x safety
# margin for per-track variance (e.g. larger bigwigs, busier GPU nodes).

# One array task = one row of manifest.tsv = one (track, mode, level, seed)
# combination (see build_manifest.py). run_one_degradation.py itself isolates
# the LanceOtron subprocess call into its own module environment, so only
# python-cbrg needs to be loaded in this script.

set -euo pipefail

REPO_ROOT="$SLURM_SUBMIT_DIR"  # sbatch must be invoked from the repo root - ${BASH_SOURCE[0]} resolves wrong since SLURM copies this script to a spool dir before running it
EXPERIMENT_DIR="$REPO_ROOT/experiments/e1_degradation_benchmark"

module load python-cbrg
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"

MANIFEST="$EXPERIMENT_DIR/manifest.tsv"                        # from build_manifest.py
MODEL="$REPO_ROOT/results/model_excl_pancreatic_pbmc.json"      # from scripts/train_model.py - NOT results/model.json,
                                                                 # which still includes pancreatic_pbmc (see E2's README
                                                                 # for why that collection was excluded)
RESULTS_DIR="$REPO_ROOT/results/e1_jobs"
mkdir -p "$RESULTS_DIR"

# SLURM_ARRAY_TASK_ID is 1-indexed; manifest.tsv line 1 is the header, so task
# N reads data row N (i.e. file line N+1).
row=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$MANIFEST")
collection_name=$(echo "$row" | cut -f1)
experiment_name=$(echo "$row" | cut -f2)
mode=$(echo "$row" | cut -f3)
level=$(echo "$row" | cut -f4)
seed=$(echo "$row" | cut -f5)

python "$EXPERIMENT_DIR/run_one_degradation.py" \
    --collection-name "$collection_name" \
    --experiment-name "$experiment_name" \
    --mode "$mode" \
    --level "$level" \
    --seed "$seed" \
    --model "$MODEL" \
    --output "$RESULTS_DIR/${SLURM_ARRAY_TASK_ID}.tsv"
