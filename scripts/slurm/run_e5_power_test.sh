#!/bin/bash
#SBATCH --partition=long
#SBATCH --job-name=e5_power_test
#SBATCH --mem=20G
#SBATCH --mail-user=<simone.riva@imm.ox.ac.uk>
#SBATCH --time=05:00:00
#SBATCH --output=%A_%a_%x.out
#SBATCH --error=%A_%a_%x.err
#SBATCH --array=1-30%6

# Full-scale run of E5(A)'s open-chromatin power/FPR test across E1's 900-row
# manifest.tsv (experiments/e5_downstream_utility/run_open_chromatin_power_test.py).
# Pure CPU/bigWig work - no GPU/LanceOtron involved (unlike E1), so
# --partition=long (CPU), matching run_e4_tss_enrichment.sh's precedent, not
# gpu-ada.
#
# Batched, not one-task-per-row like E1: run_one() here is cheap relative to
# E1's (degrade + a few thousand local-background Poisson tests, no
# LanceOtron/peak-calling/model-scoring), so CHUNK_SIZE=30 rows/task (900/30 =
# 30 array tasks). Degradation dominates per-row cost and varies noticeably by
# track - CAD tracks ran ~250-330s/row, but task 18 (array index 18,
# catlas_fetal.Adrenal_Cortical) ran 345-360s/row and was CANCELLED DUE TO
# TIME LIMIT at --time=03:00:00 after 27/30 rows (confirmed via its .out log,
# not a code bug - just a larger/denser bigWig taking longer to degrade).
# --time=05:00:00 (bumped from the original 03:00:00) gives ~40% headroom
# over the slowest track seen so far: 30 x 360s = 3.0h + loci-build/module
# overhead, vs the 2.5h estimate the original budget was based on. %6
# throttle (looser than E1's GPU %4 - CPU capacity on `long` is less scarce
# than GPU-ada's) - raise/lower depending on how busy the partition is.
#
# --n-true-positive-loci/--n-true-negative-loci passed explicitly as 2000:
# the experiment README documents 2000 as the intended default, but
# run_open_chromatin_power_test.py's actual CLI default is 500 (a
# pre-existing doc/code mismatch, left as-is rather than changed, in case
# anything else relies on the 500 default) - pass 2000 explicitly here to
# match the documented intent. Testing more loci is cheap relative to the
# per-track degradation cost and gives tighter power/FPR estimates.
#
# All CHUNK_SIZE array tasks independently resample the same loci (fixed
# --random-state=0, the CLI default) - the fixed point set is intentional
# (power/FPR must be computed against the *same* true-positive/true-negative
# loci across every degradation setting for the numbers to be comparable),
# not a bug.

set -euo pipefail

REPO_ROOT="$SLURM_SUBMIT_DIR"  # sbatch must be invoked from the repo root - ${BASH_SOURCE[0]} resolves wrong since SLURM copies this script to a spool dir before running it
EXPERIMENT_DIR="$REPO_ROOT/experiments/e5_downstream_utility"

module load python-cbrg
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"

MANIFEST="$REPO_ROOT/experiments/e1_degradation_benchmark/manifest.tsv"
RESULTS_DIR="$REPO_ROOT/results/e5_jobs"
mkdir -p "$RESULTS_DIR"

CHUNK_SIZE=30
row_start=$(((SLURM_ARRAY_TASK_ID - 1) * CHUNK_SIZE))
row_end=$((row_start + CHUNK_SIZE))

python "$EXPERIMENT_DIR/run_open_chromatin_power_test.py" \
    --manifest "$MANIFEST" \
    --n-true-positive-loci 2000 \
    --n-true-negative-loci 2000 \
    --row-start "$row_start" \
    --row-end "$row_end" \
    --output "$RESULTS_DIR/${SLURM_ARRAY_TASK_ID}.tsv"
