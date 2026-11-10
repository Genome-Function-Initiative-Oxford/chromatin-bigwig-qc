#!/bin/bash
#SBATCH --partition=gpu-ada
#SBATCH --gpus=1
#SBATCH --job-name=e8_dnase
#SBATCH --mem=24G
#SBATCH --mail-user=<simone.riva@imm.ox.ac.uk>
#SBATCH --time=00:20:00
#SBATCH --output=%A_%a_%x.out
#SBATCH --error=%A_%a_%x.err
#SBATCH --array=1-30%4
# --array matches manifest.tsv's 30 rows (10 ENCODE DNase-seq experiments per
# audit tier: clean/warning/error - see build_manifest.py). %4 throttle caps
# concurrently RUNNING tasks at 4, same as E1's GPU job, to avoid hogging
# gpu-ada.
# --gpus=1 (not --gres) per FireCrest/HYPER's GPU jobs, same as E1.
#
# Reuses E1's confirmed XLA/CUDA environment fixes (bigwig_qc.lanceotron.
# call_peaks already bakes these in - see run_e1_degradation.sh's header
# comment for the full ptxas/libdevice debugging history): no new fixes
# needed here, this job just calls the same call_peaks() function on
# different (real, not synthetically degraded) bigWigs.
#
# --time=00:20:00 is a placeholder: unlike E1, there is no expensive
# degradation step here (just LanceOtron peak calling + feature computation +
# scoring), so this should be faster than E1's calibrated ~5min/task, but the
# exact time on real ENCODE DNase-seq bigWigs (300-500MB each, downloaded via
# download_bigwigs.py) is unmeasured - confirm with a --array=1-1 smoke test
# first, then tighten.

set -euo pipefail

REPO_ROOT="$SLURM_SUBMIT_DIR"  # sbatch must be invoked from the repo root - ${BASH_SOURCE[0]} resolves wrong since SLURM copies this script to a spool dir before running it
EXPERIMENT_DIR="$REPO_ROOT/experiments/e8_transfer_assays_species"

module load python-cbrg
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"

MANIFEST="$EXPERIMENT_DIR/manifest.tsv"                          # from build_manifest.py
MODEL="$REPO_ROOT/results/model_excl_pancreatic_pbmc.json"       # same model as E1/E2/E3 - trained on human hg38 data;
                                                                  # DNase-seq is still human/hg38, so this transfers unchanged
BIGWIG_DIR="$REPO_ROOT/data/raw/encode_dnase"                    # from download_bigwigs.py
RESULTS_DIR="$REPO_ROOT/results/e8_jobs"
PEAKS_DIR="$REPO_ROOT/results/e8_peaks"
mkdir -p "$RESULTS_DIR" "$PEAKS_DIR"

# SLURM_ARRAY_TASK_ID is 1-indexed; manifest.tsv line 1 is the header, so task
# N reads data row N (i.e. file line N+1).
row=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$MANIFEST")
experiment_accession=$(echo "$row" | cut -f1)
audit_tier=$(echo "$row" | cut -f4)

python "$EXPERIMENT_DIR/run_one_dnase.py" \
    --bigwig "$BIGWIG_DIR/${experiment_accession}.bigWig" \
    --experiment-accession "$experiment_accession" \
    --audit-tier "$audit_tier" \
    --model "$MODEL" \
    --peaks-dir "$PEAKS_DIR/${SLURM_ARRAY_TASK_ID}" \
    --output "$RESULTS_DIR/${SLURM_ARRAY_TASK_ID}.tsv"
