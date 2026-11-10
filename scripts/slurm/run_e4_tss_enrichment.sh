#!/bin/bash
#SBATCH --partition=long
#SBATCH --job-name=e4_tss_enrichment
#SBATCH --mem=20G
#SBATCH --mail-user=<simone.riva@imm.ox.ac.uk>
#SBATCH --time=00-12:00:00
#SBATCH --output=%j_%x.out
#SBATCH --error=%j_%x.err
#SBATCH --ntasks=1

# Computes TSS enrichment (bigwig_qc.tss_enrichment) for every track in the
# dataset - pure CPU/bigWig I/O work, no GPU needed (unlike E1's LanceOtron
# re-calls). ~8h wall-clock at the default N_TSS_SAMPLE=2000 (490 tracks x
# 2000 TSS x 3 bw.stats calls each, ~10ms/call, measured on this cluster).
# --time padded to 12h above that estimate.
#
# For a faster, slightly noisier run, override the TSS sample size:
#   sbatch --export=N_TSS_SAMPLE=500 scripts/slurm/run_e4_tss_enrichment.sh
# (~500/2000 of the default runtime, ~2h)

set -euo pipefail

REPO_ROOT="$SLURM_SUBMIT_DIR"  # sbatch must be invoked from the repo root - ${BASH_SOURCE[0]} resolves wrong since SLURM copies this script to a spool dir before running it

module load python-cbrg
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"

N_TSS_SAMPLE="${N_TSS_SAMPLE:-2000}"

python "$REPO_ROOT/experiments/e4_qc_concordance/compute_tss_enrichment.py" \
    --dataset "$REPO_ROOT/data/interim/model_dataset.parquet" \
    --n-tss-sample "$N_TSS_SAMPLE" \
    --output "$REPO_ROOT/results/e4_tss_enrichment.tsv"
