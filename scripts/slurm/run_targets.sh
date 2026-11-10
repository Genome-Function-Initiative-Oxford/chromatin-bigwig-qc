#!/bin/bash
#SBATCH --job-name=bigwig_qual_targets
#SBATCH --mem=40G
#SBATCH --mail-user=<simone.riva@imm.ox.ac.uk>
#SBATCH --time=00-01:30:00
#SBATCH --output=%j_%x.out
#SBATCH --error=%j_%x.err
#SBATCH --ntasks=11

# Refactor of the parent repo's run_process_targets.sh, calling scripts/run_targets.py
# (bigwig_qc.references + bigwig_qc.targets) instead of process_targets.py directly.

set -euo pipefail

MAX_JOBS=11

module load python-cbrg

REPO_ROOT="$SLURM_SUBMIT_DIR"  # sbatch must be invoked from the repo root - ${BASH_SOURCE[0]} resolves wrong since SLURM copies this script to a spool dir before running it
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"

PEAKS_FOLDERS="/project/Wellcome_Discovery/datashare/CREST-GV_collection/peaks_unfiltered"
COLLECTION_NAMES=('H1_hESCs' 'catlas_fetal' 'CAD' 'immune_cell' 'MPAL' 'catlas_adult' 'ludwig2019' 'pancreatic_pbmc' 'super_PBMC' 'calderon' 'Days7_10_13_17')

OUTPUT_DIR="$REPO_ROOT/data/interim/targets"
mkdir -p "$OUTPUT_DIR"

for collection_name in "${COLLECTION_NAMES[@]}"; do
    peak_calls_location="$PEAKS_FOLDERS/$collection_name"
    out="$OUTPUT_DIR/$collection_name.tsv"

    while [ "$(jobs -r | wc -l)" -ge "$MAX_JOBS" ]; do
        sleep 1
    done

    python "$REPO_ROOT/scripts/run_targets.py" --input "$peak_calls_location" --output "$out" &
done
wait
