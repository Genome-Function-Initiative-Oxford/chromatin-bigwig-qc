#!/bin/bash
#SBATCH --job-name=bigwig_inputs
#SBATCH --mem=20G
#SBATCH --mail-user=<simone.riva@imm.ox.ac.uk>
#SBATCH --time=00-0:30:00
#SBATCH --output=%j_%x.out
#SBATCH --error=%j_%x.err
#SBATCH --ntasks=60

# Refactor of the parent repo's run_process_bigwigs.sh, calling scripts/run_features.py
# (bigwig_qc.features) instead of process_bigwig.py directly.

set -euo pipefail

MAX_JOBS=60

module load python-cbrg

REPO_ROOT="$SLURM_SUBMIT_DIR"  # sbatch must be invoked from the repo root - ${BASH_SOURCE[0]} resolves wrong since SLURM copies this script to a spool dir before running it
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"

BIGWIG_FOLDERS="/project/Wellcome_Discovery/datashare/CREST-GV_collection"
COLLECTION_NAMES=('H1_hESCs' 'catlas_fetal' 'CAD' 'immune_cell' 'MPAL' 'catlas_adult' 'ludwig2019' 'pancreatic_pbmc' 'super_PBMC' 'calderon' 'Days7_10_13_17')

OUTPUT_DIR="$REPO_ROOT/data/interim/features"
mkdir -p "$OUTPUT_DIR"

for collection_name in "${COLLECTION_NAMES[@]}"; do
    bigwigs_location="$BIGWIG_FOLDERS/$collection_name/bigwigs"
    peak_calls_location="$BIGWIG_FOLDERS/peaks_unfiltered/$collection_name"

    for bigwig in "$bigwigs_location"/*.bw; do
        stub=$(basename "$bigwig")
        stub="${stub%.*}"

        peak_call_filepath="$peak_calls_location/${stub}_L-tron.bed"
        if [ -f "$peak_call_filepath" ]; then
            while [ "$(jobs -r | wc -l)" -ge "$MAX_JOBS" ]; do
                sleep 1
            done

            out="$OUTPUT_DIR/$collection_name.$stub.json"
            python "$REPO_ROOT/scripts/run_features.py" \
                --input-bigwig "$bigwig" \
                --input-peak-calls "$peak_call_filepath" \
                --output "$out" &
        else
            echo "Expected peak call file is missing: $peak_call_filepath"
        fi
    done
done

wait
