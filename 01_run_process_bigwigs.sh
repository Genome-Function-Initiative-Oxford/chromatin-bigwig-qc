#!/bin/bash
#SBATCH --job-name=bigwig_inputs
#SBATCH --mem=20G
#SBATCH --mail-user=<user@imm.ox.ac.uk>
#SBATCH --time=00-0:30:00
#SBATCH --output=%j_%x.out
#SBATCH --error=%j_%x.err
#SBATCH --ntasks=60

module load python-cbrg

MAX_JOBS=60

module load python-cbrg

BIGWIG_FOLDERS="<path>"
COLLECTION_NAMES=('H1_hESCs' 'catlas_fetal' 'CAD' 'immune_cell' 'MPAL' 'catlas_adult' 'ludwig2019' 'pancreatic_pbmc' 'super_PBMC' 'calderon' 'Days7_10_13_17')

for collection_name in "${COLLECTION_NAMES[@]}"; do
    bigwigs_location="$BIGWIG_FOLDERS/$collection_name/bigwigs"
    peak_calls_location="$BIGWIG_FOLDERS/peaks_unfiltered/$collection_name"
    
    for bigwig in "$bigwigs_location"/*.bw; do
        stub=$(basename "$bigwig")
        stub="${stub%.*}"

        peak_call_filepath="$peak_calls_location/${stub}_L-tron.bed"
        if [ -f $peak_call_filepath ]; then
            # To not clog up the jobs queue
            while [ $(jobs | wc -l) -ge $MAX_JOBS ]; do
                sleep 1 # Sleep for a short time before checking again
            done
            
            out="02_summary_JSONs/$collection_name.$stub.json"
            python script/process_bigwig.py --input-bigwig $bigwig --input-peak-calls $peak_call_filepath --output $out &
        else
            echo "Expected peak call file is missing: $peak_call_filepath"
        fi
    done
done

wait
