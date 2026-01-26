#!/bin/bash
#SBATCH --job-name=bigwig_qual_targets
#SBATCH --mem=40G
#SBATCH --mail-user=<user@imm.ox.ac.uk>
#SBATCH --time=00-01:30:00
#SBATCH --output=%j_%x.out
#SBATCH --error=%j_%x.err
#SBATCH --ntasks=11

module load python-cbrg 

MAX_JOBS=11
PEAKS_FOLDERS="/project/Wellcome_Discovery/datashare/CREST-GV_collection/peaks_unfiltered"
COLLECTION_NAMES=('H1_hESCs' 'catlas_fetal' 'CAD' 'immune_cell' 'MPAL' 'catlas_adult' 'ludwig2019' 'pancreatic_pbmc' 'super_PBMC' 'calderon' 'Days7_10_13_17')

for collection_name in "${COLLECTION_NAMES[@]}"; do
    peak_calls_location="$PEAKS_FOLDERS/$collection_name"
    out="03_targets/$collection_name.tsv"
    
    python script/process_targets.py --input $peak_calls_location --output $out &

done
wait
