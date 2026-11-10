#!/bin/bash
#SBATCH --partition=short
#SBATCH --job-name=QC_bw
#SBATCH --ntasks=1
#SBATCH --mem=48G
#SBATCH --mail-user=<simone.riva@imm.ox.ac.uk>
#SBATCH --time=00-23:59:59
#SBATCH --output=%j_%x.out
#SBATCH --error=%j_%x.err

module load python-cbrg

python crestgv_collection.py