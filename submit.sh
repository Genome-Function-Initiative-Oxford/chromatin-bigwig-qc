#!/bin/bash
#SBATCH --partition=long
#SBATCH --job-name=QC_bw
#SBATCH --ntasks=10
#SBATCH --mem=96G
#SBATCH --mail-user=<simone.riva@imm.ox.ac.uk>
#SBATCH --time=03-23:59:59
#SBATCH --output=%j_%x.out
#SBATCH --error=%j_%x.err

module load python-cbrg

python script/crestgv_collection.py