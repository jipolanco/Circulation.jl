#!/bin/bash
#SBATCH --job-name=circulation
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40

#SBATCH --dependency=singleton

#SBATCH --hint=nomultithread  # on réserve des coeurs physiques et non logiques

#SBATCH --time=02:00:00

#SBATCH --output=logs/circulation_%j.out
#SBATCH --error=logs/circulation_%j.err

echo "Starting on $(date)"

module purge
module load gcc

# echo des commandes lancées
set -x

# Copy data to scratch (should be faster...)
field_dir="/gpfswork/rech/rhl/uxs76mg/GP/NLPot/1024/OUTPUTS/fields"
rsync -avhu "$field_dir"/*Psi.100.dat $JOBSCRATCH/fields/

export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK 

julia --project --check-bounds=no -O3 circulation.jl tangle.toml

echo "Ending on $(date)"
