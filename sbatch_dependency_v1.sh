#!/bin/bash

#SBATCH --job-name=dependency
#SBATCH --nodes=1
#SBATCH --time=0-12:00:00
#SBATCH --mem=64000MB
#SBATCH --cpus-per-task=8
#SBATCH --partition=P2
#SBATCH --output='slurm_output/%j.out'

iterations=1 # how many times you want to iterate over
jobid=202532 #$(sbatch --parsable sbatch_train_model.sh False) #Adjustement required: write the file name you want to run
for((i=0; i<$iterations; i++)); do           
    dependency="afterany:${jobid}"
    echo "dependency: $dependency"
    jobid=$(sbatch --parsable --dependency=$dependency sbatch_beam_v1.sh True) #Adjustement required: write the file name you want to run
done

