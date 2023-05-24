#!/bin/bash

#SBATCH --job-name=dependency
#SBATCH --nodes=1
#SBATCH --time=0-12:00:00
#SBATCH --mem=64000MB
#SBATCH --cpus-per-task=8
#SBATCH --partition=P1
#SBATCH --output='/home/hyeryungson/mucoco/slurm_output/%j.out'

iterations=3 # how many times you want to iterate over
jobid=310118 #$(sbatch --parsable sbatch_train_model.sh False) #Adjustement required: write the file name you want to run
# jobid=308396
for((i=0; i<$iterations; i++)); do           
    dependency="afterany:${jobid}"
    echo "dependency: $dependency"
    jobid=$(sbatch --parsable --dependency=$dependency sbatch_train_model_addmore.sh True) #Adjustement required: write the file name you want to run
done

