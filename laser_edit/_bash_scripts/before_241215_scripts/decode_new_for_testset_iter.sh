#!/bin/bash
#SBATCH --time=0-12:00:00
#SBATCH --mem=1GB
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=P1
#SBATCH --gres=gpu:0
#SBATCH --output='laser_edit/_slurm_outs/gbi_meta_%j.out'

# sbatch laser_edit/decode_new_for_testset_iter_sentiment_em.sh
# sbatch laser_edit/decode_new_for_testset_iter_sentiment_clsf.sh
# sbatch laser_edit/decode_new_for_testset_iter_toxicity_em.sh
# sbatch laser_edit/decode_new_for_testset_iter_toxicity_clsf.sh
# sbatch laser_edit/decode_new_for_testset_iter_formality_em.sh
# sbatch laser_edit/decode_new_for_testset_iter_formality_clsf.sh


jobid=$(sbatch --parsable --dependency=afterany:383477 laser_edit/decode_new_for_testset_iter_toxicity_em.sh)
dependency="afterany:${jobid}"
jobid=$(sbatch --parsable --dependency=$dependency laser_edit/decode_new_for_testset_iter_toxicity_em.sh)

jobid=$(sbatch --parsable --dependency=afterany:383478 laser_edit/decode_new_for_testset_iter_toxicity_clsf.sh)
dependency="afterany:${jobid}"
jobid=$(sbatch --parsable --dependency=$dependency laser_edit/decode_new_for_testset_iter_toxicity_clsf.sh)


# jobid=$(sbatch --parsable --dependency=afterany:374583 laser_edit/decode_new_for_testset_iter_formality_em.sh)
# dependency="afterany:${jobid}"
# jobid=$(sbatch --parsable --dependency=$dependency laser_edit/decode_new_for_testset_iter_formality_em.sh)
# dependency="afterany:${jobid}"
# jobid=$(sbatch --parsable --dependency=$dependency laser_edit/decode_new_for_testset_iter_formality_em.sh)
# dependency="afterany:${jobid}"
# jobid=$(sbatch --parsable --dependency=$dependency laser_edit/decode_new_for_testset_iter_formality_em.sh)

# jobid=$(sbatch --parsable --dependency=afterany:374584 laser_edit/decode_new_for_testset_iter_formality_clsf.sh)
# dependency="afterany:${jobid}"
# jobid=$(sbatch --parsable --dependency=$dependency laser_edit/decode_new_for_testset_iter_formality_clsf.sh)
# dependency="afterany:${jobid}"
# jobid=$(sbatch --parsable --dependency=$dependency laser_edit/decode_new_for_testset_iter_formality_clsf.sh)
# dependency="afterany:${jobid}"
# jobid=$(sbatch --parsable --dependency=$dependency laser_edit/decode_new_for_testset_iter_formality_clsf.sh)
