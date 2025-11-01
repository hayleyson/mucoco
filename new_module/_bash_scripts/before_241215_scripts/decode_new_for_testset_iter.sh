#!/bin/bash
#SBATCH --time=0-12:00:00
#SBATCH --mem=1GB
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=P1
#SBATCH --gres=gpu:0
#SBATCH --output='new_module/_slurm_outs/gbi_meta_%j.out'

# sbatch new_module/decode_new_for_testset_iter_sentiment_em.sh
# sbatch new_module/decode_new_for_testset_iter_sentiment_clsf.sh
# sbatch new_module/decode_new_for_testset_iter_toxicity_em.sh
# sbatch new_module/decode_new_for_testset_iter_toxicity_clsf.sh
# sbatch new_module/decode_new_for_testset_iter_formality_em.sh
# sbatch new_module/decode_new_for_testset_iter_formality_clsf.sh


jobid=$(sbatch --parsable --dependency=afterany:383477 new_module/decode_new_for_testset_iter_toxicity_em.sh)
dependency="afterany:${jobid}"
jobid=$(sbatch --parsable --dependency=$dependency new_module/decode_new_for_testset_iter_toxicity_em.sh)

jobid=$(sbatch --parsable --dependency=afterany:383478 new_module/decode_new_for_testset_iter_toxicity_clsf.sh)
dependency="afterany:${jobid}"
jobid=$(sbatch --parsable --dependency=$dependency new_module/decode_new_for_testset_iter_toxicity_clsf.sh)


# jobid=$(sbatch --parsable --dependency=afterany:374583 new_module/decode_new_for_testset_iter_formality_em.sh)
# dependency="afterany:${jobid}"
# jobid=$(sbatch --parsable --dependency=$dependency new_module/decode_new_for_testset_iter_formality_em.sh)
# dependency="afterany:${jobid}"
# jobid=$(sbatch --parsable --dependency=$dependency new_module/decode_new_for_testset_iter_formality_em.sh)
# dependency="afterany:${jobid}"
# jobid=$(sbatch --parsable --dependency=$dependency new_module/decode_new_for_testset_iter_formality_em.sh)

# jobid=$(sbatch --parsable --dependency=afterany:374584 new_module/decode_new_for_testset_iter_formality_clsf.sh)
# dependency="afterany:${jobid}"
# jobid=$(sbatch --parsable --dependency=$dependency new_module/decode_new_for_testset_iter_formality_clsf.sh)
# dependency="afterany:${jobid}"
# jobid=$(sbatch --parsable --dependency=$dependency new_module/decode_new_for_testset_iter_formality_clsf.sh)
# dependency="afterany:${jobid}"
# jobid=$(sbatch --parsable --dependency=$dependency new_module/decode_new_for_testset_iter_formality_clsf.sh)
