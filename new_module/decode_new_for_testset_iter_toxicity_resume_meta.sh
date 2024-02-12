#!/bin/bash
#SBATCH --time=0-12:00:00
#SBATCH --mem=1GB
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=P2
#SBATCH --gres=gpu:0
#SBATCH --output='new_module/_slurm_outs/t_gbi_meta_%j.out'

# sbatch --dependency=371781 new_module/decode_new_for_testset_iter_toxicity_em_gn.sh
# sbatch --dependency=371782 new_module/decode_new_for_testset_iter_toxicity_em.sh
# sbatch --dependency=371783 new_module/decode_new_for_testset_iter_toxicity_clsf_gn.sh
# sbatch --dependency=371784 new_module/decode_new_for_testset_iter_toxicity_clsf.sh


jobid=$(sbatch --parsable --dependency=afterany:371781 new_module/decode_new_for_testset_iter_toxicity_em_gn.sh)
dependency="afterany:${jobid}"
jobid=$(sbatch --parsable --dependency=$dependency new_module/decode_new_for_testset_iter_toxicity_em_gn.sh)
dependency="afterany:${jobid}"
jobid=$(sbatch --parsable --dependency=$dependency new_module/decode_new_for_testset_iter_toxicity_em_gn.sh)
dependency="afterany:${jobid}"
jobid=$(sbatch --parsable --dependency=$dependency new_module/decode_new_for_testset_iter_toxicity_em_gn.sh)

jobid=$(sbatch --parsable --dependency=afterany:371782 new_module/decode_new_for_testset_iter_toxicity_em.sh)
dependency="afterany:${jobid}"
jobid=$(sbatch --parsable --dependency=$dependency new_module/decode_new_for_testset_iter_toxicity_em.sh)
dependency="afterany:${jobid}"
jobid=$(sbatch --parsable --dependency=$dependency new_module/decode_new_for_testset_iter_toxicity_em.sh)
dependency="afterany:${jobid}"
jobid=$(sbatch --parsable --dependency=$dependency new_module/decode_new_for_testset_iter_toxicity_em.sh)


jobid=$(sbatch --parsable --dependency=afterany:371783 new_module/decode_new_for_testset_iter_toxicity_clsf_gn.sh)
dependency="afterany:${jobid}"
jobid=$(sbatch --parsable --dependency=$dependency new_module/decode_new_for_testset_iter_toxicity_clsf_gn.sh)
dependency="afterany:${jobid}"
jobid=$(sbatch --parsable --dependency=$dependency new_module/decode_new_for_testset_iter_toxicity_clsf_gn.sh)
dependency="afterany:${jobid}"
jobid=$(sbatch --parsable --dependency=$dependency new_module/decode_new_for_testset_iter_toxicity_clsf_gn.sh)


jobid=$(sbatch --parsable --dependency=afterany:371784 new_module/decode_new_for_testset_iter_toxicity_clsf.sh)
dependency="afterany:${jobid}"
jobid=$(sbatch --parsable --dependency=$dependency new_module/decode_new_for_testset_iter_toxicity_clsf.sh)
dependency="afterany:${jobid}"
jobid=$(sbatch --parsable --dependency=$dependency new_module/decode_new_for_testset_iter_toxicity_clsf.sh)
dependency="afterany:${jobid}"
jobid=$(sbatch --parsable --dependency=$dependency new_module/decode_new_for_testset_iter_toxicity_clsf.sh)
