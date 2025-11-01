#!/bin/bash
iterations=4
jobid=$(sbatch --parsable new_module/_bash_scripts/mlm_reranking_formality_original_em_no_embed_share.sh)
for((i=0; i<$iterations; i++)); do           
    dependency="afterany:${jobid}"
    echo "dependency: $dependency"
    jobid=$(sbatch --parsable --dependency=$dependency new_module/_bash_scripts/mlm_reranking_formality_original_em_no_embed_share.sh) #Adjustement required: write the file name you want to run
done

iterations=4
jobid=$(sbatch --parsable new_module/_bash_scripts/mlm_reranking_formality_v1_em_no_embed_share.sh)
for((i=0; i<$iterations; i++)); do           
    dependency="afterany:${jobid}"
    echo "dependency: $dependency"
    jobid=$(sbatch --parsable --dependency=$dependency new_module/_bash_scripts/mlm_reranking_formality_v1_em_no_embed_share.sh) #Adjustement required: write the file name you want to run
done

sbatch new_module/_bash_scripts/mlm_reranking_formality_original_em_no_embed_share.sh

sbatch new_module/_bash_scripts/mlm_reranking_formality_v1_em_no_embed_share.sh

sbatch new_module/_bash_scripts/mlm_reranking_sentiment_original_em_no_embed_share.sh

sbatch new_module/_bash_scripts/mlm_reranking_sentiment_v0_em_no_embed_share.sh

sbatch new_module/_bash_scripts/mlm_reranking_sentiment_v0_em.sh

sbatch new_module/_bash_scripts/mlm_reranking_sentiment_v1_em_no_embed_share.sh

sbatch new_module/_bash_scripts/mlm_reranking_toxicity_original_em_no_embed_share.sh

sbatch new_module/_bash_scripts/mlm_reranking_toxicity_v0_em_no_embed_share.sh

sbatch new_module/_bash_scripts/mlm_reranking_toxicity_v0_em.sh

sbatch new_module/_bash_scripts/mlm_reranking_toxicity_v1_em_no_embed_share.sh
