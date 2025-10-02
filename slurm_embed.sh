#!/bin/bash
#SBATCH -p gpus48
#SBATCH --gres gpu:1
#SBATCH --job-name=EMBED_192_img_512_sem_p_0.1_final
#SBATCH --nodelist=loki
#SBATCH --output=/vol/biomedic3/rrr2417/cxr-generation/src/models/slurm/embed/logs/slurm.%N.%j.log

source /vol/biomedic3/rrr2417/.bashrc

poetry run python3 -m models.diffae.scripts.train \
    -s 192 \
    -e /vol/biomedic3/rrr2417/cxr-generation/src/models/diffae/output/EMBED_192_img_512_sem_p_0.1_final
