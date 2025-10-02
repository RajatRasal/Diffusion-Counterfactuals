#!/bin/bash
#SBATCH -p gpus24
#SBATCH --gres gpu:1
#SBATCH --job-name=celeba_p_0.1_no_sem
#SBATCH --nodelist=mira09
#SBATCH --output=/vol/biomedic3/rrr2417/cxr-generation/src/models/slurm/celebahq/logs/slurm.%N.%j.log

source /vol/biomedic3/rrr2417/.bashrc

export IMAGE_SIZE=64
export EXPN="/vol/biomedic3/rrr2417/cxr-generation/src/models/diffae/output/CELEBA_p_0.1_no_sem"

poetry run python3 -m models.diffae.scripts.train --size=$IMAGE_SIZE --expn=$EXPN
