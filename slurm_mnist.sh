#!/bin/bash
#SBATCH -p gpus24
#SBATCH --gres gpu:1
#SBATCH --job-name=mnist_sem_ch_64_p_0_dim_4
##SBATCH --nodelist=mira09
#SBATCH --output=/vol/biomedic3/rrr2417/cxr-generation/src/models/slurm/mnist/logs/slurm.%N.%j.log

source /vol/biomedic3/rrr2417/.bashrc

poetry run python3 -m models.diffae.scripts.train \
    -s 28 \
    -e /vol/biomedic3/rrr2417/cxr-generation/src/models/diffae/output/mnist_sem_ch_64_p_0_dim_4
