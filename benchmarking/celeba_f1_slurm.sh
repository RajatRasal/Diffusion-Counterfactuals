#!/bin/bash
#SBATCH -p gpus24
#SBATCH --gres gpu:1
#SBATCH --job-name=f1-celeba-metrics
##SBATCH --nodelist=mira09
#SBATCH --output=src/counterfactuals/benchmarking/slurm_logs/slurm.%N.%j.log

source /vol/biomedic3/rrr2417/.bashrc
cd /vol/biomedic3/rrr2417/cxr-generation/


poetry run python3 -m counterfactuals.benchmarking.celeba_f1 \
    --batch-size 1000 \
    --last-batch 0 \
    --seed 10000 \
    --semantic \
    --cfg \
    --guidance-scale 2