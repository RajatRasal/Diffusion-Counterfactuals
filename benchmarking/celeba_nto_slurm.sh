#!/bin/bash
#SBATCH -p gpus48
#SBATCH --gres gpu:1
#SBATCH --job-name=celeba-nto-metrics
##SBATCH --nodelist=mira03
#SBATCH --output=src/counterfactuals/benchmarking/slurm_logs/slurm.%N.%j.log

source /vol/biomedic3/rrr2417/.bashrc
cd /vol/biomedic3/rrr2417/cxr-generation/

# poetry run python3 -m counterfactuals.benchmarking.celeba_nto \
#     --guidance-scale 1.5 \
#     --cta-steps 1 \
#     --learning-rate 1e-5 \
#     --last-batch 100 \
#     --seed 10000

# --learning-rate 0.000001 \

poetry run python3 -m counterfactuals.benchmarking.celeba_nto \
    --guidance-scale 2 \
    --semantic \
    --cta-steps 1 \
    --learning-rate 0.001 \
    --last-batch 100 \
    --seed 10000