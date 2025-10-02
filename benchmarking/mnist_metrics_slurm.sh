#!/bin/bash
#SBATCH -p gpus48
#SBATCH --gres gpu:1
#SBATCH --job-name=mnist-metrics
##SBATCH --nodelist=mira01
#SBATCH --output=src/counterfactuals/benchmarking/slurm_logs/slurm.%N.%j.log

source /vol/biomedic3/rrr2417/.bashrc
cd /vol/biomedic3/rrr2417/cxr-generation/

# poetry run python3 -m counterfactuals.benchmarking.morphomnist4 \
#     --model hvae \
#     --version "" \
#     --batch-size 1024 \
#     --last-batch 0 \
#     --seed 10000


# --batch-size 1024 \
# poetry run python3 -m counterfactuals.benchmarking.morphomnist4 \
#     --model diffusion \
#     --version "" \
#     --batch-size 1024 \
#     --last-batch 0 \
#     --seed 43 \
#     --cfg \
#     --guidance-scale 4.5 \
# 
# echo

poetry run python3 -m counterfactuals.benchmarking.morphomnist4_it \
    --batch-size 1024 \
    --last-batch 0 \
    --guidance-scale 3.0

echo

# poetry run python3 -m counterfactuals.benchmarking.morphomnist_causal_effects \
#     --model diffusion \
#     --cfg \
#     --guidance-scale 3 \
#     --version "" \
#     --batch-size 1024 \
#     --last-batch -1 \
#     --seed 10000