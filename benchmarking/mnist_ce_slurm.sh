#!/bin/bash
#SBATCH -p gpus48
#SBATCH --gres gpu:1
#SBATCH --job-name=mnist-metrics
#SBATCH --output=src/counterfactuals/benchmarking/slurm_logs/slurm.%N.%j.log

source /vol/biomedic3/rrr2417/.bashrc
cd /vol/biomedic3/rrr2417/cxr-generation/

poetry run python3 -m counterfactuals.benchmarking.morphomnist_causal_effects \
    --model diffusion \
    --version "" \
    --batch-size 1024 \
    --last-batch 0 \
    --seed 10000 \
    --semantic \
    --cfg \
    --guidance-scale 3
    # --batch-size 1024 \

echo

# poetry run python3 -m counterfactuals.benchmarking.morphomnist_causal_effects \
#     --model diffusion \
#     --cfg \
#     --guidance-scale 3 \
#     --version "" \
#     --batch-size 1024 \
#     --last-batch -1 \
#     --seed 10000