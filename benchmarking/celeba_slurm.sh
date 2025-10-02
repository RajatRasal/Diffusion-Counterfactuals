#!/bin/bash
#SBATCH -p gpus48
#SBATCH --gres gpu:1
#SBATCH --job-name=celeba-metrics
##SBATCH --nodelist=mira09
#SBATCH --output=src/counterfactuals/benchmarking/slurm_logs/slurm.%N.%j.log

source /vol/biomedic3/rrr2417/.bashrc
cd /vol/biomedic3/rrr2417/cxr-generation/


# poetry run python3 -m counterfactuals.benchmarking.celeba \
#     --model vae \
#     --version version_17 \
#     --batch-size 10 \
#     --last-batch 10


poetry run python3 -m counterfactuals.benchmarking.celeba \
    --model hvae \
    --version version_0 \
    --batch-size 512 \
    --last-batch -1


# poetry run python3 -m counterfactuals.benchmarking.celeba \
#     --model diffusion \
#     --version "" \
#     --batch-size 1000 \
#     --last-batch 0 \
#     --seed 10000 \
#     --semantic \
#     --cfg \
#     --guidance 2


# poetry run python3 -m counterfactuals.benchmarking.celeba \
#     --model diffusion \
#     --version "" \
#     --batch-size 128 \
#     --last-batch -1 \
#     --seed 10000 \
#     --cfg \
#     --guidance 3 \
#     --semantic


# poetry run python3 -m counterfactuals.benchmarking.celeba \
#     --model diffusion \
#     --cfg \
#     --guidance-scale 3 \
#     --version "" \
#     --batch-size 2000 \
#     --last-batch -1 \
#     --seed 10000


# poetry run python3 -m counterfactuals.benchmarking.celeba \
#     --model diffusion \
#     --semantic \
#     --cfg \
#     --guidance-scale 1 \
#     --version "" \
#     --batch-size 2000 \
#     --last-batch -1 \
#     --seed 10000


# poetry run python3 -m counterfactuals.benchmarking.celeba \
#     --model diffusion \
#     --semantic \
#     --cfg \
#     --guidance-scale 3 \
#     --version "" \
#     --batch-size 2000 \
#     --last-batch -1 \
#     --seed 10000