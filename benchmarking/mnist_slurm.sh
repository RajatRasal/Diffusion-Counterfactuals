#!/bin/bash
#SBATCH -p gpus48
#SBATCH --gres gpu:1
#SBATCH --job-name=mnist-metrics
#SBATCH --output=src/counterfactuals/benchmarking/slurm_logs/slurm.%N.%j.log

source /vol/biomedic3/rrr2417/.bashrc
cd /vol/biomedic3/rrr2417/cxr-generation/

# poetry run python3 -m counterfactuals.benchmarking.morphomnist3 \
#     --model diffusion \
#     --version enc_beta_2.0_p_0.0_1731171670.4524353 \
#     --guidance-scale 5 \
#     --timestep 15 \
#     --batch-size 2048

# --version enc_beta_2.0_p_0.1_1730888504.104032 \
# --version no_enc_beta_1.0_p_0.1_1730888535.4530525 \

# --version enc_beta_2.0_p_0.0_1731171670.4524353 \
# --version no_enc_beta_1.0_p_0.0_1731171454.736434 \


# poetry run python3 -m counterfactuals.benchmarking.morphomnist3 \
#     --model vae \
#     --version version_27 \
#     --batch-size 1024 \
#     --last-batch -1
# 
# 
# echo

poetry run python3 -m counterfactuals.benchmarking.morphomnist3 \
    --model hvae \
    --version version_14 \
    --batch-size 1024 \
    --last-batch -1

echo

# poetry run python3 -m counterfactuals.benchmarking.morphomnist3 \
#     --model hvae \
#     --version version_9 \
#     --batch-size 1024 \
#     --last-batch -1
# 
# echo