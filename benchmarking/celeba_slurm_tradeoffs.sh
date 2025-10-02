#!/bin/bash
#SBATCH -p gpus24
#SBATCH --gres gpu:1
#SBATCH --job-name=celeba-tradeoffs
##SBATCH --nodelist=mira10
#SBATCH --output=src/counterfactuals/benchmarking/slurm_logs_tradeoffs/slurm.%N.%j.log

source /vol/biomedic3/rrr2417/.bashrc
cd /vol/biomedic3/rrr2417/cxr-generation/


# for i in $(seq 2 10);
# do
#     echo -------------------------------
#     gs=$(echo "scale=2 ; $i / 2" | bc)
#     echo guidance scale: $gs;
#     poetry run python3 -m counterfactuals.benchmarking.celeba \
#         --model diffusion \
#         --semantic \
#         --cfg \
#         --guidance-scale $gs \
#         --version "" \
#         --batch-size 200 \
#         --last-batch 0 \
#         --seed 10000 \
#         --val
#     echo
# done


# for i in $(seq 2 10);
# do
#     echo -------------------------------
#     gs=$(echo "scale=2 ; $i / 2" | bc)
#     echo guidance scale: $gs;
#     poetry run python3 -m counterfactuals.benchmarking.celeba \
#         --model diffusion \
#         --cfg \
#         --guidance-scale $gs \
#         --version "" \
#         --batch-size 200 \
#         --last-batch 0 \
#         --seed 10000 \
#         --val
#     echo
# done

poetry run python3 -m counterfactuals.benchmarking.celeba \
    --model diffusion \
    --semantic \
    --cfg \
    --guidance-scale 3 \
    --version "" \
    --batch-size 200 \
    --last-batch 0 \
    --seed 10000