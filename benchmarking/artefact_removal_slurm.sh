#!/bin/bash
#SBATCH -p gpus24
#SBATCH --gres gpu:1
#SBATCH --job-name=artefact_removal
#SBATCH --nodelist=mira10
#SBATCH --output=src/counterfactuals/benchmarking/slurm_logs/slurm.%N.%j.log

source /vol/biomedic3/rrr2417/.bashrc
cd /vol/biomedic3/rrr2417/cxr-generation/

poetry run python3 -m counterfactuals.benchmarking.embed_artefact_removal \
    --last-batch -1 \
    --batch-size 128 \
    --split train \
    --edit triangle