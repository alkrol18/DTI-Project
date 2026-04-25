#!/bin/bash
#SBATCH --job-name=dti_v4
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=10:00:00
#SBATCH --output=/home/users/ak724/training_v4.log

source /usr/pkg/miniconda-23.9.0/etc/profile.d/conda.sh
conda activate dti_research

cd /home/users/ak724

python dti_cross_modal.py \
  --batch_size 32 \
  --num_workers 4 \
  --scheduler cosine \
  --grad_clip 1.0 \
  --patience 15 \
  --lora_rank 8 \
  --lora_alpha 16 \
  --dropout 0.2 \
  --split_method random \
  --use_gcn \
  --resume \
  --run_analysis
