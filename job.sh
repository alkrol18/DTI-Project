#!/usr/bin/env bash
# =============================================================================
# SLURM job script -- Cross-Modal DTI Transformer (DAVIS / Cold-Target)
# =============================================================================
# Partition   : gpu-common  (swap to scavenger for lower-priority preemptable)
# GPU         : 1x A100 or A5000  (any available via gres)
# CPUs        : 4
# RAM         : 32 GB
# Preemption  : SIGUSR1 sent 90 s before wall-time; Python handler saves
#               checkpoint and exits cleanly; --requeue relaunches the job;
#               --resume flag picks up from the last completed epoch.
# =============================================================================

#SBATCH --job-name=dti_cross_modal
#SBATCH --partition=gpu-common
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/dti_%j.out
#SBATCH --error=logs/dti_%j.err
#SBATCH --signal=SIGUSR1@90          # warn Python 90 s before wall-time
#SBATCH --requeue                    # auto-requeue on preemption / node failure
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=alexjkrol@gmail.com

# ---------------------------------------------------------------------------
# Scavenger partition variant:
# Comment out the gpu-common --partition / --gres lines above and
# uncomment these two to run at lower priority with preemption enabled.
# ---------------------------------------------------------------------------
# #SBATCH --partition=scavenger
# #SBATCH --qos=scavenger

set -euo pipefail

# ---------------------------------------------------------------------------
# Environment info
# ---------------------------------------------------------------------------
echo "========================================="
echo "Job ID      : ${SLURM_JOB_ID}"
echo "Node        : $(hostname)"
echo "GPUs        : ${CUDA_VISIBLE_DEVICES:-unset}"
echo "Start       : $(date)"
echo "========================================="

# Activate conda environment -- adjust name/path to match your cluster setup
CONDA_ENV="dti"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

# ---------------------------------------------------------------------------
# Install / update dependencies (no-op if already present)
# ---------------------------------------------------------------------------
pip install --quiet --upgrade \
    PyTDC \
    transformers \
    torch \
    numpy \
    pandas

# ---------------------------------------------------------------------------
# Paths -- all relative to the directory from which sbatch was called
# ---------------------------------------------------------------------------
WORKDIR="${SLURM_SUBMIT_DIR}"
DATA_DIR="${WORKDIR}/data"
CACHE_DIR="${WORKDIR}/hf_cache"
OUTPUT_DIR="${WORKDIR}/outputs"
CHECKPOINT="${WORKDIR}/checkpoint.pt"

mkdir -p "${DATA_DIR}" "${CACHE_DIR}" "${OUTPUT_DIR}" logs

# ---------------------------------------------------------------------------
# Resume logic
# If a checkpoint from a previous (possibly preempted) run exists, pass
# --resume so training continues from the saved epoch rather than epoch 1.
# ---------------------------------------------------------------------------
RESUME_FLAG=""
if [[ -f "${CHECKPOINT}" ]]; then
    echo "Checkpoint found -- resuming training from last saved epoch."
    RESUME_FLAG="--resume"
fi

# ---------------------------------------------------------------------------
# Launch training
# ---------------------------------------------------------------------------
python "${WORKDIR}/dti_cross_modal.py" \
    --data_dir          "${DATA_DIR}"        \
    --cache_dir         "${CACHE_DIR}"       \
    --output_dir        "${OUTPUT_DIR}"      \
    --checkpoint_path   "${CHECKPOINT}"      \
    --mol_model         "seyonec/ChemBERTa-zinc-base-v1" \
    --prot_model        "facebook/esm2_t6_8M_UR50D"      \
    --d_model           256                  \
    --n_heads           8                    \
    --n_cross_layers    2                    \
    --dropout           0.1                  \
    --max_mol_len       128                  \
    --prot_chunk_size   1020                 \
    --prot_stride       512                  \
    --epochs            50                   \
    --batch_size        16                   \
    --lr                1e-4                 \
    --weight_decay      1e-4                 \
    --warmup_ratio      0.1                  \
    --grad_clip         1.0                  \
    --fp16                                   \
    --num_workers       4                    \
    --seq_id_threshold  0.3                  \
    --val_frac          0.1                  \
    --test_frac         0.2                  \
    --seed              42                   \
    --run_analysis                           \
    --saliency_n        20                   \
    ${RESUME_FLAG}

echo "========================================="
echo "Job finished : $(date)"
echo "========================================="
