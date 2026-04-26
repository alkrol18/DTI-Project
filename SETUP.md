# Setup Guide

## System Requirements

- Python 3.10+
- CUDA 11.8+ for GPU training (CPU inference works but is slow — expect ~30× longer per epoch)
- ~6 GB GPU memory for the 35M ESM-2 model (`facebook/esm2_t12_35M_UR50D`); ~3 GB for the 8M variant

## Step 1 — Python Dependencies

```bash
pip install -r requirements.txt
```

**PyTorch Geometric note**: `torch-geometric` is listed in `requirements.txt`, but on some systems the sparse-ops backends need a separate install step:

```bash
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-${TORCH_VER}+${CUDA_VER}.html
```

Replace `${TORCH_VER}` with your PyTorch version (e.g. `2.1.0`) and `${CUDA_VER}` with your CUDA version (e.g. `cu118`). See [https://pytorch-geometric.readthedocs.io/](https://pytorch-geometric.readthedocs.io/) for the full compatibility matrix.

## Step 2 — MMseqs2 (cold-target clustering)

MMseqs2 is used to cluster proteins by sequence identity and assign test-set proteins that share <40% identity with any training protein (the "cold-target" split). This gives a biologically rigorous generalization test.

```bash
conda install -c bioconda mmseqs2
```

If MMseqs2 is unavailable, the code automatically falls back to a Levenshtein-based clustering. The fallback is slower and less precise but produces a valid cold-target split without any additional install steps.

## Step 3 — Data

DAVIS downloads automatically the first time you run training, via `tdc.multi_pred.DTI`. The download is ~50 MB and is cached under `./data/`. No API keys required.

Dataset summary:
- 25,772 drug-protein pairs
- 68 unique drugs (kinase inhibitors, SMILES strings)
- 379 unique protein targets (human kinases, amino-acid sequences)
- Labels: Kd [nM]; converted to pKd = −log₁₀(Kd / 10⁹) for training

## Step 4 — Pretrained Models

Two pretrained models are downloaded from Hugging Face on first use:

| Model | Hugging Face ID | Size |
|---|---|---|
| ChemBERTa | `seyonec/ChemBERTa-zinc-base-v1` | ~84 MB |
| ESM-2 35M | `facebook/esm2_t12_35M_UR50D` | ~140 MB |
| ESM-2 8M | `facebook/esm2_t6_8M_UR50D` | ~31 MB |

Downloads are cached in the default Hugging Face cache directory (`~/.cache/huggingface/`). No accounts or API tokens required for these models.

## Step 5 — Sanity Check

Run a single-epoch pass to verify the full pipeline (data loading, model forward pass, LoRA, GCN, evaluation):

```bash
python dti_cross_modal.py --epochs 1 --batch_size 8
```

This should complete in approximately 3 minutes on a single GPU. Expected output: a non-NaN val MSE and CI printed at the end of epoch 1.

## Full Training Run

```bash
python dti_cross_modal.py --epochs 41 --use_gcn --use_lora --esm_model facebook/esm2_t12_35M_UR50D
```

Full training takes ~8 hours on an RTX 2080 Ti / A5000. On Duke DCC use the SLURM script:

```bash
sbatch run_dti.sh
```

Pass `--resume` to continue from an existing checkpoint after preemption.

## Troubleshooting

**Out of GPU memory (OOM)**: Drop to the 8M ESM-2 variant:

```bash
python dti_cross_modal.py --epochs 41 --use_gcn --use_lora --esm_model facebook/esm2_t6_8M_UR50D
```

The 8M model uses ~3 GB and trains in ~3–4 hours on a single GPU.

**`torch_geometric` import errors**: Make sure `torch-scatter` and `torch-sparse` are installed for your exact PyTorch + CUDA version combination (see Step 1). Mismatched versions are the most common cause of PyG import failures.

**MMseqs2 not found**: The code prints a warning and falls back to Levenshtein clustering automatically — no action required unless you specifically want the MMseqs2-based split.
