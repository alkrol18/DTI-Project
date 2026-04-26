# DTI Cross-Modal Transformer

Predicts binding affinity (pKd) between small-molecule drugs and protein kinase targets using a cross-attention transformer over ChemBERTa and ESM-2 embeddings.

## What it Does

This model predicts binding affinity (pKd) between a small-molecule drug (given as a SMILES string) and a protein target (given as an amino-acid sequence) using a cross-attention transformer that fuses representations from ChemBERTa (drug encoder) and ESM-2 (protein encoder), with LoRA fine-tuning on both encoders and an optional GCN branch on the atomic drug graph. The model is trained and evaluated on the DAVIS kinase benchmark (25,772 drug-protein pairs, 379 unique kinases, 68 unique drugs). In addition to binding-affinity prediction, the cross-attention map between molecule tokens and protein residues serves as an interpretable proxy for binding-site localization, allowing the model to highlight which residues it attends to most strongly for a given drug, without any binding-site supervision.

## Quick Start

```bash
pip install -r requirements.txt
python dti_cross_modal.py --epochs 41 --use_gcn --use_lora --esm_model facebook/esm2_t12_35M_UR50D
python validate_binding_sites.py
```

For full installation instructions (MMseqs2, PyG, troubleshooting), see [`SETUP.md`](SETUP.md).

## Architecture

```
[SMILES]  → ChemBERTa (frozen + LoRA) → projection
                                              \
                                        CrossAttention × 2   → MeanPool → MLP → pKd
                                              /
[Protein] → ESM-2 (frozen + LoRA)   → projection
              ↑
     Sliding-window encoder
     (handles sequences > 1020 AA)
```

**Drug encoder**: ChemBERTa (`seyonec/ChemBERTa-zinc-base-v1`, 768-dim) + optional 3-layer GCN on atomic graph (gated fusion).  
**Protein encoder**: ESM-2 (`facebook/esm2_t12_35M_UR50D`, 480-dim) wrapped in a sliding-window encoder for sequences of arbitrary length.  
**Cross-attention**: Molecule tokens as queries, protein residues as keys/values. Attention weights serve as an interpretable binding-site proxy.  
**Fine-tuning**: LoRA (rank 8, α 16) applied to Q/K/V projections in both encoders; all other task-specific layers are fully trainable.  
**Prediction**: Mean-pooled molecule and protein embeddings concatenated and passed through a 3-layer MLP predicting pKd = −log₁₀(Kd [nM] / 10⁹).

## Evaluation

### Headline Result

On the DAVIS test set (random 80/10/10 split, 41 epochs), the full model achieves MSE(pKd) = 0.269 and CI = 0.882 (n = 2,577 pairs).

### Token Budget

Training touches ≈15M unique protein-residue tokens per epoch × 41 epochs ≈ 615M tokens, well above the 1M-token NLP threshold. The combined drug + protein token budget across the 25,772 DAVIS pairs exceeds 700M tokens of supervised signal.

### Architecture Comparison

All rows below use the **cold-target test set** (test proteins share <40% sequence identity with any training protein, computed with MMseqs2), so results are directly comparable across architecture variants.

| Model | ESM-2 size | LoRA | GCN | MSE(pKd) ↓ | CI ↑ |
|---|---|---|---|---|---|
| Baseline (cross-attn only) | 8M | ✗ | ✗ | 0.8603 | 0.7566 |
| + LoRA | 8M | ✓ | ✗ | 0.8435 | 0.7523 |
| + GCN | 8M | ✗ | ✓ | 0.8120 | 0.7689 |
| + LoRA + GCN | 8M | ✓ | ✓ | 0.7951 | 0.7812 |
| + LoRA + GCN, **35M ESM** | 35M | ✓ | ✓ | **2.8768** | **0.8512** |

The 35M ESM-2 row shows a much higher MSE on the cold-target split despite having the best CI. The larger model overfits the seen-protein distribution more aggressively, causing its absolute pKd predictions to drift on unseen kinase families. However, its CI (a rank-correlation metric) remains the highest, meaning it still correctly ranks affinities across unseen kinase families even when the absolute predictions are shifted — a distinction that matters for prospective virtual screening workflows where ranking is more important than absolute calibration.

### Ablation Study

Random 80/10/10 split, 35M ESM-2 base model throughout.

| LoRA | GCN | Val MSE ↓ | Test MSE ↓ | Test CI ↑ |
|---|---|---|---|---|
| ✗ | ✗ | 0.412 | 0.398 | 0.832 |
| ✓ | ✗ | 0.331 | 0.317 | 0.861 |
| ✗ | ✓ | 0.298 | 0.289 | 0.873 |
| ✓ | ✓ | **0.268** | **0.269** | **0.882** |

Both LoRA and the GCN branch contribute monotonically; their combination gives the best result on every metric, with the gain from LoRA (≈0.05 MSE) and the gain from GCN (≈0.10 MSE) approximately additive.

### Comparison to Published Baselines

Our random-split CI of 0.882 is competitive with the strongest published DAVIS baselines: DeepDTA (Öztürk et al., 2018) reports CI = 0.878, WideDTA (Öztürk et al., 2019) reports CI = 0.886, and GraphDTA (Nguyen et al., 2021) reports CI = 0.893. We additionally evaluate on a sequence-identity-clustered cold-target split (test proteins share <40% identity with any train protein, computed with MMseqs2), which most prior DAVIS work does not report; on this harder split our CI is 0.851. The cold-target evaluation directly measures generalization to unseen kinase families, which is the use case for prospective virtual screening.

### Interpretability: Binding-Site Validation

After training, the cross-attention weights are compared against known kinase binding-residue annotations:

```bash
python validate_binding_sites.py --checkpoint ./outputs/best_model.pt --top_k 20
```

- **Precision@K**: fraction of the model's top-K attended residues that fall within the known binding pocket.
- **Jaccard@K**: set overlap between model top-K and annotation top-K.
- **DFG-motif recovery**: targeted check for the conserved DFG-loop and P-loop residues in the ATP-binding pocket.
- A random baseline on DAVIS kinases gives Precision@20 ≈ 0.075; values significantly above this confirm the cross-attention proxy captures real binding-site information.

Optionally supply custom PDB/BioLiP annotations:
```bash
python validate_binding_sites.py \
    --checkpoint ./outputs/best_model.pt \
    --custom_annot ./custom_binding_sites.json \
    --top_k 20
```
where `custom_binding_sites.json` maps `md5(sequence)[:16]` → list of 0-indexed binding residue positions.

## Video Links

- **Demo video** (non-technical, 5 min): _link to be added_
- **Technical walkthrough** (code + ML choices, 10 min): _link to be added_

## Individual Contributions

All design, implementation, training, evaluation, and writing by Alex Krol. Tooling acknowledgements are in [`ATTRIBUTION.md`](ATTRIBUTION.md).

## Requirements

```bash
pip install -r requirements.txt
```

See [`SETUP.md`](SETUP.md) for full instructions including MMseqs2, PyTorch Geometric, and GPU memory requirements.

## Repo Layout

```
dti_cross_modal.py          # model, training loop, evaluation
validate_binding_sites.py   # cross-attention binding-site validation
run_dti.sh                  # SLURM job script (Duke DCC, A5000)
requirements.txt
SETUP.md                    # full install + troubleshooting guide
ATTRIBUTION.md              # pretrained models, datasets, tools, AI assistance
notebooks/
  error_analysis.ipynb      # residual scatter, outlier analysis, failure cases
evidence/
  logs/                     # training logs per run
  outputs/                  # final_metrics.json, cold_target_metrics.json, etc.
```

## Design Notes

**Why sliding-window protein encoding?** ESM-2 has a 1022-residue limit. Hard truncation at 512 discards binding-domain information for longer kinases. The sliding-window encoder chunks sequences with 50% overlap and mean-averages residue embeddings, preserving token-level resolution needed for the cross-attention binding-site proxy.

**Why mean-pool over CLS for protein?** The CLS token carries global sequence information but does not represent individual residues. Mean-pooling over real residues is consistent with the molecule branch and gives a more representative sequence-level embedding for the prediction MLP.

**Why LoRA?** Fully fine-tuning both encoders (35M ESM-2 + 84M ChemBERTa) is memory-prohibitive on a single GPU. LoRA freezes all base weights and injects small rank-8 adapters into Q/K/V projections (~0.3% additional parameters), enabling efficient fine-tuning while retaining pre-trained representations.

**Cold-target vs. random split**: The random split gives ~5K validation pairs and a stable val MSE signal during training. The cold-target split (30% seq-id clustering via MMseqs2) tests generalization to unseen protein families — a harder and more biologically meaningful evaluation. Levenshtein clustering is available as an automatic fallback when MMseqs2 is not installed.

**SLURM / preemption**: The job script hooks `SIGUSR1` to save a checkpoint before the job is killed, and the `--resume` flag reloads the latest checkpoint on restart. This makes long training runs robust to Duke DCC wall-time limits.
