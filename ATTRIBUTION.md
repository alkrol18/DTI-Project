# Attribution

## Pretrained Models

**ChemBERTa-zinc-base-v1**  
- Authors: Chithrananda et al. (2020), "ChemBERTa: Large-Scale Self-Supervised Pretraining for Molecular Property Prediction"  
- Hugging Face ID: `seyonec/ChemBERTa-zinc-base-v1`  
- License: MIT  
- Used as the drug encoder; fine-tuned with LoRA rank-8 adapters.

**ESM-2 (8M and 35M parameter variants)**  
- Authors: Lin et al. (2023), "Evolutionary-scale prediction of atomic-level protein structure with a language model", Science  
- Hugging Face IDs: `facebook/esm2_t6_8M_UR50D`, `facebook/esm2_t12_35M_UR50D`  
- License: MIT  
- Used as the protein encoder with a sliding-window wrapper; fine-tuned with LoRA rank-8 adapters.

## Datasets

**DAVIS Kinase Dataset**  
- Davis, M. I., et al. (2011), "Comprehensive analysis of kinase inhibitor selectivity", Nature Biotechnology  
- Accessed via PyTDC (Huang et al., 2021): `tdc.multi_pred.DTI(name='DAVIS')`  
- 25,772 drug-protein pairs, 68 unique drugs, 379 unique proteins  
- Labels: Kd [nM]; converted to pKd for regression

**PyTDC**  
- Huang, K., et al. (2021), "Therapeutics Data Commons: Machine Learning Datasets and Tasks for Drug Discovery and Development"

## Libraries

| Library | Purpose |
|---|---|
| PyTorch | Core deep learning framework |
| Hugging Face `transformers` | ChemBERTa and ESM-2 model loading and tokenization |
| Hugging Face `peft` | LoRA adapter implementation |
| RDKit | SMILES parsing and molecular graph construction |
| PyTorch Geometric | GCN layers on atomic drug graphs |
| MMseqs2 | Sequence-identity-based protein clustering for cold-target split |
| python-Levenshtein | Fallback protein clustering when MMseqs2 is unavailable |
| NumPy | Numerical utilities |
| scikit-learn | Evaluation metrics and train/val/test splitting utilities |

## AI Development Tools

I used Claude Code (Anthropic) and Cursor as AI-assisted coding tools during development of this project. The model architecture decisions (cross-attention fusion, gated GCN branch, sliding-window protein encoder, LoRA configuration), training-loop logic, ablation design, and evaluation choices were mine. The AI tools helped with boilerplate code — specifically data-loader plumbing, logging setup, checkpoint save/load logic, error handling, and the SLURM SIGUSR1 preemption hook — and with refactoring passes to keep the codebase clean as the architecture evolved. All code was reviewed, tested, and integrated by me.

## Course Resources

CS 372 (Applied Machine Learning, Duke University, Spring 2026) lecture material on transformers, multi-head attention, and parameter-efficient fine-tuning (PEFT/LoRA) informed the architectural choices made in this project.
