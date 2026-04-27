#!/usr/bin/env python3
"""
validate_binding_sites.py
=========================
Wires the cross-modal attention map to empirical binding-site annotations
as required by CRIT-3 (attention map validation scaffolding).

Purpose
-------
The cross-attention weights in CrossModalDTI act as a proxy for physical
binding-site residues.  This script provides empirical grounding for that
claim by comparing the model's top-K attended residues against two sources
of ground-truth:

  1. Motif-based annotations  — DFG motif (Asp-Phe-Gly) and P-loop
     (Gly-x-Gly-x-x-Gly) found programmatically in kinase sequences.
     These landmarks are universally conserved in the kinase ATP-binding
     pocket and require no external data download.

  2. Custom JSON annotations  — user-supplied dict mapping protein-sequence
     hash -> list of known binding residue indices (0-indexed, relative to
     the sequence as it appears in DAVIS).  Useful for extending the
     evaluation with PDB/BioLiP/sc-PDB data.

Metrics
-------
  Precision@K  : fraction of model's top-K residues that fall in the
                 annotated binding site window.
  Jaccard@K    : intersection-over-union between top-K model residues and
                 the annotated binding site set.

Usage
-----
  # After training, run from the project root:
  python validate_binding_sites.py \\
      --checkpoint    ./outputs/best_model.pt \\
      --data_dir      ./data \\
      --cache_dir     ./hf_cache \\
      --output_dir    ./outputs \\
      --top_k         20 \\
      --attn_topk     None

  # With custom PDB annotations:
  python validate_binding_sites.py \\
      --checkpoint    ./outputs/best_model.pt \\
      --custom_annot  ./custom_binding_sites.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# Import from the main training script
from dti_cross_modal import (
    CrossModalDTI,
    DTIDataset,
    dti_collate_fn,
    _top_binding_residues,
    log,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)


# ─────────────────────────────────────────────────────────────────────────────
# Motif-based binding site detection
# ─────────────────────────────────────────────────────────────────────────────

# Kinase binding-pocket window half-width.
# The DFG motif (or P-loop) is used as an anchor; residues within ±WINDOW
# positions are considered part of the ATP-binding site.
_WINDOW = 15


def find_dfg_motif(sequence: str) -> List[int]:
    """
    Returns 0-indexed positions of the D in every DFG (Asp-Phe-Gly) motif.
    DFG is the canonical gatekeeper for the ATP-binding DFG-loop in kinases.
    Multiple matches can occur in multi-domain sequences; all are returned.
    """
    return [m.start() for m in re.finditer(r"DFG", sequence)]


def find_ploop_motif(sequence: str) -> List[int]:
    """
    Returns 0-indexed positions of the first G in P-loop (GxGxxG) motifs.
    The P-loop anchors the phosphate groups of ATP in the active site.
    Regex: G[A-Z]G[A-Z]{2}G
    """
    return [m.start() for m in re.finditer(r"G[A-Z]G[A-Z]{2}G", sequence)]


def motif_binding_site(sequence: str, window: int = _WINDOW) -> Set[int]:
    """
    Derives a set of binding-site residue indices (0-indexed) from
    conserved kinase motifs found in `sequence`.

    Priority: DFG motif (most specific ATP-pocket marker).
    Fallback:  P-loop if no DFG found.
    Returns empty set if neither motif is found (non-kinase sequence).
    """
    positions = find_dfg_motif(sequence)
    source    = "DFG"
    if not positions:
        positions = find_ploop_motif(sequence)
        source    = "P-loop"
    if not positions:
        return set()

    # Use the first occurrence (N-terminal kinase domain convention)
    anchor = positions[0]
    site   = set(range(max(0, anchor - window), min(len(sequence), anchor + window + 1)))
    log.debug(f"  Motif anchor: {source} @ position {anchor}, window ±{window} -> {len(site)} residues")
    return site


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def precision_at_k(predicted: List[int], ground_truth: Set[int]) -> float:
    """Fraction of predicted residues that are in the ground-truth binding site."""
    if not predicted:
        return 0.0
    hits = sum(1 for r in predicted if r in ground_truth)
    return hits / len(predicted)


def jaccard_at_k(predicted: List[int], ground_truth: Set[int]) -> float:
    """Intersection-over-union between predicted top-K and ground-truth sets."""
    pred_set = set(predicted)
    if not pred_set and not ground_truth:
        return 1.0
    intersection = len(pred_set & ground_truth)
    union        = len(pred_set | ground_truth)
    return intersection / union if union > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Inference helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_attention_map(
    model:    CrossModalDTI,
    mol_ids:  torch.Tensor,
    mol_mask: torch.Tensor,
    prot_ids: torch.Tensor,
    prot_mask:torch.Tensor,
    device:   torch.device,
) -> np.ndarray:
    """
    Run a single forward pass with return_attentions=True and return the
    head-averaged attention map from the last cross-attention layer.

    Returns: attn_map (L_mol, L_prot) as a numpy float32 array.
    """
    model.eval()
    with torch.no_grad():
        out = model(
            mol_input_ids       = mol_ids.to(device),
            mol_attention_mask  = mol_mask.to(device),
            prot_input_ids      = prot_ids.to(device),
            prot_attention_mask = prot_mask.to(device),
            return_attentions   = True,
        )
    return out["attn_weights"].squeeze(0).cpu().numpy()   # (L_mol, L_prot)


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation loop
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_binding_site_recovery(
    model:       CrossModalDTI,
    test_df,
    mol_tok:     AutoTokenizer,
    prot_tok:    AutoTokenizer,
    device:      torch.device,
    top_k:       int           = 20,
    attn_topk:   Optional[int] = None,
    custom_annot:Optional[Dict] = None,
    max_samples: int           = 200,
    seed:        int           = 42,
) -> Dict:
    """
    For each test pair:
      1. Detect the binding site from DFG/P-loop motifs (or custom annotations).
      2. Run the model to obtain cross-attention weights.
      3. Compute Precision@K and Jaccard@K.

    Returns a summary dict with per-sample results and aggregate statistics.
    """
    rng     = np.random.default_rng(seed)
    n_samp  = min(max_samples, len(test_df))
    indices = rng.choice(len(test_df), n_samp, replace=False)

    results = []
    p_at_k_all, jaccard_all = [], []

    motif_found   = 0
    custom_used   = 0
    no_annotation = 0

    for idx in indices:
        row      = test_df.iloc[int(idx)]
        sequence = str(row["Target"])
        smiles   = str(row["Drug"])

        # ── Ground-truth binding site ─────────────────────────────────────
        seq_hash = hashlib.md5(sequence.encode()).hexdigest()[:16]

        if custom_annot and seq_hash in custom_annot:
            gt_site = set(custom_annot[seq_hash])
            annot_source = "custom"
            custom_used += 1
        else:
            gt_site = motif_binding_site(sequence)
            annot_source = "motif"
            if gt_site:
                motif_found += 1
            else:
                no_annotation += 1
                # Cannot evaluate without ground truth; skip but keep for coverage stats
                results.append({
                    "idx":         int(idx),
                    "smiles":      smiles[:40],
                    "seq_hash":    seq_hash,
                    "annot_source": "none",
                    "gt_size":     0,
                    "precision_at_k": None,
                    "jaccard_at_k":   None,
                    "top_k_residues": [],
                })
                continue

        # ── Model inference ───────────────────────────────────────────────
        single  = test_df.iloc[[int(idx)]].reset_index(drop=True)
        ds      = DTIDataset(single, mol_tok, prot_tok)
        item    = ds[0]
        mol_ids   = item["mol_input_ids"].unsqueeze(0)
        mol_mask  = item["mol_attention_mask"].unsqueeze(0)
        prot_ids  = item["prot_input_ids"].unsqueeze(0)
        prot_mask = item["prot_attention_mask"].unsqueeze(0)

        try:
            attn_map = get_attention_map(
                model, mol_ids, mol_mask, prot_ids, prot_mask, device
            )
        except Exception as exc:
            log.warning(f"  Inference failed for sample {idx}: {exc}")
            continue

        mol_len  = int(mol_mask.sum())
        prot_len = int(prot_mask.sum())

        top_residues = _top_binding_residues(
            attn_map, mol_len, prot_len,
            attn_topk=attn_topk, top_n=top_k,
        )

        p_k = precision_at_k(top_residues, gt_site)
        j_k = jaccard_at_k(top_residues, gt_site)
        p_at_k_all.append(p_k)
        jaccard_all.append(j_k)

        results.append({
            "idx":            int(idx),
            "smiles":         smiles[:40],
            "seq_hash":       seq_hash,
            "annot_source":   annot_source,
            "gt_size":        len(gt_site),
            "precision_at_k": float(p_k),
            "jaccard_at_k":   float(j_k),
            "top_k_residues": top_residues,
        })

    # ── Aggregate ─────────────────────────────────────────────────────────
    evaluated    = [r for r in results if r["precision_at_k"] is not None]
    mean_p_at_k  = float(np.mean(p_at_k_all))  if p_at_k_all  else float("nan")
    mean_jaccard = float(np.mean(jaccard_all))  if jaccard_all else float("nan")

    summary = {
        "top_k":                 top_k,
        "attn_topk":             attn_topk,
        "n_evaluated":           len(evaluated),
        "n_motif_annotation":    motif_found,
        "n_custom_annotation":   custom_used,
        "n_no_annotation":       no_annotation,
        "mean_precision_at_k":   mean_p_at_k,
        "mean_jaccard_at_k":     mean_jaccard,
        "interpretation": (
            f"Precision@{top_k} = fraction of the model's top-{top_k} attended "
            f"residues that fall within the annotated binding site (DFG/P-loop "
            f"window ±{_WINDOW} residues).  "
            f"Jaccard@{top_k} = IoU between predicted and ground-truth sets.  "
            "Values >> random (random Precision@K ~ site_size / seq_len) indicate "
            "the cross-attention proxy captures real binding-site information."
        ),
        "per_sample": results,
    }

    log.info(
        f"Binding-site recovery: "
        f"Precision@{top_k} = {mean_p_at_k:.3f}  |  "
        f"Jaccard@{top_k}   = {mean_jaccard:.3f}  "
        f"(n={len(evaluated)}, {no_annotation} samples skipped — no annotation)"
    )
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Validate cross-attention binding-site proxy against motifs/annotations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint",    required=True,
                   help="Path to best_model.pt from training")
    p.add_argument("--data_dir",      default="./data")
    p.add_argument("--cache_dir",     default="./hf_cache")
    p.add_argument("--output_dir",    default="./outputs")
    p.add_argument("--mol_model",     default="seyonec/ChemBERTa-zinc-base-v1")
    p.add_argument("--prot_model",    default="facebook/esm2_t6_8M_UR50D")
    p.add_argument("--top_k",         default=20,   type=int,
                   help="Number of top attended residues to evaluate")
    p.add_argument("--attn_topk",     default=None, type=int,
                   help="Top-K molecule body tokens to use for residue scoring "
                        "(None = all body tokens, max-pooled)")
    p.add_argument("--max_samples",   default=200,  type=int,
                   help="Max test-set samples to evaluate")
    p.add_argument("--seed",          default=42,   type=int)
    p.add_argument("--custom_annot",  default=None,
                   help="Path to JSON file mapping seq_hash -> list of binding "
                        "residue indices (0-indexed in DAVIS sequence).  "
                        "Format: {\"<md5_16hex>\": [idx1, idx2, ...], ...}")
    # Split params (must match training config to reconstruct test set)
    p.add_argument("--seq_id_threshold", default=0.30, type=float)
    p.add_argument("--val_frac",         default=0.10, type=float)
    p.add_argument("--test_frac",        default=0.20, type=float)
    p.add_argument("--max_mol_len",      default=128,  type=int)
    p.add_argument("--prot_chunk_size",  default=1020, type=int)
    p.add_argument("--prot_stride",      default=512,  type=int)
    p.add_argument("--d_model",          default=256,  type=int)
    p.add_argument("--n_heads",          default=8,    type=int)
    p.add_argument("--n_cross_layers",   default=2,    type=int)
    p.add_argument("--dropout",          default=0.10, type=float)
    return p


def main() -> None:
    cfg    = build_parser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device : {device}")

    # ── Reconstruct test split (same seed as training) ────────────────────
    log.info("Loading DAVIS and reconstructing test split ...")
    from tdc.multi_pred import DTI as TDC_DTI
    from dti_cross_modal import cold_target_split

    davis = TDC_DTI(name="DAVIS", path=cfg.data_dir)
    df    = davis.get_data()
    _, _, test_df = cold_target_split(
        df,
        seq_id_threshold = cfg.seq_id_threshold,
        val_frac         = cfg.val_frac,
        test_frac        = cfg.test_frac,
        seed             = cfg.seed,
        cache_dir        = cfg.data_dir,
    )
    log.info(f"Test set: {len(test_df)} pairs, {test_df['Target'].nunique()} proteins")

    # ── Load tokenizers ───────────────────────────────────────────────────
    mol_tok  = AutoTokenizer.from_pretrained(cfg.mol_model,  cache_dir=cfg.cache_dir)
    prot_tok = AutoTokenizer.from_pretrained(cfg.prot_model, cache_dir=cfg.cache_dir)

    # ── Load model ────────────────────────────────────────────────────────
    log.info(f"Loading model from {cfg.checkpoint} ...")
    model = CrossModalDTI(
        mol_model_name  = cfg.mol_model,
        prot_model_name = cfg.prot_model,
        d_model         = cfg.d_model,
        n_heads         = cfg.n_heads,
        n_cross_layers  = cfg.n_cross_layers,
        dropout         = cfg.dropout,
        cache_dir       = cfg.cache_dir,
        prot_chunk_size = cfg.prot_chunk_size,
        prot_stride     = cfg.prot_stride,
    ).to(device)

    ckpt = torch.load(cfg.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    log.info("Model loaded successfully.")

    # ── Optional custom annotations ───────────────────────────────────────
    custom_annot = None
    if cfg.custom_annot:
        with open(cfg.custom_annot) as fh:
            custom_annot = json.load(fh)
        log.info(f"Loaded {len(custom_annot)} custom binding-site annotations.")

    # ── Run evaluation ────────────────────────────────────────────────────
    summary = evaluate_binding_site_recovery(
        model        = model,
        test_df      = test_df,
        mol_tok      = mol_tok,
        prot_tok     = prot_tok,
        device       = device,
        top_k        = cfg.top_k,
        attn_topk    = cfg.attn_topk,
        custom_annot = custom_annot,
        max_samples  = cfg.max_samples,
        seed         = cfg.seed,
    )

    # ── Save results ──────────────────────────────────────────────────────
    out_path = Path(cfg.output_dir) / "binding_site_validation.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    log.info(f"Validation results -> {out_path}")

    # ── Print summary table ───────────────────────────────────────────────
    print("\n" + "="*60)
    print(f"  Binding-Site Validation Summary (top_k={cfg.top_k})")
    print("="*60)
    print(f"  Samples evaluated   : {summary['n_evaluated']}")
    print(f"  Motif annotations   : {summary['n_motif_annotation']}")
    print(f"  Custom annotations  : {summary['n_custom_annotation']}")
    print(f"  Skipped (no annot.) : {summary['n_no_annotation']}")
    print(f"  Mean Precision@{cfg.top_k:<3}  : {summary['mean_precision_at_k']:.4f}")
    print(f"  Mean Jaccard@{cfg.top_k:<3}   : {summary['mean_jaccard_at_k']:.4f}")
    print("="*60)
    print()
    print("  Interpretation:")
    print(f"  Random Precision@{cfg.top_k} baseline ~ (binding_site_size / seq_len)")
    print("  For DAVIS kinases with DFG window ±15 -> ~30 residues in a ~400-AA")
    print("  sequence, random Precision@20 ~ 30/400 = 0.075.")
    print("  Model precision significantly above 0.075 confirms the cross-attention")
    print("  proxy captures real binding-site information (L22/L23 bridge).")
    print()


if __name__ == "__main__":
    main()
