#!/usr/bin/env python3
"""
Cross-Modal Transformer for Drug-Target Interaction (DTI)
=========================================================
Dataset  : DAVIS (PyTDC)
Split    : Cold-Target via MMseqs2 sequence clustering (Levenshtein fallback)
Molecule : seyonec/ChemBERTa-zinc-base-v1
Protein  : facebook/esm2_t6_8M_UR50D  (sliding-window encoder for long seqs)
Bridge   : L22 (Cross-Modal Attention) <-> L23 (Graph/Structural intuition)
           cross-attention weights act as proxy for physical binding-site residues

SLURM-aware: SIGUSR1 handler saves checkpoint before preemption.

Changelog (v2)
--------------
[CRIT-1] Cold-target split now uses MMseqs2 easy-cluster at 30 % seq-id
         threshold instead of raw Levenshtein distance.  Levenshtein retained
         as an automatic fallback when the mmseqs binary is not in PATH.
[CRIT-2] Protein hard-truncation at 512 tokens replaced by a sliding-window
         encoder that chunks long sequences, processes each chunk through
         ESM-2, and mean-averages overlapping residue embeddings.  Dynamic
         per-batch padding replaces fixed global padding.
[MED-4]  Removed nan_to_num in cross-attention softmax.  Root cause (all-pad
         protein row → all-inf scores → NaN) is now fixed at the mask level:
         a safe_mask ensures at least one valid key position per query.
[MED-5]  Protein pooling changed from ESM CLS token to mean-pool over real
         residues, consistent with the molecule branch.
[MED-6]  Attention aggregation now uses max-pool over molecule body tokens
         (excluding CLS/EOS) with optional top-K filtering (--attn_topk).
[LOW-7]  Predicted and true Kd values are now reported in both log1p scale
         (training objective) and nanomolar scale (expm1 inverse transform).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import pickle
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CLI Configuration
# ─────────────────────────────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Cross-Modal Transformer DTI — DAVIS / Cold-Target",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # paths
    p.add_argument("--data_dir",        default="./data")
    p.add_argument("--cache_dir",       default="./hf_cache")
    p.add_argument("--output_dir",      default="./outputs")
    p.add_argument("--checkpoint_path", default="./checkpoint.pt")
    # split  [CRIT-1]
    p.add_argument("--seq_id_threshold", default=0.30, type=float,
                   help="MMseqs2 sequence-identity threshold for cold-target "
                        "clustering (0.30 = 30%%).  Levenshtein fallback uses "
                        "1 - seq_id_threshold as the similarity floor.")
    p.add_argument("--val_frac",        default=0.10, type=float)
    p.add_argument("--test_frac",       default=0.20, type=float)
    # model
    p.add_argument("--mol_model",       default="seyonec/ChemBERTa-zinc-base-v1")
    p.add_argument("--prot_model",      default="facebook/esm2_t6_8M_UR50D")
    p.add_argument("--d_model",         default=256,  type=int)
    p.add_argument("--n_heads",         default=8,    type=int)
    p.add_argument("--n_cross_layers",  default=2,    type=int)
    p.add_argument("--dropout",         default=0.10, type=float)
    p.add_argument("--max_mol_len",     default=128,  type=int,
                   help="SMILES token budget (hard truncation is safe; "
                        "SMILES rarely exceed 100 tokens).")
    # [CRIT-2] prot_chunk_size replaces max_prot_len as the per-window limit
    p.add_argument("--prot_chunk_size", default=1020, type=int,
                   help="Max residue tokens per ESM-2 forward pass "
                        "(ESM-2 t6_8M supports up to 1022 incl. BOS/EOS; "
                        "1020 leaves 2 slots for special tokens).")
    p.add_argument("--prot_stride",     default=512,  type=int,
                   help="Sliding-window stride for proteins longer than "
                        "prot_chunk_size.  Overlap = chunk_size - stride.")
    # training
    p.add_argument("--epochs",          default=50,   type=int)
    p.add_argument("--batch_size",      default=16,   type=int)
    p.add_argument("--lr",              default=1e-4, type=float)
    p.add_argument("--weight_decay",    default=1e-4, type=float)
    p.add_argument("--warmup_ratio",    default=0.10, type=float,
                   help="Fraction of total steps used for linear warmup "
                        "(applies to cosine and linear schedulers).")
    p.add_argument("--scheduler",       default="cosine",
                   choices=["cosine", "linear", "plateau"],
                   help="LR scheduler.  cosine = cosine decay with warmup; "
                        "linear = linear decay with warmup; "
                        "plateau = ReduceLROnPlateau (monitors val MSE, "
                        "ignores warmup_ratio).")
    p.add_argument("--plateau_factor",  default=0.5,  type=float,
                   help="ReduceLROnPlateau multiplicative factor.")
    p.add_argument("--plateau_patience",default=5,    type=int,
                   help="ReduceLROnPlateau patience (epochs).")
    p.add_argument("--patience",        default=None, type=int,
                   help="Early-stopping patience in epochs (monitors val MSE). "
                        "Training stops when val MSE has not improved for this "
                        "many consecutive epochs.  None = disabled.")
    p.add_argument("--grad_clip",       default=1.00, type=float)
    p.add_argument("--fp16",            action="store_true", default=False)
    p.add_argument("--num_workers",     default=4,    type=int)
    p.add_argument("--seed",            default=42,   type=int)
    # resume / analysis
    p.add_argument("--resume",          action="store_true", default=False)
    p.add_argument("--run_analysis",    action="store_true", default=False)
    p.add_argument("--saliency_n",      default=20,   type=int)
    # [MED-6] top-K attention filter
    p.add_argument("--attn_topk",       default=None, type=int,
                   help="If set, use only the top-K molecule body tokens "
                        "(by max attention to any residue) when computing "
                        "per-residue binding scores.  None = use all body tokens.")
    return p


# ─────────────────────────────────────────────────────────────────────────────
# Cold-Target Split  — MMseqs2 primary, Levenshtein fallback  [CRIT-1]
# ─────────────────────────────────────────────────────────────────────────────

# ── Levenshtein helpers (kept for fallback) ───────────────────────────────────
def _lev_distance(a: str, b: str) -> int:
    """Standard O(|a|*|b|) Levenshtein edit distance."""
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for c_a in a:
        curr = [prev[0] + 1]
        for j, c_b in enumerate(b):
            curr.append(min(curr[j] + 1, prev[j + 1] + 1, prev[j] + (c_a != c_b)))
        prev = curr
    return prev[-1]


def _lev_similarity(a: str, b: str, max_prefix: int = 200) -> float:
    """Normalised Levenshtein similarity in [0,1], using first max_prefix chars."""
    a, b = a[:max_prefix], b[:max_prefix]
    if not a and not b:
        return 1.0
    return 1.0 - _lev_distance(a, b) / max(len(a), len(b))


def _union_find_clusters(
    proteins: List[str],
    sim_fn,
    threshold: float,
) -> Dict[int, List[str]]:
    """
    Union-Find clustering: two proteins are merged when sim_fn(a, b) >= threshold.
    Returns a dict mapping root-index -> list of protein sequences in that cluster.
    """
    n = len(proteins)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    for i in range(n):
        for j in range(i + 1, n):
            if sim_fn(proteins[i], proteins[j]) >= threshold:
                union(i, j)

    clusters: Dict[int, List[str]] = defaultdict(list)
    for i, p in enumerate(proteins):
        clusters[find(i)].append(p)
    return clusters


def _assign_clusters_to_splits(
    cluster_list: List[List[str]],
    df: pd.DataFrame,
    val_frac: float,
    test_frac: float,
    rng: np.random.Generator,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Shuffle clusters, assign to splits, and filter df rows."""
    rng.shuffle(cluster_list)
    nc     = len(cluster_list)
    n_test = max(1, int(nc * test_frac))
    n_val  = max(1, int(nc * val_frac))

    test_prots  = set(sum(cluster_list[:n_test], []))
    val_prots   = set(sum(cluster_list[n_test:n_test + n_val], []))
    train_prots = set(sum(cluster_list[n_test + n_val:], []))

    log.info(
        f"  Proteins -- train: {len(train_prots)}, "
        f"val: {len(val_prots)}, test: {len(test_prots)}"
    )
    train_df = df[df["Target"].isin(train_prots)].reset_index(drop=True)
    val_df   = df[df["Target"].isin(val_prots)  ].reset_index(drop=True)
    test_df  = df[df["Target"].isin(test_prots) ].reset_index(drop=True)
    log.info(
        f"  Pairs    -- train: {len(train_df)}, "
        f"val: {len(val_df)}, test: {len(test_df)}"
    )
    return train_df, val_df, test_df


def _levenshtein_cold_target_split(
    df: pd.DataFrame,
    seq_id_threshold: float = 0.30,
    val_frac: float   = 0.10,
    test_frac: float  = 0.20,
    seed: int         = 42,
    cache_dir: str    = "./data",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Fallback split using Levenshtein distance.  Less biologically rigorous
    than MMseqs2 but requires no binary dependencies.

    NOTE: Levenshtein similarity is computed on the first 200 chars of each
    sequence (sufficient for DAVIS kinase-family discrimination) to bound
    O(n^2) cost.  Two proteins with similarity >= (1 - seq_id_threshold) are
    placed in the same cluster.
    """
    log.info("  [Fallback] Using Levenshtein clustering ...")
    rng      = np.random.default_rng(seed)
    proteins = df["Target"].unique().tolist()
    n        = len(proteins)

    # Cached pairwise similarity matrix
    sim_cache = Path(cache_dir) / "prot_lev_sim_matrix.pkl"
    if sim_cache.exists():
        log.info("  Loading cached Levenshtein similarity matrix ...")
        with open(sim_cache, "rb") as fh:
            S = pickle.load(fh)
    else:
        log.info(f"  Computing pairwise Levenshtein similarities for {n} proteins ...")
        S = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            S[i, i] = 1.0
            for j in range(i + 1, n):
                s = _lev_similarity(proteins[i], proteins[j])
                S[i, j] = S[j, i] = s
            if (i + 1) % 10 == 0:
                log.info(f"    {i+1}/{n} done")
        sim_cache.parent.mkdir(parents=True, exist_ok=True)
        with open(sim_cache, "wb") as fh:
            pickle.dump(S, fh)

    lev_threshold = 1.0 - seq_id_threshold  # e.g. seq_id=0.30 -> sim>=0.70 are grouped
    clusters = _union_find_clusters(
        proteins,
        sim_fn=lambda a, b: float(S[proteins.index(a), proteins.index(b)]),
        threshold=lev_threshold,
    )
    log.info(f"  {len(clusters)} clusters (Levenshtein >= {lev_threshold:.2f})")
    return _assign_clusters_to_splits(list(clusters.values()), df, val_frac, test_frac, rng)


def _mmseqs2_cold_target_split(
    df: pd.DataFrame,
    seq_id_threshold: float = 0.30,
    val_frac: float   = 0.10,
    test_frac: float  = 0.20,
    seed: int         = 42,
    cache_dir: str    = "./data",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Primary split using MMseqs2 easy-cluster.

    Clusters proteins at `seq_id_threshold` sequence identity with 80 %
    bidirectional coverage (--cov-mode 1 -c 0.8).  This is the standard
    used by CATH/SCOPe and is far more biologically meaningful than raw
    string similarity: it accounts for alignment gaps and is sensitive to
    domain coverage, not just edit distance.

    Requires: mmseqs binary in PATH.
      conda install -c conda-forge -c bioconda mmseqs2
      OR download from https://github.com/soedinglab/MMseqs2/releases
    """
    rng      = np.random.default_rng(seed)
    proteins = df["Target"].unique().tolist()

    # Stable protein IDs for the FASTA file (MD5 hex of sequence)
    prot_to_id: Dict[str, str] = {}
    id_to_prot: Dict[str, str] = {}
    for p in proteins:
        pid = hashlib.md5(p.encode()).hexdigest()[:16]
        prot_to_id[p]  = pid
        id_to_prot[pid] = p

    work_dir = Path(cache_dir) / "mmseqs2_tmp"
    work_dir.mkdir(parents=True, exist_ok=True)
    fasta_path     = work_dir / "proteins.fasta"
    result_prefix  = str(work_dir / "clusters")
    mmseqs_tmp_dir = str(work_dir / "mmseqs_tmp")

    # Write FASTA (deterministic; skip re-run if already exists)
    if not fasta_path.exists():
        with open(fasta_path, "w") as fh:
            for p in proteins:
                fh.write(f">{prot_to_id[p]}\n{p}\n")

    cluster_tsv = Path(result_prefix + "_cluster.tsv")
    if not cluster_tsv.exists():
        cmd = [
            "mmseqs", "easy-cluster",
            str(fasta_path),
            result_prefix,
            mmseqs_tmp_dir,
            "--min-seq-id", str(seq_id_threshold),
            "--cov-mode", "1",
            "-c", "0.8",
            "--cluster-mode", "1",
            "-v", "1",
        ]
        log.info(f"  Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"MMseqs2 failed (exit {result.returncode}):\n{result.stderr}"
            )

    # Parse cluster TSV  (rep_id \t member_id)
    clusters: Dict[str, List[str]] = defaultdict(list)
    seen_ids = set(id_to_prot.keys())
    with open(cluster_tsv) as fh:
        for line in fh:
            parts = line.strip().split("\t")
            if len(parts) != 2:
                continue
            rep, member = parts
            if member in seen_ids:
                clusters[rep].append(id_to_prot[member])

    if not clusters:
        raise RuntimeError(
            "MMseqs2 cluster TSV is empty or IDs did not match.  "
            "Check that proteins.fasta was generated correctly."
        )

    log.info(
        f"  {len(clusters)} clusters at seq-id >= {seq_id_threshold:.0%} "
        f"(MMseqs2 easy-cluster)"
    )
    return _assign_clusters_to_splits(list(clusters.values()), df, val_frac, test_frac, rng)


def cold_target_split(
    df: pd.DataFrame,
    seq_id_threshold: float = 0.30,
    val_frac: float   = 0.10,
    test_frac: float  = 0.20,
    seed: int         = 42,
    cache_dir: str    = "./data",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Public entry-point for cold-target splitting.

    Tries MMseqs2 first (preferred — biologically rigorous sequence-identity
    clustering).  Falls back automatically to Levenshtein clustering if the
    `mmseqs` binary is not found in PATH, logging a prominent warning.
    """
    log.info("Building cold-target split ...")
    log.info(f"  Unique target proteins : {df['Target'].nunique()}")

    if shutil.which("mmseqs") is not None:
        log.info("  MMseqs2 found in PATH — using MMseqs2 clustering.")
        return _mmseqs2_cold_target_split(
            df, seq_id_threshold=seq_id_threshold,
            val_frac=val_frac, test_frac=test_frac,
            seed=seed, cache_dir=cache_dir,
        )
    else:
        log.warning(
            "  mmseqs binary NOT found in PATH.  Falling back to Levenshtein "
            "clustering.  This is less biologically rigorous and may allow "
            "structurally similar proteins to leak across train/test boundaries.\n"
            "  Install MMseqs2:  conda install -c conda-forge -c bioconda mmseqs2"
        )
        return _levenshtein_cold_target_split(
            df, seq_id_threshold=seq_id_threshold,
            val_frac=val_frac, test_frac=test_frac,
            seed=seed, cache_dir=cache_dir,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Dataset  [CRIT-2: protein tokenised without truncation]
# ─────────────────────────────────────────────────────────────────────────────
class DTIDataset(Dataset):
    """
    Tokenises (SMILES, protein-sequence) pairs.
    Labels are log1p(Kd [nM]) -- DAVIS lower = tighter binding.

    Molecule (SMILES):
      Truncated at max_mol_len — safe because SMILES for small molecules
      are short (typically < 80 tokens) and the chemistry is fully encoded.

    Protein (amino-acid sequence):
      NOT truncated here.  Variable-length tensors are returned; the
      dti_collate_fn pads them to the batch maximum.  The
      ProteinSlidingWindowEncoder handles sequences that exceed ESM-2's
      1022-residue native limit.
    """

    def __init__(
        self,
        df:          pd.DataFrame,
        mol_tok:     AutoTokenizer,
        prot_tok:    AutoTokenizer,
        max_mol_len: int = 128,
    ) -> None:
        self.df          = df.reset_index(drop=True)
        self.mol_tok     = mol_tok
        self.prot_tok    = prot_tok
        self.max_mol_len = max_mol_len
        self.labels      = torch.tensor(
            np.log1p(df["Y"].values.astype(np.float32)), dtype=torch.float32
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]

        # Molecule: safe to truncate — SMILES are always short
        mol_enc = self.mol_tok(
            str(row["Drug"]),
            max_length=self.max_mol_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Protein: NO truncation / NO padding — handled by dti_collate_fn
        # and ProteinSlidingWindowEncoder respectively.
        prot_enc = self.prot_tok(
            str(row["Target"]),
            return_tensors="pt",
        )

        return {
            "mol_input_ids":       mol_enc["input_ids"].squeeze(0),
            "mol_attention_mask":  mol_enc["attention_mask"].squeeze(0),
            "prot_input_ids":      prot_enc["input_ids"].squeeze(0),      # variable length
            "prot_attention_mask": prot_enc["attention_mask"].squeeze(0),
            "label":               self.labels[idx],
        }


def dti_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Custom collate for DTIDataset.

    Molecule tensors are already padded to a fixed length (max_mol_len) by the
    tokenizer, so they can be stacked directly.

    Protein tensors have variable lengths (no truncation in the dataset).
    This function pads them to the maximum length within the batch.  This is
    more memory-efficient than a global fixed-length pad (e.g., 512 for a 200-AA
    protein) and avoids any truncation of long sequences.
    """
    mol_input_ids      = torch.stack([b["mol_input_ids"]       for b in batch])
    mol_attention_mask = torch.stack([b["mol_attention_mask"]   for b in batch])
    labels             = torch.stack([b["label"]                for b in batch])

    # Dynamic padding for proteins
    max_prot = max(b["prot_input_ids"].shape[0] for b in batch)
    B = len(batch)
    prot_input_ids      = torch.zeros(B, max_prot, dtype=torch.long)
    prot_attention_mask = torch.zeros(B, max_prot, dtype=torch.long)
    for i, b in enumerate(batch):
        L = b["prot_input_ids"].shape[0]
        prot_input_ids[i, :L]      = b["prot_input_ids"]
        prot_attention_mask[i, :L] = b["prot_attention_mask"]

    return {
        "mol_input_ids":       mol_input_ids,
        "mol_attention_mask":  mol_attention_mask,
        "prot_input_ids":      prot_input_ids,
        "prot_attention_mask": prot_attention_mask,
        "label":               labels,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Protein Sliding-Window Encoder  [CRIT-2]
# ─────────────────────────────────────────────────────────────────────────────
class ProteinSlidingWindowEncoder(nn.Module):
    """
    Wraps an ESM-2 backbone to handle protein sequences of arbitrary length.

    Strategy (sliding window with mean-aggregation of overlapping residues):
    -------------------------------------------------------------------------
    For sequences with <= chunk_size body tokens: standard single-pass ESM-2
    forward (no overhead).

    For longer sequences:
      1. Extract the body tokens (positions 1..seq_len-2, i.e., exclude the
         BOS and EOS special tokens added by the tokenizer).
      2. Split the body into overlapping windows of `chunk_size` with `stride`
         step.  Each window is wrapped with fresh BOS/EOS tokens.
      3. Run each chunk independently through ESM-2.
      4. Per-residue hidden states are accumulated and mean-averaged across
         all chunks that covered that position.
      5. The BOS embedding is taken from the first chunk only; EOS from last.

    Why sliding window over global mean-pool:
      Global mean-pool loses token-level resolution needed for the
      cross-attention binding-site proxy (L23 link).  Sliding window preserves
      per-residue embeddings for the full sequence while staying within ESM-2's
      positional embedding limit.

    Gradient note:
      All chunks for sequences within chunk_size receive full gradients.
      Chunks in the sliding-window path (rare for DAVIS — most kinases < 800 AA)
      run under torch.no_grad() to avoid accumulating O(n_chunks) computation
      graphs in memory during training; the projection layers that follow still
      receive gradients through the final prot_h tensor.
    """

    def __init__(
        self,
        encoder:    nn.Module,
        chunk_size: int = 1020,
        stride:     int = 512,
    ) -> None:
        super().__init__()
        self.encoder    = encoder
        self.chunk_size = chunk_size  # body tokens (excludes BOS/EOS)
        self.stride     = stride

    @property
    def config(self):
        """Expose encoder config so callers can query hidden_size etc."""
        return self.encoder.config

    def forward(
        self,
        input_ids:      torch.Tensor,   # (B, L) padded to batch max
        attention_mask: torch.Tensor,   # (B, L)
    ) -> torch.Tensor:                  # (B, L, hidden_size)

        B, L       = input_ids.shape
        hidden_dim = self.encoder.config.hidden_size
        device     = input_ids.device
        dtype      = next(self.encoder.parameters()).dtype

        actual_lens = attention_mask.sum(dim=1).long()   # true token counts per sample
        max_actual  = actual_lens.max().item()

        # ── Fast path (all sequences fit in one chunk) ────────────────────
        if max_actual <= self.chunk_size + 2:            # +2 for BOS + EOS
            return self.encoder(
                input_ids=input_ids, attention_mask=attention_mask
            ).last_hidden_state

        # ── Slow path: per-sample sliding window ─────────────────────────
        # Accumulate per-residue embeddings; average over overlapping chunks.
        outputs = torch.zeros(B, L, hidden_dim, device=device, dtype=dtype)
        counts  = torch.zeros(B, L,             device=device, dtype=torch.float32)

        for b in range(B):
            seq_len  = actual_lens[b].item()

            if seq_len <= self.chunk_size + 2:
                # This sample fits — single chunk, with full gradient.
                out = self.encoder(
                    input_ids      = input_ids[b:b+1, :seq_len],
                    attention_mask = attention_mask[b:b+1, :seq_len],
                ).last_hidden_state.squeeze(0)          # (seq_len, hidden)
                outputs[b, :seq_len] = out
                counts [b, :seq_len] = 1.0
                continue

            # Long sequence: chunk the body (positions 1..seq_len-2).
            # input_ids layout: [BOS, aa1, aa2, ..., aaN, EOS, PAD, ...]
            body_ids = input_ids[b, 1 : seq_len - 1]   # (N,) where N = seq_len - 2
            bos_id   = input_ids[b, 0:1]
            eos_id   = input_ids[b, seq_len - 1 : seq_len]
            N        = body_ids.shape[0]

            start = 0
            while True:
                end        = min(start + self.chunk_size, N)
                chunk_ids  = torch.cat([bos_id, body_ids[start:end], eos_id])
                chunk_mask = torch.ones(chunk_ids.shape[0], dtype=torch.long, device=device)

                # Run encoder (no gradient for long-seq chunks to save memory)
                with torch.no_grad():
                    chunk_out = self.encoder(
                        input_ids      = chunk_ids.unsqueeze(0),
                        attention_mask = chunk_mask.unsqueeze(0),
                    ).last_hidden_state.squeeze(0)      # (chunk_len+2, hidden)

                # Map chunk positions back to full-sequence indices.
                #   chunk_out[0]           -> full position 0 (BOS), first chunk only
                #   chunk_out[1 : end-start+1] -> full positions [start+1 : end+1]
                #   chunk_out[-1]          -> full position seq_len-1 (EOS), last chunk only
                if start == 0:
                    outputs[b, 0] += chunk_out[0]
                    counts [b, 0] += 1.0

                chunk_body_len = end - start
                full_s = start + 1
                full_e = end   + 1
                outputs[b, full_s:full_e] += chunk_out[1 : chunk_body_len + 1]
                counts [b, full_s:full_e] += 1.0

                if end == N:
                    outputs[b, seq_len - 1] += chunk_out[-1]
                    counts [b, seq_len - 1] += 1.0
                    break

                start += self.stride

        # Mean-average overlapping positions
        valid            = counts > 0
        outputs[valid]   = outputs[valid] / counts[valid].unsqueeze(-1)

        return outputs


# ─────────────────────────────────────────────────────────────────────────────
# Cross-Attention Layer  [MED-4: NaN fix]
# ─────────────────────────────────────────────────────────────────────────────
class CrossAttentionLayer(nn.Module):
    """
    Multi-Head Cross-Attention.

        Q  = molecule hidden states   (what we interpret)
        K,V= protein  hidden states   (the binding context)

    Attention scores  softmax( QK^T / sqrt(d_k) )  are the interpretability
    hook: each row maps one molecule atom to its affinity over protein residues.

    Returns
    -------
    mol_out      : (B, L_mol,  d_model)  updated molecule representations
    attn_weights : (B, H, L_mol, L_prot) raw attention probabilities
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k     = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model)

        self.drop    = nn.Dropout(dropout)
        self.norm_ca = nn.LayerNorm(d_model)
        self.norm_ff = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(
        self,
        mol_h:    torch.Tensor,                   # (B, L_mol,  d_model)
        prot_h:   torch.Tensor,                   # (B, L_prot, d_model)
        key_mask: Optional[torch.Tensor] = None,  # (B, L_prot) True = pad position
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L_mol, _   = mol_h.shape
        _,  L_prot, _ = prot_h.shape

        Q = self.W_q(mol_h)
        K = self.W_k(prot_h)
        V = self.W_v(prot_h)

        def to_heads(x: torch.Tensor) -> torch.Tensor:
            return x.view(B, -1, self.n_heads, self.d_k).transpose(1, 2)

        Q, K, V = to_heads(Q), to_heads(K), to_heads(V)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if key_mask is not None:
            # [FIX MED-4] Root cause of NaN: if ALL protein positions are masked
            # for a sample, every score in that row becomes -inf, and
            # softmax(-inf, ..., -inf) = NaN.  This cannot arise from valid
            # input (a real protein always has at least one non-pad token), but
            # we guard defensively.
            #
            # Fix: detect all-masked rows; temporarily unmask position 0 for
            # those rows so softmax has at least one finite input.  The output
            # for those positions is near-zero and harmless because the
            # corresponding protein representation is all padding anyway.
            #
            # The old code silenced this with nan_to_num, which hid real data
            # quality bugs.  This approach surfaces them with a warning while
            # keeping training stable.
            all_masked = key_mask.all(dim=-1)          # (B,) True = bad sample
            if all_masked.any():
                log.warning(
                    f"CrossAttentionLayer: {all_masked.sum().item()} sample(s) in this "
                    "batch have ALL protein tokens masked.  This indicates an empty or "
                    "fully-padding protein sequence — check your input data."
                )
                safe_mask = key_mask.clone()
                safe_mask[all_masked, 0] = False        # unmask pos 0 for bad rows only
            else:
                safe_mask = key_mask

            scores = scores.masked_fill(safe_mask[:, None, None, :], float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)        # (B,H,L_mol,L_prot)
        # NaN should not appear here; raise immediately if it does so bugs are
        # caught at the source rather than propagating silently downstream.
        if torch.isnan(attn_weights).any():
            raise RuntimeError(
                "NaN in cross-attention weights after safe masking.  "
                "Please check that your protein sequences are non-empty "
                "and that max_prot_len is large enough."
            )

        attn_out = torch.matmul(self.drop(attn_weights), V)   # (B,H,L_mol,d_k)
        attn_out = (
            attn_out.transpose(1, 2).contiguous().view(B, L_mol, self.d_model)
        )
        attn_out = self.W_o(attn_out)

        mol_h = self.norm_ca(mol_h + self.drop(attn_out))
        mol_h = self.norm_ff(mol_h + self.drop(self.ff(mol_h)))
        return mol_h, attn_weights


# ─────────────────────────────────────────────────────────────────────────────
# Full Cross-Modal DTI Model  [CRIT-2, MED-5]
# ─────────────────────────────────────────────────────────────────────────────
class CrossModalDTI(nn.Module):
    """
    Architecture
    ------------
    [SMILES]  -> ChemBERTa -> proj -> mol_h  (B, L_mol,  d_model)
                                           \\
    [protein] -> ESM-2 (sliding-window) -> proj -> prot_h (B, L_prot, d_model)
                                           /
    CrossAttentionLayer x n_cross_layers
        Q = mol_h    K,V = prot_h
        attn_weights (B,H,L_mol,L_prot) <- binding-site proxy map

    MeanPool(mol_h) || MeanPool(prot_h)  ->  MLP  ->  scalar log1p(Kd)

    Pooling note [MED-5]:
        Both branches now use mean-pooling over their real (non-pad) tokens.
        The previous code used ESM CLS (position 0) for protein but mean-pool
        for molecule.  This inconsistency is problematic because:
          (a) CLS is a single fixed-position token that may not summarise the
              full binding-domain context as well as mean-pool.
          (b) The molecule branch already sees the cross-attention update;
              using a different pooling paradigm for protein made the
              concatenation semantically uneven.
        Mean-pool for both gives a more representative and consistent input
        to the prediction MLP.
    """

    def __init__(
        self,
        mol_model_name:  str   = "seyonec/ChemBERTa-zinc-base-v1",
        prot_model_name: str   = "facebook/esm2_t6_8M_UR50D",
        d_model:         int   = 256,
        n_heads:         int   = 8,
        n_cross_layers:  int   = 2,
        dropout:         float = 0.1,
        cache_dir:       str   = "./hf_cache",
        prot_chunk_size: int   = 1020,
        prot_stride:     int   = 512,
    ) -> None:
        super().__init__()

        log.info(f"Loading molecule encoder : {mol_model_name}")
        self.mol_encoder  = AutoModel.from_pretrained(mol_model_name,  cache_dir=cache_dir)

        log.info(f"Loading protein  encoder : {prot_model_name}")
        _prot_backbone = AutoModel.from_pretrained(prot_model_name, cache_dir=cache_dir)
        # Wrap backbone with sliding-window logic  [CRIT-2]
        self.prot_encoder = ProteinSlidingWindowEncoder(
            encoder    = _prot_backbone,
            chunk_size = prot_chunk_size,
            stride     = prot_stride,
        )

        mol_dim  = self.mol_encoder.config.hidden_size    # ChemBERTa -> 768
        prot_dim = self.prot_encoder.config.hidden_size   # ESM-2 t6  -> 320

        self.mol_proj = nn.Sequential(
            nn.Linear(mol_dim,  d_model), nn.LayerNorm(d_model), nn.Dropout(dropout)
        )
        self.prot_proj = nn.Sequential(
            nn.Linear(prot_dim, d_model), nn.LayerNorm(d_model), nn.Dropout(dropout)
        )

        self.cross_layers = nn.ModuleList([
            CrossAttentionLayer(d_model, n_heads, dropout)
            for _ in range(n_cross_layers)
        ])

        self.predictor = nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model,     d_model // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def _encode(
        self,
        mol_ids:   torch.Tensor, mol_mask:  torch.Tensor,
        prot_ids:  torch.Tensor, prot_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mol_h  = self.mol_proj(
            self.mol_encoder(input_ids=mol_ids,   attention_mask=mol_mask ).last_hidden_state
        )
        # prot_encoder is ProteinSlidingWindowEncoder; returns (B, L, hidden)
        prot_h = self.prot_proj(
            self.prot_encoder(input_ids=prot_ids, attention_mask=prot_mask)
        )
        return mol_h, prot_h

    def forward(
        self,
        mol_input_ids:       torch.Tensor,
        mol_attention_mask:  torch.Tensor,
        prot_input_ids:      torch.Tensor,
        prot_attention_mask: torch.Tensor,
        return_attentions:   bool = False,
    ) -> Dict[str, torch.Tensor]:

        mol_h, prot_h = self._encode(
            mol_input_ids, mol_attention_mask,
            prot_input_ids, prot_attention_mask,
        )

        # True where protein token is padding -> masked to -inf in attention
        key_mask = (prot_attention_mask == 0)

        all_attn: List[torch.Tensor] = []
        for layer in self.cross_layers:
            mol_h, attn_w = layer(mol_h, prot_h, key_mask)
            if return_attentions:
                all_attn.append(attn_w)          # (B, H, L_mol, L_prot)

        # [MED-5] Consistent mean-pooling for both branches
        # Molecule: mean over non-padded tokens (cross-attention updated)
        mol_mask_f  = mol_attention_mask.unsqueeze(-1).float()
        mol_pool    = (mol_h * mol_mask_f).sum(1) / mol_mask_f.sum(1).clamp(min=1e-9)

        # Protein: mean over non-padded tokens (provides sequence-level identity signal)
        # Previously CLS token (position 0) — now consistent with molecule branch.
        prot_mask_f = prot_attention_mask.unsqueeze(-1).float()
        prot_pool   = (prot_h * prot_mask_f).sum(1) / prot_mask_f.sum(1).clamp(min=1e-9)

        pred = self.predictor(torch.cat([mol_pool, prot_pool], dim=-1)).squeeze(-1)

        out: Dict[str, torch.Tensor] = {"prediction": pred}
        if return_attentions and all_attn:
            # Head-averaged attention from the last cross-attention layer
            # (B, L_mol, L_prot) — the binding-site proxy map
            out["attn_weights"]     = all_attn[-1].mean(dim=1)
            out["all_attn_weights"] = all_attn
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────
def concordance_index(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    DeepDTA-style Concordance Index (CI).
    Vectorised NumPy -- O(n^2) memory, fine for DAVIS test sets (<= ~5k pairs).
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    diff_true = y_true[:, None] - y_true[None, :]
    diff_pred = y_pred[:, None] - y_pred[None, :]

    ordered    = diff_true > 0
    total      = ordered.sum()
    if total == 0:
        return 0.0

    concordant = ((diff_pred > 0)  & ordered).sum()
    ties_pred  = ((diff_pred == 0) & ordered).sum()
    return float(concordant + 0.5 * ties_pred) / float(total)


# ─────────────────────────────────────────────────────────────────────────────
# Checkpointing  (atomic rename -> safe mid-write preemption)
# ─────────────────────────────────────────────────────────────────────────────
def save_checkpoint(state: dict, path: str) -> None:
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)
    log.info(f"Checkpoint -> {path}")


def load_checkpoint(
    path:      str,
    model:     nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler=None,
) -> Tuple[int, float, list]:
    log.info(f"Resuming from {path}")
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    if scheduler and "scheduler_state" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state"])
    return (
        ckpt.get("epoch", 0),
        ckpt.get("best_val_mse", float("inf")),
        ckpt.get("history", []),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Saliency Map  (Input x Gradient)  [MED-5: prot_pool updated]
# ─────────────────────────────────────────────────────────────────────────────
def compute_saliency(
    model:  CrossModalDTI,
    batch:  Dict[str, torch.Tensor],
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Input x Gradient saliency for a single (molecule, protein) pair.

    Returns
    -------
    mol_sal  : (L_mol,)         importance score per SMILES token
    prot_sal : (L_prot,)        importance score per protein residue
    attn_map : (L_mol, L_prot)  head-averaged cross-attention from last layer
                                 -- proxy for atom <-> residue binding alignment

    Method
    ------
    Saliency[t] = || grad(pred, h_t) * h_t ||_2
    computed on the projected hidden states (post mol_proj / prot_proj).
    Both the mean-pool path and the cross-attention K,V path contribute
    gradient signal to prot_sal, giving a holistic residue importance score.
    """
    model.eval()

    mol_ids   = batch["mol_input_ids"].to(device)
    mol_mask  = batch["mol_attention_mask"].to(device)
    prot_ids  = batch["prot_input_ids"].to(device)
    prot_mask = batch["prot_attention_mask"].to(device)

    with torch.enable_grad():
        model.zero_grad()

        mol_enc_out  = model.mol_encoder(input_ids=mol_ids,   attention_mask=mol_mask)
        # Use prot_encoder directly (ProteinSlidingWindowEncoder)
        prot_raw     = model.prot_encoder(input_ids=prot_ids, attention_mask=prot_mask)

        mol_h  = model.mol_proj(mol_enc_out.last_hidden_state)  # (1, L_mol,  d)
        prot_h = model.prot_proj(prot_raw)                       # (1, L_prot, d)

        mol_h.retain_grad()
        prot_h.retain_grad()

        key_mask = (prot_mask == 0)
        all_attn: List[torch.Tensor] = []
        cur_mol = mol_h
        for layer in model.cross_layers:
            cur_mol, aw = layer(cur_mol, prot_h, key_mask)
            all_attn.append(aw)

        # [MED-5] Match the pooling in forward() — both mean-pool
        mol_mask_f  = mol_mask.unsqueeze(-1).float()
        mol_pool    = (cur_mol * mol_mask_f).sum(1) / mol_mask_f.sum(1).clamp(min=1e-9)

        prot_mask_f = prot_mask.unsqueeze(-1).float()
        prot_pool   = (prot_h * prot_mask_f).sum(1) / prot_mask_f.sum(1).clamp(min=1e-9)

        pred = model.predictor(torch.cat([mol_pool, prot_pool], dim=-1)).squeeze(-1)
        pred.sum().backward()

    mol_sal  = (mol_h.grad  * mol_h ).detach().abs().norm(dim=-1).squeeze(0).cpu().numpy()
    prot_sal = (prot_h.grad * prot_h).detach().abs().norm(dim=-1).squeeze(0).cpu().numpy()
    attn_map = all_attn[-1].mean(dim=1).squeeze(0).detach().cpu().numpy()  # (L_mol, L_prot)

    return mol_sal, prot_sal, attn_map


# ─────────────────────────────────────────────────────────────────────────────
# Training & Evaluation
# ─────────────────────────────────────────────────────────────────────────────
def train_one_epoch(
    model:     CrossModalDTI,
    loader:    DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler:    Optional[GradScaler],
    device:    torch.device,
    grad_clip: float,
) -> float:
    model.train()
    running_loss, n = 0.0, 0

    for batch in loader:
        mol_ids   = batch["mol_input_ids"].to(device)
        mol_mask  = batch["mol_attention_mask"].to(device)
        prot_ids  = batch["prot_input_ids"].to(device)
        prot_mask = batch["prot_attention_mask"].to(device)
        labels    = batch["label"].to(device)

        optimizer.zero_grad()
        with autocast(enabled=(scaler is not None)):
            out  = model(mol_ids, mol_mask, prot_ids, prot_mask)
            loss = F.mse_loss(out["prediction"], labels)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        if scheduler is not None:   # None when using plateau (stepped per epoch)
            scheduler.step()
        running_loss += loss.item() * labels.size(0)
        n            += labels.size(0)

    return running_loss / n


@torch.no_grad()
def evaluate(
    model:  CrossModalDTI,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Returns (mse_log, predictions_log, labels_log) — all on log1p(Kd [nM]) scale."""
    model.eval()
    total_loss, n = 0.0, 0
    all_preds, all_labels = [], []

    for batch in loader:
        mol_ids   = batch["mol_input_ids"].to(device)
        mol_mask  = batch["mol_attention_mask"].to(device)
        prot_ids  = batch["prot_input_ids"].to(device)
        prot_mask = batch["prot_attention_mask"].to(device)
        labels    = batch["label"].to(device)

        out  = model(mol_ids, mol_mask, prot_ids, prot_mask)
        loss = F.mse_loss(out["prediction"], labels)

        total_loss  += loss.item() * labels.size(0)
        n           += labels.size(0)
        all_preds.append(out["prediction"].cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    return (
        total_loss / n,
        np.concatenate(all_preds),
        np.concatenate(all_labels),
    )


def report_metrics(
    split_name: str,
    mse_log:    float,
    preds_log:  np.ndarray,
    labels_log: np.ndarray,
) -> Dict:
    """
    [LOW-7] Compute and log CI + MSE in both log1p scale (training objective)
    and nanomolar scale (human-interpretable).  Returns a dict for JSON export.
    """
    ci = concordance_index(labels_log, preds_log)

    # Inverse log1p transform: expm1(x) = e^x - 1
    preds_nM  = np.expm1(preds_log)
    labels_nM = np.expm1(labels_log)
    mse_nM    = float(np.mean((preds_nM - labels_nM) ** 2))

    log.info(
        f"[{split_name}]  "
        f"MSE(log1p) = {mse_log:.4f}  |  "
        f"MSE(nM)    = {mse_nM:.1f}   |  "
        f"CI         = {ci:.4f}"
    )
    return {
        "mse_log1p":  float(mse_log),
        "mse_nM":     mse_nM,
        "ci":         float(ci),
        "n_samples":  int(len(labels_log)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Post-training Analysis  [MED-6, LOW-7]
# ─────────────────────────────────────────────────────────────────────────────
def _top_binding_residues(
    attn_map:  np.ndarray,      # (L_mol, L_prot) head-averaged
    mol_len:   int,             # actual mol token count (incl. BOS/EOS)
    prot_len:  int,             # actual prot token count
    attn_topk: Optional[int],
    top_n:     int = 10,
) -> List[int]:
    """
    [MED-6] Identify top-N protein residues by cross-attention.

    Strategy:
      1. Restrict to body tokens only: exclude CLS (index 0) and EOS (index -1)
         from both molecule and protein dimensions.  Special tokens carry
         sequence-global information, not atom-level pharmacophore signal, and
         including them dilutes binding-site scores.
      2. Optional top-K molecule token filtering: if attn_topk is set, retain
         only the K molecule body tokens with the highest max-attention to any
         residue.  This focuses the scoring on the pharmacophore atoms.
      3. Max-pool over the (filtered) molecule token dimension to produce a
         per-residue score.  Max-pool is preferred over mean-pool because only
         a small subset of atoms typically drive binding; averaging suppresses
         their signal.
    """
    # Body = exclude BOS (0) and EOS (-1) from both dimensions
    mol_body  = attn_map[1 : mol_len  - 1, 1 : prot_len - 1]  # (L_mol_body, L_prot_body)

    if mol_body.shape[0] == 0 or mol_body.shape[1] == 0:
        return []

    # Optional top-K mol token selection
    if attn_topk is not None and attn_topk < mol_body.shape[0]:
        mol_importance = mol_body.max(axis=1)                   # (L_mol_body,) peak attention per atom
        topk_idx       = np.argsort(mol_importance)[-attn_topk:]
        mol_body       = mol_body[topk_idx, :]                  # (K, L_prot_body)

    # Max-pool over molecule axis -> per-residue binding score
    residue_scores = mol_body.max(axis=0)                       # (L_prot_body,)

    # Return top_n residue indices (offset by 1 to match full-sequence positions)
    top_idx_body = np.argsort(residue_scores)[-top_n:]
    return (top_idx_body + 1).tolist()                          # +1 to re-add BOS offset


def run_analysis(
    model:    CrossModalDTI,
    test_df:  pd.DataFrame,
    mol_tok:  AutoTokenizer,
    prot_tok: AutoTokenizer,
    cfg:      argparse.Namespace,
    device:   torch.device,
) -> Dict:
    """
    1. Compute CI and MSE (log and nM scale) on the cold-target test set.
    2. Generate saliency maps for cfg.saliency_n sampled pairs.
    3. Produce an attention-alignment summary using max-pool / top-K scoring.
    All outputs saved under cfg.output_dir.
    """
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Cold-target evaluation
    test_ds  = DTIDataset(test_df, mol_tok, prot_tok, cfg.max_mol_len)
    test_ldr = DataLoader(
        test_ds, batch_size=cfg.batch_size,
        num_workers=cfg.num_workers, pin_memory=True,
        collate_fn=dti_collate_fn,
    )

    log.info("Evaluating on cold-target test set ...")
    mse_log, preds_log, labels_log = evaluate(model, test_ldr, device)
    metrics = report_metrics("Cold-Target Test", mse_log, preds_log, labels_log)
    metrics["n_test_proteins"] = int(test_df["Target"].nunique())
    metrics["label_scale"]     = "log1p(Kd [nM]) for training; nM for reporting"
    (out_dir / "cold_target_metrics.json").write_text(json.dumps(metrics, indent=2))
    log.info(f"  Metrics -> {out_dir}/cold_target_metrics.json")

    # 2. Saliency maps
    rng     = np.random.default_rng(cfg.seed)
    n_samp  = min(cfg.saliency_n, len(test_df))
    indices = rng.choice(len(test_df), n_samp, replace=False)

    log.info(f"Computing saliency maps for {n_samp} samples ...")
    saliency_records = []

    for idx in indices:
        row    = test_df.iloc[int(idx)]
        single = test_df.iloc[[int(idx)]].reset_index(drop=True)
        ds     = DTIDataset(single, mol_tok, prot_tok, cfg.max_mol_len)
        item   = ds[0]
        batch  = {k: v.unsqueeze(0) for k, v in item.items() if isinstance(v, torch.Tensor)}

        try:
            mol_sal, prot_sal, attn_map = compute_saliency(model, batch, device)
            mol_len  = int(batch["mol_attention_mask"].sum())
            prot_len = int(batch["prot_attention_mask"].sum())

            top_residues = _top_binding_residues(
                attn_map, mol_len, prot_len,
                attn_topk=cfg.attn_topk, top_n=10,
            )

            saliency_records.append({
                "sample_idx":             int(idx),
                "drug_smiles":            str(row["Drug"]),
                "target_prefix":          str(row["Target"])[:30] + "...",
                "true_kd_nM":             float(row["Y"]),
                "true_kd_log1p":          float(np.log1p(row["Y"])),
                "mol_saliency":           mol_sal[:mol_len].tolist(),
                "prot_saliency":          prot_sal[:prot_len].tolist(),
                "attn_map_shape":         [mol_len, prot_len],
                "top10_binding_residues": top_residues,
                "attn_topk_used":         cfg.attn_topk,
                "interpretation": (
                    "mol_saliency[i]: input*grad importance of SMILES token i. "
                    "prot_saliency[j]: input*grad importance of residue j. "
                    "top10_binding_residues: residues with highest max cross-attention "
                    "score from pharmacophore atoms (cross-modal binding-site proxy, "
                    "body tokens only, CLS/EOS excluded)."
                ),
            })
        except Exception as exc:
            log.warning(f"  Saliency failed for sample {idx}: {exc}")

    (out_dir / "saliency_maps.json").write_text(json.dumps(saliency_records, indent=2))
    log.info(
        f"  Saliency maps -> {out_dir}/saliency_maps.json  "
        f"({len(saliency_records)} samples)"
    )

    # 3. Attention alignment summary
    alignment_summary = [
        {
            "smiles_prefix":          r["drug_smiles"][:40],
            "protein_prefix":         r["target_prefix"],
            "top10_binding_residues": r["top10_binding_residues"],
            "aggregation_method":     "max-pool over mol body tokens" + (
                f" (top-{cfg.attn_topk} atoms)" if cfg.attn_topk else " (all atoms)"
            ),
            "note": (
                "Residue positions in the protein sequence receiving the highest "
                "max cross-attention weight from pharmacophore atoms.  "
                "For kinases these should cluster near the ATP-binding pocket; "
                "cross-validate against PDB structures or run validate_binding_sites.py."
            ),
        }
        for r in saliency_records[:10]
    ]
    (out_dir / "attention_alignment_summary.json").write_text(
        json.dumps(alignment_summary, indent=2)
    )
    log.info(f"  Alignment summary -> {out_dir}/attention_alignment_summary.json")
    log.info("Analysis complete.")
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# SLURM preemption handler
# ─────────────────────────────────────────────────────────────────────────────
class PreemptionHandler:
    """
    Registers a SIGUSR1 handler.  SLURM sends SIGUSR1 via
      #SBATCH --signal=SIGUSR1@90
    90 seconds before wall-time is reached.  On receipt the handler saves
    the latest checkpoint and exits cleanly so --requeue restarts the job
    and --resume picks up exactly where it left off.
    """

    def __init__(self) -> None:
        self._state: Optional[dict] = None
        self._path:  Optional[str]  = None
        # SIGUSR1 is POSIX-only; not available on Windows.
        if hasattr(signal, "SIGUSR1"):
            signal.signal(signal.SIGUSR1, self._handle)
            log.info("SIGUSR1 preemption handler registered.")
        else:
            log.info("SIGUSR1 not available on this platform (Windows) — "
                     "preemption handler skipped.")

    def register_state(self, state: dict, path: str) -> None:
        """Call once per epoch to keep the handler state current."""
        self._state = state
        self._path  = path

    def _handle(self, signum, frame) -> None:
        log.warning("SIGUSR1 received -- preemption imminent. Saving checkpoint ...")
        if self._state and self._path:
            save_checkpoint(self._state, self._path)
            log.info("Checkpoint saved. Exiting for SLURM requeue.")
        sys.exit(0)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    cfg = build_parser().parse_args()

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device : {device}")
    if device.type == "cuda":
        log.info(f"GPU    : {torch.cuda.get_device_name(0)}")

    for d in (cfg.data_dir, cfg.cache_dir, cfg.output_dir):
        os.makedirs(d, exist_ok=True)

    # Load DAVIS
    log.info("Loading DAVIS from PyTDC ...")
    from tdc.multi_pred import DTI as TDC_DTI
    davis = TDC_DTI(name="DAVIS", path=cfg.data_dir)
    df    = davis.get_data()
    log.info(f"Total pairs : {len(df)}  columns : {df.columns.tolist()}")

    # Cold-target split  [CRIT-1]
    train_df, val_df, test_df = cold_target_split(
        df,
        seq_id_threshold=cfg.seq_id_threshold,
        val_frac=cfg.val_frac,
        test_frac=cfg.test_frac,
        seed=cfg.seed,
        cache_dir=cfg.data_dir,
    )

    # Tokenizers
    log.info("Loading tokenizers ...")
    mol_tok  = AutoTokenizer.from_pretrained(cfg.mol_model,  cache_dir=cfg.cache_dir)
    prot_tok = AutoTokenizer.from_pretrained(cfg.prot_model, cache_dir=cfg.cache_dir)

    # DataLoaders  [CRIT-2: dti_collate_fn for dynamic protein padding]
    def make_ldr(df_: pd.DataFrame, shuffle: bool) -> DataLoader:
        ds = DTIDataset(df_, mol_tok, prot_tok, cfg.max_mol_len)
        return DataLoader(
            ds,
            batch_size=cfg.batch_size,
            shuffle=shuffle,
            num_workers=cfg.num_workers,
            pin_memory=(device.type == "cuda"),
            collate_fn=dti_collate_fn,
        )

    train_ldr = make_ldr(train_df, shuffle=True)
    val_ldr   = make_ldr(val_df,   shuffle=False)

    # Model
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

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Trainable parameters : {n_params:,}")

    # Optimizer + scheduler
    optimizer   = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    total_steps  = len(train_ldr) * cfg.epochs
    warmup_steps = int(total_steps * cfg.warmup_ratio)

    if cfg.scheduler == "cosine":
        from transformers import get_cosine_schedule_with_warmup
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
        _step_scheduler_each_batch = True
    elif cfg.scheduler == "linear":
        from transformers import get_linear_schedule_with_warmup
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
        _step_scheduler_each_batch = True
    elif cfg.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min",
            factor=cfg.plateau_factor,
            patience=cfg.plateau_patience,
        )
        _step_scheduler_each_batch = False  # stepped once per epoch on val MSE
    else:
        raise ValueError(f"Unknown scheduler: {cfg.scheduler}")

    log.info(f"Scheduler : {cfg.scheduler}")
    scaler = GradScaler() if (cfg.fp16 and device.type == "cuda") else None

    # Resume
    start_epoch  = 0
    best_val_mse = float("inf")
    history: list = []

    epochs_no_improve = 0
    if cfg.resume and os.path.exists(cfg.checkpoint_path):
        start_epoch, best_val_mse, history = load_checkpoint(
            cfg.checkpoint_path, model, optimizer, scheduler
        )
        # Restore early-stopping counter so patience is not reset on requeue
        ckpt_peek = torch.load(cfg.checkpoint_path, map_location="cpu")
        epochs_no_improve = ckpt_peek.get("epochs_no_improve", 0)
        log.info(
            f"Resumed at epoch {start_epoch}, best val MSE {best_val_mse:.4f}, "
            f"no-improve streak {epochs_no_improve}"
        )

    # SLURM preemption handler
    preempt = PreemptionHandler()

    # Training loop
    best_path = os.path.join(cfg.output_dir, "best_model.pt")
    log.info(f"Training epochs {start_epoch + 1} -> {cfg.epochs} ...")
    if cfg.patience:
        log.info(f"Early stopping patience : {cfg.patience} epochs")

    for epoch in range(start_epoch, cfg.epochs):
        t0         = time.time()
        train_loss = train_one_epoch(
            model, train_ldr, optimizer,
            scheduler if _step_scheduler_each_batch else None,
            scaler, device, cfg.grad_clip,
        )
        val_mse_log, val_preds, val_labels = evaluate(model, val_ldr, device)
        val_ci  = concordance_index(val_labels, val_preds)
        elapsed = time.time() - t0

        # Step plateau scheduler once per epoch on val MSE
        if not _step_scheduler_each_batch:
            scheduler.step(val_mse_log)

        # [LOW-7] Report both log-scale (training objective) and nM (human-readable)
        val_mse_nM = float(np.mean((np.expm1(val_preds) - np.expm1(val_labels)) ** 2))
        current_lr = optimizer.param_groups[0]["lr"]
        log.info(
            f"Epoch {epoch+1:3d}/{cfg.epochs}  |  "
            f"train MSE(log) {train_loss:.4f}  |  "
            f"val MSE(log) {val_mse_log:.4f}  |  "
            f"val MSE(nM) {val_mse_nM:.1f}  |  "
            f"val CI {val_ci:.4f}  |  "
            f"lr {current_lr:.2e}  |  "
            f"{elapsed:.0f}s"
        )

        history.append(dict(
            epoch        = epoch + 1,
            train_mse    = train_loss,
            val_mse_log  = val_mse_log,
            val_mse_nM   = val_mse_nM,
            val_ci       = val_ci,
            lr           = current_lr,
        ))

        ckpt_state = dict(
            epoch              = epoch + 1,
            model_state        = model.state_dict(),
            optimizer_state    = optimizer.state_dict(),
            scheduler_state    = scheduler.state_dict(),
            best_val_mse       = best_val_mse,
            epochs_no_improve  = epochs_no_improve,
            history            = history,
            config             = vars(cfg),
        )
        save_checkpoint(ckpt_state, cfg.checkpoint_path)
        preempt.register_state(ckpt_state, cfg.checkpoint_path)

        if val_mse_log < best_val_mse:
            best_val_mse      = val_mse_log
            epochs_no_improve = 0
            save_checkpoint(ckpt_state, best_path)
            log.info(f"  New best  (val MSE(log) {val_mse_log:.4f})  -> {best_path}")
        else:
            epochs_no_improve += 1

        if cfg.patience and epochs_no_improve >= cfg.patience:
            log.info(
                f"Early stopping triggered: no improvement for "
                f"{epochs_no_improve} consecutive epochs."
            )
            break

    # Final cold-target test evaluation
    log.info("Loading best model for final cold-target test evaluation ...")
    if os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])

    test_ldr = make_ldr(test_df, shuffle=False)
    test_mse_log, test_preds, test_labels = evaluate(model, test_ldr, device)
    test_metrics = report_metrics("Cold-Target Test", test_mse_log, test_preds, test_labels)

    final = dict(
        cold_target    = test_metrics,
        best_val_mse   = float(best_val_mse),
        n_test_pairs   = int(len(test_df)),
        n_test_proteins= int(test_df["Target"].nunique()),
        history        = history,
    )
    (Path(cfg.output_dir) / "final_metrics.json").write_text(json.dumps(final, indent=2))

    if cfg.run_analysis:
        run_analysis(model, test_df, mol_tok, prot_tok, cfg, device)

    log.info("Done.")


if __name__ == "__main__":
    main()
