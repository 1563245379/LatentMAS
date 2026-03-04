#!/usr/bin/env python3
"""
t-SNE Visualization of Alignment Quality for Three LatentMAS Methods
====================================================================

Visualizes alignment quality at two levels:

**Row 1 – Vocabulary-level alignment**:
  Samples vocabulary tokens and compares W_out[t] @ W_align vs W_in[t].

**Row 2 – Inference-level alignment** (real token test):
  Runs model A on a test text to obtain actual last-layer hidden states,
  extracts predicted tokens, looks up their input embeddings in the target
  model, and compares  h_t @ W_align  vs  W_in_B[predicted_token_t].

Methods:
1. **latent_mas** (same model):  Self-alignment
2. **latent_mas_hybrid** (same family, different size):  Full-vocab cross-model
3. **latent_mas_plus** (different families):  Intersection-vocab cross-model

Usage
-----
    python scripts/tsne_alignment.py \
        --model_a Qwen/Qwen3-4B \
        --model_b_hybrid Qwen/Qwen3-8B \
        --model_b_plus meta-llama/Llama-3.1-8B \
        --n_tokens 1000 \
        --n_samples 200 \
        --perplexity 30 \
        --seed 42 \
        --output figures/tsne_alignment.png
"""

import argparse
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset
from sklearn.manifold import TSNE

# ── add project root to path ─────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from transformers import AutoModelForCausalLM, AutoTokenizer

warnings.filterwarnings("ignore", category=FutureWarning)


# =========================================================================
# Alignment matrix builders (mirroring the three methods)
# =========================================================================

def build_self_alignment(model, lambda_reg: float = 1e-5):
    """
    latent_mas: same-model self-alignment.
    W = (W_out^T W_out + λI)^{-1} W_out^T W_in
    """
    W_out = model.get_output_embeddings().weight.detach().float()  # [V, d]
    W_in = model.get_input_embeddings().weight.detach().float()   # [V, d]
    d = W_out.shape[1]
    gram = W_out.T @ W_out + lambda_reg * torch.eye(d, device=W_out.device)
    rhs = W_out.T @ W_in
    W_align = torch.linalg.solve(gram, rhs)  # [d, d]
    target_norm = W_in.norm(dim=1).mean()
    return W_align, target_norm


def build_hybrid_alignment(model_a, model_b, lambda_reg: float = 1e-5):
    """
    latent_mas_hybrid: full-vocab cross-model alignment (same-family models).
    W_cross = (W_out_A^T W_out_A + λI)^{-1} W_out_A^T W_in_B
    Uses min(vocab_A, vocab_B) tokens when vocab sizes differ.
    """
    W_out_A = model_a.get_output_embeddings().weight.detach().float()  # [V_A, d_A]
    W_in_B = model_b.get_input_embeddings().weight.detach().float()   # [V_B, d_B]
    V_A, d_A = W_out_A.shape
    V_B, d_B = W_in_B.shape
    min_V = min(V_A, V_B)
    if V_A != V_B:
        W_out_A = W_out_A[:min_V]
        W_in_B = W_in_B[:min_V]
        print(f"  [hybrid] Vocab size mismatch ({V_A} vs {V_B}), using first {min_V}")
    gram = W_out_A.T @ W_out_A + lambda_reg * torch.eye(d_A, device=W_out_A.device)
    rhs = W_out_A.T @ W_in_B
    W_align = torch.linalg.solve(gram, rhs)  # [d_A, d_B]
    target_norm = W_in_B.norm(dim=1).mean()
    return W_align, target_norm, min_V


def build_plus_alignment(model_a, model_b, tokenizer_a, tokenizer_b,
                          lambda_reg: float = 1e-4):
    """
    latent_mas_plus: intersection-vocab cross-model alignment (different families).
    Only shared tokens are used for alignment.
    """
    vocab_a = tokenizer_a.get_vocab()
    vocab_b = tokenizer_b.get_vocab()
    shared = sorted(set(vocab_a) & set(vocab_b))
    idx_a = [vocab_a[t] for t in shared]
    idx_b = [vocab_b[t] for t in shared]
    n_shared = len(shared)
    print(f"  [plus] Vocab intersection: |V^A|={len(vocab_a)}, |V^B|={len(vocab_b)}, "
          f"|V^I|={n_shared} ({100*n_shared/max(len(vocab_a),len(vocab_b)):.1f}%)")

    W_out_A = model_a.get_output_embeddings().weight.detach().float()
    W_in_B = model_b.get_input_embeddings().weight.detach().float()

    idx_a_t = torch.tensor(idx_a, dtype=torch.long, device=W_out_A.device)
    idx_b_t = torch.tensor(idx_b, dtype=torch.long, device=W_in_B.device)

    W_out_I = W_out_A[idx_a_t]    # [N_I, d_A]
    W_in_I = W_in_B[idx_b_t]      # [N_I, d_B]
    d_A = W_out_I.shape[1]

    gram = W_out_I.T @ W_out_I + lambda_reg * torch.eye(d_A, device=W_out_I.device)
    rhs = W_out_I.T @ W_in_I
    W_align = torch.linalg.solve(gram, rhs)  # [d_A, d_B]
    target_norm = W_in_I.norm(dim=1).mean()
    return W_align, target_norm, idx_a, idx_b


def apply_alignment(hidden: torch.Tensor, W_align: torch.Tensor,
                     target_norm: torch.Tensor) -> torch.Tensor:
    """Project hidden through alignment matrix and normalize."""
    aligned = (hidden.float() @ W_align)
    norms = aligned.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    aligned = aligned * (target_norm / norms)
    return aligned


# =========================================================================
# Dataset & Inference helpers
# =========================================================================

def load_gsm8k_questions(n_samples: int, seed: int = 42):
    """
    Load n_samples questions from GSM8K test set.
    Returns list of question strings.
    """
    ds = load_dataset("gsm8k", "main", split="test")
    rng = np.random.RandomState(seed)
    indices = rng.choice(len(ds), size=min(n_samples, len(ds)), replace=False)
    questions = [ds[int(i)]["question"].strip() for i in indices]
    return questions


@torch.no_grad()
def get_inference_hidden_states(model, tokenizer, text, device="cpu"):
    """
    Run model forward pass on text and return:
      - last_hidden: [seq_len, hidden_dim]  last-layer hidden states
      - predicted_ids: [seq_len]  predicted next-token IDs (argmax of logits)
      - input_ids: [seq_len]  the input token IDs
    """
    inputs = tokenizer(text, return_tensors="pt").to(device)
    outputs = model(**inputs, output_hidden_states=True)
    last_hidden = outputs.hidden_states[-1][0].cpu().float()   # [seq_len, d]
    logits = outputs.logits[0].cpu().float()                   # [seq_len, V]
    predicted_ids = logits.argmax(dim=-1)                       # [seq_len]
    return last_hidden, predicted_ids, inputs["input_ids"][0].cpu()


@torch.no_grad()
def collect_inference_vectors_gsm8k(model, tokenizer, questions, device="cpu"):
    """
    Run model on each GSM8K question and concatenate hidden states.
    Returns:
      all_hidden:       [total_tokens, d]  last-layer hidden states
      all_predicted_ids: [total_tokens]    predicted next-token IDs
    """
    all_hidden, all_pred = [], []
    for i, q in enumerate(questions):
        h, p, _ = get_inference_hidden_states(model, tokenizer, q, device)
        all_hidden.append(h)
        all_pred.append(p)
        if (i + 1) % 10 == 0 or i == len(questions) - 1:
            total = sum(x.shape[0] for x in all_hidden)
            print(f"    Processed {i+1}/{len(questions)} questions, "
                  f"{total} tokens so far")
    return torch.cat(all_hidden, dim=0), torch.cat(all_pred, dim=0)


def map_tokens_across_tokenizers(predicted_ids, tok_src, tok_tgt):
    """
    Map predicted token IDs from source tokenizer to target tokenizer
    via token string matching (for different-family models).

    Returns:
      positions: list of sequence positions where mapping succeeded
      tgt_ids:   corresponding token IDs in the target vocabulary
    """
    vocab_src_inv = {v: k for k, v in tok_src.get_vocab().items()}
    vocab_tgt = tok_tgt.get_vocab()
    positions, tgt_ids = [], []
    for pos, src_id in enumerate(predicted_ids.tolist()):
        token_str = vocab_src_inv.get(src_id)
        if token_str and token_str in vocab_tgt:
            positions.append(pos)
            tgt_ids.append(vocab_tgt[token_str])
    return positions, tgt_ids


# =========================================================================
# Model loading helper
# =========================================================================

def load_model_and_tokenizer(model_name: str, device: str = "cpu"):
    print(f"Loading model: {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,   # need float32 for linalg.solve
        device_map=device,
    )
    model.eval()
    return model, tokenizer


# =========================================================================
# Sampling & projection
# =========================================================================

def sample_token_vectors(W_out_src, W_in_tgt, W_align, target_norm,
                          sample_idx_src, sample_idx_tgt, n_tokens, rng):
    """
    Sample n_tokens shared indices and return:
      source  = W_out_src[idx]
      aligned = source @ W_align  (normalized)
      target  = W_in_tgt[idx]
    """
    total = len(sample_idx_src)
    chosen = rng.choice(total, size=min(n_tokens, total), replace=False)
    src_vecs = W_out_src[torch.tensor(sample_idx_src, dtype=torch.long)[chosen]]
    tgt_vecs = W_in_tgt[torch.tensor(sample_idx_tgt, dtype=torch.long)[chosen]]
    aligned_vecs = apply_alignment(src_vecs, W_align, target_norm)
    return src_vecs.numpy(), aligned_vecs.numpy(), tgt_vecs.numpy()


# =========================================================================
# t-SNE + Plotting
# =========================================================================

def run_tsne(data_dict, perplexity, seed):
    """
    data_dict: {label: ndarray of shape [N, D]}
    Runs t-SNE on the concatenation, returns dict of embedded 2D arrays.
    """
    labels_list = list(data_dict.keys())
    arrays = [data_dict[k] for k in labels_list]
    sizes = [a.shape[0] for a in arrays]
    combined = np.concatenate(arrays, axis=0)

    tsne = TSNE(
        n_components=2,
        perplexity=min(perplexity, max(2, combined.shape[0] // 4)),
        random_state=seed,
        init="pca",
        learning_rate="auto",
        max_iter=1000,
    )
    embedded = tsne.fit_transform(combined)

    # split back
    parts = {}
    offset = 0
    for label, sz in zip(labels_list, sizes):
        parts[label] = embedded[offset: offset + sz]
        offset += sz

    return parts


def plot_single_method(ax, parts, title, colors, markers):
    """Plot one subplot for a single method."""
    for (label, pts), c, m in zip(parts.items(), colors, markers):
        ax.scatter(pts[:, 0], pts[:, 1], c=c, marker=m, s=12, alpha=0.55,
                   label=label, edgecolors="none")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(fontsize=7, loc="upper right", framealpha=0.8)
    ax.set_xticks([])
    ax.set_yticks([])


def compute_alignment_score(aligned: np.ndarray, target: np.ndarray) -> float:
    """Mean cosine similarity between aligned and target."""
    dot = np.sum(aligned * target, axis=1)
    norm_a = np.linalg.norm(aligned, axis=1) + 1e-8
    norm_t = np.linalg.norm(target, axis=1) + 1e-8
    return float(np.mean(dot / (norm_a * norm_t)))


# =========================================================================
# Main
# =========================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="t-SNE alignment visualization for LatentMAS methods")
    p.add_argument("--model_a", type=str, required=True,
                   help="Base model (model A in all methods)")
    p.add_argument("--model_b_hybrid", type=str, default=None,
                   help="Target model for latent_mas_hybrid (same family, different size)")
    p.add_argument("--model_b_plus", type=str, default=None,
                   help="Target model for latent_mas_plus (different family)")
    p.add_argument("--n_tokens", type=int, default=1000,
                   help="Number of vocabulary tokens to sample for vocab-level test")
    p.add_argument("--n_samples", type=int, default=200,
                   help="Number of GSM8K questions to sample for inference-level test")
    p.add_argument("--perplexity", type=float, default=30,
                   help="t-SNE perplexity")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lambda_reg", type=float, default=1e-5,
                   help="Ridge regularization for self/hybrid alignment")
    p.add_argument("--lambda_plus", type=float, default=1e-4,
                   help="Ridge regularization for intersection alignment")
    p.add_argument("--device", type=str, default="cuda",
                   help="Device for computation")
    p.add_argument("--output", type=str, default="figures/tsne_alignment.png",
                   help="Output image path")
    return p.parse_args()


def main():
    args = parse_args()
    rng = np.random.RandomState(args.seed)
    torch.manual_seed(args.seed)

    # ── Determine which methods to visualize ────────────────────────────
    methods_to_plot = ["latent_mas"]  # always available
    if args.model_b_hybrid:
        methods_to_plot.append("latent_mas_hybrid")
    if args.model_b_plus:
        methods_to_plot.append("latent_mas_plus")

    n_methods = len(methods_to_plot)
    print(f"\n{'='*60}")
    print(f"  t-SNE Alignment Visualization")
    print(f"  Methods: {methods_to_plot}")
    print(f"  model_a:        {args.model_a}")
    print(f"  model_b_hybrid: {args.model_b_hybrid or '(skipped)'}")
    print(f"  model_b_plus:   {args.model_b_plus or '(skipped)'}")
    print(f"  n_tokens:       {args.n_tokens}")
    print(f"  n_samples:      {args.n_samples} (GSM8K)")
    print(f"  perplexity:     {args.perplexity}")
    print(f"{'='*60}\n")

    # ── Load models ─────────────────────────────────────────────────────
    model_a, tok_a = load_model_and_tokenizer(args.model_a, args.device)

    model_b_hybrid, tok_b_hybrid = None, None
    model_b_plus, tok_b_plus = None, None

    if args.model_b_hybrid:
        if args.model_b_hybrid == args.model_a:
            model_b_hybrid, tok_b_hybrid = model_a, tok_a
        else:
            model_b_hybrid, tok_b_hybrid = load_model_and_tokenizer(
                args.model_b_hybrid, args.device)

    if args.model_b_plus:
        if args.model_b_plus == args.model_a:
            model_b_plus, tok_b_plus = model_a, tok_a
        elif args.model_b_plus == args.model_b_hybrid:
            model_b_plus, tok_b_plus = model_b_hybrid, tok_b_hybrid
        else:
            model_b_plus, tok_b_plus = load_model_and_tokenizer(
                args.model_b_plus, args.device)

    # ── Extract weight matrices (float32 on CPU) ────────────────────────
    W_out_A = model_a.get_output_embeddings().weight.detach().cpu().float()
    W_in_A = model_a.get_input_embeddings().weight.detach().cpu().float()

    # ================================================================
    # Part 1: Build alignment matrices
    # ================================================================

    # -- Method 1: self-alignment --
    print("\n[1/3] Computing self-alignment (latent_mas) ...")
    W_self, tn_self = build_self_alignment(model_a, args.lambda_reg)
    W_self, tn_self = W_self.cpu(), tn_self.cpu()

    # -- Method 2: hybrid --
    W_hyb, tn_hyb, min_V_hyb = None, None, None
    W_in_B_hybrid = None
    if "latent_mas_hybrid" in methods_to_plot:
        print("\n[2/3] Computing hybrid alignment (latent_mas_hybrid) ...")
        W_in_B_hybrid = model_b_hybrid.get_input_embeddings().weight.detach().cpu().float()
        W_hyb, tn_hyb, min_V_hyb = build_hybrid_alignment(
            model_a, model_b_hybrid, args.lambda_reg)
        W_hyb, tn_hyb = W_hyb.cpu(), tn_hyb.cpu()

    # -- Method 3: plus --
    W_plus, tn_plus, idx_a_plus, idx_b_plus = None, None, None, None
    W_in_B_plus = None
    if "latent_mas_plus" in methods_to_plot:
        print("\n[3/3] Computing intersection alignment (latent_mas_plus) ...")
        W_in_B_plus = model_b_plus.get_input_embeddings().weight.detach().cpu().float()
        W_plus, tn_plus, idx_a_plus, idx_b_plus = build_plus_alignment(
            model_a, model_b_plus, tok_a, tok_b_plus, args.lambda_plus)
        W_plus, tn_plus = W_plus.cpu(), tn_plus.cpu()

    # ================================================================
    # Part 2: Vocabulary-level alignment vectors
    # ================================================================
    print("\n--- Vocabulary-level alignment ---")
    V_A = W_out_A.shape[0]
    all_idx = list(range(V_A))

    src_self, aligned_self, tgt_self = sample_token_vectors(
        W_out_A, W_in_A, W_self, tn_self, all_idx, all_idx, args.n_tokens, rng)
    cos_self = compute_alignment_score(aligned_self, tgt_self)
    print(f"  Self-alignment cosine: {cos_self:.4f}")

    cos_hybrid_vocab = None
    src_hybrid, aligned_hybrid, tgt_hybrid = None, None, None
    if W_hyb is not None:
        shared_idx = list(range(min_V_hyb))
        src_hybrid, aligned_hybrid, tgt_hybrid = sample_token_vectors(
            W_out_A, W_in_B_hybrid, W_hyb, tn_hyb,
            shared_idx, shared_idx, args.n_tokens, rng)
        cos_hybrid_vocab = compute_alignment_score(aligned_hybrid, tgt_hybrid)
        print(f"  Hybrid alignment cosine: {cos_hybrid_vocab:.4f}")

    cos_plus_vocab = None
    src_plus, aligned_plus, tgt_plus = None, None, None
    if W_plus is not None:
        src_plus, aligned_plus, tgt_plus = sample_token_vectors(
            W_out_A, W_in_B_plus, W_plus, tn_plus,
            idx_a_plus, idx_b_plus, args.n_tokens, rng)
        cos_plus_vocab = compute_alignment_score(aligned_plus, tgt_plus)
        print(f"  Plus alignment cosine: {cos_plus_vocab:.4f}")

    # ================================================================
    # Part 3: Inference-level alignment (real hidden states from GSM8K)
    # ================================================================
    print(f"\n--- Inference-level alignment (GSM8K, {args.n_samples} samples) ---")
    print(f"  Loading GSM8K test set ...")
    gsm8k_questions = load_gsm8k_questions(args.n_samples, args.seed)
    print(f"  Sampled {len(gsm8k_questions)} questions")
    print(f"  Example: \"{gsm8k_questions[0][:80]}...\"")

    print(f"  Running model A inference ...")
    last_hidden, predicted_ids = collect_inference_vectors_gsm8k(
        model_a, tok_a, gsm8k_questions, args.device)
    n_pos = last_hidden.shape[0]
    print(f"  Total tokens collected: {n_pos}")

    # Decode a few predicted tokens for sanity check
    pred_text = tok_a.decode(predicted_ids[:10])
    print(f"  First 10 predicted next-tokens: {repr(pred_text)}")

    # -- Self-alignment inference --
    # All vectors in d_A space: hidden[t], aligned[t]=hidden@W_self, target=W_in_A[pred]
    aligned_infer_self = apply_alignment(last_hidden, W_self, tn_self).numpy()
    tgt_infer_self = W_in_A[predicted_ids].numpy()
    hidden_self_np = last_hidden.numpy()
    cos_infer_self = compute_alignment_score(aligned_infer_self, tgt_infer_self)
    print(f"  [Inference] Self-alignment cosine: {cos_infer_self:.4f}")

    # -- Hybrid inference --
    cos_infer_hybrid = None
    aligned_infer_hybrid, tgt_infer_hybrid = None, None
    if W_hyb is not None:
        # Same family → token IDs directly correspond; filter to valid range
        valid_mask = predicted_ids < min_V_hyb
        valid_pos = torch.where(valid_mask)[0]
        if len(valid_pos) > 0:
            h_hyb = last_hidden[valid_pos]
            pred_hyb = predicted_ids[valid_pos]
            aligned_infer_hybrid = apply_alignment(h_hyb, W_hyb, tn_hyb).numpy()
            tgt_infer_hybrid = W_in_B_hybrid[pred_hyb].numpy()
            cos_infer_hybrid = compute_alignment_score(
                aligned_infer_hybrid, tgt_infer_hybrid)
            print(f"  [Inference] Hybrid alignment cosine: {cos_infer_hybrid:.4f} "
                  f"({len(valid_pos)}/{n_pos} tokens valid)")
        else:
            print("  [Inference] Hybrid: no valid tokens found")

    # -- Plus inference --
    cos_infer_plus = None
    aligned_infer_plus, tgt_infer_plus = None, None
    if W_plus is not None:
        positions, tgt_ids = map_tokens_across_tokenizers(
            predicted_ids, tok_a, tok_b_plus)
        if len(positions) > 0:
            h_plus = last_hidden[torch.tensor(positions, dtype=torch.long)]
            aligned_infer_plus = apply_alignment(h_plus, W_plus, tn_plus).numpy()
            tgt_infer_plus = W_in_B_plus[
                torch.tensor(tgt_ids, dtype=torch.long)].numpy()
            cos_infer_plus = compute_alignment_score(
                aligned_infer_plus, tgt_infer_plus)
            print(f"  [Inference] Plus alignment cosine: {cos_infer_plus:.4f} "
                  f"({len(positions)}/{n_pos} tokens mapped)")
        else:
            print("  [Inference] Plus: no tokens could be mapped")

    # ================================================================
    # Part 4: t-SNE Visualization  (2 rows × n_methods columns)
    #   Row 1 = Vocabulary-level,  Row 2 = Inference-level
    # ================================================================
    print("\nRunning t-SNE ...")

    # Color scheme
    COLOR_SRC = "#E74C3C"      # red   – source / hidden states
    COLOR_ALIGNED = "#2ECC71"  # green – aligned embeddings
    COLOR_TGT = "#3498DB"      # blue  – target embeddings
    colors3 = [COLOR_SRC, COLOR_ALIGNED, COLOR_TGT]
    markers3 = ["o", "^", "s"]
    colors2 = [COLOR_ALIGNED, COLOR_TGT]
    markers2 = ["^", "s"]

    fig, axes = plt.subplots(2, n_methods, figsize=(7 * n_methods, 12),
                              squeeze=False)

    # ── Row 1: Vocabulary-level alignment ───────────────────────────────
    col = 0

    # --- Panel (0, 0): latent_mas vocab ---
    data_self_v = {
        "Source (W_out)": src_self,
        "Aligned (W_out @ W_self)": aligned_self,
        "Target (W_in)": tgt_self,
    }
    parts_self_v = run_tsne(data_self_v, args.perplexity, args.seed)
    plot_single_method(
        axes[0, col], parts_self_v,
        f"[Vocab] latent_mas (same model)\n"
        f"{os.path.basename(args.model_a)}\n"
        f"cos = {cos_self:.4f}",
        colors3, markers3)
    col += 1

    # --- Panel (0, 1): latent_mas_hybrid vocab ---
    if "latent_mas_hybrid" in methods_to_plot:
        src_hyb_proj = (torch.from_numpy(src_hybrid).float() @ W_hyb).numpy()
        src_norms = np.linalg.norm(src_hyb_proj, axis=1, keepdims=True) + 1e-8
        src_hyb_proj = src_hyb_proj * (tn_hyb.item() / src_norms)
        data_hyb_v = {
            "Source (projected)": src_hyb_proj,
            "Aligned (W_out_A @ W_cross)": aligned_hybrid,
            "Target (W_in_B)": tgt_hybrid,
        }
        parts_hyb_v = run_tsne(data_hyb_v, args.perplexity, args.seed + 1)
        plot_single_method(
            axes[0, col], parts_hyb_v,
            f"[Vocab] latent_mas_hybrid\n"
            f"{os.path.basename(args.model_a)} → "
            f"{os.path.basename(args.model_b_hybrid)}\n"
            f"cos = {cos_hybrid_vocab:.4f}",
            colors3, markers3)
        col += 1

    # --- Panel (0, 2): latent_mas_plus vocab ---
    if "latent_mas_plus" in methods_to_plot:
        src_plus_proj = (torch.from_numpy(src_plus).float() @ W_plus).numpy()
        src_norms = np.linalg.norm(src_plus_proj, axis=1, keepdims=True) + 1e-8
        src_plus_proj = src_plus_proj * (tn_plus.item() / src_norms)
        data_plus_v = {
            "Source (projected)": src_plus_proj,
            "Aligned (W_out_A @ W_inter)": aligned_plus,
            "Target (W_in_B)": tgt_plus,
        }
        parts_plus_v = run_tsne(data_plus_v, args.perplexity, args.seed + 2)
        plot_single_method(
            axes[0, col], parts_plus_v,
            f"[Vocab] latent_mas_plus\n"
            f"{os.path.basename(args.model_a)} → "
            f"{os.path.basename(args.model_b_plus)}\n"
            f"cos = {cos_plus_vocab:.4f}",
            colors3, markers3)
        col += 1

    # ── Row 2: Inference-level alignment ────────────────────────────────
    col = 0

    # --- Panel (1, 0): latent_mas inference ---
    # Self-alignment: all vectors in d_A space → 3 clusters directly
    data_infer_self = {
        "Hidden (h_t)": hidden_self_np,
        "Aligned (h_t @ W_self)": aligned_infer_self,
        "Target (W_in[pred_t])": tgt_infer_self,
    }
    parts_infer_self = run_tsne(data_infer_self, args.perplexity, args.seed + 10)
    plot_single_method(
        axes[1, col], parts_infer_self,
        f"[Inference] latent_mas (same model)\n"
        f"{os.path.basename(args.model_a)}\n"
        f"cos = {cos_infer_self:.4f}",
        colors3, markers3)
    col += 1

    # --- Panel (1, 1): latent_mas_hybrid inference ---
    if "latent_mas_hybrid" in methods_to_plot:
        if aligned_infer_hybrid is not None:
            # Cross-model: aligned & target in d_B space → 2 clusters
            data_infer_hyb = {
                "Aligned (h_t @ W_cross)": aligned_infer_hybrid,
                "Target (W_in_B[pred_t])": tgt_infer_hybrid,
            }
            parts_infer_hyb = run_tsne(
                data_infer_hyb, args.perplexity, args.seed + 11)
            plot_single_method(
                axes[1, col], parts_infer_hyb,
                f"[Inference] latent_mas_hybrid\n"
                f"{os.path.basename(args.model_a)} → "
                f"{os.path.basename(args.model_b_hybrid)}\n"
                f"cos = {cos_infer_hybrid:.4f}",
                colors2, markers2)
        else:
            axes[1, col].text(
                0.5, 0.5, "No valid tokens", ha="center", va="center",
                transform=axes[1, col].transAxes, fontsize=12)
            axes[1, col].set_title(
                "[Inference] latent_mas_hybrid\n(no data)", fontsize=11)
        col += 1

    # --- Panel (1, 2): latent_mas_plus inference ---
    if "latent_mas_plus" in methods_to_plot:
        if aligned_infer_plus is not None:
            data_infer_plus = {
                "Aligned (h_t @ W_inter)": aligned_infer_plus,
                "Target (W_in_B[pred_t])": tgt_infer_plus,
            }
            parts_infer_plus = run_tsne(
                data_infer_plus, args.perplexity, args.seed + 12)
            plot_single_method(
                axes[1, col], parts_infer_plus,
                f"[Inference] latent_mas_plus\n"
                f"{os.path.basename(args.model_a)} → "
                f"{os.path.basename(args.model_b_plus)}\n"
                f"cos = {cos_infer_plus:.4f}",
                colors2, markers2)
        else:
            axes[1, col].text(
                0.5, 0.5, "No mappable tokens", ha="center", va="center",
                transform=axes[1, col].transAxes, fontsize=12)
            axes[1, col].set_title(
                "[Inference] latent_mas_plus\n(no data)", fontsize=11)
        col += 1

    plt.suptitle(
        "t-SNE: Latent Alignment Quality\n"
        "Row 1: Vocabulary-level  |  Row 2: Real Inference",
        fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()

    # ── Save ────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    fig.savefig(args.output, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"\nSaved figure to: {args.output}")
    plt.close(fig)

    # ── Summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Alignment Quality Summary (cosine similarity)")
    print(f"  {'':35s} {'Vocab':>8s} {'Infer':>8s}")
    print(f"  {'-'*53}")
    print(f"  {'latent_mas (self):':<35s} {cos_self:>8.4f} {cos_infer_self:>8.4f}")
    if cos_hybrid_vocab is not None:
        hyb_str = f"{cos_infer_hybrid:.4f}" if cos_infer_hybrid is not None else "  N/A "
        print(f"  {'latent_mas_hybrid (same family):':<35s} "
              f"{cos_hybrid_vocab:>8.4f} {hyb_str:>8s}")
    if cos_plus_vocab is not None:
        plus_str = f"{cos_infer_plus:.4f}" if cos_infer_plus is not None else "  N/A "
        print(f"  {'latent_mas_plus (diff family):':<35s} "
              f"{cos_plus_vocab:>8.4f} {plus_str:>8s}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
