#!/usr/bin/env python3
"""
t-SNE Visualization of Alignment Quality for Three LatentMAS Methods
====================================================================

Visualizes how well each method aligns source model hidden states to the
target model's input embedding space:

1. **latent_mas** (same model):  Self-alignment W = (W_out^T W_out + λI)^{-1} W_out^T W_in
2. **latent_mas_hybrid** (same family, different size):  Full-vocab cross-model alignment
3. **latent_mas_plus** (different families):  Intersection-vocab cross-model alignment

For each method we sample N shared vocabulary tokens and plot:
  - Source hidden vectors  (W_out_A[t])
  - Aligned embeddings     (W_out_A[t] @ W_alignment)
  - Target input embeddings (W_in_B[t])

A perfect alignment means "aligned" and "target" points overlap.

Usage
-----
    python scripts/tsne_alignment.py \
        --model_a Qwen/Qwen3-4B \
        --model_b_hybrid Qwen/Qwen3-8B \
        --model_b_plus meta-llama/Llama-3.1-8B \
        --n_tokens 1000 \
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
    if V_A != V_B:
        min_V = min(V_A, V_B)
        W_out_A = W_out_A[:min_V]
        W_in_B = W_in_B[:min_V]
        print(f"  [hybrid] Vocab size mismatch ({V_A} vs {V_B}), using first {min_V}")
    gram = W_out_A.T @ W_out_A + lambda_reg * torch.eye(d_A, device=W_out_A.device)
    rhs = W_out_A.T @ W_in_B
    W_align = torch.linalg.solve(gram, rhs)  # [d_A, d_B]
    target_norm = W_in_B.norm(dim=1).mean()
    return W_align, target_norm, min(V_A, V_B)


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

def run_tsne_and_plot(data_dict, perplexity, seed, output_path, method_name):
    """
    data_dict: {label: ndarray of shape [N, D]}
    Runs t-SNE on the concatenation, then plots per-label with different colors.
    """
    labels_list = list(data_dict.keys())
    arrays = [data_dict[k] for k in labels_list]
    sizes = [a.shape[0] for a in arrays]
    combined = np.concatenate(arrays, axis=0)

    tsne = TSNE(
        n_components=2,
        perplexity=min(perplexity, combined.shape[0] // 4),
        random_state=seed,
        init="pca",
        learning_rate="auto",
        n_iter=1000,
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
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right", framealpha=0.8)
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
    p = argparse.ArgumentParser(description="t-SNE alignment visualization for LatentMAS methods")
    p.add_argument("--model_a", type=str, required=True,
                   help="Base model (used by all agents in latent_mas, and as model_A in hybrid/plus)")
    p.add_argument("--model_b_hybrid", type=str, default=None,
                   help="Target model for latent_mas_hybrid (same family, different size)")
    p.add_argument("--model_b_plus", type=str, default=None,
                   help="Target model for latent_mas_plus (different family)")
    p.add_argument("--n_tokens", type=int, default=1000,
                   help="Number of vocabulary tokens to sample")
    p.add_argument("--perplexity", type=float, default=30,
                   help="t-SNE perplexity")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lambda_reg", type=float, default=1e-5,
                   help="Ridge regularization for alignment")
    p.add_argument("--lambda_plus", type=float, default=1e-4,
                   help="Ridge regularization for intersection alignment")
    p.add_argument("--device", type=str, default="cpu",
                   help="Device for computation (cpu recommended for visualization)")
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
    print(f"  perplexity:     {args.perplexity}")
    print(f"{'='*60}\n")

    # ── Load models ─────────────────────────────────────────────────────
    model_a, tok_a = load_model_and_tokenizer(args.model_a, args.device)

    model_b_hybrid, tok_b_hybrid = None, None
    model_b_plus, tok_b_plus = None, None

    if args.model_b_hybrid:
        # same-family: may share tokenizer
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

    # ── Extract weight matrices (float32 on CPU for t-SNE) ──────────────
    W_out_A = model_a.get_output_embeddings().weight.detach().cpu().float()
    W_in_A = model_a.get_input_embeddings().weight.detach().cpu().float()

    # ── Method 1: latent_mas (self-alignment) ───────────────────────────
    print("\n[1/3] Computing self-alignment (latent_mas) ...")
    W_self, tn_self = build_self_alignment(model_a, args.lambda_reg)
    W_self, tn_self = W_self.cpu(), tn_self.cpu()
    V_A = W_out_A.shape[0]
    all_idx = list(range(V_A))
    src_self, aligned_self, tgt_self = sample_token_vectors(
        W_out_A, W_in_A, W_self, tn_self, all_idx, all_idx, args.n_tokens, rng)
    cos_self = compute_alignment_score(aligned_self, tgt_self)
    print(f"  Self-alignment cosine similarity: {cos_self:.4f}")

    # ── Method 2: latent_mas_hybrid (same family, cross-size) ───────────
    src_hybrid, aligned_hybrid, tgt_hybrid, cos_hybrid = None, None, None, None
    if "latent_mas_hybrid" in methods_to_plot:
        print("\n[2/3] Computing hybrid alignment (latent_mas_hybrid) ...")
        W_in_B_hybrid = model_b_hybrid.get_input_embeddings().weight.detach().cpu().float()
        W_hyb, tn_hyb, min_V_hyb = build_hybrid_alignment(
            model_a, model_b_hybrid, args.lambda_reg)
        W_hyb, tn_hyb = W_hyb.cpu(), tn_hyb.cpu()
        shared_idx = list(range(min_V_hyb))
        src_hybrid, aligned_hybrid, tgt_hybrid = sample_token_vectors(
            W_out_A, W_in_B_hybrid, W_hyb, tn_hyb, shared_idx, shared_idx,
            args.n_tokens, rng)
        cos_hybrid = compute_alignment_score(aligned_hybrid, tgt_hybrid)
        print(f"  Hybrid alignment cosine similarity: {cos_hybrid:.4f}")

    # ── Method 3: latent_mas_plus (different family, intersection) ──────
    src_plus, aligned_plus, tgt_plus, cos_plus = None, None, None, None
    if "latent_mas_plus" in methods_to_plot:
        print("\n[3/3] Computing intersection alignment (latent_mas_plus) ...")
        W_in_B_plus = model_b_plus.get_input_embeddings().weight.detach().cpu().float()
        W_plus, tn_plus, idx_a_plus, idx_b_plus = build_plus_alignment(
            model_a, model_b_plus, tok_a, tok_b_plus, args.lambda_plus)
        W_plus, tn_plus = W_plus.cpu(), tn_plus.cpu()
        src_plus, aligned_plus, tgt_plus = sample_token_vectors(
            W_out_A, W_in_B_plus, W_plus, tn_plus, idx_a_plus, idx_b_plus,
            args.n_tokens, rng)
        cos_plus = compute_alignment_score(aligned_plus, tgt_plus)
        print(f"  Plus alignment cosine similarity: {cos_plus:.4f}")

    # ── t-SNE ───────────────────────────────────────────────────────────
    print("\nRunning t-SNE ...")

    # Color scheme
    COLOR_SRC = "#E74C3C"      # red   – source hidden states
    COLOR_ALIGNED = "#2ECC71"  # green – aligned embeddings
    COLOR_TGT = "#3498DB"      # blue  – target embeddings
    colors = [COLOR_SRC, COLOR_ALIGNED, COLOR_TGT]
    markers = ["o", "^", "s"]

    fig, axes = plt.subplots(1, n_methods, figsize=(7 * n_methods, 6),
                              squeeze=False)
    axes = axes.flatten()

    panel_idx = 0

    # --- Panel 1: latent_mas ---
    data_self = {
        f"Source (W_out)": src_self,
        f"Aligned (W_out @ W_self)": aligned_self,
        f"Target (W_in)": tgt_self,
    }
    parts_self = run_tsne_and_plot(data_self, args.perplexity, args.seed,
                                    args.output, "latent_mas")
    plot_single_method(
        axes[panel_idx], parts_self,
        f"latent_mas (same model)\n"
        f"{os.path.basename(args.model_a)}\n"
        f"cos_sim = {cos_self:.4f}",
        colors, markers)
    panel_idx += 1

    # --- Panel 2: latent_mas_hybrid ---
    if "latent_mas_hybrid" in methods_to_plot:
        # Aligned and target live in model_b_hybrid's embedding space,
        # but source lives in model_a's space (different dim potentially).
        # We project source into target space via W_hyb for a fair comparison,
        # OR we run t-SNE only on aligned vs target.

        # Since source dim != target dim for hybrid, we visualize 
        # aligned vs target in the target space, plus a separate panel
        # for source in its own space.
        # For clarity we project everything into target dim space.
        src_hybrid_projected = (torch.from_numpy(src_hybrid).float() @ W_hyb).numpy()
        # normalize projected source same way
        src_norms = np.linalg.norm(src_hybrid_projected, axis=1, keepdims=True) + 1e-8
        src_hybrid_projected = src_hybrid_projected * (tn_hyb.item() / src_norms)

        data_hyb = {
            f"Source (projected)": src_hybrid_projected,
            f"Aligned (W_out_A @ W_cross)": aligned_hybrid,
            f"Target (W_in_B)": tgt_hybrid,
        }
        parts_hyb = run_tsne_and_plot(data_hyb, args.perplexity, args.seed + 1,
                                       args.output, "latent_mas_hybrid")
        plot_single_method(
            axes[panel_idx], parts_hyb,
            f"latent_mas_hybrid (same family)\n"
            f"{os.path.basename(args.model_a)} → {os.path.basename(args.model_b_hybrid)}\n"
            f"cos_sim = {cos_hybrid:.4f}",
            colors, markers)
        panel_idx += 1

    # --- Panel 3: latent_mas_plus ---
    if "latent_mas_plus" in methods_to_plot:
        src_plus_projected = (torch.from_numpy(src_plus).float() @ W_plus).numpy()
        src_norms = np.linalg.norm(src_plus_projected, axis=1, keepdims=True) + 1e-8
        src_plus_projected = src_plus_projected * (tn_plus.item() / src_norms)

        data_plus = {
            f"Source (projected)": src_plus_projected,
            f"Aligned (W_out_A @ W_intersect)": aligned_plus,
            f"Target (W_in_B)": tgt_plus,
        }
        parts_plus = run_tsne_and_plot(data_plus, args.perplexity, args.seed + 2,
                                        args.output, "latent_mas_plus")
        plot_single_method(
            axes[panel_idx], parts_plus,
            f"latent_mas_plus (different family)\n"
            f"{os.path.basename(args.model_a)} → {os.path.basename(args.model_b_plus)}\n"
            f"cos_sim = {cos_plus:.4f}",
            colors, markers)
        panel_idx += 1

    plt.suptitle("t-SNE Visualization of Latent Alignment Quality", fontsize=16,
                 fontweight="bold", y=1.02)
    plt.tight_layout()

    # ── Save ────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    fig.savefig(args.output, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"\nSaved figure to: {args.output}")
    plt.close(fig)

    # ── Summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Alignment Quality Summary (cosine similarity)")
    print(f"  {'latent_mas (self):':<35} {cos_self:.4f}")
    if cos_hybrid is not None:
        print(f"  {'latent_mas_hybrid (same family):':<35} {cos_hybrid:.4f}")
    if cos_plus is not None:
        print(f"  {'latent_mas_plus (diff family):':<35} {cos_plus:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
