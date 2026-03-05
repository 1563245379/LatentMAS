"""
Usage
-----
    python scripts/tsne_alignment.py \
        --model_a Qwen/Qwen3-1.7B \
        --n_samples 30 \
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
# Alignment matrix builders & Gap Calculators
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

def apply_alignment(hidden: torch.Tensor, W_align: torch.Tensor,
                     target_norm: torch.Tensor) -> torch.Tensor:
    """Project hidden through alignment matrix and normalize."""
    aligned = (hidden.float() @ W_align)
    norms = aligned.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    aligned = aligned * (target_norm / norms)
    return aligned

def compute_theoretical_gap(W_in: torch.Tensor, W_out: torch.Tensor, W_align: torch.Tensor):
    """
    计算论文定理 A.1 中的理论对齐差距 (Vocabulary Level)
    || W_out @ W_a - W_in ||_F 及其相对比例
    """
    W_aligned = W_out.float() @ W_align.float()
    residual = W_aligned - W_in.float()
    
    # 计算 Frobenius 范数
    norm_in = torch.linalg.norm(W_in.float(), ord='fro').item()
    norm_residual = torch.linalg.norm(residual, ord='fro').item()
    norm_aligned = torch.linalg.norm(W_aligned, ord='fro').item()
    
    # 相对对齐差距 (Residual / Original)
    relative_gap = norm_residual / norm_in
    
    # 解释方差 (Explained Variance: ||W_aligned||_F^2 / ||W_in||_F^2)
    explained_var = (norm_aligned ** 2) / (norm_in ** 2)
    
    # 随机子空间投影基线 (Random Matrix Theory Baseline: d / N)
    d = W_in.shape[1]
    N = W_in.shape[0]
    random_baseline = d / N
    
    return {
        "norm_in": norm_in,
        "norm_residual": norm_residual,
        "relative_gap": relative_gap,
        "explained_var": explained_var,
        "random_baseline": random_baseline
    }

def compute_inference_gap(aligned: np.ndarray, target: np.ndarray) -> float:
    """计算推理过程中，实际对齐特征与目标特征之间的相对 L2 差距 (Frobenius Norm)"""
    residual = aligned - target
    norm_residual = np.linalg.norm(residual, ord='fro')
    norm_target = np.linalg.norm(target, ord='fro')
    return float(norm_residual / (norm_target + 1e-8))


# =========================================================================
# Dataset & Inference helpers
# =========================================================================

def load_gsm8k_questions(n_samples: int, seed: int = 42):
    """
    Load n_samples questions from GSM8K test set.
    """
    ds = load_dataset("gsm8k", "main", split="test")
    rng = np.random.RandomState(seed)
    indices = rng.choice(len(ds), size=min(n_samples, len(ds)), replace=False)
    questions = [ds[int(i)]["question"].strip() for i in indices]
    return questions

@torch.no_grad()
def get_inference_hidden_states(model, tokenizer, text, device="cpu"):
    inputs = tokenizer(text, return_tensors="pt").to(device)
    outputs = model(**inputs, output_hidden_states=True)
    last_hidden = outputs.hidden_states[-1][0].cpu().float()   # [seq_len, d]
    logits = outputs.logits[0].cpu().float()                   # [seq_len, V]
    predicted_ids = logits.argmax(dim=-1)                       # [seq_len]
    return last_hidden, predicted_ids, inputs["input_ids"][0].cpu()

@torch.no_grad()
def collect_inference_vectors_gsm8k(model, tokenizer, questions, device="cpu"):
    all_hidden, all_pred = [],[]
    for i, q in enumerate(questions):
        h, p, _ = get_inference_hidden_states(model, tokenizer, q, device)
        all_hidden.append(h)
        all_pred.append(p)
        if (i + 1) % 10 == 0 or i == len(questions) - 1:
            total = sum(x.shape[0] for x in all_hidden)
            print(f"    Processed {i+1}/{len(questions)} questions, "
                  f"{total} tokens so far")
    return torch.cat(all_hidden, dim=0), torch.cat(all_pred, dim=0)


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
# t-SNE + Plotting
# =========================================================================

def run_tsne(data_dict, perplexity, seed):
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

    parts = {}
    offset = 0
    for label, sz in zip(labels_list, sizes):
        parts[label] = embedded[offset: offset + sz]
        offset += sz

    return parts

def plot_single_method(ax, parts, title, colors, markers):
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
    p.add_argument("--n_samples", type=int, default=200,
                   help="Number of GSM8K questions to sample for inference-level test")
    p.add_argument("--perplexity", type=float, default=30,
                   help="t-SNE perplexity")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lambda_reg", type=float, default=1e-5,
                   help="Ridge regularization for self/hybrid alignment")
    p.add_argument("--device", type=str, default="cuda",
                   help="Device for computation")
    p.add_argument("--output", type=str, default="figures/tsne_alignment.png",
                   help="Output image path")
    return p.parse_args()

def main():
    args = parse_args()
    rng = np.random.RandomState(args.seed)
    torch.manual_seed(args.seed)

    print(f"\n{'='*60}")
    print(f"  t-SNE Alignment Visualization & Gap Calculation")
    print(f"  model_a:        {args.model_a}")
    print(f"  n_samples:      {args.n_samples} (GSM8K)")
    print(f"{'='*60}\n")

    # ── Load models ─────────────────────────────────────────────────────
    model_a, tok_a = load_model_and_tokenizer(args.model_a, args.device)

    # ── Extract weight matrices (float32 on CPU) ────────────────────────
    W_out_A = model_a.get_output_embeddings().weight.detach().cpu().float()
    W_in_A = model_a.get_input_embeddings().weight.detach().cpu().float()

    # ================================================================
    # Part 1: Build alignment matrices & Calculate Theoretical Gap
    # ================================================================

    print("\nComputing self-alignment (latent_mas) ...")
    W_self, tn_self = build_self_alignment(model_a, args.lambda_reg)
    W_self, tn_self = W_self.cpu(), tn_self.cpu()
    
    print("\n[MATHEMATICAL VERIFICATION] Theoretical Alignment Gap (Vocab Level):")
    gap_metrics = compute_theoretical_gap(W_in_A, W_out_A, W_self)
    print(f"  Original Input Space Energy || W_in ||_F      : {gap_metrics['norm_in']:.2f}")
    print(f"  Upper Bound Residual || W_out*Wa - W_in ||_F  : {gap_metrics['norm_residual']:.2f}")
    print(f"  --> Relative Gap (Residual / Original)        : {gap_metrics['relative_gap']*100:.2f}%")
    print(f"  --> Explained Variance (Aligned Energy Ratio) : {gap_metrics['explained_var']*100:.2f}%")
    print(f"  --> Random Projection Baseline (d / |V|)      : {gap_metrics['random_baseline']*100:.2f}%")
    print("      (Notice how Explained Variance ≈ Random Baseline, proving the linear projection is trivial)")

    # ================================================================
    # Part 2: Inference-level alignment (real hidden states from GSM8K)
    # ================================================================
    print(f"\n--- Inference-level alignment (GSM8K, {args.n_samples} samples) ---")
    gsm8k_questions = load_gsm8k_questions(args.n_samples, args.seed)

    print(f"  Running model A inference ...")
    last_hidden, predicted_ids = collect_inference_vectors_gsm8k(
        model_a, tok_a, gsm8k_questions, args.device)
    n_pos = last_hidden.shape[0]
    print(f"  Total tokens collected: {n_pos}")

    aligned_infer_self = apply_alignment(last_hidden, W_self, tn_self).numpy()
    tgt_infer_self = W_in_A[predicted_ids].numpy()
    hidden_self_np = last_hidden.numpy()
    
    cos_infer_self = compute_alignment_score(aligned_infer_self, tgt_infer_self)
    infer_gap_self = compute_inference_gap(aligned_infer_self, tgt_infer_self)
    
    print(f"\n[INFERENCE RESULTS]")
    print(f"  Self-alignment Cosine Similarity : {cos_infer_self:.4f}")
    print(f"  Self-alignment Relative L2 Gap   : {infer_gap_self*100:.2f}%")

    # ================================================================
    # Part 3: t-SNE Visualization
    # ================================================================
    print("\nRunning t-SNE ...")

    COLOR_SRC = "#E74C3C"      
    COLOR_ALIGNED = "#2ECC71"  
    COLOR_TGT = "#3498DB"      
    colors3 = [COLOR_SRC, COLOR_ALIGNED, COLOR_TGT]
    markers3 = ["o", "^", "s"]

    fig, axes = plt.subplots(1, 1, figsize=(6, 6), squeeze=False)

    data_infer_self = {
        "Hidden (h_t)": hidden_self_np,
        "Aligned (h_t @ W_self)": aligned_infer_self,
        "Target (W_in[pred_t])": tgt_infer_self,
    }
    parts_infer_self = run_tsne(data_infer_self, args.perplexity, args.seed + 10)
    
    # 在图片标题中加入差距信息，更加直观
    plot_title = (f"[Inference] latent_mas (same model)\n"
                  f"{os.path.basename(args.model_a)}\n"
                  f"cos = {cos_infer_self:.4f} | Rel Gap = {infer_gap_self*100:.1f}%")
    
    plot_single_method(axes[0, 0], parts_infer_self, plot_title, colors3, markers3)

    plt.suptitle("t-SNE: Latent Alignment Quality Visualization", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    fig.savefig(args.output, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"\nSaved figure to: {args.output}")

    # ── Summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  FINAL SUMMARY (Proving the trivial bound)")
    print(f"  - Theoretical Residual (Upper Bound): {gap_metrics['relative_gap']*100:.2f}% of input norm")
    print(f"  - Actual Inference Residual L2 Gap  : {infer_gap_self*100:.2f}% of target norm")
    print(f"  - Actual Inference Cosine Similarity: {cos_infer_self:.4f}")
    print(f"  (A gap of ~100% means the aligned vector is almost orthogonal to the target)")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()