"""
t-SNE Visualization of Alignment Quality for LatentMAS Methods
====================================================================

This script rigorously evaluates "Parameter Space Alignment" (Original) 
vs "Activation Manifold Alignment" (Improved Middle-Layer Injection) with:
1. Strict Train/Test Isolation to prove Generalization.
2. Autoregressive Temporal Shift (h_t -> h_{t+1}) to evaluate TRUE 
   next-step predictive power, avoiding feature-reconstruction illusions.

Usage
-----
    python scripts/tsne_alignment.py \
        --model_a Qwen/Qwen3-1.7B \
        --n_train 150 \
        --n_test 30 \
        --mid_layer_idx 12 \
        --perplexity 30 \
        --output figures/tsne_autoregressive.png
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
# Alignment matrix builders
# =========================================================================

def build_self_alignment(model, lambda_reg: float = 1e-5):
    """
    Method 1: latent_mas (Original)
    Computed using STATIC parameter weights. 
    Formula: W_a = (W_out^T W_out + λI)^{-1} W_out^T W_in
    """
    W_out = model.get_output_embeddings().weight.detach().float()  # [V, d]
    W_in = model.get_input_embeddings().weight.detach().float()    # [V, d]
    d = W_out.shape[1]
    gram = W_out.T @ W_out + lambda_reg * torch.eye(d, device=W_out.device)
    rhs = W_out.T @ W_in
    W_align = torch.linalg.solve(gram, rhs)  #[d, d]
    target_norm = W_in.norm(dim=1).mean()
    return W_align, target_norm


def build_mid_layer_alignment_autoregressive(source_h_L: torch.Tensor, target_h_K: torch.Tensor, lambda_reg: float = 1e-4):
    """
    Method 2: Middle-Layer Injection (Improved)
    Trained strictly on the TRAIN set WITH Temporal Shift.
    Maps h_t^(L) ---> h_{t+1}^(K)
    """
    d = source_h_L.shape[1]
    gram = source_h_L.T @ source_h_L + lambda_reg * torch.eye(d, device=source_h_L.device)
    rhs = source_h_L.T @ target_h_K
    W_mid = torch.linalg.solve(gram, rhs)  # [d, d]
    target_norm_train = target_h_K.norm(dim=1).mean() 
    return W_mid, target_norm_train


def apply_alignment(hidden: torch.Tensor, W_align: torch.Tensor, target_norm: torch.Tensor) -> torch.Tensor:
    """Project hidden through alignment matrix and scale using learned target_norm."""
    aligned = (hidden.float() @ W_align)
    norms = aligned.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    aligned = aligned * (target_norm / norms)
    return aligned


# =========================================================================
# Dataset & Inference helpers
# =========================================================================

def load_gsm8k_splits(n_train: int, n_test: int, seed: int = 42):
    """Load train and test questions from completely isolated splits."""
    ds_train = load_dataset("gsm8k", "main", split="train")
    ds_test = load_dataset("gsm8k", "main", split="test")
    
    rng = np.random.RandomState(seed)
    train_idx = rng.choice(len(ds_train), size=min(n_train, len(ds_train)), replace=False)
    test_idx = rng.choice(len(ds_test), size=min(n_test, len(ds_test)), replace=False)
    
    q_train = [ds_train[int(i)]["question"].strip() for i in train_idx]
    q_test = [ds_test[int(i)]["question"].strip() for i in test_idx]
    
    return q_train, q_test


@torch.no_grad()
def collect_shifted_inference_vectors(model, tokenizer, questions, mid_layer_idx, device="cpu", desc=""):
    """
    Runs inference and extracts vectors with strictly aligned temporal shifts:
    Source: h_t^(L)
    Target 1 (Improved): h_{t+1}^(K) (Actual next token's middle-layer state)
    Target 2 (Original): e_{t+1} (Embedding of model's predicted next token)
    """
    all_source_L =[]
    all_target_K = []
    all_target_e_ids =[]
    
    for i, q in enumerate(questions):
        inputs = tokenizer(q, return_tensors="pt").to(device)
        outputs = model(**inputs, output_hidden_states=True)
        
        # Extract features for the whole sequence: [seq_len, d]
        h_L = outputs.hidden_states[-1][0].cpu().float()       
        h_K = outputs.hidden_states[mid_layer_idx][0].cpu().float() 
        
        # logits -> predicted next tokens
        logits = outputs.logits[0].cpu().float()
        pred_ids = logits.argmax(dim=-1) # pred_ids[t] is prediction for t+1
        
        seq_len = h_L.shape[0]
        if seq_len < 2:
            continue
            
        # ===============================================================
        # THE CRITICAL AUTOREGRESSIVE SHIFT (t -> t+1) WITHIN THE SEQUENCE
        # ===============================================================
        source_L = h_L[:-1]           # h[0], h[1], ..., h[T-1]
        target_K = h_K[1:]            # h[1], h[2], ..., h[T]
        target_e_ids = pred_ids[:-1]  # predictions made at 0, 1, ..., T-1
        
        all_source_L.append(source_L)
        all_target_K.append(target_K)
        all_target_e_ids.append(target_e_ids)
        
        if (i + 1) % 10 == 0 or i == len(questions) - 1:
            total_tokens = sum(x.shape[0] for x in all_source_L)
            print(f"    [{desc}] Processed {i+1}/{len(questions)} questions. Collected {total_tokens} shifted pairs.")
            
    return torch.cat(all_source_L, dim=0), torch.cat(all_target_K, dim=0), torch.cat(all_target_e_ids, dim=0)


def load_model_and_tokenizer(model_name: str, device: str = "cpu"):
    print(f"Loading model: {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32, device_map=device
    )
    model.eval()
    return model, tokenizer


def compute_alignment_score(aligned: np.ndarray, target: np.ndarray) -> float:
    dot = np.sum(aligned * target, axis=1)
    norm_a = np.linalg.norm(aligned, axis=1) + 1e-8
    norm_t = np.linalg.norm(target, axis=1) + 1e-8
    return float(np.mean(dot / (norm_a * norm_t)))

# =========================================================================
# t-SNE + Plotting
# =========================================================================

def run_tsne(data_dict, perplexity, seed):
    labels_list = list(data_dict.keys())
    arrays = [data_dict[k] for k in labels_list]
    sizes =[a.shape[0] for a in arrays]
    combined = np.concatenate(arrays, axis=0)

    tsne = TSNE(
        n_components=2, perplexity=min(perplexity, max(2, combined.shape[0] // 4)),
        random_state=seed, init="pca", learning_rate="auto", max_iter=1000,
    )
    embedded = tsne.fit_transform(combined)

    parts, offset = {}, 0
    for label, sz in zip(labels_list, sizes):
        parts[label] = embedded[offset: offset + sz]
        offset += sz
    return parts


def plot_single_method(ax, parts, title, colors, markers):
    for (label, pts), c, m in zip(parts.items(), colors, markers):
        ax.scatter(pts[:, 0], pts[:, 1], c=c, marker=m, s=12, alpha=0.55,
                   label=label, edgecolors="none")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(fontsize=7, loc="lower left", framealpha=0.9)
    ax.set_xticks([]); ax.set_yticks([])


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_a", type=str, required=True)
    p.add_argument("--n_train", type=int, default=150, help="Questions for computing W_mid")
    p.add_argument("--n_test", type=int, default=30, help="Questions for evaluation and t-SNE")
    p.add_argument("--mid_layer_idx", type=int, default=14, help="Target layer for injection (e.g., 0, 12, 20). -1 means middle.")
    p.add_argument("--perplexity", type=float, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lambda_reg", type=float, default=1e-5)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--output", type=str, default="figures/tsne_autoregressive.png")
    return p.parse_args()


def main():
    args = parse_args()
    rng = np.random.RandomState(args.seed)
    torch.manual_seed(args.seed)

    print(f"\n{'='*70}")
    print(f"  t-SNE Alignment: Autoregressive Generalization (t -> t+1)")
    print(f"  Train samples: {args.n_train} | Test samples: {args.n_test}")
    print(f"{'='*70}")

    model_a, tok_a = load_model_and_tokenizer(args.model_a, args.device)
    num_layers = getattr(model_a.config, "num_hidden_layers")
    
    # Process target layer logic
    mid_layer_idx = args.mid_layer_idx
    print(f"  Detected {num_layers} layers. Mid-layer injection target: Layer {mid_layer_idx}.")

    # ================================================================
    # 1. Collect Data (Strict Train/Test Split + Temporal Shift)
    # ================================================================
    print("\n--- 1. Collecting Data (Temporal Shift: h_t -> h_t+1) ---")
    q_train, q_test = load_gsm8k_splits(args.n_train, args.n_test, args.seed)
    
    # Train set (only source_L and target_K are used for fitting W_mid)
    src_L_train, tgt_K_train, _ = collect_shifted_inference_vectors(
        model_a, tok_a, q_train, mid_layer_idx, args.device, desc="TRAIN")
    
    # Test set
    src_L_test, tgt_K_test, tgt_e_ids_test = collect_shifted_inference_vectors(
        model_a, tok_a, q_test, mid_layer_idx, args.device, desc="TEST")

    W_in_A = model_a.get_input_embeddings().weight.detach().cpu().float()

    # ================================================================
    # 2. Compute Alignments
    # ================================================================
    print("\n--- 2. Computing Alignment Matrices ---")
    
    # Method 1: Original LatentMAS (Static Weights)
    W_self, tn_self = build_self_alignment(model_a, args.lambda_reg)
    W_self, tn_self = W_self.cpu(), tn_self.cpu()
    print(f"  -> [Original] Computed W_self using static Model Weights (vocab space)")

    # Method 2: Improved Middle-Layer Injection (Learned from TRAIN shift pairs)
    W_mid, tn_mid = build_mid_layer_alignment_autoregressive(src_L_train, tgt_K_train, args.lambda_reg)
    W_mid, tn_mid = W_mid.cpu(), tn_mid.cpu()
    print(f"  -> [Improved] Computed W_mid using {src_L_train.shape[0]} TRAIN pairs [h_t^(L) -> h_t+1^(K)]")

    # ================================================================
    # 3. Evaluate Alignments (Strictly on TEST set)
    # ================================================================
    print("\n--- 3. Evaluating Autoregressive Predictability on Unseen TEST Data ---")

    # Original Evaluation: h_t^(L) -> e_t+1 (Input embedding of predicted token)
    aligned_infer_self_test = apply_alignment(src_L_test, W_self, tn_self).numpy()
    target_e_test = W_in_A[tgt_e_ids_test].numpy()
    cos_infer_self = compute_alignment_score(aligned_infer_self_test, target_e_test)
    print(f"  [Original] LatentMAS (h_t^(L) -> e_t+1) Cosine on TEST: {cos_infer_self:.4f}")

    # Middle-Layer Evaluation: h_t^(L) -> h_t+1^(K)
    aligned_infer_mid_test = apply_alignment(src_L_test, W_mid, tn_mid).numpy()
    target_mid_test = tgt_K_test.numpy()
    cos_infer_mid = compute_alignment_score(aligned_infer_mid_test, target_mid_test)
    print(f"  [Improved] Data-Driven (h_t^(L) -> h_t+1^({mid_layer_idx})) Cosine on TEST: {cos_infer_mid:.4f}")

    # ================================================================
    # 4. t-SNE Plotting
    # ================================================================
    print("\n--- 4. Running t-SNE Visualization on TEST Data ---")
    COLOR_SRC = "#E74C3C"      # red
    COLOR_ALIGNED = "#2ECC71"  # green
    COLOR_TGT = "#3498DB"      # blue
    colors3 =[COLOR_SRC, COLOR_ALIGNED, COLOR_TGT]
    markers3 =["o", "^", "s"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 6), squeeze=False)
    source_test_np = src_L_test.numpy()

    # Panel 1: Original Method
    data_test_self = {
        "Source (h_t^L)": source_test_np,
        "Aligned (h_t^L @ W_self)": aligned_infer_self_test,
        "Target (e_t+1)": target_e_test,
    }
    plot_single_method(axes[0, 0], run_tsne(data_test_self, args.perplexity, args.seed + 10),
        f"[Original] Parameter Space (TEST)\nh_t^(L) -> e_t+1 (Vocab Embed)\ncos = {cos_infer_self:.4f}",
        colors3, markers3)

    # Panel 2: Middle-Layer Method
    data_test_mid = {
        "Source (h_t^L)": source_test_np,
        f"Aligned (h_t^L @ W_mid)": aligned_infer_mid_test,
        f"Target (h_t+1^{mid_layer_idx})": target_mid_test,
    }
    plot_single_method(axes[0, 1], run_tsne(data_test_mid, args.perplexity, args.seed + 20),
        f"[Improved] Activation Manifold (TEST)\nh_t^(L) -> h_t+1^({mid_layer_idx}) (Data-Driven)\ncos = {cos_infer_mid:.4f}",
        colors3, markers3)

    plt.suptitle("t-SNE: Autoregressive Generalization (Train on Train-set, Evaluate on Test-set)", 
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    fig.savefig(args.output, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"\nSaved generalization comparison figure to: {args.output}")

if __name__ == "__main__":
    main()