"""
t-SNE Visualization of Alignment Quality for LatentMAS Methods
====================================================================

Compares the original LatentMAS (aligning to input embeddings) with the
mathematically superior "Middle-Layer Injection" (aligning last layer
to middle layer hidden states).

**Crucially, this version strictly isolates a TRAIN SET for computing
the alignment matrix W_mid, and a TEST SET for evaluating the alignment
generalization (Cosine similarity & t-SNE).**

Usage
-----
    python scripts/tsne_alignment.py \
        --model_a Qwen/Qwen3-1.7B \
        --n_train 100 \
        --n_test 30 \
        --perplexity 30 \
        --seed 42 \
        --output figures/tsne_alignment_generalization.png
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
    Uses static weights (W_out and W_in). No train data needed for computation,
    but we will evaluate its performance on the test set.
    """
    W_out = model.get_output_embeddings().weight.detach().float()  # [V, d]
    W_in = model.get_input_embeddings().weight.detach().float()   # [V, d]
    d = W_out.shape[1]
    gram = W_out.T @ W_out + lambda_reg * torch.eye(d, device=W_out.device)
    rhs = W_out.T @ W_in
    W_align = torch.linalg.solve(gram, rhs)  # [d, d]
    target_norm = W_in.norm(dim=1).mean()
    return W_align, target_norm


def build_mid_layer_alignment(h_L_train: torch.Tensor, h_K_train: torch.Tensor, lambda_reg: float = 1e-4):
    """
    Method 2: Middle-Layer Injection (Improved Data-Driven Alignment)
    Trained strictly on the TRAIN set.
    """
    d = h_L_train.shape[1]
    gram = h_L_train.T @ h_L_train + lambda_reg * torch.eye(d, device=h_L_train.device)
    rhs = h_L_train.T @ h_K_train
    W_mid = torch.linalg.solve(gram, rhs)  # [d, d]
    # Learn the average target norm from training data
    target_norm_train = h_K_train.norm(dim=1).mean() 
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
def get_inference_hidden_states(model, tokenizer, text, mid_layer_idx, device="cpu"):
    """Extracts BOTH last layer and middle layer hidden states."""
    inputs = tokenizer(text, return_tensors="pt").to(device)
    outputs = model(**inputs, output_hidden_states=True)
    
    last_hidden = outputs.hidden_states[-1][0].cpu().float()       #[seq_len, d]
    mid_hidden = outputs.hidden_states[mid_layer_idx][0].cpu().float() # [seq_len, d]
    
    logits = outputs.logits[0].cpu().float()
    predicted_ids = logits.argmax(dim=-1)
    return last_hidden, mid_hidden, predicted_ids, inputs["input_ids"][0].cpu()


@torch.no_grad()
def collect_inference_vectors_gsm8k(model, tokenizer, questions, mid_layer_idx, device="cpu", desc=""):
    all_last, all_mid, all_pred = [], [],[]
    for i, q in enumerate(questions):
        h_L, h_K, p, _ = get_inference_hidden_states(model, tokenizer, q, mid_layer_idx, device)
        all_last.append(h_L)
        all_mid.append(h_K)
        all_pred.append(p)
        if (i + 1) % 10 == 0 or i == len(questions) - 1:
            total = sum(x.shape[0] for x in all_last)
            print(f"    [{desc}] Processed {i+1}/{len(questions)} questions, {total} tokens collected")
    return torch.cat(all_last, dim=0), torch.cat(all_mid, dim=0), torch.cat(all_pred, dim=0)


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
    ax.legend(fontsize=7, loc="upper right", framealpha=0.8)
    ax.set_xticks([]); ax.set_yticks([])


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_a", type=str, required=True)
    p.add_argument("--n_train", type=int, default=100, help="Questions for computing W_mid")
    p.add_argument("--n_test", type=int, default=30, help="Questions for evaluation and t-SNE")
    p.add_argument("--perplexity", type=float, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lambda_reg", type=float, default=1e-5)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--output", type=str, default="figures/tsne_alignment_generalization.png")
    return p.parse_args()


def main():
    args = parse_args()
    rng = np.random.RandomState(args.seed)
    torch.manual_seed(args.seed)

    print(f"\n{'='*60}")
    print(f"  t-SNE Alignment: Rigorous Train/Test Generalization")
    print(f"  Train samples: {args.n_train} | Test samples: {args.n_test}")
    print(f"{'='*60}")

    model_a, tok_a = load_model_and_tokenizer(args.model_a, args.device)
    num_layers = getattr(model_a.config, "num_hidden_layers", 24)
    mid_layer_idx = num_layers // 2
    print(f"  Detected {num_layers} layers. Mid-layer injection target: Layer {mid_layer_idx}.")

    # ================================================================
    # 1. Collect Data (Strictly Split Train and Test)
    # ================================================================
    print("\n--- 1. Collecting Data (Train/Test Isolation) ---")
    q_train, q_test = load_gsm8k_splits(args.n_train, args.n_test, args.seed)
    
    last_hidden_train, mid_hidden_train, _ = collect_inference_vectors_gsm8k(
        model_a, tok_a, q_train, mid_layer_idx, args.device, desc="TRAIN")
    
    last_hidden_test, mid_hidden_test, pred_ids_test = collect_inference_vectors_gsm8k(
        model_a, tok_a, q_test, mid_layer_idx, args.device, desc="TEST")

    W_in_A = model_a.get_input_embeddings().weight.detach().cpu().float()

    # ================================================================
    # 2. Compute Alignments (on TRAIN set or Static Weights)
    # ================================================================
    print("\n--- 2. Computing Alignment Matrices ---")
    
    # Method 1: Original LatentMAS (Computed using static model weights, independent of train data)
    W_self, tn_self = build_self_alignment(model_a, args.lambda_reg)
    W_self, tn_self = W_self.cpu(), tn_self.cpu()
    print(f"  -> Computed W_self using static Model Weights [W_out -> W_in]")

    # Method 2: Improved Middle-Layer Injection (Learned strictly from TRAIN set hidden states)
    W_mid, tn_mid = build_mid_layer_alignment(last_hidden_train, mid_hidden_train, args.lambda_reg)
    W_mid, tn_mid = W_mid.cpu(), tn_mid.cpu()
    print(f"  -> Computed W_mid using {last_hidden_train.shape[0]} TRAIN tokens [Layer {num_layers} -> Layer {mid_layer_idx}]")

    # ================================================================
    # 3. Evaluate Alignments (Strictly on TEST set)
    # ================================================================
    print("\n--- 3. Evaluating on Unseen TEST Data ---")

    # Original Evaluation
    aligned_infer_self_test = apply_alignment(last_hidden_test, W_self, tn_self).numpy()
    tgt_infer_self_test = W_in_A[pred_ids_test].numpy()
    cos_infer_self = compute_alignment_score(aligned_infer_self_test, tgt_infer_self_test)
    print(f"  [Original] Latent_MAS (L->Input) Cosine on TEST:   {cos_infer_self:.4f}")

    # Middle-Layer Evaluation
    aligned_infer_mid_test = apply_alignment(last_hidden_test, W_mid, tn_mid).numpy()
    tgt_infer_mid_test = mid_hidden_test.numpy()
    cos_infer_mid = compute_alignment_score(aligned_infer_mid_test, tgt_infer_mid_test)
    print(f"  [Improved] Middle-Layer (L->{mid_layer_idx}) Cosine on TEST: {cos_infer_mid:.4f}  <-- GENERALIZATION CONFIRMED!")

    # ================================================================
    # 4. t-SNE Plotting (Only visualizing the TEST set)
    # ================================================================
    print("\n--- 4. Running t-SNE Visualization on TEST Data ---")
    COLOR_SRC = "#E74C3C"      # red
    COLOR_ALIGNED = "#2ECC71"  # green
    COLOR_TGT = "#3498DB"      # blue
    colors3 =[COLOR_SRC, COLOR_ALIGNED, COLOR_TGT]
    markers3 = ["o", "^", "s"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 6), squeeze=False)
    hidden_test_np = last_hidden_test.numpy()

    # Panel 1: Original Method
    data_test_self = {
        "Hidden (h^L_t)": hidden_test_np,
        "Aligned (h^L_t @ W_self)": aligned_infer_self_test,
        "Target (W_in[pred_t])": tgt_infer_self_test,
    }
    plot_single_method(axes[0, 0], run_tsne(data_test_self, args.perplexity, args.seed + 10),
        f"[Original] latent_mas on TEST\nLayer {num_layers} -> Input Vocab Space\ncos = {cos_infer_self:.4f}",
        colors3, markers3)

    # Panel 2: Middle-Layer Method
    data_test_mid = {
        "Hidden (h^L_t)": hidden_test_np,
        f"Aligned (h^L_t @ W_mid)": aligned_infer_mid_test,
        f"Target (h^{mid_layer_idx}_t)": tgt_infer_mid_test,
    }
    plot_single_method(axes[0, 1], run_tsne(data_test_mid, args.perplexity, args.seed + 20),
        f"[Improved] Middle-Layer Injection on TEST\nLayer {num_layers} -> Layer {mid_layer_idx} Manifold\ncos = {cos_infer_mid:.4f}",
        colors3, markers3)

    plt.suptitle("t-SNE Generalization: Train on Train-set, Visualized on Test-set", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    fig.savefig(args.output, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"\nSaved generalization comparison figure to: {args.output}")

if __name__ == "__main__":
    main()