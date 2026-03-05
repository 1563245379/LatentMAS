"""
t-SNE Visualization of Alignment Quality for LatentMAS Methods
====================================================================

This script rigorously evaluates 3 alignment strategies:
1. [Original] Linear Parameter Space Alignment
2. [Improved] Linear Activation Manifold Alignment (Data-Driven Ridge Regression)
3. [Advanced] Non-Linear Activation Manifold Alignment (Data-Driven MLP Adapter)

Usage
-----
    python scripts/tsne_alignment_nonlinear.py \
        --model_a Qwen/Qwen3-4B \
        --n_train 1000 \
        --n_test 50 \
        --mid_layer_idx 2 \
        --mlp_epochs 100 \
        --perplexity 30 \
        --output figures/tsne_autoregressive_nonlinear.png
"""

import argparse
import os
import sys
import warnings

import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from datasets import load_dataset
from sklearn.manifold import TSNE

# ── add project root to path ─────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from transformers import AutoModelForCausalLM, AutoTokenizer

warnings.filterwarnings("ignore", category=FutureWarning)


# =========================================================================
# 1. Linear Alignment Builders
# =========================================================================

def build_self_alignment(model, lambda_reg: float = 1e-5):
    """Method 1: latent_mas (Original) - Static Parameter Space"""
    W_out = model.get_output_embeddings().weight.detach().float()  
    W_in = model.get_input_embeddings().weight.detach().float()    
    d = W_out.shape[1]
    gram = W_out.T @ W_out + lambda_reg * torch.eye(d, device=W_out.device)
    rhs = W_out.T @ W_in
    W_align = torch.linalg.solve(gram, rhs)  
    target_norm = W_in.norm(dim=1).mean()
    return W_align, target_norm


def build_mid_layer_alignment_autoregressive(source_h_L: torch.Tensor, target_h_K: torch.Tensor, lambda_reg: float = 1e-4):
    """Method 2: Linear Data-Driven - Ridge Regression"""
    d = source_h_L.shape[1]
    gram = source_h_L.T @ source_h_L + lambda_reg * torch.eye(d, device=source_h_L.device)
    rhs = source_h_L.T @ target_h_K
    W_mid = torch.linalg.solve(gram, rhs)  
    target_norm_train = target_h_K.norm(dim=1).mean() 
    return W_mid, target_norm_train


def apply_linear_alignment(hidden: torch.Tensor, W_align: torch.Tensor, target_norm: torch.Tensor) -> torch.Tensor:
    """Project hidden through linear alignment matrix."""
    aligned = (hidden.float() @ W_align)
    norms = aligned.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    aligned = aligned * (target_norm / norms)
    return aligned


# =========================================================================
# 2. Non-Linear Alignment Builders (New)
# =========================================================================

class NonLinearAdapter(nn.Module):
    """A lightweight MLP to bridge the non-linear semantic gap."""
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 2 * d),
            nn.SiLU(),  # Standard modern LLM activation
            nn.Linear(2 * d, 2 * d),
            nn.SiLU(),
            nn.Linear(2 * d, d)
        )

    def forward(self, x):
        return self.net(x)

def build_nonlinear_alignment_autoregressive(source_train, target_train, epochs=50, lr=1e-3, batch_size=1024,
                                              source_val=None, target_val=None):
    """Method 3: Non-Linear Data-Driven - MLP Adapter Training"""
    device = source_train.device
    d = source_train.shape[-1]
    
    adapter = NonLinearAdapter(d).to(device)
    optimizer = optim.AdamW(adapter.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()

    dataset = TensorDataset(source_train.float(), target_train.float())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    has_val = source_val is not None and target_val is not None
    if has_val:
        source_val = source_val.float().to(device)
        target_val = target_val.float().to(device)

    print(f"    [MLP Training] Started for {epochs} epochs{'  (with val set)' if has_val else ''}...")
    adapter.train()

    for epoch in tqdm.tqdm(range(epochs)):
        # ---- train step ----
        adapter.train()
        total_loss = 0
        for x, y in loader:
            optimizer.zero_grad()
            pred = adapter(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            avg_train_loss = total_loss / len(loader)
            if has_val:
                adapter.eval()
                with torch.no_grad():
                    val_loss = criterion(adapter(source_val), target_val).item()
                print(f"      Epoch {epoch+1:03d}/{epochs} | Train MSE: {avg_train_loss:.6f} | Val MSE: {val_loss:.6f}")
            else:
                print(f"      Epoch {epoch+1:03d}/{epochs} | Train MSE: {avg_train_loss:.6f}")
    
    adapter.eval()
    target_norm_train = target_train.norm(dim=1).mean()
    return adapter, target_norm_train

def apply_nonlinear_alignment(hidden: torch.Tensor, adapter: nn.Module, target_norm: torch.Tensor) -> torch.Tensor:
    """Project hidden through the trained MLP adapter."""
    with torch.no_grad():
        aligned = adapter(hidden.float())
    norms = aligned.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    aligned = aligned * (target_norm / norms)
    return aligned


# =========================================================================
# Dataset & Inference helpers
# =========================================================================

def load_gsm8k_splits(n_train: int, n_test: int, seed: int = 42):
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
    all_source_L, all_target_K, all_target_e_ids = [], [],[]
    
    for i, q in enumerate(questions):
        inputs = tokenizer(q, return_tensors="pt").to(device)
        outputs = model(**inputs, output_hidden_states=True)
        
        h_L = outputs.hidden_states[-1][0].cpu().float()       
        h_K = outputs.hidden_states[mid_layer_idx][0].cpu().float() 
        
        logits = outputs.logits[0].cpu().float()
        pred_ids = logits.argmax(dim=-1) 
        
        seq_len = h_L.shape[0]
        if seq_len < 2:
            continue
            
        # Temporal shift (t -> t+1)
        all_source_L.append(h_L[:-1])           # h[0...T-1]
        all_target_K.append(h_K[1:])            # h[1...T]
        all_target_e_ids.append(pred_ids[:-1])  
        
        if (i + 1) % 10 == 0 or i == len(questions) - 1:
            total_tokens = sum(x.shape[0] for x in all_source_L)
            print(f"    [{desc}] Processed {i+1}/{len(questions)} questions. Collected {total_tokens} shifted pairs.")
            
    return torch.cat(all_source_L, dim=0), torch.cat(all_target_K, dim=0), torch.cat(all_target_e_ids, dim=0)


def load_model_and_tokenizer(model_name: str, device: str = "cpu"):
    print(f"Loading model: {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32, device_map=device)
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

    tsne = TSNE(n_components=2, perplexity=min(perplexity, max(2, combined.shape[0] // 4)),
                random_state=seed, init="pca", learning_rate="auto", max_iter=1000)
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
    ax.legend(fontsize=8, loc="lower left", framealpha=0.9)
    ax.set_xticks([]); ax.set_yticks([])


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_a", type=str, required=True)
    p.add_argument("--n_train", type=int, default=150)
    p.add_argument("--n_test", type=int, default=30)
    p.add_argument("--mid_layer_idx", type=int, default=0)
    p.add_argument("--mlp_epochs", type=int, default=50, help="Epochs for training the non-linear MLP adapter")
    p.add_argument("--perplexity", type=float, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lambda_reg", type=float, default=1e-5)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--output", type=str, default="figures/tsne_autoregressive_nonlinear.png")
    return p.parse_args()


def main():
    args = parse_args()
    rng = np.random.RandomState(args.seed)
    torch.manual_seed(args.seed)

    print(f"\n{'='*75}")
    print(f"  t-SNE Alignment: Linear vs Non-Linear Autoregressive Generalization")
    print(f"  Train samples: {args.n_train} | Test samples: {args.n_test}")
    print(f"{'='*75}")

    model_a, tok_a = load_model_and_tokenizer(args.model_a, args.device)
    num_layers = getattr(model_a.config, "num_hidden_layers", 28)
    mid_layer_idx = args.mid_layer_idx
    print(f"  Mid-layer target: Layer {mid_layer_idx} (out of {num_layers}).")

    # ================================================================
    # 1. Collect Data (Strict Train/Test Split + Temporal Shift)
    # ================================================================
    print("\n--- 1. Collecting Data (Temporal Shift: h_t -> h_t+1) ---")
    q_train, q_test = load_gsm8k_splits(args.n_train, args.n_test, args.seed)
    
    src_L_train, tgt_K_train, _ = collect_shifted_inference_vectors(
        model_a, tok_a, q_train, mid_layer_idx, args.device, desc="TRAIN")
    
    src_L_test, tgt_K_test, tgt_e_ids_test = collect_shifted_inference_vectors(
        model_a, tok_a, q_test, mid_layer_idx, args.device, desc="TEST")

    W_in_A = model_a.get_input_embeddings().weight.detach().cpu().float()

    # ================================================================
    # 2. Compute/Train Alignments
    # ================================================================
    print("\n--- 2. Computing/Training Alignment Matrices ---")
    
    # Method 1: Original LatentMAS (Static Weights)
    W_self, tn_self = build_self_alignment(model_a, args.lambda_reg)
    W_self, tn_self = W_self.cpu(), tn_self.cpu()
    print(f"  -> [Original] Computed Linear W_self (vocab space)")

    # Method 2: Linear Data-Driven
    W_mid_lin, tn_mid_lin = build_mid_layer_alignment_autoregressive(src_L_train, tgt_K_train, args.lambda_reg)
    W_mid_lin, tn_mid_lin = W_mid_lin.cpu(), tn_mid_lin.cpu()
    print(f"  -> [Improved] Computed Linear W_mid (data-driven ridge regression)")

    # Method 3: Non-Linear Data-Driven (MLP)
    mlp_device = args.device
    mlp_adapter, tn_mid_mlp = build_nonlinear_alignment_autoregressive(
        src_L_train.to(mlp_device), tgt_K_train.to(mlp_device), epochs=args.mlp_epochs, batch_size=2048,
        source_val=src_L_test.to(mlp_device), target_val=tgt_K_test.to(mlp_device))
    mlp_adapter = mlp_adapter.cpu()
    tn_mid_mlp = tn_mid_mlp.cpu()
    print(f"  -> [Advanced] Trained Non-Linear MLP Adapter")

    # ================================================================
    # 3. Evaluate Alignments (Strictly on TEST set)
    # ================================================================
    print("\n--- 3. Evaluating on Unseen TEST Data ---")

    source_test_np = src_L_test.cpu().numpy()

    # 1. Original
    aligned_self_test = apply_linear_alignment(src_L_test.cpu(), W_self, tn_self).numpy()
    target_e_test = W_in_A[tgt_e_ids_test].numpy()
    cos_self = compute_alignment_score(aligned_self_test, target_e_test)
    print(f"  [1. Original] Linear (h_t^L -> e_t+1) Cosine on TEST:    {cos_self:.4f}")

    # 2. Linear Mid-Layer
    aligned_mid_lin_test = apply_linear_alignment(src_L_test.cpu(), W_mid_lin, tn_mid_lin).numpy()
    target_mid_test = tgt_K_test.cpu().numpy()
    cos_mid_lin = compute_alignment_score(aligned_mid_lin_test, target_mid_test)
    print(f"[2. Improved] Linear (h_t^L -> h_t+1^{mid_layer_idx}) Cosine on TEST: {cos_mid_lin:.4f}")

    # 3. Non-Linear Mid-Layer
    aligned_mid_mlp_test = apply_nonlinear_alignment(src_L_test.cpu(), mlp_adapter, tn_mid_mlp).numpy()
    cos_mid_mlp = compute_alignment_score(aligned_mid_mlp_test, target_mid_test)
    print(f"  [3. Advanced] Non-Lin (h_t^L -> h_t+1^{mid_layer_idx}) Cosine on TEST: {cos_mid_mlp:.4f} <-- EXPECTED PEAK!")

    # ================================================================
    # 4. t-SNE Plotting (1x3 Panels)
    # ================================================================
    print("\n--- 4. Running t-SNE Visualization on TEST Data ---")
    COLOR_SRC = "#E74C3C"      # red (Source)
    COLOR_ALIGNED = "#2ECC71"  # green (Aligned)
    COLOR_TGT = "#3498DB"      # blue (Target)
    colors3 =[COLOR_SRC, COLOR_ALIGNED, COLOR_TGT]
    markers3 = ["o", "^", "s"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 6), squeeze=False)

    # Panel 1: Original Linear
    data_test_self = {
        "Source (h_t^L)": source_test_np,
        "Aligned (Linear W_self)": aligned_self_test,
        "Target (e_t+1)": target_e_test,
    }
    plot_single_method(axes[0, 0], run_tsne(data_test_self, args.perplexity, args.seed + 10),
        f"[Original] Linear Parameter Space\nh_t^(L) -> e_t+1\nCosine = {cos_self:.4f}",
        colors3, markers3)

    # Panel 2: Improved Linear
    data_test_mid_lin = {
        "Source (h_t^L)": source_test_np,
        f"Aligned (Linear W_mid)": aligned_mid_lin_test,
        f"Target (h_t+1^{mid_layer_idx})": target_mid_test,
    }
    plot_single_method(axes[0, 1], run_tsne(data_test_mid_lin, args.perplexity, args.seed + 20),
        f"[Improved] Linear Data-Driven\nh_t^(L) -> h_t+1^({mid_layer_idx})\nCosine = {cos_mid_lin:.4f}",
        colors3, markers3)

    # Panel 3: Advanced Non-Linear
    data_test_mid_mlp = {
        "Source (h_t^L)": source_test_np,
        f"Aligned (MLP Adapter)": aligned_mid_mlp_test,
        f"Target (h_t+1^{mid_layer_idx})": target_mid_test,
    }
    plot_single_method(axes[0, 2], run_tsne(data_test_mid_mlp, args.perplexity, args.seed + 30),
        f"[Advanced] Non-Linear MLP Adapter\nh_t^(L) -> h_t+1^({mid_layer_idx})\nCosine = {cos_mid_mlp:.4f}",
        colors3, markers3)

    plt.suptitle("t-SNE Generalization: Breaking the Non-Linear Semantic Gap (Test Set)", 
                 fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    fig.savefig(args.output, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"\nSaved 3-panel comparison figure to: {args.output}")

if __name__ == "__main__":
    main()