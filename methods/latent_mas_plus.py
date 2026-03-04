from typing import Dict, List, Optional, Tuple

from . import default_agents
from models import ModelWrapper, _past_length
from prompts import build_agent_message_sequential_latent_mas, build_agent_message_hierarchical_latent_mas
from utils import extract_gsm8k_answer, normalize_answer, extract_markdown_python_block, run_with_timeout
import torch
import argparse

try:
    from vllm import SamplingParams
except:
    print("vLLM not installed, may be fine unless vLLM use required.")

import pdb
from tqdm import tqdm

try:
    from transformers.cache_utils import Cache
except ImportError:
    Cache = None


# ---------------------------------------------------------------------------
# Step 1: Tokenizer Intersection Extraction
# ---------------------------------------------------------------------------
def compute_tokenizer_intersection(
    tokenizer_a,
    tokenizer_b,
) -> Tuple[List[int], List[int], int]:
    """
    Identify the shared semantic space (V^I = V^A ∩ V^B) between two tokenizers.

    For each tokenizer, extract all string tokens and their IDs, then find exact
    string matches.  Return two aligned index lists so that
        tokenizer_a.convert_ids_to_tokens(indices_a[i])
        == tokenizer_b.convert_ids_to_tokens(indices_b[i])
    for every i.

    Returns:
        indices_a: list of token IDs in tokenizer A for shared tokens
        indices_b: list of token IDs in tokenizer B for shared tokens
        n_intersect: size of the intersection set
    """
    # Build token-string → id mappings for both tokenizers
    vocab_a: Dict[str, int] = tokenizer_a.get_vocab()  # {str: id}
    vocab_b: Dict[str, int] = tokenizer_b.get_vocab()

    # Intersection of string keys
    shared_tokens = set(vocab_a.keys()) & set(vocab_b.keys())

    # Build aligned index lists (deterministic sorted order)
    indices_a: List[int] = []
    indices_b: List[int] = []
    for token_str in sorted(shared_tokens):
        indices_a.append(vocab_a[token_str])
        indices_b.append(vocab_b[token_str])

    n_intersect = len(indices_a)
    assert len(indices_a) == len(indices_b), (
        f"Intersection index lists must have equal length, got {len(indices_a)} vs {len(indices_b)}"
    )
    print(f"[LatentMAS+] Tokenizer intersection: |V^A|={len(vocab_a)}, |V^B|={len(vocab_b)}, "
          f"|V^I|={n_intersect} ({100*n_intersect/max(len(vocab_a),len(vocab_b)):.1f}% of larger vocab)")
    return indices_a, indices_b, n_intersect


# ---------------------------------------------------------------------------
# Step 2 & 3: Weight Extraction, Sub-setting & Alignment Matrix Computation
# ---------------------------------------------------------------------------
def compute_intersection_alignment_matrix(
    model_a: torch.nn.Module,
    model_b: torch.nn.Module,
    tokenizer_a,
    tokenizer_b,
    lambda_val: float = 1e-4,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the cross-model alignment matrix W_a using vocabulary intersection
    and Ridge Regression (Workflow.md Steps 1-3).

    The closed-form solution is:
        W_a = (W_out_intersect^T @ W_out_intersect + λI)^{-1}
              @ W_out_intersect^T @ W_in_intersect

    where W_out_intersect and W_in_intersect are sliced from the LM Head of
    Model A and the Input Embedding of Model B using the intersection indices.

    Args:
        model_a: Source model (sender) — needs .lm_head or get_output_embeddings()
        model_b: Target model (receiver) — needs .model.embed_tokens or get_input_embeddings()
        tokenizer_a: Tokenizer for model A
        tokenizer_b: Tokenizer for model B
        lambda_val: Ridge regularization coefficient
        device: Device for computation (defaults to model_a's device)

    Returns:
        W_a: Alignment matrix of shape [d_out, d_in]
        target_norm: Mean L2 norm of Model B's intersection input embeddings (for scaling)
    """
    # --- Step 1: Tokenizer intersection ---
    indices_a, indices_b, n_intersect = compute_tokenizer_intersection(tokenizer_a, tokenizer_b)
    if n_intersect == 0:
        raise RuntimeError("No shared tokens between the two tokenizers — cannot compute alignment.")

    indices_a_t = torch.tensor(indices_a, dtype=torch.long)
    indices_b_t = torch.tensor(indices_b, dtype=torch.long)

    # --- Step 2: Extract & slice weight matrices ---
    # W_out_A = model_a.lm_head.weight  (shape [N_A, d_out])
    output_embeds_a = (
        model_a.get_output_embeddings()
        if hasattr(model_a, "get_output_embeddings") and model_a.get_output_embeddings() is not None
        else getattr(model_a, "lm_head", None)
    )
    if output_embeds_a is None or not hasattr(output_embeds_a, "weight"):
        raise RuntimeError("Cannot access LM Head weight of Model A.")
    W_out_A = output_embeds_a.weight.detach()  # [N_A, d_out]

    # W_in_B = model_b.model.embed_tokens.weight  (shape [N_B, d_in])
    input_embeds_b = (
        model_b.get_input_embeddings()
        if hasattr(model_b, "get_input_embeddings") and model_b.get_input_embeddings() is not None
        else None
    )
    if input_embeds_b is None or not hasattr(input_embeds_b, "weight"):
        raise RuntimeError("Cannot access Input Embedding weight of Model B.")
    W_in_B = input_embeds_b.weight.detach()  # [N_B, d_in]

    d_out = W_out_A.shape[1]
    d_in = W_in_B.shape[1]

    # Move to computation device
    comp_device = device or W_out_A.device
    indices_a_t = indices_a_t.to(comp_device)
    indices_b_t = indices_b_t.to(comp_device)

    # Slice intersection rows (Workflow.md Step 2)
    W_out_intersect = W_out_A.to(device=comp_device, dtype=torch.float32)[indices_a_t]  # [N_intersect, d_out]
    W_in_intersect = W_in_B.to(device=comp_device, dtype=torch.float32)[indices_b_t]    # [N_intersect, d_in]

    # --- Step 3: Ridge regression closed-form solution ---
    # Gram = W_out_intersect^T @ W_out_intersect   [d_out, d_out]
    Gram = torch.matmul(W_out_intersect.T, W_out_intersect)
    # Gram_reg = Gram + λI
    Gram_reg = Gram + lambda_val * torch.eye(d_out, device=comp_device, dtype=torch.float32)
    # Target = W_out_intersect^T @ W_in_intersect   [d_out, d_in]
    Target = torch.matmul(W_out_intersect.T, W_in_intersect)
    # W_a = solve(Gram_reg, Target)   [d_out, d_in]
    W_a = torch.linalg.solve(Gram_reg, Target)

    assert W_a.shape == (d_out, d_in), (
        f"W_a shape mismatch: expected ({d_out}, {d_in}), got {W_a.shape}"
    )

    # Compute target norm for runtime scaling
    target_norm = W_in_intersect.norm(dim=1).mean().detach()

    # --- Verification (Workflow.md Quality Check 3) ---
    _verify_alignment(W_out_intersect, W_in_intersect, W_a, n_samples=5)

    print(f"[LatentMAS+] Alignment matrix W_a computed: shape={W_a.shape}, "
          f"d_out={d_out}, d_in={d_in}, target_norm={target_norm.item():.4f}")

    return W_a, target_norm


def _verify_alignment(
    W_out_intersect: torch.Tensor,
    W_in_intersect: torch.Tensor,
    W_a: torch.Tensor,
    n_samples: int = 5,
) -> None:
    """
    Verification: for randomly selected shared tokens, check that
    cosine_similarity(W_out_A[t] @ W_a, W_in_B[t]) ≈ 1.0
    """
    n = W_out_intersect.shape[0]
    if n == 0:
        return
    sample_indices = torch.randperm(n)[:min(n_samples, n)]
    cos = torch.nn.CosineSimilarity(dim=0)
    sims = []
    for idx in sample_indices:
        projected = W_out_intersect[idx] @ W_a   # [d_in]
        target = W_in_intersect[idx]               # [d_in]
        sim = cos(projected, target).item()
        sims.append(sim)
    avg_sim = sum(sims) / len(sims)
    print(f"[LatentMAS+] Alignment verification: avg cosine similarity = {avg_sim:.6f} "
          f"(sampled {len(sims)} tokens, range [{min(sims):.4f}, {max(sims):.4f}])")


# ---------------------------------------------------------------------------
# Step 4: Runtime Transfer Function
# ---------------------------------------------------------------------------
def transfer_via_intersection_alignment(
    hidden_states: torch.Tensor,
    W_a: torch.Tensor,
    target_norm: torch.Tensor,
) -> torch.Tensor:
    """
    Transfer hidden states from Model A to Model B's embedding space using the
    precomputed intersection alignment matrix (Workflow.md Step 4).

    h_aligned = h_A @ W_a

    Then normalize to match the target embedding scale.

    Args:
        hidden_states: [batch, seq_len, d_out] — hidden states from Model A
        W_a: [d_out, d_in] — precomputed alignment matrix
        target_norm: scalar — mean L2 norm of Model B's input embeddings

    Returns:
        embeddings_B: [batch, seq_len, d_in] — input embeddings for Model B
    """
    original_dtype = hidden_states.dtype
    batch_size, seq_len, d_out = hidden_states.shape

    # Project: h_aligned = h_A @ W_a   →  [batch, seq_len, d_in]
    h_flat = hidden_states.reshape(-1, d_out).float()
    W_a_dev = W_a.to(device=h_flat.device, dtype=torch.float32)
    h_aligned_flat = torch.matmul(h_flat, W_a_dev)

    # Normalize to target embedding scale
    target_norm_dev = target_norm.to(device=h_aligned_flat.device, dtype=torch.float32)
    current_norms = h_aligned_flat.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    h_aligned_flat = h_aligned_flat * (target_norm_dev / current_norms)

    d_in = W_a.shape[1]
    h_aligned = h_aligned_flat.reshape(batch_size, seq_len, d_in).to(original_dtype)
    return h_aligned


# ---------------------------------------------------------------------------
# Alignment Matrix Cache (keyed by model pair)
# ---------------------------------------------------------------------------
_alignment_cache: Dict[Tuple[str, str], Tuple[torch.Tensor, torch.Tensor]] = {}


def get_or_compute_alignment(
    model_from: ModelWrapper,
    model_to: ModelWrapper,
    lambda_val: float = 1e-4,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Return cached (W_a, target_norm) for a (model_from, model_to) pair, or
    compute and cache it.
    """
    key = (model_from.model_name, model_to.model_name)
    if key in _alignment_cache:
        W_a, target_norm = _alignment_cache[key]
        target_device = device or torch.device("cpu")
        return W_a.to(target_device), target_norm.to(target_device)

    # Access the underlying HuggingFace model
    hf_model_a = getattr(model_from, "HF_model", None) or model_from.model
    hf_model_b = getattr(model_to, "HF_model", None) or model_to.model

    W_a, target_norm = compute_intersection_alignment_matrix(
        hf_model_a,
        hf_model_b,
        model_from.tokenizer,
        model_to.tokenizer,
        lambda_val=lambda_val,
        device=device,
    )
    _alignment_cache[key] = (W_a.cpu(), target_norm.cpu())
    return W_a, target_norm


# ===========================================================================
# Main Class: LatentMASPlus
# ===========================================================================
class LatentMASPlus:
    """
    Latent Multi-Agent System with Training-Free Heterogeneous Latent
    Communication Bridge.

    Extends LatentMAS to support heterogeneous models (different architectures,
    vocabularies, and hidden dimensions) by:
        1. Computing tokenizer intersection between Model A and Model B.
        2. Extracting and sub-setting LM-Head / Input-Embedding weights to the
           intersection vocabulary.
        3. Solving a Ridge Regression to obtain an optimal alignment matrix W_a.
        4. At runtime, projecting Model A's hidden states through W_a and
           injecting as inputs_embeds into Model B.

    See Workflow.md and Theory.md for the full mathematical derivation.
    """

    def __init__(
        self,
        model: ModelWrapper,
        *,
        agent_models: Optional[List[str]] = None,
        latent_steps: int = 10,
        judger_max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        generate_bs: int = 1,
        args: argparse.Namespace = None,
    ) -> None:
        self.args = args
        self.initial_model = model
        self.latent_steps = latent_steps
        self.judger_max_new_tokens = judger_max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.generate_bs = max(1, generate_bs)
        self.agents = getattr(args, "custom_agents", None) or default_agents()
        self.method_name = "latent_mas_plus"
        self.vllm_device = args.device
        self.HF_device = args.device2
        self.latent_only = bool(getattr(args, "latent_only", False)) if args else False
        self.sequential_info_only = bool(getattr(args, "sequential_info_only", False)) if args else False
        self.first_agent_text = bool(getattr(args, "first_agent_text", False)) if args else False

        if self.latent_only:
            self.sequential_info_only = True

        try:
            self.sampling_params = SamplingParams(
                temperature=temperature,
                top_p=top_p,
                max_tokens=args.max_new_tokens,
            )
        except Exception:
            self.sampling_params = None

        self.task = args.task

        # --- Agent-to-model mapping ---
        if agent_models is None:
            self.agent_models = [model.model_name] * len(self.agents)
        else:
            assert len(agent_models) == len(self.agents), \
                f"Must specify one model per agent, got {len(agent_models)} for {len(self.agents)} agents"
            self.agent_models = agent_models

        # Load all unique models
        self.models: Dict[str, ModelWrapper] = {model.model_name: model}
        self._load_additional_models()
        self.model = model  # default / initial model

        # --- Pre-compute alignment matrices for all model pairs ---
        self._precompute_alignments()

    def _load_additional_models(self) -> None:
        """Load any models needed by agents that aren't already loaded."""
        unique_models = set(self.agent_models)
        for model_name in unique_models:
            if model_name not in self.models:
                print(f"[LatentMAS+] Loading additional model: {model_name}")
                new_model = ModelWrapper(
                    model_name,
                    self.vllm_device,
                    use_vllm=False,
                    args=self.args,
                )
                self.models[model_name] = new_model

    def _precompute_alignments(self) -> None:
        """
        Pre-compute alignment matrices for every consecutive model pair in
        the agent pipeline.  This is a one-time offline cost (Workflow.md Step 3).
        """
        for i in range(len(self.agent_models) - 1):
            src_name = self.agent_models[i]
            dst_name = self.agent_models[i + 1]
            if src_name == dst_name:
                continue  # same model, no alignment needed
            src_model = self.models[src_name]
            dst_model = self.models[dst_name]
            print(f"[LatentMAS+] Pre-computing alignment: {src_name} -> {dst_name}")
            get_or_compute_alignment(
                src_model,
                dst_model,
                lambda_val=float(getattr(self.args, "lambda_val", 1e-5)),
                device=torch.device(self.vllm_device),
            )

    # -------------------------------------------------------------------
    # Hidden-state capture (same-model latent generation)
    # -------------------------------------------------------------------
    def _capture_hidden_states_from_model(
        self,
        agent_model: ModelWrapper,
        wrapped_ids: torch.Tensor,
        wrapped_mask: torch.Tensor,
        past_kv: Optional[Tuple],
        latent_steps: int,
    ) -> Tuple[Tuple, torch.Tensor]:
        """
        Run latent generation on *agent_model* and capture RAW hidden states
        (before self-alignment).

        Returns:
            past_kv: Updated KV cache
            raw_latent_hidden_states: [batch, latent_steps, hidden_dim]
        """
        input_ids = wrapped_ids.to(agent_model.device)
        attention_mask = wrapped_mask.to(agent_model.device)

        if past_kv is not None:
            past_len = _past_length(past_kv)
            if past_len > 0:
                past_mask = torch.ones(
                    (attention_mask.shape[0], past_len),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                attention_mask = torch.cat([past_mask, attention_mask], dim=-1)

        # Initial forward pass
        outputs = agent_model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_kv,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        past = outputs.past_key_values
        last_hidden = outputs.hidden_states[-1][:, -1, :]  # [B, hidden_dim]

        raw_latent_hidden_list: List[torch.Tensor] = []
        for _ in range(latent_steps):
            raw_latent_hidden_list.append(last_hidden.unsqueeze(1))  # [B, 1, D]

            # Self-alignment for feeding back into the same model
            latent_vec = agent_model._apply_latent_realignment(last_hidden, agent_model.model)
            latent_embed = latent_vec.unsqueeze(1)

            past_len = _past_length(past)
            latent_mask = torch.ones(
                (latent_embed.shape[0], past_len + 1),
                dtype=torch.long,
                device=latent_embed.device,
            )
            outputs = agent_model.model(
                inputs_embeds=latent_embed,
                attention_mask=latent_mask,
                past_key_values=past,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
            past = outputs.past_key_values
            last_hidden = outputs.hidden_states[-1][:, -1, :]

        if latent_steps > 0:
            raw_latent_hidden_states = torch.cat(raw_latent_hidden_list, dim=1)
        else:
            batch_size = wrapped_ids.shape[0]
            hidden_dim = last_hidden.shape[-1]
            raw_latent_hidden_states = torch.zeros(
                (batch_size, 0, hidden_dim), device=last_hidden.device, dtype=last_hidden.dtype
            )

        return past, raw_latent_hidden_states

    # -------------------------------------------------------------------
    # Cross-model transfer using intersection alignment
    # -------------------------------------------------------------------
    def _transfer_hidden_states(
        self,
        hidden_states: torch.Tensor,
        model_from: ModelWrapper,
        model_to: ModelWrapper,
    ) -> torch.Tensor:
        """
        Transfer hidden states from model_from to model_to using the
        precomputed intersection alignment matrix (Workflow.md Step 4).

        Args:
            hidden_states: [B, L, d_out] raw hidden states from model_from
            model_from: source ModelWrapper
            model_to: target ModelWrapper

        Returns:
            embeddings_to: [B, L, d_in] input embeddings for model_to
        """
        W_a, target_norm = get_or_compute_alignment(
            model_from,
            model_to,
            lambda_val=float(getattr(self.args, "lambda_val", 1e-5)),
            device=hidden_states.device,
        )
        return transfer_via_intersection_alignment(hidden_states, W_a, target_norm)

    # -------------------------------------------------------------------
    # Utility: KV cache manipulation
    # -------------------------------------------------------------------
    @staticmethod
    def _slice_tensor(tensor: torch.Tensor, tokens_to_keep: int) -> torch.Tensor:
        if tokens_to_keep <= 0:
            return tensor[..., 0:0, :].contiguous()
        keep = min(tokens_to_keep, tensor.shape[-2])
        start = tensor.shape[-2] - keep
        return tensor[..., start:, :].contiguous()

    def _truncate_past(self, past_kv: Optional[Tuple], tokens_to_keep: int) -> Optional[Tuple]:
        if past_kv is None or tokens_to_keep <= 0:
            return None
        if Cache is not None and isinstance(past_kv, Cache):
            legacy = past_kv.to_legacy_cache()
            trimmed_legacy = tuple(
                tuple(self._slice_tensor(t, tokens_to_keep) for t in layer)
                for layer in legacy
            )
            return past_kv.__class__.from_legacy_cache(trimmed_legacy)
        trimmed_layers = []
        for layer in past_kv:
            if isinstance(layer, tuple):
                trimmed_layers.append(tuple(self._slice_tensor(t, tokens_to_keep) for t in layer))
            elif torch.is_tensor(layer):
                trimmed_layers.append(self._slice_tensor(layer, tokens_to_keep))
            else:
                trimmed_layers.append(layer)
        return tuple(trimmed_layers)

    # ===================================================================
    # run_batch  (HuggingFace Transformers backend)
    # ===================================================================
    @torch.no_grad()
    def run_batch(self, items: List[Dict]) -> List[Dict]:
        if len(items) > self.generate_bs:
            raise ValueError("Batch size exceeds configured generate_bs")

        batch_size = len(items)
        past_kv: Optional[Tuple] = None
        current_model_name: Optional[str] = None

        # Accumulated text prompts & latent hidden states
        cumulative_prompts: List[str] = ["" for _ in range(batch_size)]
        cumulative_latent_hiddens: Optional[torch.Tensor] = None

        agent_traces: List[List[Dict]] = [[] for _ in range(batch_size)]
        final_texts = ["" for _ in range(batch_size)]

        agent_pbar = tqdm(self.agents, desc="Agents", unit="agent")
        for agent_idx, agent in enumerate(agent_pbar):
            agent_pbar.set_description(f"Agent: {agent.name} ({agent.role})")
            is_first_agent = (agent_idx == 0)
            is_last_agent = (agent_idx == len(self.agents) - 1)
            should_generate_text = is_first_agent and self.first_agent_text

            # Determine which model this agent uses
            agent_model_name = self.agent_models[agent_idx]
            agent_model = self.models[agent_model_name]

            model_switched = (
                current_model_name is not None
                and agent_model_name != current_model_name
            )
            if model_switched:
                print(f"\n[LatentMAS+] Model switch: {current_model_name} -> {agent_model_name}")
                if cumulative_latent_hiddens is not None:
                    print(f"[LatentMAS+] Transferring latent hiddens via intersection alignment "
                          f"(shape {cumulative_latent_hiddens.shape})")

            # Build prompts for this agent
            if self.args.prompt == "sequential":
                batch_messages = [
                    build_agent_message_sequential_latent_mas(
                        role=agent.role, question=item["question"], context="",
                        method=self.method_name, args=self.args,
                    )
                    for item in items
                ]
            elif self.args.prompt == "hierarchical":
                batch_messages = [
                    build_agent_message_hierarchical_latent_mas(
                        role=agent.role, question=item["question"], context="",
                        method=self.method_name, args=self.args,
                    )
                    for item in items
                ]

            prompts, input_ids, attention_mask, tokens_batch = agent_model.prepare_chat_batch(
                batch_messages, add_generation_prompt=True,
            )

            # ----------------------------------------------------------
            # Non-last agents: generate latent thoughts
            # ----------------------------------------------------------
            if not is_last_agent:
                if should_generate_text:
                    # First agent optionally generates text
                    if self.args.think:
                        first_agent_prompts = [f"{prompt}{self.args.think}" for prompt in prompts]
                    else:
                        first_agent_prompts = prompts

                    first_encoded = agent_model.tokenizer(
                        first_agent_prompts, return_tensors="pt", padding=True, add_special_tokens=False,
                    )
                    first_ids = first_encoded["input_ids"].to(agent_model.device)
                    first_mask = first_encoded["attention_mask"].to(agent_model.device)
                    first_tokens_batch: List[List[str]] = []
                    for ids_row, mask_row in zip(first_ids, first_mask):
                        active_ids = ids_row[mask_row.bool()].tolist()
                        first_tokens_batch.append(agent_model.tokenizer.convert_ids_to_tokens(active_ids))

                    generated_batch, past_kv = agent_model.generate_text_batch(
                        first_ids, first_mask,
                        max_new_tokens=self.judger_max_new_tokens,
                        temperature=self.temperature, top_p=self.top_p,
                        past_key_values=past_kv,
                    )
                    if current_model_name is None:
                        current_model_name = agent_model_name

                    for idx in range(batch_size):
                        text_out = generated_batch[idx].strip()
                        cumulative_prompts[idx] += first_agent_prompts[idx] + text_out
                        mask = first_mask[idx].bool()
                        trimmed_ids = first_ids[idx][mask].to("cpu").tolist()
                        agent_traces[idx].append({
                            "name": agent.name, "role": agent.role,
                            "input": first_agent_prompts[idx],
                            "input_ids": trimmed_ids,
                            "input_tokens": first_tokens_batch[idx],
                            "output": text_out,
                        })
                    continue

                # Standard latent generation
                prev_past_len = _past_length(past_kv)

                if self.args.think:
                    wrapped_prompts = [f"{prompt}{self.args.think}" for prompt in prompts]
                else:
                    wrapped_prompts = prompts

                wrapped_encoded = agent_model.tokenizer(
                    wrapped_prompts, return_tensors="pt", padding=True, add_special_tokens=False,
                )
                wrapped_ids = wrapped_encoded["input_ids"].to(agent_model.device)
                wrapped_mask = wrapped_encoded["attention_mask"].to(agent_model.device)
                wrapped_tokens_batch: List[List[str]] = []
                for ids_row, mask_row in zip(wrapped_ids, wrapped_mask):
                    active_ids = ids_row[mask_row.bool()].tolist()
                    wrapped_tokens_batch.append(agent_model.tokenizer.convert_ids_to_tokens(active_ids))

                # ---- Handle model switch via intersection alignment ----
                if model_switched and cumulative_latent_hiddens is not None:
                    prev_model = self.models[current_model_name]

                    # Re-encode accumulated text prompts with NEW model's tokenizer
                    prompt_encoded = agent_model.tokenizer(
                        cumulative_prompts, return_tensors="pt", padding=True, add_special_tokens=False,
                    )
                    prompt_ids = prompt_encoded["input_ids"].to(agent_model.device)
                    prompt_mask = prompt_encoded["attention_mask"].to(agent_model.device)

                    # Native Model B embeddings for prompts
                    with torch.no_grad():
                        prompt_embeds = agent_model.model.get_input_embeddings()(prompt_ids)

                    # Transfer latent hidden states via intersection alignment (Workflow.md Step 4)
                    transferred_latent_embeds = self._transfer_hidden_states(
                        cumulative_latent_hiddens, prev_model, agent_model,
                    )

                    # Concatenate [prompt_embeds, transferred_latent_embeds]
                    combined_embeds = torch.cat([prompt_embeds, transferred_latent_embeds], dim=1)
                    combined_mask = torch.cat([
                        prompt_mask,
                        torch.ones(
                            (batch_size, transferred_latent_embeds.shape[1]),
                            dtype=prompt_mask.dtype, device=prompt_mask.device,
                        ),
                    ], dim=1)

                    # Feed combined embeddings through Model B to build KV cache
                    with torch.no_grad():
                        transfer_outputs = agent_model.model(
                            inputs_embeds=combined_embeds,
                            attention_mask=combined_mask,
                            past_key_values=None,
                            use_cache=True,
                            return_dict=True,
                        )
                        transfer_past_kv = transfer_outputs.past_key_values

                    # Generate NEW latent thoughts with Model B
                    past_kv, new_latent_hiddens = self._capture_hidden_states_from_model(
                        agent_model, wrapped_ids, wrapped_mask, transfer_past_kv, self.latent_steps,
                    )
                    cumulative_latent_hiddens = new_latent_hiddens
                    current_model_name = agent_model_name

                elif model_switched:
                    # Model switch but no latent hiddens (e.g. first agent generated text)
                    prompt_encoded = agent_model.tokenizer(
                        cumulative_prompts, return_tensors="pt", padding=True, add_special_tokens=False,
                    )
                    prompt_ids = prompt_encoded["input_ids"].to(agent_model.device)
                    prompt_mask = prompt_encoded["attention_mask"].to(agent_model.device)
                    with torch.no_grad():
                        reenc_outputs = agent_model.model(
                            input_ids=prompt_ids,
                            attention_mask=prompt_mask,
                            past_key_values=None,
                            use_cache=True,
                            return_dict=True,
                        )
                        reenc_past_kv = reenc_outputs.past_key_values
                    past_kv, new_latent_hiddens = self._capture_hidden_states_from_model(
                        agent_model, wrapped_ids, wrapped_mask, reenc_past_kv, self.latent_steps,
                    )
                    cumulative_latent_hiddens = new_latent_hiddens
                    current_model_name = agent_model_name

                else:
                    # Same model or first agent — use KV cache directly
                    past_kv, new_latent_hiddens = self._capture_hidden_states_from_model(
                        agent_model, wrapped_ids, wrapped_mask, past_kv, self.latent_steps,
                    )
                    if cumulative_latent_hiddens is None:
                        cumulative_latent_hiddens = new_latent_hiddens
                    else:
                        cumulative_latent_hiddens = torch.cat(
                            [cumulative_latent_hiddens, new_latent_hiddens], dim=1,
                        )
                    if current_model_name is None:
                        current_model_name = agent_model_name

                # Update cumulative prompts
                for idx in range(batch_size):
                    cumulative_prompts[idx] += wrapped_prompts[idx]

                if self.sequential_info_only or self.latent_only:
                    new_past_len = _past_length(past_kv)
                    tokens_added = new_past_len - prev_past_len
                    tokens_to_keep = self.latent_steps if self.latent_only else tokens_added
                    past_kv = self._truncate_past(past_kv, tokens_to_keep)

                for idx in range(batch_size):
                    mask = wrapped_mask[idx].bool()
                    trimmed_ids = wrapped_ids[idx][mask].to("cpu").tolist()
                    agent_traces[idx].append({
                        "name": agent.name, "role": agent.role,
                        "input": wrapped_prompts[idx],
                        "input_ids": trimmed_ids,
                        "input_tokens": wrapped_tokens_batch[idx],
                        "latent_steps": self.latent_steps,
                        "output": "",
                    })

            # ----------------------------------------------------------
            # Last agent: generate final text output
            # ----------------------------------------------------------
            else:
                if model_switched and cumulative_latent_hiddens is not None:
                    prev_model = self.models[current_model_name]

                    prompt_encoded = agent_model.tokenizer(
                        cumulative_prompts, return_tensors="pt", padding=True, add_special_tokens=False,
                    )
                    prompt_ids = prompt_encoded["input_ids"].to(agent_model.device)
                    prompt_mask = prompt_encoded["attention_mask"].to(agent_model.device)
                    with torch.no_grad():
                        prompt_embeds = agent_model.model.get_input_embeddings()(prompt_ids)

                    transferred_latent_embeds = self._transfer_hidden_states(
                        cumulative_latent_hiddens, prev_model, agent_model,
                    )
                    combined_embeds = torch.cat([prompt_embeds, transferred_latent_embeds], dim=1)
                    combined_mask = torch.cat([
                        prompt_mask,
                        torch.ones(
                            (batch_size, transferred_latent_embeds.shape[1]),
                            dtype=prompt_mask.dtype, device=prompt_mask.device,
                        ),
                    ], dim=1)
                    with torch.no_grad():
                        transfer_outputs = agent_model.model(
                            inputs_embeds=combined_embeds,
                            attention_mask=combined_mask,
                            past_key_values=None,
                            use_cache=True,
                            return_dict=True,
                        )
                        past_for_decoding = transfer_outputs.past_key_values

                elif model_switched:
                    prompt_encoded = agent_model.tokenizer(
                        cumulative_prompts, return_tensors="pt", padding=True, add_special_tokens=False,
                    )
                    prompt_ids = prompt_encoded["input_ids"].to(agent_model.device)
                    prompt_mask = prompt_encoded["attention_mask"].to(agent_model.device)
                    with torch.no_grad():
                        reenc_outputs = agent_model.model(
                            input_ids=prompt_ids,
                            attention_mask=prompt_mask,
                            past_key_values=None,
                            use_cache=True,
                            return_dict=True,
                        )
                        past_for_decoding = reenc_outputs.past_key_values
                else:
                    past_for_decoding = past_kv if self.latent_steps > 0 else None

                if self.args.think:
                    final_agent_prompts = [f"{prompt}{self.args.think}" for prompt in prompts]
                else:
                    final_agent_prompts = prompts

                final_agent_encoded = agent_model.tokenizer(
                    final_agent_prompts, return_tensors="pt", padding=True, add_special_tokens=False,
                )
                final_agent_ids = final_agent_encoded["input_ids"].to(agent_model.device)
                final_agent_mask = final_agent_encoded["attention_mask"].to(agent_model.device)
                final_agent_tokens_batch: List[List[str]] = []
                for ids_row, mask_row in zip(final_agent_ids, final_agent_mask):
                    active_ids = ids_row[mask_row.bool()].tolist()
                    final_agent_tokens_batch.append(agent_model.tokenizer.convert_ids_to_tokens(active_ids))

                generated_batch, _ = agent_model.generate_text_batch(
                    final_agent_ids, final_agent_mask,
                    max_new_tokens=self.judger_max_new_tokens,
                    temperature=self.temperature, top_p=self.top_p,
                    past_key_values=past_for_decoding,
                )
                for idx in range(batch_size):
                    final_text = generated_batch[idx].strip()
                    final_texts[idx] = final_text
                    mask = final_agent_mask[idx].bool()
                    trimmed_ids = final_agent_ids[idx][mask].to("cpu").tolist()
                    agent_traces[idx].append({
                        "name": agent.name, "role": agent.role,
                        "input": final_agent_prompts[idx],
                        "input_ids": trimmed_ids,
                        "input_tokens": final_agent_tokens_batch[idx],
                        "output": final_text,
                    })

        # Assemble results
        results: List[Dict] = []
        for idx, item in enumerate(items):
            final_text = final_texts[idx]
            if self.task in ["mbppplus", "humanevalplus"]:
                pred = extract_markdown_python_block(final_text)
                gold = item.get("gold", "")
                if pred is None:
                    ok = False
                    error_msg = "python error: No python code block found"
                else:
                    python_code_to_exe = pred + "\n" + gold
                    ok, error_msg = run_with_timeout(python_code_to_exe, timeout=10)
                print(f"=========================================")
                print(f"Question {idx}")
                print(f"error_msg: {error_msg}")

            elif self.task in ["aime2024", "aime2025"]:
                pred = normalize_answer(extract_gsm8k_answer(final_text))
                gold = str(item.get("gold", "")).strip()
                try:
                    if pred is None or pred == "":
                        ok = False
                        error_msg = f"Failed to extract answer from: {final_text[:100]}..."
                    else:
                        pred_int = int(pred)
                        gold_int = int(gold)
                        ok = pred_int == gold_int
                        error_msg = None
                except ValueError:
                    ok = False
                    error_msg = f"Value error in parsing answer. Pred: {pred}, Gold: {gold}"
            else:
                pred = normalize_answer(extract_gsm8k_answer(final_text))
                gold = item.get("gold", "")
                ok = (pred == gold) if (pred and gold) else False
                error_msg = None

            results.append({
                "question": item["question"],
                "gold": gold,
                "solution": item["solution"],
                "prediction": pred,
                "raw_prediction": final_text,
                "agents": agent_traces[idx],
                "correct": ok,
            })
        return results

    # ===================================================================
    # run_batch_vllm  (vLLM backend)
    # ===================================================================
    @torch.no_grad()
    def run_batch_vllm(self, items: List[Dict]) -> List[Dict]:
        raise NotImplementedError("vLLM backend not yet implemented for LatentMASPlus")
        # if len(items) > self.generate_bs:
        #     raise ValueError("Batch size exceeds configured generate_bs")

        # batch_size = len(items)
        # past_kv: Optional[Tuple] = None
        # current_model_name: Optional[str] = None
        # agent_traces: List[List[Dict]] = [[] for _ in range(batch_size)]
        # final_texts = ["" for _ in range(batch_size)]

        # embedding_record: List[torch.Tensor] = []

        # agent_pbar = tqdm(self.agents, desc="Agents", unit="agent")
        # for agent_idx, agent in enumerate(agent_pbar):
        #     agent_pbar.set_description(f"Agent: {agent.name} ({agent.role})")
        #     is_last_agent = (agent_idx == len(self.agents) - 1)

        #     agent_model_name = self.agent_models[agent_idx]
        #     agent_model = self.models[agent_model_name]
        #     model_switched = (
        #         current_model_name is not None
        #         and agent_model_name != current_model_name
        #     )

        #     if self.args.prompt == "sequential":
        #         batch_messages = [
        #             build_agent_message_sequential_latent_mas(
        #                 role=agent.role, question=item["question"], context="",
        #                 method=self.method_name, args=self.args,
        #             )
        #             for item in items
        #         ]
        #     elif self.args.prompt == "hierarchical":
        #         batch_messages = [
        #             build_agent_message_hierarchical_latent_mas(
        #                 role=agent.role, question=item["question"], context="",
        #                 method=self.method_name, args=self.args,
        #             )
        #             for item in items
        #         ]

        #     prompts, input_ids, attention_mask, tokens_batch = self.model.prepare_chat_batch(
        #         batch_messages, add_generation_prompt=True,
        #     )

        #     if not is_last_agent:
        #         prev_past_len = _past_length(past_kv)

        #         if self.args.think:
        #             wrapped_prompts = [f"{prompt}{self.args.think}" for prompt in prompts]
        #         else:
        #             wrapped_prompts = prompts

        #         wrapped_encoded = self.model.tokenizer(
        #             wrapped_prompts, return_tensors="pt", padding=True, add_special_tokens=False,
        #         )
        #         wrapped_ids = wrapped_encoded["input_ids"].to(self.model.HF_device)
        #         wrapped_mask = wrapped_encoded["attention_mask"].to(self.model.HF_device)
        #         wrapped_tokens_batch: List[List[str]] = []
        #         for ids_row, mask_row in zip(wrapped_ids, wrapped_mask):
        #             active_ids = ids_row[mask_row.bool()].tolist()
        #             wrapped_tokens_batch.append(self.model.tokenizer.convert_ids_to_tokens(active_ids))

        #         past_kv, previous_hidden_embedding = self.model.generate_latent_batch_hidden_state(
        #             wrapped_ids,
        #             attention_mask=wrapped_mask,
        #             latent_steps=self.latent_steps,
        #             past_key_values=past_kv,
        #         )
        #         if self.sequential_info_only or self.latent_only:
        #             new_past_len = _past_length(past_kv)
        #             tokens_added = new_past_len - prev_past_len
        #             tokens_to_keep = self.latent_steps if self.latent_only else tokens_added
        #             past_kv = self._truncate_past(past_kv, tokens_to_keep)

        #         if self.latent_only:
        #             if self.latent_steps > 0:
        #                 previous_hidden_embedding = previous_hidden_embedding[:, -self.latent_steps:, :]
        #             else:
        #                 previous_hidden_embedding = previous_hidden_embedding[:, 0:0, :]

        #         # Transfer via intersection alignment on model switch
        #         if model_switched and len(embedding_record) > 0:
        #             prev_model = self.models[current_model_name]
        #             stacked = torch.cat(embedding_record, dim=1)
        #             transferred = self._transfer_hidden_states(stacked, prev_model, agent_model)
        #             embedding_record = [transferred]

        #         embedding_record.append(previous_hidden_embedding)
        #         current_model_name = agent_model_name

        #         if self.sequential_info_only or self.latent_only:
        #             embedding_record = embedding_record[-1:]

        #         for idx in range(batch_size):
        #             mask = wrapped_mask[idx].bool()
        #             trimmed_ids = wrapped_ids[idx][mask].to("cpu").tolist()
        #             agent_traces[idx].append({
        #                 "name": agent.name, "role": agent.role,
        #                 "input": wrapped_prompts[idx],
        #                 "input_ids": trimmed_ids,
        #                 "input_tokens": wrapped_tokens_batch[idx],
        #                 "latent_steps": self.latent_steps,
        #                 "output": "",
        #             })
        #     else:
        #         # Last agent: generate final text
        #         if self.args.think:
        #             final_agent_prompts = [f"{prompt}{self.args.think}" for prompt in prompts]
        #         else:
        #             final_agent_prompts = prompts

        #         if self.latent_steps > 0 and embedding_record:
        #             past_embedding = torch.cat(embedding_record, dim=1).to(self.vllm_device)

        #             # Transfer if last agent uses a different model
        #             if model_switched and current_model_name is not None:
        #                 prev_model = self.models[current_model_name]
        #                 past_embedding = self._transfer_hidden_states(
        #                     past_embedding, prev_model, agent_model,
        #                 )

        #             final_agent_encoded = self.model.tokenizer(
        #                 final_agent_prompts, return_tensors="pt", padding=True, add_special_tokens=False,
        #             )
        #             final_agent_encoded_ids = final_agent_encoded["input_ids"].to(self.model.HF_device)
        #             # Keep batch dim
        #             curr_prompt_emb = self.model.embedding_layer(final_agent_encoded_ids).to(self.vllm_device)

        #             # Handle latent embedding insertion position
        #             len_of_left = []
        #             for p in final_agent_prompts:
        #                 idx_pos = p.find("<|im_start|>user\n")
        #                 if idx_pos >= 0:
        #                     left = p[:idx_pos + len("<|im_start|>user\n")]
        #                 else:
        #                     left = ""
        #                 len_of_left.append(len(self.model.tokenizer(left)["input_ids"]) if left else 0)

        #             B, L, H = curr_prompt_emb.shape
        #             _, Lp, _ = past_embedding.shape

        #             whole_prompt_emb_list = []
        #             for i in range(B):
        #                 insert_idx = len_of_left[i]
        #                 left_emb = curr_prompt_emb[i, :insert_idx, :]
        #                 right_emb = curr_prompt_emb[i, insert_idx:, :]
        #                 combined = torch.cat([left_emb, past_embedding[i], right_emb], dim=0)
        #                 whole_prompt_emb_list.append(combined)

        #             max_len = max(x.shape[0] for x in whole_prompt_emb_list)
        #             whole_prompt_emb = torch.stack([
        #                 torch.cat([x, torch.zeros(max_len - x.shape[0], H, device=x.device)], dim=0)
        #                 for x in whole_prompt_emb_list
        #             ])

        #             prompt_embeds_list = [
        #                 {"prompt_embeds": embeds} for embeds in whole_prompt_emb
        #             ]

        #             outputs = self.model.vllm_engine.generate(
        #                 prompt_embeds_list, self.sampling_params,
        #             )

        #             generated_texts = [out.outputs[0].text.strip() for out in outputs]
        #         else:
        #             # No latent context (latent_steps=0): use text prompts directly
        #             generated_texts = self.model.vllm_generate_text_batch(
        #                 final_agent_prompts,
        #                 max_new_tokens=self.judger_max_new_tokens,
        #                 temperature=self.temperature,
        #                 top_p=self.top_p,
        #             )

        #         for idx in range(batch_size):
        #             text_out = generated_texts[idx].strip()
        #             final_texts[idx] = text_out
        #             agent_traces[idx].append({
        #                 "name": agent.name, "role": agent.role,
        #                 "input": final_agent_prompts[idx],
        #                 "output": text_out,
        #             })

        # # Assemble results
        # results: List[Dict] = []
        # for idx, item in enumerate(items):
        #     final_text = final_texts[idx]
        #     if self.task in ["mbppplus", "humanevalplus"]:
        #         pred = extract_markdown_python_block(final_text)
        #         gold = item.get("gold", "")
        #         if pred is None:
        #             ok = False
        #             error_msg = "python error: No python code block found"
        #         else:
        #             python_code_to_exe = pred + "\n" + gold
        #             ok, error_msg = run_with_timeout(python_code_to_exe, timeout=10)
        #         print(f"=========================================")
        #         print(f"Question {idx}")
        #         print(f"error_msg: {error_msg}")

        #     elif self.task in ["aime2024", "aime2025"]:
        #         pred = normalize_answer(extract_gsm8k_answer(final_text))
        #         gold = str(item.get("gold", "")).strip()
        #         try:
        #             if pred is None or pred == "":
        #                 ok = False
        #                 error_msg = f"Failed to extract answer from: {final_text[:100]}..."
        #             else:
        #                 pred_int = int(pred)
        #                 gold_int = int(gold)
        #                 ok = pred_int == gold_int
        #                 error_msg = None
        #         except ValueError:
        #             ok = False
        #             error_msg = f"Value error in parsing answer. Pred: {pred}, Gold: {gold}"
        #     else:
        #         pred = normalize_answer(extract_gsm8k_answer(final_text))
        #         gold = item.get("gold", "")
        #         ok = (pred == gold) if (pred and gold) else False
        #         error_msg = None

        #     results.append({
        #         "question": item["question"],
        #         "gold": gold,
        #         "solution": item["solution"],
        #         "prediction": pred,
        #         "raw_prediction": final_text,
        #         "agents": agent_traces[idx],
        #         "correct": ok,
        #     })
        # return results

    # ===================================================================
    # run_item
    # ===================================================================
    def run_item(self, item: Dict) -> Dict:
        return self.run_batch([item])[0]
    