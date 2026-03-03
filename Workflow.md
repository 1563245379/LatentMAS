# [SYSTEM ROLE]
You are a Lead AI Research Engineer specializing in Large Language Models (LLMs), multi-agent systems, and representation engineering. Your task is to implement a **Training-Free Heterogeneous Latent Communication Bridge** between two different LLMs. 

#[CONTEXT & THEORETICAL FOUNDATION]
You are fusing two breakthrough methodologies to enable seamless, text-free thought transfer between two heterogeneous models (Model A and Model B) that have **different hidden dimensions ($d_{out} \neq d_{in}$)** and **different vocabularies ($V^A \neq V^B$)**.

1. **From LatentMAS:** A linear projection matrix $W_a$ can align continuous latent hidden states into an input embedding space. The 2-Wasserstein distance between the aligned distribution and the target distribution is strictly upper-bounded by the Frobenius norm of their difference. This proof remains strictly valid even if the mapping matrix $W_a$ projects across different dimensional spaces (no non-singular or square matrix assumptions required).
2. **From GAC:** Mismatched vocabularies can be resolved by projecting outputs into a unified vocabulary space (Union Space) using binary mapping matrices.
3. **The Fusion Conclusion:** By expanding the embedding matrices of both models to a Union Vocabulary Space and applying an Intersection Mask ($D$), we can compute an optimal, training-free alignment matrix $W_a$ using Ridge Regression. This allows Model A to send its latent hidden states directly to Model B's transformer layers without decoding to text.

# [ALGORITHM DESIGN (Mathematical Formulation)]
Let Model A (Sender) have vocabulary $V^A$ (size $N_A$), hidden dimension $d_{out}$, and Output LM Head $W_{out}^A \in \mathbb{R}^{N_A \times d_{out}}$.
Let Model B (Receiver) have vocabulary $V^B$ (size $N_B$), hidden dimension $d_{in}$, and Input Embedding $W_{in}^B \in \mathbb{R}^{N_B \times d_{in}}$.

1. **Union & Masking:** Define intersection vocabulary $V^I = V^A \cap V^B$. 
2. **Objective:** Minimize the masked Frobenius norm: 
   $$\min_{W_a} \| D (\tilde{W}_{out}^A W_a - \tilde{W}_{in}^B) \|_F^2$$
3. **Closed-Form Solution:** 
   $$W_a = \left( (W_{out\_intersect}^A)^\top W_{out\_intersect}^A + \lambda I \right)^{-1} (W_{out\_intersect}^A)^\top W_{in\_intersect}^B$$

#[IMPLEMENTATION PROTOCOL]
Execute the following steps using Python and PyTorch. 

### Step 1: Tokenizer Intersection Extraction
**Objective:** Identify the shared semantic space between the two heterogeneous models.
*   Load `Tokenizer_A` and `Tokenizer_B`.
*   Extract all string tokens and their corresponding IDs from both tokenizers.
*   Find the exact string matches (the intersection set $V^I$).
*   Create two aligned index lists: `indices_A` (IDs of shared tokens in Model A) and `indices_B` (IDs of shared tokens in Model B), ensuring the order of strings matches perfectly.

### Step 2: Weight Matrix Extraction & Sub-setting
**Objective:** Extract the relevant projection matrices, optimizing out the zero-padding of the Union Space by directly slicing the intersection.
*   Extract the LM Head weight of Model A: `W_out_A = Model_A.lm_head.weight.data` (Shape: `[N_A, d_out]`).
*   Extract the Input Embedding weight of Model B: `W_in_B = Model_B.model.embed_tokens.weight.data` (Shape: `[N_B, d_in]`).
*   Slice both matrices using the intersection indices:
    *   `W_out_intersect = W_out_A[indices_A]` (Shape: `[N_intersect, d_out]`)
    *   `W_in_intersect = W_in_B[indices_B]` (Shape: `[N_intersect, d_in]`)

### Step 3: Compute the Alignment Matrix ($W_a$)
**Objective:** Solve the ridge regression to find the optimal linear projection $W_a$.
*   Define a small regularization term $\lambda$ (e.g., `lambda_val = 1e-4`) to ensure numerical stability.
*   Compute the Gram matrix: `Gram = W_out_intersect.T @ W_out_intersect` (Shape: `[d_out, d_out]`).
*   Add the ridge penalty: `Gram_reg = Gram + lambda_val * torch.eye(d_out)`.
*   Compute the target projection: `Target = W_out_intersect.T @ W_in_intersect` (Shape: `[d_out, d_in]`).
*   Solve for $W_a$: `W_a = torch.linalg.solve(Gram_reg, Target)` (Shape: `[d_out, d_in]`).
*   *Note: This is a one-time offline calculation. Save `W_a` to disk or cache it in memory.*

### Step 4: Runtime Execution (Continuous Latent Communication)
**Objective:** Transfer reasoning from Model A to Model B during inference.
*   **Generate Latent Thought (Model A):** Run Model A forward pass without generating text. Extract the last-layer hidden state $h_A$ of the final token sequence. (Shape: `[batch_size, seq_len, d_out]`).
*   **Project Latent Thought:** Multiply the hidden state by the alignment matrix.
    *   `h_aligned = h_A @ W_a` (Shape: `[batch_size, seq_len, d_in]`).
*   **Inject to Model B:** Pass `h_aligned` directly as the `inputs_embeds` argument to Model B's forward pass (bypassing Model B's standard token embedding layer).
    *   `outputs = Model_B(inputs_embeds=h_aligned, ...)`

# [VERIFICATION & QUALITY CHECKS]
As you implement this, ensure the following asserts pass:
1. `len(indices_A) == len(indices_B)` -> The number of intersection tokens must strictly match.
2. `W_a.shape == (d_out, d_in)` -> The alignment matrix must correctly map the dimension of A's hidden states to B's embedding dimension.
3. Validate $W_a$ by randomly selecting a shared token `t`. Check if `cosine_similarity(W_out_A[t_idx_A] @ W_a, W_in_B[t_idx_B])` is close to 1.0. This confirms the Input-Output Distribution Alignment is mathematically sound.