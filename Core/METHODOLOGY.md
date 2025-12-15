# Methodology: Knowledge Graph Path Generation via Discrete Diffusion

## 1. Problem Formulation

Given a natural language question $q$ and a knowledge graph $\mathcal{G} = (\mathcal{E}, \mathcal{R})$, generate diverse reasoning paths $P = \{p_1, \ldots, p_k\}$ connecting question entities to answer entities. Each path is a relation sequence: $\mathbf{r}_i = (r_1, r_2, \ldots, r_{L-1})$.

## 2. Data Format

```json
{
    "question": "what sports are played in canada",
    "q_entity": ["Canada"],
    "a_entity": ["Ice Hockey", "Lacrosse", "Curling"],
    "graph": [
        ["Canada", "sports.sports_team.location", "Toronto Maple Leafs"],
        ["Toronto Maple Leafs", "sports.sports_team.sport", "Ice Hockey"]
    ],
    "paths": [{
        "relation_chain": "sports.sports_team.location -> sports.sports_team.sport",
        "entities": ["Canada", "Toronto Maple Leafs", "Ice Hockey"],
        "relations": ["sports.sports_team.location", "sports.sports_team.sport"]
    }]
}
```

### Data Processing
- **Vocabulary**: Special tokens `<PAD>=0, <UNK>=1, <BOS>=2, <EOS>=3, <MASK>=4` for entities; `<PAD>=0, <UNK>=1, <MASK>=2` for relations.
- **Path Encoding**: `[BOS, e1, e2, ..., en, EOS]` for entities, `[r1, r2, ..., rn-1]` for relations.
- **Multi-path**: Each question can have multiple valid paths; training samples all paths.

## 3. Model Architecture

### Question Encoder
- **Backbone**: `sentence-transformers/all-MiniLM-L6-v2`.
- **Outputs**: 
  - `Q_seq` $[B, S, D]$: Projected to `hidden_dim` for cross-attention.
  - `Q_pooled` $[B, D]$: Mean-pooled with attention mask for path count prediction.

### Path Count Predictor
- **Structure**: `Linear(D) → ReLU → Dropout → Linear(D/2) → ReLU → Dropout → Linear(max_paths)`.
- **Function**: Predicts number of paths $k \in [1, 30]$ from `Q_pooled`.
- **Loss**: Cross-entropy against ground truth path count ($k_{gt} - 1$).

### Path Diffusion Transformer (Denoiser)
- **Input**: Noisy relation tokens + position embeddings + time embeddings.
- **Time Embedding**: Sinusoidal embeddings → MLP → `time_emb` $[B, D]$.
- **Blocks** (×6): AdaLN-Self-Attention → Cross-Attention(to `Q_seq`) → AdaLN-FFN.
- **AdaLN Modulation**:
  - `(shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp) = MLP(time_emb)`.
  - **Application**:
    $$x = x + \text{gate}_{\text{msa}} \cdot \text{Attn}(\text{LN}(x) \cdot (1 + \text{scale}_{\text{msa}}) + \text{shift}_{\text{msa}})$$
    $$x = x + \text{gate}_{\text{mlp}} \cdot \text{FFN}(\text{LN}(x) \cdot (1 + \text{scale}_{\text{mlp}}) + \text{shift}_{\text{mlp}})$$

## 4. Discrete Diffusion Process

### Forward Process (Training)
Corrupt clean relations by masking with cosine schedule:
$$p_{\text{mask}}(t) = 1 - \bar{\alpha}_t, \quad \bar{\alpha}_t = \frac{f(t)}{f(0)}, \quad f(t) = \cos^2\left(\frac{t/T + 0.008}{1.008} \cdot \frac{\pi}{2}\right)$$
Sample noisy sequence: `r_t[i] = MASK if rand() < p_mask(t) else r_0[i]`.

### Reverse Process (Inference)
1. **Initialize**: `r_T = [MASK, ..., MASK]`.
2. **Iterate** $t = T \to 1$:
   - **Predict**: `logits = Denoiser(r_t, t, Q_seq)`.
   - **Sample**: `r' ~ Categorical(softmax(logits / τ))`.
   - **Update**: Unmask positions with probability $1 - \bar{\alpha}_{t-1}/\bar{\alpha}_t$.
     $$r_{t-1}[i] = r'[i] \text{ if unmask else } r_t[i]$$

## 5. Training

### Loss Functions
- **Diffusion Loss**: Cross-entropy on predicting clean relations from noisy input.
  $$\mathcal{L}_{\text{diff}} = -\sum_{i} \mathbb{1}[\text{valid}(i)] \cdot \log p_\theta(r_{0,i} | \mathbf{r}_t, \mathbf{Q}, t)$$
  *Note: Relation-only mode uses `path_mask[:, :-1]` to derive valid relation positions.*

- **Path Count Loss**: 
  $$\mathcal{L}_{\text{count}} = \text{CE}(\text{logits}_{\text{count}}, k_{\text{gt}} - 1)$$

- **Total Loss**: $\mathcal{L} = \mathcal{L}_{\text{diff}} + 0.1 \cdot \mathcal{L}_{\text{count}}$.

### Multi-Path Training
Expand batch to include all paths per question. Compute diffusion loss for each path, then average over valid paths per question.

## 6. Inference Pipeline

1. **Encode**: Question → `Q_seq, Q_pooled`.
2. **Predict Count**: `k = PathCountPredictor(Q_pooled).argmax() + 1`.
3. **Generate Paths**: For each path $i = 0 \dots k-1$:
   - **Diversity Penalty**: Scale temperature $\tau_i = \tau_0 \cdot (1 + \alpha \cdot i)$.
   - **Run Reverse Diffusion**: Generate relation chain $\mathbf{r}^{(i)}$.
4. **Retrieve Entities**: Traverse KG from topic entity using generated relations.
5. **Aggregate**: Rank answers by frequency across all $k$ paths.

## 7. Key Design Choices

| Choice | Rationale |
|--------|-----------|
| **Discrete Diffusion** | Handles categorical nature of relations; allows parallel decoding. |
| **Relation-Only** | Reduces vocabulary size and complexity; entities retrieved deterministically. |
| **AdaLN-Zero** | Adaptive Layer Norm with gating effectively conditions generation on noise level. |
| **Cosine Schedule** | Prevents abrupt information loss, improving training stability. |
| **Path Count Prediction** | Dynamically adjusts computational effort based on question complexity. |



