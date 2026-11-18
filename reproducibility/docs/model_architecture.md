# Model Architecture: DeiT-Small Baseline and EViT Pruning Variants

## 1. Baseline Architecture: DeiT-Small (patch16, 224x224)

The perception backbone is based on Data-efficient Image Transformers (DeiT-Small):
- **Input Resolution**: $224 \times 224 \times 3$ RGB driving scenes.
- **Patch Embedding**: $16 \times 16$ non-overlapping patches $\to 14 \times 14 = 196$ patch tokens.
- **Tokens**: $196\text{ visual tokens} + 1\text{ [CLS] token} = 197\text{ total tokens}$.
- **Transformer Depth**: 12 Transformer encoder blocks.
- **Hidden Dimension ($D$)**: 384.
- **Attention Heads ($H$)**: 6 heads ($d_h = 64$ per head).
- **MLP Expansion Ratio**: $4.0$ ($1536$ hidden units).
- **Parameters**: $21.67\text{ Million}$.
- **Computational Complexity**: $4.60\text{ GMACs}$.

---

## 2. EViT Dynamic Token Pruning Mechanism

EViT modifies standard Vision Transformers by introducing attention-guided token pruning and fusion at designated hierarchical stages:

### Pruning Stages (`drop_loc`)
Pruning occurs after blocks **3, 6, and 9** (0-indexed):
1. **Block 3**: Stage 1 compression.
2. **Block 6**: Stage 2 compression.
3. **Block 9**: Stage 3 compression.

### Selection and Fusion Equations
1. **CLS-Attention Scoring**:
   For each visual token $j$, its importance score $s_j$ is given by the cross-attention weight from the `[CLS]` token averaged across all 6 attention heads:
   $$s_j = \frac{1}{H} \sum_{h=1}^H A_{h, 0, j}$$

2. **Top-$k$ Retention**:
   The $K_l = \lceil k \cdot N_l \rceil$ tokens with highest $s_j$ are retained:
   $$\mathcal{T}_{\text{keep}} = \arg\operatorname{top-}K_l(s_1, \dots, s_{N_l})$$

3. **Inattentive Token Fusion**:
   Instead of discarding the remaining $N_l - K_l$ tokens outright (which causes irreversibility of context), EViT computes an attention-weighted summary token $\mathbf{x}_{\text{fused}}$:
   $$\mathbf{x}_{\text{fused}} = \sum_{j \in \mathcal{T}_{\text{discard}}} \frac{s_j}{\sum_{m \in \mathcal{T}_{\text{discard}}} s_m + \epsilon} \mathbf{x}_j$$
   This single summary token is concatenated back to the sequence, preserving background contextual information.

---

## 3. Token Sequence Lengths and Efficiency Profiles

| Model Variant | Base Keep-Rate ($k$) | Pruning Locations | Token Sequence Length | GMACs | Latency (Graph p95) | Speedup vs DeiT |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **DeiT-Small** | $1.0$ (No pruning) | None | $197 \to 197 \to 197 \to 197$ | $4.60$ | $3.68\text{ ms}$ | $1.00\times$ |
| **EViT-K09** | $0.9$ | (3, 6, 9) | $197 \to 179 \to 163 \to 148$ | $4.03$ | $3.54\text{ ms}$ | $1.04\times$ |
| **EViT-K07** | $0.7$ | (3, 6, 9) | $197 \to 140 \to 100 \to 72$ | $3.03$ | $3.49\text{ ms}$ | $1.05\times$ |
| **EViT-K05** | $0.5$ | (3, 6, 9) | $197 \to 100 \to 52 \to 28$ | $2.31$ | $2.81\text{ ms}$ | $1.31\times$ |

*Note: The token lengths include $1\text{ [CLS] token} + K_l\text{ kept tokens} + 1\text{ fused summary token}$.*
