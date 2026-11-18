# Training Protocol: Loss Formulations, Optimization, and Checkpointing

## 1. Mathematical Formulations

### A. Class-Balanced Cross-Entropy (CB-CE)
Given training class sample counts $n_c$ for $c \in \{0, \dots, C-1\}$, smoothed inverse-frequency weights are computed via:
$$w_c^{\text{raw}} = \left( \frac{N}{C \cdot n_c} \right)^\beta, \quad \beta = 0.25$$
Normalized by their mean:
$$w_c = \frac{w_c^{\text{raw}}}{\frac{1}{C} \sum_{j=0}^{C-1} w_j^{\text{raw}}}$$
For our audited training split ($N = 69,634$):
$$\mathbf{w} = [0.3393, 0.4268, 0.5167, 1.1118, 2.1507, 1.4547]$$

The weighted cross-entropy loss is:
$$\mathcal{L}_{\text{CB-CE}} = - \frac{1}{B} \sum_{i=1}^B w_{y_i} \log \frac{\exp(z_{s, i, y_i})}{\sum_{j=0}^{C-1} \exp(z_{s, i, j})}$$

---

### B. Correctness-Gated Margin Preservation Regularization (CMPR)
Let $z_{t, i}$ and $z_{s, i}$ denote logits from the frozen teacher and student models for sample $i$ with ground-truth label $y_i$.

1. **Logit Margins**:
   $$m_{t, i} = z_{t, i, y_i} - \max_{j \ne y_i} z_{t, i, j}$$
   $$m_{s, i} = z_{s, i, y_i} - \max_{j \ne y_i} z_{s, i, j}$$

2. **Teacher Correctness Gate**:
   $$g_i = \mathbb{1}\left[ \arg\max_j z_{t, i, j} = y_i \right]$$

3. **Margin Discrepancy Loss**:
   $$\mathcal{L}_{\text{margin}, i} = \left[ \max(0, m_{t, i} - m_{s, i} - \gamma) \right]^2, \quad \gamma = 0.0$$

4. **Normalized Batch Loss**:
   $$\mathcal{L}_{\text{CMPR}} = \frac{\sum_{i=1}^B g_i \mathcal{L}_{\text{margin}, i}}{\sum_{i=1}^B g_i + \epsilon}$$

5. **Composite Objective**:
   $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CB-CE}} + \lambda_{\text{CMPR}} \mathcal{L}_{\text{CMPR}}, \quad \lambda_{\text{CMPR}} = 0.5$$

---

## 2. Optimization Configuration

- **Optimizer**: AdamW ($\beta_1 = 0.9, \beta_2 = 0.999, \text{weight decay} = 0.05$).
- **Differential Learning Rates**:
  - Linear Classification Head: $\eta_{\text{head}} = 3.0 \times 10^{-4}$
  - Transformer Backbone: $\eta_{\text{backbone}} = 2.0 \times 10^{-5}$
- **Learning Rate Schedule**: Cosine annealing with 1-epoch linear warmup.
- **Batch Size**: 64 images per GPU batch.
- **Training Epochs**: 3 epochs ($3 \times 1,089 = 3,267$ gradient update steps).
- **Mixed Precision**: Automatic Mixed Precision (AMP `fp16`) via `torch.cuda.amp.GradScaler`.
- **Checkpoint Selection Rule**: Best checkpoint selected strictly by highest validation Top-1 accuracy ($\tau_{\text{val}}$) on the 9,947-image validation split. **Zero test-set tuning.**
