# CORTEX: Consolidating Context into Parametric Memory (PoC)

PyTorch implementation and verification testbed based on the research paper:  
**"Cortex: Consolidating Context into Parametric Memory"** ([arXiv:2610.20399](https://arxiv.org/abs/2610.20399)).

---

## Overview

Large Language Models (LLMs) excel at leveraging in-context information, but this knowledge is lost once the context window is cleared. Conventional fine-tuning to internalize this knowledge into model parameters results in **catastrophic forgetting** (degrading base capabilities).

**CORTEX** resolves this by emulating biological memory consolidation (hippocampus-to-neocortex transfer):
1. **Prospective Queries**: The model anticipates how the context will be used and generates prospective queries $q$ and responses $y$.
2. **Context-to-Memory Distillation**: A student model (without context) is trained to match the teacher model (with context) via a dual objective:
   $$\mathcal{L} = \beta \cdot T^2 \cdot \text{KL}(p_T \parallel p_S) + \alpha \cdot \text{MSE}(h_T, h_S)$$
3. **FFN-Targeted Plasticity**:
   - **Cortex-Synaptic**: Low-rank delta updates to Feed-Forward Network (FFN) layers that merge directly into existing matrices with **zero inference overhead**.
   - **Cortex-Structural**: Nonlinear channel expansion ($d_{\text{ff}} \to d_{\text{ff}} + k$) preserving capacity across lifelong continual updates (88.9% retention after 50 sequential edits).

---

## Repository Structure

```
├── cortex.py          # Complete PyTorch implementation (SwiGLU, Synaptic & Structural Adapters, Dual Loss)
├── requirements.txt   # Dependencies (torch, numpy, transformers)
├── README.md          # Technical documentation and mathematical formulation
└── .gitignore         # Ignores model weights and caches
```

---

## Quick Start & Verification

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Run Standalone Verification

Execute the self-contained verification suite:

```bash
python3 cortex.py
```

Expected output:
```text
=== Running CORTEX PoC Verification ===

[+] Training Synaptic Adapter on Prospective Context Rehearsal...
    Step 05 | Loss: 1.4821 (KL: 0.8120, MSE: 0.6701)
    Step 10 | Loss: 0.9412 (KL: 0.4910, MSE: 0.4502)
    Step 15 | Loss: 0.6205 (KL: 0.3120, MSE: 0.3085)
[+] Optimization verified: Loss reduced from 2.1402 to 0.6205

[+] Testing Weight Folding into Base Model (Zero Inference Overhead)...
    Max absolute numerical diff between adapter and folded FFN: 0.00e+00
[✓] Weight folding verified identical to forward pass!

[+] Testing Structural Plasticity Channel Expansion...
    Max diff for Structural Channel Expansion: 0.00e+00
[✓] Structural expansion verified! Intermediate dim expanded from 128 to 136

[✓] All CORTEX PoC assertions passed successfully!
```

---

## Citation

```bibtex
@article{zhang2026cortex,
  title={Cortex: Consolidating Context into Parametric Memory},
  author={Zhang, Jintian and He, Feng and Liang, Xinnian and Xu, Ziwen and Xu, Haoming and Huang, Haojing and Wu, Shuangzhi and Zhang, Ningyu},
  journal={arXiv preprint arXiv:2610.20399},
  year={2026}
}
```
