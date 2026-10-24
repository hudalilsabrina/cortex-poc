"""
CORTEX: Consolidating Context into Parametric Memory (PyTorch PoC)
Based on: 'Cortex: Consolidating Context into Parametric Memory' (arXiv:2610.20399)

Implements:
1. SwiGLU Feed-Forward Network layer.
2. Cortex-Synaptic Adapter (Low-Rank update mergeable into weights).
3. Cortex-Structural Adapter (Nonlinear Channel Expansion d_ff -> d_ff + k).
4. Memory Consolidation Dual-Objective Loss (Logit KL + Post-FFN Hidden MSE).
5. Self-contained verification pipeline.
"""

import math
from typing import Tuple, Optional

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    torch = None
    nn = None
    F = None


if torch is not None:
    class SwiGLUFFN(nn.Module):
        """
        Standard SwiGLU Feed-Forward Network as used in modern architectures
        (LLaMA, Qwen, Mistral, Gemma).
        Computes: down_proj(SiLU(gate_proj(x)) * up_proj(x))
        """
        def __init__(self, hidden_dim: int, intermediate_dim: int):
            super().__init__()
            self.hidden_dim = hidden_dim
            self.intermediate_dim = intermediate_dim
            self.gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
            self.up_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
            self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


    class CortexSynapticAdapter(nn.Module):
        """
        Cortex-Synaptic Parameterization:
        Low-rank synaptic plasticity updates to existing projections.
        Trains 3 * r * (dh + dff) parameters.
        Zero latency at inference: folds directly into base weight matrices.
        """
        def __init__(self, ffn: SwiGLUFFN, rank: int = 16, alpha: float = 32.0):
            super().__init__()
            self.ffn = ffn
            self.rank = rank
            self.scaling = alpha / rank

            # Freeze original parameters
            for p in self.ffn.parameters():
                p.requires_grad = False

            dh = ffn.hidden_dim
            dff = ffn.intermediate_dim

            # Low-rank adapters for gate, up, down projections
            self.gate_A = nn.Linear(dh, rank, bias=False)
            self.gate_B = nn.Linear(rank, dff, bias=False)

            self.up_A = nn.Linear(dh, rank, bias=False)
            self.up_B = nn.Linear(rank, dff, bias=False)

            self.down_A = nn.Linear(dff, rank, bias=False)
            self.down_B = nn.Linear(rank, dh, bias=False)

            self._reset_parameters()

        def _reset_parameters(self):
            nn.init.kaiming_uniform_(self.gate_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.gate_B.weight)
            nn.init.kaiming_uniform_(self.up_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.up_B.weight)
            nn.init.kaiming_uniform_(self.down_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.down_B.weight)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            gate = self.ffn.gate_proj(x) + self.gate_B(self.gate_A(x)) * self.scaling
            up = self.ffn.up_proj(x) + self.up_B(self.up_A(x)) * self.scaling
            inter = F.silu(gate) * up
            out = self.ffn.down_proj(inter) + self.down_B(self.down_A(inter)) * self.scaling
            return out

        def merge_and_fold(self):
            """Folds synaptic delta updates directly into base FFN weight matrices."""
            with torch.no_grad():
                self.ffn.gate_proj.weight += (self.gate_B.weight @ self.gate_A.weight) * self.scaling
                self.ffn.up_proj.weight += (self.up_B.weight @ self.up_A.weight) * self.scaling
                self.ffn.down_proj.weight += (self.down_B.weight @ self.down_A.weight) * self.scaling


    class CortexStructuralAdapter(nn.Module):
        """
        Cortex-Structural Parameterization:
        Structural plasticity via newly grown nonlinear channels.
        Expands intermediate dimension from d_ff to (d_ff + k).
        Trains 3 * k * dh parameters (fewer trainable params when k=r).
        Provides orthogonal capacity for lifelong continual/sequential updates.
        """
        def __init__(self, ffn: SwiGLUFFN, k_channels: int = 16):
            super().__init__()
            self.ffn = ffn
            self.k_channels = k_channels

            for p in self.ffn.parameters():
                p.requires_grad = False

            dh = ffn.hidden_dim
            # Growth channels
            self.gate_new = nn.Linear(dh, k_channels, bias=False)
            self.up_new = nn.Linear(dh, k_channels, bias=False)
            self.down_new = nn.Linear(k_channels, dh, bias=False)

            self._reset_parameters()

        def _reset_parameters(self):
            nn.init.kaiming_uniform_(self.gate_new.weight, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.up_new.weight, a=math.sqrt(5))
            nn.init.zeros_(self.down_new.weight)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            base_out = self.ffn(x)
            # Parallel structural channel pass
            branch_out = self.down_new(F.silu(self.gate_new(x)) * self.up_new(x))
            return base_out + branch_out

        def merge_and_fold(self) -> SwiGLUFFN:
            """
            Folds structural branch into a wider SwiGLUFFN:
            Concatenates weights along intermediate dimension (dff -> dff + k).
            """
            dh = self.ffn.hidden_dim
            new_dff = self.ffn.intermediate_dim + self.k_channels
            expanded_ffn = SwiGLUFFN(dh, new_dff)

            with torch.no_grad():
                expanded_ffn.gate_proj.weight.copy_(
                    torch.cat([self.ffn.gate_proj.weight, self.gate_new.weight], dim=0)
                )
                expanded_ffn.up_proj.weight.copy_(
                    torch.cat([self.ffn.up_proj.weight, self.up_new.weight], dim=0)
                )
                expanded_ffn.down_proj.weight.copy_(
                    torch.cat([self.ffn.down_proj.weight, self.down_new.weight], dim=1)
                )

            return expanded_ffn


    def cortex_loss(
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        student_hidden: torch.Tensor,
        teacher_hidden: torch.Tensor,
        alpha: float = 1.0,
        beta: float = 1.0,
        temperature: float = 2.0,
        top_kv: Optional[int] = 50
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        CORTEX Memory Consolidation Loss:
        Combines output logit distillation (truncated KL) with post-FFN representation matching (MSE).
        L = beta * T^2 * KL(p_T || p_S) + alpha * MSE(h_S, h_T)
        """
        # 1. Output-distribution matching via truncated KL
        if top_kv is not None and top_kv < teacher_logits.size(-1):
            topk_vals, topk_indices = torch.topk(teacher_logits, k=top_kv, dim=-1)
            p_teacher = F.softmax(topk_vals / temperature, dim=-1)
            # Gather student logits for matching tokens
            student_selected = torch.gather(student_logits, dim=-1, index=topk_indices)
            log_p_student = F.log_softmax(student_selected / temperature, dim=-1)
            kl_div = F.kl_div(log_p_student, p_teacher, reduction="batchmean") * (temperature ** 2)
        else:
            p_teacher = F.softmax(teacher_logits / temperature, dim=-1)
            log_p_student = F.log_softmax(student_logits / temperature, dim=-1)
            kl_div = F.kl_div(log_p_student, p_teacher, reduction="batchmean") * (temperature ** 2)

        # 2. Representation alignment (post-FFN hidden state MSE)
        mse_loss = F.mse_loss(student_hidden, teacher_hidden)

        # Composite objective
        total_loss = (beta * kl_div) + (alpha * mse_loss)
        return total_loss, kl_div, mse_loss


def run_standalone_verification():
    """Runs functional assertion test verifying zero-error gradient flow and weight folding."""
    if torch is None:
        print("[!] PyTorch is not installed in the current environment.")
        print("[!] Install with: pip install torch numpy")
        return

    print("=== Running CORTEX PoC Verification ===")
    torch.manual_seed(42)

    batch_size = 4
    seq_len = 16
    hidden_dim = 64
    intermediate_dim = 128
    rank = 8

    # 1. Instantiate Teacher and Base Student FFN
    base_ffn = SwiGLUFFN(hidden_dim, intermediate_dim)
    adapter = CortexSynapticAdapter(base_ffn, rank=rank)

    x_input = torch.randn(batch_size, seq_len, hidden_dim)

    # Synthetic Teacher targets (representing model output with explicit context)
    target_hidden = torch.randn(batch_size, seq_len, hidden_dim)
    target_logits = torch.randn(batch_size, seq_len, 256)

    # Linear projection to simulate LM head
    lm_head = nn.Linear(hidden_dim, 256, bias=False)

    optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-3)

    # 2. Train adapter for 15 steps
    print("\n[+] Training Synaptic Adapter on Prospective Context Rehearsal...")
    initial_loss = None
    for step in range(15):
        optimizer.zero_grad()
        out_hidden = adapter(x_input)
        out_logits = lm_head(out_hidden)

        loss, kl, mse = cortex_loss(
            out_logits, target_logits, out_hidden, target_hidden,
            alpha=1.0, beta=1.0, temperature=2.0
        )
        if initial_loss is None:
            initial_loss = loss.item()

        loss.backward()
        optimizer.step()

        if (step + 1) % 5 == 0:
            print(f"    Step {step+1:02d} | Loss: {loss.item():.4f} (KL: {kl.item():.4f}, MSE: {mse.item():.4f})")

    final_loss = loss.item()
    assert final_loss < initial_loss, "Loss failed to decrease during training"
    print(f"[+] Optimization verified: Loss reduced from {initial_loss:.4f} to {final_loss:.4f}")

    # 3. Verify Weight Merging (Fold)
    print("\n[+] Testing Weight Folding into Base Model (Zero Inference Overhead)...")
    out_before_merge = adapter(x_input)
    adapter.merge_and_fold()
    out_after_merge = base_ffn(x_input)

    diff = torch.max(torch.abs(out_before_merge - out_after_merge)).item()
    print(f"    Max absolute numerical diff between adapter and folded FFN: {diff:.2e}")
    assert diff < 1e-5, f"Folded weights mismatch: {diff}"
    print("[✓] Weight folding verified identical to forward pass!")

    # 4. Test Structural Plasticity Branch
    print("\n[+] Testing Structural Plasticity Channel Expansion...")
    base_ffn2 = SwiGLUFFN(hidden_dim, intermediate_dim)
    struct_adapter = CortexStructuralAdapter(base_ffn2, k_channels=8)
    struct_out = struct_adapter(x_input)
    expanded_ffn = struct_adapter.merge_and_fold()
    expanded_out = expanded_ffn(x_input)
    diff_struct = torch.max(torch.abs(struct_out - expanded_out)).item()
    print(f"    Max diff for Structural Channel Expansion: {diff_struct:.2e}")
    assert diff_struct < 1e-5, f"Structural folding mismatch: {diff_struct}"
    print(f"[✓] Structural expansion verified! Intermediate dim expanded from {intermediate_dim} to {expanded_ffn.intermediate_dim}")

    print("\n[✓] All CORTEX PoC assertions passed successfully!")


if __name__ == "__main__":
    run_standalone_verification()
