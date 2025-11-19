import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class CosineWithWarmup:
    def __init__(self, opt, total_steps, warmup_steps, base_lr, min_lr=0.0, start_step=0):
        self.optimizer, self.total_steps, self.warmup_steps = opt, max(1,int(total_steps)), max(0,int(warmup_steps))
        self.base_lr, self.min_lr = float(base_lr), float(min_lr)
        self.step_num = int(start_step)
        for pg in self.optimizer.param_groups:
            pg.setdefault("base_lr", self.base_lr)
        self._set_lr()  # initialize LR at step_num

    def _lr_scale(self, t):
        if self.warmup_steps > 0 and t <= self.warmup_steps:
            return t / self.warmup_steps
        if self.total_steps <= self.warmup_steps:
            return 1.0
        progress = (t - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        progress = min(1.0, max(0.0, progress))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
        
    def _set_lr(self):
        scale = self._lr_scale(self.step_num)
        for pg in self.optimizer.param_groups:
            base = pg.get("base_lr", self.base_lr)
            pg["lr"] = max(self.min_lr, base * scale)

    def step(self):
        self.step_num += 1
        self._set_lr()


class ResidualBlock(nn.Module):
    """Pre-activation residual MLP block with LayerNorm and GELU."""
    def __init__(self, dim: int, dropout: float = 0.1, layernorm: bool = True):
        super().__init__()
        self.ln  = nn.LayerNorm(dim) if layernorm else nn.Identity()
        self.fc1 = nn.Linear(dim, dim * 4)
        self.fc2 = nn.Linear(dim * 4, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Pre-activation
        h = self.ln(x)
        h = F.gelu(self.fc1(h))
        h = self.dropout(h)
        h = self.fc2(h)
        return x + h


class ResidualMLP(nn.Module):
    """
    Stack of residual blocks with input projection and optional concat of the ORIGINAL input at each down stage.

    NOTE: Callers should normally do `self.trunk(x)` and NOT pass `x_skip`.
    If you do pass `x_skip`, it must be the original raw input to this MLP, not the evolving hidden state,
    otherwise widths explode and training degrades.
    """
    def __init__(self, in_dim: int, hidden_dims: List[int],
                 dropout: float = 0.1, layernorm: bool = True, use_input_skip: bool = True):
        super().__init__()
        self.use_input_skip = use_input_skip
        self.layernorm = layernorm
        self.dropout = dropout
        self.in_dim = in_dim
        self.hidden_dims = hidden_dims

        # Initial projection to first hidden width
        self.in_proj = nn.Linear(in_dim, hidden_dims[0])

        # Two preact residual blocks at first hidden width
        self.pre_blocks = nn.ModuleList([ResidualBlock(hidden_dims[0], dropout, layernorm),
                                         ResidualBlock(hidden_dims[0], dropout, layernorm),
                                         ])

        # Build "down" stages: (concat -> LayerNorm(concat_dim) -> Linear(concat_dim -> next_hidden) -> GELU -> Dropout)
        self.downs = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            prev_w = hidden_dims[i]
            concat_w = prev_w + (in_dim if use_input_skip else 0)
            block = nn.ModuleDict({"ln": nn.LayerNorm(concat_w) if layernorm else nn.Identity(),
                                   "fc": nn.Linear(concat_w, hidden_dims[i+1]),
                                   "act": nn.GELU(),
                                   "drop": nn.Dropout(dropout),
                                   })
            self.downs.append(block)

        self.final_dim = hidden_dims[-1]

    def forward(self, x, x_skip=None):        
        if x_skip is None:
            x_skip = x # usually the raw input to this MLP
        else:
            # Dev-time guard: ensure the skip is the same tensor shape as original input
            if x_skip.shape[-1] != self.in_dim:
                raise ValueError(f"x_skip last dim {x_skip.shape[-1]} != in_dim {self.in_dim}")
        
        # Project and do pre-activation residuals at the first hidden width
        h = self.in_proj(x)
        for blk in self.pre_blocks:
            h = blk(h)

        # Down stages with explicit concat-normalize-linear
        for blk in self.downs:
            
            if self.use_input_skip:
                h_cat = torch.cat([h, x_skip], dim=-1)
            else:
                h_cat = h
            
            h = blk["ln"](h_cat)
            h = blk["fc"](h)
            h = blk["act"](h)
            h = blk["drop"](h)
        return h


class GaussianHead(nn.Module):
    """Outputs mean vector; we treat reconstruction loss as MSE between predicted mean and target log1p expression"""
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, h):
        # We predict the mean directly; optionally could clamp/ext transform.
        return self.fc(h)
    

class DeltaHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        in_dim = cfg.gene_embed_dim + cfg.context_dim
        hid    = cfg.hidden_dims[-1]

        self.fc1 = nn.Linear(in_dim, hid)
        self.ln_hid = nn.LayerNorm(hid) if cfg.layernorm else nn.Identity()
        self.fc2 = nn.Linear(hid, cfg.latent_dim)
        self.dropout = nn.Dropout(cfg.dropout)

        nn.init.xavier_uniform_(self.fc1.weight); nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.weight); nn.init.zeros_(self.fc2.bias)

        self.register_buffer("delta_scale_max", torch.tensor(1.0))

    def forward(self, g, u):
        h = torch.cat([g, u], dim=-1)     # [B, in_dim]
        h = F.gelu(self.fc1(h))           # [B, hid]
        h = self.ln_hid(h)                # [B, hid]
        h = self.dropout(h)
        raw = self.fc2(h)                 # [B, Dz]
        return torch.tanh(raw) * self.delta_scale_max
    

class FiLM(nn.Module):
    """Feature-wise Linear Modulation by a context vector: gamma, beta -> y = gamma * h + beta"""
    def __init__(self, h_dim: int, cond_dim: int):
        super().__init__()
        self.to_gamma = nn.Linear(cond_dim, h_dim)
        self.to_beta  = nn.Linear(cond_dim, h_dim)
        
        # gentle init so gamma ~ 0 at start
        nn.init.zeros_(self.to_gamma.bias)
        nn.init.zeros_(self.to_beta.bias)
        nn.init.normal_(self.to_gamma.weight, std=1e-3)
        nn.init.normal_(self.to_beta.weight,  std=1e-3)

    def forward(self, h, c):
        gamma = self.to_gamma(c)
        beta  = self.to_beta(c)
        return h * (1 + gamma) + beta


class AdvDiscriminator(nn.Module):
    """
    Optional adversarial head trying to predict target gene from z_c, to enforce invariance
    The encoder is trained to fool this head (gradient reversal or detached loss flipping)
    """
    def __init__(self, latent_dim: int, num_genes: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(latent_dim, hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, num_genes),
                                 )

    def forward(self, z):
        return self.net(z)


class LowRankAdapter(nn.Module):
    """
    Applies a low-rank update h <- h + A (B^T h).
    A in R^{D x r}, B in R^{D x r}, r << D.
    """
    def __init__(self, dim: int, rank: int):
        super().__init__()
        self.dim = dim
        self.rank = max(1, int(rank))

    def forward(self, h: torch.Tensor, A: torch.Tensor, B: torch.Tensor):
        # h: [B, D], A,B: [B, D, r]
        # v = B^T h -> [B, r]
        v = torch.einsum("bdr,bd->br", B, h)
        upd = torch.einsum("bdr,br->bd", A, v)
        return h + upd


class HyperDelta(nn.Module):
    """
    Hypernetwork for Δ:
      - Produces a base delta vector in latent space (Dz)
      - Produces low-rank adapter factors A,B for latent modulation
    Inputs: gene embedding g [B, Dg], context u [B, Du]
    """
    def __init__(self, z_dim: int, g_dim: int, u_dim: int, hidden: int = 512, rank: int = 8, dropout: float = 0.1, use_ln: bool = True):
        super().__init__()
        self.rank = max(1, int(rank))
        self.z_dim = int(z_dim)

        in_dim = g_dim + u_dim
        h = hidden
        
        self.in_ln = nn.LayerNorm(in_dim) if use_ln else nn.Identity()
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, h), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(h, h), nn.GELU(), nn.Dropout(dropout),
        )
        self.h_ln = nn.LayerNorm(h) if use_ln else nn.Identity()
        
        # base delta (vector)
        self.to_delta = nn.Linear(h, z_dim)
        nn.init.zeros_(self.to_delta.weight); nn.init.zeros_(self.to_delta.bias)

        # factors A,B as flat outputs, then reshape to [B, D, r]
        self.to_A = nn.Linear(h, z_dim * self.rank)
        self.to_B = nn.Linear(h, z_dim * self.rank)
        # small init to keep early training stable
        nn.init.normal_(self.to_A.weight, std=1e-3); nn.init.zeros_(self.to_A.bias)
        nn.init.normal_(self.to_B.weight, std=1e-3); nn.init.zeros_(self.to_B.bias)
    
    def forward(self, g: torch.Tensor, u: torch.Tensor):
        x = torch.cat([g, u], dim=-1)
        x = self.in_ln(x)
        h = self.trunk(x)
        h = self.h_ln(h)
        delta_vec = torch.tanh(self.to_delta(h))
        A = self.to_A(h).view(-1, self.z_dim, self.rank)
        B = self.to_B(h).view(-1, self.z_dim, self.rank)
        return {"delta_vec": delta_vec, "A": A, "B": B}