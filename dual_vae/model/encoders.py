from typing import Tuple
import torch
import torch.nn as nn
from .blocks import ResidualMLP
from ..config import ModelConfig

class EncoderC(nn.Module):
    """Control encoder q(z_c | x, u) -> (mu, logvar)."""
    def __init__(self, input_dim: int, cfg: ModelConfig, use_softplus_var: bool = False):
        super().__init__()
        self.cfg = cfg
        self.use_softplus_var = bool(use_softplus_var)

        in_dim      = input_dim + cfg.context_dim
        self.trunk  = ResidualMLP(in_dim, cfg.hidden_dims, cfg.dropout, cfg.layernorm, use_input_skip=True)
        self.mu     = nn.Linear(self.trunk.final_dim, cfg.latent_dim)
        self.logvar = nn.Linear(self.trunk.final_dim, cfg.latent_dim)

        # gentle init
        nn.init.zeros_(self.mu.bias)
        nn.init.constant_(self.logvar.bias, -2.0)  # std ≈ e^-1 ≈ 0.37

        # clamp bounds (allow override if present on cfg)
        self.logvar_min = self.cfg.logvar_min
        self.logvar_max = self.cfg.logvar_max

    def forward(self, x: torch.Tensor, u: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:        
        x_in = torch.cat([x, u], dim=-1)
        h = self.trunk(x_in, x_skip=x_in)
        mu = self.mu(h)
        if self.use_softplus_var:
            # logvar = log(softplus(raw))  → strictly positive variance w/ smooth grads
            raw = self.logvar(h)
            logvar = torch.log(torch.nn.functional.softplus(raw) + 1e-6)
        else:
            logvar = torch.clamp(self.logvar(h), min=self.logvar_min, max=self.logvar_max)
        return mu, logvar


class EncoderP(nn.Module):
    """Perturbed encoder q(z_p | x, u, g) -> (mu, logvar)."""
    def __init__(self, input_dim: int, cfg: ModelConfig, use_softplus_var: bool = False):
        super().__init__()
        self.cfg = cfg
        self.use_softplus_var = bool(use_softplus_var)

        in_dim      = input_dim + self.cfg.context_dim + self.cfg.gene_embed_dim
        self.trunk  = ResidualMLP(in_dim, self.cfg.hidden_dims, self.cfg.dropout, self.cfg.layernorm, use_input_skip=True)
        self.mu     = nn.Linear(self.trunk.final_dim, self.cfg.latent_dim)
        self.logvar = nn.Linear(self.trunk.final_dim, self.cfg.latent_dim)

        nn.init.zeros_(self.mu.bias)
        nn.init.constant_(self.logvar.bias, -2.0)

        self.logvar_min = self.cfg.logvar_min
        self.logvar_max = self.cfg.logvar_max

    def forward(self, x: torch.Tensor, u: torch.Tensor, g: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:            
        x_in = torch.cat([x, u, g], dim=-1)
        h = self.trunk(x_in, x_skip=x_in)
        mu = self.mu(h)
        if self.use_softplus_var:
            raw = self.logvar(h)
            logvar = torch.log(torch.nn.functional.softplus(raw) + 1e-6)
        else:
            logvar = torch.clamp(self.logvar(h), min=self.logvar_min, max=self.logvar_max)
        return mu, logvar