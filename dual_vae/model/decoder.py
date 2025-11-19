import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import ResidualMLP, GaussianHead, FiLM
from ..config import ModelConfig
from ..utils import bounded_output_tanh, bounded_output_sigmoid

class Decoder(nn.Module):
    """Shared decoder: p(x | z, u) with optional FiLM conditioning on context u."""
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg   = cfg
        in_dim     = self.cfg.latent_dim + self.cfg.context_dim
        self.trunk = ResidualMLP(in_dim, self.cfg.hidden_dims[::-1], self.cfg.dropout, self.cfg.layernorm, use_input_skip=True)
        self.head  = GaussianHead(self.trunk.final_dim, self.cfg.input_dim)
        self.film  = FiLM(self.trunk.final_dim, self.cfg.context_dim) if self.cfg.use_film_in_decoder else None

        # Bounds for log1p space (typically [0, 8] for scRNA-seq)
        self.min_val = self.cfg.output_min
        self.max_val = self.cfg.output_max
        self.bound = self.cfg.bound_mode  # "tanh" (recommended), "sigmoid", or None

        # Optional count head for Poisson/NB/ZINB losses
        # Transforms bounded log1p predictions to positive count rates: rate = softplus(w * x_hat + b)
        if self.cfg.use_count_head:
            # Initialize to neutral: softplus(0) ≈ 0.693
            self.count_w = nn.Parameter(torch.zeros(self.cfg.input_dim))
            self.count_b = nn.Parameter(torch.zeros(self.cfg.input_dim))

            # ZINB zero-inflation head (predicts per-gene π from x_hat_log1p)
            count_link = self.cfg.count_link.lower()
            if count_link.startswith("zinb"):
                zinb_hidden = self.cfg.zinb_pi_hidden
                self.zinb_pi_head = nn.Sequential(
                    nn.Linear(self.cfg.input_dim, zinb_hidden),
                    nn.GELU(),
                    nn.Dropout(self.cfg.dropout),
                    nn.Linear(zinb_hidden, self.cfg.input_dim),
                )
                # Initialize to low zero-inflation: sigmoid(-2) ≈ 0.12
                with torch.no_grad():
                    self.zinb_pi_head[-1].bias.fill_(-2.0)
            else:
                self.zinb_pi_head = None

            def reset_count_head():
                """Reset count head parameters to neutral initialization."""
                with torch.no_grad():
                    self.count_w.zero_()
                    self.count_b.zero_()
                    if self.zinb_pi_head is not None:
                        for m in self.zinb_pi_head.modules():
                            if isinstance(m, nn.Linear):
                                nn.init.zeros_(m.weight)
                                nn.init.constant_(m.bias, -2.0)
            self.reset_count_head = reset_count_head
        else:
            self.register_parameter("count_w", None)
            self.register_parameter("count_b", None)
            self.zinb_pi_head = None

    def forward(self, z: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Decode latent z conditioned on context u to log1p expression space."""
        x_in = torch.cat([z, u], dim=-1)
        h = self.trunk(x_in, x_skip=x_in)

        if self.film is not None:
            h = self.film(h, u)

        x_hat = self.head(h)

        # Apply output bounds appropriate for log1p-normalized expression
        if self.bound == "sigmoid":
            x_hat = bounded_output_sigmoid(x_hat, min_val=self.min_val, max_val=self.max_val)
        elif self.bound == "tanh":
            x_hat = bounded_output_tanh(x_hat, min_val=self.min_val, max_val=self.max_val)
        else:
            x_hat = torch.clamp(x_hat, min=self.min_val, max=self.max_val)

        return x_hat

    def counts_rate(self, x_hat_log1p: torch.Tensor):
        """
        Convert bounded log1p predictions to positive count rates via learned affine transform.
        
        Returns:
            rate: [B, G] positive rates for Poisson/NB, or None if count head disabled
        """
        if (self.count_w is None) or (self.count_b is None):
            return None
        return F.softplus(x_hat_log1p * self.count_w + self.count_b)
    
    @staticmethod
    def _prune_topk(rate: torch.Tensor, k_keep: int | None) -> torch.Tensor:
        """
        Keep only top-K genes per cell by rate, zero out the rest.
        Does NOT renormalize internally (caller handles depth matching).
        """
        if not k_keep or k_keep <= 0:
            return rate
        N, G = rate.shape
        k = min(int(k_keep), G)
        vals, idx = torch.topk(rate, k=k, dim=1)
        pruned = torch.zeros_like(rate)
        pruned.scatter_(1, idx, vals)
        return pruned

    @staticmethod
    def _prune_quantile(rate: torch.Tensor, q: float | None) -> torch.Tensor:
        """
        Keep genes above per-cell quantile q, zero out the rest.
        Does NOT renormalize internally (caller handles depth matching).
        """
        if q is None or not (0.0 < float(q) < 1.0):
            return rate
        thr = torch.quantile(rate, float(q), dim=1, keepdim=True)
        pruned = torch.where(rate >= thr, rate, torch.zeros_like(rate))
        return pruned
    
    @torch.no_grad()
    def expected_log1p(
        self,
        x_hat_log1p: torch.Tensor,
        target_median_depth: float | None = None,
        rate_sharpen_beta: float = 1.0,      # Power transform for global sharpening (1.0 = off)
        mix_sharpen_p: float = 0.0,          # Blend ratio for soft sharpening (0.0 = hard)
        topk_keep_only: int | None = None,   # Sparsify: keep top-K genes per cell
        prune_quantile: float | None = None, # Alternative: keep genes above quantile
        topk_boost_k: int = 0,               # Boost top-K genes locally (0 = off)
        topk_boost_gamma: float = 1.0,       # Boost multiplier (>1.0 amplifies heads)
        zero_floor: float = 0.0,             # Hard zero threshold (diagnostic only)
        max_rate: float | None = None,       # Rate cap after all transforms (rarely needed)
        target_sum: float | None = None,     # Normalizes to this sum if provided
    ) -> torch.Tensor:
        """
        Generate deterministic expected log1p output with depth matching and optional transforms.
        
        Pipeline:
        1. Extract positive rates from count head (or expm1 fallback)
        2. Match to target depth
        3. Top-K boost (local head enhancement) → renorm
        4. Global sharpening (power transform) → renorm
        5. Sparsification (top-K or quantile) → renorm
        6. Optional rate cap
        7. Convert to normalized log1p
        
        Use for diagnostics/visualization. For competition metrics, prefer sample_counts() 
        followed by scanpy normalize_total() → log1p.
        """
        eps = 1e-8

        # 1) Extract positive rates
        if (self.count_w is None) or (self.count_b is None):
            rate = torch.expm1(x_hat_log1p)  # Fallback: inverse of log1p
        else:
            rate = self.counts_rate(x_hat_log1p)  # Learned count head
            
        # Optional hard sparsification for diagnostics
        if zero_floor and zero_floor > 0.0:
            rate = torch.where(rate < zero_floor, torch.zeros_like(rate), rate)

        # 2) Determine target per-cell depth
        if target_sum is not None:
            desired = float(target_sum)
        elif (target_median_depth is not None) and (target_median_depth > 0):
            desired = float(target_median_depth)
        else:
            desired = float(torch.median(rate.sum(dim=1)).item())

        def _renorm_to_desired(r: torch.Tensor) -> torch.Tensor:
            """Rescale rates to match desired per-cell total."""
            s = r.sum(dim=1, keepdim=True) + eps
            return r * (desired / s)

        # 3) Top-K boost: amplify local heads before global sharpening
        if (topk_boost_k is not None) and (topk_boost_k > 0) and \
           (topk_boost_gamma is not None) and (topk_boost_gamma > 1.0):
            K = min(int(topk_boost_k), rate.size(1))
            _, idx = torch.topk(rate, k=K, dim=1)
            mask = torch.zeros_like(rate, dtype=torch.bool)
            mask.scatter_(1, idx, True)
            rate = torch.where(mask, rate * float(topk_boost_gamma), rate)
            # rate = _renorm_to_desired(rate)

        # 4) Global sharpening: power transform to enhance dynamic range
        if rate_sharpen_beta and rate_sharpen_beta != 1.0:
            if rate_sharpen_beta <= 0:
                raise ValueError(f"rate_sharpen_beta must be positive, got {rate_sharpen_beta}")
            rate_orig = rate
            if mix_sharpen_p and mix_sharpen_p > 0.0:
                # Soft blend: mix original and sharpened
                sharpened = torch.pow(rate_orig, rate_sharpen_beta)
                rate = (1.0 - mix_sharpen_p) * rate_orig + mix_sharpen_p * sharpened
            else:
                # Hard sharpening
                rate = torch.pow(rate_orig, rate_sharpen_beta)
            # rate = _renorm_to_desired(rate)
        
        # 5) Sparsification: mimic real scRNA-seq dropout
        if topk_keep_only is not None and topk_keep_only > 0:
            rate = self._prune_topk(rate, topk_keep_only)
            # rate = _renorm_to_desired(rate)
        elif prune_quantile is not None:
            rate = self._prune_quantile(rate, prune_quantile)
            # rate = _renorm_to_desired(rate)

        # 6) Optional rate cap (prevents extreme values)
        if max_rate is not None:
            rate = torch.clamp(rate, min=0.0, max=float(max_rate))
        
        if desired is not None:
            rate = _renorm_to_desired(rate)

        # 7) Convert to normalized log1p expression
        probs = rate / (rate.sum(dim=1, keepdim=True) + eps)
        mu = probs * desired
        return torch.log1p(mu.clamp_min(0.0))

    @torch.no_grad()
    def sample_counts(
        self,
        x_hat_log1p: torch.Tensor,
        target_median_depth: float | None = None,   # Target per-cell UMI count
        max_rate: float | None = None,              # Rate cap (rarely needed)
        nb_theta: float | None = None,              # NB dispersion (lower = heavier tails)
        rate_sharpen_beta: float = 1.0,             # Global sharpening (1.0 = off; try 1.05–1.2)
        mix_sharpen_p: float = 0.0,                 # Soft sharpening blend (0.0 = hard)
        topk_keep_only: int | None = None,          # Sparsify: keep top-K genes
        prune_quantile: float | None = None,        # Alternative: keep above quantile
        topk_boost_k: int = 0,                      # Boost top-K locally (try 50–100)
        topk_boost_gamma: float = 1.0,              # Boost multiplier (try 1.5–2.5)
        link: str | None = None,                    # "nb" (default) or "poisson"
        allow_fallback_expm1: bool = False,         # Allow expm1 if count head missing
        count_clip_min: float = 0.0,                # Floor for rates (usually 0)
    ) -> torch.Tensor:
        """
        Sample integer counts from learned rate distribution with depth matching and transforms.
        
        Pipeline:
        1. Extract positive rates from count head
        2. Match to target depth
        3. Top-K boost (amplify local heads) → renorm
        4. Global sharpening (power transform) → renorm
        5. Sparsification (mimic dropout) → renorm
        6. Optional rate cap
        7. Sample counts via Negative Binomial (Gamma-Poisson mixture) or Poisson
        
        For VCC competition: Use NB with theta ≈ 6–10 for realistic overdispersion.
        """
        eps = 1e-8

        # ZINB not implemented in sampling; downgrade to NB
        if link and link.startswith("zinb"):
            link = "nb"

        # 1) Extract positive rates
        using_head = (self.count_w is not None) and (self.count_b is not None)
       
        if not using_head and not allow_fallback_expm1:
            raise RuntimeError("Count head disabled; enable cfg.use_count_head=True or allow_fallback_expm1=True.")
        
        rate = self.counts_rate(x_hat_log1p) if using_head else torch.expm1(x_hat_log1p)
       
        # Floor rates and clean numerical issues
        if count_clip_min and count_clip_min > 0.0:
            rate = torch.clamp_min(rate, float(count_clip_min))
        rate = torch.nan_to_num(rate, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)

        # 2) Determine target per-cell depth
        desired = float(target_median_depth) if (target_median_depth is not None and target_median_depth > 0) else None

        def _renorm_to_desired(r: torch.Tensor) -> torch.Tensor:
            """Rescale rates to match desired per-cell total (or preserve current if None)."""
            total = r.sum(dim=1, keepdim=True) + eps
            tgt = total if desired is None else torch.full_like(total, desired)
            return r * (tgt / total)

        # # Initial depth match (if target provided)
        # if desired is not None:
        #     rate = _renorm_to_desired(rate)

        # 3) Top-K boost: enhance local maxima before global transforms
        if topk_boost_k and topk_boost_k > 0 and topk_boost_gamma and topk_boost_gamma > 1.0:
            K = min(int(topk_boost_k), rate.size(1))
            _, idx = torch.topk(rate, k=K, dim=1)
            mask = torch.zeros_like(rate, dtype=torch.bool)
            mask.scatter_(1, idx, True)
            rate = torch.where(mask, rate * float(topk_boost_gamma), rate)
            # rate = _renorm_to_desired(rate)

        # 4) Global sharpening: power transform to increase dynamic range
        if rate_sharpen_beta and rate_sharpen_beta != 1.0:
            if rate_sharpen_beta <= 0:
                raise ValueError(f"rate_sharpen_beta must be positive, got {rate_sharpen_beta}")
            
            r0 = rate
            if mix_sharpen_p and mix_sharpen_p > 0.0:
                # Soft blend: partial application of power transform
                rate = (1.0 - mix_sharpen_p) * r0 + mix_sharpen_p * torch.pow(r0, rate_sharpen_beta)
            else:
                # Hard sharpening
                rate = torch.pow(r0, rate_sharpen_beta)
            # rate = _renorm_to_desired(rate)

        # 5) Sparsification: mimic scRNA-seq dropout patterns
        if topk_keep_only and topk_keep_only > 0:
            rate = self._prune_topk(rate, topk_keep_only)
            # rate = _renorm_to_desired(rate)
        elif prune_quantile is not None:
            rate = self._prune_quantile(rate, prune_quantile)
            # rate = _renorm_to_desired(rate)

        # 6) Optional rate cap (prevents sampling extreme outliers)
        if max_rate is not None:
            rate = torch.clamp(rate, min=0.0, max=float(max_rate))
        
        # Initial depth match (if target provided)
        if desired is not None:
            rate = _renorm_to_desired(rate)

        # 7) Sample integer counts
        if link and link.startswith("nb"):
            # Negative Binomial: Gamma-Poisson mixture for overdispersion
            theta = float(nb_theta) if nb_theta is not None else float(getattr(self.cfg, "nb_theta", 8.0))
            theta = max(theta, 0.1)  # Ensure positive
            
            # Sample Gamma latent rates, then Poisson counts
            lam = torch.zeros_like(rate)
            mask = rate > 0
            if mask.any().item():
                # Gamma parameterization: shape=theta, rate=theta/mean
                gp_rate = torch.clamp(theta / (rate[mask] + eps), max=1.0/eps)
                gamma = torch.distributions.Gamma(
                    concentration=torch.full_like(rate[mask], theta), 
                    rate=gp_rate
                )
                lam_mask = gamma.sample()
                if max_rate is not None:
                    lam_mask = torch.clamp_max(lam_mask, float(max_rate))
                lam[mask] = lam_mask
            y = torch.poisson(lam)
        else:
            # Poisson sampling (simpler, less realistic for scRNA-seq)
            y = torch.poisson(rate)

        return y.clamp_min_(0).to(torch.int64)