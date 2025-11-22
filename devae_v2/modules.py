"""
Attention modules and building blocks for AE-DEVAE.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List


# ============================================================================
# ResidualMLP with Skip Connections
# ============================================================================

class ResidualBlock(nn.Module):
    """Pre-activation residual MLP block with LayerNorm and GELU."""
    def __init__(self, dim: int, dropout: float = 0.1, layernorm: bool = True):
        super().__init__()
        self.ln = nn.LayerNorm(dim) if layernorm else nn.Identity()
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
    Stack of residual blocks with input projection and optional concat of 
    the ORIGINAL input at each down stage.
    
    This provides better gradient flow and information preservation compared
    to standard sequential MLPs.
    
    Args:
        in_dim: Input dimension
        hidden_dims: List of hidden layer dimensions (high→low)
        dropout: Dropout probability
        layernorm: Whether to use layer normalization
        use_input_skip: Whether to concatenate original input at each stage
    
    NOTE: Callers should pass x_skip=x_in (the original input) if using skip connections.
    """
    def __init__(
        self,
        in_dim: int,
        hidden_dims: List[int],
        dropout: float = 0.1,
        layernorm: bool = True,
        use_input_skip: bool = True
    ):
        super().__init__()
        self.use_input_skip = use_input_skip
        self.layernorm = layernorm
        self.dropout = dropout
        self.in_dim = in_dim
        self.hidden_dims = hidden_dims

        # Initial projection to first hidden width
        self.in_proj = nn.Linear(in_dim, hidden_dims[0])

        # Two pre-activation residual blocks at first hidden width
        self.pre_blocks = nn.ModuleList([
            ResidualBlock(hidden_dims[0], dropout, layernorm),
            ResidualBlock(hidden_dims[0], dropout, layernorm),
        ])

        # Build "down" stages with skip connection support
        self.downs = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            prev_w = hidden_dims[i]
            concat_w = prev_w + (in_dim if use_input_skip else 0)
            block = nn.ModuleDict({
                "ln": nn.LayerNorm(concat_w) if layernorm else nn.Identity(),
                "fc": nn.Linear(concat_w, hidden_dims[i+1]),
                "act": nn.GELU(),
                "drop": nn.Dropout(dropout),
            })
            self.downs.append(block)

        self.final_dim = hidden_dims[-1]

    def forward(self, x, x_skip=None):
        """
        Forward pass.
        
        Args:
            x: Input tensor [B, in_dim]
            x_skip: Optional original input for skip connections (should equal x initially)
        
        Returns:
            h: Output tensor [B, final_dim]
        """
        if x_skip is None:
            x_skip = x  # Use input as skip connection
        else:
            # Safety check
            if x_skip.shape[-1] != self.in_dim:
                raise ValueError(f"x_skip last dim {x_skip.shape[-1]} != in_dim {self.in_dim}")
        
        # Project and apply pre-activation residuals
        h = self.in_proj(x)
        for blk in self.pre_blocks:
            h = blk(h)

        # Down stages with skip connections
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


# ============================================================================
# CosineWithWarmup Scheduler 
# ============================================================================

class CosineWithWarmup:
    """
    Cosine annealing learning rate scheduler with linear warmup.
    
    Better than PyTorch's CosineAnnealingLR because:
    - Built-in linear warmup
    - Resume-friendly with start_step parameter
    - Per-param-group base_lr support
    
    Args:
        optimizer: PyTorch optimizer
        total_steps: Total number of training steps
        warmup_steps: Number of warmup steps
        base_lr: Base learning rate
        min_lr: Minimum learning rate
        start_step: Starting step (for resuming)
    """
    def __init__(
        self,
        optimizer,
        total_steps: int,
        warmup_steps: int,
        base_lr: float,
        min_lr: float = 0.0,
        start_step: int = 0
    ):
        self.optimizer = optimizer
        self.total_steps = max(1, int(total_steps))
        self.warmup_steps = max(0, int(warmup_steps))
        self.base_lr = float(base_lr)
        self.min_lr = float(min_lr)
        self.step_num = int(start_step)
        
        # Store base_lr in each param group
        for pg in self.optimizer.param_groups:
            pg.setdefault("base_lr", pg.get("lr", self.base_lr))
        
        # Initialize LR at start_step
        self._set_lr()

    def _lr_scale(self, t: int) -> float:
        """Compute LR scale factor at step t."""
        # Linear warmup
        if self.warmup_steps > 0 and t <= self.warmup_steps:
            return t / self.warmup_steps
        
        # Constant after warmup if no annealing
        if self.total_steps <= self.warmup_steps:
            return 1.0
        
        # Cosine annealing
        progress = (t - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        progress = min(1.0, max(0.0, progress))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
        
    def _set_lr(self):
        """Update learning rates in optimizer."""
        scale = self._lr_scale(self.step_num)
        for pg in self.optimizer.param_groups:
            base = pg.get("base_lr", self.base_lr)
            pg["lr"] = max(self.min_lr, base * scale)

    def step(self):
        """Step the scheduler (call after each optimizer.step())."""
        self.step_num += 1
        self._set_lr()
    
    def get_last_lr(self):
        """Get current learning rates (for logging)."""
        return [pg["lr"] for pg in self.optimizer.param_groups]


# ============================================================================
# Gene-Perturbation Attention (Optimized for Memory)
# ============================================================================

class GenePerturbationAttention(nn.Module):
    """
    Multi-head cross-attention between cellular latent state and gene embeddings.
    
    This module learns which genes are most relevant for each perturbation by
    attending over gene embeddings conditioned on the cellular state.
    
    MEMORY OPTIMIZED:
    Projects keys/values once per gene set, rather than expanding for every batch item.
    This fixes the massive memory overhead when G > 15k.
    
    Args:
        latent_dim: Dimension of cell latent representations
        gene_embed_dim: Dimension of gene embeddings
        num_genes: Total number of genes
        num_heads: Number of attention heads
        dropout: Dropout probability
    """
    def __init__(
        self,
        latent_dim: int,
        gene_embed_dim: int,
        num_genes: int,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.gene_embed_dim = gene_embed_dim
        self.num_heads = num_heads
        self.head_dim = latent_dim // num_heads
        
        assert latent_dim % num_heads == 0, f"latent_dim ({latent_dim}) must be divisible by num_heads ({num_heads})"
        
        # Projections
        self.q_proj = nn.Linear(latent_dim, latent_dim)
        
        # Key/Value projections (Input: Gene Emb Dim -> Latent Dim)
        self.k_proj = nn.Linear(gene_embed_dim, latent_dim)
        self.v_proj = nn.Linear(gene_embed_dim, latent_dim)
        
        # Output projection to aggregate heads
        self.out_proj = nn.Linear(latent_dim, latent_dim)
        
        self.scale = self.head_dim ** -0.5
        self.dropout = nn.Dropout(dropout)
        
        # Gentle initialization for stability
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with gentle scaling."""
        nn.init.xavier_uniform_(self.q_proj.weight, gain=0.5)
        nn.init.xavier_uniform_(self.k_proj.weight, gain=0.5)
        nn.init.xavier_uniform_(self.v_proj.weight, gain=0.5)
        nn.init.zeros_(self.q_proj.bias)
        nn.init.zeros_(self.k_proj.bias)
        nn.init.zeros_(self.v_proj.bias)
        
    def forward(
        self,
        z_latent: torch.Tensor,
        gene_embeddings: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            z_latent: [B, D] - cell latent representations
            gene_embeddings: [G, D_gene] - gene embeddings
        
        Returns:
            attn_output: [B, D] - aggregated context from attention
            attn_weights_avg: [B, G] - attention weights averaged over heads (The Gate Signal)
        """
        B = z_latent.size(0)
        G = gene_embeddings.size(0)
        
        # 1. Project Query (Per batch)
        # [B, D] -> [B, 1, H, D/H]
        # Reshape: [B, 1, D]
        Q = self.q_proj(z_latent).unsqueeze(1)
        # View heads: [B, 1, H, Dh] -> Permute to [B, H, 1, Dh]
        Q = Q.view(B, 1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # 2. Project Keys/Values (Once per gene set - Memory Efficient!)
        # [G, D_gene] -> [G, D]
        K_all = self.k_proj(gene_embeddings)
        V_all = self.v_proj(gene_embeddings)
        
        # Reshape for multi-head attention: [G, H, Dh]
        # Then permute to [1, H, Dh, G] (for K) and [1, H, G, Dh] (for V) to broadcast against B
        
        # Prepare K for matmul: [B, H, 1, Dh] @ [1, H, Dh, G] -> [B, H, 1, G]
        K_t = K_all.view(G, self.num_heads, self.head_dim).permute(1, 2, 0).unsqueeze(0)
        
        # Prepare V for matmul: [B, H, 1, G] @ [1, H, G, Dh] -> [B, H, 1, Dh]
        V = V_all.view(G, self.num_heads, self.head_dim).permute(1, 0, 2).unsqueeze(0)
        
        # 3. Compute Attention Scores
        attn_scores = torch.matmul(Q, K_t) * self.scale  # [B, H, 1, G]
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights_dropout = self.dropout(attn_weights)
        
        # 4. Apply attention to values
        attn_output = torch.matmul(attn_weights_dropout, V)  # [B, H, 1, Dh]
        
        # 5. Recombine heads and project
        # [B, H, 1, Dh] -> [B, 1, H, Dh] -> [B, D]
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(B, self.latent_dim)
        attn_output = self.out_proj(attn_output)  # [B, D]
        
        # 6. Average attention weights for the Gating mechanism
        # [B, H, 1, G] -> [B, G]
        attn_weights_avg = attn_weights.squeeze(2).mean(dim=1)
        
        return attn_output, attn_weights_avg


class AttentionDeltaPredictor(nn.Module):
    """
    Predicts perturbation effects using Disentangled Gating.
    
    Combines:
    1. Global latent space shift (delta) via hypernet -> MAGNITUDE/DIRECTION
    2. Gene-specific probability (gate) via attention -> SELECTION
    
    Args:
        latent_dim: Dimension of latent space
        gene_embed_dim: Dimension of gene embeddings
        num_genes: Total number of genes
        context_dim: Dimension of context embeddings (batch/celltype/etc)
        num_heads: Number of attention heads
    """
    def __init__(
        self,
        latent_dim: int,
        gene_embed_dim: int,
        num_genes: int,
        context_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_genes = num_genes
        
        # Gene-perturbation attention (Optimized)
        self.gene_attention = GenePerturbationAttention(
            latent_dim=latent_dim,
            gene_embed_dim=gene_embed_dim,
            num_genes=num_genes,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Global delta predictor (Physics of the perturbation)
        self.global_delta_net = nn.Sequential(
            nn.Linear(latent_dim + gene_embed_dim + context_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, latent_dim)
        )
        
        # Gate Refinement: Maps raw attention weights to sharp [0, 1] probabilities
        # Simple 1x1 conv equivalent (element-wise scaling)
        self.gate_scale = nn.Parameter(torch.ones(1) * 10.0) # Initialize sharp
        self.gate_bias = nn.Parameter(torch.zeros(1) - 2.0)  # Initialize sparse
        
        # Gentle initialization
        self._init_weights()
    
    def _init_weights(self):
        """Initialize final delta layer to near-zero for stable early training."""
        # Zero-init the final layer of global_delta_net
        nn.init.zeros_(self.global_delta_net[-1].weight)
        nn.init.zeros_(self.global_delta_net[-1].bias)
        
    def forward(
        self,
        z_c: torch.Tensor,
        gene_emb_target: torch.Tensor,
        gene_embeddings_all: torch.Tensor,
        context: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            z_c: [B, D] - control latent state
            gene_emb_target: [B, D_gene] - target gene embedding
            gene_embeddings_all: [G, D_gene] - all gene embeddings
            context: [B, D_context] - context embedding
        
        Returns:
            delta: [B, D] - perturbation effect in latent space
            gate_probs: [B, G] - probability of DE (The Gate)
            attn_weights: [B, G] - raw attention weights
        """
        # 1. Global Delta (Latent Space Shift)
        global_input = torch.cat([z_c, gene_emb_target, context], dim=-1)
        delta_global = self.global_delta_net(global_input)  # [B, D]
        
        # 2. Gene Attention (Selection/Gating)
        # attn_output is the aggregated context (can be used for delta refinement if needed)
        attn_output, attn_weights = self.gene_attention(z_c, gene_embeddings_all)  # [B, D], [B, G]
        
        # 3. Refine Attention into Gate Probabilities
        # Apply sharpening/bias and sigmoid
        gate_logits = attn_weights * self.gate_scale + self.gate_bias
        gate_probs = torch.sigmoid(gate_logits) # [B, G]
        
        return delta_global, gate_probs, attn_weights


class AttentionCountHead(nn.Module):
    """
    Count decoder with Gated Modulation.
    
    Uses the Gate Probabilities from the Delta Predictor to explicitly 
    boost or suppress changes in the count rates.
    
    Args:
        latent_dim: Dimension of latent space
        num_genes: Total number of genes
        nb_theta: Negative binomial dispersion parameter
        use_attention_modulation: Whether to use gate to modulate rates
    """
    def __init__(
        self,
        latent_dim: int,
        num_genes: int,
        nb_theta: float = 10.0,
        use_attention_modulation: bool = True
    ):
        super().__init__()
        self.num_genes = num_genes
        self.nb_theta = nb_theta
        self.use_attention_modulation = use_attention_modulation
        
        # Base count predictor
        self.count_decoder = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 2048),
            nn.LayerNorm(2048),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(2048, num_genes)
        )
        
        # Learnable gene-specific biases
        self.gene_bias = nn.Parameter(torch.zeros(num_genes))
        
        # Gate modulation gain (how strong is the gate's effect?)
        if use_attention_modulation:
            self.mod_gain = nn.Parameter(torch.ones(1) * 2.0)
        
    def forward(
        self,
        z: torch.Tensor,
        gate_probs: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            z: [B, D] - latent state
            gate_probs: [B, G] - optional gate probabilities [0, 1]
        
        Returns:
            rates: [B, G] - NB mean parameters (always positive)
        """
        # Base log-rates
        log_rates = self.count_decoder(z) + self.gene_bias  # [B, G]
        
        # Modulate with gate if provided and enabled
        if self.use_attention_modulation and gate_probs is not None:
            # Boost genes where gate_prob is high
            # This helps the model separate "background noise" (gate=0) from "signal" (gate=1)
            log_rates = log_rates + self.mod_gain * gate_probs
        
        # Ensure positive rates via softplus
        rates = F.softplus(log_rates)
        
        return rates


class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation layer.
    
    Applies affine transformation γ * x + β where γ and β
    are predicted from a conditioning signal.
    """
    def __init__(self, feature_dim: int, cond_dim: int):
        super().__init__()
        self.scale_shift_net = nn.Linear(cond_dim, 2 * feature_dim)
        
        # Gentle initialization (start near identity)
        nn.init.zeros_(self.scale_shift_net.bias)
        nn.init.normal_(self.scale_shift_net.weight, std=1e-3)
        
    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, D_feat] - features to modulate
            cond: [B, D_cond] - conditioning signal
        
        Returns:
            modulated: [B, D_feat]
        """
        scale_shift = self.scale_shift_net(cond)  # [B, 2*D_feat]
        scale, shift = scale_shift.chunk(2, dim=-1)
        return x * (1 + scale) + shift