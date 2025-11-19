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
# Gene-Perturbation Attention
# ============================================================================

class GenePerturbationAttention(nn.Module):
    """
    Multi-head cross-attention between cellular latent state and gene embeddings.
    
    This module learns which genes are most relevant for each perturbation by
    attending over gene embeddings conditioned on the cellular state.
    
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
        self.k_proj = nn.Linear(gene_embed_dim, latent_dim)
        self.v_proj = nn.Linear(gene_embed_dim, latent_dim)
        
        # Output projection to per-gene scores
        self.out_proj = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, num_genes)
        )
        
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
            gene_attention: [B, G] - per-gene attention scores
            attn_weights: [B, H, G] - raw attention weights (for visualization)
        """
        B = z_latent.size(0)
        G = gene_embeddings.size(0)
        
        # Expand gene embeddings for batch
        gene_emb_expanded = gene_embeddings.unsqueeze(0).expand(B, -1, -1)  # [B, G, D_gene]
        
        # Project to Q, K, V
        Q = self.q_proj(z_latent).unsqueeze(1)  # [B, 1, D]
        K = self.k_proj(gene_emb_expanded)      # [B, G, D]
        V = self.v_proj(gene_emb_expanded)      # [B, G, D]
        
        # Reshape for multi-head attention
        Q = Q.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, 1, D/H]
        K = K.view(B, G, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, G, D/H]
        V = V.view(B, G, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, G, D/H]
        
        # Compute attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [B, H, 1, G]
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights_dropout = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights_dropout, V)  # [B, H, 1, D/H]
        
        # Reshape and project to gene scores
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, self.latent_dim)
        gene_attention = self.out_proj(attn_output)  # [B, G]
        
        return gene_attention, attn_weights.squeeze(2)  # [B, G], [B, H, G]


class AttentionDeltaPredictor(nn.Module):
    """
    Predicts perturbation effects using attention mechanism.
    
    Combines:
    1. Global latent space shift (delta_global) via hypernet
    2. Gene-specific modulation (delta_gene) via attention
    
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
        
        # Gene-perturbation attention
        self.gene_attention = GenePerturbationAttention(
            latent_dim=latent_dim,
            gene_embed_dim=gene_embed_dim,
            num_genes=num_genes,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Global delta predictor (hypernet)
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
        
        # Gene-specific modulation
        self.gene_modulation = nn.Sequential(
            nn.Linear(num_genes, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, latent_dim),
            nn.Tanh()
        )
        
        # Learnable mixing parameter (initialized to 0.5)
        self.log_alpha = nn.Parameter(torch.tensor(0.0))  # sigmoid(0) = 0.5
        
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
            gene_attention: [B, G] - gene-level attention scores
            attn_weights: [B, H, G] - raw attention weights
        """
        # Global delta (latent space shift)
        global_input = torch.cat([z_c, gene_emb_target, context], dim=-1)
        delta_global = self.global_delta_net(global_input)  # [B, D]
        
        # Gene-specific attention
        gene_attention, attn_weights = self.gene_attention(z_c, gene_embeddings_all)  # [B, G], [B, H, G]
        
        # Gene-specific modulation of delta
        delta_gene_mod = self.gene_modulation(gene_attention)  # [B, D]
        
        # Combine global and gene-specific with learnable weight
        alpha = torch.sigmoid(self.log_alpha)
        delta = delta_global + alpha * delta_gene_mod
        
        return delta, gene_attention, attn_weights


class AttentionCountHead(nn.Module):
    """
    Count decoder with attention-based gene selection.
    
    Uses attention scores to modulate gene-specific rate parameters,
    boosting rates for genes that the attention mechanism identifies
    as relevant for the perturbation.
    
    Args:
        latent_dim: Dimension of latent space
        num_genes: Total number of genes
        nb_theta: Negative binomial dispersion parameter
        use_attention_modulation: Whether to use attention to modulate rates
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
        
        # Attention-to-rate gain (learnable)
        if use_attention_modulation:
            self.attn_gain = nn.Parameter(torch.ones(1) * 2.0)
        
    def forward(
        self,
        z: torch.Tensor,
        gene_attention: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            z: [B, D] - latent state
            gene_attention: [B, G] - optional attention scores
        
        Returns:
            rates: [B, G] - NB mean parameters (always positive)
        """
        # Base log-rates
        log_rates = self.count_decoder(z) + self.gene_bias  # [B, G]
        
        # Modulate with attention if provided and enabled
        if self.use_attention_modulation and gene_attention is not None:
            # Normalize attention to roughly [-1, 1] range centered at 0
            attn_normalized = torch.tanh(gene_attention * 0.5)
            
            # Modulate log-rates: boost attended genes, suppress others
            log_rates = log_rates + self.attn_gain * attn_normalized
        
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