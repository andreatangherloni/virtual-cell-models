"""
Main model architecture for AE-DEVAE (Attention-Enhanced Dual Encoder VAE).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from .config import ModelConfig
from .modules import (AttentionDeltaPredictor,
                      AttentionCountHead,
                      FiLMLayer,
                      ResidualMLP
                      )


class Encoder(nn.Module):
    """
    Encoder network: x → (μ, logσ²) with ResidualMLP option and softplus variance.
    
    Args:
        input_dim: Input dimension (number of genes)
        latent_dim: Latent space dimension
        hidden_dims: List of hidden layer dimensions
        dropout: Dropout probability
        use_layernorm: Whether to use layer normalization
        use_residual_mlp: Whether to use ResidualMLP with skip connections
        use_input_skip: Whether to use input skip connections in ResidualMLP
        use_softplus_var: Whether to use softplus for smooth positive variance
    """
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dims: list,
        dropout: float = 0.2,
        use_layernorm: bool = True,
        use_residual_mlp: bool = True,
        use_input_skip: bool = True,
        use_softplus_var: bool = True,
        logvar_min: float = -4.0,
        logvar_max: float = 2.0
    ):
        super().__init__()
        self.use_softplus_var = use_softplus_var
        self.logvar_min = logvar_min
        self.logvar_max = logvar_max
        
        # Use ResidualMLP if requested, otherwise standard Sequential
        if use_residual_mlp:
            self.encoder = ResidualMLP(
                in_dim=input_dim,
                hidden_dims=hidden_dims,
                dropout=dropout,
                layernorm=use_layernorm,
                use_input_skip=use_input_skip
            )
            final_dim = self.encoder.final_dim
        else:
            # Standard sequential encoder (fallback)
            layers = []
            dims = [input_dim] + hidden_dims
            
            for i in range(len(dims) - 1):
                layers.append(nn.Linear(dims[i], dims[i+1]))
                if use_layernorm:
                    layers.append(nn.LayerNorm(dims[i+1]))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
            
            self.encoder = nn.Sequential(*layers)
            final_dim = hidden_dims[-1]
        
        # Separate heads for mean and log-variance
        self.mu_head = nn.Linear(final_dim, latent_dim)
        self.logvar_head = nn.Linear(final_dim, latent_dim)
        
        # Gentle initialization for stable early training
        nn.init.zeros_(self.mu_head.bias)
        nn.init.constant_(self.logvar_head.bias, -2.0)  # std ≈ e^(-1) ≈ 0.37
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, G] - input gene expression
        
        Returns:
            mu: [B, D] - latent mean
            logvar: [B, D] - latent log-variance
        """
        # Encode
        if isinstance(self.encoder, ResidualMLP):
            h = self.encoder(x, x_skip=x)  # Pass input as skip connection
        else:
            h = self.encoder(x)
        
        # Predict mean
        mu = self.mu_head(h)
        
        # Predict variance with optional softplus for smooth gradients
        if self.use_softplus_var:
            # Softplus ensures strictly positive variance with smooth gradients
            raw_logvar = self.logvar_head(h)
            logvar = torch.log(F.softplus(raw_logvar) + 1e-6)
        else:
            # Standard clamping
            logvar = torch.clamp(self.logvar_head(h), self.logvar_min, self.logvar_max)
        
        return mu, logvar


class Decoder(nn.Module):
    """
    Decoder network: z → x̂ with ResidualMLP option.
    
    Args:
        latent_dim: Latent space dimension
        output_dim: Output dimension (number of genes)
        hidden_dims: List of hidden layer dimensions (reversed from encoder)
        dropout: Dropout probability
        use_layernorm: Whether to use layer normalization
        use_film: Whether to use FiLM conditioning
        context_dim: Dimension of context for FiLM (if used)
        bound_mode: Output bounding ("tanh", "sigmoid", or None)
        output_min: Minimum output value (if bounding)
        output_max: Maximum output value (if bounding)
        use_residual_mlp: Whether to use ResidualMLP
        use_input_skip: Whether to use input skip connections
    """
    def __init__(
        self,
        latent_dim: int,
        output_dim: int,
        hidden_dims: list,
        dropout: float = 0.2,
        use_layernorm: bool = True,
        use_film: bool = False,
        context_dim: int = 256,
        bound_mode: str = "tanh",
        output_min: float = 0.0,
        output_max: float = 8.0,
        use_residual_mlp: bool = True,
        use_input_skip: bool = True
    ):
        super().__init__()
        self.use_film = use_film
        self.bound_mode = bound_mode
        self.output_min = output_min
        self.output_max = output_max
        
        # Reverse hidden dims for decoder
        hidden_dims = list(reversed(hidden_dims))
        
        # Use ResidualMLP if requested
        if use_residual_mlp:
            self.decoder = ResidualMLP(
                in_dim=latent_dim,
                hidden_dims=hidden_dims,
                dropout=dropout,
                layernorm=use_layernorm,
                use_input_skip=use_input_skip
            )
            final_dim = self.decoder.final_dim
            self.film_layers = None  # FiLM not yet integrated with ResidualMLP
        else:
            # Standard decoder with FiLM support
            dims = [latent_dim] + hidden_dims
            layers = []
            film_layers = []
            
            for i in range(len(dims) - 1):
                layers.append(nn.Linear(dims[i], dims[i+1]))
                if use_layernorm:
                    layers.append(nn.LayerNorm(dims[i+1]))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
                
                if use_film:
                    film_layers.append(FiLMLayer(dims[i+1], context_dim))
            
            self.decoder = nn.ModuleList(layers)
            self.film_layers = nn.ModuleList(film_layers) if use_film else None
            final_dim = hidden_dims[-1]
        
        # Output projection
        self.output_proj = nn.Linear(final_dim, output_dim)
        
    def forward(
        self,
        z: torch.Tensor,
        context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            z: [B, D] - latent representation
            context: [B, D_context] - optional context for FiLM
        
        Returns:
            x_hat: [B, G] - reconstructed gene expression
        """
        # Decode
        if isinstance(self.decoder, ResidualMLP):
            h = self.decoder(z, x_skip=z)
        else:
            h = z
            film_idx = 0
            
            for i, layer in enumerate(self.decoder):
                h = layer(h)
                
                # Apply FiLM after ReLU activations
                if self.use_film and isinstance(layer, nn.ReLU):
                    if context is not None and film_idx < len(self.film_layers):
                        h = self.film_layers[film_idx](h, context)
                        film_idx += 1
        
        x_hat = self.output_proj(h)
        
        # Apply output bounds
        if self.bound_mode == "tanh":
            mid = (self.output_max + self.output_min) / 2
            scale = (self.output_max - self.output_min) / 2
            x_hat = mid + scale * torch.tanh(x_hat)
        elif self.bound_mode == "sigmoid":
            x_hat = self.output_min + (self.output_max - self.output_min) * torch.sigmoid(x_hat)
        
        return x_hat


class AttentionEnhancedDualEncoderVAE(nn.Module):
    """
    Attention-Enhanced Dual Encoder VAE for perturbation prediction with:
    - ResidualMLP encoders/decoders with skip connections
    - Softplus variance for smooth gradients
    - Neighbor mixing for unseen gene generalization
    - Gentle initialization for stable training
    
    Architecture:
    1. Dual encoders (control + perturbed) with ResidualMLP
    2. Disentangled Delta Predictor (Gate + Magnitude)
    3. Decoder with optional FiLM conditioning
    4. Gated Count Head
    
    Args:
        cfg: Model configuration
    """
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        
        # Auto-detect input dim if not provided
        if cfg.input_dim is None:
            raise ValueError("input_dim must be provided or detected from data")
        
        # ============================================================
        # 1. ENCODERS (with ResidualMLP)
        # ============================================================
        
        self.enc_c = Encoder(
            input_dim=cfg.input_dim,
            latent_dim=cfg.latent_dim,
            hidden_dims=cfg.hidden_dims,
            dropout=cfg.dropout,
            use_layernorm=cfg.layernorm,
            use_residual_mlp=cfg.use_residual_mlp,
            use_input_skip=cfg.use_input_skip,
            use_softplus_var=cfg.use_softplus_var,
            logvar_min=cfg.logvar_min,
            logvar_max=cfg.logvar_max
        )
        
        self.enc_p = Encoder(
            input_dim=cfg.input_dim,
            latent_dim=cfg.latent_dim,
            hidden_dims=cfg.hidden_dims,
            dropout=cfg.dropout,
            use_layernorm=cfg.layernorm,
            use_residual_mlp=cfg.use_residual_mlp,
            use_input_skip=cfg.use_input_skip,
            use_softplus_var=cfg.use_softplus_var,
            logvar_min=cfg.logvar_min,
            logvar_max=cfg.logvar_max
        )
        
        # ============================================================
        # 2. EMBEDDINGS (Auto-sized from data)
        # ============================================================

        # Gene embeddings (can be initialized from pretrained)
        self.gene_emb = nn.Embedding(cfg.input_dim, cfg.gene_embed_dim)
        nn.init.normal_(self.gene_emb.weight, mean=0.0, std=0.02)

        # Context embeddings - sizes will be set from data
        self.batch_emb = None
        self.celltype_emb = None
        self.h1_emb = nn.Embedding(2, cfg.context_dim // 4)  # Binary, always 2

        # These will be set when we see the data
        self.num_batches = None
        self.num_celltypes = None

        # Library size projection (if used)
        if cfg.use_libsize_covariate:
            self.libsize_proj = nn.Sequential(
                nn.Linear(1, cfg.libsize_proj_hidden),
                nn.ReLU(),
                nn.Linear(cfg.libsize_proj_hidden, cfg.context_dim // 4)
            )

        # Context fusion - dynamically sized input
        context_input_dim = 3 * (cfg.context_dim // 4)  # batch + celltype + h1
        if cfg.use_libsize_covariate:
            context_input_dim += cfg.context_dim // 4  # + libsize

        self.context_fusion = nn.Sequential(
            nn.Linear(context_input_dim, cfg.context_dim),
            nn.LayerNorm(cfg.context_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout)
        )
        
        # ============================================================
        # 3. DELTA PREDICTOR (Attention-based)
        # ============================================================
        
        if cfg.use_attention:
            self.delta_predictor = AttentionDeltaPredictor(
                latent_dim=cfg.latent_dim,
                gene_embed_dim=cfg.gene_embed_dim,
                num_genes=cfg.input_dim,
                context_dim=cfg.context_dim,
                num_heads=cfg.num_attention_heads,
                dropout=cfg.attention_dropout
            )
        else:
            # Fallback to simple hypernet (no attention)
            self.delta_predictor = nn.Sequential(
                nn.Linear(cfg.latent_dim + cfg.gene_embed_dim + cfg.context_dim, 512),
                nn.LayerNorm(512),
                nn.ReLU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(512, 256),
                nn.LayerNorm(256),
                nn.ReLU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(256, cfg.latent_dim)
            )
        
        # ============================================================
        # 4. DECODER (with ResidualMLP)
        # ============================================================
        
        self.dec = Decoder(
            latent_dim=cfg.latent_dim,
            output_dim=cfg.input_dim,
            hidden_dims=cfg.hidden_dims,
            dropout=cfg.dropout,
            use_layernorm=cfg.layernorm,
            use_film=cfg.use_film_in_decoder,
            context_dim=cfg.context_dim,
            bound_mode=cfg.bound_mode,
            output_min=cfg.output_min,
            output_max=cfg.output_max,
            use_residual_mlp=cfg.use_residual_mlp,
            use_input_skip=cfg.use_input_skip 
        )
        
        # ============================================================
        # 5. COUNT HEAD
        # ============================================================
        
        if cfg.use_count_head:
            self.count_head = AttentionCountHead(
                latent_dim=cfg.latent_dim,
                num_genes=cfg.input_dim,
                nb_theta=cfg.nb_theta,
                use_attention_modulation=cfg.attention_modulated_counts
            )
        else:
            self.count_head = None
        
        # ZINB pi head (if using ZINB)
        if cfg.count_link == "zinb":
            self.zinb_pi_head = nn.Sequential(
                nn.Linear(cfg.latent_dim, cfg.zinb_pi_hidden),
                nn.ReLU(),
                nn.Linear(cfg.zinb_pi_hidden, cfg.input_dim)
            )
        else:
            self.zinb_pi_head = None
        
        # ============================================================
        # 6. ADDITIONAL ATTRIBUTES
        # ============================================================
        
        self.logvar_min = cfg.logvar_min
        self.logvar_max = cfg.logvar_max
        self.nb_theta = cfg.nb_theta
        self.gene_w = None  # Optional per-gene weights
        
    def _init_context_embeddings(self, num_batches: int, num_celltypes: int):
        """
        Initialize context embeddings with correct sizes from data.
        
        Args:
            num_batches: Number of unique batches
            num_celltypes: Number of unique cell types
        """
        device = next(self.parameters()).device
        
        self.num_batches = num_batches
        self.num_celltypes = num_celltypes
        
        # Create embeddings with correct sizes
        self.batch_emb = nn.Embedding(num_batches, self.cfg.context_dim // 4).to(device)
        self.celltype_emb = nn.Embedding(num_celltypes, self.cfg.context_dim // 4).to(device)
        
        # Gentle initialization
        nn.init.normal_(self.batch_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.celltype_emb.weight, mean=0.0, std=0.02)
        
        print(f"[model] ✓ Model correctly initialized: {num_batches} batches, {num_celltypes} cell types, {self.gene_emb.num_embeddings:,} genes")
    
    def _init_gene_embeddings(self, path: str):
        """
        Initialize gene embeddings from pretrained file.
        
        Enhanced to handle:
        - Dictionary format: {'gene_name': tensor([...]), ...}
        - Tensor format: [num_genes, embed_dim]
        - Missing genes: random initialization
        - Extra genes: filtered out
        - Case-insensitive matching (optional)
        
        Args:
            path: Path to pretrained embeddings (.pt file)
        """
        try:
            print(f"[gene_emb] Loading pretrained embeddings from {path}")
            pretrained = torch.load(path, map_location='cpu')
            
            # ====================================================================
            # PARSE PRETRAINED FORMAT
            # ====================================================================
            
            if isinstance(pretrained, dict):
                # Check if it's a wrapper dict with 'embeddings' key
                if 'embeddings' in pretrained:
                    pretrained = pretrained['embeddings']
                
                # NEW: Handle dictionary format {'gene_name': tensor}
                if all(isinstance(k, str) for k in list(pretrained.keys())[:10]):
                    print(f"[gene_emb] Detected dictionary format with gene names")
                    emb_dict = pretrained
                    is_dict_format = True
                else:
                    # Old format: assume it's a tensor
                    emb_dict = None
                    is_dict_format = False
            else:
                # Direct tensor format
                emb_dict = None
                is_dict_format = False
            
            # ====================================================================
            # DICTIONARY FORMAT: Match gene names
            # ====================================================================
            
            if is_dict_format and emb_dict is not None:
                print(f"[gene_emb] Matching genes by name...")
                
                # Get model gene names (from self.gene_emb)
                num_genes = self.gene_emb.num_embeddings
                embed_dim_target = self.cfg.gene_embed_dim
                
                # Get first embedding to determine source dimension
                first_emb = next(iter(emb_dict.values()))
                embed_dim_source = first_emb.size(0)
                
                print(f"[gene_emb] Source embeddings: {len(emb_dict)} genes, dim={embed_dim_source}")
                print(f"[gene_emb] Target model: {num_genes} genes, dim={embed_dim_target}")
                
                # ✅ Case-insensitive matching (optional)
                if self.cfg.pretrained_case_insensitive:
                    print(f"[gene_emb] Using case-insensitive matching")
                    emb_dict_lower = {k.lower(): v for k, v in emb_dict.items()}
                else:
                    emb_dict_lower = emb_dict
                
                # Get gene names from model (need to be set from dataset)
                # This requires gene_names to be stored when _init_context_embeddings is called
                if not hasattr(self, 'gene_names'):
                    raise RuntimeError(
                        "gene_names not set! Call model._set_gene_names(gene_list) before loading embeddings"
                    )
                
                gene_names = self.gene_names
                
                # Build embedding matrix
                matched = 0
                missing = 0
                
                # Initialize with random (for missing genes)
                emb_matrix = torch.randn(num_genes, embed_dim_source) * 0.02
                
                for idx, gene_name in enumerate(gene_names):
                    # Try to find embedding
                    lookup_name = gene_name.lower() if self.cfg.pretrained_case_insensitive else gene_name
                    
                    if lookup_name in emb_dict_lower:
                        emb_matrix[idx] = emb_dict_lower[lookup_name]
                        matched += 1
                    else:
                        # Already initialized randomly above
                        missing += 1
                
                print(f"[gene_emb] Matched: {matched}/{num_genes} genes ({100*matched/num_genes:.1f}%)")
                if missing > 0:
                    print(f"[gene_emb] Missing: {missing} genes (randomly initialized)")
                
                emb = emb_matrix
                
            # ====================================================================
            # TENSOR FORMAT: Direct size matching
            # ====================================================================
            
            else:
                if isinstance(pretrained, dict):
                    emb = pretrained  # Shouldn't happen, but fallback
                else:
                    emb = pretrained
                
                print(f"[gene_emb] Detected tensor format: {emb.shape}")
                
                # Check size match
                if emb.size(0) != self.gene_emb.num_embeddings:
                    print(f"[gene_emb] ⚠ Size mismatch: pretrained has {emb.size(0)} genes, model has {self.gene_emb.num_embeddings}")
                    print(f"[gene_emb] Using random initialization")
                    return
            
            # ====================================================================
            # DIMENSION PROJECTION (if needed)
            # ====================================================================
            
            if emb.size(1) != self.cfg.gene_embed_dim:
                print(f"[gene_emb] Projecting from dim {emb.size(1)} → {self.cfg.gene_embed_dim}")
                proj = nn.Linear(emb.size(1), self.cfg.gene_embed_dim, bias=False)
                nn.init.xavier_uniform_(proj.weight)
                emb = proj(emb)
            
            # ====================================================================
            # NORMALIZATION (optional)
            # ====================================================================
            
            if self.cfg.pretrained_norm == "l2":
                print(f"[gene_emb] Applying L2 normalization")
                emb = F.normalize(emb, p=2, dim=1)
            elif self.cfg.pretrained_norm == "unit":
                print(f"[gene_emb] Applying unit normalization")
                emb = emb / emb.norm(dim=1, keepdim=True).clamp_min(1e-8)
            
            # ====================================================================
            # COPY TO MODEL
            # ====================================================================
            
            self.gene_emb.weight.data.copy_(emb)
            print(f"[gene_emb] ✓ Successfully loaded pretrained embeddings")
            
        except Exception as e:
            print(f"[gene_emb] ✗ Failed to load pretrained: {e}")
            print(f"[gene_emb] Using random initialization")
            import traceback
            traceback.print_exc()


    def _set_gene_names(self, gene_names: list):
        """
        Set gene names for embedding matching.
            
        Args:
            gene_names: List of gene names in same order as gene_emb indices
        """
        if len(gene_names) != self.gene_emb.num_embeddings:
            raise ValueError(
                f"gene_names length ({len(gene_names)}) must match "
                f"gene_emb.num_embeddings ({self.gene_emb.num_embeddings})"
            )
        
        self.gene_names = gene_names
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick: z = μ + σ * ε, where ε ~ N(0, 1)
        
        Args:
            mu: [B, D] - latent mean
            logvar: [B, D] - latent log-variance
        
        Returns:
            z: [B, D] - sampled latent
        """
        # Clamp logvar for stability
        logvar = torch.clamp(logvar, self.logvar_min, self.logvar_max)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def encode_control(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode control cells."""
        mu, logvar = self.enc_c(x)
        logvar = torch.clamp(logvar, self.logvar_min, self.logvar_max)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar
    
    def encode_perturbed(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode perturbed cells."""
        mu, logvar = self.enc_p(x)
        logvar = torch.clamp(logvar, self.logvar_min, self.logvar_max)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar
    
    def build_context(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Build context embedding from batch metadata.
        """
        # Check if embeddings are initialized
        if self.batch_emb is None or self.celltype_emb is None:
            raise RuntimeError(
                "Context embeddings not initialized! "
                "Call model._init_context_embeddings() before forward pass."
            )
        
        batch_id = batch.get('batch_id', torch.zeros(batch['x'].size(0), dtype=torch.long, device=batch['x'].device))
        celltype_id = batch.get('celltype_id', torch.zeros(batch['x'].size(0), dtype=torch.long, device=batch['x'].device))
        is_h1 = batch.get('is_h1', torch.zeros(batch['x'].size(0), dtype=torch.long, device=batch['x'].device))
        
        # Clamp IDs to valid range (based on actual vocab sizes)
        batch_id = torch.clamp(batch_id, 0, self.num_batches - 1)
        celltype_id = torch.clamp(celltype_id, 0, self.num_celltypes - 1)
        is_h1 = torch.clamp(is_h1, 0, 1)
        
        # Embed
        batch_emb = self.batch_emb(batch_id)
        celltype_emb = self.celltype_emb(celltype_id)
        h1_emb = self.h1_emb(is_h1)
        
        # Collect context parts
        context_parts = [batch_emb, celltype_emb, h1_emb]
        
        # Add library size if used
        if self.cfg.use_libsize_covariate and 'libsize' in batch:
            libsize = batch['libsize'].unsqueeze(-1)
            libsize_emb = self.libsize_proj(libsize)
            context_parts.append(libsize_emb)
        
        # Concatenate and fuse
        context = torch.cat(context_parts, dim=-1)
        context = self.context_fusion(context)
        
        return context
    
    def predict_delta(
        self,
        z_c: torch.Tensor,
        target_gene_idx: torch.Tensor,
        context: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Predict perturbation effect delta AND gate probabilities.
        
        Args:
            z_c: [B, D] - control latent state
            target_gene_idx: [B] - GENE INDEX (0 to num_genes-1)
            context: [B, D_context] - context embedding
        
        Returns:
            delta: [B, D] - perturbation effect (Magnitude)
            gate_probs: [B, G] - probability of DE (Selection)
            attn_weights: [B, G] - raw attention weights
        """
        # Use gene embeddings directly (works for ANY gene, including unseen targets!)
        target_gene_idx = torch.clamp(target_gene_idx, 0, self.gene_emb.num_embeddings - 1)
        gene_emb_target = self.gene_emb(target_gene_idx)  # [B, D_gene]
        
        # Get all gene embeddings for attention
        gene_embeddings_all = self.gene_emb.weight  # [G, D_gene]
        
        if self.cfg.use_attention:
            # Gated prediction: Delta + Gate
            delta, gate_probs, attn_weights = self.delta_predictor(
                z_c, gene_emb_target, gene_embeddings_all, context
            )
            return delta, gate_probs, attn_weights
        else:
            # Simple hypernet (no attention)
            delta_input = torch.cat([z_c, gene_emb_target, context], dim=-1)
            delta = self.delta_predictor(delta_input)
            return delta, None, None
    
    # ============================================================================
    # NEIGHBOR MIXING FOR UNSEEN GENES
    # ============================================================================
    
    @torch.no_grad()
    def predict_with_neighbor_mixing(
        self,
        z_c: torch.Tensor,
        context: torch.Tensor,
        target_gene_idx: torch.Tensor,
        k: int = 8,
        tau: float = 0.07,
        include_self: bool = True,
        delta_gain: float = 1.0
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Predict perturbation with k-nearest-neighbor mixing in gene embedding space.
        
        This is crucial for generalizing to unseen perturbations!
        For a new gene, we average the predicted effects of its k nearest neighbors.
        
        Args:
            z_c: [B, D] - control latent state
            context: [B, D_context] - context embedding
            target_gene_idx: [B] - target gene indices
            k: Number of nearest neighbors
            tau: Softmax temperature (lower = more focused)
            include_self: Whether to include query gene in mixture
            delta_gain: Scale factor for delta
        
        Returns:
            x_pred: [B, G] - predicted perturbed expression
            gate_mixed: [B, G] - mixed gate probabilities (if using attention)
        """
        device = z_c.device
        B = z_c.size(0)
        
        # Get gene embeddings
        gene_emb_all = self.gene_emb.weight  # [G, D_gene]
        target_emb = self.gene_emb(target_gene_idx)  # [B, D_gene]
        
        # Normalize for cosine similarity
        target_norm = F.normalize(target_emb, dim=-1)
        gene_norm = F.normalize(gene_emb_all, dim=-1)
        
        # Compute cosine similarity
        sims = target_norm @ gene_norm.t()  # [B, G]
        
        # Get top-k similar genes
        k_eff = min(k, gene_emb_all.size(0) - 1)
        topk_sims, topk_idx = torch.topk(sims, k=k_eff, dim=1)  # [B, k]
        
        # Include self if requested
        if include_self:
            neighbor_embs = torch.cat([
                target_emb.unsqueeze(1),
                gene_emb_all[topk_idx]
            ], dim=1)  # [B, k+1, D_gene]
            
            neighbor_sims = torch.cat([
                torch.ones(B, 1, device=device),
                topk_sims
            ], dim=1)  # [B, k+1]
        else:
            neighbor_embs = gene_emb_all[topk_idx]  # [B, k, D_gene]
            neighbor_sims = topk_sims  # [B, k]
        
        # Softmax weights with temperature
        weights = F.softmax(neighbor_sims / tau, dim=1)  # [B, K]
        K = neighbor_embs.size(1)
        
        # Predict delta for each neighbor
        deltas = []
        gates = []
        
        for i in range(K):
            neighbor_emb = neighbor_embs[:, i, :]  # [B, D_gene]
            
            # Predict delta for this neighbor gene
            if self.cfg.use_attention:
                delta, gate_probs, _ = self.delta_predictor(
                    z_c, neighbor_emb, gene_emb_all, context
                )
                if gate_probs is not None:
                    gates.append(gate_probs)
            else:
                delta_input = torch.cat([z_c, neighbor_emb, context], dim=-1)
                delta = self.delta_predictor(delta_input)
            
            deltas.append(delta)
        
        # Stack and weight
        deltas_stacked = torch.stack(deltas, dim=1)  # [B, K, D_latent]
        delta_mixed = (weights.unsqueeze(-1) * deltas_stacked).sum(dim=1)  # [B, D_latent]
        
        # Scale delta if requested
        if delta_gain != 1.0:
            delta_mixed = delta_mixed * delta_gain
        
        # Average gates if available
        if gates:
            gates_stacked = torch.stack(gates, dim=1)  # [B, K, G]
            gate_mixed = (weights.unsqueeze(-1) * gates_stacked).sum(dim=1)  # [B, G]
        else:
            gate_mixed = None
        
        # Decode
        x_pred = self.decode(z_c + delta_mixed, context)
        
        return x_pred, gate_mixed
    
    # ============================================================================
    # Standard methods
    # ============================================================================
    
    def decode(
        self,
        z: torch.Tensor,
        context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Decode latent to gene expression.
        
        Args:
            z: [B, D] - latent representation
            context: [B, D_context] - optional context for FiLM
        
        Returns:
            x_hat: [B, G] - reconstructed gene expression (log1p space)
        """
        return self.dec(z, context)
    
    def counts_rate(
        self,
        z: torch.Tensor,
        gate_probs: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Predict count rate parameters.
        
        Args:
            z: [B, D] - latent representation
            gate_probs: [B, G] - optional gate probabilities
        
        Returns:
            rates: [B, G] - NB/Poisson rate parameters
        """
        if self.count_head is None:
            return None
        return self.count_head(z, gate_probs)
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Full forward pass.
        
        Args:
            batch: Dictionary containing:
                - x: [B, G] gene expression (log1p normalized)
                - is_control: [B] binary mask (1=control, 0=perturbed)
                - gene_idx: [B] target gene indices
                - batch_id, celltype_id, is_h1, libsize: metadata
        
        Returns:
            Dictionary containing all intermediate outputs and predictions
        """
        x = batch['x']
        is_control = batch['is_control']
        gene_idx = batch.get('gene_idx', torch.zeros(x.size(0), dtype=torch.long, device=x.device))
        
        B, G = x.shape
        device = x.device
        
        # Build context
        context = self.build_context(batch)
        
        # ============================================================
        # ENCODE
        # ============================================================
        
        # Encode all cells with both encoders
        z_c, mu_c, logvar_c = self.encode_control(x)
        z_p, mu_p, logvar_p = self.encode_perturbed(x)
        
        # ============================================================
        # PREDICT DELTA
        # ============================================================
        
        delta_raw, gate_probs, attn_weights = self.predict_delta(z_c, gene_idx, context)
        
        # Control cells should have no perturbation effect
        mask_perturbed = (1 - is_control).unsqueeze(1)  # [B, 1]
        delta = delta_raw * mask_perturbed  # Zero out delta for controls
        
        # ============================================================
        # DECODE
        # ============================================================
        
        # Reconstruct from own encoder
        x_rec_c = self.decode(z_c, context)
        x_rec_p = self.decode(z_p, context)
        
        # Cross-domain predictions
        x_pred_from_c = self.decode(z_c + delta, context)
        x_pred_from_p_to_c = self.decode(z_p - delta, context)
        
        # ============================================================
        # COUNT HEAD
        # ============================================================
        
        rates_c = self.counts_rate(z_c, gate_probs)
        rates_p = self.counts_rate(z_p, gate_probs)
        rates_pred = self.counts_rate(z_c + delta, gate_probs)
        
        # ZINB pi if applicable
        if self.zinb_pi_head is not None:
            pi_logits_c = self.zinb_pi_head(z_c)
            pi_logits_p = self.zinb_pi_head(z_p)
            pi_logits_pred = self.zinb_pi_head(z_c + delta)
        else:
            pi_logits_c = None
            pi_logits_p = None
            pi_logits_pred = None
        
        # ============================================================
        # OUTPUT
        # ============================================================
        
        out = {
            # Latent representations
            'z_c': z_c,
            'z_p': z_p,
            'mu_c': mu_c,
            'mu_p': mu_p,
            'logvar_c': logvar_c,
            'logvar_p': logvar_p,
            
            # Delta & Gating
            'delta': delta,
            'delta_raw': delta_raw,  # Unmasked version
            'gate_probs': gate_probs,       # NEW
            'attn_weights': attn_weights,
            'gene_attention': gate_probs,   # ALIAS for compatibility with loss logging
            
            # Reconstructions
            'x_rec_c': x_rec_c,
            'x_rec_p': x_rec_p,
            'x_pred_from_c': x_pred_from_c,
            'x_pred_from_p_to_c': x_pred_from_p_to_c,
            
            # Counts
            'rates_c': rates_c,
            'rates_p': rates_p,
            'rates_pred': rates_pred,
            'pi_logits_c': pi_logits_c,
            'pi_logits_p': pi_logits_p,
            'pi_logits_pred': pi_logits_pred,
            
            # Context
            'context': context,
            'is_control': is_control,
            'gene_idx': gene_idx,
        }
        
        return out