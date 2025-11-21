"""
Configuration classes for AE-DEVAE (Attention-Enhanced Dual Encoder VAE).
"""
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class DataConfig:
    """Data loading and preprocessing configuration."""
    train_h5ad: str = "data/adata_Training.h5ad"
    val_h5ad: Optional[str] = None
    fewshot_h1_h5ad: Optional[str] = None
    control_pool_h5ad: Optional[str] = None

    
    # Column names in AnnData.obs
    col_target: str = "target_gene"
    col_batch: str = "batch"
    col_celltype: str = "cell_type"
    col_is_h1: str = "is_H1"
    control_token: str = "non-targeting"
    
    # Preprocessing
    zscore: bool = False
    normalize: bool = True
    target_sum: Optional[float] = None
    drop_all_zero: bool = False
    
    # Data loading
    num_workers: int = 4
    pin_memory: bool = True


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    # Dimensions
    input_dim: Optional[int] = None  # Auto-detected from data
    latent_dim: int = 128
    hidden_dims: List[int] = field(default_factory=lambda: [1536, 768, 384])
    context_dim: int = 256
    gene_embed_dim: int = 128
    
    # Regularization
    dropout: float = 0.2
    layernorm: bool = True
    
    # ResidualMLP options
    use_residual_mlp: bool = True  # Use ResidualMLP with skip connections
    use_input_skip: bool = True    # Concatenate input at each down stage
    
    # Attention
    num_attention_heads: int = 4
    attention_dropout: float = 0.1
    use_attention: bool = True
    attention_mixing_alpha: float = 0.5  # Initial value for learnable parameter
    
    # Delta prediction
    delta_type: str = "attention"  # "attention" or "hyper"
    delta_rank: int = 8
    hyperdelta_use_ln: bool = True
    
    # Decoder
    use_film_in_decoder: bool = True
    
    # Count head
    use_count_head: bool = True
    count_link: str = "nb"  # "poisson", "nb", or "zinb"
    nb_theta: float = 10.0
    zinb_pi_hidden: int = 128
    attention_modulated_counts: bool = True  # Use attention to modulate count rates
    
    # Output bounds
    bound_mode: str = "tanh"
    output_min: float = 0.0
    output_max: float = 8.0
    
    # Gene embeddings
    pretrained_gene_emb_path: Optional[str] = "data/ESM2_pert_features.pt"
    pretrained_norm: str = "l2"
    pretrained_case_insensitive: bool = True
    use_pca_init: bool = False
    
    # Library size
    use_libsize_covariate: bool = True
    libsize_proj_hidden: int = 128
    
    # VAE variance options
    logvar_min: float = -4.0
    logvar_max: float = 2.0
    use_softplus_var: bool = True  # Use softplus for smooth positive variance
    
    #  Neighbor mixing for unseen genes
    neighbor_mix_k: int = 12              # Number of nearest neighbors
    neighbor_mix_tau: float = 0.07        # Softmax temperature
    neighbor_mix_include_self: bool = True  # Include query gene in mixture


@dataclass
class TrainConfig:
    """Training configuration."""
    # I/O
    pretrained_outdir: Optional[str] = None
    outdir: str = "outputs/default"
    ckpt_name: str = "model.pt"
    seed: int = 42
    
    # Training schedule
    epochs_warmup: int = 0
    kl_warmup_epochs: int = 10
    epochs_main: int = 50
    
    # Optimization
    batch_size: int = 128
    lr: float = 2.0e-4
    weight_decay: float = 5.0e-2
    grad_clip: float = 1.0
    huber_delta: float = 0.5
    amp: bool = False
    reset_optimizer: bool = False
    
    # Freezing (for curriculum learning)
    freeze_enc_c: bool = False
    freeze_enc_p: bool = False
    freeze_decoder_main: bool = False
    freeze_count_head: bool = False
    freeze_zinb_head: bool = False
    freeze_delta_module: bool = False
    freeze_attention: bool = False
    freeze_adapter: bool = False
    freeze_gene_emb: bool = False
    freeze_batch_emb: bool = False
    freeze_ct_emb: bool = False
    freeze_h1_emb: bool = False
    freeze_lib_proj: bool = False
    
    # Loss weights
    lambda_rec: float = 0.70
    lambda_kl: float = 0.02
    lambda_xrec: float = 0.10
    lambda_delta: float = 0.15
    lambda_orth: float = 0.0
    lambda_nce: float = 0.0
    lambda_smooth: float = 0.0
    lambda_mmd: float = 0.0
    
    # Count losses
    lambda_count_rec: float = 0.30
    lambda_count_xrec: float = 0.10
    
    # Competition losses
    lambda_attention_focus: float = 0.0   # Stage 2+
    lambda_topk_supervision: float = 0.0  # Stage 3
    lambda_lfc_magnitude: float = 0.0
    
    # Auxiliary
    nce_temperature: float = 0.2
    smooth_k: int = 8
    count_max_rate: float = 60000.0
    lambda_zinb_pi_reg: float = 1.0e-4
    zinb_pi_reg_type: str = "l2_logit"
    zinb_beta_a: float = 1.0
    zinb_beta_b: float = 9.0
    
    compute_gene_weights: bool = False
    
    # EMA and scheduler
    use_ema: bool = True
    ema_decay: float = 0.999
    use_cosine: bool = True           # Use CosineWithWarmup scheduler
    warmup_frac: float = 0.10         # Fraction of steps for warmup
    min_lr_ratio: float = 0.01        # Minimum LR as fraction of base LR
    
    pretrained_freeze_epochs: int = 0
    pretrained_emb_lr_scale: float = 0.05  # LR multiplier for embeddings
    
    # Logging
    log_every: int = 20
    save_every: int = 5
    resume_from: Optional[str] = None
    early_stop_patience: int = 30


@dataclass
class PredictConfig:
    """Prediction/inference configuration for competition submission."""
    
 # ========================================================================
    # Input/Output Paths
    # ========================================================================
    ckpt_path: str = "outputs/default/best_model.pt"
    perturb_list_csv: str = "data/test_perturbations.csv"   # CSV: target_gene, n_cells, [median_umi_per_cell]
    out_h5ad: str = "predictions.h5ad"                      # Output path
    gene_order_file: Optional[str] = None                   # Optional gene ordering file
    
    
    # ========================================================================
    # Model Inference Settings
    # ========================================================================
    batch_size: int = 256
    use_ema: bool = True              # Use EMA weights if available
    delta_gain: float = 1.0           # Scale delta predictions
    n_samples: int = 1                # Number of samples from latent (>1 for uncertainty)
    
    # ========================================================================
    # Neighbor Mixing (for unseen genes)
    # ========================================================================
    neighbor_mix_k: int = 12                # Number of nearest neighbors
    neighbor_mix_tau: float = 0.07          # Softmax temperature
    neighbor_mix_include_self: bool = True  # Include query gene in mixture
    
    # ========================================================================
    # Control Cell Settings
    # ========================================================================
    h1_flag_value: int = 1            # Value indicating H1 cells in col_is_h1
    n_control_cells: int = 0          # Number of real controls to include (0=none, -1=all)
    
    # ========================================================================
    # Output Format
    # ========================================================================
    output_scale: str = "counts"      # "counts" or "log1p"
    compression: str = "gzip"         # H5AD compression
    
    # ========================================================================
    # Count Sampling (when output_scale='counts')
    # ========================================================================
    count_link: str = "nb"            # "nb", "poisson", or "round"
    nb_theta: float = 10.0            # NB dispersion parameter
    count_max_rate: float = 60000.0   # Maximum rate for count sampling
    
    # Depth matching
    depth_column: str = "median_umi_per_cell"  # Column in CSV for target depth
    
    # Rate shaping
    rate_sharpen_beta: float = 1.0    # Exponent for rate sharpening (1.0=no sharpening)
    mix_sharpen_p: float = 0.0        # Mixing proportion for sharpened rates
    
    # Post-processing
    topk_keep_only: Optional[int] = None      # Keep only top-K genes (None=keep all)
    prune_quantile: Optional[float] = None    # Prune genes below quantile (None=no pruning)
    topk_boost_k: int = 0                     # Number of top genes to boost (0=no boosting)
    topk_boost_gamma: float = 1.0             # Boost factor for top-K genes
    
    # ========================================================================
    # Analysis
    # ========================================================================
    compute_attention_maps: bool = False  # Save attention for analysis (not implemented yet)


@dataclass
class Config:
    """Main configuration container."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    predict: PredictConfig = field(default_factory=PredictConfig)