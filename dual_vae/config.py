from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import yaml

def load_yaml_cfg(path: str) -> Dict[str, Any]:
    """Load a YAML file into a nested dict."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


# -------------------------
# Model hyperparameters
# -------------------------
@dataclass
class ModelConfig:
    # Core dimensions
    input_dim: int                       # Number of genes (e.g., 18080); set at runtime from AnnData
    latent_dim: int = 128                # Latent space size
    hidden_dims: Optional[List[int]] = None  # MLP widths high→low (e.g., [1536, 768, 384]); if None, code will set a default
    context_dim: int = 256               # Context embedding dim (batch/ct/H1 are embedded and summed to this size)
    gene_embed_dim: int = 128            # Target-gene embedding dim
    
    # --- Pretrained embeddings ---    
    pretrained_gene_emb_path: Optional[str] = None  # .pt dict {gene_name->Tensor[D_pre]}
    pretrained_norm: str = "l2"                    # {"l2","none"}
    pretrained_case_insensitive: bool = True

    # PCA initialization toggle and limits
    use_pca_init: bool = True                      # if False → random init
    init_max_cells: int = 200_000 

    # Encoder variance clamp in log-space for stability
    logvar_min: float = -4.0
    logvar_max: float = 2.0
    
    # Decoder output range in log1p-space
    output_min: float = 0.0
    output_max: float = 8.0
    bound_mode: Optional[str] = None  # None: no bound; "sigmoid"/"tanh" for bounded decoder output

    # Gene vocabulary (row 0 reserved for control/non-targeting). Will be overwritten at runtime.
    num_genes_vocab: int = 20000

    # Regularization & normalization
    dropout: float = 0.2
    layernorm: bool = True

    # Decoder conditioning
    use_film_in_decoder: bool = True     # FiLM(gamma,beta) modulation by context u

    # (Optional) adversarial head to encourage z_c invariance to target gene
    use_adv_invariance: bool = False
    adv_hidden: int = 128
    
    # libsize covariate
    use_libsize_covariate: bool = False  # per-cell log1p library size as covariate
    libsize_proj_hidden: int = 128      # Tiny hidden width to project scalar libsize

    # Δ hypernetwork/adapter (used by the extended VAE)
    delta_type: str = "hyper"
    delta_rank: int = 8                  # Low-rank adapter size r for A,B
    hyperdelta_use_ln: bool = True       # LayerNorm inside HyperDelta MLP for stability

    # Count head on top of decoder’s bounded log1p mean (used if you add a count loss)
    use_count_head: bool = False         # Default off; enable when using lambda_count > 0
    count_link: str = "poisson"          # "poisson" or "nb"
    nb_theta: float = 10.0               # NB inverse-dispersion (scalar) if using NB
    zinb_pi_hidden: int = 128

    # Inference-time neighbor mixing in gene-embedding space (helps unseen targets)
    neighbor_mix_k: int = 8
    neighbor_mix_tau: float = 0.07       # Softmax temperature for neighbor weighting


# -------------------------
# Training hyperparameters
# -------------------------
@dataclass
class TrainConfig:
    pretrained_outdir: Optional[str] = None
    # Schedule
    seed: int = 42
    epochs_warmup: int = 10              # Warm-up epochs (also used to ramp Δ/xrec if enabled)
    kl_warmup_epochs: int = 10           # Linear ramp of KL weight for these epochs
    epochs_main: int = 100               # Additional epochs after warm-up

    # Optimization
    batch_size: int = 256
    lr: float = 2e-3
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    huber_delta: float = 0.5
    amp: bool = True                     # Automatic mixed precision
    
    #: Fine-grained freezing control
    freeze_enc_c: bool = False           # Freeze control encoder
    freeze_enc_p: bool = False           # Freeze perturbed encoder
    freeze_decoder_main: bool = False    # Freeze decoder trunk (not count head)
    freeze_count_head: bool = False      # Freeze count_w, count_b
    freeze_zinb_head: bool = False       # Freeze ZINB zero-inflation head
    freeze_delta_module: bool = False    # Freeze delta hypernetwork
    freeze_adapter: bool = False         # Freeze low-rank adapter
    freeze_gene_emb: bool = False        # Freeze gene embeddings
    freeze_batch_emb: bool = False       # Freeze batch embeddings
    freeze_ct_emb: bool = False          # Freeze cell-type embeddings
    freeze_h1_emb: bool = False          # Freeze H1 flag embedding
    freeze_lib_proj: bool = False        # Freeze library size projection
    freeze_adv: bool = False             # Freeze adversarial head (if present)
    
    
    # -------- Loss weights (defaults: only recon & KL on) --------
    lambda_rec: float      = 1.0            # Reconstruction in log1p space
    lambda_xrec: float     = 0.0            # Cross-reconstruction (NTC→Perturbed, Perturbed→NTC)
    lambda_kl: float       = 0.05           # KL(q||N) (often warmed up)
    lambda_delta: float    = 0.0            # Latent Δ consistency (z_p ≈ z_c + Δ)
    lambda_orth: float     = 0.0            # Encourage orthogonality between z_c and Δ
    lambda_nce: float      = 0.0            # InfoNCE contrastive term (anchor=z_c+Δ, pos=z_p)
    lambda_adv: float      = 0.0            # Adversarial invariance head on z_c
    lambda_smooth: float   = 0.0            # Smooth Δ across nearest neighbors in gene-embedding space
    nce_temperature: float = 0.0            # Temperature for InfoNCE (if enabled)

    # Optional likelihood / distribution-matching objectives (default off)
    lambda_count_rec: float  = 0.0            # Poisson/NB loss on (approx) counts
    lambda_count_xrec: float = 0.0            # Cross-reconstruction (NTC→Perturbed, Perturbed→NTC)
    lambda_mmd: float        = 0.0            # MMD between predicted vs real distributions
    # lambda_rank: float       = 0.0            # Rank/logFC alignment loss

    lambda_soft_topk: float = 0.0         # Smooth top-K overlap (Jaccard surrogate)
    lambda_soft_sign: float = 0.0         # Smooth sign agreement (PDS surrogate)
    lambda_rank_corr: float = 0.0         # Smooth rank correlation (Spearman-like)
    lambda_lfc_magnitude: float = 0.0     # Weighted MSE on LFC values
    lambda_lfc_scale: float = 0.0         # Variance/scale matching
    lambda_listnet: float = 0.0           # Smooth ListNet ranking
    
    lambda_focal_gene_selection: float = 0.0 
    lambda_ranking_hinge: float = 0.0 
    lambda_hard_gene_selection: float = 0.0 
    
    count_max_rate: float    = 50000.0         # Max rate (counts) used to clip Poisson/NB mean; avoids inf gradients
    
    # ZINB regularization
    lambda_zinb_pi_reg: float = 1.0e-4     # try 1e-4 … 1e-3
    zinb_pi_reg_type: str = "l2_logit"     # or "kl_beta"
    zinb_beta_a: float = 1.0               # if using "kl_beta"
    zinb_beta_b: float = 9.0
        
    # Δ-smooth k-NN (only used if lambda_smooth > 0)
    smooth_k: int = 8

    # Gene-weighted MSE (compute per-gene weights on controls; default off for simplicity)
    compute_gene_weights: bool = False
    gw_max_cells: int = 200000
    gw_clip_min: float = 0.1
    gw_clip_max: float = 10.0
    gw_stratify_by_batch: bool = True

    # EMA of model params (stability / better validation)
    use_ema: bool = True
    ema_decay: float = 0.999
    
    early_stop_patience: int = 40    # Stop if no improvement for 40 epochs

    # LR schedule
    use_cosine: bool = True
    warmup_frac: float = 0.1

    # Logging / checkpoints
    log_every: int = 100
    save_every: int = 1
    outdir: str = "outputs"
    ckpt_name: str = "dual_vae.pt"

    # Resume / reset
    reset_optimizer: bool = True         # If resuming, start a fresh optimizer by default
    resume_from: Optional[str] = None    # Path to a checkpoint to load model_state from
    
    # Freeze policy for pretrained emb/projector during early epochs
    pretrained_freeze_epochs: int = 0          # 0 = don’t freeze

    # LR multipliers for pretrained blocks (always applied, frozen or not)
    pretrained_emb_lr_scale: float = 0.1       # multiply base LR for emb table

# -------------------------
# Data settings
# -------------------------
@dataclass
class DataConfig:
    train_h5ad: str
    val_h5ad: Optional[str] = None               # Optional validation AnnData
    fewshot_h1_h5ad: Optional[str] = None        # Optional H1 few-shot set
    control_pool_h5ad_for_init: Optional[str] = None  # If None, use train_h5ad for init

    # .obs column names
    col_target: str = "target_gene"
    col_batch: str = "batch"
    col_celltype: str = "cell_type"
    col_is_h1: str = "is_H1"
    control_token: str = "non-targeting"

    # Preprocessing flags
    # apply_log1p: bool = True
    normalize: bool = True          # build a log1p(normalized) view for x
    target_sum: float | None = None # if None -> default to median inside loader
    val_only_h1: bool = False       # If True, filter validation to H1
    zscore: bool = False
    zscore_eps: float = 1e-6
    drop_all_zero: bool = True      # Drop cells with all-zero expression
    
    include_libsize_covariate: bool = True
    libsize_zscore: bool = True
    libsize_eps: float = 1e-6

# -------------------------
# Prediction settings
# -------------------------
@dataclass
class PredictConfig:
    # ---------------------------------------------------------------------
    # Input / output files
    # ---------------------------------------------------------------------
    perturb_list_csv: str                        # CSV: target_gene, n_cells, [median_umi_per_cell]
    control_pool_h5ad: str                       # AnnData with control cells (e.g., H1 controls)
    out_h5ad: str = "predictions.h5ad"
    n_control_cells: int = 5000                  # -1 or None -> include all controls
    artifacts_outdir: Optional[str] = None
    gene_order_file: Optional[str] = None

    # ---------------------------------------------------------------------
    # Data normalization
    # ---------------------------------------------------------------------
    normalize: bool = True                       # build log1p(normalized) view for x
    target_sum: Optional[float] = None           # if None -> median-depth normalization

    # Context flag for H1 in obs[col_is_h1]
    h1_flag_value: int = 1
    n_infer_samples: int = 1                     # >1 → sample z_c multiple times per control

    # ---------------------------------------------------------------------
    # Output options
    # ---------------------------------------------------------------------
    output_scale: str = "counts"                 # "counts" or "log1p"
    compression: str = "gzip"                    # h5ad compression ("gzip" or "lzf")
    depth_column: str = "median_umi_per_cell"    # CSV column for target depth

    # ---------------------------------------------------------------------
    # Count-sampling parameters
    # ---------------------------------------------------------------------
    count_max_rate: Optional[float] = None       # cap for stability; None disables
    count_link: str = "nb"                       # "poisson" or "nb"/"zinb"
    nb_theta: float = 10.0                       # NB dispersion (smaller → heavier tails)

    # Optional rate shaping before sampling
    rate_sharpen_beta: float = 1.05              # power transform on rates (1.0 = off)
    mix_sharpen_p: float = 0.25                  # 0..1 blend between rate and rate**beta
    
    topk_keep_only: Optional[int] = None      # e.g., 9000 to match ~50% zeros on 18k genes
    prune_quantile: Optional[float] = None    # e.g., 0.50 to keep top half by value
    
    topk_boost_k: int = 0                        # boost top-K genes per cell (0 = off)
    topk_boost_gamma: float = 1.0                # >1.0 to amplify top-K
    
    delta_gain: float = 1.0  # NEW: inference-time multiplier on perturbation delta

    # ---------------------------------------------------------------------
    # Neighbor-mixing for unseen genes
    # ---------------------------------------------------------------------
    neighbor_mix_k: int = 8
    neighbor_mix_tau: float = 0.07

    # ---------------------------------------------------------------------
    # Optional per-gene calibration (post-hoc)
    # ---------------------------------------------------------------------
    apply_affine_calibration: bool = False
    calib_alpha_npy: Optional[str] = None        # path to npy array [G] with multipliers