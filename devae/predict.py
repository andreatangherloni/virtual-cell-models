"""
Inference for AE-DEVAE
Enhanced with competition submission workflow
"""
import torch
import numpy as np
import pandas as pd
import scanpy as sc
from pathlib import Path
from tqdm import tqdm
from scipy import sparse
from typing import Optional, Dict, Tuple
import json

from .config import Config
from .vae import AttentionEnhancedDualEncoderVAE
from .data import load_control_pool


def get_device():
    """Get best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def batched(iterable, n):
    """Yield successive n-sized chunks from iterable."""
    items = list(iterable)
    for i in range(0, len(items), n):
        yield items[i:i + n]


def sample_counts_from_log1p(X_hat_log1p: np.ndarray, target_depth: float, cfg) -> np.ndarray:
    """
    Sample counts from log1p predictions using Negative Binomial.
    
    Args:
        X_hat_log1p: [n_cells, n_genes] predictions in log1p space
        target_depth: Target median depth for sampling
        cfg: PredictConfig
    
    Returns:
        [n_cells, n_genes] count matrix (int32)
    """
    # Convert back to rates
    rates = np.expm1(X_hat_log1p)  # log1p -> raw scale
    
    # Normalize to target depth
    current_depths = rates.sum(axis=1, keepdims=True)
    rates = rates / (current_depths + 1e-8) * target_depth
    
    # Apply rate shaping if configured
    rate_sharpen_beta = getattr(cfg, 'rate_sharpen_beta', 1.0)
    mix_sharpen_p = getattr(cfg, 'mix_sharpen_p', 0.0)
    
    if rate_sharpen_beta != 1.0:
        rates_sharp = np.power(rates, rate_sharpen_beta)
        rates = (1 - mix_sharpen_p) * rates + mix_sharpen_p * rates_sharp
    
    # Clip rates
    count_max_rate = getattr(cfg, 'count_max_rate', None)
    if count_max_rate is not None:
        rates = np.clip(rates, 0, count_max_rate)
    
    # Sample from NB
    count_link = getattr(cfg, 'count_link', 'nb')
    if count_link == 'nb':
        theta = getattr(cfg, 'nb_theta', 10.0)
        p = theta / (theta + rates + 1e-8)
        counts = np.random.negative_binomial(theta, p)
    elif count_link == 'poisson':
        counts = np.random.poisson(rates)
    else:
        # Fallback: round rates
        counts = np.rint(rates)
    
    # Apply top-k/pruning if configured
    topk_keep_only = getattr(cfg, 'topk_keep_only', None)
    if topk_keep_only is not None and topk_keep_only > 0:
        for i in range(counts.shape[0]):
            topk_idx = np.argsort(counts[i])[-topk_keep_only:]
            mask = np.zeros(counts.shape[1], dtype=bool)
            mask[topk_idx] = True
            counts[i, ~mask] = 0
    
    prune_quantile = getattr(cfg, 'prune_quantile', None)
    if prune_quantile is not None and prune_quantile > 0:
        for i in range(counts.shape[0]):
            thresh = np.quantile(counts[i], prune_quantile)
            counts[i, counts[i] < thresh] = 0
    
    # Apply top-k boosting if configured
    topk_boost_k = getattr(cfg, 'topk_boost_k', 0)
    topk_boost_gamma = getattr(cfg, 'topk_boost_gamma', 1.0)
    
    if topk_boost_k > 0 and topk_boost_gamma != 1.0:
        for i in range(counts.shape[0]):
            topk_idx = np.argsort(counts[i])[-topk_boost_k:]
            counts[i, topk_idx] = np.rint(counts[i, topk_idx] * topk_boost_gamma)
    
    return counts.astype(np.int32)


import pandas as pd
import scanpy as sc

def reorder_genes(adata: sc.AnnData, gene_order_file: str) -> sc.AnnData:
    """
    Reorder genes according to a gene order file using vectorized operations.
    
    Args:
        adata: AnnData object
        gene_order_file: Path to text file with gene names (one per line)
    
    Returns:
        Reordered AnnData
    """
    # 1. Fast Load: Use pandas to read file (faster than looping over lines)
    target_genes = pd.read_csv(gene_order_file, header=None, dtype=str).iloc[:, 0].values
    target_idx = pd.Index(target_genes)
    
    # 2. Fast Lookup: isin() uses hash tables (O(1) lookup per gene)
    mask_present = target_idx.isin(adata.var_names)
    
    # 3. Split using boolean masking (vectorized)
    matched = target_idx[mask_present]
    missing_count = len(target_idx) - len(matched)
    
    if missing_count > 0:
        print(f"⚠ Gene order: {missing_count}/{len(target_genes)} genes missing from predictions")
    
    # 4. Reorder
    return adata[:, matched].copy()


def mix_params_for_target(
    tg: str,
    gene2id: Dict[str, int],
    control_token: str,
    default_k: int,
    default_tau: float,
    seen_k: int = 0,
    seen_tau: Optional[float] = None
) -> Tuple[int, float]:
    """
    Determine neighbor mixing parameters for a target gene.
    
    Args:
        tg: Target gene name
        gene2id: Gene vocabulary
        control_token: Control token string
        default_k: Default k for unseen genes
        default_tau: Default tau for unseen genes
        seen_k: k for seen genes (0 = use default)
        seen_tau: tau for seen genes (None = use default)
    
    Returns:
        (k, tau) tuple
    """
    gid = gene2id.get(tg, 0)
    
    if gid == 0 or tg == control_token:
        # Control or unseen → no mixing needed
        return 0, 1.0
    
    # Check if gene was seen during training (has non-zero embedding)
    # For simplicity, use default mixing for all
    # In practice, you might want different k for seen vs unseen
    if seen_k > 0 and seen_tau is not None:
        return seen_k, seen_tau
    else:
        return default_k, default_tau


@torch.no_grad()
def predict(cfg: Config):
    """
    Generate predictions for competition submission.
    
    Workflow:
    1. Load checkpoint and model
    2. Load control cell pool with depth statistics
    3. Encode control cells
    4. Load perturbation list
    5. Generate perturbed cells
    6. Assemble and save output
    """
    device = get_device()
    
    print(f"{'─'*80}")
    print(f"PREDICTION PROCEDURE")
    print(f"{'─'*80}")
    print(f"- Output: {cfg.predict.out_h5ad}")
    print(f"- Device: {device}")
    print(f"- Scale: {cfg.predict.output_scale}\n")
    
    # ========================================================================
    # CHECKPOINT & MODEL
    # ========================================================================
    
    print(f"{'─'*80}")
    print(f"CHECKPOINT")
    print(f"{'─'*80}")
    
    ckpt_path = Path(cfg.predict.ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    if 'config' in ckpt:
        model_cfg = ckpt['config'].model
    else:
        model_cfg = cfg.model
    
    # Load vocabulary mappings
    ckpt_dir = ckpt_path.parent
    mapping_path = ckpt_dir / 'vocab_mappings.json'
    
    if not mapping_path.exists():
        raise FileNotFoundError(f"vocab_mappings.json not found at {mapping_path}")
    
    with open(mapping_path, 'r') as f:
        mappings = json.load(f)
    
    gene_to_idx = mappings['gene_to_idx']
    batch_to_id = mappings['batch_to_id']
    celltype_to_id = mappings['celltype_to_id']
        
    # Build model
    model = AttentionEnhancedDualEncoderVAE(model_cfg).to(device)
    
    # Set gene names for neighbor mixing
    train_gene_names = list(gene_to_idx.keys())
    model._set_gene_names(train_gene_names)
    
    # Initialize context embeddings
    model._init_context_embeddings(len(batch_to_id), len(celltype_to_id))
    
    # Load weights
    if cfg.predict.use_ema and 'ema_state_dict' in ckpt:
        model.load_state_dict(ckpt['ema_state_dict'])
        print(f"✓ Loaded EMA weights from {ckpt_path.name}")
    else:
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"✓ Loaded weights from {ckpt_path.name}")
    
    model.eval()
    
    # ========================================================================
    # CONTROL POOL (load once, get everything!)
    # ========================================================================
    
    print(f"\n{'─'*80}")
    print(f"CONTROL POOL")
    print(f"{'─'*80}")
    
    X_pool, obs_pool, pool_batch_vocab, pool_ct_vocab, global_median_depth, depth_by_batch, X_raw = load_control_pool(
        cfg=cfg.data,
        gene_to_idx=gene_to_idx,
        h1_flag_value=cfg.predict.h1_flag_value,
    )
    
    # Compute library sizes
    libsize_log1p = np.log1p(np.expm1(X_pool).sum(axis=1)).astype(np.float32)
    obs_pool['libsize_log1p'] = libsize_log1p
    
    pool_size = X_pool.shape[0]
    out_var_index = pd.Index(train_gene_names, dtype=str)
    
    print(f"Pool statistics:")
    print(f"- Cells: {pool_size:,}")
    print(f"- Genes: {X_pool.shape[1]:,}")
    print(f"- Global median depth: {global_median_depth:.1f} counts")
    if depth_by_batch:
        print(f"- Per-batch medians: {len(depth_by_batch)} batches")
    
    # ========================================================================
    # ENCODE CONTROLS
    # ========================================================================

    print(f"\n{'─'*80}")
    print(f"ENCODING")
    print(f"{'─'*80}")

    # Prepare context
    col_batch = cfg.data.col_batch
    col_celltype = cfg.data.col_celltype
    col_target = cfg.data.col_target
    control_token = cfg.data.control_token

    batch_ids = np.array([batch_to_id.get(str(b), 0) for b in obs_pool.get(col_batch, pd.Series(['unknown'] * len(obs_pool)))])
    ct_ids = np.array([celltype_to_id.get(str(c), 0) for c in obs_pool.get(col_celltype, pd.Series(['unknown'] * len(obs_pool)))])
    is_h1 = np.ones(len(obs_pool), dtype=np.int64)  # All H1

    # Encode z_c
    n_samples = getattr(cfg.predict, 'n_samples', 1)
    zc_bank, context_bank = [], []

    bs = cfg.predict.batch_size  # Use config batch size
    with tqdm(total=pool_size, desc="Encoding controls", unit="cells") as pbar:
        for sl in batched(range(pool_size), bs):
            x_b = torch.from_numpy(X_pool[sl]).to(device).float()
            
            # Build batch dict (match training format exactly!)
            batch_dict = {
                'x': x_b,
                'batch_idx': torch.from_numpy(batch_ids[sl]).to(device),
                'celltype_idx': torch.from_numpy(ct_ids[sl]).to(device),
                'is_h1': torch.from_numpy(is_h1[sl]).to(device),
                'libsize': torch.from_numpy(libsize_log1p[sl]).to(device).float()
            }
            
            if n_samples > 1:
                # Repeat for multiple samples
                for k in batch_dict:
                    if isinstance(batch_dict[k], torch.Tensor):
                        batch_dict[k] = batch_dict[k].repeat_interleave(n_samples, dim=0)
            
            # Encode
            z_c, _, _ = model.encode_control(batch_dict['x'])
            context = model.build_context(batch_dict)
            
            zc_bank.append(z_c)
            context_bank.append(context)
            
            pbar.update(len(sl))
    
    zc_bank = torch.cat(zc_bank, dim=0)
    context_bank = torch.cat(context_bank, dim=0)
    
    pool_size_eff = zc_bank.shape[0]
    print(f"\n✓ Encoded {pool_size_eff:,} control representations")
    print(f"- z_c shape: {tuple(zc_bank.shape)}")
    print(f"- context shape: {tuple(context_bank.shape)}")
    
    # ========================================================================
    # PERTURBATION LIST
    # ========================================================================
    
    print(f"\n{'─'*80}")
    print(f"PERTURBATION LIST")
    print(f"{'─'*80}")
    
    df_list = pd.read_csv(cfg.predict.perturb_list_csv)
    
    if 'target_gene' not in df_list.columns or 'n_cells' not in df_list.columns:
        raise ValueError("perturb_list_csv must have columns: target_gene, n_cells")
    
    print(f"Loaded perturbations:")
    print(f"- Total: {len(df_list)}")
    print(f"- Unique genes: {df_list['target_gene'].nunique()}")
    print(f"- Total cells: {df_list['n_cells'].sum():,}")
    
    # Depth column
    depth_col = getattr(cfg.predict, 'depth_column', 'median_umi_per_cell')
    has_depth = depth_col in df_list.columns
    df_has_batch = col_batch in df_list.columns
    
    if has_depth:
        print(f"- Depth info: Found '{depth_col}' column")
    else:
        print(f"⚠ Depth info: Using global median ({global_median_depth:.1f})")
    
    def resolve_target_depth(row):
        """Resolve target depth for a perturbation."""
        if has_depth:
            v = row.get(depth_col, np.nan)
            if pd.notna(v) and np.isfinite(v) and float(v) > 0:
                return float(v)
        if df_has_batch:
            b = str(row.get(col_batch, ''))
            if b in depth_by_batch:
                return depth_by_batch[b]
        return global_median_depth
    
    # Neighbor mixing defaults
    default_k = cfg.predict.neighbor_mix_k
    default_tau = cfg.predict.neighbor_mix_tau
    
    print(f"\nNeighbor mixing:")
    print(f"- k: {default_k}")
    print(f"- tau: {default_tau}")
    print(f"- include_self: {cfg.predict.neighbor_mix_include_self}")
    
    # ========================================================================
    # GENERATE CELLS
    # ========================================================================
    
    print(f"\n{'─'*80}")
    print(f"GENERATION")
    print(f"{'─'*80}")
    
    X_blocks, obs_blocks = [], []
    output_scale = cfg.predict.output_scale.lower()
    
    # Optional: include real controls (fast - just index into X_raw!)
    n_ctrl = getattr(cfg.predict, 'n_control_cells', 0)
    include_real_ctrl = (n_ctrl is not None and n_ctrl != 0)
    
    if include_real_ctrl:
        if n_ctrl == -1 or n_ctrl is None:
            n_ctrl = pool_size
        n_ctrl = min(n_ctrl, pool_size)
        
        idx_ctrl = np.random.choice(pool_size, size=n_ctrl, replace=False)
        obs_ctrl = obs_pool.iloc[idx_ctrl].copy()
        obs_ctrl[col_target] = control_token
        obs_ctrl['is_synthetic'] = 0
        
        if output_scale == 'counts':
            # Use pre-loaded raw counts (already in correct gene order!)
            X_ctrl = X_raw[idx_ctrl]
            obs_ctrl['median_umi_per_cell'] = global_median_depth
            X_blocks.append(X_ctrl)
            obs_blocks.append(obs_ctrl)
            print(f"✓ Included {n_ctrl:,} real control cells (counts)")
        else:
            # Log1p (already in correct order)
            X_ctrl = X_pool[idx_ctrl]
            X_blocks.append(X_ctrl)
            obs_blocks.append(obs_ctrl)
            print(f"✓ Included {n_ctrl:,} real control cells (log1p)")
    
    # Generate per target
    bs_gen = cfg.predict.batch_size  # Use config batch size
    
    print(f"\nGenerating perturbed cells...")
    for _, row in tqdm(df_list.iterrows(), total=len(df_list), desc="Perturbations"):
        tg = str(row['target_gene'])
        n_cells = int(row['n_cells'])
        target_depth = resolve_target_depth(row)
        
        # Get neighbor mixing params
        k_tg, tau_tg = mix_params_for_target(
            tg, gene_to_idx, control_token, default_k, default_tau
        )
        
        gid = gene_to_idx.get(tg, 0)
        idx = np.random.choice(pool_size_eff, size=n_cells, replace=(n_cells > pool_size_eff))
        
        zc_sel = zc_bank[idx]
        context_sel = context_bank[idx]
        
        # ============================================================
        # CONTROL/NON-TARGETING
        # ============================================================
        
        if gid == 0 or tg == control_token:
            obs_t = obs_pool.iloc[idx % len(obs_pool)].copy().reset_index(drop=True)
            obs_t[col_target] = tg
            obs_t['is_synthetic'] = 1
            
            # Generate synthetic controls (decode without perturbation)
            outs = []
            for sl in batched(range(n_cells), bs_gen):
                x_hat = model.decode(zc_sel[sl], context_sel[sl])
                outs.append(x_hat.cpu().numpy())
            
            X_hat_log1p = np.concatenate(outs, axis=0)
            
            if output_scale == 'counts':
                X_counts = sample_counts_from_log1p(X_hat_log1p, target_depth, cfg.predict)
                obs_t['median_umi_per_cell'] = target_depth
                X_blocks.append(X_counts)
                obs_blocks.append(obs_t)
            else:
                X_blocks.append(X_hat_log1p)
                obs_blocks.append(obs_t)
            continue
        
        # ============================================================
        # PERTURBATION
        # ============================================================
        
        gid_t = torch.full((n_cells,), gid, device=device, dtype=torch.long)
        outs = []
        
        for sl in batched(range(n_cells), bs_gen):
            # Predict with neighbor mixing
            x_hat, _ = model.predict_with_neighbor_mixing(
                z_c=zc_sel[sl],
                context=context_sel[sl],
                target_gene_idx=gid_t[sl],
                k=k_tg if k_tg > 0 else default_k,
                tau=tau_tg if k_tg > 0 else default_tau,
                include_self=cfg.predict.neighbor_mix_include_self,
                delta_gain=getattr(cfg.predict, 'delta_gain', 1.0)
            )
            outs.append(x_hat.cpu().numpy())
        
        X_hat_log1p = np.concatenate(outs, axis=0)
        
        obs_t = obs_pool.iloc[idx % len(obs_pool)].copy().reset_index(drop=True)
        obs_t[col_target] = tg
        obs_t['is_synthetic'] = 1
        
        if output_scale == 'counts':
            X_counts = sample_counts_from_log1p(X_hat_log1p, target_depth, cfg.predict)
            obs_t['median_umi_per_cell'] = target_depth
            X_blocks.append(X_counts)
            obs_blocks.append(obs_t)
        else:
            X_blocks.append(X_hat_log1p)
            obs_blocks.append(obs_t)
    
    # ========================================================================
    # ASSEMBLY
    # ========================================================================
    
    print(f"\n{'─'*80}")
    print(f"ASSEMBLY")
    print(f"{'─'*80}")
    
    any_sparse = any(sparse.issparse(b) for b in X_blocks)
    if any_sparse:
        X_out = sparse.vstack([
            b.tocsr() if sparse.issparse(b) else sparse.csr_matrix(b) 
            for b in X_blocks
        ], format='csr')
    else:
        X_out = np.vstack(X_blocks)
    
    obs_out = pd.concat(obs_blocks, axis=0, ignore_index=True)
    obs_out.index = obs_out.index.astype(str)
    
    var_out = pd.DataFrame(index=out_var_index)
    
    # Final cleanup
    if output_scale == 'counts':
        if sparse.issparse(X_out):
            X_out.data = np.rint(X_out.data).clip(min=0).astype(np.int32)
            X_out.eliminate_zeros()
        else:
            X_out = np.rint(X_out).clip(min=0).astype(np.int32)
    else:
        if 'median_umi_per_cell' in obs_out.columns:
            obs_out = obs_out.drop(columns=['median_umi_per_cell'])
    
    adata_out = sc.AnnData(X_out, obs=obs_out, var=var_out)
    adata_out.obs_names_make_unique()
    adata_out.var_names_make_unique()
    
    print(f"Output shape: {adata_out.shape}")
    print(f"- Cells: {adata_out.n_obs:,}")
    print(f"- Genes: {adata_out.n_vars:,}")
    print(f"- Targets: {obs_out[col_target].nunique()}")
    
    # Gene ordering
    if cfg.predict.gene_order_file is not None:
        adata_out = reorder_genes(adata_out, cfg.predict.gene_order_file)
        print(f"✓ Reordered genes according to {cfg.predict.gene_order_file}")
    
    # ========================================================================
    # OUTPUT
    # ========================================================================
    
    print(f"\n{'─'*80}")
    print(f"OUTPUT")
    print(f"{'─'*80}")
    
    output_path = Path(cfg.predict.out_h5ad)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if output_scale == 'counts' and not sparse.issparse(adata_out.X):
        adata_out.X = sparse.csr_matrix(adata_out.X)
    
    adata_out.write_h5ad(output_path, compression=getattr(cfg.predict, 'compression', 'gzip'))
    
    print(f"Saved: {output_path}")
    
    if output_scale == 'counts':
        depths = np.asarray(adata_out.X.sum(axis=1)).ravel() if sparse.issparse(adata_out.X) else adata_out.X.sum(axis=1)
        nnz = adata_out.X.count_nonzero() if sparse.issparse(adata_out.X) else np.count_nonzero(adata_out.X)
        sparsity = 100 * (1 - nnz / (adata_out.n_obs * adata_out.n_vars))
        
        print(f"\nCount statistics:")
        print(f"- Mean depth: {depths.mean():.1f}")
        print(f"- Median depth: {np.median(depths):.1f}")
        print(f"- Sparsity: {sparsity:.1f}%")
    
    print(f"\n{'─'*80}")
    print(f"PREDICTION COMPLETE")
    print(f"{'─'*80}\n")