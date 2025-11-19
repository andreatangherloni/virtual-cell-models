"""
Inference for AE-DEVAE.
Enhanced with neighbor mixing for unseen genes and better prediction workflow.
"""
import torch
import numpy as np
import scanpy as sc
from pathlib import Path
from tqdm import tqdm

from .config import Config
from .vae import AttentionEnhancedDualEncoderVAE
from .data import get_dataloader, PerturbationDataset

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

@torch.no_grad()
def predict(cfg: Config):
    """
    Generate predictions on test data.
    
    Enhanced with:
    - Neighbor mixing for unseen genes
    - Vocabulary mapping consistency
    - Attention map saving
    - Multiple sampling options
    """
    device = get_device()
    
    print(f"{'─'*60}")
    print(f"PREDICTING PROCEDURE")
    print(f"{'─'*60}")
    print(f"- Device: {device}")
    
    print(f"Loading checkpoint from {cfg.predict.ckpt_path}")
    
    # ========================================================================
    # LOAD MODEL
    # ========================================================================
    
    ckpt = torch.load(cfg.predict.ckpt_path, map_location=device)
    
    if 'config' in ckpt:
        model_cfg = ckpt['config'].model
        print(f"Using model config from checkpoint")
    else:
        model_cfg = cfg.model
        print(f"Using model config from current config")
    
    model = AttentionEnhancedDualEncoderVAE(model_cfg).to(device)
    
    # Load weights (EMA or standard)
    if cfg.predict.use_ema and 'ema_state_dict' in ckpt:
        print("✓ Using EMA weights")
        model.load_state_dict(ckpt['ema_state_dict'])
    else:
        print("Using standard weights")
        model.load_state_dict(ckpt['model_state_dict'])
    
    model.eval()
    
    # ========================================================================
    # LOAD VOCABULARY MAPPINGS
    # ========================================================================
    
    # Load gene_to_idx from training for consistency
    ckpt_dir = Path(cfg.predict.ckpt_path).parent
    mapping_path = ckpt_dir / 'vocab_mappings.json'
    
    gene_to_idx = None
    if mapping_path.exists():
        print(f"✓ Loading vocabulary mappings from {mapping_path}")
        mappings = PerturbationDataset.load_mappings(mapping_path)
        gene_to_idx = mappings.get('gene_to_idx')
    else:
        print(f"⚠ No vocabulary mappings found at {mapping_path} - Using test data gene order (may cause inconsistency!)")
    
    # ========================================================================
    # LOAD DATA
    # ========================================================================
    
    print(f"\nLoading test data from {cfg.predict.test_h5ad}")
    test_loader = get_dataloader(
        cfg.predict.test_h5ad,
        cfg.data,
        batch_size=cfg.predict.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        gene_to_idx=gene_to_idx  # Use same mapping as training!
    )
    
    # Get dataset info
    dataset = test_loader.dataset
    num_batches_test = len(dataset.batch_to_id)
    num_celltypes_test = len(dataset.celltype_to_id)
    
    print(f"Test data:")
    print(f"- Cells: {len(dataset):,}")
    print(f"- Genes: {dataset.adata.n_vars}")
    print(f"- Batches: {num_batches_test}")
    print(f"- Cell types: {num_celltypes_test}")
    
    # Initialize context embeddings if needed
    if model.batch_emb is None or model.celltype_emb is None:
        print(f"Initializing context embeddings...")
        model._init_context_embeddings(num_batches_test, num_celltypes_test)
    
    # ========================================================================
    # GENERATE PREDICTIONS
    # ========================================================================
    
    print(f"\nGenerating predictions...")
    
    # Check if neighbor mixing should be used
    use_neighbor_mixing = cfg.predict.use_neighbor_mixing
    if use_neighbor_mixing:
        print(f"✓ Using neighbor mixing for unseen genes:")
        print(f"- k={cfg.predict.neighbor_mix_k}")
        print(f"- tau={cfg.predict.neighbor_mix_tau}")
        print(f"- include_self={cfg.predict.neighbor_mix_include_self}")
    
    if cfg.predict.delta_gain != 1.0:
        print(f"✓ Applying delta gain: {cfg.predict.delta_gain}")
    
    all_predictions = []
    all_attentions = []
    
    # Multiple sampling support (for uncertainty estimation)
    n_samples = cfg.predict.n_samples
    if n_samples > 1:
        print(f"✓ Generating {n_samples} samples per cell for uncertainty estimation")
    
    for batch_idx, batch in enumerate(tqdm(test_loader, desc="Predicting")):
        # Move to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                 for k, v in batch.items()}
        
        # ============================================================
        # STANDARD PREDICTION (with optional neighbor mixing)
        # ============================================================
        
        if use_neighbor_mixing:
            # Use neighbor mixing for better generalization to unseen genes
            
            # Encode controls
            z_c, _, _ = model.encode_control(batch['x'])
            context = model.build_context(batch)
            gene_idx = batch['gene_idx']
            
            # Predict with neighbor mixing
            k = cfg.predict.neighbor_mix_k if cfg.predict.neighbor_mix_k is not None else cfg.model.neighbor_mix_k
            tau = cfg.predict.neighbor_mix_tau if cfg.predict.neighbor_mix_tau is not None else cfg.model.neighbor_mix_tau
            include_self = cfg.predict.neighbor_mix_include_self if cfg.predict.neighbor_mix_include_self is not None else cfg.model.neighbor_mix_include_self
            
            x_pred, attn = model.predict_with_neighbor_mixing(
                z_c=z_c,
                context=context,
                target_gene_idx=gene_idx,
                k=k,
                tau=tau,
                include_self=include_self,
                delta_gain=cfg.predict.delta_gain
            )
            
            # Store attention if computed
            if cfg.predict.compute_attention_maps and attn is not None:
                all_attentions.append(attn.cpu().numpy())
        
        else:
            # Standard forward pass
            out = model(batch)
            
            # Get predictions (control → perturbed)
            if cfg.predict.delta_gain != 1.0:
                # Apply delta gain
                delta_scaled = out['delta'] * cfg.predict.delta_gain
                x_pred = model.decode(out['z_c'] + delta_scaled, out['context'])
            else:
                x_pred = out['x_pred_from_c']
            
            # Save attention if requested
            if cfg.predict.compute_attention_maps and out['gene_attention'] is not None:
                attn = out['gene_attention'].cpu().numpy()
                all_attentions.append(attn)
        
        # Move to CPU and store
        x_pred = x_pred.cpu().numpy()
        all_predictions.append(x_pred)
    
    # ========================================================================
    # AGGREGATE RESULTS
    # ========================================================================
    
    # Concatenate all batches
    predictions = np.vstack(all_predictions)
    
    if all_attentions:
        attentions = np.vstack(all_attentions)
        print(f"✓ Collected attention maps: {attentions.shape}")
    else:
        attentions = None
    
    print(f"✓ Generated predictions: {predictions.shape}")
    
    # ========================================================================
    # SAVE RESULTS
    # ========================================================================
    
    print(f"\nSaving to {cfg.predict.output_h5ad}")
    
    # Load original test data to get metadata
    adata_test = sc.read_h5ad(cfg.predict.test_h5ad)
    
    # Create new AnnData with predictions
    adata_pred = sc.AnnData(
        X=predictions,
        obs=adata_test.obs.copy(),
        var=adata_test.var.copy()
    )
    
    # Add metadata
    adata_pred.uns['prediction_config'] = {
        'checkpoint': str(cfg.predict.ckpt_path),
        'use_ema': cfg.predict.use_ema,
        'delta_gain': cfg.predict.delta_gain,
        'use_neighbor_mixing': use_neighbor_mixing,
        'neighbor_mix_k': cfg.predict.neighbor_mix_k if use_neighbor_mixing else None,
        'neighbor_mix_tau': cfg.predict.neighbor_mix_tau if use_neighbor_mixing else None,
    }
    
    # Add attention maps to obsm if computed
    if attentions is not None:
        adata_pred.obsm['gene_attention'] = attentions
        print("✓ Saved attention maps to .obsm['gene_attention']")
    
    # Save
    output_path = Path(cfg.predict.output_h5ad)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    adata_pred.write_h5ad(output_path)
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    
    print(f"\n")
    print(f"PREDICTION COMPLETED")
    print(f"- Input: {cfg.predict.test_h5ad}")
    print(f"- Output: {cfg.predict.output_h5ad}")
    print(f"- Predictions: {predictions.shape}")
    if attentions is not None:
        print(f"- Attention maps: {attentions.shape}")
    print(f"- Method: {'Neighbor mixing' if use_neighbor_mixing else 'Standard'}")
    print(f"{'='*80}")


@torch.no_grad()
def predict_with_uncertainty(
    cfg: Config,
    n_samples: int = 10
) -> tuple:
    """
    Generate predictions with uncertainty estimation.
    
    Uses multiple samples from the VAE latent space to estimate
    prediction uncertainty.
    
    Args:
        cfg: Configuration
        n_samples: Number of samples per cell
    
    Returns:
        (mean_predictions, std_predictions, all_samples)
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Generating {n_samples} samples per cell...")
    
    # Load model (same as predict())
    ckpt = torch.load(cfg.predict.ckpt_path, map_location=device)
    model_cfg = ckpt.get('config', cfg).model
    model = AttentionEnhancedDualEncoderVAE(model_cfg).to(device)
    
    if cfg.predict.use_ema and 'ema_state_dict' in ckpt:
        model.load_state_dict(ckpt['ema_state_dict'])
    else:
        model.load_state_dict(ckpt['model_state_dict'])
    
    model.eval()
    
    # Load data (same as predict())
    ckpt_dir = Path(cfg.predict.ckpt_path).parent
    mapping_path = ckpt_dir / 'vocab_mappings.json'
    gene_to_idx = None
    if mapping_path.exists():
        mappings = PerturbationDataset.load_mappings(mapping_path)
        gene_to_idx = mappings.get('gene_to_idx')
    
    test_loader = get_dataloader(
        cfg.predict.test_h5ad,
        cfg.data,
        batch_size=cfg.predict.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        gene_to_idx=gene_to_idx
    )
    
    # Initialize embeddings
    dataset = test_loader.dataset
    model._init_context_embeddings(len(dataset.batch_to_id), len(dataset.celltype_to_id))
    
    # Collect samples
    all_samples = []  # [n_samples, n_cells, n_genes]
    
    for sample_idx in range(n_samples):
        print(f"[predict_uncertainty] Sample {sample_idx + 1}/{n_samples}")
        
        sample_predictions = []
        
        for batch in tqdm(test_loader, desc=f"Sample {sample_idx+1}", leave=False):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                     for k, v in batch.items()}
            
            # Sample from latent space (not deterministic!)
            z_c, _, _ = model.encode_control(batch['x'])  # Samples from q(z|x)
            context = model.build_context(batch)
            
            # Predict delta
            delta, _, _ = model.predict_delta(z_c, batch['gene_idx'], context)
            
            # Apply delta gain
            delta = delta * cfg.predict.delta_gain
            
            # Decode
            x_pred = model.decode(z_c + delta, context)
            
            sample_predictions.append(x_pred.cpu().numpy())
        
        all_samples.append(np.vstack(sample_predictions))
    
    # Stack samples: [n_samples, n_cells, n_genes]
    all_samples = np.stack(all_samples, axis=0)
    
    # Compute statistics
    mean_predictions = all_samples.mean(axis=0)  # [n_cells, n_genes]
    std_predictions = all_samples.std(axis=0)    # [n_cells, n_genes]
    
    print(f"Mean predictions: {mean_predictions.shape}")
    print(f"Std predictions: {std_predictions.shape}")
    print(f"Average uncertainty: {std_predictions.mean():.4f}")
    
    return mean_predictions, std_predictions, all_samples