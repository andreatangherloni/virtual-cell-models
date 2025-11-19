"""
Training loop for AE-DEVAE.
Enhanced with CosineWithWarmup scheduler, clean logging, and vocabulary persistence.
"""
import torch
import torch.nn as nn
from torch.optim import AdamW
from pathlib import Path
import numpy as np
from tqdm import tqdm
from typing import Optional

from .config import Config, TrainConfig
from .vae import AttentionEnhancedDualEncoderVAE
from .losses import compute_losses
from .data import get_dataloader, PerturbationDataset
from .modules import CosineWithWarmup


class EMA:
    """
    Exponential Moving Average for model parameters.
    
    Maintains a shadow copy of model weights that are updated with exponential decay.
    Often leads to better generalization and more stable predictions.
    """
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        """Update shadow parameters with exponential decay."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1 - self.decay)
    
    def apply_shadow(self):
        """Apply shadow parameters to model (for saving/evaluation)."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])
    
    def restore(self):
        """Restore original parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.backup[name])
        self.backup = {}


def get_device():
    """Get best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_freezing(model: AttentionEnhancedDualEncoderVAE, cfg: TrainConfig):
    """
    Apply freezing according to config.
    
    Useful for curriculum learning:
    - Stage 1: Train everything
    - Stage 2: Freeze encoders, train only delta predictor
    - Stage 3: Fine-tune with specific components frozen
    """
    frozen_components = []
    
    # Encoders
    if cfg.freeze_enc_c:
        for param in model.enc_c.parameters():
            param.requires_grad = False
        frozen_components.append("enc_c")
        
    if cfg.freeze_enc_p:
        for param in model.enc_p.parameters():
            param.requires_grad = False
        frozen_components.append("enc_p")
    
    # Decoder
    if cfg.freeze_decoder_main:
        for param in model.dec.parameters():
            param.requires_grad = False
        frozen_components.append("decoder")
    
    # Count head
    if cfg.freeze_count_head and model.count_head is not None:
        for param in model.count_head.parameters():
            param.requires_grad = False
        frozen_components.append("count_head")
    
    # ZINB head
    if cfg.freeze_zinb_head and model.zinb_pi_head is not None:
        for param in model.zinb_pi_head.parameters():
            param.requires_grad = False
        frozen_components.append("zinb_head")
    
    # Delta predictor
    if cfg.freeze_delta_module:
        if hasattr(model.delta_predictor, 'parameters'):
            for param in model.delta_predictor.parameters():
                param.requires_grad = False
        frozen_components.append("delta_predictor")
    
    # Attention
    if cfg.freeze_attention and hasattr(model.delta_predictor, 'gene_attention'):
        for param in model.delta_predictor.gene_attention.parameters():
            param.requires_grad = False
        frozen_components.append("attention")
    
    # Embeddings
    if cfg.freeze_gene_emb:
        model.gene_emb.weight.requires_grad = False
        frozen_components.append("gene_emb")
        
    if cfg.freeze_batch_emb and model.batch_emb is not None:
        model.batch_emb.weight.requires_grad = False
        frozen_components.append("batch_emb")
        
    if cfg.freeze_ct_emb and model.celltype_emb is not None:
        model.celltype_emb.weight.requires_grad = False
        frozen_components.append("celltype_emb")
        
    if cfg.freeze_h1_emb:
        model.h1_emb.weight.requires_grad = False
        frozen_components.append("h1_emb")
    
    # Library size projection
    if cfg.freeze_lib_proj and hasattr(model, 'libsize_proj'):
        for param in model.libsize_proj.parameters():
            param.requires_grad = False
        frozen_components.append("libsize_proj")
    
    if frozen_components:
        print(f"Frozen: {', '.join(frozen_components)}")


def get_optimizer(model: nn.Module, cfg: TrainConfig):
    """
    Create optimizer with parameter groups.
    
    Embeddings get lower learning rate for stability.
    """
    emb_params = []
    other_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        # Lower LR for embeddings
        if 'emb' in name.lower() or 'embedding' in name.lower():
            emb_params.append(param)
        else:
            other_params.append(param)
    
    param_groups = [
        {
            'params': other_params,
            'lr': cfg.lr,
            'base_lr': cfg.lr
        },
        {
            'params': emb_params,
            'lr': cfg.lr * cfg.pretrained_emb_lr_scale,
            'base_lr': cfg.lr * cfg.pretrained_emb_lr_scale
        }
    ]
    
    n_main = sum(p.numel() for p in other_params)
    n_emb = sum(p.numel() for p in emb_params)
    print(f"Optimizer: {n_main:,} main params (lr={cfg.lr:.2e}), {n_emb:,} emb params (lr={cfg.lr * cfg.pretrained_emb_lr_scale:.2e})")
    
    return AdamW(param_groups, lr=cfg.lr, weight_decay=cfg.weight_decay)


def train_epoch(
    model: AttentionEnhancedDualEncoderVAE,
    dataloader,
    optimizer,
    scheduler,
    cfg: TrainConfig,
    epoch: int,
    ema: Optional[EMA] = None,
    device: torch.device = torch.device('cuda')
):
    """Train for one epoch."""
    model.train()
    
    total_loss = 0.0
    loss_acc = {}
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch:03d}")
    
    for batch_idx, batch in enumerate(pbar):
        # Move to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                 for k, v in batch.items()}
        
        # Forward
        out = model(batch)
        
        # Compute losses
        loss, loss_dict = compute_losses(batch, out, cfg, model)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        
        optimizer.step()
        
        # Step scheduler per batch
        if scheduler is not None:
            scheduler.step()
        
        # EMA update
        if ema is not None:
            ema.update()
        
        # Accumulate losses
        total_loss += loss.item()
        for k, v in loss_dict.items():
            loss_acc[k] = loss_acc.get(k, 0.0) + v
        num_batches += 1
        
        # Update progress bar
        if batch_idx % cfg.log_every == 0:
            pbar.set_postfix({k: f"{v:.4f}" for k, v in loss_dict.items()})
    
    # Average losses
    avg_loss_dict = {k: v / num_batches for k, v in loss_acc.items()}
    
    return total_loss / num_batches, avg_loss_dict


def train(cfg: Config):
    """
    Main training function.
    
    Args:
        cfg: Full configuration
    """
    # ========================================================================
    # SETUP
    # ========================================================================
    
    torch.manual_seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)
    device = get_device()
    
    outdir = Path(cfg.train.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    print(f"{'─'*60}")
    print(f"TRAINING PROCEDURE")
    print(f"{'─'*60}")
    print(f"- Output: {outdir}")
    print(f"- Device: {device}")
    print(f"- Seed: {cfg.train.seed}\n")
    
    # ========================================================================
    # DATA
    # ========================================================================
    
    # Load mappings if fine-tuning
    gene_to_idx = None
    if cfg.train.pretrained_outdir is not None:
        mapping_path = Path(cfg.train.pretrained_outdir) / 'vocab_mappings.json'
        if mapping_path.exists():
            mappings = PerturbationDataset.load_mappings(mapping_path)
            gene_to_idx = mappings.get('gene_to_idx')
    
    train_loader = get_dataloader(
        cfg.data.train_h5ad,
        cfg.data,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        gene_to_idx=gene_to_idx
    )
    
    # Get dataset info
    dataset = train_loader.dataset
    sample_batch = next(iter(train_loader))
    cfg.model.input_dim = sample_batch['x'].shape[1]
    num_batches = len(dataset.batch_to_id)
    num_celltypes = len(dataset.celltype_to_id)
    
    # Clean data summary
    print(f"\n{'─'*60}")
    print(f"DATA")
    print(f"{'─'*60}")
    
    print(f"Train data:")
    print(f"- Cells: {len(dataset):,}")
    print(f"- Genes: {cfg.model.input_dim:,}")
    print(f"- Batches: {num_batches}")
    print(f"- Cell types: {num_celltypes}")
    
    # Save mappings (only if new)
    if gene_to_idx is None:
        dataset.save_mappings(cfg.train.outdir)
    
    # ========================================================================
    # MODEL
    # ========================================================================
    
    print(f"\n{'─'*60}")
    print(f"MODEL")
    print(f"{'─'*60}")
    
    model = AttentionEnhancedDualEncoderVAE(cfg.model).to(device)
    
    # Set gene names and initialize embeddings
    gene_names = list(dataset.adata.var_names)
    model._set_gene_names(gene_names)
    model._init_context_embeddings(num_batches, num_celltypes)
    
    # Load pretrained gene embeddings
    if cfg.model.pretrained_gene_emb_path:
        model._init_gene_embeddings(cfg.model.pretrained_gene_emb_path)
    
    # ========================================================================
    # CHECKPOINT LOADING (WITH PROPER RESUME vs FINE-TUNE)
    # ========================================================================
    
    print(f"\n{'─'*60}")
    print(f"CHECKPOINT")
    print(f"{'─'*60}")
    
    start_epoch = 0
    resume_optimizer = False
    
    # PRIORITY 1: Direct --resume flag (for resuming training)
    if cfg.train.resume_from is not None and cfg.train.resume_from:
        resume_path = Path(cfg.train.resume_from)
        if resume_path.exists():
            print(f"Loading checkpoint: {resume_path}")
            ckpt = torch.load(resume_path, map_location=device)
            
            # Load model weights
            missing, unexpected = model.load_state_dict(ckpt['model_state_dict'], strict=False)
            if missing or unexpected:
                print(f"⚠ Partial load (expected for new embeddings)")
            
            # Start from next epoch, restore optimizer
            if 'epoch' in ckpt:
                start_epoch = ckpt['epoch'] + 1
                print(f"✓ Resuming from epoch {start_epoch}")
            
            resume_optimizer = True  # Flag to restore optimizer state later
            optimizer_state = ckpt.get('optimizer_state_dict')
            
        else:
            print(f"✗ Checkpoint not found: {resume_path}")
    
    # PRIORITY 2: Fine-tuning from pretrained_outdir (if no --resume)
    elif cfg.train.pretrained_outdir is not None:
        ckpt_path = Path(cfg.train.pretrained_outdir) / "best_model.pt"
        if ckpt_path.exists():
            print(f"Loading pretrained: {ckpt_path.name}")
            ckpt = torch.load(ckpt_path, map_location=device)
            
            # Load model weights
            missing, unexpected = model.load_state_dict(ckpt['model_state_dict'], strict=False)
            if missing or unexpected:
                print(f"⚠ Partial load (expected for new embeddings)")
            
            # Fne-tuning: Start from epoch 0, optionally restore optimizer
            start_epoch = 0  # Always start from 0 for fine-tuning
            print(f"Starting from epoch 0 (fine-tuning)")
            
            if not cfg.train.reset_optimizer:
                resume_optimizer = True
                optimizer_state = ckpt.get('optimizer_state_dict')
        else:
            print(f"✗ Pretrained checkpoint not found: {ckpt_path}")
    
    # Apply freezing
    set_freezing(model, cfg.train)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pct = 100 * trainable_params / total_params
    print(f"\nParameters: {total_params:,} total, {trainable_params:,} trainable ({pct:.1f}%)")
    
    # ========================================================================
    # OPTIMIZER & SCHEDULER
    # ========================================================================
    
    print(f"\n{'─'*60}")
    print(f"OPTIMIZATION")
    print(f"{'─'*60}")
    
    optimizer = get_optimizer(model, cfg.train)
    
    # Restore optimizer state if resuming
    if resume_optimizer and optimizer_state is not None:
        try:
            optimizer.load_state_dict(optimizer_state)
            print(f"✓ Restored optimizer state")
        except Exception as e:
            print(f"⚠ Failed to restore optimizer: {e}")
    
    if cfg.train.use_cosine:
        total_steps = len(train_loader) * cfg.train.epochs_main
        warmup_steps = int(cfg.train.warmup_frac * total_steps)
        
        scheduler = CosineWithWarmup(
            optimizer,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            base_lr=cfg.train.lr,
            min_lr=cfg.train.lr * cfg.train.min_lr_ratio,
            start_step=start_epoch * len(train_loader)
        )
        print(f"Scheduler: CosineWithWarmup ({warmup_steps:,} warmup steps)")
    else:
        scheduler = None
    
    ema = EMA(model, decay=cfg.train.ema_decay) if cfg.train.use_ema else None
    
    # ========================================================================
    # TRAINING LOOP
    # ========================================================================
    
    print(f"\n{'─'*60}")
    print(f"TRAINING")
    print(f"{'─'*60}")
    if 'counts' in sample_batch:
        print(f"✓ Count losses: Using real raw counts")
    else:
        print(f"⚠ Count losses: Approximating from log1p (suboptimal)")
    
    print(f"Epochs: {start_epoch} → {cfg.train.epochs_main}\n")
    
    best_loss = float('inf')
    patience_counter = 0
    
    # Store original lambda_kl for warmup
    original_lambda_kl = cfg.train.lambda_kl
    
    for epoch in range(start_epoch, cfg.train.epochs_main):
        # KL warmup
        if epoch < cfg.train.kl_warmup_epochs:
            kl_weight = (epoch + 1) / cfg.train.kl_warmup_epochs
            cfg.train.lambda_kl = original_lambda_kl * kl_weight
        
        # Train epoch
        avg_loss, loss_dict = train_epoch(
            model, train_loader, optimizer, scheduler, cfg.train, epoch, ema, device
        )
        
        # Get current learning rate
        if scheduler is not None:
            current_lrs = scheduler.get_last_lr()
            lr_str = f"{current_lrs[0]:.2e}"
            if len(current_lrs) > 1:
                lr_str += f" (emb: {current_lrs[1]:.2e})"
        else:
            lr_str = f"{cfg.train.lr:.2e}"
        
        # Logging
        print(f"Epoch {epoch:03d} | LR: {lr_str} | Loss: {avg_loss:.6f}")
        loss_str = " | ".join([f"{k}:{v:.4f}" for k, v in loss_dict.items()])
        print(f"{loss_str}")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            
            save_dict = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'loss_dict': loss_dict,
                'config': cfg,
            }
            
            # Save EMA weights
            if ema is not None:
                ema.apply_shadow()
                save_dict['ema_state_dict'] = {
                    name: param.data.clone() 
                    for name, param in model.named_parameters()
                }
                ema.restore()
            
            torch.save(save_dict, outdir / "best_model.pt")
            print(f"✓ Best model saved\n")
        else:
            patience_counter += 1
            if patience_counter <= cfg.train.early_stop_patience:
                print(f"✗ No improvement ({patience_counter}/{cfg.train.early_stop_patience})\n")
        
        # Periodic checkpoint
        if (epoch + 1) % cfg.train.save_every == 0:
            checkpoint_dict = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }
            torch.save(checkpoint_dict, outdir / f"checkpoint_epoch{epoch:03d}.pt")
        
        # Early stopping
        if patience_counter >= cfg.train.early_stop_patience:
            print(f"\n✗ Early stopping (no improvement for {patience_counter} epochs)")
            break
    
    # ========================================================================
    # COMPLETE
    # ========================================================================
    
    print(f"\n")
    print(f"TRAINING COMPLETED")
    print(f"- Best loss: {best_loss:.6f}")
    print(f"- Saved to: {outdir / 'best_model.pt'}")
    print(f"{'='*80}")