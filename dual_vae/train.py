import os
import time
import torch
import torch.nn as nn
from dataclasses import asdict
from typing import Optional
from torch.utils.data import DataLoader
from .config import TrainConfig
from .losses import compute_losses
from .vae import DualEncoderVAE
from .utils import param_groups_no_decay, logger
from .model.blocks import CosineWithWarmup
from copy import deepcopy

class EMA:
    """Exponential Moving Average of model parameters."""
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module):
        d = self.decay
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(d).add_(v.detach(), alpha=1.0 - d)
            else:
                self.shadow[k] = v.clone()

    @torch.no_grad()
    def load_into(self, model: nn.Module):
        model.load_state_dict(self.shadow, strict=False)


# ============================================================================ #
#                              TRAINING LOOP
# ============================================================================ #
def train_loop(model: DualEncoderVAE,
               train_loader: DataLoader,
               val_loader: Optional[DataLoader],
               train_cfg: TrainConfig,
               device: torch.device,
               outdir: str):
    """
    Modern, AMP-ready training loop with:
      - KL and loss-component warm-ups
      - gradient clipping
      - cosine LR with warmup (optional)
      - EMA tracking (optional)
      - checkpointing (best + periodic)
      - pretrained gene_emb freezing/unfreezing
      - automatic disabling of count head if no raw-count targets exist
      - selective component freezing via config flags
    """
    os.makedirs(outdir, exist_ok=True)
    use_cuda = (device.type == "cuda")
    scaler = torch.amp.GradScaler(enabled=train_cfg.amp and use_cuda)

    # elective parameter freezing based on config flags
    freeze_flags = [
        train_cfg.freeze_enc_c, train_cfg.freeze_enc_p, 
        train_cfg.freeze_decoder_main, train_cfg.freeze_count_head,
        train_cfg.freeze_zinb_head, train_cfg.freeze_delta_module,
        train_cfg.freeze_adapter, train_cfg.freeze_gene_emb,
        train_cfg.freeze_batch_emb, train_cfg.freeze_ct_emb,
        train_cfg.freeze_h1_emb, train_cfg.freeze_lib_proj,
        train_cfg.freeze_adv,
    ]
    
    if any(freeze_flags):
        logger("[freeze]", "Applying granular parameter freezing...")
        frozen_params = []
        trainable_params = []
        
        for name, param in model.named_parameters():
            should_freeze = False
            
            # Control encoder
            if train_cfg.freeze_enc_c and 'enc_c.' in name:
                should_freeze = True
            
            # Perturbed encoder
            if train_cfg.freeze_enc_p and 'enc_p.' in name:
                should_freeze = True
            
            # Decoder main (trunk + head, excluding count/ZINB)
            if train_cfg.freeze_decoder_main and 'dec.' in name:
                # Exclude count head and ZINB head
                if not any(x in name for x in ['count_w', 'count_b', 'zinb_pi_head']):
                    should_freeze = True
            
            # Count head (w and b parameters)
            if train_cfg.freeze_count_head and ('dec.count_w' in name or 'dec.count_b' in name):
                should_freeze = True
            
            # ZINB zero-inflation head
            if train_cfg.freeze_zinb_head and 'dec.zinb_pi_head' in name:
                should_freeze = True
            
            # Delta module (hypernetwork)
            if train_cfg.freeze_delta_module and 'delta_module.' in name:
                should_freeze = True
            
            # Low-rank adapter
            if train_cfg.freeze_adapter and 'adapter.' in name:
                should_freeze = True
            
            # Gene embeddings
            if train_cfg.freeze_gene_emb and 'gene_emb.' in name:
                should_freeze = True
            
            # Batch embeddings
            if train_cfg.freeze_batch_emb and 'batch_emb.' in name:
                should_freeze = True
            
            # Cell-type embeddings
            if train_cfg.freeze_ct_emb and 'ct_emb.' in name:
                should_freeze = True
            
            # H1 flag embedding
            if train_cfg.freeze_h1_emb and 'h1_emb.' in name:
                should_freeze = True
            
            # Library size projection
            if train_cfg.freeze_lib_proj and 'lib_proj.' in name:
                should_freeze = True
            
            # Adversarial head (if exists)
            if train_cfg.freeze_adv and 'adv.' in name:
                should_freeze = True
            
            # Apply freeze decision
            param.requires_grad = not should_freeze
            
            if should_freeze:
                frozen_params.append(name)
            else:
                trainable_params.append(name)
        
        logger("[freeze]", f"Frozen: {len(frozen_params)} parameters")
        logger("[freeze]", f"Trainable: {len(trainable_params)} parameters")
        
        # Detailed logging for small parameter sets
        if len(trainable_params) <= 20:
            logger("[freeze]", f"Trainable params: {trainable_params}")
        else:
            logger("[freeze]", f"Trainable examples: {trainable_params[:10]}...")
        
        if len(frozen_params) <= 20:
            logger("[freeze]", f"Frozen params: {frozen_params}")
        else:
            logger("[freeze]", f"Frozen examples: {frozen_params[:10]}...")

    # ------------------ Setup Embedding Freezing / Scaling ------------------ #
    freeze_epochs = train_cfg.pretrained_freeze_epochs
    emb_lr_scale  = train_cfg.pretrained_emb_lr_scale
    has_gene_emb  = hasattr(model, "gene_emb") and isinstance(model.gene_emb, nn.Embedding)

    # AdamW (no weight decay for LN/bias)
    opt = torch.optim.AdamW(param_groups_no_decay(model, train_cfg.weight_decay), lr=train_cfg.lr)
    for pg in opt.param_groups:
        pg["base_lr"] = train_cfg.lr

    # Track parameters belonging to gene_emb for LR scaling
    gene_emb_param_ids = set()
    if has_gene_emb:
        for p in model.gene_emb.parameters():
            gene_emb_param_ids.add(id(p))

    def _group_has_gene_emb(pg):
        return any(id(p) in gene_emb_param_ids for p in pg["params"])

    def _set_gene_emb_lr_scale(scale: float):
        """Rescale LR for gene_emb param group(s)."""
        if not has_gene_emb or scale == 1.0:
            return
        for pg in opt.param_groups:
            if _group_has_gene_emb(pg):
                new_lr = train_cfg.lr * scale
                pg["lr"] = new_lr
                pg["base_lr"] = new_lr
        logger("[emb]", f"applied LR scale x{scale:.3f} to gene_emb group(s)")

    # Apply scaled LR immediately if not freezing
    if has_gene_emb and freeze_epochs <= 0 and emb_lr_scale != 1.0:
        _set_gene_emb_lr_scale(emb_lr_scale)

    # Optionally freeze gene embeddings (separate from freeze_gene_emb flag)
    if has_gene_emb and freeze_epochs > 0:
        for p in model.gene_emb.parameters():
            p.requires_grad = False
        logger("[emb]", f"freezing gene_emb for {freeze_epochs} epoch(s)")

    # ------------------ Scheduler + EMA ------------------ #
    steps_per_epoch = max(1, len(train_loader))
    total_epochs = train_cfg.epochs_warmup + train_cfg.epochs_main
    total_steps = total_epochs * steps_per_epoch
    warmup_steps = int(train_cfg.warmup_frac * total_steps) if train_cfg.use_cosine else 0

    sched = CosineWithWarmup(opt, total_steps=total_steps, warmup_steps=warmup_steps, base_lr=train_cfg.lr) \
            if train_cfg.use_cosine else None

    ema = EMA(model, decay=train_cfg.ema_decay) if train_cfg.use_ema else None
    global_step = 0

    # ------------------ Best Checkpoint Tracking ------------------ #
    best_metric = float("inf")
    best_ckpt_path = os.path.join(outdir, "best_" + train_cfg.ckpt_name)
    gene_emb_unfroze = (freeze_epochs <= 0)
    epoch_best = 0

    # ------------------ Count Availability Check ------------------ #
    try:
        first_batch = next(iter(train_loader))
        counts_available = ("y_counts" in first_batch)
    except StopIteration:
        raise RuntimeError("Training loader is empty — no samples.")
    except Exception as e:
        logger("[warn]", f"could not check first batch for count targets: {e}")
        counts_available = False

    count_losses_enabled = model.cfg.use_count_head
    if not counts_available and count_losses_enabled:
        # Disable count head and related lambdas
        model.cfg.use_count_head = False
        if hasattr(model, "dec") and hasattr(model.dec, "use_count_head"):
            model.dec.use_count_head = False
        train_cfg.lambda_count_rec = 0.0
        train_cfg.lambda_count_xrec = 0.0
        count_losses_enabled = False
        logger("[count-head]", "no raw count targets found → disabling count losses/head")
    elif counts_available and count_losses_enabled:
        logger("[count-head]", "raw count targets detected → count losses/head enabled")
    else:
        logger("[count-head]", "count head disabled by configuration or missing targets")

    epochs_without_improvement = 0
    
    # ------------------ Training Loop ------------------ #
    for epoch in range(total_epochs):
        model.train()
        t0 = time.time()

        # ---- Dynamic Loss Weighting ----
        if epoch < train_cfg.kl_warmup_epochs:
            kl_weight = train_cfg.lambda_kl * (epoch + 1) / train_cfg.kl_warmup_epochs
        else:
            kl_weight = train_cfg.lambda_kl

        cfg_epoch = TrainConfig(**asdict(train_cfg))
        cfg_epoch.lambda_kl = float(kl_weight)
        
        warm = float(min(1.0, (epoch + 1) / max(1, train_cfg.epochs_warmup)))
        
        cfg_epoch.lambda_delta = train_cfg.lambda_delta * warm
        cfg_epoch.lambda_xrec = train_cfg.lambda_xrec * warm
        cfg_epoch.lambda_count_rec = train_cfg.lambda_count_rec * warm
        cfg_epoch.lambda_count_xrec = train_cfg.lambda_count_xrec * warm

        running_epoch = {"loss": 0.0}
        seen_batches = 0
        train_loss_sum = 0.0
        train_sample_count = 0

        for batch in train_loader:
            for k in batch:
                batch[k] = batch[k].to(device, non_blocking=use_cuda)

            opt.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, enabled=train_cfg.amp and use_cuda):
                out = model(batch)
                loss, d = compute_losses(batch, out, cfg_epoch, model)

            if not torch.isfinite(loss):
                logger("[warn]", "non-finite loss — skipping batch")
                continue

            scaler.scale(loss).backward()
            
            if train_cfg.grad_clip is not None:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=train_cfg.grad_clip)

            scaler.step(opt)
            
            if ema is not None:
                ema.update(model)
            scaler.update()
            
            if sched is not None:
                sched.step()

            global_step += 1
            seen_batches += 1
            bs = int(batch["x"].size(0))
            train_loss_sum += loss.detach().item() * bs
            train_sample_count += bs

            for k, v in d.items():
                running_epoch[k] = running_epoch.get(k, 0.0) + float(v)

        # ---- Epoch Summary ----
        dt = time.time() - t0
        denom = max(1, seen_batches)
        msg = " | ".join([f"{k}:{running_epoch[k] / denom:3.4f}" for k in sorted(running_epoch.keys())])
        print(f"* Epoch [train] {epoch:03d} | {msg}")
        print(f"* Epoch {epoch} done in {dt:.1f}s")

        # ---- Validation ----
        if val_loader is not None:
            was_training = model.training
            if ema is not None:
                bak_state = deepcopy(model.state_dict())
                ema.load_into(model)

            model.eval()
            val_loss_sum = 0.0
            val_sample_count = 0
            with torch.no_grad():
                for batch in val_loader:
                    for k in batch:
                        batch[k] = batch[k].to(device, non_blocking=use_cuda)
                    with torch.autocast(device_type=device.type, enabled=train_cfg.amp and use_cuda):
                        out = model(batch)
                        loss, _ = compute_losses(batch, out, cfg_epoch, model)
                    if torch.isfinite(loss):
                        bs = int(batch["x"].size(0))
                        val_loss_sum += loss.detach().item() * bs
                        val_sample_count += bs

            monitor_value = val_loss_sum / max(1, val_sample_count)
            print(f"* Epoch [val] {epoch:03d} | val_loss={monitor_value:.6f}")

            if ema is not None:
                model.load_state_dict(bak_state, strict=False)
            if was_training:
                model.train()
        else:
            monitor_value = train_loss_sum / max(1, train_sample_count)
            print(f"* Epoch [train] {epoch:03d} | train_loss(avg)={monitor_value:.6f}")

        # ---- Unfreeze Embeddings ----
        if has_gene_emb and (not gene_emb_unfroze) and ((epoch + 1) >= freeze_epochs):
            for p in model.gene_emb.parameters():
                p.requires_grad = True
            _set_gene_emb_lr_scale(emb_lr_scale)
            gene_emb_unfroze = True
            logger("[emb]", f"unfrozen gene_emb at epoch {epoch + 1}")

        # ---- Best Checkpoint ----
        improved = monitor_value < best_metric
        if improved:
            best_metric = monitor_value
            epochs_without_improvement = 0
            state_to_save = (ema.shadow if (ema is not None) else model.state_dict())
            best_ckpt = {
                "epoch": epoch,
                "model_state": state_to_save,
                "cfg_model": asdict(model.cfg),
                "cfg_train": asdict(train_cfg),
                "count_head_enabled": count_losses_enabled,
            }
            torch.save(best_ckpt, best_ckpt_path)
            logger("[best]", f"new best checkpoint saved → {best_ckpt_path} | metric={best_metric:.6f}")
            epoch_best = epoch
        else:
            epochs_without_improvement += 1
            logger("[best]", f"no improvement (best={best_metric:.6f}) | last best at epoch {epoch_best}")
            
            # ✅ Early stopping check
            if hasattr(train_cfg, 'early_stop_patience') and train_cfg.early_stop_patience > 0:
                if epochs_without_improvement >= train_cfg.early_stop_patience:
                    logger("[early-stop]", f"No improvement for {epochs_without_improvement} epochs. Stopping...")
                    break

        # ---- Periodic Checkpoint ----
        if ((epoch + 1) % train_cfg.save_every == 0) or ((epoch + 1) == total_epochs):
            ckpt_path = os.path.join(outdir, train_cfg.ckpt_name)
            state_to_save = (ema.shadow if (ema is not None) else model.state_dict())
            ckpt = {
                "epoch": epoch,
                "model_state": state_to_save,
                "cfg_model": asdict(model.cfg),
                "cfg_train": asdict(train_cfg),
                "count_head_enabled": count_losses_enabled,
            }
            torch.save(ckpt, ckpt_path)
            logger("[checkpoint]", f"saved periodic checkpoint → {ckpt_path} (EMA={ema is not None})")