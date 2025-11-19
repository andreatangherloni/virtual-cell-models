"""
Loss functions for AE-DEVAE.
Enhanced with real count usage, better error handling, and cleaner structure.
"""
import torch
import torch.nn.functional as F
from typing import Dict
from .config import TrainConfig
from .vae import AttentionEnhancedDualEncoderVAE


# ============================================================================
# Helper Functions
# ============================================================================

def _to_float(x):
    """Convert tensor to float for logging."""
    if isinstance(x, torch.Tensor):
        return x.detach().item()
    return float(x)

# ============================================================================
# VAE BASE LOSSES
# ============================================================================

def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    KL(q(z|x) || p(z)) where p(z) = N(0, I).
    
    Args:
        mu: [B, D] - latent mean
        logvar: [B, D] - latent log-variance
    
    Returns:
        [B] - KL per sample
    """
    return 0.5 * torch.sum(torch.exp(logvar) + mu.pow(2) - 1.0 - logvar, dim=-1)


def mse_loss_weighted(
    x_pred: torch.Tensor,
    x_true: torch.Tensor,
    gene_w: torch.Tensor = None,
    huber_delta: float = 0.5
) -> torch.Tensor:
    """
    Weighted MSE (or Huber) loss over genes.
    
    Args:
        x_pred: [B, G] - predicted expression
        x_true: [B, G] - true expression
        gene_w: [G] - optional per-gene weights
        huber_delta: Huber loss delta (None for standard MSE)
    
    Returns:
        [B] - loss per sample
    """
    if huber_delta is not None and huber_delta > 0:
        base = F.smooth_l1_loss(x_pred, x_true, reduction="none", beta=huber_delta)
    else:
        base = (x_pred - x_true) ** 2
    
    if gene_w is not None:
        if gene_w.dim() == 1:
            gene_w = gene_w.view(1, -1)
        base = base * gene_w
        denom = gene_w.sum(dim=1, keepdim=True).clamp_min(1e-8)
        return base.sum(dim=-1) / denom.squeeze(1)
    
    return base.mean(dim=-1)


def delta_consistency_loss(z_p: torch.Tensor, z_c: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """
    Consistency: z_p should equal z_c + delta.
    
    Args:
        z_p: [B, D] - perturbed latent
        z_c: [B, D] - control latent
        delta: [B, D] - predicted perturbation effect
    
    Returns:
        [B] - MSE per sample
    """
    return ((z_p - (z_c + delta)) ** 2).mean(dim=-1)


# ============================================================================
# COUNT DISTRIBUTION LOSSES
# ============================================================================

def poisson_nll(rate: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    """
    Poisson negative log-likelihood.
    
    Args:
        rate: [B, G] - Poisson rate parameters (λ)
        counts: [B, G] - observed counts
    
    Returns:
        [B] - NLL per sample
    """
    eps = 1e-8
    rate_safe = rate.clamp_min(eps)
    return (rate_safe - counts * torch.log(rate_safe)).mean(dim=-1)


def nb_nll(mean: torch.Tensor, theta: float, counts: torch.Tensor) -> torch.Tensor:
    """
    Negative Binomial negative log-likelihood.
    
    Args:
        mean: [B, G] - NB mean parameters
        theta: Dispersion parameter (inverse overdispersion)
        counts: [B, G] - observed counts
    
    Returns:
        [B] - NLL per sample
    """
    th = torch.as_tensor(theta, dtype=mean.dtype, device=mean.device)
    eps = 1e-8
    
    mean_safe = mean.clamp_min(eps)
    th_safe = th.clamp_min(eps)
    
    t1 = torch.lgamma(counts + th_safe) - torch.lgamma(th_safe) - torch.lgamma(counts + 1.0)
    t2 = th_safe * (torch.log(th_safe) - torch.log(th_safe + mean_safe))
    t3 = counts * (torch.log(mean_safe) - torch.log(th_safe + mean_safe))
    
    return -(t1 + t2 + t3).mean(dim=-1)


def zinb_nll(mean: torch.Tensor, theta: float, pi: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    """
    Zero-Inflated Negative Binomial NLL.
    
    Args:
        mean: [B, G] - NB mean parameters
        theta: Dispersion parameter
        pi: [B, G] - zero-inflation probability
        counts: [B, G] - observed counts
    
    Returns:
        [B] - NLL per sample
    """
    eps = 1e-6
    th = torch.as_tensor(theta, dtype=mean.dtype, device=mean.device)
    
    mean_safe = mean.clamp_min(eps)
    th_safe = th.clamp_min(eps)
    pi_safe = pi.clamp(eps, 1.0 - eps)
    
    # NB log pmf
    log_nb = (
        torch.lgamma(counts + th_safe)
        - torch.lgamma(th_safe)
        - torch.lgamma(counts + 1.0)
        + th_safe * (torch.log(th_safe) - torch.log(th_safe + mean_safe))
        + counts * (torch.log(mean_safe) - torch.log(th_safe + mean_safe))
    )
    
    is_zero = (counts < eps).to(mean.dtype)
    is_nonzero = 1.0 - is_zero
    
    # Zero probability: mixture of inflated zeros and NB zeros
    log_zero = torch.logaddexp(torch.log(pi_safe), torch.log(1.0 - pi_safe) + log_nb)
    
    # Likelihood
    loglik = is_zero * log_zero + is_nonzero * (torch.log(1.0 - pi_safe) + log_nb)
    
    return -loglik.mean(dim=-1)


def zinb_nll_with_reg(
    mean: torch.Tensor,
    theta: float,
    pi_logits: torch.Tensor,
    counts: torch.Tensor,
    pi_reg_weight: float = 0.0,
    pi_reg_type: str = "l2_logit",
    beta_prior_ab: tuple = (1.0, 9.0)
) -> torch.Tensor:
    """
    ZINB NLL with optional regularization on zero-inflation.
    
    Args:
        mean: [B, G] - NB mean parameters
        theta: Dispersion parameter
        pi_logits: [B, G] - zero-inflation logits
        counts: [B, G] - observed counts
        pi_reg_weight: Regularization weight
        pi_reg_type: "l2_logit" or "kl_beta"
        beta_prior_ab: Beta prior parameters for KL regularization
    
    Returns:
        [B] - NLL per sample (with regularization)
    """
    pi = torch.sigmoid(pi_logits).clamp(1e-6, 1 - 1e-6)
    base = zinb_nll(mean, theta, pi, counts)
    
    if pi_reg_weight <= 0:
        return base
    
    if pi_reg_type == "l2_logit":
        # Penalize large logits (encourages pi near 0.5)
        reg = (pi_logits ** 2).mean()
    elif pi_reg_type == "kl_beta":
        # KL divergence to Beta(a, b) prior
        a, b = beta_prior_ab
        eps = 1e-8
        reg = -((a - 1.0) * torch.log(pi + eps) + (b - 1.0) * torch.log(1.0 - pi + eps)).mean()
    else:
        raise ValueError(f"Unknown pi_reg_type: {pi_reg_type}")
    
    return base + pi_reg_weight * reg


# ============================================================================
# ATTENTION SUPERVISION LOSSES
# ============================================================================

def attention_focus_loss(
    gene_attention: torch.Tensor,
    lfc_true: torch.Tensor
) -> torch.Tensor:
    """
    Encourages attention to correlate with true differential expression.
    
    Uses rank correlation (differentiable Spearman-like).
    
    Args:
        gene_attention: [G] - attention scores
        lfc_true: [G] - true log-fold-changes
    
    Returns:
        scalar loss (1 - correlation)
    """
    # Compute DE score from true LFC
    true_de_score = torch.abs(lfc_true)
    
    # Rank correlation (Spearman-like)
    attn_rank = gene_attention.argsort().argsort().float() / gene_attention.numel()
    de_rank = true_de_score.argsort().argsort().float() / true_de_score.numel()
    
    # Pearson correlation on ranks
    attn_centered = attn_rank - attn_rank.mean()
    de_centered = de_rank - de_rank.mean()
    
    corr = (attn_centered * de_centered).sum() / (
        torch.sqrt((attn_centered ** 2).sum() * (de_centered ** 2).sum()) + 1e-8
    )
    
    return 1.0 - corr


def topk_supervision_loss(
    lfc_pred: torch.Tensor,
    lfc_true: torch.Tensor,
    k: int = 200,
    gamma: float = 2.0
) -> torch.Tensor:
    """
    Focal loss for top-K gene selection.
    
    Teaches model which genes should be in top-K DE genes.
    
    Args:
        lfc_pred: [G] - predicted log-fold-changes
        lfc_true: [G] - true log-fold-changes
        k: number of top genes
        gamma: focal loss focusing parameter
    
    Returns:
        scalar focal loss
    """
    k = min(k, lfc_pred.numel())
    
    # Get true top-K genes
    _, true_topk_idx = torch.topk(torch.abs(lfc_true), k=k)
    
    # Create binary target: 1 for top-K, 0 for others
    target = torch.zeros_like(lfc_pred)
    target[true_topk_idx] = 1.0
    
    # Predicted "probability" of being in top-K
    pred_abs = torch.abs(lfc_pred)
    pred_prob = torch.sigmoid(pred_abs - pred_abs.median())
    
    # Focal loss
    eps = 1e-8
    pos_loss = -(1 - pred_prob) ** gamma * torch.log(pred_prob + eps)
    neg_loss = -pred_prob ** gamma * torch.log(1 - pred_prob + eps)
    
    loss = target * pos_loss + (1 - target) * neg_loss
    
    return loss.mean()


def lfc_magnitude_loss(
    lfc_pred: torch.Tensor,
    lfc_true: torch.Tensor,
    focus_top_k: int = 1000
) -> torch.Tensor:
    """
    MSE on LFC magnitudes, focused on top-K DE genes.
    
    Args:
        lfc_pred: [G] - predicted log-fold-changes
        lfc_true: [G] - true log-fold-changes
        focus_top_k: focus on top-K most DE genes
    
    Returns:
        scalar MSE
    """
    k = min(focus_top_k, lfc_pred.numel())
    
    # Get top-K by true LFC magnitude
    _, true_topk_idx = torch.topk(torch.abs(lfc_true), k=k)
    
    # MSE on these genes
    return F.mse_loss(lfc_pred[true_topk_idx], lfc_true[true_topk_idx])


# ============================================================================
# MAIN LOSS COMPUTATION
# ============================================================================

def compute_losses(
    batch: Dict[str, torch.Tensor],
    out: Dict[str, torch.Tensor],
    cfg: TrainConfig,
    model: AttentionEnhancedDualEncoderVAE
) -> tuple:
    """
    Compute all losses for training.
    
    Args:
        batch: Input batch (must contain 'x', 'is_control', optionally 'counts')
        out: Model outputs
        cfg: Training configuration
        model: Model instance
    
    Returns:
        (total_loss, loss_dict)
    """
    x = batch['x']
    is_control = out['is_control']
    device = x.device
    
    mask_c = (is_control > 0.5)
    mask_p = (is_control < 0.5)
    
    # Per-gene weights (optional)
    gene_w = None
    if hasattr(model, "gene_w") and model.gene_w is not None:
        gene_w = model.gene_w.to(device=device, dtype=x.dtype)
    
    # ========================================================================
    # 1. RECONSTRUCTION
    # ========================================================================
    
    L_rec_c = mse_loss_weighted(out['x_rec_c'], x, gene_w, cfg.huber_delta)
    L_rec_p = mse_loss_weighted(out['x_rec_p'], x, gene_w, cfg.huber_delta)
    
    L_rec = torch.tensor(0.0, device=device)
    if mask_c.any():
        L_rec += L_rec_c[mask_c].mean()
    if mask_p.any():
        L_rec += L_rec_p[mask_p].mean()
    L_rec = cfg.lambda_rec * L_rec
    
    # ========================================================================
    # 2. CROSS-DOMAIN RECONSTRUCTION
    # ========================================================================
    
    L_xrec_cp = mse_loss_weighted(out['x_pred_from_c'], x, gene_w, cfg.huber_delta)
    L_xrec_pc = mse_loss_weighted(out['x_pred_from_p_to_c'], x, gene_w, cfg.huber_delta)
    
    L_xrec = torch.tensor(0.0, device=device)
    if mask_p.any():
        L_xrec += L_xrec_cp[mask_p].mean()
    if mask_c.any():
        L_xrec += L_xrec_pc[mask_c].mean()
    L_xrec = cfg.lambda_xrec * L_xrec
    
    # ========================================================================
    # 3. KL DIVERGENCE
    # ========================================================================
    
    L_kl_c = kl_divergence(out['mu_c'], out['logvar_c'])
    L_kl_p = kl_divergence(out['mu_p'], out['logvar_p'])
    
    L_kl = torch.tensor(0.0, device=device)
    if mask_c.any():
        L_kl += L_kl_c[mask_c].mean()
    if mask_p.any():
        L_kl += L_kl_p[mask_p].mean()
    L_kl = cfg.lambda_kl * L_kl
    
    # ========================================================================
    # 4. DELTA CONSISTENCY
    # ========================================================================
    
    L_delta = torch.tensor(0.0, device=device)
    if mask_p.any() and cfg.lambda_delta > 0:
        L_delta_all = delta_consistency_loss(out['z_p'], out['z_c'], out['delta'])
        L_delta = cfg.lambda_delta * L_delta_all[mask_p].mean()
    
    # ========================================================================
    # 5. COUNT LOSSES
    # ========================================================================
    
    L_count_rec = torch.tensor(0.0, device=device)
    L_count_xrec = torch.tensor(0.0, device=device)
    
    use_counts = (cfg.lambda_count_rec > 0 or cfg.lambda_count_xrec > 0) and \
                 out['rates_c'] is not None
    
    if use_counts:
        # Use real counts if available, otherwise approximate from log1p
        if 'counts' in batch:
            y_true = batch['counts']  # [B, G] - real raw counts!
        else:
            y_true = torch.expm1(x).clamp_min(0.0)  # [B, G] - approximation
        
        count_link = model.cfg.count_link.lower()
        
        def _compute_count_nll(rates, pi_logits=None):
            """Helper to compute count NLL based on model type."""
            if count_link == "zinb" and pi_logits is not None:
                return zinb_nll_with_reg(
                    rates, model.nb_theta, pi_logits, y_true,
                    pi_reg_weight=cfg.lambda_zinb_pi_reg,
                    pi_reg_type=cfg.zinb_pi_reg_type,
                    beta_prior_ab=(cfg.zinb_beta_a, cfg.zinb_beta_b)
                )
            elif count_link == "nb":
                return nb_nll(rates, model.nb_theta, y_true)
            else:
                return poisson_nll(rates, y_true)
        
        # Reconstruction count loss
        if cfg.lambda_count_rec > 0:
            nll_c = _compute_count_nll(out['rates_c'], out['pi_logits_c'])
            nll_p = _compute_count_nll(out['rates_p'], out['pi_logits_p'])
            
            if mask_c.any():
                L_count_rec += nll_c[mask_c].mean()
            if mask_p.any():
                L_count_rec += nll_p[mask_p].mean()
            
            L_count_rec = cfg.lambda_count_rec * L_count_rec
        
        # Cross-reconstruction count loss
        if cfg.lambda_count_xrec > 0:
            nll_pred = _compute_count_nll(out['rates_pred'], out['pi_logits_pred'])
            
            if mask_p.any():
                L_count_xrec = cfg.lambda_count_xrec * nll_pred[mask_p].mean()
    
    # ========================================================================
    # 6. ATTENTION & TOP-K SUPERVISION (PER-PERTURBATION)
    # ========================================================================
    
    L_attention_focus = torch.tensor(0.0, device=device)
    L_topk_supervision = torch.tensor(0.0, device=device)
    L_lfc_magnitude = torch.tensor(0.0, device=device)
    
    if mask_p.any() and (cfg.lambda_attention_focus > 0 or 
                         cfg.lambda_topk_supervision > 0 or 
                         cfg.lambda_lfc_magnitude > 0):
        
        batch_gene_indices = batch.get('gene_idx')
        
        if batch_gene_indices is not None:
            unique_genes = torch.unique(batch_gene_indices[mask_p])
            
            per_target_attn_focus = []
            per_target_topk = []
            per_target_lfc_mag = []
            
            for gene_idx in unique_genes:
                # Group by gene index
                target_mask_p = (batch_gene_indices == gene_idx) & mask_p
                target_mask_c = mask_c
                
                # Need minimum samples
                if target_mask_p.sum() < 3 or target_mask_c.sum() < 3:
                    continue
                
                # Compute pseudobulk LFCs
                ctrl_bulk = x[target_mask_c].mean(0)  # [G]
                pert_real_bulk = x[target_mask_p].mean(0)  # [G]
                lfc_real = pert_real_bulk - ctrl_bulk
                
                x_pred_p = out['x_pred_from_c'][target_mask_p]
                pert_pred_bulk = x_pred_p.mean(0)  # [G]
                lfc_pred = pert_pred_bulk - ctrl_bulk
                
                # Attention focus loss
                if cfg.lambda_attention_focus > 0 and out['gene_attention'] is not None:
                    attn = out['gene_attention'][target_mask_p].mean(0)  # [G]
                    loss = attention_focus_loss(attn, lfc_real)
                    per_target_attn_focus.append(loss)
                
                # Top-K supervision
                if cfg.lambda_topk_supervision > 0:
                    loss = topk_supervision_loss(lfc_pred, lfc_real, k=200, gamma=2.0)
                    per_target_topk.append(loss)
                
                # LFC magnitude
                if cfg.lambda_lfc_magnitude > 0:
                    loss = lfc_magnitude_loss(lfc_pred, lfc_real, focus_top_k=1000)
                    per_target_lfc_mag.append(loss)
            
            # Aggregate across perturbations
            if per_target_attn_focus:
                L_attention_focus = cfg.lambda_attention_focus * torch.stack(per_target_attn_focus).mean()
            if per_target_topk:
                L_topk_supervision = cfg.lambda_topk_supervision * torch.stack(per_target_topk).mean()
            if per_target_lfc_mag:
                L_lfc_magnitude = cfg.lambda_lfc_magnitude * torch.stack(per_target_lfc_mag).mean()
    
    # ========================================================================
    # TOTAL LOSS
    # ========================================================================
    
    total = (L_rec + L_xrec + L_kl + L_delta + 
             L_count_rec + L_count_xrec + 
             L_attention_focus + L_topk_supervision + L_lfc_magnitude)
    
    # ========================================================================
    # LOSS DICTIONARY (for logging)
    # ========================================================================
    
    loss_dict = {
        'loss': _to_float(total),
        'rec': _to_float(L_rec),
        'xrec': _to_float(L_xrec),
        'kl': _to_float(L_kl),
        'delta': _to_float(L_delta),
        'cnt_rec': _to_float(L_count_rec),
        'cnt_xrec': _to_float(L_count_xrec),
        'attn_focus': _to_float(L_attention_focus),
        'topk_super': _to_float(L_topk_supervision),
        'lfc_mag': _to_float(L_lfc_magnitude),
    }
    
    return total, loss_dict