import torch
import torch.nn.functional as F
from .config import TrainConfig
from .vae import DualEncoderVAE

# --------------------------- utilities ---------------------------
def _to_float(x):
    """Log scalars (convert tensors to Python floats safely)."""
    if isinstance(x, torch.Tensor):
        return x.detach().item()
    return float(x)

# --------------------------- Base terms ---------------------------
def kl_divergence(mu, logvar):
    """Elementwise KL(q(z|x)||N(0,1)) reduced over latent dim, returns [B]."""
    return 0.5 * torch.sum(torch.exp(logvar) + mu.pow(2) - 1.0 - logvar, dim=-1)

def mse_loss_weighted(x_pred, x_true, gene_w=None, huber_delta: float = 0.5):
    """
    Per-cell weighted MSE over genes. If gene_w (shape [G]) is provided,
    returns sum_j w_j * (e_ij^2) / sum_j w_j. Otherwise mean over genes.
    Output shape: [B].
    """
    err = x_pred - x_true
    if huber_delta is not None and huber_delta > 0:
        base = F.smooth_l1_loss(x_pred, x_true, reduction="none", beta=huber_delta)  # [B,G]
    else:
        base = err**2
    if gene_w is not None:
        if gene_w.dim() == 1: gene_w = gene_w.view(1, -1)
        base = base * gene_w
        denom = gene_w.sum(dim=1, keepdim=True).clamp_min(1e-8)
        return base.sum(dim=-1) / denom.squeeze(1)
    return base.mean(dim=-1)

def delta_consistency(z_p, z_c, Delta):
    """|| z_p - (z_c + Δ) ||^2 averaged over latent dims, returns [B]."""
    return ((z_p - (z_c + Delta)) ** 2).mean(dim=-1)

def cosine_orthogonality(z_c, Delta):
    """
    Squared cosine similarity between z_c and Δ (encourage orthogonality).
    Returns [B] reduced over latent dims.
    """
    eps = 1e-6
    num = torch.sum(z_c * Delta, dim=-1)
    den = (torch.norm(z_c, dim=-1) * torch.norm(Delta, dim=-1)).clamp_min(eps)
    cos = num / den
    return cos ** 2

def info_nce(anchor, positives, temperature=0.2):
    """
    InfoNCE over a batch: anchor[i] should match positives[i], others act as negatives.
    Inputs: [B, D]; returns scalar CE loss.
    """
    a = F.normalize(anchor, dim=-1)
    p = F.normalize(positives, dim=-1)
    logits = (a @ p.t()) / max(1e-4, float(temperature))  # [B, B]
    labels = torch.arange(a.size(0), device=a.device)
    return F.cross_entropy(logits, labels)

# --------------------------- Count losses ---------------------------
def poisson_nll(rate, counts):
    """
    counts ~ Poisson(rate), NLL up to const: rate - counts * log(rate)
    Assumes rate > 0 (softplus in decoder).
    Returns per-cell mean across genes: [B]
    """
    eps = max(1e-8, float(torch.finfo(rate.dtype).tiny))
    return (rate - counts * torch.log(rate.clamp_min(eps))).mean(dim=-1)

def nb_nll(mean, theta, counts):
    """
    Negative Binomial NLL with mean μ (>0) and inverse-dispersion θ (>0).
    Assume μ>0 (softplus). Returns per-cell mean across genes: [B]
    """
    th = torch.as_tensor(theta, dtype=mean.dtype, device=mean.device)
    eps = max(1e-8, float(torch.finfo(mean.dtype).tiny))
    t1 = torch.lgamma(counts + th) - torch.lgamma(th) - torch.lgamma(counts + 1.0)
    t2 = th * (torch.log(th.clamp_min(eps)) - torch.log((th + mean).clamp_min(eps)))
    t3 = counts * (torch.log(mean.clamp_min(eps)) - torch.log((th + mean).clamp_min(eps)))
    return -(t1 + t2 + t3).mean(dim=-1)

def zinb_nll(mean, theta, pi, counts):
    """
    ZINB negative log-likelihood.
    mean  : [B, G]  NB mean μ (>0)
    theta : broadcastable to [B, G]  inverse-dispersion θ (>0)
    pi    : [B, G]  zero-inflation probability π in [0,1]
    counts: [B, G]  integer counts
    Returns: [B] per-cell mean NLL across genes
    """
    eps = 1e-6
    th = torch.as_tensor(theta, dtype=mean.dtype, device=mean.device)

    mean = mean.clamp_min(eps)
    th   = th.clamp_min(eps)
    pi   = pi.clamp(eps, 1.0 - eps)

    # log NB pmf
    log_nb = (
        torch.lgamma(counts + th)
        - torch.lgamma(th)
        - torch.lgamma(counts + 1.0)
        + th * (torch.log(th) - torch.log(th + mean))
        + counts * (torch.log(mean) - torch.log(th + mean))
    )

    is_zero    = (counts < 1e-8).to(mean.dtype)
    is_nonzero = 1.0 - is_zero

    # zeros come from mixture of inflated-zeros and NB zeros
    log_zero = torch.logaddexp(
        torch.log(pi),
        torch.log(1.0 - pi) + log_nb
    )

    loglik = is_zero * log_zero + is_nonzero * (torch.log(1.0 - pi) + log_nb)
    return -loglik.mean(dim=-1)

def _sigmoid_clamped(x, eps=1e-6):
    return torch.sigmoid(x).clamp(eps, 1 - eps)

def zinb_nll_with_reg(mean, theta, pi_logits, counts,
                      pi_reg_weight: float = 0.0,
                      pi_reg_type: str = "l2_logit",
                      beta_prior_ab: tuple = (1.0, 9.0)):
    """
    ZINB NLL + optional regularization on zero-inflation.
    pi_logits    : [B, G] raw logits of π (preferred for stable reg)
    pi_reg_type  : 'l2_logit' or 'kl_beta'
    beta_prior_ab: (a,b) for Beta prior if 'kl_beta'
    """
    pi = _sigmoid_clamped(pi_logits)
    base = zinb_nll(mean, theta, pi, counts)

    if pi_reg_weight <= 0:
        return base

    if pi_reg_type == "l2_logit":
        reg = (pi_logits ** 2).mean()
    elif pi_reg_type == "kl_beta":
        a, b = beta_prior_ab
        eps = 1e-8
        reg = -((a - 1.0) * torch.log(pi + eps) + (b - 1.0) * torch.log(1.0 - pi + eps)).mean()
    else:
        raise ValueError(f"Unknown pi_reg_type: {pi_reg_type}")

    return base + pi_reg_weight * reg

# --------------------------- MMD (RBF) ---------------------------
def _rbf(x, y, gamma):
    x2 = (x**2).sum(dim=1, keepdim=True)
    y2 = (y**2).sum(dim=1, keepdim=True)
    dist = x2 + y2.t() - 2.0 * (x @ y.t())
    return torch.exp(-gamma * torch.clamp(dist, min=0.0))

def mmd_rbf(x, y, gammas=(0.5, 1.0, 2.0)):
    """
    Unbiased multi-kernel MMD between two batches x,y (both [N,D]).
    Returns scalar.
    """
    Kxx = 0; Kyy = 0; Kxy = 0
    for g in gammas:
        Kxx = Kxx + _rbf(x, x, g)
        Kyy = Kyy + _rbf(y, y, g)
        Kxy = Kxy + _rbf(x, y, g)
    n = x.size(0)
    m = y.size(0)
    mmd = (Kxx.sum() - Kxx.diag().sum()) / (n*(n-1) + 1e-8) \
        + (Kyy.sum() - Kyy.diag().sum()) / (m*(m-1) + 1e-8) \
        - 2.0 * Kxy.mean()
    return mmd

# ============================================================================
# ✅ SMOOTH DIFFERENTIABLE COMPETITION LOSSES (NO HARD TOP-K!)
# ============================================================================

def soft_topk_overlap_loss(pred_lfc, true_lfc, temperature=0.05):
    """
    ✅ SMOOTH approximation of top-K Jaccard using softmax weighting.
    
    Instead of hard top-K selection, uses softmax to create soft weights
    that emphasize genes with high absolute LFC.
    
    FULLY DIFFERENTIABLE!
    
    Args:
        pred_lfc: [G] predicted log-fold-change
        true_lfc: [G] ground truth log-fold-change
        temperature: controls sharpness (lower = more focused on top genes)
    
    Returns:
        scalar loss in [0, 1]
    """
    # Get absolute LFCs (we care about magnitude, not direction here)
    pred_abs = torch.abs(pred_lfc)
    true_abs = torch.abs(true_lfc)
    
    # Soft weights via softmax (temperature controls focus)
    # Lower temp = more focused on truly top genes
    pred_weights = torch.softmax(pred_abs / temperature, dim=0)  # [G]
    true_weights = torch.softmax(true_abs / temperature, dim=0)  # [G]
    
    # Overlap: high where both assign high weight
    overlap = (pred_weights * true_weights).sum()
    
    # Normalize by geometric mean of weight sums
    norm = torch.sqrt(pred_weights.sum() * true_weights.sum()) + 1e-8
    
    # Loss = 1 - normalized overlap
    return 1.0 - (overlap / norm)


def soft_sign_agreement_loss(pred_lfc, true_lfc, temperature=0.05):
    """
    ✅ SMOOTH sign agreement loss (PDS surrogate).
    
    Encourages matching signs (up/down regulation) with emphasis on
    genes that have high true LFC magnitude.
    
    FULLY DIFFERENTIABLE!
    
    Args:
        pred_lfc: [G] predicted log-fold-change
        true_lfc: [G] ground truth log-fold-change
        temperature: controls focus on high-magnitude genes
    
    Returns:
        scalar loss in [0, 1]
    """
    # Soft weights based on true LFC magnitude
    # We care most about sign agreement for truly DE genes
    true_abs = torch.abs(true_lfc)
    weights = torch.softmax(true_abs / temperature, dim=0)  # [G]
    
    # Smooth sign function using tanh (differentiable!)
    # tanh is smooth sigmoid-like version of sign
    pred_signs_smooth = torch.tanh(pred_lfc * 2.0)  # Scale for sharper transition
    true_signs_smooth = torch.tanh(true_lfc * 2.0)
    
    # Agreement: 1 when signs match, 0 when opposite
    agreement = (pred_signs_smooth * true_signs_smooth + 1.0) / 2.0  # Map [-1,1] to [0,1]
    
    # Weighted agreement (emphasize important genes)
    weighted_agreement = (weights * agreement).sum()
    
    # Loss = 1 - agreement
    return 1.0 - weighted_agreement


def smooth_rank_correlation_loss(pred_lfc, true_lfc):
    """
    ✅ SMOOTH differentiable Spearman-like correlation.
    
    Encourages monotonic relationship between predicted and true LFC rankings.
    Uses normalized values instead of discrete ranks.
    
    FULLY DIFFERENTIABLE!
    
    Args:
        pred_lfc: [G] predicted log-fold-change
        true_lfc: [G] ground truth log-fold-change
    
    Returns:
        scalar loss in [0, 2] (typically [0, 1])
    """
    # Normalize to [0, 1] range (smooth version of ranking)
    pred_min, pred_max = pred_lfc.min(), pred_lfc.max()
    true_min, true_max = true_lfc.min(), true_lfc.max()
    
    pred_norm = (pred_lfc - pred_min) / (pred_max - pred_min + 1e-8)
    true_norm = (true_lfc - true_min) / (true_max - true_min + 1e-8)
    
    # Pearson correlation on normalized values
    pred_centered = pred_norm - pred_norm.mean()
    true_centered = true_norm - true_norm.mean()
    
    numerator = (pred_centered * true_centered).sum()
    denominator = torch.sqrt((pred_centered ** 2).sum() * (true_centered ** 2).sum()) + 1e-8
    
    correlation = numerator / denominator
    
    # Loss = 1 - correlation (correlation in [-1, 1], loss in [0, 2])
    return 1.0 - correlation


def lfc_magnitude_mse_loss(pred_lfc, true_lfc, focus_top_k_fraction=0.2):
    """
    ✅ SMOOTH magnitude matching loss with soft top-K focus.
    
    MSE on LFC values with soft weighting toward high-magnitude genes.
    
    FULLY DIFFERENTIABLE!
    
    Args:
        pred_lfc: [G] predicted log-fold-change
        true_lfc: [G] ground truth log-fold-change
        focus_top_k_fraction: fraction of genes to emphasize (via soft weights)
    
    Returns:
        scalar MSE loss
    """
    # Soft weights based on true LFC magnitude
    true_abs = torch.abs(true_lfc)
    
    # Create soft top-K weights using sigmoid-scaled magnitudes
    # This smoothly emphasizes high-magnitude genes
    k_threshold = torch.quantile(true_abs, 1.0 - focus_top_k_fraction)
    weights = torch.sigmoid((true_abs - k_threshold) * 10.0)  # Sharp but smooth
    weights = weights / (weights.sum() + 1e-8)  # Normalize
    
    # Weighted MSE
    squared_errors = (pred_lfc - true_lfc) ** 2
    return (weights * squared_errors).sum()


def lfc_scale_loss(pred_lfc, true_lfc):
    """
    ✅ SMOOTH LFC variance/scale matching loss.
    
    Encourages predicted LFCs to have similar spread (variance) as true LFCs.
    
    FULLY DIFFERENTIABLE!
    
    Args:
        pred_lfc: [G] predicted log-fold-change
        true_lfc: [G] ground truth log-fold-change
    
    Returns:
        scalar loss
    """
    pred_std = pred_lfc.std() + 1e-6
    true_std = true_lfc.std() + 1e-6
    
    # Log-space MSE (encourages matching orders of magnitude)
    return F.mse_loss(torch.log(pred_std), torch.log(true_std))


def listnet_loss_smooth(pred_lfc, true_lfc, temperature=0.1):
    """
    ✅ SMOOTH ListNet loss with temperature control.
    
    Cross-entropy between softmax distributions over gene scores.
    
    FULLY DIFFERENTIABLE!
    
    Args:
        pred_lfc: [G] predicted scores
        true_lfc: [G] target scores
        temperature: softmax temperature
    
    Returns:
        scalar cross-entropy loss
    """
    # Use absolute LFC for ranking (magnitude matters)
    pred_abs = torch.abs(pred_lfc)
    true_abs = torch.abs(true_lfc)
    
    # Clamp to prevent numerical issues
    pred_abs = torch.clamp(pred_abs, min=-10.0, max=10.0)
    true_abs = torch.clamp(true_abs, min=-10.0, max=10.0)
    
    # Softmax distributions
    P_true = torch.softmax(true_abs / temperature, dim=0)
    log_P_pred = torch.log_softmax(pred_abs / temperature, dim=0)
    
    # Cross-entropy: -sum(P_true * log(P_pred))
    return -(P_true * log_P_pred).sum()

def hard_gene_selection_loss(pred_lfc, true_lfc, k_list=[20, 50, 200]):
    """
    ✅ SUPERVISED loss that directly teaches which genes should be top-K.
    
    For the TRUE top-K genes, we want predicted LFC to be HIGH.
    For the TRUE bottom genes, we want predicted LFC to be LOW.
    
    This is differentiable and directly optimizes gene selection!
    """
    total_loss = 0.0
    
    for k in k_list:
        k = min(k, pred_lfc.numel())
        
        # Get TRUE top-K and bottom-K genes
        true_abs = torch.abs(true_lfc)
        _, true_topk_idx = torch.topk(true_abs, k=k, largest=True)
        _, true_bottomk_idx = torch.topk(true_abs, k=k, largest=False)
        
        # For top-K genes: penalize if predicted magnitude is LOW
        pred_topk_abs = torch.abs(pred_lfc[true_topk_idx])
        # We want these to be HIGH, so minimize: -log(pred_abs)
        # Or equivalently: 1 / (pred_abs + eps)
        loss_topk = (1.0 / (pred_topk_abs + 0.1)).mean()
        
        # For bottom-K genes: penalize if predicted magnitude is HIGH
        pred_bottomk_abs = torch.abs(pred_lfc[true_bottomk_idx])
        # We want these to be LOW, so minimize: pred_abs
        loss_bottomk = pred_bottomk_abs.mean()
        
        # Combine (top-K should be high, bottom-K should be low)
        total_loss += loss_topk + loss_bottomk
    
    return total_loss / len(k_list)


def focal_gene_selection_loss(pred_lfc, true_lfc, k=200, gamma=2.0):
    """
    ✅ FOCAL loss variant for gene selection.
    
    Heavily penalizes misranking of truly important genes.
    Uses focal loss idea: focus on hard examples.
    """
    # Get true top-K genes
    true_abs = torch.abs(true_lfc)
    _, true_topk_idx = torch.topk(true_abs, k=k)
    
    # Create target: 1 for top-K genes, 0 for others
    target = torch.zeros_like(pred_lfc)
    target[true_topk_idx] = 1.0
    
    # Predicted "probability" of being in top-K (via sigmoid)
    pred_abs = torch.abs(pred_lfc)
    pred_prob = torch.sigmoid(pred_abs - pred_abs.median())  # Center around median
    
    # Focal loss: -(1-p)^gamma * log(p) for positive, -p^gamma * log(1-p) for negative
    pos_loss = -(1 - pred_prob) ** gamma * torch.log(pred_prob + 1e-8)
    neg_loss = -pred_prob ** gamma * torch.log(1 - pred_prob + 1e-8)
    
    # Apply based on target
    loss = target * pos_loss + (1 - target) * neg_loss
    
    return loss.mean()


def ranking_hinge_loss(pred_lfc, true_lfc, margin=0.5):
    """
    ✅ RANKING hinge loss: top genes should be ranked higher than bottom genes.
    
    For every (top_gene, bottom_gene) pair:
    We want: |pred_lfc[top]| > |pred_lfc[bottom]| + margin
    """
    true_abs = torch.abs(true_lfc)
    pred_abs = torch.abs(pred_lfc)
    
    # Get top 500 and bottom 500 genes
    k = min(500, pred_lfc.numel() // 4)
    _, top_idx = torch.topk(true_abs, k=k, largest=True)
    _, bottom_idx = torch.topk(true_abs, k=k, largest=False)
    
    # Sample pairs (for efficiency)
    n_samples = min(1000, k * k)
    top_samples = top_idx[torch.randint(0, k, (n_samples,))]
    bottom_samples = bottom_idx[torch.randint(0, k, (n_samples,))]
    
    # Hinge loss: max(0, margin - (pred[top] - pred[bottom]))
    pred_top = pred_abs[top_samples]
    pred_bottom = pred_abs[bottom_samples]
    
    hinge = torch.relu(margin + pred_bottom - pred_top)
    
    return hinge.mean()


# ============================================================================
# FULL LOSS COMPUTATION
# ============================================================================

def compute_losses(batch, out, cfg: TrainConfig, model: DualEncoderVAE):
    """
    Full loss with SMOOTH, DIFFERENTIABLE competition losses.
    
    All hard top-K selection replaced with soft approximations!
    """
    x          = batch["x"]
    is_control = out["is_control"]
    device     = x.device

    # --- per-gene weights (optional) ---
    gene_w = None
    if hasattr(model, "gene_w") and (model.gene_w is not None):
        gw = model.gene_w
        if isinstance(gw, torch.Tensor) and gw.ndimension() == 1 and gw.numel() == x.size(1):
            gene_w = gw.to(device=device, dtype=x.dtype)

    # Masks
    mask_c = (is_control > 0.5)   # controls
    mask_p = (is_control < 0.5)   # perturbed

    # --- reconstruction (weighted MSE) ---
    delta = cfg.huber_delta
    L_rec_c = mse_loss_weighted(out["x_rec_c"], x, gene_w=gene_w, huber_delta=delta)
    L_rec_p = mse_loss_weighted(out["x_rec_p"], x, gene_w=gene_w, huber_delta=delta)
    L_rec   = cfg.lambda_rec * (
        (L_rec_c[mask_c].mean() if mask_c.any() else torch.zeros((), device=device))
      + (L_rec_p[mask_p].mean() if mask_p.any() else torch.zeros((), device=device))
    )

    # --- cross-domain reconstruction ---
    L_xrec_cp = mse_loss_weighted(out["x_pred_from_c"],      x, gene_w=gene_w, huber_delta=delta)
    L_xrec_pc = mse_loss_weighted(out["x_pred_from_p_to_c"], x, gene_w=gene_w, huber_delta=delta)
    L_xrec    = torch.zeros((), device=device)
    if mask_p.any():
        L_xrec = L_xrec + cfg.lambda_xrec * L_xrec_cp[mask_p].mean()
    if mask_c.any():
        L_xrec = L_xrec + cfg.lambda_xrec * L_xrec_pc[mask_c].mean()

    # --- KL ---
    L_kl_c = kl_divergence(out["mu_c"], out["logvar_c"])
    L_kl_p = kl_divergence(out["mu_p"], out["logvar_p"])
    L_kl   = cfg.lambda_kl * (
        (L_kl_c[mask_c].mean() if mask_c.any() else torch.zeros((), device=device))
      + (L_kl_p[mask_p].mean() if mask_p.any() else torch.zeros((), device=device))
    )

    # --- Δ consistency ---
    L_delta = torch.zeros((), device=device)
    if mask_p.any() and (cfg.lambda_delta > 0.0):
        L_delta_all = delta_consistency(out["z_p"], out["z_c"], out["Delta"])
        L_delta     = cfg.lambda_delta * L_delta_all[mask_p].mean()

    # --- orthogonality ---
    L_orth = torch.zeros((), device=device)
    if cfg.lambda_orth > 0.0:
        L_orth = cfg.lambda_orth * cosine_orthogonality(out["z_c"], out["Delta"]).mean()

    # --- InfoNCE ---
    L_nce = torch.zeros((), device=device)
    if (mask_p.any()) and (cfg.lambda_nce > 0.0):
        anchor = out["z_c"][mask_p] + out["Delta"][mask_p]
        pos    = out["z_p"][mask_p]
        if anchor.size(0) > 1:
            L_nce = cfg.lambda_nce * info_nce(anchor, pos, temperature=cfg.nce_temperature)

    # --- adversarial ---
    L_adv = torch.zeros((), device=device)
    if (model.adv is not None) and mask_p.any() and (cfg.lambda_adv > 0.0):
        zc_p   = out["z_c"][mask_p]
        gid_p  = batch["gid"][mask_p]
        logits_cls = model.adv(zc_p.detach())
        L_adv_cls  = F.cross_entropy(logits_cls, gid_p)
        logits_enc = model.adv(zc_p)
        L_adv_rev  = F.cross_entropy(logits_enc, gid_p)
        L_adv      = cfg.lambda_adv * (L_adv_cls - L_adv_rev)

    # --- Δ-smoothness ---
    L_smooth = torch.zeros((), device=device)
    if cfg.lambda_smooth > 0.0:
        gid   = batch["gid"]
        gvec  = out["g"]
        Delta = out["Delta"]
        mask_seen = (gid > 0)
        if mask_seen.any():
            Gn = F.normalize(gvec[mask_seen], dim=-1)
            D  = Delta[mask_seen]
            S  = Gn @ Gn.t()
            S.fill_diagonal_(-1.0)
            k = cfg.smooth_k
            k = max(1, min(k, Gn.size(0) - 1))
            if k >= 1 and Gn.size(0) >= 2:
                vals, idx = torch.topk(S, k=k, dim=1)
                diffs = D.unsqueeze(1) - D[idx]
                w = torch.relu(vals).unsqueeze(-1)
                L_smooth = cfg.lambda_smooth * (w * (diffs ** 2)).mean()

    # --- count losses ---
    L_count_rec  = torch.zeros((), device=device)
    L_count_xrec = torch.zeros((), device=device)

    use_counts = ((cfg.lambda_count_rec> 0.0) or (cfg.lambda_count_xrec > 0.0)) \
                 and hasattr(model, "dec") and hasattr(model.dec, "counts_rate")

    if use_counts:
        y_true   = torch.expm1(x).clamp_min_(0.0)
        max_rate = cfg.count_max_rate
        count_link = model.cfg.count_link.lower()
        use_zinb = count_link.startswith("zinb")
        use_nb = count_link.startswith("nb") and not use_zinb

        def _min_pos(t):
            return max(1e-8, float(torch.finfo(t.dtype).tiny))

        def _nll_from_xhat(xhat, apply_mask=None):
            rate = model.dec.counts_rate(xhat.float())
            if rate is None:
                return None
            rate = rate.clamp(min=_min_pos(rate), max=max_rate)
            
            if use_zinb:
                if hasattr(model.dec, "zinb_pi_head") and model.dec.zinb_pi_head is not None:
                    pi_logits = model.dec.zinb_pi_head(xhat.float())
                else:
                    pi_logits = torch.full_like(rate, -10.0)
                
                theta = model.cfg.nb_theta
                pi_reg_weight = cfg.lambda_zinb_pi_reg
                pi_reg_type = cfg.zinb_pi_reg_type
                beta_a = cfg.zinb_beta_a
                beta_b = cfg.zinb_beta_b
                
                nll = zinb_nll_with_reg(
                    rate, theta, pi_logits, y_true,
                    pi_reg_weight=pi_reg_weight,
                    pi_reg_type=pi_reg_type,
                    beta_prior_ab=(beta_a, beta_b)
                )
            elif use_nb:
                theta = model.cfg.nb_theta
                nll = nb_nll(rate, theta, y_true)
            else:
                nll = poisson_nll(rate, y_true)
            
            if apply_mask is not None and apply_mask.any():
                nll = nll[apply_mask].mean()
            else:
                nll = nll.mean()
            
            return nll

        if cfg.lambda_count_rec > 0.0:
            nll_rec_c = _nll_from_xhat(out["x_rec_c"], apply_mask=mask_c if mask_c.any() else None)
            nll_rec_p = _nll_from_xhat(out["x_rec_p"], apply_mask=mask_p if mask_p.any() else None)
            
            if nll_rec_c is not None:
                L_count_rec = L_count_rec + nll_rec_c
            if nll_rec_p is not None:
                L_count_rec = L_count_rec + nll_rec_p
            
            L_count_rec = cfg.lambda_count_rec * L_count_rec

        if cfg.lambda_count_xrec > 0.0:
            nll_cp = _nll_from_xhat(out["x_pred_from_c"], apply_mask=mask_p if mask_p.any() else None)
            nll_pc = _nll_from_xhat(out["x_pred_from_p_to_c"], apply_mask=mask_c if mask_c.any() else None)
            
            if nll_cp is not None:
                L_count_xrec = L_count_xrec + nll_cp
            if nll_pc is not None:
                L_count_xrec = L_count_xrec + nll_pc
            
            L_count_xrec = cfg.lambda_count_xrec * L_count_xrec
    
    # --- MMD ---
    L_mmd = torch.zeros((), device=device)
    if (mask_p.any()) and (cfg.lambda_mmd > 0.0):
        X_real = x[mask_p]
        X_pred = out["x_pred_from_c"][mask_p]
        Xr = F.layer_norm(X_real, X_real.shape[-1:])
        Xp = F.layer_norm(X_pred, X_pred.shape[-1:])
        L_mmd = cfg.lambda_mmd * mmd_rbf(Xp, Xr)

    # ========================================================================
    # ✅ SMOOTH COMPETITION LOSSES (PER-PERTURBATION PSEUDOBULK)
    # ========================================================================
    L_soft_topk      = torch.zeros((), device=device)
    L_soft_sign      = torch.zeros((), device=device)
    L_rank_corr      = torch.zeros((), device=device)
    L_lfc_magnitude  = torch.zeros((), device=device)
    L_lfc_scale      = torch.zeros((), device=device)
    L_listnet        = torch.zeros((), device=device)
    
    L_hard_gene_selection  = torch.zeros((), device=device)
    L_focal_gene_selection = torch.zeros((), device=device)
    L_ranking_hinge        = torch.zeros((), device=device)

    if mask_p.any() and (cfg.lambda_soft_topk > 0.0 or cfg.lambda_soft_sign > 0.0 
                          or cfg.lambda_rank_corr > 0.0 or cfg.lambda_lfc_magnitude > 0.0
                          or cfg.lambda_lfc_scale > 0.0 or cfg.lambda_listnet > 0.0):
        
        batch_targets = batch.get("gid", None)
        
        if batch_targets is not None:
            unique_targets = torch.unique(batch_targets[mask_p])
            
            per_target_soft_topk = []
            per_target_soft_sign = []
            per_target_rank_corr = []
            per_target_lfc_mag   = []
            per_target_lfc_scale = []
            per_target_listnet   = []
            per_target_hard_gene = []
            per_target_focal     = []
            per_target_hinge     = []
            
            for target_id in unique_targets:
                target_mask_p = (batch_targets == target_id) & mask_p
                target_mask_c = mask_c
                
                if target_mask_p.sum() < 3 or target_mask_c.sum() < 3:
                    continue
                
                # PSEUDOBULK
                ctrl_bulk = x[target_mask_c].mean(dim=0)  # [G]
                pert_real_bulk = x[target_mask_p].mean(dim=0)  # [G]
                lfc_real = pert_real_bulk - ctrl_bulk
                
                x_pred_p = out["x_pred_from_c"][target_mask_p]
                pert_pred_bulk = x_pred_p.mean(dim=0)  # [G]
                lfc_pred = pert_pred_bulk - ctrl_bulk
                
                # ✅ ALL SMOOTH LOSSES
                if cfg.lambda_soft_topk > 0.0:
                    loss = soft_topk_overlap_loss(lfc_pred, lfc_real, temperature=0.05)
                    per_target_soft_topk.append(loss)
                
                if cfg.lambda_soft_sign > 0.0:
                    loss = soft_sign_agreement_loss(lfc_pred, lfc_real, temperature=0.05)
                    per_target_soft_sign.append(loss)
                
                if cfg.lambda_rank_corr > 0.0:
                    loss = smooth_rank_correlation_loss(lfc_pred, lfc_real)
                    per_target_rank_corr.append(loss)
                
                if cfg.lambda_lfc_magnitude > 0.0:
                    loss = lfc_magnitude_mse_loss(lfc_pred, lfc_real, focus_top_k_fraction=0.2)
                    per_target_lfc_mag.append(loss)
                
                if cfg.lambda_lfc_scale > 0.0:
                    loss = lfc_scale_loss(lfc_pred, lfc_real)
                    per_target_lfc_scale.append(loss)
                
                if cfg.lambda_listnet > 0.0:
                    loss = listnet_loss_smooth(lfc_pred, lfc_real, temperature=0.1)
                    per_target_listnet.append(loss)
                
                # NEW SUPERVISED GENE SELECTION LOSSES
                if cfg.lambda_hard_gene_selection > 0.0:
                    loss = hard_gene_selection_loss(lfc_pred, lfc_real, k_list=[20, 50, 200])
                    per_target_hard_gene.append(loss)
                
                if cfg.lambda_focal_gene_selection > 0.0:
                    loss = focal_gene_selection_loss(lfc_pred, lfc_real, k=200, gamma=2.0)
                    per_target_focal.append(loss)
                
                if cfg.lambda_ranking_hinge > 0.0:
                    loss = ranking_hinge_loss(lfc_pred, lfc_real, margin=0.5)
                    per_target_hinge.append(loss)
            
            # Average over perturbations in batch
            if per_target_soft_topk:
                L_soft_topk = cfg.lambda_soft_topk * torch.stack(per_target_soft_topk).mean()
            if per_target_soft_sign:
                L_soft_sign = cfg.lambda_soft_sign * torch.stack(per_target_soft_sign).mean()
            if per_target_rank_corr:
                L_rank_corr = cfg.lambda_rank_corr * torch.stack(per_target_rank_corr).mean()
            if per_target_lfc_mag:
                L_lfc_magnitude = cfg.lambda_lfc_magnitude * torch.stack(per_target_lfc_mag).mean()
            if per_target_lfc_scale:
                L_lfc_scale = cfg.lambda_lfc_scale * torch.stack(per_target_lfc_scale).mean()
            if per_target_listnet:
                L_listnet = cfg.lambda_listnet * torch.stack(per_target_listnet).mean()
            
            if per_target_hard_gene:
                L_hard_gene_selection = cfg.lambda_hard_gene_selection * torch.stack(per_target_hard_gene).mean()
            if per_target_focal:
                L_focal_gene_selection = cfg.lambda_focal_gene_selection * torch.stack(per_target_focal).mean()
            if per_target_hinge:
                L_ranking_hinge = cfg.lambda_ranking_hinge * torch.stack(per_target_hinge).mean()
        

    # --- Total loss ---
    total = (L_rec + L_xrec + L_kl + L_delta + L_orth + L_nce + L_adv + L_smooth 
             + L_count_rec + L_count_xrec + L_mmd 
             + L_soft_topk + L_soft_sign + L_rank_corr 
             + L_lfc_magnitude + L_lfc_scale + L_listnet
             + L_hard_gene_selection
             + L_focal_gene_selection
             +L_ranking_hinge)

    # --- Loss dictionary ---
    loss_dict = {
        "loss":         _to_float(total),
        "rec":          _to_float(L_rec),
        "xrec":         _to_float(L_xrec),
        "kl":           _to_float(L_kl),
        "delta":        _to_float(L_delta),
        "orth":         _to_float(L_orth),
        "nce":          _to_float(L_nce),
        "adv":          _to_float(L_adv),
        "smo":          _to_float(L_smooth),
        "cnt_rec":      _to_float(L_count_rec),
        "cnt_xrec":     _to_float(L_count_xrec),
        "mmd":          _to_float(L_mmd),
        "soft_topk":    _to_float(L_soft_topk),     # ✅ Smooth Jaccard
        "soft_sign":    _to_float(L_soft_sign),     # ✅ Smooth PDS
        "rank_corr":    _to_float(L_rank_corr),     # ✅ Spearman-like
        "lfc_mag":      _to_float(L_lfc_magnitude), # ✅ Magnitude MSE
        "lfc_scale":    _to_float(L_lfc_scale),     # ✅ Variance matching
        "listnet":      _to_float(L_listnet),       # ✅ Smooth ranking
        "hard_gene":    _to_float(L_hard_gene_selection),   # ✅ Smooth ranking
        "focal_gene":   _to_float(L_focal_gene_selection),  # ✅ Smooth ranking
        "rank_hinge":   _to_float(L_ranking_hinge),         # ✅ Smooth ranking
    }

    return total, loss_dict