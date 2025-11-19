import os
import random
import torch
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from typing import Optional, Tuple, List, Dict, Sequence
from sklearn.preprocessing import LabelEncoder
from .config import PredictConfig


def set_all_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def logger(stage:str, msg: str):
    print(f"{stage} {msg}")

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def ensure_csr(X):
    return X if sparse.isspmatrix_csr(X) else X.tocsr()

def detect_logged_data(X, check_n: int = 200, int_frac_tol: float = 0.98, max_log_cap: float = 20.0) -> bool:
    """
    Detects whether data appear to be log-transformed already.
    Criteria:
      - non-negative
      - mostly non-integer values
      - upper bound reasonable for log1p-normalized data
    """
    if sparse.issparse(X):
        sample = X[:min(check_n, X.shape[0])].toarray()
    else:
        sample = np.asarray(X[:min(check_n, X.shape[0])])

    if sample.size == 0 or np.allclose(sample, 0.0):
        return False

    nonneg = (sample >= 0).all()
    intlike_frac = np.mean(np.isclose(sample, np.round(sample), atol=1e-6))
    maxv = float(sample.max())
    return bool(nonneg and (intlike_frac < int_frac_tol) and (maxv <= max_log_cap))

def prepare_normalized_views(
    adata: ad.AnnData,
    normalize: bool = True,
    target_sum: float | None = None,
    return_adata: bool = False,):    

    """
    Given raw-count AnnData, return:
      - X_counts_csr (raw)
      - X_log1p (normalized + log1p)
    """
    
    X_counts_csr = ensure_csr(adata.X)
    
    if not normalize:
        logger("[prep]", "skipping normalization (log1p only)")
        adata_log = adata.copy()
        sc.pp.log1p(adata_log)
        return (X_counts_csr, adata_log.X, adata_log) if return_adata else (X_counts_csr, adata_log.X)
    
    ts = None if target_sum is None else float(target_sum)
    logger("[prep]", f"normalizing to target_sum={ts} and applying log1p...")
    
    adata_norm = adata.copy()
    sc.pp.normalize_total(adata_norm, target_sum=ts, inplace=True)
    sc.pp.log1p(adata_norm)
    
    return (X_counts_csr, adata_norm.X, adata_norm) if return_adata else (X_counts_csr, adata_norm.X)

def bounded_output_sigmoid(x, min_val=0.0, max_val=12.0):
    return torch.sigmoid(x) * (max_val - min_val) + min_val

def bounded_output_tanh(x, min_val=-4.0, max_val=12.0):
    scale = (max_val - min_val) / 2.0
    shift = (max_val + min_val) / 2.0
    return torch.tanh(x) * scale + shift

def param_groups_no_decay(model, weight_decay):
    no_decay = []
    decay = []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.endswith(".bias") or "ln" in n.lower() or "layernorm" in n.lower():
            no_decay.append(p)
        else:
            decay.append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]

def load_label_encoders(outdir: str, cols: List[str]) -> Dict[str, LabelEncoder]:
    encoders = {}
    for col in cols:
        npy_path = os.path.join(outdir, f"{col}_classes.npy")
        if os.path.exists(npy_path):
            classes = np.load(npy_path, allow_pickle=True)
            le = LabelEncoder()
            le.classes_ = np.array(classes, dtype=str)
            encoders[col] = le
    return encoders

def build_gene_vocab(adata: ad.AnnData, control_token: str = "non-targeting", col_target: str = "target_gene") -> Dict[str, int]:
    """
    Build a FULL mapping:
      id 0 -> control_token
      1..K -> all perturbed genes observed in .obs[col_target] (excluding control_token), sorted
      K+1.. -> all remaining genes from .var_names, sorted, that are not already in the map

    Result length ~= 1 + (#unique perturbation targets) + (#var genes not in targets)
    """
    # start with control at 0
    gene2id: Dict[str, int] = {control_token: 0}
    next_id = 1

    # 1) add observed perturbation targets (if column exists)
    perturbed = []
    if col_target in adata.obs.columns:
        vals = adata.obs[col_target].astype(str).values
        unique_targets = sorted(set(vals) - {control_token})
        perturbed = unique_targets
        for g in unique_targets:
            if g not in gene2id:
                gene2id[g] = next_id
                next_id += 1

    # 2) add all remaining assay genes from var_names
    var_genes = list(map(str, adata.var_names))
    for g in sorted(var_genes):
        if g not in gene2id:
            gene2id[g] = next_id
            next_id += 1

    return gene2id

def compute_gene_pc_embedding_from_controls(adata,
                                            col_target: str,
                                            control_token: str,
                                            embed_dim: int,
                                            max_cells: int = 200_000,
                                            random_state: int = 42):
    """
    Sparse-safe PCA-like init for gene embeddings using TruncatedSVD on controls.

    Steps:
      1) Filter controls and (optionally) subsample to `max_cells`.
      2) Ensure log1p has been applied upstream (your AnnDataDataset already does this).
      3) (Optional but recommended) per-gene variance scaling using a sparse right-multiply by diag(1/std).
      4) Fit TruncatedSVD (no centering) on the sparse matrix.
      5) Use components_.T as gene embeddings, then L2-normalize rows.

    Returns:
      gene_emb_var : np.ndarray [n_genes, embed_dim]
      var_names    : List[str]   (training gene order)
    """

    # 1) filter to controls
    if col_target not in adata.obs.columns:
        raise ValueError(f"Column '{col_target}' not found in .obs.")
    is_ctrl = (adata.obs[col_target].astype(str).values == str(control_token))
    if is_ctrl.sum() == 0:
        raise RuntimeError("No control cells found to initialize gene embeddings.")
    adata_ctrl = adata[is_ctrl].copy()

    # 2) subsample for memory, if needed
    if adata_ctrl.n_obs > max_cells:
        idx = np.random.choice(adata_ctrl.n_obs, size=max_cells, replace=False)
        adata_ctrl = adata_ctrl[idx].copy()

    # 3) get sparse X (already log1p upstream)
    X = adata_ctrl.X
    if not sparse.issparse(X):
        X = sparse.csr_matrix(X)
    X = X.tocsr()  # ensure CSR for fast row ops

    # 4) per-gene scale (variance normalize) without centering
    N = X.shape[0]
    sums = np.asarray(X.sum(axis=0)).ravel().astype(np.float64)
    sq_sums = np.asarray(X.power(2).sum(axis=0)).ravel().astype(np.float64)
    mean = sums / max(1, N)
    var  = (sq_sums / max(1, N)) - (mean ** 2)
    std  = np.sqrt(np.clip(var, 0.0, None)) + 1e-6

    # Right-multiply by diag(1/std) to scale columns sparsely
    inv_std = (1.0 / std).astype(np.float32)
    # X_scaled = X @ sparse.diags(inv_std, offsets=0, shape=(X.shape[1], X.shape[1]), dtype=np.float32)
    X_scaled = X.copy()
    X_scaled.data *= inv_std[X_scaled.indices]

    # 5) TruncatedSVD on sparse (no explicit centering)
    k = int(min(embed_dim, X_scaled.shape[0] - 1, X_scaled.shape[1] - 1))
    k = max(1, k)
    if k > min(X_scaled.shape) - 1:
        k = min(X_scaled.shape) - 1
    if k <= 0:
        raise ValueError(f"embed_dim too large for matrix shape {X_scaled.shape}.")
    svd = TruncatedSVD(n_components=k, random_state=random_state)
    svd.fit(X_scaled)

    # components_: [k, n_genes] → transpose to [n_genes, k]
    gene_emb_var = svd.components_.T.astype(np.float32, copy=False)

    # 6) L2-normalize per gene
    norms = np.linalg.norm(gene_emb_var, axis=1, keepdims=True) + 1e-6
    gene_emb_var = gene_emb_var / norms

    return gene_emb_var, list(map(str, adata.var_names))

def _subsample_indices_stratified(obs, mask, n_max, by_col="batch", seed=42):
    """Return indices of at most n_max True entries in `mask`, stratified by obs[by_col] if present."""
    rng = np.random.RandomState(seed)
    idx_all = np.where(mask)[0]
    if len(idx_all) <= n_max:
        return idx_all
    if (by_col is None) or (by_col not in obs.columns):
        return rng.choice(idx_all, size=n_max, replace=False)
    # per-batch proportional sampling
    obs_sub = obs.iloc[idx_all]
    counts = obs_sub[by_col].value_counts()
    frac = n_max / float(len(idx_all))
    take_per_batch = (counts * frac).astype(int).clip(lower=1)    
    chosen = []
    for b, k in take_per_batch.items():
        cand = idx_all[obs_sub[by_col].values == b]
        k = min(k, len(cand))
        chosen.append(rng.choice(cand, size=k, replace=False))    
    chosen = np.concatenate(chosen)
    # pad if rounding caused deficit
    if chosen.size < n_max:
        remaining = np.setdiff1d(idx_all, chosen, assume_unique=False)
        pad_k = min(n_max - chosen.size, remaining.size)
        if pad_k > 0:
            chosen = np.concatenate([chosen, rng.choice(remaining, size=pad_k, replace=False)])
    return chosen

def compute_gene_weights_from_controls_sparse(
    adata: ad.AnnData,
    col_target: str,
    control_token: str,
    max_cells: int = 200_000,
    stratify_by_batch: bool = True,
    seed: int = 42,
    clip_min: float = 0.1,
    clip_max: float = 10.0,
) -> np.ndarray:
    """
    Compute inverse-variance per-gene weights on *log1p* data from controls only, using sparse ops.
    Returns weights shape [G], normalized to mean=1, clipped to [clip_min, clip_max] pre-normalization.
    """
    # Controls mask
    if col_target not in adata.obs.columns:
        raise ValueError(f"Column '{col_target}' not found in .obs.")
    is_ctrl = (adata.obs[col_target].astype(str).values == str(control_token))
    if is_ctrl.sum() == 0:
        raise RuntimeError("No control cells found for gene weighting.")
    idx_ctrl = np.where(is_ctrl)[0]

    # Subsample for memory
    if idx_ctrl.size > max_cells:
        idx_ctrl = _subsample_indices_stratified(
            adata.obs, is_ctrl, max_cells,
            by_col=("batch" if stratify_by_batch and "batch" in adata.obs.columns else None),
            seed=seed
        )

    # Ensure CSR
    X = adata.X[idx_ctrl]
    if not sparse.isspmatrix_csr(X):
        X = sparse.csr_matrix(X)

    N = X.shape[0]
    # per-gene sums and squared sums (on log1p scale)
    sums   = np.asarray(X.sum(axis=0)).ravel().astype(np.float64)
    sqsum  = np.asarray(X.power(2).sum(axis=0)).ravel().astype(np.float64)

    mean = sums / max(1, N)
    var  = (sqsum / max(1, N)) - mean**2
    var[var < 1e-8] = 1e-8  # guard

    # inverse-variance weights
    w = 1.0 / var
    # clip extreme weights to avoid exploding gradients
    w = np.clip(w, clip_min, clip_max)
    # normalize to mean 1 (keeps overall loss scale stable)
    w = (w / w.mean()).astype(np.float32)

    return w

def build_full_init_matrix_for_gene_embedding(
    gene2id: Dict[str, int],
    var_names: List[str],
    gene_emb_var: np.ndarray,
    embed_dim: int,
    init_std: float = 0.02,
    case_insensitive: bool = True,
) -> np.ndarray:
    V = max(gene2id.values()) + 1
    W = np.random.normal(0.0, init_std, size=(V, embed_dim)).astype(np.float32)
    W[0].fill(0.0)

    # deduplicate var_names preserving first
    seen = set()
    name2row = {}
    for i, g in enumerate(var_names):
        key = g.upper() if case_insensitive else str(g)
        if key in seen:
            continue
        seen.add(key)
        name2row[key] = i

    for g, gid in gene2id.items():
        if gid == 0:
            continue
        key = g.upper() if case_insensitive else str(g)
        j = name2row.get(key)
        if j is not None and j < gene_emb_var.shape[0]:
            W[gid] = gene_emb_var[j]
    return W

def _load_pt_gene_embeddings(pt_path: str) -> Tuple[List[str], np.ndarray]:
    """Load .pt dict {gene -> tensor[D_pre]} -> (names, np.array [M, D_pre])."""
    if not os.path.exists(pt_path):
        raise FileNotFoundError(f"Pretrained embedding file not found: {pt_path}")
    obj = torch.load(pt_path, map_location="cpu")
    if not isinstance(obj, dict):
        raise TypeError(f"Expected dict in {pt_path}, got {type(obj)}")
    names, rows = [], []
    dim_ref = None
    for k, v in obj.items():
        name = str(k)
        if isinstance(v, torch.Tensor):
            vec = v.detach().cpu().float().view(-1)
            if dim_ref is None:
                dim_ref = vec.numel()
            elif vec.numel() != dim_ref:
                continue
            names.append(name)
            rows.append(vec.numpy())
    if not rows:
        raise RuntimeError(f"No consistent tensors in {pt_path}.")
    mat = np.stack(rows, axis=0).astype(np.float32, copy=False)
    return names, mat   # [M], [M, D_pre]

def prepare_gene_embedding(
    gene2id: Dict[str, int],
    var_names: List[str],
    embed_dim: int,
    pretrained_pt_path: Optional[str] = None,    # .pt dict {gene -> tensor[D_pre]}
    adata_for_pca: Optional[ad.AnnData] = None,  # for PCA init or fill
    col_target: str = "target_gene",
    control_token: str = "non-targeting",
    norm: str = "l2",
    case_insensitive: bool = True,
    zero_control_row: bool = True,
    random_state: int = 42,
    use_pca_init: bool = True,                   # toggles PCA init ON/OFF
    max_cells: int = 200_000,                    # passed to PCA helper
) -> Optional[np.ndarray]:
    """
    Returns W_init [V, embed_dim] or None (random init).

    Logic:
    - If pretrained_pt_path is provided:
        * Copy overlapping genes from pretrained table.
        * If use_pca_init=True and adata_for_pca available, fill missing genes with PCA.
        * Reduce to embed_dim using TruncatedSVD.
    - If no pretrained:
        * If use_pca_init=True → build via PCA on controls in adata_for_pca.
        * Else return None → random init inside model.
    """
    rng = np.random.RandomState(random_state)
    V = max(gene2id.values()) + 1

    def _name_map(names: List[str]) -> Dict[str, int]:
        return ({str(n).upper(): i for i, n in enumerate(names)}
                if case_insensitive else {str(n): i for i, n in enumerate(names)})

    # --------------------------------------------------------------------------
    # A) Pretrained embeddings path
    # --------------------------------------------------------------------------
    if pretrained_pt_path is not None:
        pt_names, pt_mat = _load_pt_gene_embeddings(pretrained_pt_path)
        D_pre = pt_mat.shape[1]
        logger("[pretrain]", f"loaded pretrained table with {len(pt_names)} genes, dim={D_pre}")

        # base random table
        W_full_pre = rng.normal(0.0, 0.02, size=(V, D_pre)).astype(np.float32)
        if zero_control_row:
            W_full_pre[0].fill(0.0)

        map_pt  = _name_map(pt_names)
        map_var = _name_map(var_names)

        # Fill overlaps
        hits = 0
        placed_mask = np.zeros(V, dtype=bool)
        for g, gid in gene2id.items():
            if gid == 0:
                continue
            j = map_pt.get(str(g).upper() if case_insensitive else str(g))
            if j is not None:
                W_full_pre[gid] = pt_mat[j]
                placed_mask[gid] = True
                hits += 1
        logger("[pretrain]", f"found overlaps for {hits}/{V} genes")

        # Fill missing with PCA if allowed
        if use_pca_init and adata_for_pca is not None:
            try:
                logger("[pretrain]", f"filling missing genes via PCA (max_cells={max_cells})...")
                Gpca, _ = compute_gene_pc_embedding_from_controls(
                    adata_for_pca, col_target=col_target, control_token=control_token,
                    embed_dim=D_pre, max_cells=max_cells, random_state=random_state
                )
                norms = np.linalg.norm(Gpca, axis=1, keepdims=True) + 1e-6
                Gpca = (Gpca / norms).astype(np.float32, copy=False)
                filled = 0
                for g, gid in gene2id.items():
                    if gid == 0 or placed_mask[gid]:
                        continue
                    k = map_var.get(str(g).upper() if case_insensitive else str(g))
                    if (k is not None) and (k < Gpca.shape[0]):
                        W_full_pre[gid] = Gpca[k]
                        filled += 1
                logger("[pretrain]", f"PCA-filled {filled} missing genes")
            except Exception as e:
                logger("[warn]", f"PCA fill failed, keeping random for missing. Reason: {e}")
        else:
            logger("[pretrain]", "skipping PCA fill (use_pca_init=False or no adata_for_pca)")

        # Optional normalization + SVD projection
        if norm.lower() == "l2":
            norms = np.linalg.norm(W_full_pre[1:], axis=1, keepdims=True) + 1e-6
            W_full_pre[1:] = (W_full_pre[1:] / norms).astype(np.float32, copy=False)

        k = int(min(embed_dim, W_full_pre.shape[0] - 1, W_full_pre.shape[1] - 1))
        svd = TruncatedSVD(n_components=k, random_state=random_state)
        svd.fit(W_full_pre)
        W_init = svd.transform(W_full_pre).astype(np.float32, copy=False)

        if k < embed_dim:
            pad = rng.normal(0, 0.02, size=(V, embed_dim - k)).astype(np.float32)
            W_init = np.concatenate([W_init, pad], axis=1)
        if zero_control_row:
            W_init[0].fill(0.0)

        logger("[pretrain]", f"gene embeddings projected from pretrained ({D_pre}→{embed_dim}), shape={W_init.shape}")
        return W_init

    # --------------------------------------------------------------------------
    # B) No pretrained → PCA or random
    # --------------------------------------------------------------------------
    if use_pca_init:
        if adata_for_pca is None:
            logger("[warn]", "PCA init requested but adata_for_pca=None → random init")
            return None
        try:
            logger("[pretrain]", f"computing PCA on control cells (max_cells={max_cells})...")
            Gpca, _ = compute_gene_pc_embedding_from_controls(
                adata_for_pca, col_target=col_target, control_token=control_token,
                embed_dim=embed_dim, max_cells=max_cells, random_state=random_state
            )
            W_init = build_full_init_matrix_for_gene_embedding(
                gene2id=gene2id, var_names=var_names, gene_emb_var=Gpca,
                embed_dim=embed_dim, init_std=0.02, case_insensitive=case_insensitive
            )
            if norm.lower() == "l2":
                norms = np.linalg.norm(W_init[1:], axis=1, keepdims=True) + 1e-6
                W_init[1:] = (W_init[1:] / norms).astype(np.float32, copy=False)
            if zero_control_row:
                W_init[0].fill(0.0)
            logger("[pretrain]", f"gene embedding initialized from PCA controls (shape={W_init.shape})")
            return W_init
        except Exception as e:
            logger("[warn]", f"PCA init failed → random init. Reason: {e}")
            return None
    else:
        logger("[pretrain]", "using random gene embedding (PCA disabled).")
        return None

def encode_context_cols(obs_df: pd.DataFrame,
                        encoders: Dict[str, LabelEncoder],
                        col_batch: str,
                        col_celltype: str,
                        col_is_h1: str):
    def enc(col):
        if (col in obs_df.columns) and (col in encoders):
            return encoders[col].transform(obs_df[col].astype(str).values).astype(np.int64)
        return np.zeros(len(obs_df), dtype=np.int64)
    
    batch_id = enc(col_batch)
    ct_id    = enc(col_celltype)
    is_h1    = (obs_df[col_is_h1].astype(int).values
                if col_is_h1 in obs_df.columns else np.zeros(len(obs_df), dtype=np.int64))
    return batch_id, ct_id, is_h1

def batched(iterable, batch_size):
    for i in range(0, len(iterable), batch_size):
        yield slice(i, min(i + batch_size, len(iterable)))


# ---------- cli helpers ----------
def extend_embedding(old_emb: torch.nn.Embedding, new_num_embeddings: int, device: torch.device) -> torch.nn.Embedding:
    """Resize embedding, preserving existing weights and randomly initializing new rows."""
    if new_num_embeddings <= old_emb.num_embeddings:
        return old_emb
    new_emb = torch.nn.Embedding(new_num_embeddings, old_emb.embedding_dim).to(device)
    with torch.no_grad():
        new_emb.weight[:old_emb.num_embeddings].copy_(old_emb.weight)
        torch.nn.init.normal_(new_emb.weight[old_emb.num_embeddings:], mean=0.0, std=0.02)
        if old_emb.padding_idx is not None:
            new_emb.weight[old_emb.padding_idx].zero_()
    return new_emb


def fit_label_encoder_from_union(existing_le: LabelEncoder | None, new_series: pd.Series) -> LabelEncoder:
    """Merge existing label encoder with new categories."""
    le = LabelEncoder()
    new_classes = np.array(sorted(map(str, pd.unique(new_series.astype(str)))), dtype=str)
    if existing_le is None:
        le.classes_ = new_classes
    else:
        union = np.array(sorted(set(existing_le.classes_.astype(str)).union(set(new_classes))), dtype=str)
        le.classes_ = union
    return le


def resolve_artifacts_root(cfg_all, ckpt_path: str) -> str:
    """Locate directory containing gene2id.json (training artifacts)."""
    cands = [
        cfg_all.get("predict", {}).get("artifacts_outdir"),
        cfg_all.get("finetune", {}).get("outdir"),
        cfg_all.get("finetune", {}).get("pretrained_outdir"),
        cfg_all.get("train", {}).get("outdir"),
        os.path.dirname(os.path.abspath(ckpt_path)),
    ]
    for p in cands:
        if p and os.path.isfile(os.path.join(p, "gene2id.json")):
            return p
    raise FileNotFoundError("Could not locate training artifacts (gene2id.json). " "Checked: " + ", ".join([str(p) for p in cands if p]))


def dense_log_features_from_adata(adata, pred_cfg: PredictConfig) -> np.ndarray:
        """
        Prepare dense log1p-normalized matrix from AnnData:
        - If already log-transformed, pass through.
        - Else: optionally normalize_total(target_sum) + log1p.
        """
        X = adata.X
        if detect_logged_data(X, check_n=200):
            logger("[prediction]", "detected log-like input → using as-is")
            X_log = X.toarray() if sparse.issparse(X) else np.asarray(X)
            return X_log.astype(np.float32, copy=False)
        normalize  = pred_cfg.normalize
        target_sum = pred_cfg.target_sum
        _, X_log = prepare_normalized_views(adata, normalize=normalize, target_sum=target_sum)
        return (X_log.toarray() if sparse.issparse(X_log) else np.asarray(X_log)).astype(np.float32, copy=False)


# --- Adaptive neighbor-mix helpers ------------------------------------------
def seen_target(tg: str, gene2id: dict, control_token: str) -> bool:
    gid = gene2id.get(tg, 0)
    return (tg != control_token) and (gid != 0)


def mix_params_for_target(
    tg: str,
    gene2id: dict,
    control_token: str,
    default_k: int,
    default_tau: float,
    seen_k: int = 0,
    seen_tau: float | None = None
):
    """Return (k, tau) mix parameters based on whether the target was seen during training."""
    if seen_target(tg, gene2id, control_token):
        return int(seen_k), float(default_tau if seen_tau is None else seen_tau)
    else:
        return int(default_k), float(default_tau)

def _load_gene_list(path: str) -> Sequence[str]:
    p = str(path).lower()
    if p.endswith(".npy"):
        arr = np.load(path, allow_pickle=True)
        return [str(x) for x in arr.tolist()]
    try:
        df = pd.read_csv(path, sep=None, engine="python", header=None)
        col = df.iloc[:, 0].dropna().astype(str).str.strip()
        return col[col != ""].tolist()
    except Exception:
        with open(path, "r") as f:
            return [ln.strip() for ln in f if ln.strip()]

def reorder_var_to_gene_file(adata_in: ad.AnnData, gene_file: str) -> Tuple[ad.AnnData, dict]:
    target_genes = pd.Index([str(g) for g in _load_gene_list(gene_file)], dtype=str)
    cur = pd.Index(adata_in.var_names.astype(str))
    pos = cur.get_indexer(target_genes)  # -1 if missing
    have = pos >= 0

    if sparse.issparse(adata_in.X):
        X = adata_in.X.tocsr()
        parts = []
        if have.any():
            parts.append(X[:, pos[have]])
        if (~have).any():
            parts.append(sparse.csr_matrix((X.shape[0], (~have).sum()), dtype=X.dtype))
        X_out = sparse.hstack(parts, format="csr")
    else:
        X = np.asarray(adata_in.X)
        X_out = np.zeros((adata_in.n_obs, len(target_genes)), dtype=X.dtype)
        if have.any():
            X_out[:, have] = X[:, pos[have]]

    adata_out = ad.AnnData(X_out, obs=adata_in.obs.copy(), var=pd.DataFrame(index=target_genes))
    adata_out.obs_names = adata_in.obs_names.copy()
    adata_out.var_names_make_unique()

    info = {
        "n_target": int(len(target_genes)),
        "n_matched": int(have.sum()),
        "n_missing": int((~have).sum()),
        "missing_sample": target_genes[~have][:10].tolist()
    }
    return adata_out, info
# -----------------------------------------------------------------------------