import argparse
import os, sys
import json
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
from scipy import sparse

from .data import collate_batch, make_dataloader, PairedPerturbBatchSampler
from .config import load_yaml_cfg, DataConfig, ModelConfig, TrainConfig, PredictConfig
from .utils import *
from .vae import DualEncoderVAE
from .train import train_loop

# ---------- TRAIN ----------
def main_train(args):
    logger("[main-train]", f"loading the config from {args.config}...")
    cfg_all   = load_yaml_cfg(args.config)
    data_cfg  = DataConfig(**cfg_all["data"])
    model_cfg = ModelConfig(**cfg_all["model"])
    train_cfg = TrainConfig(**cfg_all["train"])

    set_all_seeds(train_cfg.seed)
    device = get_device()
    os.makedirs(train_cfg.outdir, exist_ok=True)

    logger("[main-train]", f"loading AnnData from {data_cfg.train_h5ad}...")
    adata_train = sc.read_h5ad(data_cfg.train_h5ad)

    # --- Vocabulary & encoders ---
    logger("[main-train]", "building the vocabulary...")
    gene2id = build_gene_vocab(adata_train, control_token=data_cfg.control_token, col_target=data_cfg.col_target)
    model_cfg.num_genes_vocab = len(gene2id)
    model_cfg.input_dim = adata_train.n_vars
    if model_cfg.hidden_dims is None:
        model_cfg.hidden_dims = [1024, 512, 256]

    encoders = {}
    for col in [data_cfg.col_batch, data_cfg.col_celltype]:
        if col in adata_train.obs.columns:
            encoders[col] = LabelEncoder().fit(adata_train.obs[col].astype(str).values)

    # --- Dataloaders ---
    logger("[main-train]", "creating the dataloaders...")
    ds_train, _ = make_dataloader(
        data_cfg.train_h5ad, data_cfg, gene2id, encoders,
        train_cfg.batch_size, shuffle=True, zscore=data_cfg.zscore, zstats=None
    )

    if model_cfg.use_count_head and ds_train.is_logged_input:
        logger("[warn]", "count head enabled but data appears log-transformed → will be auto-disabled in training loop")

    sampler = PairedPerturbBatchSampler(dataset=ds_train,
                                        batch_size=train_cfg.batch_size,
                                        min_pos_per_target=8,
                                        match_batch=True,
                                        match_h1=True,
                                        ctrl_per_pos=1,
                                        shuffle=True,
                                        drop_last=True,
                                        seed=train_cfg.seed,
                                        prioritize_batch=True
                                        )
    
    dl_train = DataLoader(ds_train, batch_sampler=sampler, num_workers=0,
                          collate_fn=collate_batch, pin_memory=(device.type == "cuda"))

    val_loader = None
    if data_cfg.val_h5ad and os.path.exists(data_cfg.val_h5ad):
        _, val_loader = make_dataloader(
            data_cfg.val_h5ad, data_cfg, gene2id, encoders,
            train_cfg.batch_size, shuffle=False, zscore=data_cfg.zscore, zstats=None
        )

    # --- Build model ---
    logger("[main-train]", "building the model...")
    n_batches = len(encoders.get(data_cfg.col_batch, LabelEncoder()).classes_) if data_cfg.col_batch in encoders else 1
    n_ct      = len(encoders.get(data_cfg.col_celltype, LabelEncoder()).classes_) if data_cfg.col_celltype in encoders else 1
    var_names = list(map(str, adata_train.var_names))

    resuming = train_cfg.resume_from and os.path.exists(train_cfg.resume_from)
    W_init = None
    if not resuming:
        logger("[main-train]", "preparing gene embedding initialization...")
        W_init = prepare_gene_embedding(
            gene2id=gene2id,
            var_names=var_names,
            embed_dim=model_cfg.gene_embed_dim,
            pretrained_pt_path=model_cfg.pretrained_gene_emb_path,
            adata_for_pca=adata_train,
            col_target=data_cfg.col_target,
            control_token=data_cfg.control_token,
            norm=model_cfg.pretrained_norm,
            case_insensitive=model_cfg.pretrained_case_insensitive,
            zero_control_row=True,
            random_state=train_cfg.seed,
            use_pca_init=model_cfg.use_pca_init,
            max_cells=model_cfg.init_max_cells,
        )

    model = DualEncoderVAE(model_cfg, n_batches, n_ct, model_cfg.num_genes_vocab, gene_emb_init=W_init).to(device)
    logger("[main-train]", f"model size: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters")

    # --- Resume (optional) ---
    if resuming:
        logger("[resume]", f"loading model weights from {train_cfg.resume_from}...")
        ckpt = torch.load(train_cfg.resume_from, map_location=device)
        state = ckpt.get("model_state", ckpt)
        model.load_state_dict(state, strict=False)
        logger("[resume]", "model resumed successfully")
    
    # --- Dataset preprocessing summary ---
    print("\n=== DATA PREPROCESSING SUMMARY ===")
    print(f" - Log-scale input detected:  {ds_train.is_logged_input}")
    print(f" - Normalized during load:    {ds_train.used_normalization}")
    print(f" - Raw count supervision:     {ds_train.has_counts_targets}")
    print(f" - Z-score normalization:     {data_cfg.zscore}")
    print(f" - Target sum normalization:  {data_cfg.target_sum}")
    print("=================================\n")

    logger("[main-train]", "starting the procedure...")
    train_loop(model, dl_train, val_loader, train_cfg, device, train_cfg.outdir)
    logger("[main-train]", "procedure completed!")
    
    # --- Post-training quick diagnostic on ranges ---
    try:
        model.eval()
        with torch.no_grad():
            # fetch one small batch directly from dataset to avoid sampler coupling
            idx = np.random.choice(len(ds_train), size=min(64, len(ds_train)), replace=False)
            samples = [ds_train[i] for i in idx]
            b = collate_batch(samples)
            for k in b:
                if isinstance(b[k], torch.Tensor):
                    b[k] = b[k].to(device)
            out = model(b)

            x_in = b["x"].float().detach().cpu().numpy()
            x_in_min, x_in_max = float(np.min(x_in)), float(np.max(x_in))

            # try common output keys (model returns x_rec_c and x_rec_p)
            x_keys = ["x_rec_c", "x_rec_p", "x_hat_log1p", "x_rec", "x_hat"]
            x_out = None
            for k in x_keys:
                if isinstance(out, dict) and (k in out):
                    x_out = out[k]
                    break
            if isinstance(x_out, torch.Tensor):
                x_out = x_out.detach().cpu().numpy()
                
            if x_out is not None:
                x_out_min, x_out_max = float(np.min(x_out)), float(np.max(x_out))
                in_band = (x_out_min >= -0.1) and (x_out_max <= 12.0)
                print("\n=== POST-TRAIN DIAGNOSTIC ===")
                print(f" - Input(log1p) range:   [{x_in_min:.3f}, {x_in_max:.3f}]")
                print(f" - Recon(log1p) range:   [{x_out_min:.3f}, {x_out_max:.3f}]")
                print(f" - Recon within [0,12]?:  {in_band}")
                print("   (If out of band, check decoder bounds or data scale.)")
                print("================================\n")
            else:
                logger("[diag]", "could not find reconstruction tensor in model output (looked for keys: x_rec_c/x_rec_p/x_hat_log1p/x_rec/x_hat). Skipping range check.\n")
    except Exception as e:
        logger("[diag]", f"post-training diagnostic failed gracefully: {e}\n")

    # --- Save artifacts ---
    np.save(os.path.join(train_cfg.outdir, "var_names.npy"), np.array(adata_train.var_names.astype(str)))
    with open(os.path.join(train_cfg.outdir, "gene2id.json"), "w") as f:
        json.dump(gene2id, f)
    for col, le in encoders.items():
        np.save(os.path.join(train_cfg.outdir, f"{col}_classes.npy"), le.classes_)
    logger("[main-train]", f"saved vocab/encoders to {train_cfg.outdir}")

# ---------- FINETUNE ----------
def main_finetune(args):
    """
    Fine-tune from a pretrained checkpoint on a new dataset.
    - Loads encoders from pretraining outdir and merges with new classes (union).
    - Expands batch/celltype embeddings to fit union sizes (via extend_embedding).
    - Uses the same training loop & sampler.
    """
    logger("[main-finetune]", f"loading config from {args.config}...")
    cfg_all   = load_yaml_cfg(args.config)
    data_cfg  = DataConfig(**cfg_all["data"])
    model_cfg = ModelConfig(**cfg_all["model"])
    train_cfg = TrainConfig(**cfg_all["finetune"])

    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")

    set_all_seeds(train_cfg.seed)
    device = get_device()
    os.makedirs(train_cfg.outdir, exist_ok=True)

    # Dataset
    ft_path = data_cfg.fewshot_h1_h5ad or data_cfg.train_h5ad
    logger("[main-finetune]", f"loading AnnData from {ft_path}...")    
    adata_ft = sc.read_h5ad(ft_path)

    # ---- determine where to load pretraining artifacts from
    # 1) explicit override in config: finetune.pretrained_outdir
    pre_outdir = train_cfg.pretrained_outdir
    # 2) else infer from the checkpoint path
    if not pre_outdir:
        pre_outdir = os.path.dirname(os.path.abspath(args.ckpt))
    # 3) final fallback (not recommended): train.outdir in this config
    if (not pre_outdir) or (not os.path.isdir(pre_outdir)):
        cand = cfg_all.get("train", {}).get("outdir")
        if cand and os.path.isdir(cand):
            pre_outdir = cand

    if not os.path.isfile(os.path.join(pre_outdir, "gene2id.json")):
        raise FileNotFoundError(
            f"gene2id.json not found in '{pre_outdir}'. "
            "Set finetune.pretrained_outdir in your YAML or put gene2id.json next to the checkpoint."
        )

    with open(os.path.join(pre_outdir, "gene2id.json"), "r") as f:
        gene2id = json.load(f)
    model_cfg.num_genes_vocab = max(gene2id.values()) + 1

    # Build encoders: union of pretrained encoders and current FT data
    enc_train = load_label_encoders(pre_outdir, [data_cfg.col_batch, data_cfg.col_celltype])
    encoders = {}
    if data_cfg.col_batch in adata_ft.obs.columns:
        encoders[data_cfg.col_batch] = fit_label_encoder_from_union(enc_train.get(data_cfg.col_batch), adata_ft.obs[data_cfg.col_batch])
    if data_cfg.col_celltype in adata_ft.obs.columns:
        encoders[data_cfg.col_celltype] = fit_label_encoder_from_union(enc_train.get(data_cfg.col_celltype), adata_ft.obs[data_cfg.col_celltype])

    # Loaders
    ds_train, _ = make_dataloader(ft_path, data_cfg, gene2id, encoders,
                                  train_cfg.batch_size, shuffle=True, zscore=data_cfg.zscore, zstats=None)
    
    sampler = PairedPerturbBatchSampler(dataset=ds_train,
                                        batch_size=train_cfg.batch_size,
                                        min_pos_per_target=8,
                                        match_batch=True,
                                        match_h1=True,
                                        ctrl_per_pos=1,
                                        shuffle=True,
                                        drop_last=True,
                                        seed=train_cfg.seed,
                                        prioritize_batch=True)

    dl_train = DataLoader(ds_train, batch_sampler=sampler, num_workers=0,
                          collate_fn=collate_batch, pin_memory=(device.type == "cuda"))

    # Build model and load checkpoint (shape-safe)
    logger("[main-finetune]", f"loading the checkpoint from {args.ckpt}...")    
    ckpt = torch.load(args.ckpt, map_location=device)
    ckpt_state = ckpt.get("model_state", ckpt)
    ckpt_mcfg = ckpt.get("cfg_model", {})

    # Keep critical shape params consistent with checkpoint where provided
    for k in ["input_dim", "latent_dim", "hidden_dims", "context_dim", "gene_embed_dim",
              "delta_rank", "num_genes_vocab"]:
        if k in ckpt_mcfg:
            setattr(model_cfg, k, ckpt_mcfg[k])

    # Helper to read embedding sizes from ckpt tensors if present
    def _rows(name, fallback):
        w = ckpt_state.get(name + ".weight", None)
        return int(w.shape[0]) if (w is not None) else int(fallback)

    ckpt_batch_rows = _rows("batch_emb", 1)
    ckpt_ct_rows    = _rows("ct_emb",    1)
    ckpt_gene_rows  = _rows("gene_emb",  model_cfg.num_genes_vocab)

    # Union sizes from current FT data
    n_batches_union = len(encoders.get(data_cfg.col_batch, LabelEncoder()).classes_) if data_cfg.col_batch in encoders else 1
    n_ct_union      = len(encoders.get(data_cfg.col_celltype, LabelEncoder()).classes_) if data_cfg.col_celltype in encoders else 1

    # Instantiate with at least ckpt sizes
    n_batches_init = max(ckpt_batch_rows, n_batches_union)
    n_ct_init      = max(ckpt_ct_rows,    n_ct_union)
    n_gene_init    = ckpt_gene_rows  # use ckpt vocab exactly

    model_cfg.num_genes_vocab = n_gene_init
    model = DualEncoderVAE(model_cfg, n_batches_init, n_ct_init, n_gene_init).to(device)

    # Load weights (allow missing/extra)
    missing, unexpected = model.load_state_dict(ckpt_state, strict=False)
    if missing or unexpected:
        logger("[main-finetune]", f"state_dict loaded with missing={len(missing)}, unexpected={len(unexpected)}")

    # If union > ckpt, extend embeddings AFTER loading so new classes can be learned
    with torch.no_grad():
        if n_batches_union > ckpt_batch_rows:
            model.batch_emb = extend_embedding(model.batch_emb, n_batches_union, device)
            logger("[main-finetune]", f"batch_emb expanded to {n_batches_union} (from ckpt {ckpt_batch_rows})")
        if n_ct_union > ckpt_ct_rows:
            model.ct_emb = extend_embedding(model.ct_emb, n_ct_union, device)
            logger("[main-finetune]", f"ct_emb expanded to {n_ct_union} (from ckpt {ckpt_ct_rows})")

        # Count head compatibility: initialize if YAML enables it but ckpt didn't have it
        if model.cfg.use_count_head:
            has_w = ("dec.count_w" in ckpt_state)
            has_b = ("dec.count_b" in ckpt_state)
            if not (has_w and has_b):
                if hasattr(model.dec, "reset_count_head"):
                    model.dec.reset_count_head()
                else:
                    model.dec.count_w.zero_()
                    model.dec.count_b.zero_()
                logger("[main-finetune]", "initialized count head (dec.count_w/dec.count_b)")
    
    # --- Dataset preprocessing summary ---
    print("\n=== DATA PREPROCESSING SUMMARY ===")
    print(f" - Log-scale input detected:  {ds_train.is_logged_input}")
    print(f" - Normalized during load:    {ds_train.used_normalization}")
    print(f" - Raw count supervision:     {ds_train.has_counts_targets}")
    print(f" - Z-score normalization:     {data_cfg.zscore}")
    print(f" - Target sum normalization:  {data_cfg.target_sum}")
    print("=================================\n")

    logger("[main-finetune]", "starting the procedure...")
    train_loop(model, dl_train, None, train_cfg, device, train_cfg.outdir)
    logger("[main-finetune]", "procedure completed!")
    
    # --- Post-finetuning quick diagnostic on ranges ---
    try:
        model.eval()
        with torch.no_grad():
            # fetch one small batch directly from dataset to avoid sampler coupling
            idx = np.random.choice(len(ds_train), size=min(64, len(ds_train)), replace=False)
            samples = [ds_train[i] for i in idx]
            b = collate_batch(samples)
            for k in b:
                if isinstance(b[k], torch.Tensor):
                    b[k] = b[k].to(device)
            out = model(b)

            x_in = b["x"].float().detach().cpu().numpy()
            x_in_min, x_in_max = float(np.min(x_in)), float(np.max(x_in))

            # try common output keys (model returns x_rec_c and x_rec_p)
            x_keys = ["x_rec_c", "x_rec_p", "x_hat_log1p", "x_rec", "x_hat"]
            x_out = None
            for k in x_keys:
                if isinstance(out, dict) and (k in out):
                    x_out = out[k]
                    break
            if isinstance(x_out, torch.Tensor):
                x_out = x_out.detach().cpu().numpy()
                
            if x_out is not None:
                x_out_min, x_out_max = float(np.min(x_out)), float(np.max(x_out))
                in_band = (x_out_min >= -0.1) and (x_out_max <= 12.0)
                print("\n=== POST-FINETUNING DIAGNOSTIC ===")
                print(f" - Input(log1p) range:   [{x_in_min:.3f}, {x_in_max:.3f}]")
                print(f" - Recon(log1p) range:   [{x_out_min:.3f}, {x_out_max:.3f}]")
                print(f" - Recon within [0,12]?:  {in_band}")
                print("   (If out of band, check decoder bounds or data scale.)")
                print("================================\n")
            else:
                logger("[diag]", "could not find reconstruction tensor in model output (looked for keys: x_rec_c/x_rec_p/x_hat_log1p/x_rec/x_hat). Skipping range check.\n")
    except Exception as e:
        logger("[diag]", f"post-finetuning diagnostic failed gracefully: {e}\n")

    # --- Save artifacts ---
    np.save(os.path.join(train_cfg.outdir, "var_names.npy"), np.array(adata_ft.var_names.astype(str)))
    with open(os.path.join(train_cfg.outdir, "gene2id.json"), "w") as f:
        json.dump(gene2id, f)
    for col, le in encoders.items():
        np.save(os.path.join(train_cfg.outdir, f"{col}_classes.npy"), le.classes_)
    logger("[main-finetune]", f"saved vocab/encoders to {train_cfg.outdir}")
    
# ---------- PREDICT ----------
@torch.no_grad()
def main_predict(args):
    """
    Generate synthetic cells per PredictConfig:
      - per-target depth matching using 'median_umi_per_cell' if present (batch-aware fallback)
      - strict training gene order in output
      - optional: include real controls (n_control_cells: -1|None => all)
      - generate synthetic controls if 'non-targeting' appears in CSV
      - export in 'counts' (with per-cell median_umi_per_cell) or 'log1p' (drop the depth column)
      - neighbor-mixed Δ for unseen genes (neighbor_mix_k / neighbor_mix_tau)
    """

    # -------------------------------------------------------------------------
    # Load config and artifacts
    # -------------------------------------------------------------------------
    logger("[prediction]", f"loading the config from {args.config}...")
    cfg_all   = load_yaml_cfg(args.config)
    data_cfg  = DataConfig(**cfg_all["data"])
    model_cfg = ModelConfig(**cfg_all["model"])
    pred_cfg  = PredictConfig(**cfg_all["predict"])

    device = get_device()
    n_z = max(1, pred_cfg.n_infer_samples)

    # Optional per-gene multiplicative calibration in log1p space (not common)
    calib_alpha = None
    if pred_cfg.apply_affine_calibration:
        path = pred_cfg.calib_alpha_npy
        if path and os.path.exists(path):
            calib_alpha = np.load(path).astype(np.float32)
            logger("[prediction]", f"loaded per-gene calibration alpha from {path}")
        else:
            logger("[warn]", "apply_affine_calibration=True but calib_alpha_npy missing/not found; skipping")

    # -------------------------------------------------------------------------
    # Load training artifacts
    # -------------------------------------------------------------------------
    outdir = resolve_artifacts_root(cfg_all, args.ckpt)
    with open(os.path.join(outdir, "gene2id.json"), "r") as f:
        gene2id = json.load(f)
    model_cfg.num_genes_vocab = max(gene2id.values()) + 1
    logger("[prediction]", f"loaded gene2id (V={model_cfg.num_genes_vocab}) from {outdir}")

    # Strict training gene order (if saved)
    train_genes = None
    p_txt = os.path.join(outdir, "var_names.txt")
    p_npy = os.path.join(outdir, "var_names.npy")
    if os.path.exists(p_npy):
        train_genes = list(np.load(p_npy, allow_pickle=True))
    elif os.path.exists(p_txt):
        with open(p_txt, "r") as f:
            train_genes = [ln.strip() for ln in f if ln.strip()]
    if train_genes is None:
        logger("[warn]", "training var_names.* not found; will use control pool .var order")
    else:
        logger("[prediction]", f"loaded training gene order: {len(train_genes)} genes")

    encoders = load_label_encoders(outdir, [data_cfg.col_batch, data_cfg.col_celltype])

    # -------------------------------------------------------------------------
    # Load control pool (H1 controls)
    # -------------------------------------------------------------------------
    logger("[prediction]", f"loading control pool from {pred_cfg.control_pool_h5ad}...")
    adata_pool = sc.read_h5ad(pred_cfg.control_pool_h5ad)
    obs_all = adata_pool.obs.copy()
    
    model_cfg.input_dim = adata_pool.n_vars

    mask_h1   = (obs_all[data_cfg.col_is_h1].astype(int) == pred_cfg.h1_flag_value) if data_cfg.col_is_h1 in obs_all.columns else np.ones(len(obs_all), dtype=bool)
    mask_ctrl = (obs_all[data_cfg.col_target].astype(str) == data_cfg.control_token)
    sel_series = pd.Series(mask_h1 & mask_ctrl)
    if sel_series.sum() == 0:
        raise RuntimeError("No H1 controls found for given flags/columns")
    logger("[prediction]", f"H1 control cells available: {int(sel_series.sum())}")

    # Prepare dense log1p features (model input) for pool
    X_pool_all = dense_log_features_from_adata(adata_pool, pred_cfg)  # float log1p
    libsize_log1p_all = np.log1p(np.expm1(X_pool_all).sum(axis=1)).astype(np.float32)
    obs_all["libsize_log1p"] = libsize_log1p_all

    # Convert selection to indices (sparse-safe)
    sel_np  = sel_series.to_numpy(dtype=bool)
    row_idx = np.nonzero(sel_np)[0]

    # Filter log features / obs / libsize to controls
    X_pool = X_pool_all[row_idx]
    obs    = obs_all.loc[sel_np].reset_index(drop=True)
    libsize_log1p = libsize_log1p_all[row_idx]
    pool_size = X_pool.shape[0]
    logger("[prediction]", f"filtered pool: X={X_pool.shape}, obs={obs.shape}")

    # RAW-counts pool for depth stats (controls-only)
    if sparse.issparse(adata_pool.X):
        X_pool_counts = adata_pool.X[row_idx]
        pool_depths = np.asarray(X_pool_counts.sum(axis=1)).ravel()
    else:
        X_pool_counts = adata_pool.X[row_idx]
        pool_depths = X_pool_counts.sum(axis=1).astype(np.float64)
    global_median_depth = float(np.median(pool_depths)) if pool_depths.size else 0.0
    if not (np.isfinite(global_median_depth) and global_median_depth > 0):
        raise RuntimeError("Global control median depth is invalid; check that control pool contains raw counts.")
    logger("[prediction]", f"global control median depth (counts): {global_median_depth:.1f}")

    # Optional: per-batch medians (post-filter batches)
    depth_by_batch = {}
    if data_cfg.col_batch in obs.columns:
        for b, idxs in obs.groupby(data_cfg.col_batch, observed=False).groups.items():
            row_idx_b = row_idx[np.asarray(list(idxs), dtype=int)]
            if sparse.issparse(adata_pool.X):
                depths_b = np.asarray(adata_pool.X[row_idx_b].sum(axis=1)).ravel()
            else:
                depths_b = adata_pool.X[row_idx_b].sum(axis=1).astype(np.float64)
            if depths_b.size > 0 and np.isfinite(depths_b).all():
                depth_by_batch[b] = float(np.median(depths_b))
        if depth_by_batch:
            logger("[prediction]", f"per-batch depth medians cached for {len(depth_by_batch)} batches")

    # -------------------------------------------------------------------------
    # Load model
    # -------------------------------------------------------------------------
    logger("[prediction]", f"loading the model from checkpoint {args.ckpt}...")
    ckpt = torch.load(args.ckpt, map_location=device)
    if model_cfg.hidden_dims is None:
        model_cfg.hidden_dims = [1024, 512, 256]

    model = DualEncoderVAE(
        model_cfg,
        num_batches=len(encoders.get(data_cfg.col_batch, LabelEncoder()).classes_) if data_cfg.col_batch in encoders else 1,
        num_celltypes=len(encoders.get(data_cfg.col_celltype, LabelEncoder()).classes_) if data_cfg.col_celltype in encoders else 1,
        num_genes_vocab=model_cfg.num_genes_vocab,
    ).to(device)
    model.load_state_dict(ckpt.get("model_state", ckpt), strict=False)
    model.eval()
    logger("[prediction]", "model loaded and ready for inference")

    # -------------------------------------------------------------------------
    # Enforce training gene order
    # -------------------------------------------------------------------------
    pool_genes = list(map(str, adata_pool.var_names))
    if train_genes is not None:
        col_map = {g: i for i, g in enumerate(pool_genes)}
        expected_G = len(train_genes)
        X_pool_reordered = np.zeros((X_pool.shape[0], expected_G), dtype=np.float32)
        matched = 0
        missing_genes = []
        for out_idx, gene in enumerate(train_genes):
            pool_idx = col_map.get(gene)
            if pool_idx is not None:
                X_pool_reordered[:, out_idx] = X_pool[:, pool_idx]
                matched += 1
            else:
                missing_genes.append(gene)
        X_pool = X_pool_reordered
        out_var_index = pd.Index(train_genes, dtype=str)
        logger("[prediction]", f"gene order enforced: {matched}/{expected_G} genes present in pool")
        if missing_genes:
            logger("[warn]", f"{len(missing_genes)} training genes missing from pool (will be zeros): {missing_genes[:5]}...")
    else:
        out_var_index = pd.Index(pool_genes, dtype=str)
        expected_G = len(pool_genes)
        logger("[warn]", "using control pool gene order (training order not found)")
    logger("[prediction]", f"output gene dimension: {expected_G}")

    # Build column indexer to reorder RAW COUNT matrices to training order
    pool_var = pd.Index(map(str, adata_pool.var_names))
    col_sel = pool_var.get_indexer(out_var_index)   # -1 if missing
    have_all = np.all(col_sel >= 0)

    # -------------------------------------------------------------------------
    # Encode control contexts
    # -------------------------------------------------------------------------
    batch_id, ct_id, is_h1 = encode_context_cols(
        obs, encoders, data_cfg.col_batch, data_cfg.col_celltype, data_cfg.col_is_h1
    )
    logger("[prediction]", "encoding control latent representations (z_c, u)...")
    zc_bank, u_bank = [], []
    bs = 1024
    for sl in batched(range(pool_size), bs):
        x_b = torch.from_numpy(X_pool[sl]).to(device).float()
        b_b = torch.from_numpy(batch_id[sl]).to(device)
        c_b = torch.from_numpy(ct_id[sl]).to(device)
        h_b = torch.from_numpy(is_h1[sl]).to(device)
        l_b = torch.from_numpy(libsize_log1p[sl]).to(device).float().unsqueeze(1)

        if n_z > 1:
            x_rep = x_b.repeat_interleave(n_z, dim=0)
            b_rep = b_b.repeat_interleave(n_z, dim=0)
            c_rep = c_b.repeat_interleave(n_z, dim=0)
            h_rep = h_b.repeat_interleave(n_z, dim=0)
            l_rep = l_b.repeat_interleave(n_z, dim=0)
            zc, u = model.encode_controls(x_rep, b_rep, c_rep, h_rep, libsize=l_rep, sample=True)
        else:
            zc, u = model.encode_controls(x_b, b_b, c_b, h_b, libsize=l_b, sample=True)
        zc_bank.append(zc); u_bank.append(u)

    zc_bank = torch.cat(zc_bank, dim=0)  # shape ~ (pool_size * n_z, Dz)
    u_bank  = torch.cat(u_bank,  dim=0)  # shape ~ (pool_size * n_z, Du)
    pool_size_eff = zc_bank.shape[0]
    logger("[prediction]", f"control bank encoded: zc={zc_bank.shape}, u={u_bank.shape}")

    # -------------------------------------------------------------------------
    # Targets to synthesize
    # -------------------------------------------------------------------------
    df_list = pd.read_csv(pred_cfg.perturb_list_csv)
    if "target_gene" not in df_list.columns or "n_cells" not in df_list.columns:
        raise ValueError("perturb_list_csv must have columns: target_gene, n_cells")
    logger("[prediction]", f"loaded {len(df_list)} targets to generate from {pred_cfg.perturb_list_csv}")

    # Defaults from YAML (used for unseen targets)
    default_k   = pred_cfg.neighbor_mix_k
    default_tau = pred_cfg.neighbor_mix_tau

    # Depth helpers
    depth_col = pred_cfg.depth_column
    has_depth = depth_col in df_list.columns
    df_has_batch = (data_cfg.col_batch in df_list.columns)
    
    def _resolve_target_depth(row):
        if has_depth:
            v = row.get(depth_col, np.nan)
            if pd.notna(v) and np.isfinite(v) and float(v) > 0:
                return float(v)
        if df_has_batch:
            b = row.get(data_cfg.col_batch, None)
            if b in depth_by_batch and np.isfinite(depth_by_batch[b]) and depth_by_batch[b] > 0:
                return float(depth_by_batch[b])
        return global_median_depth

    X_blocks, obs_blocks = [], []

    # -------------------------------------------------------------------------
    # Optional: include real controls (with counts/log1p parity)
    # -------------------------------------------------------------------------
    output_scale      = pred_cfg.output_scale.lower()
    n_ctrl            = pred_cfg.n_control_cells
    delta_gain        = pred_cfg.delta_gain
    rate_sharpen_beta = pred_cfg.rate_sharpen_beta
    mix_sharpen_p     = pred_cfg.mix_sharpen_p
    topk_keep_only    = pred_cfg.topk_keep_only
    prune_quantile    = pred_cfg.prune_quantile
    topk_boost_k      = pred_cfg.topk_boost_k
    topk_boost_gamma  = pred_cfg.topk_boost_gamma
    max_rate          = pred_cfg.count_max_rate
    
    if n_ctrl is None or n_ctrl == -1:
        include_real_ctrl = True
        n_ctrl = pool_size
    elif n_ctrl > 0:
        include_real_ctrl = True
    else:
        include_real_ctrl = False

    if include_real_ctrl:
        if n_ctrl > pool_size:
            logger("[warn]", f"requested {n_ctrl} controls but only {pool_size} available; sampling with replacement")
        idx_ctrl = np.random.choice(pool_size, size=n_ctrl, replace=(n_ctrl > pool_size))
        obs_ctrl = obs.iloc[idx_ctrl].copy()
        obs_ctrl[data_cfg.col_target] = data_cfg.control_token
        obs_ctrl["is_synthetic"] = 0

        if output_scale == "counts":
            # RAW counts in training gene order
            Xc_rows = adata_pool.X[row_idx[idx_ctrl]]
            if sparse.issparse(Xc_rows):
                if have_all:
                    X_ctrl = Xc_rows[:, col_sel].tocsr(copy=False)
                else:
                    present = (col_sel >= 0)
                    X_present = Xc_rows[:, col_sel[present]].tocsr()
                    if np.any(~present):
                        n_missing = int((~present).sum())
                        zeros = sparse.csr_matrix((X_present.shape[0], n_missing), dtype=X_present.dtype)
                        parts, pi = [], 0
                        for j in range(len(out_var_index)):
                            if present[j]:
                                parts.append(X_present[:, pi]); pi += 1
                            else:
                                parts.append(zeros[:, 0])
                        X_ctrl = sparse.hstack(parts, format="csr", dtype=X_present.dtype)
            else:
                Xc_dense = np.asarray(Xc_rows)
                X_ctrl = np.zeros((Xc_dense.shape[0], len(out_var_index)), dtype=np.int32)
                hit = (col_sel >= 0)
                X_ctrl[:, hit] = Xc_dense[:, col_sel[hit]].astype(np.int32, copy=False)

            obs_ctrl["median_umi_per_cell"] = float(global_median_depth)
            X_blocks.append(X_ctrl); obs_blocks.append(obs_ctrl)
            logger("[prediction]", f"included {n_ctrl} real control cells (raw counts)")
        else:
            # LOG1P path
            X_ctrl = X_pool[idx_ctrl]
            X_blocks.append(X_ctrl)
            obs_blocks.append(obs_ctrl)
            logger("[prediction]", f"included {n_ctrl} real control cells (logs)")

    # -------------------------------------------------------------------------
    # Generate per-target
    # -------------------------------------------------------------------------
    if output_scale == "counts":
        nb_theta = pred_cfg.nb_theta
        link     = pred_cfg.count_link.lower()
    else:
        target_sum = pred_cfg.target_sum

    bs_gen = 1024
    for _, row in df_list.iterrows():
        tg = str(row["target_gene"])
        n_cells = int(row["n_cells"])
        target_depth = _resolve_target_depth(row)
        if not (500.0 <= target_depth <= 200_000.0):
            # logger("[warn]", f"{tg}: unusual target_depth={target_depth:.1f}; using global control median")
            target_depth = global_median_depth

        # adaptive neighbor-mix for this target
        k_tg, tau_tg = mix_params_for_target(
            tg=tg,
            gene2id=gene2id,
            control_token=data_cfg.control_token,
            default_k=default_k,
            default_tau=default_tau,
            seen_k=0,
            seen_tau=None
        )
        
        gid = gene2id.get(tg, 0)
        idx = np.random.choice(pool_size_eff, size=n_cells, replace=(n_cells > pool_size_eff))
        zc_sel = zc_bank[idx]
        u_sel = u_bank[idx]

        # ---------------- Controls / Non-targeting ----------------
        if gid == 0:
            obs_t = obs.iloc[idx % len(obs)].copy()
            obs_t[data_cfg.col_target] = tg
            
            if output_scale == "counts":
                # synthetic controls → sample counts at control-like depth
                outs_cnt = []
                for sl in batched(range(n_cells), bs_gen):
                    X_hat_log1p = model.dec(zc_sel[sl], u_sel[sl])
                    y_b = model.dec.sample_counts(x_hat_log1p=X_hat_log1p,
                                                  target_median_depth=target_depth,
                                                  rate_sharpen_beta=1.0,
                                                  mix_sharpen_p=0.0,
                                                  topk_keep_only=topk_keep_only,
                                                  prune_quantile=prune_quantile,
                                                  topk_boost_k=0,
                                                  topk_boost_gamma=1.0,
                                                  max_rate=max_rate,
                                                  nb_theta=nb_theta,
                                                  link=link,
                                                  allow_fallback_expm1=True,
                                                  )
                    
                    outs_cnt.append(y_b.cpu().numpy())
               
                Xc = np.concatenate(outs_cnt, axis=0).astype(np.int32, copy=False)
                obs_t["is_synthetic"] = 1
                obs_t["median_umi_per_cell"] = float(target_depth)
                X_blocks.append(Xc); obs_blocks.append(obs_t)
                logger("[prediction]", f"{tg}: {n_cells} synthetic control cells (counts)")
            else:
                # log1p path (export log1p)
                outs_log = []
                for sl in batched(range(n_cells), bs_gen):
                    X_hat_log1p = model.dec(zc_sel[sl], u_sel[sl])
                    y_b = model.dec.expected_log1p(x_hat_log1p=X_hat_log1p,
                                                   target_median_depth=target_depth,
                                                   rate_sharpen_beta=rate_sharpen_beta,
                                                   mix_sharpen_p=mix_sharpen_p,
                                                   topk_keep_only=topk_keep_only,
                                                   prune_quantile=prune_quantile,
                                                   topk_boost_k = topk_boost_k,
                                                   topk_boost_gamma = topk_boost_gamma,
                                                   max_rate=max_rate,
                                                   target_sum=target_sum
                                                   )
                    outs_log.append(y_b.cpu().numpy())
                    # X_hat_log1p = np.clip(X_hat_log1p, 0.0, float(model.cfg.output_max))  # data-domain
                    # outs.append(X_hat_log1p)
                Xc = np.concatenate(outs_log, axis=0).astype(np.float32, copy=False)
                obs_t["is_synthetic"] = 1
                X_blocks.append(Xc); obs_blocks.append(obs_t)
                logger("[prediction]", f"{tg}: {n_cells} synthetic control cells (log1p)")
            continue

        # ---------------- Valid perturbation target ----------------
        gid_t  = torch.full((n_cells,), gid, device=device, dtype=torch.long)
        outs = []
        for sl in batched(range(n_cells), bs_gen):
            x_hat = model.predict_perturbation(
                zc_sel[sl], u_sel[sl], gid_t[sl],
                neighbor_mix_k=k_tg,
                neighbor_mix_tau=tau_tg,
                delta_gain=delta_gain
            ).cpu().numpy()
            outs.append(x_hat)
        
        X_hat_log1p = np.concatenate(outs, axis=0)

        # (Decoder forward already bounded; no CLI-side bounding)
        if X_hat_log1p.shape[1] != expected_G:
            raise ValueError(f"Model output {X_hat_log1p.shape[1]} genes != expected {expected_G} genes for target {tg}")

        if calib_alpha is not None:
            if calib_alpha.shape[0] == X_hat_log1p.shape[1]:
                X_hat_log1p *= calib_alpha[None, :]
            else:
                logger("[warn]", f"calib_alpha length {calib_alpha.shape[0]} != genes {X_hat_log1p.shape[1]}; skipping calibration for {tg}")

        if output_scale == "counts":
            outs_b = []
            for sl in batched(range(n_cells), bs_gen):
                xhat_b = torch.from_numpy(X_hat_log1p[sl]).to(device=device, dtype=torch.float32)
                y_b = model.dec.sample_counts(x_hat_log1p=xhat_b,
                                              target_median_depth=target_depth,
                                              rate_sharpen_beta=rate_sharpen_beta,
                                              topk_keep_only=topk_keep_only,
                                              prune_quantile=prune_quantile,
                                              topk_boost_k = topk_boost_k,
                                              topk_boost_gamma = topk_boost_gamma,
                                              mix_sharpen_p=mix_sharpen_p,
                                              max_rate=max_rate,
                                              nb_theta=nb_theta,
                                              link=link,
                                              allow_fallback_expm1=True,
                                              )
                
                outs_b.append(y_b.cpu().numpy())
            X_blocks.append(np.concatenate(outs_b, axis=0))

            obs_t = obs.iloc[idx % len(obs)].copy()
            obs_t[data_cfg.col_target] = tg
            obs_t["is_synthetic"] = 1
            obs_t["median_umi_per_cell"] = target_depth
            obs_blocks.append(obs_t)
            # logger("[prediction]", f"{tg}: {n_cells} cells (k={k_tg}, depth={target_depth})")
        else:
            outs_b = []
            for sl in batched(range(n_cells), bs_gen):
                xhat_b = torch.from_numpy(X_hat_log1p[sl]).to(device=device, dtype=torch.float32)                
                y_b = model.dec.expected_log1p(x_hat_log1p=xhat_b,
                                               target_median_depth=target_depth,
                                               rate_sharpen_beta=rate_sharpen_beta,
                                               topk_keep_only=topk_keep_only,
                                               prune_quantile=prune_quantile,
                                               topk_boost_k = topk_boost_k,
                                               topk_boost_gamma = topk_boost_gamma,
                                               mix_sharpen_p=mix_sharpen_p,
                                               max_rate=max_rate,
                                               target_sum=target_sum,
                                               )
                
                # test = y_b.cpu().numpy()
                # print(f"Pred sparsity: {(test == 0).mean()*100:.1f}%")
                
                outs_b.append(y_b.cpu().numpy())
            X_blocks.append(np.concatenate(outs_b, axis=0))
            
            obs_t = obs.iloc[idx % len(obs)].copy()
            obs_t[data_cfg.col_target] = tg
            obs_t["is_synthetic"] = 1
            obs_blocks.append(obs_t)
            logger("[prediction]", f"{tg}: {n_cells} cells (log1p export, k={k_tg})")

    # -------------------------------------------------------------------------
    # Assemble AnnData with strict var order
    # -------------------------------------------------------------------------
    logger("[prediction]", "assembling output AnnData...")
    any_sparse = any(sparse.issparse(b) for b in X_blocks)
    if any_sparse:
        X_out = sparse.vstack(
            [b.tocsr() if sparse.issparse(b) else sparse.csr_matrix(b) for b in X_blocks],
            format="csr"
        )
    else:
        X_out = np.vstack(X_blocks)

    obs_out = pd.concat(obs_blocks, axis=0).reset_index(drop=True)
    obs_out.index = obs_out.index.astype(str)
    var_out = pd.DataFrame(index=out_var_index)

    # Final safety + per-scale handling
    if output_scale == "counts":
        if sparse.issparse(X_out):
            X_out = X_out.tocsr(copy=False)
            X_out.data = np.rint(X_out.data).clip(min=0).astype(np.int32, copy=False)
            X_out.eliminate_zeros()
        else:
            X_out = np.rint(X_out).clip(min=0).astype(np.int32, copy=False)
    else:
        # Drop median_umi_per_cell for log1p exports
        if "median_umi_per_cell" in obs_out.columns:
            obs_out = obs_out.drop(columns=["median_umi_per_cell"])

    # CRITICAL VALIDATION
    if X_out.shape[1] != len(out_var_index):
        raise ValueError(f"Output matrix has {X_out.shape[1]} genes but expected {len(out_var_index)} genes!")

    generated_targets = set(obs_out[data_cfg.col_target].astype(str).unique())
    expected_targets = set(df_list["target_gene"].astype(str))
    if include_real_ctrl:
        expected_targets |= {data_cfg.control_token}
    missing_targets = expected_targets - generated_targets
    if missing_targets:
        logger("[warn]", f"Missing some expected targets: {missing_targets}")

    adata_out = sc.AnnData(X_out, obs=obs_out, var=var_out)
    adata_out.obs_names_make_unique()
    adata_out.var_names_make_unique()
    
    if pred_cfg.gene_order_file is not None:
        adata_out, info = reorder_var_to_gene_file(adata_out, pred_cfg.gene_order_file)
        logger("[prediction]", f"Reordered to gene file ({info['n_matched']}/{info['n_target']} matched; missing={info['n_missing']})")

    # Diagnostics (optional)
    logger("[prediction]", "validating output...")
    print("\n=== OUTPUT VALIDATION ===")
    print(f" - Total cells generated:     {adata_out.n_obs}")
    print(f" - Total genes:               {adata_out.n_vars}")
    print(f" - Unique targets:            {obs_out[data_cfg.col_target].nunique()}")
    print(f" - Output scale:              {output_scale}")
    print(f" - Sparse format:             {sparse.issparse(adata_out.X)}")
    if output_scale == "counts":
        depths = (np.asarray(adata_out.X.sum(axis=1)).ravel()
                  if sparse.issparse(adata_out.X) else adata_out.X.sum(axis=1))
        nz = (np.asarray(adata_out.X>0).sum() if not sparse.issparse(adata_out.X) else adata_out.X.count_nonzero())
        total = adata_out.n_obs * adata_out.n_vars
        print(f" - Sparsity:                  {100*(1-nz/total):.2f}% zeros")
        print(f" - Count range:               [{adata_out.X.min():.0f}, {adata_out.X.max():.0f}]")
        print(f" - Mean counts per cell:      {float(depths.mean()):.1f}")
        # Quick median by target (head)
        med_by_t = pd.Series(depths, index=obs_out.index).groupby(obs_out[data_cfg.col_target]).median().sort_index()
        print(" - Achieved median depth (first 5 targets):")
        print(med_by_t.head().round(1).to_string())
    else:
        X_dense_preview = (adata_out.X if not sparse.issparse(adata_out.X) else adata_out.X.toarray())
        print(f" - Log1p range:               [{X_dense_preview.min():.3f}, {X_dense_preview.max():.3f}]")
    print("========================\n")

    # Write
    logger("[prediction]", "writing output file...")
    if output_scale == "counts" and not sparse.issparse(adata_out.X):
        adata_out.X = sparse.csr_matrix(adata_out.X)
    comp = pred_cfg.compression
    adata_out.write_h5ad(pred_cfg.out_h5ad, compression=comp)
    logger("[prediction]", f"✓ Successfully wrote {pred_cfg.out_h5ad}")
    logger("[prediction]", f"  Shape: {adata_out.shape} | Scale: {output_scale} | Compression: {comp}")
    
# ---------- ARGPARSE ----------
def build_argparser():
    p = argparse.ArgumentParser(description="Dual-Encoder VAE pipeline (train / finetune / predict)")
    sub = p.add_subparsers(dest="cmd")

    p_train = sub.add_parser("train", help="Train the model")
    p_train.add_argument("--config", required=True)
    p_train.set_defaults(func=main_train)

    p_ft = sub.add_parser("finetune", help="Fine-tune a pretrained model")
    p_ft.add_argument("--config", required=True)
    p_ft.add_argument("--ckpt", required=True)
    p_ft.set_defaults(func=main_finetune)

    p_pred = sub.add_parser("predict", help="Generate perturbed cells")
    p_pred.add_argument("--config", required=True)
    p_pred.add_argument("--ckpt", required=True)
    p_pred.set_defaults(func=main_predict)

    return p

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("\nUsage:\n"
              "  python dual_vae_pipeline.py train --config config.yaml\n"
              "  python dual_vae_pipeline.py finetune --config config.yaml --ckpt model.pt\n"
              "  python dual_vae_pipeline.py predict --config config.yaml --ckpt model.pt\n")
        sys.exit(0)

    parser = build_argparser()
    args = parser.parse_args()
    args.func(args)