"""
Dataset and DataLoader for AE-DEVAE.
"""
import torch
from torch.utils.data import Dataset, DataLoader
import scanpy as sc
import numpy as np
import pandas as pd
from pathlib import Path
import json
from typing import Optional, Dict, Tuple
from scipy import sparse
from .config import DataConfig


# ============================================================================
# PREPROCESSING FUNCTIONS
# ============================================================================

def detect_data_type(X) -> bool:
    """
    Detect if data is raw counts or normalized/log-transformed.
    
    Returns:
        True if raw counts, False if already processed
    """
    if hasattr(X, 'toarray'):
        X_sample = X[:1000].toarray()
    else:
        X_sample = X[:1000]
    
    is_int = np.allclose(X_sample, X_sample.astype(int), rtol=0.01)
    max_val = X_sample.max()
    
    return is_int and max_val > 100


def compute_sparsity(X) -> float:
    """Compute sparsity of data matrix."""
    if hasattr(X, 'nnz'):
        return 1.0 - (X.nnz / (X.shape[0] * X.shape[1]))
    else:
        return 0.0


def should_keep_sparse(X, threshold: float = 0.5) -> bool:
    """Decide if data should be kept sparse."""
    if not hasattr(X, 'toarray'):
        return False  # Already dense
    
    sparsity = compute_sparsity(X)
    return sparsity > threshold    

def preprocess_adata(adata: sc.AnnData, cfg: DataConfig, store_counts: bool = True) -> sc.AnnData:
    """
    Preprocess AnnData according to config.
    
    Args:
        adata: Input AnnData
        cfg: DataConfig with preprocessing settings
        store_counts: Whether to store raw counts in .layers['counts']
    
    Returns:
        Preprocessed AnnData (modifies in place, but also returns for chaining)
    """
    is_raw_counts = detect_data_type(adata.X)
    
    # Store raw counts before preprocessing
    if store_counts and is_raw_counts:
        print("[prep] Storing raw counts in .layers['counts']")
        if 'counts' not in adata.layers:
            adata.layers['counts'] = adata.X.copy()
    elif not is_raw_counts:
        print("[WARNING] Data is already normalized/log-transformed!")
        if store_counts:
            print("[WARNING] Cannot recover true counts - count losses will be approximate")
            adata.layers['counts'] = None
    
    # Apply preprocessing
    if is_raw_counts:
        print("[prep] Normalizing raw counts...")
        sc.pp.normalize_total(adata, target_sum=cfg.target_sum)
        
        print("[prep] Applying log1p...")
        sc.pp.log1p(adata)
    
    if cfg.zscore:
        print("[prep] Z-score normalization...")
        sc.pp.scale(adata, zero_center=True)
    
    if cfg.drop_all_zero:
        print("[prep] Filtering all-zero genes...")
        n_genes_before = adata.n_vars
        sc.pp.filter_genes(adata, min_cells=1)
        n_genes_after = adata.n_vars
        if n_genes_before != n_genes_after:
            print(f"[prep] Filtered {n_genes_before - n_genes_after} all-zero genes")
    
    return adata


def build_vocab(obs: pd.DataFrame, col_batch: str, col_celltype: str) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    Build batch and celltype vocabularies.
    Ensures vocabularies are never empty to prevent model initialization errors.
    """
    batch_to_id = {}
    celltype_to_id = {}
    
    # Build Batch Vocab
    if col_batch in obs.columns:
        unique_batches = obs[col_batch].dropna().unique()
        if len(unique_batches) > 0:
            batch_to_id = {str(b): i for i, b in enumerate(sorted(unique_batches))}
            print(f"[vocab] Found {len(batch_to_id)} unique batches")
    
    if not batch_to_id:
        print(f"[vocab] ⚠ Column '{col_batch}' empty or missing. Using default batch.")
        batch_to_id = {'unknown': 0}
    
    # Build Celltype Vocab
    if col_celltype in obs.columns:
        unique_celltypes = obs[col_celltype].dropna().unique()
        if len(unique_celltypes) > 0:
            celltype_to_id = {str(c): i for i, c in enumerate(sorted(unique_celltypes))}
    
    if not celltype_to_id:
        print(f"[vocab] ⚠ Column '{col_celltype}' empty or missing. Using default celltype.")
        celltype_to_id = {'unknown': 0}
    
    return batch_to_id, celltype_to_id

# ============================================================================
# CONTROL POOL LOADING (for prediction)
# ============================================================================
def enforce_gene_order(X, var_names: list, target_gene_order: list):
    """
    Reorder genes to match target order, filling missing genes with zeros.
    Handles both dense numpy arrays and scipy sparse matrices.
    
    Args:
        X: [n_cells, n_genes] array (dense numpy or scipy sparse)
        var_names: Current gene names
        target_gene_order: Desired gene order
    
    Returns:
        X_reordered: [n_cells, n_target_genes] (same type as input)
    """
    n_cells = X.shape[0]
    n_target = len(target_gene_order)
    
    # 1. Map target genes to current indices using Pandas
    # returns -1 for missing genes, index integer otherwise
    idx_mapper = pd.Index(var_names).get_indexer(target_gene_order)
    
    # 2. Identify found vs missing
    mask_found = idx_mapper != -1
    source_indices = idx_mapper[mask_found]
    
    n_found = np.sum(mask_found)
    n_missing = n_target - n_found
    
    # 3. Construct Result Matrix
    # We slice the existing X to get all found genes in the correct target relative order
    X_subset = X[:, source_indices]
    
    if n_missing == 0:
        # Optimization: If perfect match, just return the sliced subset
        print(f"[gene_order] Perfect match ({n_target} genes).")
        return X_subset

    print(f"[gene_order] Matched {n_found}/{n_target} genes. Filling {n_missing} with zeros.")

    if sparse.issparse(X):
        # For sparse matrices, constructing a new matrix is faster than random assignment
        # LIL is best for constructing sparse matrices with structure changes
        X_reordered = sparse.lil_matrix((n_cells, n_target), dtype=X.dtype)
        
        target_indices = np.where(mask_found)[0]
        
        # This bulk assignment is efficient in LIL
        X_reordered[:, target_indices] = X_subset
        
        return X_reordered.tocsr()
        
    else:
        # For dense arrays, simple fancy indexing works instantly
        X_reordered = np.zeros((n_cells, n_target), dtype=X.dtype)
        X_reordered[:, mask_found] = X_subset
        return X_reordered


def load_control_pool(
    cfg: DataConfig,
    gene_to_idx: Dict[str, int],
    h1_flag_value: int = 1,
) -> Tuple[np.ndarray, pd.DataFrame, Dict[str, int], Dict[str, int], float, Dict[str, float], Optional[np.ndarray]]:
    """
    Load and preprocess control cell pool for prediction.
    
    Args:
        cfg: DataConfig with preprocessing settings and control_pool_h5ad path
        gene_to_idx: Gene vocabulary from training
        h1_flag_value: Value indicating H1 cells
    
    Returns:
        X_pool: [n_cells, n_genes] log1p features (in training gene order)
        obs_pool: Metadata dataframe
        batch_to_id: Batch vocabulary
        celltype_to_id: Celltype vocabulary
        global_median_depth: Median depth across all control cells
        depth_by_batch: Dict mapping batch -> median depth
        X_raw: [n_cells, n_genes] raw counts (in training gene order) - sparse or dense
    """
    print(f"[control_pool] Loading from {cfg.control_pool_h5ad}...")
    adata = sc.read_h5ad(cfg.control_pool_h5ad)
    
    # ========================================================================
    # FILTER TO H1 CONTROLS
    # ========================================================================
    
    mask_h1 = (adata.obs[cfg.col_is_h1].astype(int) == h1_flag_value) if cfg.col_is_h1 in adata.obs.columns else np.ones(len(adata), dtype=bool)
    mask_ctrl = (adata.obs[cfg.col_target].astype(str) == cfg.control_token)
    
    sel = mask_h1 & mask_ctrl
    if sel.sum() == 0:
        raise RuntimeError("No H1 controls found in control pool")
    
    adata = adata[sel].copy()
    print(f"[control_pool] Filtered to {len(adata)} H1 control cells")
    
    # ========================================================================
    # REORDER GENES IN ADATA (ONCE!)
    # ========================================================================
    
    train_genes = list(gene_to_idx.keys())
    print(f"[control_pool] Reordering genes to match training...")
    
    adata.X = enforce_gene_order(adata.X, list(adata.var_names), train_genes)
    adata.var = pd.DataFrame(index=train_genes)
    
    # ========================================================================
    # STORE RAW COUNTS (already in correct order!)
    # ========================================================================
    
    if sparse.issparse(adata.X):
        X_raw = adata.X.copy()
    else:
        X_raw = adata.X.copy()
    
    # ========================================================================
    # COMPUTE DEPTH STATISTICS (from raw counts, already ordered)
    # ========================================================================
    
    print(f"[control_pool] Computing depth statistics...")
    
    if sparse.issparse(adata.X):
        pool_depths = np.asarray(adata.X.sum(axis=1)).ravel()
    else:
        pool_depths = adata.X.sum(axis=1)
    
    global_median_depth = float(np.median(pool_depths))
    print(f"[control_pool] ✓ Global median depth: {global_median_depth:.1f} counts")
    
    # Per-batch depth medians
    depth_by_batch = {}
    if cfg.col_batch in adata.obs.columns:
        for batch in adata.obs[cfg.col_batch].unique():
            batch_mask = (adata.obs[cfg.col_batch] == batch).values
            if sparse.issparse(adata.X):
                depths_b = np.asarray(adata.X[batch_mask].sum(axis=1)).ravel()
            else:
                depths_b = adata.X[batch_mask].sum(axis=1)
            if len(depths_b) > 0:
                depth_by_batch[str(batch)] = float(np.median(depths_b))
        print(f"[control_pool] ✓ Per-batch depth medians: {len(depth_by_batch)} batches")
    
    # ========================================================================
    # PREPROCESS (genes already in correct order!)
    # ========================================================================
    
    adata = preprocess_adata(adata, cfg, store_counts=False)
    
    # Extract log1p features (already in correct order!)
    if sparse.issparse(adata.X):
        X_pool = adata.X.toarray().astype(np.float32)
    else:
        X_pool = adata.X.astype(np.float32)
    
    # Build vocabularies
    batch_to_id, celltype_to_id = build_vocab(adata.obs, cfg.col_batch, cfg.col_celltype)
    
    obs_pool = adata.obs.copy().reset_index(drop=True)
    
    return X_pool, obs_pool, batch_to_id, celltype_to_id, global_median_depth, depth_by_batch, X_raw

# ============================================================================
# DATALOADER
# ============================================================================
def get_dataloader(
    h5ad_path: str,
    cfg: DataConfig,
    batch_size: int = 128,
    shuffle: bool = True,
    num_workers: int = 4,
    gene_to_idx: Optional[Dict[str, int]] = None
) -> DataLoader:
    """Create DataLoader for perturbation data."""
    dataset = PerturbationDataset(h5ad_path, cfg, gene_to_idx)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=cfg.pin_memory
    )


# ============================================================================
# DATASET CLASS
# ============================================================================
class PerturbationDataset(Dataset):
    """
    Dataset for single-cell perturbation data.
    Uses shared preprocessing functions to ensure consistency.
    """
    
    def __init__(self, h5ad_path: str, cfg: DataConfig, gene_to_idx: Optional[Dict[str, int]] = None):
        super().__init__()
        self.cfg = cfg
        
        print(f"[data] Loading AnnData from {h5ad_path}...")
        self.adata = sc.read_h5ad(h5ad_path)
        
        # Use shared preprocessing function
        self.adata = preprocess_adata(self.adata, cfg, store_counts=True)
        
        # Build or load gene mapping
        if gene_to_idx is not None:
            print(f"[data] Using provided gene_to_idx mapping ({len(gene_to_idx)} genes)")
            self.gene_to_idx = gene_to_idx
            
            if len(self.gene_to_idx) != self.adata.n_vars:
                print(f"[WARNING] gene_to_idx size ({len(self.gene_to_idx)}) != current data genes ({self.adata.n_vars})")
        else:
            print(f"[data] Building gene_to_idx mapping from data")
            self.gene_to_idx = {
                gene: idx for idx, gene in enumerate(self.adata.var_names)
            }
        
        # Build vocabulary using shared function
        self.batch_to_id, self.celltype_to_id = build_vocab(
            self.adata.obs, cfg.col_batch, cfg.col_celltype
        )
        
        # Decide sparse vs dense
        self.keep_sparse = should_keep_sparse(self.adata.X)
        
        sparsity_pct = compute_sparsity(self.adata.X) * 100 if self.keep_sparse else 0
        sparse_str = f", {sparsity_pct:.1f}% sparse" if self.keep_sparse else ""
        print(f"[data] ✓ Loaded: {len(self):,} cells × {self.adata.n_vars:,} genes{sparse_str}")
    
    def __len__(self):
        return self.adata.n_obs
    
    def __getitem__(self, idx):
        """Get a single cell."""
        # Extract features
        if self.keep_sparse:
            x = self.adata.X[idx].toarray().ravel().astype(np.float32)
        else:
            x = self.adata.X[idx].astype(np.float32)
        
        # Get raw counts if available
        if self.adata.layers.get('counts') is not None:
            if sparse.issparse(self.adata.layers['counts']):
                counts = self.adata.layers['counts'][idx].toarray().ravel().astype(np.float32)
            else:
                counts = self.adata.layers['counts'][idx].astype(np.float32)
        else:
            counts = np.zeros_like(x)
        
        # Metadata
        obs = self.adata.obs.iloc[idx]
        target_gene = str(obs[self.cfg.col_target])
        
        # Get gene index
        gene_idx = self.gene_to_idx.get(target_gene, 0)
        
        # Context encoding
        # Use .get with a default that exists in the vocab (0, which we map to 'unknown' if vocab was empty)
        # If batch/celltype column exists, these will be correct IDs.
        # If not, build_vocab ensured vocab has {'unknown': 0}.
        batch_val = str(obs.get(self.cfg.col_batch, 'unknown'))
        batch_id = self.batch_to_id.get(batch_val, 0) # Default to 0 (unknown) if unexpected value
        
        ct_val = str(obs.get(self.cfg.col_celltype, 'unknown'))
        celltype_id = self.celltype_to_id.get(ct_val, 0) # Default to 0 (unknown)
        
        is_h1 = int(obs.get(self.cfg.col_is_h1, 0))
        
        # Library size (in log1p space)
        libsize = np.log1p(np.expm1(x).sum())
        
        return {
            'x': torch.from_numpy(x),
            'counts': torch.from_numpy(counts),
            'gene_idx': gene_idx,
            'batch_idx': batch_id,
            'celltype_idx': celltype_id,
            'is_h1': is_h1,
            'libsize': libsize,
            'is_control': int(target_gene == self.cfg.control_token)
        }
    
    @staticmethod
    def _save_mappings_to_file(gene_to_idx: Dict, batch_to_id: Dict, celltype_to_id: Dict, outdir: str):
        """Save vocabulary mappings to JSON (Implementation)."""
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        
        mappings = {
            'gene_to_idx': gene_to_idx,
            'batch_to_id': batch_to_id,
            'celltype_to_id': celltype_to_id
        }
        
        with open(outdir / 'vocab_mappings.json', 'w') as f:
            json.dump(mappings, f, indent=2)
        
        print(f"[data] ✓ Saved vocabulary mappings to {outdir / 'vocab_mappings.json'}")
    
    def save_mappings(self, outdir: str):
        """Save this dataset's mappings."""
        self._save_mappings_to_file(self.gene_to_idx, self.batch_to_id, self.celltype_to_id, outdir)
    
    @staticmethod
    def load_mappings(path: str) -> Dict:
        """Load vocabulary mappings from JSON."""
        with open(path, 'r') as f:
            return json.load(f)