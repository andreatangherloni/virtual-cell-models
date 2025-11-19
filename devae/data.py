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
from typing import Optional, Dict
from .config import DataConfig


class PerturbationDataset(Dataset):
    """
    Dataset for single-cell perturbation data.
    Features:
    - Stores raw counts before preprocessing
    - Saves/loads vocabulary mappings for consistency
    - Efficient sparse matrix handling
    """
    
    def __init__(self, h5ad_path: str, cfg: DataConfig, gene_to_idx: Optional[Dict[str, int]] = None):
        super().__init__()
        self.cfg = cfg
        
        print(f"[data] Loading AnnData from {h5ad_path}...")
        self.adata = sc.read_h5ad(h5ad_path)
        
        # Detect data type
        self._detect_data_type()
        
        # Store raw counts before preprocessing!
        if self.is_raw_counts:
            print("[data] Storing raw counts in .layers['counts']")
            # Save to layers (works with sparse matrices)
            if 'counts' not in self.adata.layers:
                self.adata.layers['counts'] = self.adata.X.copy()
        else:
            print("[WARNING] Data is already normalized/log-transformed!")
            print("[WARNING] Cannot recover true counts - count losses will be approximate")
            self.adata.layers['counts'] = None
        
        # Preprocess (this modifies .X)
        self._preprocess()
        
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
        
        # Build vocabulary for context variables
        self._build_vocab()
        
        self.keep_sparse = self._should_keep_sparse()
        
        sparsity_pct = self._compute_sparsity() * 100 if self.keep_sparse else 0
        sparse_str = f", {sparsity_pct:.1f}% sparse" if self.keep_sparse else ""
        print(f"[data] ✓ Loaded: {len(self):,} cells × {self.adata.n_vars:,} genes{sparse_str}")

    
    def _should_keep_sparse(self) -> bool:
        """Decide if we should keep data sparse."""
        X = self.adata.X
        
        if not hasattr(X, 'toarray'):
            return False  # Already dense
        
        # Compute sparsity
        sparsity = self._compute_sparsity()
        
        # Keep sparse if > 50% zeros
        return sparsity > 0.5
    
    def _compute_sparsity(self) -> float:
        """Compute sparsity of data matrix."""
        X = self.adata.X
        if hasattr(X, 'nnz'):
            return 1.0 - (X.nnz / (X.shape[0] * X.shape[1]))
        else:
            return 0.0
    
    def _detect_data_type(self):
        """Detect if data is raw counts or normalized."""
        X = self.adata.X
        if hasattr(X, 'toarray'):
            X_sample = X[:1000].toarray()
        else:
            X_sample = X[:1000]
        
        is_int = np.allclose(X_sample, X_sample.astype(int))
        max_val = X_sample.max()
        
        if is_int and max_val > 100:
            self.is_raw_counts = True
            print("[data] Detected RAW COUNTS")
        else:
            self.is_raw_counts = False
            print("[data] Detected NORMALIZED/LOG1P data")
    
    def _preprocess(self):
        """Preprocess data according to config."""
        if self.is_raw_counts:
            print("[prep] Normalizing raw counts...")
            sc.pp.normalize_total(self.adata)
            
            print("[prep] Applying log1p...")
            sc.pp.log1p(self.adata)
        
        if self.cfg.zscore:
            print("[prep] Z-score normalization...")
            sc.pp.scale(self.adata, zero_center=True)
        
        if self.cfg.drop_all_zero:
            # Be careful: this changes gene ordering!
            # Only filter if we haven't loaded a gene_to_idx
            if not hasattr(self, 'gene_to_idx') or self.gene_to_idx is None:
                print("[prep] Filtering all-zero genes...")
                n_genes_before = self.adata.n_vars
                sc.pp.filter_genes(self.adata, min_cells=1)
                n_genes_after = self.adata.n_vars
                print(f"[prep] Filtered {n_genes_before - n_genes_after} all-zero genes")
    
    def _build_vocab(self):
        """Build vocabulary mappings (batch, celltype only)."""
        obs = self.adata.obs
        
        # Batches
        if self.cfg.col_batch in obs.columns:
            unique_batches = sorted(obs[self.cfg.col_batch].unique())
            self.batch_to_id = {batch: idx for idx, batch in enumerate(unique_batches)}
            print(f"[vocab] Found {len(self.batch_to_id)} unique batches")
        else:
            self.batch_to_id = {'unknown': 0}
            print("[vocab] No batch column found, using default")
        
        # Cell types
        if self.cfg.col_celltype in obs.columns:
            unique_celltypes = sorted(obs[self.cfg.col_celltype].unique())
            self.celltype_to_id = {ct: idx for idx, ct in enumerate(unique_celltypes)}
            print(f"[vocab] Found {len(self.celltype_to_id)} unique cell types")
        else:
            self.celltype_to_id = {'unknown': 0}
            print("[vocab] No celltype column found, using default")
    
    # ============================================================================
    # Save/Load Vocabulary Mappings
    # ============================================================================
    
    def save_mappings(self, output_dir: str):
        """
        Save vocabulary mappings for future use.
        
        This ensures consistency across training stages (stage1 → stage2 → prediction).
        
        Args:
            output_dir: Directory to save mappings
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        mappings = {
            'gene_to_idx': self.gene_to_idx,
            'batch_to_id': self.batch_to_id,
            'celltype_to_id': self.celltype_to_id,
        }
        
        mapping_path = output_dir / 'vocab_mappings.json'
        with open(mapping_path, 'w') as f:
            json.dump(mappings, f, indent=2)
        
        print(f"[data] ✓ Saved mappings → {mapping_path}")
    
    @staticmethod
    def load_mappings(mapping_path: str) -> Dict:
        """
        Load vocabulary mappings from file.
        
        Args:
            mapping_path: Path to vocab_mappings.json
        
        Returns:
            Dictionary with gene_to_idx, batch_to_id, celltype_to_id
        """
        with open(mapping_path, 'r') as f:
            mappings = json.load(f)
        
        print(f"[data] Loaded vocabulary mappings from {mapping_path}")
        
        return mappings
    
    # ============================================================================
    # Dataset methods
    # ============================================================================
    
    def __len__(self):
        return self.adata.n_obs
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample."""
        # Gene expression (log1p normalized)
        x = self.adata.X[idx]
       
        # Only densify if needed (small memory footprint per sample)
        if hasattr(x, 'toarray'):
            x = x.toarray().squeeze()
            
        x = torch.from_numpy(x.astype(np.float32))
        
        # Raw counts (if available)
        if self.adata.layers.get('counts') is not None:
            counts = self.adata.layers['counts'][idx]
            if hasattr(counts, 'toarray'):
                counts = counts.toarray().squeeze()
            counts = torch.from_numpy(counts.astype(np.float32))
        else:
            counts = None
        
        obs = self.adata.obs.iloc[idx]
        
        # Target gene: convert NAME → GENE INDEX
        target_gene_name = obs.get(self.cfg.col_target, self.cfg.control_token)
        if pd.isna(target_gene_name):
            target_gene_name = self.cfg.control_token
        
        if target_gene_name == self.cfg.control_token:
            gene_idx = 0
            is_control = 1
        else:
            gene_idx = self.gene_to_idx.get(target_gene_name, 0)
            is_control = 0
            
            if gene_idx == 0 and target_gene_name != list(self.gene_to_idx.keys())[0]:
                # Only warn once per unique gene
                if not hasattr(self, '_warned_genes'):
                    self._warned_genes = set()
                if target_gene_name not in self._warned_genes:
                    print(f"[WARNING] Target gene '{target_gene_name}' not found in var_names!")
                    self._warned_genes.add(target_gene_name)
        
        # Batch
        batch_name = obs.get(self.cfg.col_batch, 'unknown')
        if pd.isna(batch_name):
            batch_name = 'unknown'
        batch_id = self.batch_to_id.get(batch_name, 0)
        
        # Cell type
        celltype = obs.get(self.cfg.col_celltype, 'unknown')
        if pd.isna(celltype):
            celltype = 'unknown'
        celltype_id = self.celltype_to_id.get(celltype, 0)
        
        # H1 flag
        is_h1 = int(obs.get(self.cfg.col_is_h1, 0))
        
        # Library size (from raw counts if available, else approximate)
        if counts is not None:
            libsize = counts.sum().item()
        else:
            libsize = torch.expm1(x).sum().item()
        
        item = {
            'x': x,
            'gene_idx': torch.tensor(gene_idx, dtype=torch.long),
            'is_control': torch.tensor(is_control, dtype=torch.float32),
            'batch_id': torch.tensor(batch_id, dtype=torch.long),
            'celltype_id': torch.tensor(celltype_id, dtype=torch.long),
            'is_h1': torch.tensor(is_h1, dtype=torch.long),
            'libsize': torch.tensor(libsize, dtype=torch.float32),
        }
        
        # Include raw counts if available
        if counts is not None:
            item['counts'] = counts
        
        return item


def collate_fn(batch):
    """
    Collate function for DataLoader.
    
    Handles sparse data efficiently:
    - Keeps data sparse during loading
    - Densifies only when creating batches
    - Moves to device happens later in training loop
    """
    # Check if we have counts
    has_counts = 'counts' in batch[0]
    
    # Stack tensors
    collated = {
        'x': torch.stack([item['x'] for item in batch]),
        'gene_idx': torch.stack([item['gene_idx'] for item in batch]),
        'is_control': torch.stack([item['is_control'] for item in batch]),
        'batch_id': torch.stack([item['batch_id'] for item in batch]),
        'celltype_id': torch.stack([item['celltype_id'] for item in batch]),
        'is_h1': torch.stack([item['is_h1'] for item in batch]),
        'libsize': torch.stack([item['libsize'] for item in batch]),
    }
    
    # Add counts if available
    if has_counts:
        collated['counts'] = torch.stack([item['counts'] for item in batch])
    
    return collated


def get_dataloader(
    h5ad_path: str,
    cfg: DataConfig,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4,
    gene_to_idx: Optional[Dict[str, int]] = None
) -> DataLoader:
    """
    Create DataLoader with optional gene mapping for consistency.
    
    Args:
        h5ad_path: Path to h5ad file
        cfg: Data configuration
        batch_size: Batch size
        shuffle: Whether to shuffle
        num_workers: Number of worker processes
        gene_to_idx: Optional pre-computed gene-to-index mapping (for consistency across stages)
    
    Returns:
        DataLoader with PerturbationDataset
    """
    dataset = PerturbationDataset(h5ad_path, cfg, gene_to_idx=gene_to_idx)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=cfg.pin_memory,
        drop_last=True if shuffle else False
    )