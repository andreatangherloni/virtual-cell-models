import random
import numpy as np
from scipy import sparse
import anndata as ad
import torch
from collections import defaultdict
from typing import Dict, Any, List, Optional, Tuple, Iterable
from torch.utils.data import Dataset, DataLoader, Sampler
from sklearn.preprocessing import LabelEncoder
from .config import DataConfig
from .utils import ensure_csr, detect_logged_data, prepare_normalized_views, logger

class AnnDataDataset(Dataset):
    """
    AnnData-backed dataset that cleanly handles:
      - Raw-count input  → optional normalize_total(target_sum) + log1p
      - Already-log input → pass-through (no re-logging)
    Provides:
      - X_log  : normalized/log1p features (CSR)
      - X_counts : raw counts if available (CSR)
      - has_counts_targets : True if X_counts exists (used for count head)
      - preprocess_info : summary dict for downstream inspection
    """

    def __init__(self,
                 adata: ad.AnnData,
                 data_cfg: DataConfig,
                 gene2id: Dict[str, int],
                 encoders: Dict[str, LabelEncoder] = None,
                 zscore: bool = False,
                 zscore_stats: Optional[Tuple[np.ndarray, np.ndarray]] = None):
        super().__init__()
        self.adata = adata
        self.data_cfg = data_cfg
        self.gene2id = gene2id
        self.encoders = encoders or {}
        self.zscore = zscore

        # ------------------------------------------------------------------ #
        # 1) Detect whether input is already log-transformed
        # ------------------------------------------------------------------ #
        self.is_logged_input = bool(detect_logged_data(adata.X, check_n=200))
        self.used_normalization = False
        self.X_counts = None
        self.has_counts_targets = False

        normalize  = self.data_cfg.normalize
        target_sum = self.data_cfg.target_sum
        if target_sum is not None:
            try:
                target_sum = float(target_sum)
            except Exception:
                target_sum = None

        # ------------------------------------------------------------------ #
        # 2) Build X_log (model input) and optionally X_counts (for count head)
        # ------------------------------------------------------------------ #
        if self.is_logged_input:
            # Already log-like → no normalization or count targets
            logger("[data]", "detected log-transformed input → using as model features")
            self.X_log = ensure_csr(adata.X) if sparse.issparse(adata.X) else sparse.csr_matrix(adata.X)
        else:
            # Raw counts path
            logger("[data]", "detected raw counts input")
            if normalize:
                X_counts_csr, X_log1p = prepare_normalized_views(
                    adata, normalize=True, target_sum=target_sum
                )
                self.used_normalization = True
            else:
                logger("[data]", "using raw counts with log1p only (no normalization)")
                X_counts_csr, X_log1p = prepare_normalized_views(
                    adata, normalize=False, target_sum=None
                )
                self.used_normalization = False

            self.X_counts = ensure_csr(X_counts_csr)
            self.X_log = ensure_csr(X_log1p) if sparse.issparse(X_log1p) else sparse.csr_matrix(X_log1p)
            self.has_counts_targets = True

        # ------------------------------------------------------------------ #
        # 3) Drop all-zero rows if requested
        # ------------------------------------------------------------------ #
        self.obs = adata.obs.reset_index(drop=True)
        if self.data_cfg.drop_all_zero:
            nnz_per_row = np.diff(self.X_log.indptr)
            mask = (nnz_per_row > 0)
            if mask.mean() < 1.0:
                self.X_log = self.X_log[mask]
                if self.X_counts is not None:
                    self.X_counts = self.X_counts[mask]
                self.obs = self.obs.iloc[mask].reset_index(drop=True)

        # ------------------------------------------------------------------ #
        # 4) Optional z-scoring (on log1p features)
        # ------------------------------------------------------------------ #
        self.zmean = None
        self.zstd = None
        if self.zscore:
            if zscore_stats is None:
                N = self.X_log.shape[0]
                sums = np.asarray(self.X_log.sum(axis=0)).ravel()
                sq_sums = np.asarray(self.X_log.power(2).sum(axis=0)).ravel()
                mean = (sums / max(1, N)).astype(np.float32, copy=False)[None, :]
                var = (sq_sums / max(1, N)) - (mean.ravel() ** 2)
                eps = (self.data_cfg.zscore_eps)
                std = (np.sqrt(np.maximum(var, 0.0)) + eps).astype(np.float32, copy=False)[None, :]
                self.zmean, self.zstd = mean, std
            else:
                self.zmean, self.zstd = zscore_stats

        # ------------------------------------------------------------------ #
        # 5) Build categorical encodings
        # ------------------------------------------------------------------ #
        self.col_target   = self.data_cfg.col_target
        self.col_batch    = self.data_cfg.col_batch
        self.col_celltype = self.data_cfg.col_celltype
        self.col_is_h1    = self.data_cfg.col_is_h1

        for col in [self.col_batch, self.col_celltype]:
            if col in self.obs.columns:
                le = self.encoders.get(col, LabelEncoder())
                vals = self.obs[col].astype(str).values
                le.fit(vals)
                self.encoders[col] = le

        tg = self.obs[self.col_target].astype(str).values
        self.gid = np.array([self.gene2id.get(g, 0) for g in tg], dtype=np.int64)
        self.batch_id = self._encode_cat(self.col_batch)
        self.ct_id = self._encode_cat(self.col_celltype)
        self.is_h1 = self.obs[self.col_is_h1].values.astype(np.int64) \
            if self.col_is_h1 in self.obs.columns else np.zeros(len(self.obs), dtype=np.int64)
        self.is_control = (tg == self.data_cfg.control_token).astype(np.int64)

        # ------------------------------------------------------------------ #
        # 6) Summarize preprocessing status
        # ------------------------------------------------------------------ #
        self.preprocess_info = {
            "is_logged_input": self.is_logged_input,
            "used_normalization": self.used_normalization,
            "has_counts_targets": self.has_counts_targets,
        }
        logger("[data]", f"preprocessing summary: {self.preprocess_info}")

        # Warn if count head is enabled but no raw counts available
        if not self.has_counts_targets and self.data_cfg.use_count_head:
            logger("[warn]", "count head requested but no raw-count data available → disabling at runtime")

    # ---------------------------------------------------------------------- #
    # Dataset Interface
    # ---------------------------------------------------------------------- #
    def _encode_cat(self, col: str) -> np.ndarray:
        if col in self.obs.columns:
            le = self.encoders[col]
            return le.transform(self.obs[col].astype(str).values).astype(np.int64)
        else:
            return np.zeros(len(self.obs), dtype=np.int64)

    def __len__(self):
        return self.X_log.shape[0]

    def __getitem__(self, idx):
        x = self.X_log[idx].toarray().ravel().astype(np.float32, copy=False)
        if self.zscore and (self.zmean is not None) and (self.zstd is not None):
            x = (x - self.zmean.ravel()) / self.zstd.ravel()

        out = {
            "x": x,
            "gid": int(self.gid[idx]),
            "batch_id": int(self.batch_id[idx]),
            "ct_id": int(self.ct_id[idx]),
            "is_h1": int(self.is_h1[idx]),
            "is_control": int(self.is_control[idx]),
        }

        if self.has_counts_targets and (self.X_counts is not None):
            y = self.X_counts[idx].toarray().ravel().astype(np.float32, copy=False)
            out["y_counts"] = y

        return out


# -------------------------------------------------------------------------- #
# Collation + Loader
# -------------------------------------------------------------------------- #
def collate_batch(samples):
    """Collate list of dicts into batch tensors."""
    assert len(samples) > 0, "Empty batch"
    batch = {}
    has_counts = "y_counts" in samples[0]

    X = [np.asarray(s["x"], dtype=np.float32) for s in samples]
    batch["x"] = torch.from_numpy(np.stack(X, axis=0))

    if has_counts:
        Y = [np.asarray(s["y_counts"], dtype=np.float32) for s in samples]
        batch["y_counts"] = torch.from_numpy(np.stack(Y, axis=0))

    for k in ["gid", "batch_id", "ct_id", "is_h1", "is_control"]:
        vals = np.asarray([int(s[k]) for s in samples], dtype=np.int64)
        batch[k] = torch.from_numpy(vals)

    return batch


def make_dataloader(adata_path: str,
                    data_cfg: DataConfig,
                    gene2id: Dict[str, int],
                    encoders: Dict[str, LabelEncoder],
                    batch_size: int,
                    shuffle: bool,
                    zscore: bool,
                    zstats: Optional[Tuple[np.ndarray, np.ndarray]]):
    adata = ad.read_h5ad(adata_path)
    ds = AnnDataDataset(
        adata, data_cfg, gene2id, encoders=encoders,
        zscore=zscore, zscore_stats=zstats
    )
    pin_memory = torch.cuda.is_available()
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=collate_batch,
        drop_last=True,
        pin_memory=pin_memory,
        persistent_workers=False,
    )
    return ds, dl


# -------------------------------------------------------------------------- #
# Paired Sampler (unchanged)
# -------------------------------------------------------------------------- #
def _key_tuple(ct_id: int, batch_id: Optional[int], is_h1: Optional[int],
               match_batch: bool, match_h1: bool) -> Tuple[int, int, int]:
    b = batch_id if match_batch else -1
    h = is_h1 if match_h1 else -1
    return (ct_id, b, h)


class PairedPerturbBatchSampler(Sampler[List[int]]):
    """Builds batches with (perturbed, matched-control) pairs."""
    def __init__(self,
                 dataset,
                 batch_size: int,
                 min_pos_per_target: int = 8,
                 match_batch: bool = True,
                 match_h1: bool = False,
                 ctrl_per_pos: int = 1,
                 shuffle: bool = True,
                 drop_last: bool = True,
                 seed: int = 42,
                 prioritize_batch: bool = True):
        super().__init__()
        self.ds = dataset
        self.N = len(dataset)
        self.batch_size = int(batch_size)
        self.min_pos_per_target = int(min_pos_per_target)
        self.match_batch = bool(match_batch)
        self.match_h1 = bool(match_h1)
        self.ctrl_per_pos = int(ctrl_per_pos)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.prioritize_batch = bool(prioritize_batch)
        self.rng = random.Random(seed)

        self.has_batch = (self.ds.col_batch in self.ds.obs.columns)
        self.has_ct    = (self.ds.col_celltype in self.ds.obs.columns)
        self.has_h1    = (self.ds.col_is_h1 in self.ds.obs.columns)

        gid      = np.asarray(self.ds.gid, dtype=np.int64)
        ct_id    = np.asarray(self.ds.ct_id, dtype=np.int64)    if self.has_ct    else np.zeros(self.N, dtype=np.int64)
        batch_id = np.asarray(self.ds.batch_id, dtype=np.int64) if self.has_batch else np.zeros(self.N, dtype=np.int64)
        is_h1    = np.asarray(self.ds.is_h1, dtype=np.int64)    if self.has_h1    else np.zeros(self.N, dtype=np.int64)
        is_ctrl  = np.asarray(self.ds.is_control, dtype=np.int64)

        self.ctrl_global: List[int] = np.where(is_ctrl == 1)[0].tolist()
        self.pos_global:  List[int] = np.where(is_ctrl == 0)[0].tolist()

        self.ctrl_by_key: Dict[Tuple[int,int,int], List[int]] = defaultdict(list)
        self.pos_by_key_gid: Dict[Tuple[int,int,int,int], List[int]] = defaultdict(list)
        self.pos_by_key: Dict[Tuple[int,int,int], List[int]] = defaultdict(list)

        for idx in range(self.N):
            key = _key_tuple(int(ct_id[idx]), int(batch_id[idx]), int(is_h1[idx]),
                             self.match_batch, self.match_h1)
            if is_ctrl[idx] == 1:
                self.ctrl_by_key[key].append(idx)
            else:
                self.pos_by_key[key].append(idx)
                self.pos_by_key_gid[(key[0], key[1], key[2], int(gid[idx]))].append(idx)

        self.units: List[Tuple[Tuple[int,int,int], int]] = []
        for (ct, b, h, g) in self.pos_by_key_gid.keys():
            self.units.append(((ct, b, h), g))

        self.batches_present: List[int] = []
        self.units_by_batch: Dict[int, List[Tuple[Tuple[int,int,int], int]]] = defaultdict(list)

        if self.has_batch and self.prioritize_batch:
            if self.match_batch:
                batch_vals = set()
                for (key, g) in self.units:
                    b = key[1]
                    if b != -1:
                        self.units_by_batch[b].append((key, g))
                        batch_vals.add(b)
                self.batches_present = sorted(batch_vals)
            else:
                from itertools import islice
                unit_to_batches = defaultdict(set)
                for (key, g), idxs in self.pos_by_key_gid.items():
                    for j in islice(idxs, 0, 32):
                        unit_to_batches[(key, g)].add(int(batch_id[j]))
                batch_vals = set()
                for (key, g), bset in unit_to_batches.items():
                    for b in bset:
                        self.units_by_batch[b].append((key, g))
                        batch_vals.add(b)
                self.batches_present = sorted(batch_vals)

        if self.shuffle:
            self.rng.shuffle(self.units)
            if self.batches_present:
                self.rng.shuffle(self.batches_present)
                for b in self.batches_present:
                    self.rng.shuffle(self.units_by_batch[b])

        self.num_batches_est = max(1, (len(self.pos_global) * (1 + self.ctrl_per_pos)) // self.batch_size)

    def __len__(self) -> int:
        return self.num_batches_est if self.drop_last else self.num_batches_est + 1

    def _sample_from(self, pool: List[int], k: int) -> List[int]:
        if not pool or k <= 0:
            return []
        if k <= len(pool):
            return self.rng.sample(pool, k)
        return [self.rng.choice(pool) for _ in range(k)]

    def _yield_from_units(self, units_iter: List[Tuple[Tuple[int,int,int], int]]):
        batch: List[int] = []
        unit_ptr = 0
        units = list(units_iter)

        while unit_ptr < len(units):
            batch.clear()
            capacity = self.batch_size

            while capacity > 0 and unit_ptr < len(units):
                key, gid = units[unit_ptr]
                unit_ptr += 1

                pos_pool = self.pos_by_key_gid.get((key[0], key[1], key[2], gid), [])
                if not pos_pool:
                    continue

                block = (1 + self.ctrl_per_pos)
                max_pos_here = max(1, capacity // block)
                n_pos = min(self.min_pos_per_target, max_pos_here, len(pos_pool))
                if n_pos <= 0:
                    continue

                pos_idxs = self._sample_from(pos_pool, n_pos)
                batch.extend(pos_idxs)

                ctrl_pool = self.ctrl_by_key.get(key, [])
                n_ctrl = n_pos * self.ctrl_per_pos
                if not ctrl_pool:
                    ctrl_pool = self.ctrl_global
                ctrl_idxs = self._sample_from(ctrl_pool, n_ctrl)
                batch.extend(ctrl_idxs)

                capacity = self.batch_size - len(batch)

            if not batch:
                break

            if len(batch) == self.batch_size or not self.drop_last:
                yield batch

    def __iter__(self) -> Iterable[List[int]]:
        if self.has_batch and self.prioritize_batch and self.batches_present:
            batches = list(self.batches_present)
            if self.shuffle:
                self.rng.shuffle(batches)
                for b in batches:
                    self.rng.shuffle(self.units_by_batch[b])
            for b in batches:
                yield from self._yield_from_units(self.units_by_batch[b])
            return

        units = list(self.units)
        if self.shuffle:
            self.rng.shuffle(units)
        yield from self._yield_from_units(units)