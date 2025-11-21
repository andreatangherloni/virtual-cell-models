"""
Debug prediction pipeline to understand scale mismatches and
calculate proxy metrics for the Virtual Cell Challenge.

Usage:
    python debug_predictions_v2.py \
        --pred predictions_stage_b.h5ad \
        --truth data/val_split_h1.h5ad \
        --config stage_b_h1_finetune.yaml
"""

import argparse
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
import matplotlib.pyplot as plt
import yaml  # <--- NEW: To read config for control token
# <--- NEW: Added pearson, spearman, and wasserstein (EMD)
from scipy.stats import pearsonr, spearmanr, wasserstein_distance

def detect_scale(X, name="Data"):
    """Detect if data is in counts or log1p scale."""
    if sparse.issparse(X):
        # Sample to avoid dense conversion of huge matrix
        idx = np.random.choice(X.shape[0], size=min(500000, X.shape[0]), replace=False)
        X_sample = X[idx].toarray()
    else:
        idx = np.random.choice(X.shape[0], size=min(500000, X.shape[0]), replace=False)
        X_sample = np.asarray(X[idx])
        
    if X_sample.size == 0:
        print(f"\n=== {name} Scale Detection ===")
        print("   (No data to analyze)")
        return "unknown", {}
        
    max_val = X_sample.max()
    min_val = X_sample.min()
    mean_val = X_sample.mean()
    
    # Check if integer-like
    is_int = np.allclose(X_sample, np.round(X_sample), atol=1e-3)
    # Check sparsity
    sparsity = (X_sample == 0).mean()
    
    print(f"\n=== {name} Scale Detection ===")
    print(f"Min: {min_val:.4f}")
    print(f"Max: {max_val:.4f}")
    print(f"Mean: {mean_val:.4f}")
    print(f"Median: {np.median(X_sample):.4f}")
    print(f"Sparsity: {sparsity*100:.1f}%")
    print(f"Integer-like: {is_int}")
    
    # Determine scale
    if is_int and max_val > 100:
        scale = "raw_counts"
        print(f"⚠️  Detected: RAW COUNTS")
    elif is_int and max_val < 20 and max_val > 0:
        scale = "normalized_counts"  
        print(f"⚠️  Detected: NORMALIZED COUNTS (CPM/TPM scaled)")
    elif (not is_int) and max_val < 20 and min_val >= -0.1:
        scale = "log1p"
        print(f"✅ Detected: LOG1P NORMALIZED")
    else:
        scale = "unknown"
        print(f"❌ Detected: UNKNOWN SCALE! (Min: {min_val}, Max: {max_val}, Int: {is_int})")
    
    print("="*40)
    
    return scale, {
        'max': max_val,
        'min': min_val,
        'mean': mean_val,
        'sparsity': sparsity,
        'is_int': is_int,
    }


def compare_per_target(pred_adata, truth_adata, target_col='target_gene', control_token='non-targeting', top_k_de=200): # <--- NEW params
    """Compare predictions vs truth per target."""
    
    print("\n" + "="*70)
    print("PER-TARGET ANALYSIS (METRIC PROXIES)")
    print("="*70)
    
    pred_targets = set(pred_adata.obs[target_col].unique())
    truth_targets = set(truth_adata.obs[target_col].unique())
    
    common_targets = (pred_targets & truth_targets) - {control_token} # <--- NEW: Exclude control
    print(f"\nCommon perturbation targets: {len(common_targets)}")
    
    if not common_targets:
        print("❌ NO COMMON TARGETS! Check target_gene column names and control_token.")
        return
    
    # <--- NEW: Get control pseudobulks once
    eps = 1e-8
    pred_ctrl_mask = pred_adata.obs[target_col] == control_token
    truth_ctrl_mask = truth_adata.obs[target_col] == control_token
    
    if not pred_ctrl_mask.any() or not truth_ctrl_mask.any():
        print("❌ Cannot find control cells. Skipping LFC analysis.")
        return
        
    pred_ctrl_bulk = np.asarray(pred_adata[pred_ctrl_mask].X.mean(axis=0)).ravel()
    truth_ctrl_bulk = np.asarray(truth_adata[truth_ctrl_mask].X.mean(axis=0)).ravel()
    
    # Log-transform the bulks
    pred_ctrl_logbulk = np.log1p(pred_ctrl_bulk)
    truth_ctrl_logbulk = np.log1p(truth_ctrl_bulk)
    
    # Sample a few targets
    sample_targets = sorted(list(common_targets))[:5]
    print(f"Analyzing {len(sample_targets)} sample targets: {sample_targets}\n")
    
    for target in sample_targets:
        pred_mask = pred_adata.obs[target_col] == target
        truth_mask = truth_adata.obs[target_col] == target
        
        pred_X = pred_adata[pred_mask].X
        truth_X = truth_adata[truth_mask].X
        
        if sparse.issparse(pred_X):
            pred_X = pred_X.toarray()
        if sparse.issparse(truth_X):
            truth_X = truth_X.toarray()
        
        print(f"--- Target: {target} ---")
        
        # --- Basic Stats (from your original script) ---
        print(f"[Stats] Pred cells: {pred_X.shape[0]}, Truth cells: {truth_X.shape[0]}")
        print(f"[Stats] Pred mean UMI: {pred_X.sum(axis=1).mean():.1f}")
        print(f"[Stats] Truth mean UMI: {truth_X.sum(axis=1).mean():.1f}")
        print(f"[Stats] Pred sparsity: {(pred_X == 0).mean()*100:.1f}%")
        print(f"[Stats] Truth sparsity: {(truth_X == 0).mean()*100:.1f}%")
        
        # --- Pseudobulk LFC Calculation ---
        pred_pert_bulk = np.asarray(pred_X.mean(axis=0)).ravel()
        truth_pert_bulk = np.asarray(truth_X.mean(axis=0)).ravel()
        
        lfc_pred = np.log1p(pred_pert_bulk) - pred_ctrl_logbulk
        lfc_truth = np.log1p(truth_pert_bulk) - truth_ctrl_logbulk
        
        # --- Metric 1: LFC EMD (Lower is better) ---
        lfc_emd = wasserstein_distance(lfc_pred, lfc_truth)
        print(f"    METRIC 1 (LFC EMD): {lfc_emd:.4f}")
        
        # --- Metric 2 (Proxy): LFC Magnitude Ratio (Near 1.0 is better) ---
        mag_pred = np.linalg.norm(lfc_pred)
        mag_truth = np.linalg.norm(lfc_truth)
        mag_ratio = (mag_pred + eps) / (mag_truth + eps)
        print(f"    METRIC 2 (LFC Mag Ratio): {mag_ratio:.3f}  (PredMag={mag_pred:.2f}, TruthMag={mag_truth:.2f})")

        # --- Metric 3 (Proxy): Top-K DE Jaccard (Higher is better) ---
        top_k = min(top_k_de, len(lfc_pred))
        top_genes_pred_idx = np.argsort(-np.abs(lfc_pred))[:top_k]
        top_genes_truth_idx = np.argsort(-np.abs(lfc_truth))[:top_k]
        
        top_genes_pred = set(pred_adata.var_names[top_genes_pred_idx])
        top_genes_truth = set(truth_adata.var_names[top_genes_truth_idx])
        
        jaccard = len(top_genes_pred & top_genes_truth) / len(top_genes_pred | top_genes_truth)
        print(f"    METRIC 3 (Top-{top_k} Jaccard): {jaccard:.3f}\n")


def check_cell_eval_format(adata, name="AnnData"):
    """Check if AnnData is in correct format for cell-eval."""
    
    print(f"\n=== {name} Format Check ===")
    
    # Check obs
    required_cols = ['target_gene']
    for col in required_cols:
        if col in adata.obs.columns:
            print(f"✅ Has '{col}' column")
            print(f"   Unique values: {adata.obs[col].nunique()}")
        else:
            print(f"❌ Missing '{col}' column")
    
    # Check var
    print(f"\nVar names: {len(adata.var_names)} genes")
    print(f"First 5: {list(adata.var_names[:5])}")
    
    # Check X
    print(f"\nX matrix:")
    print(f"  Shape: {adata.X.shape}")
    print(f"  Sparse: {sparse.issparse(adata.X)}")
    print(f"  Dtype: {adata.X.dtype}")
    
    print("="*40)


def plot_distributions(pred_adata, truth_adata, control_token, output_path=None): # <--- NEW param
    """Plot distribution comparisons."""
    
    try:
        import matplotlib.pyplot as plt
        
        # --- Get data ---
        pred_X = pred_adata.X
        truth_X = truth_adata.X
        
        if sparse.issparse(pred_X):
            pred_X = pred_X.toarray()
        if sparse.issparse(truth_X):
            truth_X = truth_X.toarray()
        
        # --- Library sizes ---
        pred_lib = pred_X.sum(axis=1)
        truth_lib = truth_X.sum(axis=1)
        
        fig, axes = plt.subplots(3, 2, figsize=(12, 15)) # <--- NEW: 3x2 grid
        
        # Library size histograms
        axes[0, 0].hist(pred_lib[pred_lib > 0], bins=50, alpha=0.7, label='Pred', density=True)
        axes[0, 0].hist(truth_lib[truth_lib > 0], bins=50, alpha=0.7, label='Truth', density=True)
        axes[0, 0].set_xlabel('Library Size')
        axes[0, 0].set_ylabel('Density')
        axes[0, 0].set_title('Library Size Distribution (non-zero)')
        axes[0, 0].legend()
        axes[0, 0].set_yscale('log')
        
        # --- Gene expression distributions (non-zero) ---
        pred_nonzero = pred_X[pred_X > 0].flatten()
        truth_nonzero = truth_X[truth_X > 0].flatten()
        
        # Sample for speed
        if len(pred_nonzero) > 50000:
            pred_nonzero = np.random.choice(pred_nonzero, 50000, replace=False)
        if len(truth_nonzero) > 50000:
            truth_nonzero = np.random.choice(truth_nonzero, 50000, replace=False)
        
        axes[0, 1].hist(pred_nonzero, bins=50, alpha=0.7, label='Pred', density=True)
        axes[0, 1].hist(truth_nonzero, bins=50, alpha=0.7, label='Truth', density=True)
        axes[0, 1].set_xlabel('Expression (non-zero)')
        axes[0, 1].set_ylabel('Density')
        axes[0, 1].set_title('Non-Zero Expression Distribution')
        axes[0, 1].legend()
        axes[0, 1].set_yscale('log')
        
        # --- Sparsity per gene ---
        pred_sparsity = (pred_X == 0).mean(axis=0)
        truth_sparsity = (truth_X == 0).mean(axis=0)
        
        axes[1, 0].scatter(truth_sparsity, pred_sparsity, alpha=0.1, s=1)
        axes[1, 0].plot([0, 1], [0, 1], 'r--', label='y=x')
        axes[1, 0].set_xlabel('Truth Sparsity')
        axes[1, 0].set_ylabel('Pred Sparsity')
        axes[1, 0].set_title('Per-Gene Sparsity')
        axes[1, 0].legend()
        
        # --- Mean expression per gene ---
        pred_mean = pred_X.mean(axis=0)
        truth_mean = truth_X.mean(axis=0)
        
        axes[1, 1].scatter(truth_mean, pred_mean, alpha=0.1, s=1)
        axes[1, 1].plot([truth_mean.min(), truth_mean.max()], 
                       [truth_mean.min(), truth_mean.max()], 'r--', label='y=x')
        axes[1, 1].set_xlabel('Truth Mean Expression')
        axes[1, 1].set_ylabel('Pred Mean Expression')
        axes[1, 1].set_title('Per-Gene Mean Expression')
        axes[1, 1].set_xscale('log')
        axes[1, 1].set_yscale('log')
        axes[1, 1].legend()

        # <--- NEW: LFC Plots ---
        try:
            # Global LFC (all perts vs. all controls)
            pred_ctrl_mask = pred_adata.obs['target_gene'] == control_token
            truth_ctrl_mask = truth_adata.obs['target_gene'] == control_token
            pred_pert_mask = pred_adata.obs['target_gene'] != control_token
            truth_pert_mask = truth_adata.obs['target_gene'] != control_token

            pred_ctrl_bulk = np.log1p(np.asarray(pred_adata[pred_ctrl_mask].X.mean(axis=0)).ravel())
            truth_ctrl_bulk = np.log1p(np.asarray(truth_adata[truth_ctrl_mask].X.mean(axis=0)).ravel())
            pred_pert_bulk = np.log1p(np.asarray(pred_adata[pred_pert_mask].X.mean(axis=0)).ravel())
            truth_pert_bulk = np.log1p(np.asarray(truth_adata[truth_pert_mask].X.mean(axis=0)).ravel())

            lfc_pred_global = pred_pert_bulk - pred_ctrl_bulk
            lfc_truth_global = truth_pert_bulk - truth_ctrl_bulk
            
            # LFC Histogram
            axes[2, 0].hist(lfc_pred_global, bins=50, alpha=0.7, label='Pred', density=True)
            axes[2, 0].hist(lfc_truth_global, bins=50, alpha=0.7, label='Truth', density=True)
            axes[2, 0].set_xlabel('Global LFC')
            axes[2, 0].set_ylabel('Density')
            axes[2, 0].set_title('Global LFC Distribution')
            axes[2, 0].legend()
            
            # LFC Correlation Scatter
            axes[2, 1].scatter(lfc_truth_global, lfc_pred_global, alpha=0.1, s=1)
            r, _ = pearsonr(lfc_truth_global, lfc_pred_global)
            axes[2, 1].plot([lfc_truth_global.min(), lfc_truth_global.max()], 
                           [lfc_truth_global.min(), lfc_truth_global.max()], 'r--', label=f'y=x (r={r:.3f})')
            axes[2, 1].set_xlabel('Truth Global LFC')
            axes[2, 1].set_ylabel('Pred Global LFC')
            axes[2, 1].set_title('Per-Gene Global LFC Correlation')
            axes[2, 1].legend()
        except Exception as e:
            print(f"\n⚠️  LFC plot generation failed: {e}")
            axes[2, 0].text(0.5, 0.5, 'LFC Plot Failed', horizontalalignment='center')
            axes[2, 1].text(0.5, 0.5, 'LFC Plot Failed', horizontalalignment='center')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"\n📊 Saved plots to {output_path}")
        else:
            plt.savefig('prediction_debug.png', dpi=150, bbox_inches='tight')
            print(f"\n📊 Saved plots to prediction_debug.png")
        
        plt.close()
        
    except ImportError:
        print("\n⚠️  matplotlib not available, skipping plots")
    except Exception as e:
        print(f"\n⚠️  Plot generation failed: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred', required=True, help='Predictions h5ad')
    parser.add_argument('--truth', required=True, help='Ground truth h5ad')
    parser.add_argument('--config', help='Config yaml (optional, to find control_token)')
    parser.add_argument('--plot', default='debug_plot.png', help='Output plot path')
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("PREDICTION DEBUGGING TOOL (v2 with Metric Proxies)")
    print("="*70)

    # <--- NEW: Load control_token from config ---
    control_token = 'non-targeting' # Default
    if args.config:
        try:
            with open(args.config, 'r') as f:
                cfg = yaml.safe_load(f)
            control_token = cfg.get('data', {}).get('control_token', control_token)
            print(f"\nℹ️  Using control token: '{control_token}' (from config)")
        except Exception as e:
            print(f"\n⚠️  Could not read config {args.config}. Using default control token. Error: {e}")
    else:
        print(f"\nℹ️  Using default control token: '{control_token}' (no config provided)")
    
    # Load data
    print("\n📂 Loading predictions...")
    pred_adata = sc.read_h5ad(args.pred)
    print(f"   Shape: {pred_adata.shape}")
    
    print("\n📂 Loading ground truth...")
    truth_adata = sc.read_h5ad(args.truth)
    print(f"   Shape: {truth_adata.shape}")
    
    # 1. Detect scales
    pred_scale, pred_stats = detect_scale(pred_adata.X, "PREDICTIONS")
    truth_scale, truth_stats = detect_scale(truth_adata.X, "GROUND TRUTH")
    
    # 2. Check format
    check_cell_eval_format(pred_adata, "PREDICTIONS")
    check_cell_eval_format(truth_adata, "GROUND TRUTH")
    
    # 3. Scale mismatch warning
    if pred_scale != truth_scale:
        print("\n" + "!"*70)
        print("🚨 CRITICAL: SCALE MISMATCH DETECTED!")
        print(f"   Predictions: {pred_scale}")
        print(f"   Ground truth: {truth_scale}")
        print("!"*70)
        print("\nThis is the most likely cause of bad metrics.")
        print("Both should be in the SAME scale (e.g., both raw_counts)")
        print("\nTo fix:")
        if pred_scale == "log1p" and truth_scale in ["raw_counts", "normalized_counts"]:
            print("  - Set output_scale: 'counts' in your predict: config")
            print("  - Regenerate predictions")
        elif pred_scale in ["raw_counts", "normalized_counts"] and truth_scale == "log1p":
            print("  - Ground truth is log1p but predictions are counts")
            print("  - Either: convert truth to counts OR set output_scale: 'log1p'")
    else:
        print(f"\n✅ Scales match: both in {pred_scale}")
    
    # 4. Gene space check
    print("\n" + "="*70)
    print("GENE SPACE COMPARISON")
    print("="*70)
    
    pred_genes = set(pred_adata.var_names)
    truth_genes = set(truth_adata.var_names)
    
    print(f"\nPred genes: {len(pred_genes)}")
    print(f"Truth genes: {len(truth_genes)}")
    print(f"Overlap: {len(pred_genes & truth_genes)}")
    
    missing_in_pred = truth_genes - pred_genes
    extra_in_pred = pred_genes - truth_genes
    
    if len(pred_genes & truth_genes) < len(truth_genes):
        print(f"\n⚠️  CRITICAL: Gene space mismatch!")
        print(f"   {len(missing_in_pred)} genes in truth but not in pred")
        print(f"   Examples: {list(missing_in_pred)[:5]}")
        print(f"   {len(extra_in_pred)} genes in pred but not in truth")
        print(f"   Examples: {list(extra_in_pred)[:5]}")
        print("   This will result in a score of 0. Ensure 'gene_order_file' in your predict config matches the truth.")
    else:
        print(f"\n✅ Gene space matches.")
    
    # 5. Gene order check
    if list(pred_adata.var_names) != list(truth_adata.var_names):
        print(f"\n⚠️  CRITICAL: Gene order differs!")
        print(f"   This will cause all metrics to be 0.")
        print(f"   Ensure 'gene_order_file' in your predict config matches the truth gene order.")
    else:
        print(f"\n✅ Gene order matches.")
    
    # 6. Per-target analysis
    compare_per_target(pred_adata, truth_adata, control_token=control_token)
    
    # 7. Plot distributions
    plot_distributions(pred_adata, truth_adata, control_token, args.plot)
    
    # 8. Final summary
    print("\n" + "="*70)
    print("SUMMARY & RECOMMENDATIONS")
    print("="*70)
    
    issues = []
    
    if pred_scale != truth_scale:
        issues.append("CRITICAL: SCALE MISMATCH - Fix output_scale in config.")
    
    if len(missing_in_pred) > 0:
        issues.append(f"CRITICAL: {len(missing_in_pred)} genes missing from predictions. Check gene_order_file.")

    if list(pred_adata.var_names) != list(truth_adata.var_names):
         issues.append(f"CRITICAL: Gene order differs. Check gene_order_file.")
    
    if 'sparsity' in pred_stats and 'sparsity' in truth_stats:
        if pred_stats['sparsity'] < (truth_stats['sparsity'] - 0.1): # <--- More tolerant
            issues.append("Sparsity may be too low - check sampling (topk_keep_only)")
        
        if pred_stats['sparsity'] > (truth_stats['sparsity'] + 0.05): # <--- More tolerant
            issues.append("Sparsity may be too high - check sampling (topk_keep_only)")
    
    if 'mean' in pred_stats and 'mean' in truth_stats:
        lib_ratio = (pred_stats['mean'] * pred_adata.shape[1]) / (truth_stats['mean'] * truth_adata.shape[1] + 1e-8)
        if lib_ratio < 0.5 or lib_ratio > 2.0:
            issues.append(f"Library size differs significantly (ratio: {lib_ratio:.2f}). Check 'depth_column' in predict config.")
    
    if not issues:
        print("\n✅ No critical format/scale/gene issues detected!")
        print("   You can now focus on the metric proxy scores (LFC EMD, Jaccard, Mag Ratio).")
        print("   If those are bad, try tuning:")
        print("   1. `delta_gain`: (in predict config) Tune this first! Try [0.8, 1.0, 1.2, 1.5]")
        print("   2. `lambda_delta` / `lambda_xrec`: (in train config) Affects disentanglement.")
        print("   3. `nb_theta` / `rate_sharpen_beta`: (in predict config) Affects count distribution shape.")

    else:
        print("\n🚨 CRITICAL ISSUES FOUND:")
        for i, issue in enumerate(issues, 1):
            print(f"   {i}. {issue}")
    
    print("\n" + "="*70)
    print("Debug complete!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()