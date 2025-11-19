"""
Command-line interface for AE-DEVAE.
Enhanced with better argument handling and validation
"""
import argparse
import yaml
from pathlib import Path
from dataclasses import asdict
import sys
import numpy as np
import scanpy as sc

from .config import Config, DataConfig, ModelConfig, TrainConfig, PredictConfig
from .train import train
from .predict import predict, predict_with_uncertainty


def load_config(config_path: str) -> Config:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
    
    Returns:
        Config object
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    print(f"\nLoading config from {config_path}\n")
    
    with open(config_path, 'r') as f:
        cfg_dict = yaml.safe_load(f)
    
    # Parse into dataclasses with error handling
    try:
        data_cfg = DataConfig(**cfg_dict.get('data', {}))
        model_cfg = ModelConfig(**cfg_dict.get('model', {}))
        train_cfg = TrainConfig(**cfg_dict.get('train', {}))
        predict_cfg = PredictConfig(**cfg_dict.get('predict', {}))
    except TypeError as e:
        print(f"[ERROR] Invalid config format: {e}")
        print("[ERROR] Check that all config fields match the dataclass definitions")
        sys.exit(1)
    
    return Config(
        data=data_cfg,
        model=model_cfg,
        train=train_cfg,
        predict=predict_cfg
    )


def save_config(cfg: Config, output_path: str):
    """
    Save configuration to YAML file.
    
    Args:
        cfg: Configuration object
        output_path: Path to save YAML file
    """
    cfg_dict = {
        'data': asdict(cfg.data),
        'model': asdict(cfg.model),
        'train': asdict(cfg.train),
        'predict': asdict(cfg.predict),
    }
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        yaml.dump(cfg_dict, f, default_flow_style=False, sort_keys=False)

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="AE-DEVAE: Attention-Enhanced Dual Encoder VAE for Perturbation Prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train a model
  python -m ae_devae.cli train --config configs/stage1.yaml
  
  # Resume training
  python -m ae_devae.cli train --config configs/stage2.yaml --resume outputs/stage1/best_model.pt
  
  # Generate predictions
  python -m ae_devae.cli predict --config configs/predict.yaml --ckpt outputs/stage2/best_model.pt
  
  # Predict with neighbor mixing (for unseen genes)
  python -m ae_devae.cli predict --config configs/predict.yaml --ckpt outputs/stage2/best_model.pt --neighbor-mixing
  
  # Predict with uncertainty estimation
  python -m ae_devae.cli predict --config configs/predict.yaml --ckpt outputs/stage2/best_model.pt --uncertainty --n-samples 10
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # ========================================================================
    # TRAIN COMMAND
    # ========================================================================
    
    train_parser = subparsers.add_parser('train', help='Train model')
    train_parser.add_argument(
        '--config', 
        type=str, 
        required=True, 
        help='Path to config YAML file'
    )
    train_parser.add_argument(
        '--resume', 
        type=str, 
        default=None, 
        help='Resume from checkpoint (path to .pt file)'
    )
    train_parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Override output directory from config'
    )
    train_parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Override random seed from config'
    )
    
    # ========================================================================
    # PREDICT COMMAND
    # ========================================================================
    
    predict_parser = subparsers.add_parser('predict', help='Generate predictions')
    predict_parser.add_argument(
        '--config', 
        type=str, 
        required=True, 
        help='Path to config YAML file'
    )
    predict_parser.add_argument(
        '--ckpt', 
        type=str, 
        required=True, 
        help='Path to model checkpoint (.pt file)'
    )
    predict_parser.add_argument(
        '--output', 
        type=str, 
        default='predictions.h5ad', 
        help='Output h5ad file path'
    )
    predict_parser.add_argument(
        '--neighbor-mixing',
        action='store_true',
        help='Enable neighbor mixing (critical for unseen genes!)'
    )
    predict_parser.add_argument(
        '--neighbor-k',
        type=int,
        default=None,
        help='Number of neighbors for mixing (default: from config)'
    )
    predict_parser.add_argument(
        '--delta-gain',
        type=float,
        default=None,
        help='Scale factor for delta predictions (default: 1.0)'
    )
    predict_parser.add_argument(
        '--uncertainty',
        action='store_true',
        help='Compute prediction uncertainty with multiple samples'
    )
    predict_parser.add_argument(
        '--n-samples',
        type=int,
        default=10,
        help='Number of samples for uncertainty estimation (default: 10)'
    )
    predict_parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Override batch size from config'
    )
    
    # ========================================================================
    # PARSE ARGS
    # ========================================================================
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(0)
    
    # ========================================================================
    # TRAIN COMMAND
    # ========================================================================
    
    if args.command == 'train':
        print("="*80)
        print("AE-DEVAE: Attention-Enhanced Dual Encoder VAE for Perturbation Prediction")
        print("="*80)
        
        # Load config
        cfg = load_config(args.config)
        
        # Override settings from command line
        if args.resume:
            cfg.train.resume_from = args.resume
            print(f"Resuming from: {args.resume}")
        
        if args.output_dir:
            cfg.train.outdir = args.output_dir
        
        if args.seed is not None:
            cfg.train.seed = args.seed
        
        # Save config to output directory
        outdir = Path(cfg.train.outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        save_config(cfg, outdir / "config.yaml")
        
        train(cfg)        
        
    # ========================================================================
    # PREDICT COMMAND
    # ========================================================================
    
    elif args.command == 'predict':
        print("="*80)
        print("AE-DEVAE: Attention-Enhanced Dual Encoder VAE for Perturbation Prediction")
        print("="*80)
        
        # Load config
        cfg = load_config(args.config)
        
        # Override settings from command line
        cfg.predict.ckpt_path = args.ckpt
        cfg.predict.output_h5ad = args.output
        
        if args.neighbor_mixing:
            cfg.predict.use_neighbor_mixing = True
        
        if args.neighbor_k is not None:
            cfg.predict.neighbor_mix_k = args.neighbor_k
        
        if args.delta_gain is not None:
            cfg.predict.delta_gain = args.delta_gain
        
        if args.batch_size is not None:
            cfg.predict.batch_size = args.batch_size
                
        # Uncertainty estimation mode
        if args.uncertainty:
            
            mean_pred, std_pred, all_samples = predict_with_uncertainty(cfg, n_samples=args.n_samples)
            
            # Save results
            adata_test = sc.read_h5ad(cfg.predict.test_h5ad)
            
            # Mean predictions
            adata_mean = sc.AnnData(
                X=mean_pred,
                obs=adata_test.obs.copy(),
                var=adata_test.var.copy()
            )
            adata_mean.uns['prediction_type'] = 'mean'
            adata_mean.uns['n_samples'] = args.n_samples
            
            # Uncertainty (std)
            adata_std = sc.AnnData(
                X=std_pred,
                obs=adata_test.obs.copy(),
                var=adata_test.var.copy()
            )
            adata_std.uns['prediction_type'] = 'uncertainty'
            
            # Save
            output_base = Path(args.output).stem
            output_dir = Path(args.output).parent
            
            adata_mean.write_h5ad(output_dir / f"{output_base}_mean.h5ad")
            adata_std.write_h5ad(output_dir / f"{output_base}_uncertainty.h5ad")
        else:
            # Standard prediction
            predict(cfg)

if __name__ == '__main__':
    main()