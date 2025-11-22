"""
Command-line interface for AE-DEVAE.
"""
import argparse
import yaml
from pathlib import Path
from dataclasses import asdict
import sys

from .config import Config, DataConfig, ModelConfig, TrainConfig, PredictConfig
from .train import train
from .predict import predict


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
    
    print(f"Loading config from {config_path}\n")
    
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
  # Train Stage 1 (from scratch)
  python -m devae.cli train --config configs/stage1.yaml
  
  # Train Stage 2 (fine-tuning)
  python -m devae.cli train --config configs/stage2.yaml
  
  # Resume training from checkpoint
  python -m devae.cli train --config configs/stage2.yaml --resume outputs/stage1/checkpoint_epoch050.pt
  
  # Generate competition predictions
  python -m devae.cli predict --config configs/predict.yaml
  
  # Override output path
  python -m devae.cli predict --config configs/predict.yaml --output predictions_final.h5ad
  
  # Use different neighbor mixing parameters
  python -m devae.cli predict --config configs/predict.yaml --neighbor-k 16 --neighbor-tau 0.1
  
  # Scale delta predictions
  python -m devae.cli predict --config configs/predict.yaml --delta-gain 1.5
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # ========================================================================
    # TRAIN COMMAND
    # ========================================================================
    
    train_parser = subparsers.add_parser(
        'train', 
        help='Train model',
        description='Train a new model or fine-tune from pretrained weights'
    )
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
        help='Resume training from checkpoint (path to .pt file)'
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
    train_parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Override number of epochs from config'
    )
    
    # ========================================================================
    # PREDICT COMMAND
    # ========================================================================
    
    predict_parser = subparsers.add_parser(
        'predict', 
        help='Generate competition predictions',
        description='Generate predictions for competition submission using control pool and perturbation list'
    )
    predict_parser.add_argument(
        '--config', 
        type=str, 
        required=True, 
        help='Path to config YAML file (must specify control_pool_h5ad and perturb_list_csv)'
    )
    predict_parser.add_argument(
        '--ckpt', 
        type=str,
        default=None,
        help='Override checkpoint path from config (path to .pt file)'
    )
    predict_parser.add_argument(
        '--output', 
        type=str,
        default=None,
        help='Override output h5ad file path from config'
    )
    predict_parser.add_argument(
        '--control-pool',
        type=str,
        default=None,
        help='Override control pool h5ad path from config'
    )
    predict_parser.add_argument(
        '--perturb-list',
        type=str,
        default=None,
        help='Override perturbation list CSV path from config'
    )
    predict_parser.add_argument(
        '--neighbor-k',
        type=int,
        default=None,
        help='Number of neighbors for mixing (default: from config)'
    )
    predict_parser.add_argument(
        '--neighbor-tau',
        type=float,
        default=None,
        help='Temperature for neighbor mixing (default: from config)'
    )
    predict_parser.add_argument(
        '--delta-gain',
        type=float,
        default=None,
        help='Scale factor for delta predictions (default: 1.0)'
    )
    predict_parser.add_argument(
        '--output-scale',
        type=str,
        choices=['counts', 'log1p'],
        default=None,
        help='Output scale: counts or log1p (default: from config)'
    )
    predict_parser.add_argument(
        '--n-samples',
        type=int,
        default=None,
        help='Number of samples from latent for uncertainty (default: 1)'
    )
    predict_parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Override batch size from config'
    )
    predict_parser.add_argument(
        '--gene-order',
        type=str,
        default=None,
        help='Override gene order file from config (for final reordering)'
    )
    predict_parser.add_argument(
        '--no-ema',
        action='store_true',
        help='Do not use EMA weights (use standard weights instead)'
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
        print("\n" + "="*80)
        print("AE-DEVAE: Attention-Enhanced Dual Encoder VAE for Perturbation Prediction")
        print("="*80 + "\n")
        
        # Load config
        cfg = load_config(args.config)
        
        # Override settings from command line
        if args.resume:
            cfg.train.resume_from = args.resume
        
        if args.output_dir:
            cfg.train.outdir = args.output_dir
        
        if args.seed is not None:
            cfg.train.seed = args.seed
        
        if args.epochs is not None:
            cfg.train.epochs_main = args.epochs
        
        # Save config to output directory
        outdir = Path(cfg.train.outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        save_config(cfg, outdir / "config.yaml")
        
        # Train
        train(cfg)
        
    # ========================================================================
    # PREDICT COMMAND
    # ========================================================================
    
    elif args.command == 'predict':
        print("\n" + "="*80)
        print("AE-DEVAE: Attention-Enhanced Dual Encoder VAE for Perturbation Prediction")
        print("="*80 + "\n")
        
        # Load config
        cfg = load_config(args.config)
        
        # Override settings from command line
        if args.ckpt:
            cfg.predict.ckpt_path = args.ckpt
        
        if args.output:
            cfg.predict.out_h5ad = args.output
        
        if args.control_pool:
            cfg.data.control_pool_h5ad = args.control_pool
        
        if args.perturb_list:
            cfg.predict.perturb_list_csv = args.perturb_list
        
        if args.neighbor_k is not None:
            cfg.predict.neighbor_mix_k = args.neighbor_k
        
        if args.neighbor_tau is not None:
            cfg.predict.neighbor_mix_tau = args.neighbor_tau
        
        if args.delta_gain is not None:
            cfg.predict.delta_gain = args.delta_gain
        
        if args.output_scale:
            cfg.predict.output_scale = args.output_scale
        
        if args.n_samples is not None:
            cfg.predict.n_samples = args.n_samples
        
        if args.batch_size is not None:
            cfg.predict.batch_size = args.batch_size
        
        if args.gene_order:
            cfg.predict.gene_order_file = args.gene_order
        
        if args.no_ema:
            cfg.predict.use_ema = False
        
        # Validate required fields
        if not Path(cfg.predict.ckpt_path).exists():
            print(f"[ERROR] Checkpoint not found: {cfg.predict.ckpt_path}")
            sys.exit(1)
        
        if not Path(cfg.data.control_pool_h5ad).exists():
            print(f"[ERROR] Control pool not found: {cfg.data.control_pool_h5ad}")
            sys.exit(1)
        
        if not Path(cfg.predict.perturb_list_csv).exists():
            print(f"[ERROR] Perturbation list not found: {cfg.predict.perturb_list_csv}")
            sys.exit(1)
        
        # Run prediction
        predict(cfg)
        
if __name__ == '__main__':
    main()