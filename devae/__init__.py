"""
AE-DEVAE: Attention-Enhanced Dual Encoder VAE for Perturbation Prediction:
- ResidualMLP architecture for better gradient flow
- Neighbor mixing for unseen gene generalization
- CosineWithWarmup scheduler
- Real count usage for accurate likelihoods
- Vocabulary mapping persistence
- Uncertainty estimation
"""

__version__ = "1.0.0"
__author__ = "Andrea Tangherloni"

from .config import Config, DataConfig, ModelConfig, TrainConfig, PredictConfig
from .vae import AttentionEnhancedDualEncoderVAE
from .train import train
from .predict import predict, predict_with_uncertainty
from .data import PerturbationDataset, get_dataloader
from .modules import (
    ResidualMLP,
    CosineWithWarmup,
    GenePerturbationAttention,
    AttentionDeltaPredictor,
    AttentionCountHead,
    FiLMLayer
)

__all__ = [
    # Configuration
    'Config',
    'DataConfig',
    'ModelConfig',
    'TrainConfig',
    'PredictConfig',
    
    # Main model
    'AttentionEnhancedDualEncoderVAE',
    
    # Training and prediction
    'train',
    'predict',
    'predict_with_uncertainty',
    
    # Data handling
    'PerturbationDataset',
    'get_dataloader',
    
    # Modules and components
    'ResidualMLP',
    'CosineWithWarmup',
    'GenePerturbationAttention',
    'AttentionDeltaPredictor',
    'AttentionCountHead',
    'FiLMLayer',
]