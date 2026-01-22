"""Configuration Management"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """Model configuration"""
    vocab_size: int = 50257
    embed_dim: int = 768
    num_layers: int = 12
    num_heads: int = 12
    ffn_dim: int = 3072
    max_seq_len: int = 1024
    dropout: float = 0.1
    layer_norm_eps: float = 1e-5
    bias: bool = True


@dataclass
class TrainingConfig:
    """Training configuration"""
    batch_size: int = 8
    gradient_accumulation_steps: int = 1
    learning_rate: float = 6e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    max_epochs: int = 10
    warmup_steps: int = 2000
    max_grad_norm: float = 1.0
    use_mixed_precision: bool = True
    eval_interval: int = 500
    save_interval: int = 1000
    log_interval: int = 10
    resume_from: Optional[str] = None


@dataclass
class DataConfig:
    """Data configuration"""
    data_dir: str = "./data"
    max_length: int = 1024
    num_workers: int = 4
    pin_memory: bool = True


@dataclass
class WandBConfig:
    """Weights & Biases configuration"""
    project: str = "gpt2-training"
    entity: Optional[str] = None
    name: Optional[str] = None
    tags: list = field(default_factory=list)
    enabled: bool = True


@dataclass
class Config:
    """Main configuration class"""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    wandb: WandBConfig = field(default_factory=WandBConfig)
    output_dir: str = "./outputs"
    seed: int = 42
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """Create Config from dictionary"""
        config = cls()
        
        if 'model' in config_dict:
            config.model = ModelConfig(**config_dict['model'])
        if 'training' in config_dict:
            config.training = TrainingConfig(**config_dict['training'])
        if 'data' in config_dict:
            config.data = DataConfig(**config_dict['data'])
        if 'wandb' in config_dict:
            config.wandb = WandBConfig(**config_dict['wandb'])
        if 'output_dir' in config_dict:
            config.output_dir = config_dict['output_dir']
        if 'seed' in config_dict:
            config.seed = config_dict['seed']
        
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert Config to dictionary"""
        return {
            'model': self.model.__dict__,
            'training': self.training.__dict__,
            'data': self.data.__dict__,
            'wandb': self.wandb.__dict__,
            'output_dir': self.output_dir,
            'seed': self.seed
        }


def load_config(config_path: str) -> Config:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Config object
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    return Config.from_dict(config_dict)
