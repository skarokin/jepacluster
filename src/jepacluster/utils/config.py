"""
This file contains the configuration management.
"""
from pydantic import BaseModel
import yaml

########################################################
# clustering configuration
########################################################
class UMAPConfig(BaseModel):
    n_components: int
    n_neighbors: int
    min_dist: float

class HDBSCANConfig(BaseModel):
    min_cluster_size: int
    min_samples: int
    prediction_threshold: float

class ClusteringConfig(BaseModel):
    umap: UMAPConfig
    hdbscan: HDBSCANConfig

########################################################
# training configuration
########################################################
class TrainingConfig(BaseModel):
    batch_size: int
    learning_rate: float
    epochs: int
    max_steps: int = 25000
    ema_decay: float
    loss_type: str
    min_freq: int = 2
    max_vocab_size: int = 50000
    weight_decay: float = 0.001
    warmup_epochs: int = 3
    clip_grad_norm: float = 2.0
    use_amp: bool = True
    resume_checkpoint: str | None = None

########################################################
# model configuration
########################################################
class EncoderConfig(BaseModel):
    num_layers: int
    num_heads: int
    dropout: float

class PredictorConfig(BaseModel):
    """Predictor FFN width and depth; JEPAArchitecture clamps these to stay strictly smaller than the encoder."""

    hidden_dim: int
    num_layers: int = 2
    num_heads: int = 4

class ModelConfig(BaseModel):
    name: str
    window_size: int
    latent_dim: int
    encoder: EncoderConfig
    predictor: PredictorConfig
    vocab_size: int | None = None
    tokenizer_path: str | None = None

class Config(BaseModel):
    model: ModelConfig
    training: TrainingConfig
    clustering: ClusteringConfig

    @classmethod
    def validate_config(cls, config_file: str):
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
        return cls(**config)
