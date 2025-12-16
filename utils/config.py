"""
Configuration management for the VLA navigation system.

This module provides a centralized configuration system
for hyperparameters and experiment settings.
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class VisionConfig:
    """Configuration for the vision encoder."""
    input_channels: int = 3
    embedding_dim: int = 256
    image_size: Tuple[int, int] = (64, 64)


@dataclass
class LanguageConfig:
    """Configuration for the language encoder."""
    vocab_size: int = 10000
    embedding_dim: int = 256
    hidden_dim: int = 512
    max_seq_length: int = 128


@dataclass
class FusionConfig:
    """Configuration for the fusion module."""
    vision_dim: int = 256
    language_dim: int = 256
    fused_dim: int = 512
    hidden_dim: int = 512


@dataclass
class PolicyConfig:
    """Configuration for the policy network."""
    input_dim: int = 512
    hidden_dim: int = 256
    num_actions: int = 4


@dataclass
class TrainingConfig:
    """Configuration for training."""
    learning_rate: float = 1e-4
    batch_size: int = 32
    num_epochs: int = 100
    device: str = "cpu"
    seed: int = 42


@dataclass
class Config:
    """Main configuration class combining all sub-configs."""
    vision: VisionConfig = None
    language: LanguageConfig = None
    fusion: FusionConfig = None
    policy: PolicyConfig = None
    training: TrainingConfig = None
    
    def __post_init__(self):
        """Initialize default sub-configs if not provided."""
        if self.vision is None:
            self.vision = VisionConfig()
        if self.language is None:
            self.language = LanguageConfig()
        if self.fusion is None:
            self.fusion = FusionConfig()
        if self.policy is None:
            self.policy = PolicyConfig()
        if self.training is None:
            self.training = TrainingConfig()

