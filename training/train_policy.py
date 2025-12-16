"""
Training script for the vision-language-action policy.

This module contains the training loop and optimization logic
for learning the navigation policy from visual and language inputs.
"""

import torch
import torch.optim as optim
from typing import Dict, Any
import numpy as np

from models.vision_encoder import VisionEncoder
from models.language_encoder import LanguageEncoder
from models.fusion import FusionModule
from models.policy import PolicyNetwork


class PolicyTrainer:
    """
    Trainer for the VLA navigation policy.
    
    Handles the training loop, loss computation, and optimization
    for the complete vision-language-action system.
    """
    
    def __init__(self,
                 vision_encoder: VisionEncoder,
                 language_encoder: LanguageEncoder,
                 fusion_module: FusionModule,
                 policy_network: PolicyNetwork,
                 learning_rate: float = 1e-4,
                 device: str = "cpu"):
        """
        Initialize the policy trainer.
        
        Args:
            vision_encoder: Vision encoder model
            language_encoder: Language encoder model
            fusion_module: Fusion module
            policy_network: Policy network
            learning_rate: Learning rate for optimization
            device: Device to run training on
        """
        self.vision_encoder = vision_encoder.to(device)
        self.language_encoder = language_encoder.to(device)
        self.fusion_module = fusion_module.to(device)
        self.policy_network = policy_network.to(device)
        
        self.device = device
        
        # Combine all parameters for optimization
        all_params = list(vision_encoder.parameters()) + \
                    list(language_encoder.parameters()) + \
                    list(fusion_module.parameters()) + \
                    list(policy_network.parameters())
        
        self.optimizer = optim.Adam(all_params, lr=learning_rate)
        
    def train_step(self,
                   images: torch.Tensor,
                   text_tokens: torch.Tensor,
                   actions: torch.Tensor) -> Dict[str, float]:
        """
        Perform one training step.
        
        Args:
            images: Batch of RGB images
            text_tokens: Batch of text token sequences
            actions: Batch of target actions
            
        Returns:
            Dictionary with loss and other metrics
        """
        self.vision_encoder.train()
        self.language_encoder.train()
        self.fusion_module.train()
        self.policy_network.train()
        
        # Forward pass
        vision_emb = self.vision_encoder(images)
        language_emb = self.language_encoder(text_tokens)
        fused_emb = self.fusion_module(vision_emb, language_emb)
        action_logits = self.policy_network(fused_emb)
        
        # Compute loss
        loss = torch.nn.functional.cross_entropy(action_logits, actions)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return {
            "loss": loss.item()
        }

