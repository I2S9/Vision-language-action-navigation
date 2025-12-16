"""
Fusion module for combining vision and language representations.

This module implements a simple and explicit fusion strategy
to combine visual and textual embeddings into a unified representation.
"""

import torch
import torch.nn as nn


class FusionModule(nn.Module):
    """
    Module for fusing vision and language embeddings.
    
    Uses concatenation followed by a multi-layer perceptron
    to create a unified representation from visual and textual inputs.
    """
    
    def __init__(self,
                 vision_dim: int = 256,
                 language_dim: int = 256,
                 fused_dim: int = 512,
                 hidden_dim: int = 512):
        """
        Initialize the fusion module.
        
        Args:
            vision_dim: Dimension of vision embeddings
            language_dim: Dimension of language embeddings
            fused_dim: Dimension of the fused representation
            hidden_dim: Hidden dimension for the MLP
        """
        super().__init__()
        input_dim = vision_dim + language_dim
        
        # Multi-layer perceptron for fusion
        self.fusion_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, fused_dim),
            nn.ReLU()
        )
        
    def forward(self, 
                vision_embedding: torch.Tensor,
                language_embedding: torch.Tensor) -> torch.Tensor:
        """
        Fuse vision and language embeddings.
        
        Args:
            vision_embedding: Visual embedding tensor of shape (batch, vision_dim)
            language_embedding: Language embedding tensor of shape (batch, language_dim)
            
        Returns:
            Fused representation tensor of shape (batch, fused_dim)
        """
        # Concatenate vision and language embeddings
        combined = torch.cat([vision_embedding, language_embedding], dim=1)
        
        # Apply MLP to create fused representation
        fused = self.fusion_mlp(combined)
        
        return fused

