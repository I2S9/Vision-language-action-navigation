"""
Fusion module for combining vision and language representations.

This module implements a simple and explicit fusion strategy
to combine visual and textual embeddings into a unified representation.

Fusion Strategy:
1. Concatenation: Visual and language embeddings are concatenated along
   the feature dimension to create a combined representation.
2. MLP Processing: A simple multi-layer perceptron processes the concatenated
   features to create a unified embedding space.
3. Output: The fused representation captures both visual and linguistic
   information in a single vector suitable for action prediction.

This approach is simple, interpretable, and effective for vision-language
navigation tasks.
"""

import torch
import torch.nn as nn
from typing import Tuple


class FusionModule(nn.Module):
    """
    Module for fusing vision and language embeddings.
    
    Fusion Strategy:
    - Step 1: Concatenate visual and language embeddings along feature dimension
    - Step 2: Apply a simple MLP (2 layers) with ReLU activations
    - Step 3: Output a unified representation of fixed dimension
    
    The concatenation preserves all information from both modalities,
    and the MLP learns to combine them effectively for downstream tasks.
    
    Example:
        >>> fusion = FusionModule(vision_dim=256, language_dim=384, fused_dim=512)
        >>> visual_emb = torch.randn(4, 256)  # batch_size=4
        >>> language_emb = torch.randn(4, 384)
        >>> fused = fusion(visual_emb, language_emb)
        >>> fused.shape  # (4, 512)
    """
    
    def __init__(self,
                 vision_dim: int = 256,
                 language_dim: int = 384,
                 fused_dim: int = 512,
                 hidden_dim: int = 512):
        """
        Initialize the fusion module.
        
        Args:
            vision_dim: Dimension of vision embeddings (default: 256)
            language_dim: Dimension of language embeddings (default: 384 for all-MiniLM-L6-v2)
            fused_dim: Dimension of the fused representation (default: 512)
            hidden_dim: Hidden dimension for the MLP (default: 512)
        """
        super().__init__()
        self.vision_dim = vision_dim
        self.language_dim = language_dim
        self.fused_dim = fused_dim
        self.hidden_dim = hidden_dim
        
        # Input dimension after concatenation
        input_dim = vision_dim + language_dim
        
        # Simple MLP for fusion
        # Architecture: Linear -> ReLU -> Linear -> ReLU
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
        
        Fusion process:
        1. Concatenate embeddings: [vision_emb || language_emb]
        2. Apply MLP to create unified representation
        
        Args:
            vision_embedding: Visual embedding tensor of shape (batch, vision_dim)
            language_embedding: Language embedding tensor of shape (batch, language_dim)
            
        Returns:
            Fused representation tensor of shape (batch, fused_dim)
            
        Raises:
            ValueError: If input dimensions don't match expected dimensions
        """
        # Verify input dimensions
        batch_size = vision_embedding.shape[0]
        
        if vision_embedding.shape[1] != self.vision_dim:
            raise ValueError(
                f"Vision embedding dimension mismatch: expected {self.vision_dim}, "
                f"got {vision_embedding.shape[1]}"
            )
        
        if language_embedding.shape[1] != self.language_dim:
            raise ValueError(
                f"Language embedding dimension mismatch: expected {self.language_dim}, "
                f"got {language_embedding.shape[1]}"
            )
        
        if vision_embedding.shape[0] != language_embedding.shape[0]:
            raise ValueError(
                f"Batch size mismatch: vision={vision_embedding.shape[0]}, "
                f"language={language_embedding.shape[0]}"
            )
        
        # Step 1: Concatenate vision and language embeddings along feature dimension
        # Shape: (batch, vision_dim) + (batch, language_dim) -> (batch, vision_dim + language_dim)
        combined = torch.cat([vision_embedding, language_embedding], dim=1)
        
        # Step 2: Apply MLP to create fused representation
        # Shape: (batch, input_dim) -> (batch, hidden_dim) -> (batch, fused_dim)
        fused = self.fusion_mlp(combined)
        
        # Verify output dimensions
        assert fused.shape == (batch_size, self.fused_dim), \
            f"Output shape mismatch: expected ({batch_size}, {self.fused_dim}), got {fused.shape}"
        
        return fused

