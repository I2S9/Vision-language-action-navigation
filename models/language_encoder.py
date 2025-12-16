"""
Language encoder module for processing textual instructions.

This module uses a pre-trained sentence encoder to convert natural language
instructions into fixed-size embeddings.
"""

import torch
import torch.nn as nn
from typing import Union, List
from sentence_transformers import SentenceTransformer


class LanguageEncoder(nn.Module):
    """
    Pre-trained sentence encoder for encoding natural language instructions.
    
    Uses a pre-trained sentence transformer model to convert text instructions
    into fixed-size embeddings. The pre-trained weights are frozen by default.
    
    Features:
    - Pre-trained sentence transformer (frozen weights)
    - Normalized embeddings
    - Simple interface: takes string instructions directly
    """
    
    def __init__(self, 
                 model_name: str = "all-MiniLM-L6-v2",
                 embedding_dim: int = 384,
                 freeze_weights: bool = True,
                 normalize: bool = True):
        """
        Initialize the language encoder.
        
        Args:
            model_name: Name of the pre-trained sentence transformer model
            embedding_dim: Dimension of the output embedding (model-dependent)
            freeze_weights: Whether to freeze the pre-trained model weights
            normalize: Whether to normalize the embeddings to unit length
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.normalize = normalize
        
        # Load pre-trained sentence transformer
        self.sentence_model = SentenceTransformer(model_name)
        
        # Get actual embedding dimension from the model
        # Test with a dummy sentence to get the output dimension
        with torch.no_grad():
            test_embedding = self.sentence_model.encode(["test"], convert_to_tensor=True)
            self.embedding_dim = test_embedding.shape[1]
        
        # Freeze pre-trained weights if requested
        if freeze_weights:
            for param in self.sentence_model.parameters():
                param.requires_grad = False
            self.sentence_model.eval()
        
        # Optional projection layer if we need a different embedding dimension
        # For now, we use the model's native dimension
        self.projection = None
        if embedding_dim != self.embedding_dim:
            self.projection = nn.Linear(self.embedding_dim, embedding_dim)
            self.embedding_dim = embedding_dim
    
    def forward(self, instructions: Union[str, List[str]]) -> torch.Tensor:
        """
        Encode text instructions into language embeddings.
        
        Args:
            instructions: Single instruction string or list of instruction strings
            
        Returns:
            Language embedding tensor of shape (batch, embedding_dim)
            If single string, batch=1
        """
        # Handle single string input
        if isinstance(instructions, str):
            instructions = [instructions]
        
        # Encode instructions using pre-trained model
        # If weights are frozen, use no_grad for efficiency
        if not any(p.requires_grad for p in self.sentence_model.parameters()):
            # All weights frozen, use no_grad
            with torch.no_grad():
                embeddings = self.sentence_model.encode(
                    instructions,
                    convert_to_tensor=True,
                    normalize_embeddings=False  # We'll normalize manually after projection
                )
        else:
            # Some weights trainable, enable gradients
            embeddings = self.sentence_model.encode(
                instructions,
                convert_to_tensor=True,
                normalize_embeddings=False  # We'll normalize manually after projection
            )
        
        # Apply projection if needed
        if self.projection is not None:
            embeddings = self.projection(embeddings)
        
        # Normalize embeddings to unit length
        if self.normalize:
            embeddings = nn.functional.normalize(embeddings, p=2, dim=1)
        
        return embeddings
    
    def encode(self, instruction: Union[str, List[str]]) -> torch.Tensor:
        """
        Alias for forward method for more explicit interface.
        
        Args:
            instruction: Single instruction string or list of instruction strings
            
        Returns:
            Language embedding tensor of shape (batch, embedding_dim)
        """
        return self.forward(instruction)

