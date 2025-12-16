"""
Language encoder module for processing textual instructions.

This module implements a text embedding model that encodes
natural language instructions into fixed-size embeddings.
"""

import torch
import torch.nn as nn
from typing import Optional


class LanguageEncoder(nn.Module):
    """
    Text embedding model for encoding natural language instructions.
    
    Takes instruction strings as input and produces fixed-size language
    embeddings suitable for fusion with visual representations.
    """
    
    def __init__(self, 
                 vocab_size: int = 10000,
                 embedding_dim: int = 256,
                 hidden_dim: int = 512,
                 max_seq_length: int = 128):
        """
        Initialize the language encoder.
        
        Args:
            vocab_size: Size of the vocabulary
            embedding_dim: Dimension of the output embedding
            hidden_dim: Hidden dimension for the encoder
            max_seq_length: Maximum sequence length for input text
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_seq_length = max_seq_length
        
        # Word embedding layer
        self.word_embedding = nn.Embedding(vocab_size, hidden_dim)
        
        # LSTM for sequence encoding
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, 
                           batch_first=True, 
                           bidirectional=True)
        
        # Projection to embedding dimension
        self.projection = nn.Linear(hidden_dim * 2, embedding_dim)
        
    def forward(self, text_tokens: torch.Tensor) -> torch.Tensor:
        """
        Encode text tokens into a language embedding.
        
        Args:
            text_tokens: Input token indices of shape (batch, seq_length)
            
        Returns:
            Language embedding tensor of shape (batch, embedding_dim)
        """
        # Embed tokens
        embedded = self.word_embedding(text_tokens)
        
        # Encode sequence
        lstm_out, (hidden, _) = self.lstm(embedded)
        
        # Use the last hidden state from both directions
        # Concatenate forward and backward hidden states
        hidden_forward = hidden[0]
        hidden_backward = hidden[1]
        combined_hidden = torch.cat([hidden_forward, hidden_backward], dim=1)
        
        # Project to embedding dimension
        embedding = self.projection(combined_hidden)
        
        return embedding

