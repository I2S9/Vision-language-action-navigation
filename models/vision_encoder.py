"""
Vision encoder module for processing RGB images.

This module implements a convolutional neural network that encodes
visual observations into fixed-size embeddings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class VisionEncoder(nn.Module):
    """
    Simple convolutional neural network for encoding RGB images.
    
    Takes RGB images as input, normalizes them, and produces fixed-size
    visual embeddings suitable for fusion with language representations.
    
    Architecture:
    - Input normalization (uint8 [0, 255] -> float [0, 1])
    - Three convolutional layers with ReLU activations
    - Flattening and fully connected layer
    - Output: fixed-size embedding vector
    """
    
    def __init__(self, 
                 input_channels: int = 3,
                 embedding_dim: int = 256,
                 image_size: Tuple[int, int] = (64, 64)):
        """
        Initialize the vision encoder.
        
        Args:
            input_channels: Number of input channels (3 for RGB)
            embedding_dim: Dimension of the output embedding
            image_size: Expected input image size (height, width)
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.image_size = image_size
        self.input_channels = input_channels
        
        # Simple CNN architecture
        # Layer 1: 8x8 kernel, stride 4 -> reduces size by 4
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4, padding=0)
        
        # Layer 2: 4x4 kernel, stride 2 -> reduces size by 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0)
        
        # Layer 3: 3x3 kernel, stride 1 -> maintains size
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)
        
        # Calculate flattened size after convolutions
        self.flattened_size = self._calculate_flattened_size()
        
        # Fully connected layer to produce fixed-size embedding
        self.fc = nn.Linear(self.flattened_size, embedding_dim)
        
    def _calculate_flattened_size(self) -> int:
        """
        Calculate the size of flattened features after convolutions.
        
        Uses a dummy forward pass to determine the exact size.
        
        Returns:
            Size of flattened feature vector
        """
        with torch.no_grad():
            # Create dummy input with batch size 1
            dummy_input = torch.zeros(1, self.input_channels, *self.image_size)
            
            # Forward through conv layers
            x = F.relu(self.conv1(dummy_input))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            
            # Calculate flattened size
            flattened_size = int(x.view(1, -1).size(1))
            
        return flattened_size
    
    def _normalize_image(self, image: torch.Tensor) -> torch.Tensor:
        """
        Normalize input image from uint8 [0, 255] to float [0, 1].
        
        Args:
            image: Input image tensor (can be uint8 or float)
            
        Returns:
            Normalized image tensor in range [0, 1]
        """
        # If image is uint8, convert to float and normalize
        if image.dtype == torch.uint8:
            image = image.float() / 255.0
        # If image is already float but in range [0, 255], normalize
        elif image.dtype == torch.float32 or image.dtype == torch.float64:
            if image.max() > 1.0:
                image = image / 255.0
        
        # Ensure values are in [0, 1] range
        image = torch.clamp(image, 0.0, 1.0)
        
        return image
    
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Encode an RGB image into a visual embedding.
        
        Args:
            image: Input RGB image tensor
                - Shape: (batch, channels, height, width) or (batch, height, width, channels)
                - Type: uint8 [0, 255] or float [0, 1] or [0, 255]
        
        Returns:
            Visual embedding tensor of shape (batch, embedding_dim)
        """
        # Handle different input formats
        if len(image.shape) == 3:
            # Single image: (height, width, channels) -> (1, channels, height, width)
            image = image.unsqueeze(0)
            if image.shape[-1] == 3:
                # Channels last -> channels first
                image = image.permute(0, 3, 1, 2)
        elif len(image.shape) == 4:
            # Batch of images: check if channels are last
            if image.shape[-1] == 3 or image.shape[-1] == 1:
                # (batch, height, width, channels) -> (batch, channels, height, width)
                image = image.permute(0, 3, 1, 2)
        
        # Verify input shape
        if image.shape[1] != self.input_channels:
            raise ValueError(
                f"Expected {self.input_channels} channels, got {image.shape[1]}. "
                f"Input shape: {image.shape}"
            )
        
        # Normalize image to [0, 1]
        image = self._normalize_image(image)
        
        # Convolutional layers with ReLU activation
        x = F.relu(self.conv1(image))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Flatten feature maps
        x = x.reshape(x.size(0), -1)
        
        # Fully connected layer to produce embedding
        visual_embedding = self.fc(x)
        
        # Verify output dimensions
        assert visual_embedding.shape[1] == self.embedding_dim, \
            f"Output dimension mismatch: expected {self.embedding_dim}, got {visual_embedding.shape[1]}"
        
        return visual_embedding

