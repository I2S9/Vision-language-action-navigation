"""
Vision encoder module for processing RGB images.

This module implements a convolutional neural network that encodes
visual observations into fixed-size embeddings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VisionEncoder(nn.Module):
    """
    Convolutional neural network for encoding RGB images.
    
    Takes RGB images as input and produces fixed-size visual embeddings
    suitable for fusion with language representations.
    """
    
    def __init__(self, 
                 input_channels: int = 3,
                 embedding_dim: int = 256,
                 image_size: tuple = (64, 64)):
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
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Calculate flattened size after convolutions
        # This is a placeholder calculation - adjust based on actual image size
        self.flattened_size = self._calculate_flattened_size()
        
        # Fully connected layer to produce embedding
        self.fc = nn.Linear(self.flattened_size, embedding_dim)
        
    def _calculate_flattened_size(self) -> int:
        """
        Calculate the size of flattened features after convolutions.
        
        Returns:
            Size of flattened feature vector
        """
        # Placeholder calculation - should be computed based on image_size
        # For 64x64 input: after conv layers, approximate size
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, *self.image_size)
            x = F.relu(self.conv1(dummy_input))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            return int(x.numel())
    
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Encode an RGB image into a visual embedding.
        
        Args:
            image: Input RGB image tensor of shape (batch, channels, height, width)
            
        Returns:
            Visual embedding tensor of shape (batch, embedding_dim)
        """
        x = F.relu(self.conv1(image))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

