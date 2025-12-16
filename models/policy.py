"""
Policy network for navigation decision-making.

This module implements the policy network that takes fused
vision-language representations and outputs discrete action logits.
"""

import torch
import torch.nn as nn


class PolicyNetwork(nn.Module):
    """
    Policy network for navigation actions.
    
    Takes fused vision-language embeddings as input and outputs
    discrete action logits for navigation decisions.
    """
    
    def __init__(self,
                 input_dim: int = 512,
                 hidden_dim: int = 256,
                 num_actions: int = 4):
        """
        Initialize the policy network.
        
        Args:
            input_dim: Dimension of the fused input representation
            hidden_dim: Hidden dimension for the policy network
            num_actions: Number of discrete actions (e.g., forward, left, right, stop)
        """
        super().__init__()
        self.num_actions = num_actions
        
        # Policy network layers
        self.policy_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions)
        )
        
    def forward(self, fused_embedding: torch.Tensor) -> torch.Tensor:
        """
        Compute action logits from fused embedding.
        
        Args:
            fused_embedding: Fused vision-language embedding of shape (batch, input_dim)
            
        Returns:
            Action logits tensor of shape (batch, num_actions)
        """
        logits = self.policy_mlp(fused_embedding)
        return logits

