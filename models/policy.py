"""
Policy network for navigation decision-making.

This module implements the policy network that takes fused
vision-language representations and outputs discrete action logits.

The policy network maps the fused vision-language embedding to action logits,
which can be used to select discrete navigation actions.
"""

import torch
import torch.nn as nn
from typing import Dict


class PolicyNetwork(nn.Module):
    """
    Policy network for navigation actions.
    
    Takes fused vision-language embeddings as input and outputs
    discrete action logits for navigation decisions.
    
    Architecture:
    - Simple MLP with 2 hidden layers
    - Input: fused vision-language embedding
    - Output: action logits (one logit per action)
    
    Action Space (MiniGrid standard):
    - Action 0: Turn Left
    - Action 1: Turn Right
    - Action 2: Move Forward
    - Action 3: Pick Up / Drop / Toggle / Done (context-dependent)
    
    The action with the highest logit is typically selected using argmax.
    """
    
    # Action space definition for MiniGrid
    ACTION_SPACE: Dict[int, str] = {
        0: "Turn Left",
        1: "Turn Right",
        2: "Move Forward",
        3: "Pick Up / Drop / Toggle / Done"
    }
    
    def __init__(self,
                 input_dim: int = 512,
                 hidden_dim: int = 256,
                 num_actions: int = 4):
        """
        Initialize the policy network.
        
        Args:
            input_dim: Dimension of the fused input representation (default: 512)
            hidden_dim: Hidden dimension for the policy network (default: 256)
            num_actions: Number of discrete actions (default: 4 for MiniGrid)
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_actions = num_actions
        
        # Simple MLP for action prediction
        # Architecture: Linear -> ReLU -> Linear -> ReLU -> Linear (logits)
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
            These are raw logits (not probabilities) suitable for argmax or softmax.
            
        Raises:
            ValueError: If input dimension doesn't match expected dimension
        """
        # Verify input dimension
        if fused_embedding.shape[1] != self.input_dim:
            raise ValueError(
                f"Input dimension mismatch: expected {self.input_dim}, "
                f"got {fused_embedding.shape[1]}"
            )
        
        # Compute action logits through MLP
        action_logits = self.policy_mlp(fused_embedding)
        
        # Verify output dimension
        batch_size = fused_embedding.shape[0]
        assert action_logits.shape == (batch_size, self.num_actions), \
            f"Output shape mismatch: expected ({batch_size}, {self.num_actions}), " \
            f"got {action_logits.shape}"
        
        return action_logits
    
    def get_action_name(self, action: int) -> str:
        """
        Get the name of an action by its index.
        
        Args:
            action: Action index
            
        Returns:
            Action name string
        """
        return self.ACTION_SPACE.get(action, f"Unknown action {action}")
    
    def select_action(self, fused_embedding: torch.Tensor, method: str = "argmax") -> torch.Tensor:
        """
        Select an action from fused embedding.
        
        Args:
            fused_embedding: Fused vision-language embedding
            method: Selection method ("argmax" for greedy, "sample" for sampling)
            
        Returns:
            Selected action indices of shape (batch,)
        """
        action_logits = self.forward(fused_embedding)
        
        if method == "argmax":
            # Greedy selection: choose action with highest logit
            actions = torch.argmax(action_logits, dim=1)
        elif method == "sample":
            # Sample from softmax distribution
            probs = torch.softmax(action_logits, dim=1)
            actions = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            raise ValueError(f"Unknown selection method: {method}")
        
        return actions

