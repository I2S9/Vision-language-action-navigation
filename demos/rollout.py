"""
Rollout script for visualizing agent behavior.

This module provides functionality to run the trained agent
in the environment and record or visualize its behavior.
"""

import torch
import numpy as np
from typing import List, Tuple, Dict, Any

from models.vision_encoder import VisionEncoder
from models.language_encoder import LanguageEncoder
from models.fusion import FusionModule
from models.policy import PolicyNetwork


class RolloutRunner:
    """
    Runner for agent rollouts in the environment.
    
    Executes the agent policy and collects trajectories
    for visualization or analysis.
    """
    
    def __init__(self,
                 vision_encoder: VisionEncoder,
                 language_encoder: LanguageEncoder,
                 fusion_module: FusionModule,
                 policy_network: PolicyNetwork,
                 device: str = "cpu"):
        """
        Initialize the rollout runner.
        
        Args:
            vision_encoder: Vision encoder model
            language_encoder: Language encoder model
            fusion_module: Fusion module
            policy_network: Policy network
            device: Device to run inference on
        """
        self.vision_encoder = vision_encoder.to(device)
        self.language_encoder = language_encoder.to(device)
        self.fusion_module = fusion_module.to(device)
        self.policy_network = policy_network.to(device)
        
        self.device = device
        
        # Set models to evaluation mode
        self.vision_encoder.eval()
        self.language_encoder.eval()
        self.fusion_module.eval()
        self.policy_network.eval()
        
    def select_action(self,
                     image: np.ndarray,
                     text_tokens: torch.Tensor) -> int:
        """
        Select an action given an observation and instruction.
        
        Args:
            image: Current RGB image observation
            text_tokens: Text instruction tokens
            
        Returns:
            Selected action index
        """
        with torch.no_grad():
            # Convert image to tensor
            image_tensor = torch.FloatTensor(image).unsqueeze(0).to(self.device)
            if len(image_tensor.shape) == 3:
                image_tensor = image_tensor.unsqueeze(0)
            
            # Encode vision and language
            vision_emb = self.vision_encoder(image_tensor)
            language_emb = self.language_encoder(text_tokens.unsqueeze(0).to(self.device))
            
            # Fuse representations
            fused_emb = self.fusion_module(vision_emb, language_emb)
            
            # Get action logits
            action_logits = self.policy_network(fused_emb)
            
            # Select action (greedy for now)
            action = torch.argmax(action_logits, dim=1).item()
            
            return action
    
    def run_rollout(self,
                   env,
                   instruction: str,
                   text_tokens: torch.Tensor,
                   max_steps: int = 1000) -> Dict[str, Any]:
        """
        Run a single rollout episode.
        
        Args:
            env: Environment instance
            instruction: Natural language instruction
            text_tokens: Tokenized instruction
            max_steps: Maximum number of steps in the episode
            
        Returns:
            Dictionary with rollout trajectory and metadata
        """
        obs, info = env.reset()
        trajectory = []
        
        for step in range(max_steps):
            action = self.select_action(obs, text_tokens)
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            trajectory.append({
                "observation": obs,
                "action": action,
                "reward": reward
            })
            
            obs = next_obs
            
            if terminated or truncated:
                break
        
        return {
            "trajectory": trajectory,
            "episode_length": len(trajectory),
            "instruction": instruction
        }

