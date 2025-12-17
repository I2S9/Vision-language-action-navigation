"""
Evaluation script for the vision-language-action navigation system.

This module contains evaluation metrics and protocols for assessing
the performance of the navigation policy on unseen environments.
"""

import torch
import numpy as np
from typing import Dict, List, Any, Optional
import json
from pathlib import Path
from datetime import datetime

from models.vision_encoder import VisionEncoder
from models.language_encoder import LanguageEncoder
from models.fusion import FusionModule
from models.policy import PolicyNetwork
from env.environment import create_navigation_env


class PolicyEvaluator:
    """
    Evaluator for the VLA navigation policy.
    
    Tests the agent on unseen environments and computes success rate
    and other metrics. Records trajectories for analysis.
    """
    
    def __init__(self,
                 vision_encoder: VisionEncoder,
                 language_encoder: LanguageEncoder,
                 fusion_module: FusionModule,
                 policy_network: PolicyNetwork,
                 device: str = "cpu"):
        """
        Initialize the policy evaluator.
        
        Args:
            vision_encoder: Vision encoder model
            language_encoder: Language encoder model
            fusion_module: Fusion module
            policy_network: Policy network
            device: Device to run evaluation on
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
    
    def select_action(self, image: np.ndarray, instruction: str) -> int:
        """
        Select an action given an observation and instruction.
        
        Args:
            image: Current RGB image observation
            instruction: Natural language instruction
            
        Returns:
            Selected action index
        """
        with torch.no_grad():
            # Convert image to tensor
            if isinstance(image, np.ndarray):
                image_tensor = torch.from_numpy(image).float()
            else:
                image_tensor = image.float()
            
            # Ensure correct shape (batch, channels, height, width)
            if len(image_tensor.shape) == 3:
                image_tensor = image_tensor.unsqueeze(0)
            if image_tensor.shape[-1] == 3:
                image_tensor = image_tensor.permute(0, 3, 1, 2)
            
            image_tensor = image_tensor.to(self.device)
            
            # Encode vision and language
            vision_emb = self.vision_encoder(image_tensor)
            language_emb = self.language_encoder(instruction)
            
            # Fuse representations
            fused_emb = self.fusion_module(vision_emb, language_emb)
            
            # Get action logits
            action_logits = self.policy_network(fused_emb)
            
            # Select action (greedy)
            action = torch.argmax(action_logits, dim=1).item()
            
            return action
    
    def evaluate_episode(self,
                        env,
                        instruction: str,
                        max_steps: int = 1000,
                        record_trajectory: bool = True) -> Dict[str, Any]:
        """
        Evaluate a single episode in an environment.
        
        Args:
            env: Environment instance
            instruction: Natural language instruction
            max_steps: Maximum number of steps
            record_trajectory: Whether to record the full trajectory
            
        Returns:
            Dictionary with evaluation results
        """
        # Reset environment
        obs, info = env.reset()
        image = obs["image"]
        env_instruction = obs["instruction"]
        
        # Use environment instruction if provided, otherwise use given instruction
        if env_instruction:
            instruction = env_instruction
        
        trajectory = []
        total_reward = 0.0
        success = False
        
        for step in range(max_steps):
            # Select action
            action = self.select_action(image, instruction)
            
            # Step environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            next_image = next_obs["image"]
            
            total_reward += reward
            
            # Record step if requested
            if record_trajectory:
                trajectory.append({
                    "step": step,
                    "action": int(action),
                    "reward": float(reward),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated)
                })
            
            # Check for success (typically when reward > 0 or task completed)
            if terminated:
                # In MiniGrid, positive reward usually indicates success
                success = reward > 0 or terminated
                break
            
            if truncated:
                # Time limit reached
                break
            
            image = next_image
        
        result = {
            "success": success,
            "episode_length": step + 1,
            "total_reward": float(total_reward),
            "instruction": instruction,
            "max_steps_reached": step + 1 >= max_steps
        }
        
        if record_trajectory:
            result["trajectory"] = trajectory
        
        return result
    
    def evaluate_on_environments(self,
                                 env_configs: List[Dict[str, Any]],
                                 num_episodes_per_env: int = 10,
                                 max_steps: int = 1000,
                                 record_trajectories: bool = True,
                                 output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Evaluate policy on multiple unseen environments.
        
        Args:
            env_configs: List of environment configurations
                Each config should have: env_name, instruction, seed (optional)
            num_episodes_per_env: Number of episodes to run per environment
            max_steps: Maximum steps per episode
            record_trajectories: Whether to record trajectories
            output_dir: Directory to save results and trajectories
            
        Returns:
            Dictionary with aggregated metrics and per-episode results
        """
        all_results = []
        trajectories = []
        
        print(f"Evaluating on {len(env_configs)} environments")
        print(f"{num_episodes_per_env} episodes per environment")
        print("="*60)
        
        for env_idx, env_config in enumerate(env_configs):
            env_name = env_config["env_name"]
            instruction = env_config.get("instruction", "Navigate to the goal")
            seed = env_config.get("seed", None)
            
            print(f"\nEnvironment {env_idx + 1}/{len(env_configs)}: {env_name}")
            print(f"Instruction: {instruction}")
            
            # Create environment
            env = create_navigation_env(
                env_name=env_name,
                instruction=instruction,
                seed=seed
            )
            
            # Run episodes
            env_results = []
            for episode_idx in range(num_episodes_per_env):
                # Use different seed for each episode
                episode_seed = seed + episode_idx if seed is not None else None
                if episode_seed is not None:
                    env.reset(seed=episode_seed)
                
                result = self.evaluate_episode(
                    env=env,
                    instruction=instruction,
                    max_steps=max_steps,
                    record_trajectory=record_trajectories
                )
                
                result["env_name"] = env_name
                result["episode_idx"] = episode_idx
                env_results.append(result)
                all_results.append(result)
                
                if record_trajectories and "trajectory" in result:
                    trajectories.append({
                        "env_name": env_name,
                        "episode_idx": episode_idx,
                        "instruction": instruction,
                        "trajectory": result["trajectory"],
                        "success": result["success"],
                        "episode_length": result["episode_length"]
                    })
                
                status = "✓" if result["success"] else "✗"
                print(f"  Episode {episode_idx + 1}: {status} "
                      f"(length={result['episode_length']}, reward={result['total_reward']:.2f})")
            
            # Clean up environment if needed
            try:
                env.close()
            except AttributeError:
                pass
        
        # Compute aggregated metrics
        metrics = self.compute_metrics(all_results)
        
        # Prepare output
        evaluation_results = {
            "metrics": metrics,
            "per_episode_results": all_results,
            "timestamp": datetime.now().isoformat(),
            "num_environments": len(env_configs),
            "num_episodes_per_env": num_episodes_per_env,
            "total_episodes": len(all_results)
        }
        
        if record_trajectories:
            evaluation_results["trajectories"] = trajectories
        
        # Save results if output directory provided
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Save metrics and results
            results_file = output_path / "evaluation_results.json"
            with open(results_file, 'w') as f:
                json.dump(evaluation_results, f, indent=2)
            print(f"\nResults saved to {results_file}")
        
        return evaluation_results
    
    def compute_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Compute aggregate metrics from evaluation results.
        
        Args:
            results: List of episode evaluation results
            
        Returns:
            Dictionary with aggregate metrics
        """
        if not results:
            return {}
        
        total_episodes = len(results)
        successes = sum(1 for r in results if r.get("success", False))
        success_rate = successes / total_episodes if total_episodes > 0 else 0.0
        
        episode_lengths = [r.get("episode_length", 0) for r in results]
        total_rewards = [r.get("total_reward", 0.0) for r in results]
        
        avg_episode_length = np.mean(episode_lengths) if episode_lengths else 0.0
        avg_reward = np.mean(total_rewards) if total_rewards else 0.0
        
        # Per-environment metrics
        env_metrics = {}
        env_groups = {}
        for result in results:
            env_name = result.get("env_name", "unknown")
            if env_name not in env_groups:
                env_groups[env_name] = []
            env_groups[env_name].append(result)
        
        for env_name, env_results in env_groups.items():
            env_successes = sum(1 for r in env_results if r.get("success", False))
            env_success_rate = env_successes / len(env_results) if env_results else 0.0
            env_metrics[f"{env_name}_success_rate"] = env_success_rate
        
        metrics = {
            "success_rate": success_rate,
            "total_episodes": total_episodes,
            "successes": successes,
            "failures": total_episodes - successes,
            "avg_episode_length": float(avg_episode_length),
            "avg_reward": float(avg_reward),
            "min_episode_length": float(np.min(episode_lengths)) if episode_lengths else 0.0,
            "max_episode_length": float(np.max(episode_lengths)) if episode_lengths else 0.0,
            **env_metrics
        }
        
        return metrics

