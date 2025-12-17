"""
Main evaluation script for the VLA navigation policy.

This script loads a trained model and evaluates it on unseen environments.
"""

import torch
import argparse
from pathlib import Path
import json

from models.vision_encoder import VisionEncoder
from models.language_encoder import LanguageEncoder
from models.fusion import FusionModule
from models.policy import PolicyNetwork
from evaluation.evaluate import PolicyEvaluator
from utils.seed import set_seed


def load_models_from_checkpoint(checkpoint_path: str, device: str = "cpu"):
    """
    Load models from a checkpoint file.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load models on
        
    Returns:
        Tuple of (vision_encoder, language_encoder, fusion_module, policy_network)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create models
    vision_encoder = VisionEncoder(embedding_dim=256, image_size=(64, 64))
    language_encoder = LanguageEncoder()
    
    fusion_module = FusionModule(
        vision_dim=256,
        language_dim=language_encoder.embedding_dim,
        fused_dim=512,
        hidden_dim=512
    )
    
    policy_network = PolicyNetwork(
        input_dim=512,
        hidden_dim=256,
        num_actions=4
    )
    
    # Load state dicts
    vision_encoder.load_state_dict(checkpoint["vision_encoder_state_dict"])
    language_encoder.load_state_dict(checkpoint["language_encoder_state_dict"])
    fusion_module.load_state_dict(checkpoint["fusion_module_state_dict"])
    policy_network.load_state_dict(checkpoint["policy_network_state_dict"])
    
    return vision_encoder, language_encoder, fusion_module, policy_network


def get_test_environments():
    """
    Get list of test environment configurations for evaluation.
    
    These are unseen environments to test generalization.
    
    Returns:
        List of environment configuration dictionaries
    """
    env_configs = [
        {
            "env_name": "MiniGrid-Empty-8x8-v0",
            "instruction": "Navigate to the green goal",
            "seed": 100
        },
        {
            "env_name": "MiniGrid-Empty-8x8-v0",
            "instruction": "Go to the goal and avoid obstacles",
            "seed": 101
        },
        {
            "env_name": "MiniGrid-Empty-16x16-v0",
            "instruction": "Navigate to the green goal",
            "seed": 200
        },
        {
            "env_name": "MiniGrid-Empty-16x16-v0",
            "instruction": "Move forward and reach the target",
            "seed": 201
        },
    ]
    
    return env_configs


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate VLA navigation policy")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--device", type=str, default="cpu",
                       help="Device (cpu/cuda)")
    parser.add_argument("--num_episodes", type=int, default=10,
                       help="Number of episodes per environment")
    parser.add_argument("--max_steps", type=int, default=1000,
                       help="Maximum steps per episode")
    parser.add_argument("--output_dir", type=str, default="evaluation_results",
                       help="Directory to save evaluation results")
    parser.add_argument("--record_trajectories", action="store_true",
                       help="Record full trajectories")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--env_config_file", type=str, default=None,
                       help="Path to JSON file with environment configurations")
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Device
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = "cpu"
    
    print("="*60)
    print("VLA Navigation Policy Evaluation")
    print("="*60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {device}")
    print(f"Episodes per environment: {args.num_episodes}")
    print(f"Max steps per episode: {args.max_steps}")
    print("="*60)
    
    # Load models
    print("\nLoading models from checkpoint...")
    vision_encoder, language_encoder, fusion_module, policy_network = \
        load_models_from_checkpoint(args.checkpoint, device=device)
    print("Models loaded successfully")
    
    # Create evaluator
    evaluator = PolicyEvaluator(
        vision_encoder=vision_encoder,
        language_encoder=language_encoder,
        fusion_module=fusion_module,
        policy_network=policy_network,
        device=device
    )
    
    # Get test environments
    if args.env_config_file:
        with open(args.env_config_file, 'r') as f:
            env_configs = json.load(f)
    else:
        env_configs = get_test_environments()
    
    # Run evaluation
    print("\nStarting evaluation...")
    results = evaluator.evaluate_on_environments(
        env_configs=env_configs,
        num_episodes_per_env=args.num_episodes,
        max_steps=args.max_steps,
        record_trajectories=args.record_trajectories,
        output_dir=args.output_dir
    )
    
    # Print summary
    print("\n" + "="*60)
    print("Evaluation Summary")
    print("="*60)
    metrics = results["metrics"]
    print(f"Total episodes: {metrics['total_episodes']}")
    print(f"Successes: {metrics['successes']}")
    print(f"Failures: {metrics['failures']}")
    print(f"\nSuccess rate: {metrics['success_rate']:.2%}")
    print(f"Average episode length: {metrics['avg_episode_length']:.2f}")
    print(f"Average reward: {metrics['avg_reward']:.2f}")
    print(f"Min episode length: {metrics['min_episode_length']:.0f}")
    print(f"Max episode length: {metrics['max_episode_length']:.0f}")
    
    # Per-environment metrics
    print("\nPer-environment success rates:")
    for key, value in metrics.items():
        if key.endswith("_success_rate"):
            env_name = key.replace("_success_rate", "")
            print(f"  {env_name}: {value:.2%}")
    
    print("\n" + "="*60)
    print("Evaluation completed!")
    print(f"Results saved to: {args.output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()

