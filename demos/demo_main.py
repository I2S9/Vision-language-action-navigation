"""
Main script for generating visual demonstrations of agent behavior.

This script loads a trained model and generates GIF or video demonstrations
showing the agent following navigation instructions.
"""

import torch
import argparse
from pathlib import Path

from models.vision_encoder import VisionEncoder
from models.language_encoder import LanguageEncoder
from models.fusion import FusionModule
from models.policy import PolicyNetwork
from demos.rollout import RolloutRunner
from env.environment import create_navigation_env
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


def main():
    """Main function for generating demonstrations."""
    parser = argparse.ArgumentParser(description="Generate visual demonstrations of agent behavior")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--env_name", type=str, default="MiniGrid-Empty-8x8-v0",
                       help="Environment name")
    parser.add_argument("--instruction", type=str,
                       default="Navigate to the green goal",
                       help="Natural language instruction")
    parser.add_argument("--output", type=str, default="demos/demonstration.gif",
                       help="Output path for demonstration")
    parser.add_argument("--format", type=str, default="gif", choices=["gif", "mp4"],
                       help="Output format (gif or mp4)")
    parser.add_argument("--fps", type=int, default=2,
                       help="Frames per second")
    parser.add_argument("--max_steps", type=int, default=1000,
                       help="Maximum steps per episode")
    parser.add_argument("--device", type=str, default="cpu",
                       help="Device (cpu/cuda)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--no_annotate", action="store_true",
                       help="Disable frame annotations")
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Device
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = "cpu"
    
    print("="*60)
    print("VLA Navigation Agent Demonstration")
    print("="*60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Environment: {args.env_name}")
    print(f"Instruction: {args.instruction}")
    print(f"Output: {args.output}")
    print(f"Format: {args.format}")
    print(f"Device: {device}")
    print("="*60)
    
    # Load models
    print("\nLoading models from checkpoint...")
    vision_encoder, language_encoder, fusion_module, policy_network = \
        load_models_from_checkpoint(args.checkpoint, device=device)
    print("Models loaded successfully")
    
    # Create rollout runner
    runner = RolloutRunner(
        vision_encoder=vision_encoder,
        language_encoder=language_encoder,
        fusion_module=fusion_module,
        policy_network=policy_network,
        device=device
    )
    
    # Create environment
    print(f"\nCreating environment: {args.env_name}")
    env = create_navigation_env(
        env_name=args.env_name,
        instruction=args.instruction,
        seed=args.seed
    )
    
    # Generate demonstration
    print("\nGenerating demonstration...")
    result = runner.generate_demonstration(
        env=env,
        instruction=args.instruction,
        output_path=args.output,
        max_steps=args.max_steps,
        format=args.format,
        fps=args.fps,
        annotate=not args.no_annotate
    )
    
    # Print summary
    print("\n" + "="*60)
    print("Demonstration Summary")
    print("="*60)
    print(f"Success: {result['success']}")
    print(f"Episode length: {result['episode_length']} steps")
    print(f"Total reward: {result['total_reward']:.2f}")
    print(f"Output saved to: {result['output_path']}")
    print("="*60)
    
    env.close()


if __name__ == "__main__":
    main()

