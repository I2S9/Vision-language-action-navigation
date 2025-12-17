"""
Script simple pour générer une démonstration GIF.

Ce script peut fonctionner avec ou sans checkpoint entraîné.
Sans checkpoint, il utilise des modèles non entraînés pour tester le système.
"""

import torch
from pathlib import Path

from models.vision_encoder import VisionEncoder
from models.language_encoder import LanguageEncoder
from models.fusion import FusionModule
from models.policy import PolicyNetwork
from demos.rollout import RolloutRunner
from env.environment import create_navigation_env
from utils.seed import set_seed


def load_models_from_checkpoint(checkpoint_path: str, device: str = "cpu"):
    """Load models from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
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
    
    vision_encoder.load_state_dict(checkpoint["vision_encoder_state_dict"])
    language_encoder.load_state_dict(checkpoint["language_encoder_state_dict"])
    fusion_module.load_state_dict(checkpoint["fusion_module_state_dict"])
    policy_network.load_state_dict(checkpoint["policy_network_state_dict"])
    
    return vision_encoder, language_encoder, fusion_module, policy_network


def create_untrained_models(device: str = "cpu"):
    """Create untrained models for testing."""
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
    
    return vision_encoder, language_encoder, fusion_module, policy_network


def main():
    """Générer une démonstration GIF."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Générer une démonstration GIF")
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Path to checkpoint (optional, uses untrained models if not provided)")
    parser.add_argument("--output", type=str, default="demos/demonstration.gif",
                       help="Output path for GIF")
    parser.add_argument("--env_name", type=str, default="MiniGrid-Empty-8x8-v0",
                       help="Environment name")
    parser.add_argument("--instruction", type=str, default="Navigate to the green goal",
                       help="Instruction")
    parser.add_argument("--fps", type=int, default=2, help="Frames per second")
    parser.add_argument("--max_steps", type=int, default=50, help="Max steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    device = "cpu"
    
    print("="*60)
    print("Génération d'une démonstration GIF")
    print("="*60)
    
    # Charger ou créer les modèles
    if args.checkpoint and Path(args.checkpoint).exists():
        print(f"Chargement du checkpoint: {args.checkpoint}")
        vision_encoder, language_encoder, fusion_module, policy_network = \
            load_models_from_checkpoint(args.checkpoint, device=device)
        print("Modèles chargés depuis le checkpoint")
    else:
        print("Utilisation de modèles non entraînés (pour test)")
        vision_encoder, language_encoder, fusion_module, policy_network = \
            create_untrained_models(device=device)
    
    # Créer le rollout runner
    runner = RolloutRunner(
        vision_encoder=vision_encoder,
        language_encoder=language_encoder,
        fusion_module=fusion_module,
        policy_network=policy_network,
        device=device
    )
    
    # Créer l'environnement
    print(f"\nCréation de l'environnement: {args.env_name}")
    env = create_navigation_env(
        env_name=args.env_name,
        instruction=args.instruction,
        seed=args.seed
    )
    
    # Générer la démonstration
    print(f"\nGénération du GIF...")
    print(f"Instruction: {args.instruction}")
    print(f"Sortie: {args.output}")
    
    result = runner.generate_demonstration(
        env=env,
        instruction=args.instruction,
        output_path=args.output,
        max_steps=args.max_steps,
        format="gif",
        fps=args.fps,
        annotate=True
    )
    
    # Résumé
    print("\n" + "="*60)
    print("Résumé")
    print("="*60)
    print(f"Succès: {result['success']}")
    print(f"Longueur: {result['episode_length']} steps")
    print(f"Reward total: {result['total_reward']:.2f}")
    print(f"\n✓ GIF généré: {result['output_path']}")
    print("="*60)
    
    env.close()


if __name__ == "__main__":
    main()

