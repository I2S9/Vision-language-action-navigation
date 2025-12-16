"""
Exemple minimal d'utilisation du PolicyNetwork.

Cet exemple montre comment utiliser le réseau de politique:
    action_logits = policy(fused_embedding)
    action = torch.argmax(action_logits)
"""

import torch
from models.policy import PolicyNetwork
from models.fusion import FusionModule
from models.vision_encoder import VisionEncoder
from models.language_encoder import LanguageEncoder
import numpy as np


def main():
    """Exemple minimal d'utilisation."""
    # Créer tous les modules
    vision_encoder = VisionEncoder(embedding_dim=256)
    language_encoder = LanguageEncoder()
    
    fusion = FusionModule(
        vision_dim=256,
        language_dim=language_encoder.embedding_dim,
        fused_dim=512,
        hidden_dim=512
    )
    
    policy = PolicyNetwork(
        input_dim=512,
        hidden_dim=256,
        num_actions=4
    )
    
    # Créer des exemples d'entrées
    image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    image_tensor = torch.from_numpy(image)
    instruction = "Navigate to the green goal"
    
    # Encoder l'image et l'instruction
    visual_embedding = vision_encoder(image_tensor)
    language_embedding = language_encoder(instruction)
    
    # Fusionner les embeddings
    fused_embedding = fusion(visual_embedding, language_embedding)
    
    print(f"Fused embedding shape: {fused_embedding.shape}")
    
    # Obtenir les logits d'actions
    action_logits = policy(fused_embedding)
    
    print(f"Action logits shape: {action_logits.shape}")
    print(f"Action logits: {action_logits}")
    
    # Sélectionner l'action (méthode argmax comme dans l'exemple)
    action = torch.argmax(action_logits)
    
    print(f"\nSelected action: {action.item()}")
    print(f"Action name: {policy.get_action_name(action.item())}")
    
    # Afficher toutes les actions possibles
    print("\nAvailable actions:")
    for idx, name in policy.ACTION_SPACE.items():
        print(f"  {idx}: {name}")
    
    print("\n✓ Policy network completed successfully!")


if __name__ == "__main__":
    main()

