"""
Exemple minimal d'utilisation du FusionModule.

Cet exemple montre comment utiliser le module de fusion:
    fused = fusion(visual_embedding, language_embedding)
"""

import torch
from models.fusion import FusionModule
from models.vision_encoder import VisionEncoder
from models.language_encoder import LanguageEncoder
import numpy as np


def main():
    """Exemple minimal d'utilisation."""
    # Créer les encodeurs
    vision_encoder = VisionEncoder(embedding_dim=256)
    language_encoder = LanguageEncoder()
    
    # Créer le module de fusion
    # Note: language_encoder.embedding_dim peut être différent (384 pour all-MiniLM-L6-v2)
    fusion = FusionModule(
        vision_dim=256,
        language_dim=language_encoder.embedding_dim,
        fused_dim=512,
        hidden_dim=512
    )
    
    # Créer des exemples d'entrées
    # Image d'exemple (format numpy uint8, channels last)
    image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    image_tensor = torch.from_numpy(image)
    
    # Instruction d'exemple
    instruction = "Navigate to the green goal"
    
    # Encoder l'image et l'instruction
    visual_embedding = vision_encoder(image_tensor)
    language_embedding = language_encoder(instruction)
    
    print(f"Visual embedding shape: {visual_embedding.shape}")
    print(f"Language embedding shape: {language_embedding.shape}")
    
    # Fusionner les embeddings
    fused = fusion(visual_embedding, language_embedding)
    
    print(f"\nFused embedding shape: {fused.shape}")
    print(f"Fused embedding dtype: {fused.dtype}")
    print(f"\nFused vector (first 10 values): {fused[0, :10]}")
    
    # Le vecteur fusionné est maintenant prêt pour le réseau de politique
    print("\n✓ Fusion completed successfully!")


if __name__ == "__main__":
    main()

