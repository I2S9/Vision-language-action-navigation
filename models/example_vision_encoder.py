"""
Exemple minimal d'utilisation du VisionEncoder.

Cet exemple montre comment utiliser l'encodeur visuel:
    encoder = VisionEncoder()
    visual_embedding = encoder(image)
"""

import torch
import numpy as np
from models.vision_encoder import VisionEncoder


def main():
    """Exemple minimal d'utilisation."""
    # Créer l'encodeur
    encoder = VisionEncoder()
    
    # Créer une image d'exemple (format numpy uint8, channels last)
    # Dans un vrai cas, cette image viendrait de l'environnement
    image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    image_tensor = torch.from_numpy(image)
    
    # Encoder l'image
    visual_embedding = encoder(image_tensor)
    
    print(f"Image shape: {image.shape}")
    print(f"Visual embedding shape: {visual_embedding.shape}")
    print(f"Visual embedding dtype: {visual_embedding.dtype}")
    
    # L'embedding est maintenant prêt pour la fusion avec le langage
    print(f"\nEmbedding vector (first 10 values): {visual_embedding[0, :10]}")


if __name__ == "__main__":
    main()

