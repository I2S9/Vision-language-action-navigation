"""
Exemple minimal d'utilisation du LanguageEncoder.

Cet exemple montre comment utiliser l'encodeur de langage:
    language_embedding = language_encoder(instruction)
"""

from models.language_encoder import LanguageEncoder


def main():
    """Exemple minimal d'utilisation."""
    # Créer l'encodeur
    language_encoder = LanguageEncoder()
    
    # Encoder une instruction
    instruction = "Navigate to the green goal"
    language_embedding = language_encoder(instruction)
    
    print(f"Instruction: '{instruction}'")
    print(f"Language embedding shape: {language_embedding.shape}")
    print(f"Language embedding dtype: {language_embedding.dtype}")
    
    # L'embedding est maintenant prêt pour la fusion avec la vision
    print(f"\nEmbedding vector (first 10 values): {language_embedding[0, :10]}")
    
    # Vérifier la normalisation
    norm = language_embedding.norm(p=2, dim=1).item()
    print(f"L2 norm (should be ~1.0): {norm:.6f}")


if __name__ == "__main__":
    main()

