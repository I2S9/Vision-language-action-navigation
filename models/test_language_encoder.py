"""
Test script for the LanguageEncoder module.

This script verifies that the language encoder works correctly with
different input formats and produces normalized embeddings.
"""

import torch
from models.language_encoder import LanguageEncoder


def test_language_encoder():
    """Test the LanguageEncoder with various inputs."""
    print("Testing LanguageEncoder...")
    
    # Create encoder
    encoder = LanguageEncoder(
        model_name="all-MiniLM-L6-v2",
        freeze_weights=True,
        normalize=True
    )
    
    print(f"Encoder created with embedding_dim={encoder.embedding_dim}")
    
    # Check that weights are frozen
    frozen_params = sum(1 for p in encoder.sentence_model.parameters() if not p.requires_grad)
    total_params = sum(1 for p in encoder.sentence_model.parameters())
    print(f"Frozen parameters: {frozen_params}/{total_params}")
    assert frozen_params == total_params, "All pre-trained weights should be frozen"
    
    # Test 1: Single instruction string
    print("\nTest 1: Single instruction string")
    instruction = "Navigate to the green goal"
    language_embedding = encoder(instruction)
    print(f"Input: '{instruction}'")
    print(f"Output shape: {language_embedding.shape}")
    print(f"Output dtype: {language_embedding.dtype}")
    assert language_embedding.shape == (1, encoder.embedding_dim), \
        f"Expected (1, {encoder.embedding_dim}), got {language_embedding.shape}"
    print("✓ Test 1 passed")
    
    # Test 2: List of instructions
    print("\nTest 2: List of instructions")
    instructions = [
        "Go to the red door",
        "Navigate to the goal",
        "Avoid obstacles and reach the target"
    ]
    language_embeddings = encoder(instructions)
    print(f"Input: {len(instructions)} instructions")
    print(f"Output shape: {language_embeddings.shape}")
    assert language_embeddings.shape == (3, encoder.embedding_dim), \
        f"Expected (3, {encoder.embedding_dim}), got {language_embeddings.shape}"
    print("✓ Test 2 passed")
    
    # Test 3: Verify normalization
    print("\nTest 3: Verify normalization")
    instruction = "Move forward and turn left"
    embedding = encoder(instruction)
    
    # Check that embedding is normalized (L2 norm should be ~1.0)
    norm = torch.norm(embedding, p=2, dim=1).item()
    print(f"L2 norm of embedding: {norm:.6f}")
    assert abs(norm - 1.0) < 1e-5, f"Embedding should be normalized (norm={norm})"
    print("✓ Test 3 passed: Embeddings are normalized")
    
    # Test 4: Verify fixed output size
    print("\nTest 4: Verify fixed output size")
    test_instructions = [
        "Short",
        "This is a longer instruction with more words",
        "Go!"
    ]
    embeddings = encoder(test_instructions)
    assert embeddings.shape[1] == encoder.embedding_dim, \
        f"Output dimension must be {encoder.embedding_dim}"
    assert embeddings.shape[0] == len(test_instructions), \
        "Batch size should match number of instructions"
    print("✓ Test 4 passed: Fixed output size verified")
    
    # Test 5: Test encode() method alias
    print("\nTest 5: Test encode() method")
    instruction = "Test instruction"
    embedding1 = encoder(instruction)
    embedding2 = encoder.encode(instruction)
    assert torch.allclose(embedding1, embedding2), "encode() should match forward()"
    print("✓ Test 5 passed")
    
    # Test 6: Different instructions produce different embeddings
    print("\nTest 6: Different instructions produce different embeddings")
    inst1 = "Go to the goal"
    inst2 = "Avoid the wall"
    emb1 = encoder(inst1)
    emb2 = encoder(inst2)
    diff = torch.norm(emb1 - emb2).item()
    print(f"Difference between embeddings: {diff:.6f}")
    assert diff > 0.1, "Different instructions should produce different embeddings"
    print("✓ Test 6 passed")
    
    print("\n" + "="*50)
    print("All tests passed!")
    print("="*50)


if __name__ == "__main__":
    test_language_encoder()

