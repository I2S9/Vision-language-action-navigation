"""
Test script for the VisionEncoder module.

This script verifies that the vision encoder works correctly with
different input formats and produces fixed-size embeddings.
"""

import torch
import numpy as np
from models.vision_encoder import VisionEncoder


def test_vision_encoder():
    """Test the VisionEncoder with various inputs."""
    print("Testing VisionEncoder...")
    
    # Create encoder
    encoder = VisionEncoder(
        input_channels=3,
        embedding_dim=256,
        image_size=(64, 64)
    )
    
    print(f"Encoder created with embedding_dim={encoder.embedding_dim}")
    print(f"Flattened size after conv layers: {encoder.flattened_size}")
    
    # Test 1: Single image as numpy array (uint8)
    print("\nTest 1: Single image (numpy uint8, channels last)")
    image_np = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    image_tensor = torch.from_numpy(image_np)
    visual_embedding = encoder(image_tensor)
    print(f"Input shape: {image_tensor.shape}")
    print(f"Output shape: {visual_embedding.shape}")
    print(f"Output dtype: {visual_embedding.dtype}")
    assert visual_embedding.shape == (1, 256), f"Expected (1, 256), got {visual_embedding.shape}"
    print("✓ Test 1 passed")
    
    # Test 2: Batch of images (uint8)
    print("\nTest 2: Batch of images (uint8)")
    batch_images = np.random.randint(0, 255, (4, 64, 64, 3), dtype=np.uint8)
    batch_tensor = torch.from_numpy(batch_images)
    visual_embeddings = encoder(batch_tensor)
    print(f"Input shape: {batch_tensor.shape}")
    print(f"Output shape: {visual_embeddings.shape}")
    assert visual_embeddings.shape == (4, 256), f"Expected (4, 256), got {visual_embeddings.shape}"
    print("✓ Test 2 passed")
    
    # Test 3: Image with channels first (float)
    print("\nTest 3: Image with channels first (float [0, 1])")
    image_float = torch.rand(1, 3, 64, 64)
    visual_embedding = encoder(image_float)
    print(f"Input shape: {image_float.shape}")
    print(f"Output shape: {visual_embedding.shape}")
    assert visual_embedding.shape == (1, 256), f"Expected (1, 256), got {visual_embedding.shape}"
    print("✓ Test 3 passed")
    
    # Test 4: Batch with channels first (float)
    print("\nTest 4: Batch with channels first (float)")
    batch_float = torch.rand(8, 3, 64, 64)
    visual_embeddings = encoder(batch_float)
    print(f"Input shape: {batch_float.shape}")
    print(f"Output shape: {visual_embeddings.shape}")
    assert visual_embeddings.shape == (8, 256), f"Expected (8, 256), got {visual_embeddings.shape}"
    print("✓ Test 4 passed")
    
    # Test 5: Verify normalization
    print("\nTest 5: Verify normalization")
    image_uint8 = torch.randint(0, 255, (1, 3, 64, 64), dtype=torch.uint8)
    embedding1 = encoder(image_uint8)
    
    image_float = image_uint8.float() / 255.0
    embedding2 = encoder(image_float)
    
    # Embeddings should be very similar (allowing for small numerical differences)
    diff = torch.abs(embedding1 - embedding2).max().item()
    print(f"Max difference between uint8 and normalized float: {diff:.6f}")
    assert diff < 1e-5, "Normalization should produce similar results"
    print("✓ Test 5 passed")
    
    # Test 6: Verify fixed output size
    print("\nTest 6: Verify fixed output size")
    for batch_size in [1, 2, 4, 8, 16]:
        test_image = torch.rand(batch_size, 3, 64, 64)
        embedding = encoder(test_image)
        assert embedding.shape[1] == 256, f"Output dimension must be 256, got {embedding.shape[1]}"
        assert embedding.shape[0] == batch_size, f"Batch size mismatch"
    print("✓ Test 6 passed: Fixed output size verified")
    
    print("\n" + "="*50)
    print("All tests passed!")
    print("="*50)


if __name__ == "__main__":
    test_vision_encoder()

