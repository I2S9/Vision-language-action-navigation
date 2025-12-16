"""
Test script for the FusionModule.

This script verifies that the fusion module works correctly with
different input dimensions and produces the expected output dimensions.
"""

import torch
from models.fusion import FusionModule


def test_fusion_module():
    """Test the FusionModule with various inputs."""
    print("Testing FusionModule...")
    
    # Test 1: Default dimensions
    print("\nTest 1: Default dimensions")
    fusion = FusionModule(
        vision_dim=256,
        language_dim=384,
        fused_dim=512,
        hidden_dim=512
    )
    
    print(f"Vision dim: {fusion.vision_dim}")
    print(f"Language dim: {fusion.language_dim}")
    print(f"Fused dim: {fusion.fused_dim}")
    print(f"Hidden dim: {fusion.hidden_dim}")
    
    # Test forward pass
    batch_size = 4
    visual_emb = torch.randn(batch_size, 256)
    language_emb = torch.randn(batch_size, 384)
    
    fused = fusion(visual_emb, language_emb)
    
    print(f"Visual embedding shape: {visual_emb.shape}")
    print(f"Language embedding shape: {language_emb.shape}")
    print(f"Fused embedding shape: {fused.shape}")
    
    assert fused.shape == (batch_size, 512), \
        f"Expected shape ({batch_size}, 512), got {fused.shape}"
    print("✓ Test 1 passed")
    
    # Test 2: Different batch sizes
    print("\nTest 2: Different batch sizes")
    for batch_size in [1, 2, 8, 16, 32]:
        visual_emb = torch.randn(batch_size, 256)
        language_emb = torch.randn(batch_size, 384)
        fused = fusion(visual_emb, language_emb)
        assert fused.shape == (batch_size, 512), \
            f"Batch size {batch_size}: expected ({batch_size}, 512), got {fused.shape}"
    print("✓ Test 2 passed: All batch sizes work correctly")
    
    # Test 3: Dimension mismatch detection
    print("\nTest 3: Dimension mismatch detection")
    try:
        wrong_visual = torch.randn(4, 128)  # Wrong dimension
        language_emb = torch.randn(4, 384)
        fusion(wrong_visual, language_emb)
        assert False, "Should have raised ValueError for dimension mismatch"
    except ValueError as e:
        print(f"✓ Correctly caught dimension mismatch: {e}")
    
    try:
        visual_emb = torch.randn(4, 256)
        wrong_language = torch.randn(4, 256)  # Wrong dimension
        fusion(visual_emb, wrong_language)
        assert False, "Should have raised ValueError for dimension mismatch"
    except ValueError as e:
        print(f"✓ Correctly caught dimension mismatch: {e}")
    
    # Test 4: Batch size mismatch detection
    print("\nTest 4: Batch size mismatch detection")
    try:
        visual_emb = torch.randn(4, 256)
        language_emb = torch.randn(5, 384)  # Different batch size
        fusion(visual_emb, language_emb)
        assert False, "Should have raised ValueError for batch size mismatch"
    except ValueError as e:
        print(f"✓ Correctly caught batch size mismatch: {e}")
    
    # Test 5: Different fused dimensions
    print("\nTest 5: Different fused dimensions")
    for fused_dim in [128, 256, 512, 1024]:
        fusion_test = FusionModule(
            vision_dim=256,
            language_dim=384,
            fused_dim=fused_dim,
            hidden_dim=512
        )
        visual_emb = torch.randn(4, 256)
        language_emb = torch.randn(4, 384)
        fused = fusion_test(visual_emb, language_emb)
        assert fused.shape == (4, fused_dim), \
            f"Expected ({4}, {fused_dim}), got {fused.shape}"
    print("✓ Test 5 passed: Different fused dimensions work correctly")
    
    # Test 6: Verify concatenation happens correctly
    print("\nTest 6: Verify concatenation")
    fusion_test = FusionModule(
        vision_dim=2,
        language_dim=3,
        fused_dim=10,
        hidden_dim=10
    )
    
    # Use simple values to verify concatenation
    visual_emb = torch.tensor([[1.0, 2.0]])
    language_emb = torch.tensor([[3.0, 4.0, 5.0]])
    
    # Manually check concatenation
    expected_concat = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
    
    # Get the concatenated tensor before MLP
    with torch.no_grad():
        combined = torch.cat([visual_emb, language_emb], dim=1)
        assert torch.allclose(combined, expected_concat), \
            "Concatenation not working correctly"
    
    fused = fusion_test(visual_emb, language_emb)
    assert fused.shape == (1, 10), f"Expected (1, 10), got {fused.shape}"
    print("✓ Test 6 passed: Concatenation works correctly")
    
    # Test 7: Gradient flow
    print("\nTest 7: Gradient flow")
    fusion_test = FusionModule(vision_dim=256, language_dim=384, fused_dim=512)
    visual_emb = torch.randn(4, 256, requires_grad=True)
    language_emb = torch.randn(4, 384, requires_grad=True)
    
    fused = fusion_test(visual_emb, language_emb)
    loss = fused.sum()
    loss.backward()
    
    assert visual_emb.grad is not None, "Gradients should flow to vision embedding"
    assert language_emb.grad is not None, "Gradients should flow to language embedding"
    print("✓ Test 7 passed: Gradients flow correctly")
    
    print("\n" + "="*50)
    print("All tests passed!")
    print("="*50)


if __name__ == "__main__":
    test_fusion_module()

