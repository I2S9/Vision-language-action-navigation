"""
Test script for the PolicyNetwork.

This script verifies that the policy network works correctly and
produces valid action logits.
"""

import torch
from models.policy import PolicyNetwork


def test_policy_network():
    """Test the PolicyNetwork with various inputs."""
    print("Testing PolicyNetwork...")
    
    # Test 1: Default initialization
    print("\nTest 1: Default initialization")
    policy = PolicyNetwork(
        input_dim=512,
        hidden_dim=256,
        num_actions=4
    )
    
    print(f"Input dim: {policy.input_dim}")
    print(f"Hidden dim: {policy.hidden_dim}")
    print(f"Num actions: {policy.num_actions}")
    print(f"Action space: {policy.ACTION_SPACE}")
    
    # Test forward pass
    batch_size = 4
    fused_embedding = torch.randn(batch_size, 512)
    action_logits = policy(fused_embedding)
    
    print(f"Fused embedding shape: {fused_embedding.shape}")
    print(f"Action logits shape: {action_logits.shape}")
    
    assert action_logits.shape == (batch_size, 4), \
        f"Expected shape ({batch_size}, 4), got {action_logits.shape}"
    print("✓ Test 1 passed")
    
    # Test 2: Different batch sizes
    print("\nTest 2: Different batch sizes")
    for batch_size in [1, 2, 8, 16, 32]:
        fused_emb = torch.randn(batch_size, 512)
        logits = policy(fused_emb)
        assert logits.shape == (batch_size, 4), \
            f"Batch size {batch_size}: expected ({batch_size}, 4), got {logits.shape}"
    print("✓ Test 2 passed: All batch sizes work correctly")
    
    # Test 3: Dimension mismatch detection
    print("\nTest 3: Dimension mismatch detection")
    try:
        wrong_input = torch.randn(4, 256)  # Wrong dimension
        policy(wrong_input)
        assert False, "Should have raised ValueError for dimension mismatch"
    except ValueError as e:
        print(f"✓ Correctly caught dimension mismatch: {e}")
    
    # Test 4: Action selection (argmax)
    print("\nTest 4: Action selection (argmax)")
    fused_emb = torch.randn(4, 512)
    actions = policy.select_action(fused_emb, method="argmax")
    
    print(f"Selected actions: {actions}")
    print(f"Actions shape: {actions.shape}")
    
    assert actions.shape == (4,), f"Expected shape (4,), got {actions.shape}"
    assert torch.all((actions >= 0) & (actions < 4)), \
        "All actions should be in valid range [0, 3]"
    print("✓ Test 4 passed")
    
    # Test 5: Action selection (sample)
    print("\nTest 5: Action selection (sample)")
    actions = policy.select_action(fused_emb, method="sample")
    
    print(f"Sampled actions: {actions}")
    assert actions.shape == (4,), f"Expected shape (4,), got {actions.shape}"
    assert torch.all((actions >= 0) & (actions < 4)), \
        "All actions should be in valid range [0, 3]"
    print("✓ Test 5 passed")
    
    # Test 6: Action names
    print("\nTest 6: Action names")
    for action_idx in range(4):
        action_name = policy.get_action_name(action_idx)
        print(f"Action {action_idx}: {action_name}")
        assert action_name in policy.ACTION_SPACE.values() or \
               action_name.startswith("Unknown"), \
            f"Invalid action name: {action_name}"
    print("✓ Test 6 passed")
    
    # Test 7: Logits are raw (not probabilities)
    print("\nTest 7: Logits are raw (not probabilities)")
    fused_emb = torch.randn(1, 512)
    logits = policy(fused_emb)
    
    # Logits should not sum to 1 (they're not probabilities)
    logit_sum = logits.sum().item()
    print(f"Sum of logits: {logit_sum}")
    assert abs(logit_sum - 1.0) > 0.1, "Logits should not sum to 1 (they're not probabilities)"
    
    # Convert to probabilities and verify
    probs = torch.softmax(logits, dim=1)
    prob_sum = probs.sum().item()
    print(f"Sum of probabilities (after softmax): {prob_sum:.6f}")
    assert abs(prob_sum - 1.0) < 1e-5, "Probabilities should sum to 1"
    print("✓ Test 7 passed")
    
    # Test 8: Gradient flow
    print("\nTest 8: Gradient flow")
    fused_emb = torch.randn(4, 512, requires_grad=True)
    logits = policy(fused_emb)
    loss = logits.sum()
    loss.backward()
    
    assert fused_emb.grad is not None, "Gradients should flow to input"
    print("✓ Test 8 passed: Gradients flow correctly")
    
    # Test 9: Example usage as in requirements
    print("\nTest 9: Example usage (as in requirements)")
    fused_emb = torch.randn(1, 512)
    action_logits = policy(fused_emb)
    action = torch.argmax(action_logits)
    
    print(f"Action logits: {action_logits}")
    print(f"Selected action: {action.item()}")
    print(f"Action name: {policy.get_action_name(action.item())}")
    
    assert action.item() in range(4), "Action should be in valid range"
    print("✓ Test 9 passed")
    
    print("\n" + "="*50)
    print("All tests passed!")
    print("="*50)


if __name__ == "__main__":
    test_policy_network()

