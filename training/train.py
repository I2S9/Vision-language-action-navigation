"""
Main training script for the VLA navigation policy.

This script sets up the models, creates a simple dataset,
and runs the training loop.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Tuple
import argparse

from models.vision_encoder import VisionEncoder
from models.language_encoder import LanguageEncoder
from models.fusion import FusionModule
from models.policy import PolicyNetwork
from training.train_policy import PolicyTrainer
from utils.seed import set_seed
from utils.config import Config


class SimpleNavigationDataset(Dataset):
    """
    Simple dataset for navigation training.
    
    This is a placeholder dataset that generates synthetic data.
    In a real scenario, this would load actual environment data.
    """
    
    def __init__(self, num_samples: int = 1000, image_size: Tuple[int, int] = (64, 64)):
        """
        Initialize the dataset.
        
        Args:
            num_samples: Number of samples in the dataset
            image_size: Size of images (height, width)
        """
        self.num_samples = num_samples
        self.image_size = image_size
        
        # Generate synthetic data
        self.images = []
        self.instructions = []
        self.actions = []
        
        # Sample instructions
        instruction_templates = [
            "Navigate to the green goal",
            "Go to the red door",
            "Move forward and turn left",
            "Avoid obstacles and reach the target",
            "Turn right and move forward"
        ]
        
        for i in range(num_samples):
            # Random RGB image
            image = np.random.randint(0, 255, (*image_size, 3), dtype=np.uint8)
            self.images.append(image)
            
            # Random instruction
            instruction = np.random.choice(instruction_templates)
            self.instructions.append(instruction)
            
            # Random action (0-3 for MiniGrid)
            action = np.random.randint(0, 4)
            self.actions.append(action)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return (
            self.images[idx],
            self.instructions[idx],
            torch.tensor(self.actions[idx], dtype=torch.long)
        )


def collate_fn(batch):
    """
    Collate function for DataLoader.
    
    Args:
        batch: List of (image, instruction, action) tuples
        
    Returns:
        Tuple of (images, instructions, actions)
    """
    images, instructions, actions = zip(*batch)
    
    # Convert images to tensor
    images = np.stack(images)
    images = torch.from_numpy(images).float()
    
    # Instructions are already strings
    instructions = list(instructions)
    
    # Stack actions
    actions = torch.stack(actions)
    
    return images, instructions, actions


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train VLA navigation policy")
    parser.add_argument("--config", type=str, help="Path to config file (optional)")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of training samples")
    parser.add_argument("--save_interval", type=int, default=10, help="Checkpoint save interval")
    parser.add_argument("--log_interval", type=int, default=10, help="Logging interval")
    
    args = parser.parse_args()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Device
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = "cpu"
    
    print("="*60)
    print("VLA Navigation Policy Training")
    print("="*60)
    print(f"Device: {device}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Seed: {args.seed}")
    print("="*60)
    
    # Create models
    print("\nCreating models...")
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
    
    print(f"Vision encoder: {sum(p.numel() for p in vision_encoder.parameters())} parameters")
    print(f"Language encoder: {sum(p.numel() for p in language_encoder.parameters())} parameters")
    print(f"Fusion module: {sum(p.numel() for p in fusion_module.parameters())} parameters")
    print(f"Policy network: {sum(p.numel() for p in policy_network.parameters())} parameters")
    
    # Create trainer
    trainer = PolicyTrainer(
        vision_encoder=vision_encoder,
        language_encoder=language_encoder,
        fusion_module=fusion_module,
        policy_network=policy_network,
        learning_rate=args.learning_rate,
        device=device,
        checkpoint_dir=args.checkpoint_dir
    )
    
    # Create dataset and dataloader
    print(f"\nCreating dataset with {args.num_samples} samples...")
    dataset = SimpleNavigationDataset(num_samples=args.num_samples)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    # Train
    trainer.train(
        train_dataloader=dataloader,
        num_epochs=args.epochs,
        save_interval=args.save_interval,
        log_interval=args.log_interval
    )
    
    # Final checkpoint and plots
    trainer.save_checkpoint("final_checkpoint.pt")
    trainer.plot_training_curves()
    
    print("\nTraining completed successfully!")


if __name__ == "__main__":
    main()

