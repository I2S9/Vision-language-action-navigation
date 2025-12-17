"""
Training script for the vision-language-action policy.

This module contains the training loop and optimization logic
for learning the navigation policy from visual and language inputs.
"""

import torch
import torch.optim as optim
from typing import Dict, Any, List, Optional
import numpy as np
import os
import json
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime

from models.vision_encoder import VisionEncoder
from models.language_encoder import LanguageEncoder
from models.fusion import FusionModule
from models.policy import PolicyNetwork


class PolicyTrainer:
    """
    Trainer for the VLA navigation policy.
    
    Handles the training loop, loss computation, optimization,
    checkpoint saving, and metric logging.
    """
    
    def __init__(self,
                 vision_encoder: VisionEncoder,
                 language_encoder: LanguageEncoder,
                 fusion_module: FusionModule,
                 policy_network: PolicyNetwork,
                 learning_rate: float = 1e-4,
                 device: str = "cpu",
                 checkpoint_dir: str = "checkpoints"):
        """
        Initialize the policy trainer.
        
        Args:
            vision_encoder: Vision encoder model
            language_encoder: Language encoder model
            fusion_module: Fusion module
            policy_network: Policy network
            learning_rate: Learning rate for optimization
            device: Device to run training on
            checkpoint_dir: Directory to save checkpoints
        """
        self.vision_encoder = vision_encoder.to(device)
        self.language_encoder = language_encoder.to(device)
        self.fusion_module = fusion_module.to(device)
        self.policy_network = policy_network.to(device)
        
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Combine all trainable parameters for optimization
        # Note: language encoder weights are frozen by default
        trainable_params = []
        trainable_params.extend([p for p in vision_encoder.parameters() if p.requires_grad])
        trainable_params.extend([p for p in language_encoder.parameters() if p.requires_grad])
        trainable_params.extend([p for p in fusion_module.parameters() if p.requires_grad])
        trainable_params.extend([p for p in policy_network.parameters() if p.requires_grad])
        
        self.optimizer = optim.Adam(trainable_params, lr=learning_rate)
        
        # Training history
        self.train_losses: List[float] = []
        self.train_accuracies: List[float] = []
        self.epoch = 0
        
    def train_step(self,
                   images: torch.Tensor,
                   instructions: List[str],
                   actions: torch.Tensor) -> Dict[str, float]:
        """
        Perform one training step.
        
        Args:
            images: Batch of RGB images of shape (batch, H, W, 3) or (batch, 3, H, W)
            instructions: List of instruction strings
            actions: Batch of target actions of shape (batch,)
            
        Returns:
            Dictionary with loss, accuracy, and other metrics
        """
        self.vision_encoder.train()
        self.language_encoder.train()
        self.fusion_module.train()
        self.policy_network.train()
        
        # Move images to device and ensure correct format
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images).float()
        images = images.to(self.device)
        
        # Ensure images are in (batch, 3, H, W) format
        if len(images.shape) == 4 and images.shape[-1] == 3:
            images = images.permute(0, 3, 1, 2)
        
        actions = actions.to(self.device)
        
        # Forward pass
        vision_emb = self.vision_encoder(images)
        language_emb = self.language_encoder(instructions)
        fused_emb = self.fusion_module(vision_emb, language_emb)
        action_logits = self.policy_network(fused_emb)
        
        # Compute loss
        loss = torch.nn.functional.cross_entropy(action_logits, actions)
        
        # Compute accuracy
        predicted_actions = torch.argmax(action_logits, dim=1)
        accuracy = (predicted_actions == actions).float().mean().item()
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return {
            "loss": loss.item(),
            "accuracy": accuracy
        }
    
    def train_epoch(self,
                   dataloader,
                   log_interval: int = 10) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            dataloader: DataLoader providing batches of (images, instructions, actions)
            log_interval: Interval for logging metrics
            
        Returns:
            Dictionary with average metrics for the epoch
        """
        epoch_losses = []
        epoch_accuracies = []
        
        for batch_idx, batch in enumerate(dataloader):
            images, instructions, actions = batch
            
            # Training step
            metrics = self.train_step(images, instructions, actions)
            
            epoch_losses.append(metrics["loss"])
            epoch_accuracies.append(metrics["accuracy"])
            
            # Logging
            if (batch_idx + 1) % log_interval == 0:
                print(
                    f"Epoch {self.epoch + 1}, Batch {batch_idx + 1}/{len(dataloader)}, "
                    f"Loss: {metrics['loss']:.4f}, Accuracy: {metrics['accuracy']:.4f}"
                )
        
        # Average metrics for the epoch
        avg_loss = np.mean(epoch_losses)
        avg_accuracy = np.mean(epoch_accuracies)
        
        self.train_losses.append(avg_loss)
        self.train_accuracies.append(avg_accuracy)
        
        return {
            "loss": avg_loss,
            "accuracy": avg_accuracy
        }
    
    def save_checkpoint(self, filename: Optional[str] = None) -> str:
        """
        Save training checkpoint.
        
        Args:
            filename: Optional checkpoint filename. If None, uses timestamp.
            
        Returns:
            Path to saved checkpoint
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"checkpoint_epoch_{self.epoch}_{timestamp}.pt"
        
        checkpoint_path = self.checkpoint_dir / filename
        
        checkpoint = {
            "epoch": self.epoch,
            "vision_encoder_state_dict": self.vision_encoder.state_dict(),
            "language_encoder_state_dict": self.language_encoder.state_dict(),
            "fusion_module_state_dict": self.fusion_module.state_dict(),
            "policy_network_state_dict": self.policy_network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "train_losses": self.train_losses,
            "train_accuracies": self.train_accuracies,
        }
        
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")
        
        return str(checkpoint_path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Load training checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.vision_encoder.load_state_dict(checkpoint["vision_encoder_state_dict"])
        self.language_encoder.load_state_dict(checkpoint["language_encoder_state_dict"])
        self.fusion_module.load_state_dict(checkpoint["fusion_module_state_dict"])
        self.policy_network.load_state_dict(checkpoint["policy_network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epoch = checkpoint["epoch"]
        self.train_losses = checkpoint.get("train_losses", [])
        self.train_accuracies = checkpoint.get("train_accuracies", [])
        
        print(f"Checkpoint loaded from {checkpoint_path}")
        print(f"Resuming from epoch {self.epoch}")
    
    def plot_training_curves(self, save_path: Optional[str] = None):
        """
        Plot and save training curves (loss and accuracy).
        
        Args:
            save_path: Optional path to save the plot. If None, saves to checkpoint_dir.
        """
        if len(self.train_losses) == 0:
            print("No training data to plot")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss curve
        epochs = range(1, len(self.train_losses) + 1)
        ax1.plot(epochs, self.train_losses, 'b-', label='Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy curve
        ax2.plot(epochs, self.train_accuracies, 'r-', label='Training Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.checkpoint_dir / "training_curves.png"
        else:
            save_path = Path(save_path)
        
        plt.savefig(save_path)
        print(f"Training curves saved to {save_path}")
        plt.close()
    
    def train(self,
              train_dataloader,
              num_epochs: int,
              save_interval: int = 10,
              log_interval: int = 10):
        """
        Main training loop.
        
        Args:
            train_dataloader: DataLoader for training data
            num_epochs: Number of epochs to train
            save_interval: Interval for saving checkpoints (in epochs)
            log_interval: Interval for logging metrics (in batches)
        """
        print(f"Starting training for {num_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Checkpoint directory: {self.checkpoint_dir}")
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            
            print(f"\n{'='*50}")
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"{'='*50}")
            
            # Train one epoch
            metrics = self.train_epoch(train_dataloader, log_interval=log_interval)
            
            # Log epoch summary
            print(
                f"\nEpoch {epoch + 1} Summary: "
                f"Loss: {metrics['loss']:.4f}, Accuracy: {metrics['accuracy']:.4f}"
            )
            
            # Save checkpoint
            if (epoch + 1) % save_interval == 0 or epoch == num_epochs - 1:
                self.save_checkpoint()
                self.plot_training_curves()
        
        print("\nTraining completed!")
        print(f"Final loss: {self.train_losses[-1]:.4f}")
        print(f"Final accuracy: {self.train_accuracies[-1]:.4f}")

