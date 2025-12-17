"""
Rollout script for visualizing agent behavior.

This module provides functionality to run the trained agent
in the environment, record rollouts, and generate visual demonstrations
(video or GIF) with annotations.
"""

import torch
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import animation
from PIL import Image, ImageDraw, ImageFont
import imageio
from pathlib import Path

from models.vision_encoder import VisionEncoder
from models.language_encoder import LanguageEncoder
from models.fusion import FusionModule
from models.policy import PolicyNetwork


class RolloutRunner:
    """
    Runner for agent rollouts in the environment.
    
    Executes the agent policy, collects trajectories, and generates
    visual demonstrations with annotations.
    """
    
    # Action names for annotations
    ACTION_NAMES = {
        0: "Turn Left",
        1: "Turn Right",
        2: "Move Forward",
        3: "Pick Up/Drop/Toggle"
    }
    
    def __init__(self,
                 vision_encoder: VisionEncoder,
                 language_encoder: LanguageEncoder,
                 fusion_module: FusionModule,
                 policy_network: PolicyNetwork,
                 device: str = "cpu"):
        """
        Initialize the rollout runner.
        
        Args:
            vision_encoder: Vision encoder model
            language_encoder: Language encoder model
            fusion_module: Fusion module
            policy_network: Policy network
            device: Device to run inference on
        """
        self.vision_encoder = vision_encoder.to(device)
        self.language_encoder = language_encoder.to(device)
        self.fusion_module = fusion_module.to(device)
        self.policy_network = policy_network.to(device)
        
        self.device = device
        
        # Set models to evaluation mode
        self.vision_encoder.eval()
        self.language_encoder.eval()
        self.fusion_module.eval()
        self.policy_network.eval()
    
    def select_action(self, image: np.ndarray, instruction: str) -> int:
        """
        Select an action given an observation and instruction.
        
        Args:
            image: Current RGB image observation
            instruction: Natural language instruction
            
        Returns:
            Selected action index
        """
        with torch.no_grad():
            # Convert image to tensor
            if isinstance(image, np.ndarray):
                image_tensor = torch.from_numpy(image).float()
            else:
                image_tensor = image.float()
            
            # Ensure correct shape (batch, channels, height, width)
            if len(image_tensor.shape) == 3:
                image_tensor = image_tensor.unsqueeze(0)
            if image_tensor.shape[-1] == 3:
                image_tensor = image_tensor.permute(0, 3, 1, 2)
            
            image_tensor = image_tensor.to(self.device)
            
            # Encode vision and language
            vision_emb = self.vision_encoder(image_tensor)
            language_emb = self.language_encoder(instruction)
            
            # Fuse representations
            fused_emb = self.fusion_module(vision_emb, language_emb)
            
            # Get action logits
            action_logits = self.policy_network(fused_emb)
            
            # Select action (greedy)
            action = torch.argmax(action_logits, dim=1).item()
            
            return action
    
    def run_rollout(self,
                   env,
                   instruction: str,
                   max_steps: int = 1000,
                   record_images: bool = True) -> Dict[str, Any]:
        """
        Run a single rollout episode and record frames.
        
        Args:
            env: Environment instance
            instruction: Natural language instruction
            max_steps: Maximum number of steps in the episode
            record_images: Whether to record images for visualization
            
        Returns:
            Dictionary with rollout trajectory and metadata
        """
        obs, info = env.reset()
        image = obs["image"]
        env_instruction = obs.get("instruction", instruction)
        if env_instruction:
            instruction = env_instruction
        
        frames = []
        trajectory = []
        total_reward = 0.0
        success = False
        
        # Record initial frame
        if record_images:
            frames.append(image.copy())
        
        for step in range(max_steps):
            # Select action
            action = self.select_action(image, instruction)
            
            # Step environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            next_image = next_obs["image"]
            
            total_reward += reward
            
            # Record step
            trajectory.append({
                "step": step,
                "action": action,
                "action_name": self.ACTION_NAMES.get(action, f"Action {action}"),
                "reward": reward,
                "total_reward": total_reward,
                "terminated": terminated,
                "truncated": truncated
            })
            
            # Record frame
            if record_images:
                frames.append(next_image.copy())
            
            # Check for success
            if terminated:
                success = reward > 0 or terminated
                break
            
            if truncated:
                break
            
            image = next_image
        
        return {
            "trajectory": trajectory,
            "frames": frames if record_images else [],
            "episode_length": len(trajectory),
            "instruction": instruction,
            "success": success,
            "total_reward": total_reward
        }
    
    def annotate_frame(self,
                      frame: np.ndarray,
                      instruction: str,
                      action: Optional[int] = None,
                      step: Optional[int] = None,
                      reward: Optional[float] = None) -> np.ndarray:
        """
        Annotate a frame with instruction and action information.
        
        Args:
            frame: RGB image frame
            instruction: Natural language instruction
            action: Action index (optional)
            step: Step number (optional)
            reward: Current reward (optional)
            
        Returns:
            Annotated frame as numpy array
        """
        # Convert to PIL Image for easier annotation
        if isinstance(frame, np.ndarray):
            pil_image = Image.fromarray(frame)
        else:
            pil_image = frame
        
        # Create a copy for drawing
        annotated = pil_image.copy()
        draw = ImageDraw.Draw(annotated)
        
        # Try to use a default font, fallback to basic if not available
        try:
            font_large = ImageFont.truetype("arial.ttf", 16)
            font_small = ImageFont.truetype("arial.ttf", 12)
        except:
            font_large = ImageFont.load_default()
            font_small = ImageFont.load_default()
        
        # Draw instruction at the top
        instruction_text = f"Instruction: {instruction}"
        draw.rectangle([(5, 5), (pil_image.width - 5, 35)], fill=(0, 0, 0, 200))
        draw.text((10, 10), instruction_text, fill=(255, 255, 255), font=font_large)
        
        # Draw action information at the bottom
        info_lines = []
        if step is not None:
            info_lines.append(f"Step: {step}")
        if action is not None:
            action_name = self.ACTION_NAMES.get(action, f"Action {action}")
            info_lines.append(f"Action: {action_name} ({action})")
        if reward is not None:
            info_lines.append(f"Reward: {reward:.2f}")
        
        if info_lines:
            info_text = " | ".join(info_lines)
            text_bbox = draw.textbbox((0, 0), info_text, font=font_small)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            
            y_pos = pil_image.height - text_height - 10
            draw.rectangle([(5, y_pos - 5), (pil_image.width - 5, pil_image.height - 5)],
                          fill=(0, 0, 0, 200))
            draw.text((10, y_pos), info_text, fill=(255, 255, 255), font=font_small)
        
        # Convert back to numpy array
        return np.array(annotated)
    
    def create_gif(self,
                  frames: List[np.ndarray],
                  output_path: str,
                  fps: int = 2,
                  duration: Optional[float] = None):
        """
        Create a GIF from frames.
        
        Args:
            frames: List of image frames
            output_path: Path to save the GIF
            fps: Frames per second
            duration: Duration per frame in seconds (overrides fps if provided)
        """
        if not frames:
            print("No frames to create GIF")
            return
        
        # Calculate duration
        if duration is None:
            duration = 1.0 / fps
        
        # Ensure frames are uint8
        processed_frames = []
        for frame in frames:
            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)
            processed_frames.append(frame)
        
        # Save as GIF
        imageio.mimsave(output_path, processed_frames, duration=duration)
        print(f"GIF saved to {output_path}")
    
    def create_video(self,
                    frames: List[np.ndarray],
                    output_path: str,
                    fps: int = 2):
        """
        Create a video from frames.
        
        Args:
            frames: List of image frames
            output_path: Path to save the video
            fps: Frames per second
        """
        if not frames:
            print("No frames to create video")
            return
        
        # Ensure frames are uint8
        processed_frames = []
        for frame in frames:
            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)
            processed_frames.append(frame)
        
        # Save as video (MP4)
        imageio.mimsave(output_path, processed_frames, fps=fps, codec='libx264')
        print(f"Video saved to {output_path}")
    
    def generate_demonstration(self,
                              env,
                              instruction: str,
                              output_path: str,
                              max_steps: int = 1000,
                              format: str = "gif",
                              fps: int = 2,
                              annotate: bool = True) -> Dict[str, Any]:
        """
        Generate a visual demonstration (GIF or video) of agent behavior.
        
        Args:
            env: Environment instance
            instruction: Natural language instruction
            output_path: Path to save the demonstration
            max_steps: Maximum number of steps
            format: Output format ("gif" or "mp4")
            fps: Frames per second
            annotate: Whether to annotate frames with instruction and actions
            
        Returns:
            Dictionary with rollout results and metadata
        """
        # Run rollout
        rollout_result = self.run_rollout(
            env=env,
            instruction=instruction,
            max_steps=max_steps,
            record_images=True
        )
        
        frames = rollout_result["frames"]
        trajectory = rollout_result["trajectory"]
        
        # Annotate frames if requested
        if annotate:
            annotated_frames = []
            for i, frame in enumerate(frames):
                # Frame i shows the state after step i-1 (or initial state for i=0)
                if i == 0:
                    # First frame: initial state, no action taken yet
                    action = None
                    step = 0
                    reward = 0.0
                elif i <= len(trajectory):
                    # Frame after action at step i-1
                    action = trajectory[i-1]["action"]
                    step = i - 1
                    reward = trajectory[i-1]["total_reward"]
                else:
                    # Beyond trajectory (shouldn't happen)
                    action = trajectory[-1]["action"] if trajectory else None
                    step = len(trajectory) - 1
                    reward = trajectory[-1]["total_reward"] if trajectory else 0.0
                
                annotated_frame = self.annotate_frame(
                    frame=frame,
                    instruction=instruction,
                    action=action,
                    step=step,
                    reward=reward
                )
                annotated_frames.append(annotated_frame)
            frames = annotated_frames
        
        # Create output directory if needed
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Generate GIF or video
        if format.lower() == "gif":
            self.create_gif(frames, str(output_path), fps=fps)
        elif format.lower() in ["mp4", "video"]:
            self.create_video(frames, str(output_path), fps=fps)
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'gif' or 'mp4'")
        
        # Add output path to results
        rollout_result["output_path"] = str(output_path)
        rollout_result["format"] = format
        
        return rollout_result

