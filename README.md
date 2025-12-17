# Vision-Language-Action Navigation

> This project implements a Vision-Language-Action (VLA) system for robotic navigation in a simulated environment. The system designs an embodied AI agent that can execute natural language navigation instructions using visual perception and learned decision-making policies.

![Agent Demonstration](https://github.com/I2S9/Vision-language-action-navigation/raw/main/demos/demonstration.gif)

The GIF above shows the agent navigating in a simulated environment while following the instruction "Navigate to the green goal". Each frame displays the current observation, the instruction, the selected action, and the cumulative reward.

## Problem Statement

Traditional navigation systems require explicit programming or mapping, limiting their flexibility. This project addresses the challenge of creating an agent that can understand natural language instructions and translate them into navigation actions using only visual observations. The agent must generalize to unseen environments and follow diverse instructions without task-specific training.

## Architecture

The system follows a modular architecture that separates perception, language understanding, and action selection into distinct components.

### Vision Encoder

A simple convolutional neural network processes RGB images from the environment. The encoder uses three convolutional layers with ReLU activations to extract visual features, followed by a fully connected layer that produces fixed-size embeddings. Input images are normalized from uint8 [0, 255] to float [0, 1] for stable training.

### Language Encoder

Natural language instructions are encoded using a pre-trained sentence transformer (all-MiniLM-L6-v2). The model's weights are frozen during training to leverage learned semantic representations while keeping the system lightweight. Embeddings are normalized to unit length for consistent fusion.

### Fusion Module

Visual and language embeddings are concatenated along the feature dimension, then processed through a two-layer MLP. This simple fusion strategy preserves information from both modalities while allowing the network to learn effective combinations for navigation decisions.

### Policy Network

The fused representation is mapped to discrete action logits through another MLP. The policy network outputs logits for four actions: turn left, turn right, move forward, and pick up/drop/toggle. Actions are selected using argmax during evaluation.

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

The project requires:
- PyTorch for model implementation
- Gymnasium and gymnasium-minigrid for the simulation environment
- sentence-transformers for language encoding
- imageio and Pillow for visualization

## Usage

### Training

Train the policy network on navigation tasks:

```bash
python training/train.py --epochs 50 --batch_size 32 --num_samples 1000
```

The training script creates a synthetic dataset and trains all trainable components (vision encoder, fusion module, and policy network) end-to-end using cross-entropy loss. Checkpoints are saved automatically at specified intervals.

### Evaluation

Evaluate the trained model on unseen environments:

```bash
python evaluation/evaluate_main.py --checkpoint checkpoints/final_checkpoint.pt --num_episodes 20
```

The evaluation computes success rate, average episode length, and per-environment metrics. Results are saved to a JSON file for analysis.

### Demonstration

Generate a visual demonstration (GIF or video):

```bash
python demos/demo_main.py --checkpoint checkpoints/final_checkpoint.pt --output demos/demo.gif --format gif
```

The demonstration script runs the agent in an environment and creates an annotated visualization showing the agent's behavior.

## Design Rationale

The architecture prioritizes simplicity and interpretability over complexity. Each component has a clear, single responsibility, making the system easy to understand and modify. The use of a pre-trained language encoder reduces training requirements while providing strong semantic representations. The simple concatenation-based fusion avoids complex attention mechanisms that would obscure the decision-making process.

Training uses supervised learning with synthetic data, which allows for rapid iteration and testing. While this limits the diversity of training scenarios, it provides a controlled environment for validating the core architecture before scaling to more complex data collection methods.

## Evaluation

The system is evaluated on multiple unseen MiniGrid environments with different layouts and instructions. The primary metric is task success rate, calculated as the fraction of episodes where the agent reaches the goal. Additional metrics include average episode length, reward statistics, and per-environment breakdowns.

Evaluation results are saved with full trajectory information, enabling detailed analysis of failure modes and successful strategies. The system is designed to be reproducible through fixed random seeds and explicit configuration.

## Limitations

The current implementation has several limitations that reflect its research-oriented design. The synthetic training data does not capture the full complexity of real-world navigation scenarios. The simple fusion strategy may not scale to more complex multi-modal reasoning tasks. The policy network uses a basic MLP that may struggle with long-horizon planning.

The system is evaluated only in simulated grid-world environments, which have limited visual complexity compared to real-world scenes. Generalization to more realistic environments would require additional training data and potentially architectural modifications. The discrete action space limits the agent's ability to perform fine-grained movements.

Future work could address these limitations through reinforcement learning for more diverse exploration, attention mechanisms for better fusion, and continuous action spaces for smoother navigation.

## Project Structure

```
vision-language-action-navigation/
├── env/              # Environment wrappers and utilities
├── models/           # Vision, language, fusion, and policy models
├── training/         # Training loops and optimization
├── evaluation/       # Evaluation scripts and metrics
├── demos/            # Visualization and demonstration scripts
├── utils/            # Configuration and utility functions
└── checkpoints/      # Saved model checkpoints
```