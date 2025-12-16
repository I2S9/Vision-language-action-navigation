"""
Random seed management for reproducibility.

This module provides utilities for setting random seeds
across different libraries to ensure reproducible experiments.
"""

import random
import numpy as np
import torch
import gymnasium as gym


def set_seed(seed: int):
    """
    Set random seeds for reproducibility.
    
    Sets seeds for Python's random module, NumPy, PyTorch, and Gymnasium
    to ensure consistent results across runs.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Set gymnasium seed
    try:
        gym.utils.seeding.np_random.seed(seed)
    except AttributeError:
        # Fallback if gymnasium API changes
        pass
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

