"""
Random seed management for reproducibility.

This module provides utilities for setting random seeds
across different libraries to ensure reproducible experiments.
"""

import random
import numpy as np
import torch


def set_seed(seed: int):
    """
    Set random seeds for reproducibility.
    
    Sets seeds for Python's random module, NumPy, and PyTorch
    to ensure consistent results across runs.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

