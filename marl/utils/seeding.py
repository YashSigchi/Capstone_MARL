"""
Seeding utilities for reproducibility.

Status: Implemented
"""
import random
import os
import numpy as np
import torch

def set_all_seeds(seed: int) -> None:
    """Set seeds for reproducibility across all libraries.
    
    Sets: Python random, NumPy, PyTorch (CPU + CUDA).
    Also sets PYTHONHASHSEED and torch deterministic flags.
    
    Status: Implemented
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_rng(seed: int) -> np.random.Generator:
    """Create a numpy Generator with the given seed.
    
    Args:
        seed (int): The seed for the random number generator.
        
    Returns:
        np.random.Generator: A configured NumPy random number generator.
    """
    return np.random.default_rng(seed)
