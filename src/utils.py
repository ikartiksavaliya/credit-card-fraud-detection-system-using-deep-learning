"""
src/utils.py
============
General utility functions used across all notebooks and modules.
Contents (to be filled as notebooks progress):
- set_seed(): Set random seeds for reproducibility (Python, NumPy, PyTorch)
- get_device(): Return 'cuda' if GPU available, else 'cpu'
- save_figure(): Save matplotlib figures to outputs/figures/
- save_model(): Save PyTorch model checkpoint to outputs/models/
- load_model(): Load a saved checkpoint
- format_metrics(): Pretty-print metric dictionary
- count_parameters(): Count trainable parameters in a PyTorch model
"""

import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt

def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility across Python random, NumPy, and PyTorch.
    
    Args:
        seed (int): The seed value to use. Defaults to 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed}")

def get_device() -> torch.device:
    """
    Return 'cuda' device if GPU is available, else 'cpu'.
    
    Returns:
        torch.device: The PyTorch device object.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device

def save_figure(fig, filename: str) -> None:
    """
    Save a matplotlib/seaborn figure to outputs/figures/.
    Creates the directory if it does not exist.
    
    Args:
        fig (matplotlib.figure.Figure): The figure object to save.
        filename (str): The filename (e.g., 'class_distribution.png').
    """
    # Find project root relative to this utils.py file (which is in src/)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    out_dir = os.path.join(project_root, "outputs", "figures")
    os.makedirs(out_dir, exist_ok=True)
    filepath = os.path.join(out_dir, filename)
    fig.savefig(filepath, dpi=300, bbox_inches="tight")
    print(f"Figure saved to {filepath}")
