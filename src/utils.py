"""
src/utils.py
============
General utility functions used across all notebooks and modules.
Contents:
- set_seed(): Set random seeds for reproducibility (Python, NumPy, PyTorch, CUDA)
- seed_worker(): Worker init function for reproducible DataLoaders
- create_deterministic_dataloader(): Helper to build reproducible PyTorch DataLoaders
- log_experiment(): Log metrics and configs to CSV and JSON files
- get_device(): Return 'cuda' if GPU available, else 'cpu'
- save_figure(): Save matplotlib figures to outputs/figures/
"""

import os
import random
import csv
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

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
        # Ensure deterministic behavior in CUDNN
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed}")

def seed_worker(worker_id: int) -> None:
    """
    Worker initialization function for PyTorch DataLoaders to ensure deterministic
    sampling inside workers.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def create_deterministic_dataloader(
    dataset,
    batch_size: int,
    shuffle: bool = False,
    sampler = None,
    num_workers: int = 0,
    seed: int = 42
) -> DataLoader:
    """
    Create a PyTorch DataLoader configured with deterministic generator and worker seeds.
    """
    if shuffle and sampler is not None:
        raise ValueError("shuffle and sampler cannot be set together")
        
    g = torch.Generator()
    g.manual_seed(seed)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        worker_init_fn=seed_worker if num_workers > 0 else None,
        generator=g
    )

def log_experiment(
    name: str,
    config: dict,
    metrics: dict,
    csv_path: str = None,
    json_path: str = None
) -> None:
    """
    Log experiment configuration and metrics to CSV and JSON formats.
    """
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    
    if csv_path is None:
        csv_path = os.path.join(project_root, "outputs", "reports", "experiment_logs.csv")
    if json_path is None:
        json_path = os.path.join(project_root, "outputs", "reports", "experiment_logs.json")
        
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    
    # 1. Log to CSV
    row = {"experiment_name": name}
    row.update({f"config_{k}": v for k, v in config.items()})
    row.update({f"metric_{k}": v for k, v in metrics.items()})
    
    file_exists = os.path.isfile(csv_path)
    # Get all existing headers if file exists, to handle column mismatches gracefully
    headers = list(row.keys())
    if file_exists:
        try:
            with open(csv_path, 'r', newline='') as f:
                reader = csv.reader(f)
                existing_headers = next(reader)
                # Merge headers
                for h in existing_headers:
                    if h not in headers:
                        headers.append(h)
        except Exception:
            pass
            
    # Read existing rows to rewrite with complete merged headers if necessary
    existing_rows = []
    if file_exists:
        try:
            with open(csv_path, 'r', newline='') as f:
                reader = csv.DictReader(f)
                existing_rows = list(reader)
        except Exception:
            pass
            
    # Append the new row
    existing_rows.append(row)
    
    # Write back all rows with the full combined headers
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for r in existing_rows:
            # write only keys present in headers, fill others with empty string
            writer.writerow({k: r.get(k, "") for k in headers})
            
    # 2. Log to JSON
    all_runs = []
    if os.path.isfile(json_path):
        try:
            with open(json_path, 'r') as f:
                all_runs = json.load(f)
        except Exception:
            pass
            
    all_runs.append({
        "experiment_name": name,
        "config": config,
        "metrics": metrics
    })
    
    with open(json_path, 'w') as f:
        json.dump(all_runs, f, indent=4)
        
    print(f"Logged experiment metrics to {csv_path} and {json_path}")

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
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    out_dir = os.path.join(project_root, "outputs", "figures")
    os.makedirs(out_dir, exist_ok=True)
    filepath = os.path.join(out_dir, filename)
    fig.savefig(filepath, dpi=300, bbox_inches="tight")
    print(f"Figure saved to {filepath}")
