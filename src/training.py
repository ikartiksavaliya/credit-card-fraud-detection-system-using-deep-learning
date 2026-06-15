"""
src/training.py
===============
Training loop, optimizer configuration, schedulers, and early stopping utilities
for the Credit Card Fraud Detection System.

Contents:
- Trainer class: Encapsulates training + validation loops
- EarlyStopping: Callback-style early stopping
- get_optimizer(): Factory function for all optimizers
- get_scheduler(): Factory function for LR schedulers (TBD in Phase 6)
- compute_class_weights(): For weighted BCE loss (TBD in Phase 9)
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
from typing import Dict, List, Any

class EarlyStopping:
    """
    Monitors validation loss and stops training if it doesn't improve after a patience threshold.
    Saves the best model weights to disk.
    """
    def __init__(self, patience: int = 5, min_delta: float = 0.0, checkpoint_path: str = "outputs/models/best_model.pt"):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved. Defaults to 5.
            min_delta (float): Minimum change in monitored quantity to qualify as an improvement. Defaults to 0.0.
            checkpoint_path (str): Filepath to save the best model weights. Defaults to 'outputs/models/best_model.pt'.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.checkpoint_path = checkpoint_path
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss: float, model: nn.Module):
        # Check if loss improved
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            # Save model checkpoint
            dir_name = os.path.dirname(self.checkpoint_path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
            torch.save(model.state_dict(), self.checkpoint_path)
            print(f"  Validation loss decreased to {val_loss:.6f}. Model saved to {self.checkpoint_path}")
        else:
            self.counter += 1
            print(f"  EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

class Trainer:
    """
    Encapsulates PyTorch training and validation loops, metric tracking, and early stopping.
    """
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        scheduler: Any = None,
        early_stopping: EarlyStopping = None
    ):
        """
        Args:
            model (nn.Module): The PyTorch neural network to train.
            criterion (nn.Module): The loss function.
            optimizer (torch.optim.Optimizer): The optimizer.
            device (torch.device): Device to run training on (cpu or cuda).
            scheduler (Any): Optional learning rate scheduler.
            early_stopping (EarlyStopping): Optional EarlyStopping callback.
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.early_stopping = early_stopping

    def train_epoch(self, loader: DataLoader) -> float:
        """Runs a single epoch of training using mini-batches."""
        self.model.train()
        total_loss = 0.0
        
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            logits = self.model(X_batch).squeeze(-1)
            loss = self.criterion(logits, y_batch.float())
            
            # Backward pass & step
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item() * X_batch.size(0)
            
        return total_loss / len(loader.dataset)

    def val_epoch(self, loader: DataLoader) -> float:
        """Runs validation loop on a dataset split."""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                logits = self.model(X_batch).squeeze(-1)
                loss = self.criterion(logits, y_batch.float())
                total_loss += loss.item() * X_batch.size(0)
                
        return total_loss / len(loader.dataset)

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int) -> Dict[str, List[float]]:
        """
        Train the model for a specified number of epochs.
        
        Args:
            train_loader (DataLoader): DataLoader for the training set.
            val_loader (DataLoader): DataLoader for the validation set.
            epochs (int): Number of epochs to train for.
            
        Returns:
            Dict: History dictionary containing list of 'train_loss' and 'val_loss'.
        """
        history = {"train_loss": [], "val_loss": []}
        
        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.val_epoch(val_loader)
            
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            
            if self.scheduler:
                self.scheduler.step()
                
            print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
            
            if self.early_stopping:
                self.early_stopping(val_loss, self.model)
                if self.early_stopping.early_stop:
                    print("Early stopping triggered. Training stopped.")
                    # Load best weights back into model
                    self.model.load_state_dict(torch.load(self.early_stopping.checkpoint_path))
                    break
                    
        return history

def get_optimizer(
    model: nn.Module,
    opt_name: str = "adam",
    lr: float = 0.001,
    weight_decay: float = 0.0
) -> torch.optim.Optimizer:
    """
    Factory function for PyTorch optimizers.
    
    Args:
        model (nn.Module): The model containing parameters to optimize.
        opt_name (str): Name of the optimizer ('adam', 'sgd', 'rmsprop', 'adamw'). Defaults to 'adam'.
        lr (float): Learning rate. Defaults to 0.001.
        weight_decay (float): L2 regularization factor. Defaults to 0.0.
        
    Returns:
        torch.optim.Optimizer: Configured optimizer.
    """
    opt_name = opt_name.lower()
    if opt_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif opt_name == "rmsprop":
        return torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")

def compute_class_weights(y: pd.Series) -> float:
    """
    Compute positive class weight (pos_weight) for BCEWithLogitsLoss.
    Formula: pos_weight = negative_samples / positive_samples
    
    Args:
        y (pd.Series): Target class series.
        
    Returns:
        float: Calculated positive class weight.
    """
    neg_count = (y == 0).sum()
    pos_count = (y == 1).sum()
    if pos_count == 0:
        return 1.0
    return float(neg_count) / pos_count
