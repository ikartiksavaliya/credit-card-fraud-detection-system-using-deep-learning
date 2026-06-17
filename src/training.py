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
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
from typing import Dict, List, Any
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, LambdaLR

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
        early_stopping: EarlyStopping = None,
        max_grad_norm: float = None,
        l1_lambda: float = 0.0
    ):
        """
        Args:
            model (nn.Module): The PyTorch neural network to train.
            criterion (nn.Module): The loss function.
            optimizer (torch.optim.Optimizer): The optimizer.
            device (torch.device): Device to run training on (cpu or cuda).
            scheduler (Any): Optional learning rate scheduler.
            early_stopping (EarlyStopping): Optional EarlyStopping callback.
            max_grad_norm (float): Optional maximum gradient norm for clipping. Defaults to None.
            l1_lambda (float): L1 regularization weight. Defaults to 0.0 (no L1 penalty).
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.early_stopping = early_stopping
        self.max_grad_norm = max_grad_norm
        self.l1_lambda = l1_lambda

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
            
            # Add L1 regularization penalty if requested
            if self.l1_lambda > 0.0:
                l1_penalty = sum(p.abs().sum() for p in self.model.parameters())
                loss = loss + self.l1_lambda * l1_penalty
            
            # Backward pass & step
            loss.backward()
            
            # Perform gradient norm clipping if requested
            if self.max_grad_norm is not None:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                
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
    weight_decay: float = 0.0,
    momentum: float = 0.9,
    nesterov: bool = False
) -> torch.optim.Optimizer:
    """
    Factory function for PyTorch optimizers.
    
    Args:
        model (nn.Module): The model containing parameters to optimize.
        opt_name (str): Name of the optimizer ('adam', 'sgd', 'rmsprop', 'adamw', 'adagrad'). Defaults to 'adam'.
        lr (float): Learning rate. Defaults to 0.001.
        weight_decay (float): L2 regularization factor. Defaults to 0.0.
        momentum (float): Momentum factor for SGD and RMSprop. Defaults to 0.9.
        nesterov (bool): Enables Nesterov momentum for SGD. Defaults to False.
        
    Returns:
        torch.optim.Optimizer: Configured optimizer.
    """
    opt_name = opt_name.lower()
    if opt_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, nesterov=nesterov, weight_decay=weight_decay)
    elif opt_name == "rmsprop":
        return torch.optim.RMSprop(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif opt_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_name == "adagrad":
        return torch.optim.Adagrad(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")

def get_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_name: str = "step",
    step_size: int = 10,
    gamma: float = 0.1,
    T_max: int = 50,
    eta_min: float = 0.0,
    warmup_epochs: int = 5,
    total_epochs: int = 50
):
    """
    Factory function for PyTorch learning rate schedulers.
    
    Args:
        optimizer (Optimizer): The optimizer for which to schedule the learning rate.
        scheduler_name (str): Name of the scheduler ('step', 'cosine', 'warmup_cosine').
        step_size (int): Period of learning rate decay for StepLR. Defaults to 10.
        gamma (float): Multiplicative factor of learning rate decay. Defaults to 0.1.
        T_max (int): Maximum number of iterations for CosineAnnealingLR. Defaults to 50.
        eta_min (float): Minimum learning rate for CosineAnnealingLR. Defaults to 0.0.
        warmup_epochs (int): Number of warmup epochs for warmup_cosine. Defaults to 5.
        total_epochs (int): Total training epochs for warmup_cosine. Defaults to 50.
        
    Returns:
        lr_scheduler: Configured PyTorch scheduler.
    """
    scheduler_name = scheduler_name.lower()
    if scheduler_name == "step":
        return StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_name == "cosine":
        return CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
    elif scheduler_name == "warmup_cosine":
        def lr_lambda(current_epoch):
            if current_epoch < warmup_epochs:
                return float(current_epoch + 1) / float(max(1, warmup_epochs))
            progress = float(current_epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
            min_lr_ratio = eta_min / optimizer.defaults['lr'] if optimizer.defaults['lr'] > 0 else 0.0
            return min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (1.0 + math.cos(math.pi * progress))
        return LambdaLR(optimizer, lr_lambda)
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")

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

def run_experiment(
    exp_name: str,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    criterion: nn.Module = None,
    opt_name: str = "adamw",
    lr: float = 0.001,
    weight_decay: float = 0.0,
    momentum: float = 0.9,
    nesterov: bool = False,
    scheduler_name: str = None,
    scheduler_params: dict = None,
    epochs: int = 50,
    patience: int = 5,
    l1_lambda: float = 0.0,
    max_grad_norm: float = None,
    checkpoint_prefix: str = "optimizer"
) -> dict:
    """
    Unified training-evaluation wrapper to run modeling experiments.
    Saves best model checkpoints and logs metrics automatically.
    """
    from src.evaluation import evaluate_model
    from src.utils import log_experiment
    import numpy as np

    # 1. Loss function
    if criterion is None:
        criterion = nn.BCEWithLogitsLoss()
        
    # 2. Optimizer
    optimizer = get_optimizer(
        model, opt_name=opt_name, lr=lr, weight_decay=weight_decay,
        momentum=momentum, nesterov=nesterov
    )
    
    # 3. Scheduler
    scheduler = None
    if scheduler_name:
        params = scheduler_params or {}
        scheduler = get_scheduler(optimizer, scheduler_name=scheduler_name, **params)
        
    # 4. Checkpoint path
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    checkpoint_dir = os.path.join(project_root, "outputs", "models")
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"{checkpoint_prefix}_{exp_name}_best_model.pt")
    
    # 5. Early stopping callback
    if patience is not None and patience > 0:
        early_stopping = EarlyStopping(patience=patience, checkpoint_path=checkpoint_path)
    else:
        early_stopping = None
    
    # 6. Trainer instantiation and training
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        scheduler=scheduler,
        early_stopping=early_stopping,
        max_grad_norm=max_grad_norm,
        l1_lambda=l1_lambda
    )
    
    print(f"\n==================================================")
    print(f"RUNNING EXPERIMENT: {exp_name.upper()}")
    print(f"==================================================")
    
    history = trainer.fit(train_loader, val_loader, epochs=epochs)
    
    # Load best checkpoint weights or save final epoch weights
    if early_stopping is not None:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    else:
        torch.save(model.state_dict(), checkpoint_path)
    
    # Evaluate model on splits
    train_metrics = evaluate_model(model, train_loader, device, criterion=criterion)
    val_metrics = evaluate_model(model, val_loader, device, criterion=criterion)
    test_metrics = evaluate_model(model, test_loader, device, criterion=criterion)
    
    # 7. Log configurations and metrics
    config = {
        "opt_name": opt_name,
        "lr": lr,
        "weight_decay": weight_decay,
        "scheduler_name": scheduler_name or "none",
        "l1_lambda": l1_lambda,
        "max_grad_norm": max_grad_norm or 0.0,
        "checkpoint_path": checkpoint_path
    }
    
    # Flatten metrics to dictionary of floats for logging
    logged_metrics = {}
    for prefix, split_metrics in [("train", train_metrics), ("val", val_metrics), ("test", test_metrics)]:
        for k, v in split_metrics.items():
            if isinstance(v, (int, float, np.float32, np.float64, np.int64)):
                logged_metrics[f"{prefix}_{k}"] = float(v)
            elif isinstance(v, dict):
                for sub_k, sub_v in v.items():
                    if isinstance(sub_v, (int, float, np.float32, np.float64, np.int64)):
                        logged_metrics[f"{prefix}_{k}_{sub_k}"] = float(sub_v)
                        
    log_experiment(exp_name, config, logged_metrics)
    
    return {
        "history": history,
        "train": train_metrics,
        "val": val_metrics,
        "test": test_metrics,
        "epochs_run": len(history["train_loss"])
    }

