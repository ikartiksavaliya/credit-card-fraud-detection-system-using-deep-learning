"""
src/evaluation.py
=================
Evaluation utilities for Credit Card Fraud Detection. Computes metrics (Accuracy,
Precision, Recall, F1, ROC-AUC, PR-AUC) and plots curves (Confusion Matrix,
ROC Curve, PR Curve, Training Curves).

Contents:
- evaluate_model(): Compute metrics suite
- plot_confusion_matrix()
- plot_roc_curve()
- plot_pr_curve()
- plot_training_curves()
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, precision_recall_curve, auc, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Tuple
from src.utils import save_figure

def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
    criterion: nn.Module = nn.BCEWithLogitsLoss()
) -> Dict[str, Any]:
    """
    Evaluate PyTorch model performance on a given dataset split.
    
    Args:
        model (nn.Module): The model to evaluate.
        loader (DataLoader): DataLoader containing evaluation data.
        device (torch.device): Device to run evaluation on.
        threshold (float): Classification threshold. Defaults to 0.5.
        criterion (nn.Module): The loss function. Defaults to BCEWithLogitsLoss.
        
    Returns:
        Dict: Dictionary containing loss, accuracy, precision, recall, f1, roc_auc, pr_auc,
              and arrays for true labels, probabilities, and binary predictions.
    """
    model.eval()
    all_logits = []
    all_targets = []
    
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch).squeeze(-1)
            all_logits.append(logits.cpu())
            all_targets.append(y_batch)
            
    all_logits = torch.cat(all_logits)
    all_targets = torch.cat(all_targets)
    
    # Compute loss (using logits for stability)
    loss = criterion(all_logits.to(device), all_targets.float().to(device)).item()
    
    # Calculate probabilities and predictions
    probs = torch.sigmoid(all_logits).numpy()
    y_true = all_targets.numpy()
    preds = (probs >= threshold).astype(int)
    
    # Sklearn classification metrics
    accuracy = accuracy_score(y_true, preds)
    precision = precision_score(y_true, preds, zero_division=0)
    recall = recall_score(y_true, preds, zero_division=0)
    f1 = f1_score(y_true, preds, zero_division=0)
    
    # Calculate ROC-AUC
    try:
        fpr, tpr, _ = roc_curve(y_true, probs)
        roc_auc = auc(fpr, tpr)
    except ValueError:
        roc_auc = 0.0
        
    # Calculate PR-AUC
    try:
        precisions, recalls, _ = precision_recall_curve(y_true, probs)
        pr_auc = auc(recalls, precisions)
    except ValueError:
        pr_auc = 0.0
        
    return {
        "loss": loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "y_true": y_true,
        "probs": probs,
        "preds": preds
    }

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, save_path: str = None):
    """
    Plot and optionally save a confusion matrix heatmap.
    
    Args:
        y_true (np.ndarray): True labels.
        y_pred (np.ndarray): Predicted binary labels.
        save_path (str): Filename to save plot (relative to outputs/figures/). Defaults to None.
    """
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", cbar=False,
        xticklabels=["Legitimate", "Fraud"],
        yticklabels=["Legitimate", "Fraud"],
        annot_kws={"size": 14}, ax=ax
    )
    
    ax.set_title("Confusion Matrix Heatmap", fontsize=14, fontweight="bold", pad=15)
    ax.set_ylabel("Actual Label", fontsize=12)
    ax.set_xlabel("Predicted Label", fontsize=12)
    
    plt.tight_layout()
    if save_path:
        save_figure(fig, save_path)
    plt.show()

def plot_roc_curve(y_true: np.ndarray, y_probs: np.ndarray, save_path: str = None):
    """
    Plot and optionally save the Receiver Operating Characteristic (ROC) curve.
    
    Args:
        y_true (np.ndarray): True labels.
        y_probs (np.ndarray): Model output probabilities.
        save_path (str): Filename to save plot. Defaults to None.
    """
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="darkorange", lw=2.5, label=f"ROC Curve (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--")
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate (Recall)", fontsize=12)
    ax.set_title("ROC Curve Analysis", fontsize=14, fontweight="bold", pad=15)
    ax.legend(loc="lower right", fontsize=11)
    
    plt.tight_layout()
    if save_path:
        save_figure(fig, save_path)
    plt.show()

def plot_pr_curve(y_true: np.ndarray, y_probs: np.ndarray, save_path: str = None):
    """
    Plot and optionally save the Precision-Recall (PR) curve.
    
    Args:
        y_true (np.ndarray): True labels.
        y_probs (np.ndarray): Model output probabilities.
        save_path (str): Filename to save plot. Defaults to None.
    """
    precisions, recalls, _ = precision_recall_curve(y_true, y_probs)
    pr_auc = auc(recalls, precisions)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recalls, precisions, color="forestgreen", lw=2.5, label=f"PR Curve (AUC = {pr_auc:.4f})")
    
    # Plot random baseline (positive class ratio)
    pos_ratio = y_true.sum() / len(y_true)
    ax.plot([0, 1], [pos_ratio, pos_ratio], color="grey", lw=1.5, linestyle="--", label=f"Baseline ({pos_ratio:.4f})")
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curve Analysis", fontsize=14, fontweight="bold", pad=15)
    ax.legend(loc="upper right", fontsize=11)
    
    plt.tight_layout()
    if save_path:
        save_figure(fig, save_path)
    plt.show()

def plot_training_curves(history: Dict[str, list], save_path: str = None):
    """
    Plot and optionally save training and validation loss curves.
    
    Args:
        history (Dict): Output of Trainer.fit containing train_loss and val_loss.
        save_path (str): Filename to save plot. Defaults to None.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = range(1, len(history["train_loss"]) + 1)
    
    ax.plot(epochs, history["train_loss"], "b-", label="Training Loss", lw=2)
    ax.plot(epochs, history["val_loss"], "r-", label="Validation Loss", lw=2)
    
    ax.set_title("Training & Validation Loss Curve", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Epochs", fontsize=12)
    ax.set_ylabel("Loss (BCE)", fontsize=12)
    ax.set_xlim([1, len(epochs)])
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    if save_path:
        save_figure(fig, save_path)
    plt.show()
