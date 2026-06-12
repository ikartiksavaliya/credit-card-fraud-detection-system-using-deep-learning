"""
src/models.py
=============
PyTorch model definitions for the Credit Card Fraud Detection System.

Contents:
- MLP: Configurable Multi-Layer Perceptron
  - Configurable depth (number of hidden layers)
  - Configurable width (neurons per layer)
  - Configurable activation function
  - Configurable dropout rate
  - Optional Batch Normalization
  - Configurable weight initialization (TBD in Phase 7)
"""

import torch
import torch.nn as nn
from typing import List, Union

class MLP(nn.Module):
    """
    Configurable Multi-Layer Perceptron (MLP) for binary classification.
    Outputs raw logits (pre-sigmoid) for numerical stability during training.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        activation: str = "relu",
        dropout_rate: float = 0.0,
        use_batch_norm: bool = False
    ):
        """
        Args:
            input_dim (int): Number of input features.
            hidden_dims (List[int]): List containing size of each hidden layer.
            activation (str): Activation function name ('relu', 'sigmoid', 'tanh', 'leaky_relu', 'elu', 'gelu').
            dropout_rate (float): Dropout probability. Defaults to 0.0 (no dropout).
            use_batch_norm (bool): Whether to apply Batch Normalization after linear layers. Defaults to False.
        """
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Helper to retrieve activation function
        def get_activation(act_name: str) -> nn.Module:
            act_name = act_name.lower()
            if act_name == "relu":
                return nn.ReLU()
            elif act_name == "sigmoid":
                return nn.Sigmoid()
            elif act_name == "tanh":
                return nn.Tanh()
            elif act_name == "leaky_relu":
                return nn.LeakyReLU()
            elif act_name == "elu":
                return nn.ELU()
            elif act_name == "gelu":
                return nn.GELU()
            else:
                raise ValueError(f"Unsupported activation function: {act_name}")
                
        for h_dim in hidden_dims:
            # Linear layer
            layers.append(nn.Linear(prev_dim, h_dim))
            # Batch Normalization if requested
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(h_dim))
            # Activation function
            layers.append(get_activation(activation))
            # Dropout layer if requested
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            prev_dim = h_dim
            
        # Output layer - single neuron representing logit for fraud class
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input features of shape (batch_size, input_dim).
            
        Returns:
            torch.Tensor: Model logits of shape (batch_size, 1).
        """
        return self.network(x)
