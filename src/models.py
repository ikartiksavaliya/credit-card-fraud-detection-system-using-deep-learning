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
        use_batch_norm: bool = False,
        weight_init: str = "default"
    ):
        """
        Args:
            input_dim (int): Number of input features.
            hidden_dims (List[int]): List containing size of each hidden layer.
            activation (str): Activation function name ('relu', 'sigmoid', 'tanh', 'leaky_relu', 'elu', 'gelu').
            dropout_rate (float): Dropout probability. Defaults to 0.0 (no dropout).
            use_batch_norm (bool): Whether to apply Batch Normalization after linear layers. Defaults to False.
            weight_init (str): Weight initialization name ('default', 'uniform', 'normal', 'xavier', 'xavier_normal', 'kaiming', 'kaiming_normal').
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
        
        # Apply custom weight initialization
        self._initialize_weights(weight_init, activation)
        
    def _initialize_weights(self, init_type: str, activation_name: str) -> None:
        """
        Applies various weight initialization schemes to the linear layers of the network.
        """
        init_type = init_type.lower()
        if init_type == "default":
            # PyTorch's default initialization (kaiming_uniform_ with a=sqrt(5) for Linear)
            return
            
        # Determine nonlinearity name and slope a for PyTorch init functions
        nonlinearity = activation_name.lower()
        if nonlinearity == "relu":
            py_nonlinearity = "relu"
            a = 0.0
        elif nonlinearity == "leaky_relu":
            py_nonlinearity = "leaky_relu"
            a = 0.01  # Default slope for Leaky ReLU in PyTorch
        elif nonlinearity in ["sigmoid", "tanh"]:
            py_nonlinearity = nonlinearity
            a = 0.0
        else:
            # Fallback for ELU/GELU to relu to compute gain
            py_nonlinearity = "relu"
            a = 0.0
            
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Weight initialization
                if init_type in ["uniform", "random_uniform"]:
                    nn.init.uniform_(m.weight, a=-0.05, b=0.05)
                elif init_type in ["normal", "random_normal"]:
                    nn.init.normal_(m.weight, mean=0.0, std=0.05)
                elif init_type in ["xavier", "xavier_uniform"]:
                    nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain(py_nonlinearity, a))
                elif init_type == "xavier_normal":
                    nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain(py_nonlinearity, a))
                elif init_type in ["he", "kaiming", "kaiming_uniform"]:
                    nn.init.kaiming_uniform_(m.weight, a=a, nonlinearity=py_nonlinearity)
                elif init_type in ["he_normal", "kaiming_normal"]:
                    nn.init.kaiming_normal_(m.weight, a=a, nonlinearity=py_nonlinearity)
                else:
                    raise ValueError(f"Unknown weight initialization type: {init_type}")
                
                # Bias initialization (standard practice is setting to zero)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input features of shape (batch_size, input_dim).
            
        Returns:
            torch.Tensor: Model logits of shape (batch_size, 1).
        """
        return self.network(x)


class ResNetBlock(nn.Module):
    """
    A single residual block for tabular data.
    Uses LayerNorm instead of BatchNorm for stability on highly imbalanced data.
    """
    def __init__(self, dim: int, dropout_rate: float = 0.0, use_layer_norm: bool = True):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim) if use_layer_norm else nn.Identity()
        self.activation = nn.LeakyReLU(negative_slope=0.01)
        self.dropout1 = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        
        self.linear2 = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim) if use_layer_norm else nn.Identity()
        self.dropout2 = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear1(x)
        out = self.norm1(out)
        out = self.activation(out)
        out = self.dropout1(out)
        
        out = self.linear2(out)
        out = self.norm2(out)
        out = self.dropout2(out)
        
        # Residual connection followed by activation
        return self.activation(out + x)


class TabularResNet(nn.Module):
    """
    Tabular ResNet consisting of an input projection layer, multiple ResNetBlocks,
    and a final linear output layer.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_blocks: int = 3,
        dropout_rate: float = 0.0,
        use_layer_norm: bool = True,
        weight_init: str = "uniform"
    ):
        super().__init__()
        # Project input to hidden dimension
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.input_activation = nn.LeakyReLU(negative_slope=0.01)
        
        self.blocks = nn.ModuleList([
            ResNetBlock(hidden_dim, dropout_rate, use_layer_norm)
            for _ in range(num_blocks)
        ])
        
        # Classification head
        self.output_layer = nn.Linear(hidden_dim, 1)
        
        self._initialize_weights(weight_init)
        
    def _initialize_weights(self, init_type: str) -> None:
        init_type = init_type.lower()
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if init_type in ["uniform", "random_uniform"]:
                    nn.init.uniform_(m.weight, a=-0.05, b=0.05)
                elif init_type in ["normal", "random_normal"]:
                    nn.init.normal_(m.weight, mean=0.0, std=0.05)
                elif init_type in ["xavier", "xavier_uniform"]:
                    nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain("leaky_relu", 0.01))
                elif init_type == "xavier_normal":
                    nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain("leaky_relu", 0.01))
                elif init_type in ["kaiming", "kaiming_uniform"]:
                    nn.init.kaiming_uniform_(m.weight, a=0.01, nonlinearity="leaky_relu")
                elif init_type == "kaiming_normal":
                    nn.init.kaiming_normal_(m.weight, a=0.01, nonlinearity="leaky_relu")
                
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.input_activation(self.input_layer(x))
        for block in self.blocks:
            out = block(out)
        return self.output_layer(out)


class GatedMLPBlock(nn.Module):
    """
    A single MLP layer with a Gated Linear Unit (GLU) mechanism.
    GLU gating dynamically filters features: GLU(x) = LeakyReLU(Linear_val(x)) * Sigmoid(Linear_gate(x))
    """
    def __init__(self, input_dim: int, output_dim: int, dropout_rate: float = 0.0):
        super().__init__()
        self.linear_val = nn.Linear(input_dim, output_dim)
        self.linear_gate = nn.Linear(input_dim, output_dim)
        
        self.activation = nn.LeakyReLU(negative_slope=0.01)
        self.gate_activation = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        val = self.activation(self.linear_val(x))
        gate = self.gate_activation(self.linear_gate(x))
        out = val * gate
        return self.dropout(out)


class GatedMLP(nn.Module):
    """
    Gated Multi-Layer Perceptron (GLU MLP) for tabular classification.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        dropout_rate: float = 0.0,
        weight_init: str = "uniform"
    ):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(GatedMLPBlock(prev_dim, h_dim, dropout_rate))
            prev_dim = h_dim
            
        self.network = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev_dim, 1)
        
        self._initialize_weights(weight_init)
        
    def _initialize_weights(self, init_type: str) -> None:
        init_type = init_type.lower()
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if init_type in ["uniform", "random_uniform"]:
                    nn.init.uniform_(m.weight, a=-0.05, b=0.05)
                elif init_type in ["normal", "random_normal"]:
                    nn.init.normal_(m.weight, mean=0.0, std=0.05)
                elif init_type in ["xavier", "xavier_uniform"]:
                    nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain("leaky_relu", 0.01))
                elif init_type == "xavier_normal":
                    nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain("leaky_relu", 0.01))
                elif init_type in ["kaiming", "kaiming_uniform"]:
                    nn.init.kaiming_uniform_(m.weight, a=0.01, nonlinearity="leaky_relu")
                elif init_type == "kaiming_normal":
                    nn.init.kaiming_normal_(m.weight, a=0.01, nonlinearity="leaky_relu")
                
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.network(x)
        return self.output_layer(out)
