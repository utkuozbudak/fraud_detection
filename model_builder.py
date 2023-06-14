"""
Contains the neural network model for fraud detection task.
"""
import torch
from torch import nn

class FraudDetectionModel(nn.Module):
    """Creates the model for fraud detection task.

    Args:
        input_shape (int): Indicates number of input features.
        output_shape (int): Indicates number of output features.
        hidden_units (int): Indicates number of hidden units in the model.
    """
    def __init__(self,
                 input_shape: int,
                 output_shape: int,
                 hidden_units: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(num_features=input_shape),
            nn.Linear(in_features=input_shape, 
                      out_features=hidden_units),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=hidden_units),
            nn.Linear(in_features=hidden_units, 
                      out_features=hidden_units),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=hidden_units),
            nn.Linear(in_features=hidden_units, 
                      out_features=hidden_units),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=hidden_units),
            nn.Dropout(0.25),
            nn.Linear(in_features=hidden_units, 
                      out_features=output_shape)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)