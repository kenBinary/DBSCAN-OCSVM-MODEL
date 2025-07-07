import torch
import torch.nn as nn
from typing import List, Optional


class BatchNormAutoencoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 128,
        hidden_dims: List[int] = [64, 32],
        latent_dim: int = 16,
        negative_slope: float = 0.2,
        activation_type: str = "ReLU",
        output_activation_type: Optional[str] = None,
    ) -> None:
        super(BatchNormAutoencoder, self).__init__()

        # Select activation function
        activation: nn.Module
        if activation_type == "ReLU":
            activation = nn.ReLU()
        elif activation_type == "LeakyReLU":
            activation = nn.LeakyReLU(negative_slope)  # Better negative gradient
        elif activation_type == "ELU":
            activation = nn.ELU()
        elif activation_type == "GELU":  # Adding GELU option
            activation = nn.GELU()
        else:
            raise ValueError("Unknown activation type provided")

        # Build encoder
        encoder_layers: List[nn.Module] = []
        current_dim: int = input_dim
        for h_dim in hidden_dims:
            encoder_layers.append(nn.Linear(current_dim, h_dim))
            encoder_layers.append(nn.BatchNorm1d(h_dim))
            encoder_layers.append(activation)
            current_dim = h_dim

        # Latent layer
        encoder_layers.append(nn.Linear(current_dim, latent_dim))
        self.encoder: nn.Sequential = nn.Sequential(*encoder_layers)

        # Select output activation function
        output_activation: Optional[nn.Module] = None
        if output_activation_type == "ReLU":
            output_activation = nn.ReLU()
        elif output_activation_type == "LeakyReLU":
            output_activation = nn.LeakyReLU()
        elif output_activation_type == "ELU":
            output_activation = nn.ELU()
        elif output_activation_type == "Sigmoid":
            output_activation = nn.Sigmoid()
        elif output_activation_type == "Tanh":
            output_activation = nn.Tanh()
        elif output_activation_type is None:
            output_activation = None
        else:
            raise ValueError("Unknown activation type provided")

        # Build decoder
        decoder_layers: List[nn.Module] = []
        current_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(current_dim, h_dim))
            decoder_layers.append(nn.BatchNorm1d(h_dim))
            decoder_layers.append(activation)
            current_dim = h_dim

        # Add final output layer (no batch norm on output layer)
        decoder_layers.append(nn.Linear(current_dim, input_dim))

        # Add output activation if specified
        if output_activation is not None:
            decoder_layers.append(output_activation)

        self.decoder: nn.Sequential = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded: torch.Tensor = self.encoder(x)
        decoded: torch.Tensor = self.decoder(encoded)
        return decoded

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)
