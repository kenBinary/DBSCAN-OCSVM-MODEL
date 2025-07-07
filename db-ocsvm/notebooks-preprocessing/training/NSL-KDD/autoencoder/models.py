import torch
import torch.nn as nn
from typing import List, Optional


class BatchNormAutoencoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 128,
        hidden_dims: List[int] = [64, 32],
        latent_dim: int = 16,
        activation_type: str = "ReLU",
        output_activation_type: Optional[str] = None,
    ) -> None:
        super(BatchNormAutoencoder, self).__init__()

        # Select activation function
        activation: nn.Module
        if activation_type == "ReLU":
            activation = nn.ReLU()
        elif activation_type == "LeakyReLU":
            activation = nn.LeakyReLU()
        elif activation_type == "ELU":
            activation = nn.ELU()
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


class DeepSparseDenoisingAutoencoder(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dims=[
            64,
            32,
            16,
        ],  # Dimensions of encoder layers (decoder will be symmetric)
        activation="relu",
        dropout_rate=0.2,
        noise_factor=0.2,
        sparsity_weight=1e-3,
        sparsity_target=0.05,
    ):
        super(DeepSparseDenoisingAutoencoder, self).__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.bottleneck_dim = hidden_dims[-1]
        self.noise_factor = noise_factor
        self.sparsity_weight = sparsity_weight
        self.sparsity_target = sparsity_target

        # Set activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        else:
            raise ValueError(f"Activation {activation} not supported")

        # Dropout layer for regularization
        self.dropout = nn.Dropout(dropout_rate)

        # Build encoder
        encoder_layers = []
        prev_dim = input_dim

        for dim in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, dim))
            encoder_layers.append(self.activation)
            encoder_layers.append(self.dropout)
            prev_dim = dim

        self.encoder = nn.Sequential(*encoder_layers)

        # Build decoder (symmetric to encoder)
        decoder_layers = []
        hidden_dims_reversed = list(reversed(hidden_dims))

        prev_dim = hidden_dims[-1]  # Start from bottleneck

        for i, dim in enumerate(hidden_dims_reversed):
            if i < len(hidden_dims_reversed) - 1:
                next_dim = hidden_dims_reversed[i + 1]
            else:
                next_dim = input_dim

            decoder_layers.append(nn.Linear(prev_dim, next_dim))

            # Only add activation for all but the last layer
            if i < len(hidden_dims_reversed) - 1:
                decoder_layers.append(self.activation)
                decoder_layers.append(self.dropout)
            else:
                # Output layer activation is sigmoid since data is normalized to [0,1]
                decoder_layers.append(nn.Sigmoid())
                pass

            prev_dim = next_dim

        self.decoder = nn.Sequential(*decoder_layers)

    def add_noise(self, x):
        # Add Gaussian noise for denoising capability
        noise = torch.randn_like(x) * self.noise_factor
        return x + noise

    def forward(self, x):
        # Add noise to input (only during training)
        if self.training:
            x_noisy = self.add_noise(x)
        else:
            x_noisy = x

        # Encode
        encoded = self.encoder(x_noisy)

        # Decode
        decoded = self.decoder(encoded)

        return decoded, encoded

    def get_bottleneck_representation(self, x):
        """Extract the bottleneck features for dimensionality reduction"""
        with torch.no_grad():
            self.eval()  # Set to evaluation mode
            encoded = self.encoder(x)
        return encoded

    def kl_divergence_loss(self, activations):
        """Calculate KL divergence for sparsity constraint"""
        # Average activation across batch
        rho_hat = torch.mean(activations, dim=0)
        # KL divergence between rho_hat and target sparsity level
        kl_loss = self.sparsity_target * torch.log(
            self.sparsity_target / (rho_hat + 1e-10)
        ) + (1 - self.sparsity_target) * torch.log(
            (1 - self.sparsity_target) / (1 - rho_hat + 1e-10)
        )
        return torch.sum(kl_loss)
