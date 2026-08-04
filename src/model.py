import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    """
    Arquitectura Autoencoder para Reducción Dimensional y Detección de Anomalías.
    Estructura simétrica de compresión (Encoder) y reconstrucción (Decoder).
    """

    def __init__(
        self,
        input_dim: int = 165,
        bottleneck_dim: int = 16,
        dropout_rate: float = 0.1,
        hidden_dims: tuple | list | None = None,
    ):
        super(Autoencoder, self).__init__()

        if hidden_dims is None:
            hidden_dims = (64, 32)
        hidden_dims = list(hidden_dims)

        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.LeakyReLU(0.2),
                ]
            )
            if dropout_rate > 0:
                encoder_layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim

        encoder_layers.extend([nn.Linear(prev_dim, bottleneck_dim), nn.LeakyReLU(0.2)])
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        prev_dim = bottleneck_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.LeakyReLU(0.2),
                ]
            )
            if dropout_rate > 0:
                decoder_layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim

        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed