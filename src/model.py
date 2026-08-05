import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    """
    Autoencoder en PyTorch configurado para detección de anomalías/fraude
    en el dataset Credit Card.
    """
    def __init__(self, input_dim: int = 29, bottleneck_dim: int = 14, dropout_rate: float = 0.1):
        super(Autoencoder, self).__init__()
        
        # Encoder: compresión progresiva de características
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),
            
            nn.Linear(32, bottleneck_dim),
            nn.LeakyReLU(0.2)
        )
        
        # Decoder: reconstrucción del vector original
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),
            
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),
            
            nn.Linear(64, input_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Obtiene la representación latente (bottleneck)."""
        return self.encoder(x)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Reconstruye la entrada a partir del vector latente."""
        return self.decoder(latent)