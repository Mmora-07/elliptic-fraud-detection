import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    """
    Arquitectura Autoencoder para Reducción Dimensional y Detección de Anomalías.
    Estructura simétrica de compresión (Encoder) y reconstrucción (Decoder).
    """
    def __init__(self, input_dim: int = 165, bottleneck_dim: int = 16, dropout_rate: float = 0.1):
        super(Autoencoder, self).__init__()
        
        # --- ENCODER ---
        # Reduce la dimensionalidad: 165 -> 64 -> 32 -> bottleneck_dim (16)
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
        
        # --- DECODER ---
        # Reconstruye la dimensión original: bottleneck_dim (16) -> 32 -> 64 -> 165
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
            # Sin activación final al trabajar con variables estandarizadas (Z-score)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed