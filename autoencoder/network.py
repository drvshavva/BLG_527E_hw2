import torch.nn as nn


# 2. Autoencoder Implementation
class Encoder(nn.Module):
    """Encoder module - compresses 28x28 images to latent space"""

    def __init__(self, latent_dim=64):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten image
        return self.encoder(x)


class Decoder(nn.Module):
    """Decoder module - reconstructs images from latent space"""

    def __init__(self, latent_dim=64):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 28 * 28),
            nn.Sigmoid()  # Output between [0, 1]
        )

    def forward(self, x):
        x = self.decoder(x)
        return x.view(x.size(0), 1, 28, 28)  # Reshape to image format


class Autoencoder(nn.Module):
    """Complete autoencoder combining encoder and decoder"""

    def __init__(self, latent_dim=64):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
        self.latent_dim = latent_dim

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)
