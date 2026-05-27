import torch
import torch.nn as nn

class VariationalAutoencoder(nn.Module):
    """
    A Variational Autoencoder (VAE) architecture customized for 
    generating synthetic tabular financial transaction records.
    """
    def __init__(self, input_dim, hidden_dim=32, latent_dim=4):
        super(VariationalAutoencoder, self).__init__()
        
        # --- Encoder Layers ---
        # Compresses input features into a shared hidden representation
        self.encoder_hidden = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        # Latent space fully connected layers (Mean and Log-Variance)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # --- Decoder Layers ---
        # Reconstructs the compressed latent vectors back to the original layout
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim) # Outputs reconstructed raw features
        )
        
    def encode(self, x):
        """Passes the input through the encoder to extract distribution parameters."""
        hidden = self.encoder_hidden(x)
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """
        Applies the reparameterization trick to allow backpropagation.
        Samples from N(mu, var) via mu + epsilon * std.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std) # Stochastic noise component
        return mu + eps * std

    def decode(self, z):
        """Maps latent space samples back onto the original feature dimensions."""
        return self.decoder(z)

    def forward(self, x):
        """Executes the complete VAE end-to-end forward pass pipeline."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstructed_x = self.decode(z)
        return reconstructed_x, mu, logvar