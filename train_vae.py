import torch
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd

# Import the modules we built previously
# (Ensure preprocessor.py and vae_model.py are in the same directory)
from preprocessor import load_and_sample, preprocess_for_vae, create_data_loader
from vae_model import VariationalAutoencoder

def vae_loss_function(reconstructed_x, original_x, mu, logvar):
    """
    Computes the Evidence Lower Bound (ELBO) Loss for the VAE.
    Since our tabular data is standardized (mean=0, variance=1) via StandardScaler,
    we use Mean Squared Error (MSE) for the reconstruction loss.
    """
    # 1. Reconstruction Loss: How well did the decoder recreate the original input?
    recon_loss = F.mse_loss(reconstructed_x, original_x, reduction='sum')
    
    # 2. Kullback-Leibler Divergence (KLD): Forces the latent space to follow a normal distribution
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Total Loss
    return recon_loss + kld_loss

def train_vae(file_path, epochs=15, batch_size=64, learning_rate=1e-3):
    """Main training loop for the Variational Autoencoder."""
    print("=== Phase 1: Data Preparation ===")
    df_initial = load_and_sample(file_path, n_samples=50000) # Using 50k rows for faster training
    df_processed, dataset_scaler = preprocess_for_vae(df_initial)
    train_loader = create_data_loader(df_processed, batch_size=batch_size)
    
    # Dynamically determine the input dimension based on the processed dataframe
    input_dimension = df_processed.shape[1]
    
    print("\n=== Phase 2: Initializing Model ===")
    model = VariationalAutoencoder(input_dim=input_dimension, hidden_dim=32, latent_dim=4)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Set model to training mode
    model.train()
    
    print("\n=== Phase 3: Commencing Training Loop ===")
    for epoch in range(epochs):
        train_loss = 0.0
        
        for batch_idx, (data,) in enumerate(train_loader):
            # Zero the gradients to prevent accumulation from previous iterations
            optimizer.zero_grad()
            
            # Forward pass: push data through the network
            reconstructed_batch, mu, logvar = model(data)
            
            # Calculate the mathematical loss (ELBO)
            loss = vae_loss_function(reconstructed_batch, data, mu, logvar)
            
            # Backward pass: compute gradients
            loss.backward()
            
            # Update network weights
            train_loss += loss.item()
            optimizer.step()
            
        # Calculate average loss for the epoch
        avg_loss = train_loss / len(train_loader.dataset)
        print(f"Epoch [{epoch + 1}/{epochs}] | Average Loss: {avg_loss:.4f}")

    print("\n=== Phase 4: Saving Model Weights ===")
    # Save the trained model parameters so we can load them later in the Streamlit app
    torch.save(model.state_dict(), 'vae_weights.pth')
    print("Training complete! Model saved as 'vae_weights.pth'.")
    
    return model, dataset_scaler

if __name__ == "__main__":
    # Replace with the actual path to your PaySim Kaggle CSV file
    DATASET_PATH = 'your_dataset_file.csv' 
    trained_model, final_scaler = train_vae(DATASET_PATH, epochs=15)