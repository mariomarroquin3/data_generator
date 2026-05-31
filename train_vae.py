import torch
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import joblib # Exportador del escalador matemático

# Importaciones locales
from preprocessor import load_and_sample, preprocess_for_vae, create_data_loader
from vae_model import VariationalAutoencoder

def vae_loss_function(reconstructed_x, original_x, mu, logvar):
    """
    Computes the Evidence Lower Bound (ELBO) Loss for the VAE.
    """
    recon_loss = F.mse_loss(reconstructed_x, original_x, reduction='sum')
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kld_loss

def train_vae(file_path, epochs=20, batch_size=64, learning_rate=1e-3):
    print("=== Phase 1: Data Preparation ===")
    df_initial = load_and_sample(file_path, n_samples=50000)
    df_processed, dataset_scaler = preprocess_for_vae(df_initial)
    train_loader = create_data_loader(df_processed, batch_size=batch_size)
    
    input_dimension = df_processed.shape[1]
    
    print("\n=== Phase 2: Initializing Model ===")
    model = VariationalAutoencoder(input_dim=input_dimension, hidden_dim=32, latent_dim=4)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    model.train()
    
    print("\n=== Phase 3: Commencing Training Loop ===")
    for epoch in range(epochs):
        train_loss = 0.0
        
        for batch_idx, (data,) in enumerate(train_loader):
            optimizer.zero_grad()
            
            reconstructed_batch, mu, logvar = model(data)
            loss = vae_loss_function(reconstructed_batch, data, mu, logvar)
            
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            
        avg_loss = train_loss / len(train_loader.dataset)
        print(f"Epoch [{epoch + 1}/{epochs}] | Average Loss: {avg_loss:.4f}")

    print("\n=== Phase 4: Saving Model Weights and Scaler ===")
    # Guardamos los pesos de la red neuronal
    torch.save(model.state_dict(), 'vae_weights.pth')
    print("Training complete! Model saved as 'vae_weights.pth'.")
    
    # Guardamos el escalador para que tu app local sepa cómo revertir los datos
    joblib.dump(dataset_scaler, 'vae_scaler.pkl')
    print("Scaler saved as 'vae_scaler.pkl'.")
    
    return model, dataset_scaler

if __name__ == "__main__":
    # Sustituye por el nombre exacto de tu archivo en Colab
    DATASET_PATH = 'dataset.csv' 
    trained_model, final_scaler = train_vae(DATASET_PATH, epochs=20)