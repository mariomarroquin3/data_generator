import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

def load_and_sample(file_path, n_samples=1000):
    print("Loading original dataset...")
    df = pd.read_csv(file_path)
    df_sample = df.sample(n=n_samples, random_state=42).reset_index(drop=True)
    return df_sample

def preprocess_for_vae(df):
    print("Starting data preprocessing...")
    columns_to_drop = ['nameOrig', 'nameDest', 'step', 'isFlaggedFraud']
    df_cleaned = df.drop(columns=columns_to_drop)
    
    df_encoded = pd.get_dummies(df_cleaned, columns=['type'], dtype=float)
    
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_encoded)
    df_final = pd.DataFrame(scaled_data, columns=df_encoded.columns)
    
    return df_final, scaler

def create_data_loader(df, batch_size=64):
    """Converts the processed DataFrame into a PyTorch DataLoader."""
    # Convert pandas DataFrame to PyTorch Tensor
    tensor_data = torch.tensor(df.values, dtype=torch.float32)
    
    # Create a TensorDataset and a DataLoader
    dataset = TensorDataset(tensor_data)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return data_loader

# --- Usage Example ---
if __name__ == "__main__":
    # This block only runs if you execute this file directly
    df_initial = load_and_sample('dataset.csv')
    df_processed, my_scaler = preprocess_for_vae(df_initial)
    train_loader = create_data_loader(df_processed)
    print("Everything is ready for the VAE training!")