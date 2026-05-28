import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

def load_and_sample(file_path, n_samples=100000):
    print("Loading original dataset...")
    df = pd.read_csv(file_path)
    
    # --- NUEVA VALIDACIÓN DE COLUMNAS CRÍTICAS ---
    required_columns = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'type']
    missing_cols = [col for col in required_columns if col not in df.columns]
    
    if missing_cols:
        raise ValueError(f"The uploaded dataset is missing mandatory columns: {missing_cols}")
    
    # Muestreo aleatorio estructurado
    df_sample = df.sample(n=min(n_samples, len(df)), random_state=42).reset_index(drop=True)
    return df_sample

def preprocess_for_vae(df):
    print("Starting data preprocessing and defensive cleaning...")
    
    # 1. Eliminar filas duplicadas si las hay
    df = df.drop_duplicates().reset_index(drop=True)
    
    # 2. Manejo defensivo de valores nulos (NaN)
    # Si hay nulos en columnas numéricas, los llenamos con la mediana para no romper el Scaler
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
            
    # Si hay nulos en variables categóricas, los llenamos con la moda
    if df['type'].isnull().sum() > 0:
        df['type'] = df['type'].fillna(df['type'].mode()[0])
    
    # 3. Dropeo de columnas de alta cardinalidad o irrelevantes para el VAE
    columns_to_drop = ['nameOrig', 'nameDest', 'step', 'isFlaggedFraud']
    # Usamos errors='ignore' por si el usuario ya subió un dataset sin estas columnas
    df_cleaned = df.drop(columns=columns_to_drop, errors='ignore')
    
    # 4. Transformación de categorías a variables dummy (One-Hot Encoding)
    df_encoded = pd.get_dummies(df_cleaned, columns=['type'], dtype=float)
    
    # 5. Escalamiento estándar (Media 0, Varianza 1)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_encoded)
    df_final = pd.DataFrame(scaled_data, columns=df_encoded.columns)
    
    return df_final, scaler

def create_data_loader(df, batch_size=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensor_data = torch.tensor(df.values, dtype=torch.float32).to(device)
    
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