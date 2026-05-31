import pandas as pd
import numpy as np 
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

def load_and_sample(file_path, n_samples=100000):
    print("Loading original dataset...")
    df = pd.read_csv(file_path)
    
    # --- VALIDACIÓN DE COLUMNAS CRÍTICAS ---
    required_columns = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'type']
    missing_cols = [col for col in required_columns if col not in df.columns]
    
    if missing_cols:
        raise ValueError(f"The uploaded dataset is missing mandatory columns: {missing_cols}")
    
    # Muestreo aleatorio estructurado
    df_sample = df.sample(n=min(n_samples, len(df)), random_state=42).reset_index(drop=True)
    return df_sample

def preprocess_for_vae(df):
    print("Starting data preprocessing and defensive cleaning...")
    
    # Clonamos el DataFrame para evitar el molesto SettingWithCopyWarning
    df = df.copy()
    
    # 1. Eliminar filas duplicadas si las hay
    df = df.drop_duplicates().reset_index(drop=True)
    
    # 2. Manejo defensivo de valores nulos (NaN)
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
            
    # Si hay nulos en variables categóricas, los llenamos con la moda
    if df['type'].isnull().sum() > 0:
        df['type'] = df['type'].fillna(df['type'].mode()[0])

    # --- FIJAR CATEGORÍAS ANTES DE DUMMIES ---
    # Esto garantiza que df_encoded SIEMPRE tenga 11 columnas (5 de categorías + 5 numéricas)
    # sin importar la muestra aleatoria que se extraiga.
    all_categories = ['PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEBIT', 'CASH_IN']
    df['type'] = pd.Categorical(df['type'], categories=all_categories)

    # --- TRANSFORMACIÓN LOGARÍTMICA ---
    financial_cols = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
    for col in financial_cols:
        if col in df.columns:
            df[col] = np.log1p(df[col]) # Aplica log(x + 1)
    
    # 3. Dropeo de columnas de alta cardinalidad o irrelevantes para el VAE
    columns_to_drop = ['nameOrig', 'nameDest', 'step', 'isFlaggedFraud']
    df_cleaned = df.drop(columns=columns_to_drop, errors='ignore')
    
    # 4. Transformación de categorías a variables dummy (One-Hot Encoding)
    # Al estar predefinido como Categorical, respetará el orden y número de columnas
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
    df_initial = load_and_sample('dataset.csv')
    df_processed, my_scaler = preprocess_for_vae(df_initial)
    train_loader = create_data_loader(df_processed)
    print(f"Everything is ready for the VAE training! Total input features: {df_processed.shape[1]}")