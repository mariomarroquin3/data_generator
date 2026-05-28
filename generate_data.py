import torch
import pandas as pd
import numpy as np
import random
from vae_model import VariationalAutoencoder
from preprocessor import load_and_sample, preprocess_for_vae

def generate_synthetic_transactions(uploaded_file, num_samples=5000, output_file="synthetic_data.csv"):
    print("=== Phase 1: Reconstructing the Environment ===")
    
    uploaded_file.seek(0)
    df_real_preview = pd.read_csv(uploaded_file, nrows=5)
    original_columns = df_real_preview.columns
    
    # Usamos un chunk grande directo para asegurar mejor cobertura
    uploaded_file.seek(0) 
    df_dummy = pd.read_csv(uploaded_file, nrows=50000) 
    uploaded_file.seek(0) 
    
    # ---------------------------------------------------------
    # FIX 1: ALINEACIÓN FORZADA DE FEATURES 
    # Aseguramos que get_dummies genere las 5 columnas siempre, 
    # incluso si la rara categoría 'DEBIT' no está en la muestra.
    # ---------------------------------------------------------
    if 'type' in df_dummy.columns:
        df_dummy['type'] = pd.Categorical(
            df_dummy['type'], 
            categories=['PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEBIT', 'CASH_IN']
        )
        
    _, scaler = preprocess_for_vae(df_dummy)
    input_dimension_scaler = len(scaler.feature_names_in_)
    
    print("\n=== Phase 2: Loading the AI Model ===")
    # ---------------------------------------------------------
    # FIX 2: CARGA ESTRICTA DE PESOS
    # Forzamos input_dim=11 porque el checkpoint .pth exige exactamente esa dimensión.
    # ---------------------------------------------------------
    model = VariationalAutoencoder(input_dim=11, hidden_dim=32, latent_dim=4)
    model.load_state_dict(torch.load('vae_weights.pth', map_location=torch.device('cpu')))
    model.eval() 
    
    print(f"\n=== Phase 3: Generating {num_samples} Synthetic Records ===")
    with torch.no_grad(): 
        z_random = torch.randn(num_samples, 4) 
        scaled_synthetic_tensors = model.decode(z_random)
        
    print("=== Phase 4: Formatting and Exporting ===")
    scaled_synthetic_data = scaled_synthetic_tensors.numpy()
    
    # ---------------------------------------------------------
    # FIX 3: RECORTADOR DE SEGURIDAD (Padding dinámico)
    # Si el VAE generó 11 datos por fila pero el Scaler solo espera 10,
    # descartamos la columna huérfana para evitar que inverse_transform crashee.
    # ---------------------------------------------------------
    if input_dimension_scaler == 10 and scaled_synthetic_data.shape[1] == 11:
        scaled_synthetic_data = scaled_synthetic_data[:, :10]
        
    real_values_data = scaler.inverse_transform(scaled_synthetic_data)
    
    feature_names = scaler.feature_names_in_
    df_synthetic = pd.DataFrame(real_values_data, columns=feature_names)
    
    # --- Limpieza de Negocio ---
    
    # A. Invertir One-Hot Encoding
    type_columns = [col for col in df_synthetic.columns if col.startswith('type_')]
    if type_columns:
        df_synthetic['type'] = df_synthetic[type_columns].idxmax(axis=1).str.replace('type_', '')
        df_synthetic = df_synthetic.drop(columns=type_columns)
        
    # B. Limpiar Numéricos (Nada de dinero negativo)
    numeric_cols = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
    valid_numeric_cols = [c for c in numeric_cols if c in df_synthetic.columns]
    for col in valid_numeric_cols:
        df_synthetic[col] = df_synthetic[col].abs().round(2)

    # C. Metadatos y Variables de Alta Cardinalidad
    def generate_fake_account(prefix='C'):
        return f"{prefix}{random.randint(100000000, 9999999999)}"
        
    df_synthetic['step'] = np.random.randint(1, 744, size=num_samples)
    df_synthetic['nameOrig'] = [generate_fake_account('C') for _ in range(num_samples)]
    df_synthetic['nameDest'] = [generate_fake_account(random.choice(['C', 'M'])) for _ in range(num_samples)]
    df_synthetic['isFlaggedFraud'] = 0 
    
    # D. Forzar Estructura Original (Drop-in Replacement)
    for col in original_columns:
        if col not in df_synthetic.columns:
            df_synthetic[col] = 0 
            
    df_synthetic = df_synthetic[original_columns]
    
    return df_synthetic