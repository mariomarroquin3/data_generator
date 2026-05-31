import torch
import pandas as pd
import numpy as np
import random
import joblib # <-- NUEVO: Para cargar el escalador matemático
from vae_model import VariationalAutoencoder

def generate_synthetic_transactions(uploaded_file, num_samples=5000, output_file="synthetic_data_example.csv"):
    print("=== Phase 1: Loading Original Structure & Scaler ===")
    
    # Solo leemos 5 líneas del archivo para saber cómo ordenarle las columnas al usuario final
    uploaded_file.seek(0)
    df_real_preview = pd.read_csv(uploaded_file, nrows=5)
    original_columns = df_real_preview.columns
    
    # --- EL CAMBIO CRÍTICO ---
    # Cargamos la misma regla de medición exacta que usamos en el entrenamiento
    scaler = joblib.load('vae_scaler.pkl')
    
    print("\n=== Phase 2: Loading the AI Model ===")
    # Nuestro input_dim ahora está garantizado en 11 gracias a las categorías fijas
    model = VariationalAutoencoder(input_dim=11, hidden_dim=32, latent_dim=4)
    model.load_state_dict(torch.load('vae_weights (2).pth', map_location=torch.device('cpu')))
    model.eval() 
    
    print(f"\n=== Phase 3: Generating {num_samples} Synthetic Records ===")
    with torch.no_grad(): 
        z_random = torch.randn(num_samples, 4) 
        scaled_synthetic_tensors = model.decode(z_random)
        
    print("=== Phase 4: Formatting and Exporting ===")
    scaled_synthetic_data = scaled_synthetic_tensors.numpy()
    
    # El scaler devuelve los datos a su estado ANTES de escalar
    real_values_data = scaler.inverse_transform(scaled_synthetic_data)
    
    feature_names = scaler.feature_names_in_
    df_synthetic = pd.DataFrame(real_values_data, columns=feature_names)
    
    # --- Limpieza de Negocio ---
    
    # A. Invertir One-Hot Encoding
    type_columns = [col for col in df_synthetic.columns if col.startswith('type_')]
    if type_columns:
        df_synthetic['type'] = df_synthetic[type_columns].idxmax(axis=1).str.replace('type_', '')
        df_synthetic = df_synthetic.drop(columns=type_columns)
        
    # B. REVERTIR LOGARITMO Y LIMPIAR NUMÉRICOS
    numeric_cols = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
    valid_numeric_cols = [c for c in numeric_cols if c in df_synthetic.columns]
    
    for col in valid_numeric_cols:
        # 1. Función exponencial inversa
        df_synthetic[col] = np.expm1(df_synthetic[col])
        # 2. Forzar piso en cero (clip) para evitar saldos negativos por ruido estadístico
        df_synthetic[col] = df_synthetic[col].clip(lower=0).round(2)

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