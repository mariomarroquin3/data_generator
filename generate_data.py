import torch
import pandas as pd
import random
from vae_model import VariationalAutoencoder
from preprocessor import load_and_sample, preprocess_for_vae

def generate_synthetic_transactions(original_file_path, num_samples=5000, output_file="synthetic_data.csv"):
    """Generates pure synthetic financial data using the trained VAE."""
    print("=== Phase 1: Reconstructing the Environment ===")
    
    # We load a tiny sample of the original data just to perfectly recreate 
    # the StandardScaler and dynamically get the exact input dimension.
    # This ensures we don't have to hardcode anything.
    df_dummy = load_and_sample(original_file_path, n_samples=500)
    _, scaler = preprocess_for_vae(df_dummy)
    
    # The input dimension is exactly the number of columns the scaler knows about
    input_dimension = len(scaler.feature_names_in_)
    
    print("\n=== Phase 2: Loading the AI Model ===")
    model = VariationalAutoencoder(input_dim=input_dimension, hidden_dim=32, latent_dim=4)
    model.load_state_dict(torch.load('vae_weights.pth'))
    
    # CRITICAL: Put the model in evaluation mode. 
    # This locks the weights and disables training-specific behaviors.
    model.eval() 
    
    print(f"\n=== Phase 3: Generating {num_samples} Synthetic Records ===")
    # 1. Sample pure random noise from the latent space (Standard Normal Distribution)
    with torch.no_grad(): # We don't need gradients for generating, saves memory
        z_random = torch.randn(num_samples, 4) # 4 is our latent_dim
        
        # 2. Decode the noise into scaled transactional data
        scaled_synthetic_tensors = model.decode(z_random)
        
    print("=== Phase 4: Formatting and Exporting ===")
    # 3. Convert tensors back to NumPy and reverse the scaling
    scaled_synthetic_data = scaled_synthetic_tensors.numpy()
    real_values_data = scaler.inverse_transform(scaled_synthetic_data)
    
    # 4. Reconstruct the Pandas DataFrame
    feature_names = scaler.feature_names_in_
    df_synthetic = pd.DataFrame(real_values_data, columns=feature_names)
    
    # --- Data Cleanup (Business Logic Rules) ---
    # The neural net might output a transaction amount of -5.20 or 0.8 for a One-Hot encoded column.
    # We apply strict business rules to fix these mathematical anomalies.
    
    # A. Fix One-Hot Encoded columns (Must be exactly 0 or 1)
    type_columns = [col for col in df_synthetic.columns if 'type_' in col]
    for col in type_columns:
        df_synthetic[col] = df_synthetic[col].apply(lambda x: 1 if x >= 0.5 else 0)
        
    # B. Fix Amounts and Balances (Cannot be negative)
    numeric_cols = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
    valid_numeric_cols = [c for c in numeric_cols if c in df_synthetic.columns]
    for col in valid_numeric_cols:
        df_synthetic[col] = df_synthetic[col].apply(lambda x: max(0.0, round(x, 2)))

    # 5. Inject High-Cardinality Variables (Account IDs)
    print("Injecting synthetic account identifiers...")
    def generate_fake_account(prefix='C'):
        return f"{prefix}{random.randint(100000000, 999999999)}"
        
    # Adding the columns we explicitly dropped during preprocessing
    df_synthetic['nameOrig'] = [generate_fake_account('C') for _ in range(num_samples)]
    df_synthetic['nameDest'] = [generate_fake_account('M') for _ in range(num_samples)]
    df_synthetic['isFlaggedFraud'] = 0 # Defaulting to 0 for standard synthetic data
    
    # 6. Save final output
    df_synthetic.to_csv(output_file, index=False)
    print(f"\nSuccess! Saved pure synthetic data to '{output_file}'.")
    return df_synthetic

if __name__ == "__main__":
    # Replace with the exact path to your PaySim dataset
    DATASET_PATH = 'your_dataset_file.csv' 
    
    # Run the generator!
    final_df = generate_synthetic_transactions(DATASET_PATH, num_samples=1000)
    
    # Print the first few rows to verify it looks like real financial data
    print("\nPreview of Synthetic Data:")
    print(final_df.head())