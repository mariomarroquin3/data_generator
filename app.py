import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import os
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Import pipelines built previously
from preprocessor import load_and_sample, preprocess_for_vae
from generate_data import generate_synthetic_transactions

st.set_page_config(page_title="AI Financial Data Synthesizer", layout="wide")

st.title("📊 Generative AI Financial Data Synthesizer")
st.markdown("""
This platform utilizes a **Variational Autoencoder (VAE)** to learn the statistical properties of underlying 
financial transaction records and generate completely synthetic, privacy-compliant datasets for software testing.
""")

# Sidebar settings
st.sidebar.header("Configuration Panel")
dataset_path = st.sidebar.text_input("Dataset File Path (Kaggle CSV)", "your_dataset_file.csv")
num_to_generate = st.sidebar.slider("Records to Generate", min_value=500, max_value=5000, value=1000, step=500)

# Check if model weights exist
weights_exist = os.path.exists("vae_weights.pth")
if not weights_exist:
    st.warning("⚠️ No trained model weights found (`vae_weights.pth`). Please run your `train_vae.py` script first to train the backend.")

# Target action button
generate_btn = st.sidebar.button("Generate Synthetic Data", disabled=not weights_exist)

if generate_btn:
    with st.spinner("Executing generation pipeline and decoding latent space..."):
        try:
            # 1. Run the generation script
            df_synthetic = generate_synthetic_transactions(dataset_path, num_samples=num_to_generate)
            
            # 2. Extract a baseline sample of real data for side-by-side comparison
            df_real_raw = load_and_sample(dataset_path, n_samples=num_to_generate)
            
            st.success(f"Successfully generated {num_to_generate} synthetic rows!")
            
            # --- TABS FOR ORGANIZED VIEW ---
            tab_preview, tab_pca, tab_distribution = st.tabs(["📋 Data Preview", "📉 PCA Space Alignment", "📊 Feature Distribution"])
            
            with tab_preview:
                st.subheader("Generated Synthetic Output Preview")
                st.dataframe(df_synthetic.head(50))
                
                # Download link for the generated CSV
                csv_buffer = df_synthetic.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Synthetic CSV Dataset",
                    data=csv_buffer,
                    file_name="synthetic_financial_export.csv",
                    mime="text/csv"
                )
                
            with tab_pca:
                st.subheader("Principal Component Analysis (PCA) Overlay")
                st.markdown("If the synthetic points (orange) overlap tightly with the real points (blue), it proves the VAE accurately mapped the logical transaction distribution.")
                
                # Preprocess both arrays identically for PCA mapping
                # Drop non-numeric generated IDs for mathematical alignment
                cols_to_drop = ['nameOrig', 'nameDest', 'isFlaggedFraud']
                
                real_numeric = df_real_raw.drop(columns=['nameOrig', 'nameDest', 'step', 'isFlaggedFraud'], errors='ignore')
                real_numeric = pd.get_dummies(real_numeric, columns=['type'], dtype=float)
                
                synth_numeric = df_synthetic.drop(columns=cols_to_drop, errors='ignore')
                
                # Align structural columns in case one-hot variations slightly mismatch
                for col in real_numeric.columns:
                    if col not in synth_numeric.columns:
                        synth_numeric[col] = 0.0
                synth_numeric = synth_numeric[real_numeric.columns]
                
                # Standardize values uniformly before applying PCA dimensions
                combined_scaler = StandardScaler()
                real_scaled = combined_scaler.fit_transform(real_numeric)
                synth_scaled = combined_scaler.transform(synth_numeric)
                
                # Fit PCA on real data and project both sets down to 2 components
                pca = PCA(n_components=2, random_state=42)
                pca_real = pca.fit_transform(real_scaled)
                pca_synth = pca.transform(synth_scaled)
                
                # Structure results into a clean unified DataFrame for Altair visualization
                df_pca_real = pd.DataFrame(pca_real, columns=['PC1', 'PC2'])
                df_pca_real['Dataset'] = 'Real Baseline Data'
                
                df_pca_synth = pd.DataFrame(pca_synth, columns=['PC1', 'PC2'])
                df_pca_synth['Dataset'] = 'VAE Generated Data'
                
                df_pca_combined = pd.concat([df_pca_real, df_pca_synth])
                
                # Build beautiful interactive Altair scatter plot
                pca_chart = alt.Chart(df_pca_combined).mark_circle(size=40, opacity=0.6).encode(
                    x=alt.X('PC1:Q', title='Principal Component 1'),
                    y=alt.Y('PC2:Q', title='Principal Component 2'),
                    color=alt.Color('Dataset:N', scale=alt.Scale(domain=['Real Baseline Data', 'VAE Generated Data'], range=['#1f77b4', '#ff7f0e'])),
                    tooltip=['PC1', 'PC2', 'Dataset']
                ).properties(width=850, height=500).interactive()
                
                st.altair_chart(pca_chart, use_container_width=True)
                
            with tab_distribution:
                st.subheader("Transaction Volume Comparison")
                
                # Simple quantitative summary table comparison
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Real Dataset Metrics Summary:**")
                    st.write(df_real_raw['amount'].describe())
                with col2:
                    st.write("**Synthetic Dataset Metrics Summary:**")
                    st.write(df_synthetic['amount'].describe())
                    
        except Exception as e:
            st.error(f"An unexpected data engineering pipeline mismatch occurred: {e}")