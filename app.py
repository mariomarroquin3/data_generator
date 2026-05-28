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

# --- 1. PREMIUM PAGE CONFIGURATION ---
st.set_page_config(
    page_title="AI Data Synthesizer", 
    page_icon="🌌", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS FOR METRICS AND BUTTONS ---
st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        font-weight: bold;
    }
    div[data-testid="metric-container"] {
        background-color: rgba(28, 131, 225, 0.1);
        border: 1px solid rgba(28, 131, 225, 0.1);
        padding: 5% 5% 5% 10%;
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# --- HEADER SECTION ---
st.title("🌌 Generative AI Financial Data Synthesizer")
st.markdown("""
> **Enterprise Privacy & Testing Engine** > This platform utilizes a **Variational Autoencoder (VAE)** to learn the latent statistical properties of financial transaction records. It generates completely synthetic, privacy-compliant datasets mathematically aligned with real-world behavior for secure software testing.
""")
st.divider()

# --- SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2103/2103284.png", width=80) 
    st.header("⚙️ Engine Configuration")
    
    with st.expander("📂 Dataset Settings", expanded=True):
        # NUEVO: Reemplazamos el text_input por un file_uploader interactivo
        uploaded_file = st.file_uploader("Upload your Real Dataset (CSV)", type=["csv"])
        num_to_generate = st.slider("Volume to Generate", min_value=500, max_value=5000, value=1000, step=500)

    # Check if model weights exist
    weights_exist = os.path.exists("vae_weights.pth")
    if not weights_exist:
        st.error("⚠️ **System Offline:** Model weights (`vae_weights.pth`) not found. Please train the backend first.")

    st.markdown("<br>", unsafe_allow_html=True)
    
    # NUEVO: El botón solo se activa si los pesos existen Y si el usuario ha subido un archivo
    can_generate = weights_exist and uploaded_file is not None
    generate_btn = st.button("🚀 Generate Synthetic Data", type="primary", disabled=not can_generate)
    
    if not uploaded_file:
        st.info("👆 Please upload a CSV dataset to activate the generator.")
# --- MAIN EXECUTION LOGIC ---
if generate_btn:
    with st.spinner("🧠 Initializing Neural Network and decoding latent space..."):
        try:
            # 1. Run the generation script using the uploaded file in memory
            df_synthetic = generate_synthetic_transactions(uploaded_file, num_samples=num_to_generate)
            
            # CRÍTICO: Rebobinar el archivo en memoria antes de leerlo por segunda vez
            uploaded_file.seek(0)
            
            # 2. Extract a baseline sample of real data for side-by-side comparison
            df_real_raw = load_and_sample(uploaded_file, n_samples=num_to_generate)
            
            # Sleek toast notification instead of a massive banner
            st.toast(f"Success! {num_to_generate} synthetic rows generated.", icon="✅")
            
            # --- STRUCTURED TABS ---
            tab_preview, tab_pca, tab_distribution = st.tabs([
                "📋 Data Output & Export", 
                "📉 PCA Space Alignment", 
                "📊 Feature Analytics"
            ])
            
            with tab_preview:
                st.subheader("Generated Synthetic Output")
                st.dataframe(df_synthetic.head(100), use_container_width=True)
                
                # Download link 
                csv_buffer = df_synthetic.to_csv(index=False).encode('utf-8')
                
                st.markdown("<br>", unsafe_allow_html=True)
                col1, col2, col3 = st.columns([1,2,1])
                with col2: # Center the download button
                    st.download_button(
                        label="📥 Download Dataset as CSV",
                        data=csv_buffer,
                        file_name="synthetic_financial_export.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                
            with tab_pca:
                st.subheader("Latent Space Validation (PCA)")
                st.caption("If the synthetic points (orange) overlap tightly with the real points (blue), it proves the VAE accurately mapped the logical transaction distribution.")
                
                # Structural alignment and scaling
                cols_to_drop = ['nameOrig', 'nameDest', 'isFlaggedFraud']
                real_numeric = df_real_raw.drop(columns=['nameOrig', 'nameDest', 'step', 'isFlaggedFraud'], errors='ignore')
                real_numeric = pd.get_dummies(real_numeric, columns=['type'], dtype=float)
                synth_numeric = df_synthetic.drop(columns=cols_to_drop, errors='ignore')
                
                for col in real_numeric.columns:
                    if col not in synth_numeric.columns:
                        synth_numeric[col] = 0.0
                synth_numeric = synth_numeric[real_numeric.columns]
                
                combined_scaler = StandardScaler()
                real_scaled = combined_scaler.fit_transform(real_numeric)
                synth_scaled = combined_scaler.transform(synth_numeric)
                
                # PCA Projection
                pca = PCA(n_components=2, random_state=42)
                pca_real = pca.fit_transform(real_scaled)
                pca_synth = pca.transform(synth_scaled)
                
                df_pca_real = pd.DataFrame(pca_real, columns=['PC1', 'PC2'])
                df_pca_real['Dataset'] = 'Real Baseline Data'
                
                df_pca_synth = pd.DataFrame(pca_synth, columns=['PC1', 'PC2'])
                df_pca_synth['Dataset'] = 'VAE Generated Data'
                
                df_pca_combined = pd.concat([df_pca_real, df_pca_synth])
                
                # Premium Altair Chart
                pca_chart = alt.Chart(df_pca_combined).mark_circle(size=60, opacity=0.5).encode(
                    x=alt.X('PC1:Q', title='Principal Component 1', axis=alt.Axis(grid=False)),
                    y=alt.Y('PC2:Q', title='Principal Component 2', axis=alt.Axis(grid=False)),
                    color=alt.Color('Dataset:N', 
                                    scale=alt.Scale(domain=['Real Baseline Data', 'VAE Generated Data'], 
                                                    range=['#1f77b4', '#ff7f0e']),
                                    legend=alt.Legend(title="Data Source", orient="top-right")),
                    tooltip=['PC1', 'PC2', 'Dataset']
                ).properties(height=500).interactive()
                
                st.altair_chart(pca_chart, use_container_width=True)
                
            with tab_distribution:
                st.subheader("Financial Volume Analytics")
                st.caption("Comparing macro-level financial metrics between the original and AI-generated environments.")
                
                # Premium KPI Metrics Dashboard
                st.markdown("### Transaction Amounts (Global)")
                col_m1, col_m2, col_m3 = st.columns(3)
                
                real_mean = df_real_raw['amount'].mean()
                synth_mean = df_synthetic['amount'].mean()
                diff_mean = synth_mean - real_mean
                
                real_max = df_real_raw['amount'].max()
                synth_max = df_synthetic['amount'].max()
                
                with col_m1:
                    st.metric("Real Avg. Amount", f"${real_mean:,.2f}")
                with col_m2:
                    st.metric("Synthetic Avg. Amount", f"${synth_mean:,.2f}", delta=f"{diff_mean:,.2f} variance", delta_color="off")
                with col_m3:
                    st.metric("Max Generated Amount", f"${synth_max:,.2f}")

                st.markdown("<hr>", unsafe_allow_html=True)
                
                # Detailed describe tables
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Real Dataset (Raw)**")
                    st.dataframe(df_real_raw['amount'].describe(), use_container_width=True)
                with c2:
                    st.markdown("**Synthetic Dataset (Generated)**")
                    st.dataframe(df_synthetic['amount'].describe(), use_container_width=True)
                    
        # --- DEFENSIVE ERROR HANDLING ---
        except ValueError as ve:
            st.error("❌ **Data Validation Failed**")
            st.warning(f"Detalle técnico: {ve}")
            st.info("Asegúrate de que el archivo CSV contenga exactamente las columnas requeridas (amount, oldbalanceOrg, etc).")
        except Exception as e:
            st.error("⚠️ **Unexpected System Failure**")
            st.exception(e)