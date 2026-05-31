import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import os
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from streamlit_option_menu import option_menu
from streamlit_extras.stylable_container import stylable_container

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

/* ---------- GLOBAL ---------- */
.stApp {
    background:
        radial-gradient(circle at top left, rgba(37,99,235,.15), transparent 30%),
        radial-gradient(circle at top right, rgba(16,185,129,.08), transparent 25%),
        #020617;
    color: #e2e8f0;
}

/* Hide streamlit decoration */
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}


/* ---------- HERO ---------- */
.hero-card {
    background: rgba(15,23,42,0.75);
    border: 1px solid rgba(255,255,255,0.08);
    backdrop-filter: blur(20px);
    border-radius: 28px;
    padding: 2rem;
    margin-bottom: 2rem;
    box-shadow: 0 0 50px rgba(59,130,246,.08);
}

.hero-title {
    font-size: 3rem;
    font-weight: 800;
    line-height: 1;
    letter-spacing: -.04em;
    color: white;
}

.gradient-text {
    background: linear-gradient(90deg,#60a5fa,#34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-subtitle {
    color: #94a3b8;
    font-size: 1.1rem;
    line-height: 1.8;
}

/* ---------- BUTTONS ---------- */
.stButton > button,
.stDownloadButton > button {
    width: 100%;
    border-radius: 18px;
    border: none;
    height: 52px;
    font-weight: 700;
    background: linear-gradient(
        135deg,
        #2563eb,
        #0ea5e9
    );
    color: white;
    transition: all .25s ease;
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 0 35px rgba(37,99,235,.35);
}

/* ---------- METRICS ---------- */
div[data-testid="metric-container"] {
    background: rgba(15,23,42,.72);
    border: 1px solid rgba(255,255,255,.08);
    border-radius: 24px;
    padding: 1.5rem;
    backdrop-filter: blur(20px);
}

/* ---------- DATAFRAME ---------- */
[data-testid="stDataFrame"] {
    border-radius: 22px;
    overflow: hidden;
    border: 1px solid rgba(255,255,255,.08);
}

/* ---------- SIDEBAR ---------- */
section[data-testid="stSidebar"] {
    background:
        linear-gradient(
            180deg,
            rgba(15,23,42,.98),
            rgba(2,6,23,.98)
        );
    border-right: 1px solid rgba(255,255,255,.06);
}

.sidebar-card {
    background: rgba(30,41,59,.5);
    border-radius: 24px;
    padding: 1rem;
    border: 1px solid rgba(255,255,255,.08);
}

/* ---------- TABS ---------- */
button[data-baseweb="tab"] {
    background: rgba(15,23,42,.6);
    border-radius: 16px;
    padding: .8rem 1.3rem;
    color: #94a3b8;
}

button[data-baseweb="tab"][aria-selected="true"] {
    background: rgba(37,99,235,.18);
    color: white;
}

/* ---------- HR ---------- */
hr {
    border: none;
    height: 1px;
    background: rgba(255,255,255,.08);
}

</style>
""", unsafe_allow_html=True)

weights_exist = os.path.exists("vae_weights (2).pth")

st.markdown(f"""
<div class="hero-card">

<p style="letter-spacing:4px;
text-transform:uppercase;
font-size:.8rem;
font-weight:700;
color:#60a5fa;">
Enterprise Privacy & Testing Engine
</p>

<div class="hero-title">
Generative AI <br>
<span class="gradient-text">
Financial Data Synthesizer
</span>
</div>

<br>

<p class="hero-subtitle">
This platform utilizes a <b>Variational Autoencoder (VAE)</b> to learn
latent statistical patterns of financial transactions and generate
synthetic privacy-compliant datasets mathematically aligned with
real-world behavior for secure software testing.
</p>

</div>
""", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)

c1.success("🧠 VAE Enabled")
c2.success("🔒 Privacy Safe")

if weights_exist:
    c3.success("🟢 Model Online")
else:
    c3.error("🔴 Model Offline")

c4.info("📊 Enterprise Mode")

# --- SIDEBAR CONFIGURATION ---
with st.sidebar:

    st.markdown("## 🌌 AI Synthesizer")

   

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <div class="sidebar-card">
    <h4>⚙️ Engine Configuration</h4>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload Financial Dataset",
        type=["csv"]
    )
    st.markdown("<br>", unsafe_allow_html=True)

    num_to_generate = st.slider(
        "Synthetic Volume",
        min_value=500,
        max_value=5000,
        value=1000,
        step=500
    )

    weights_exist = os.path.exists("vae_weights (2).pth")

    if weights_exist:
        st.success("Model Ready")
    else:
        st.error("Missing vae_weights.pth")

    can_generate = (
        weights_exist and
        uploaded_file is not None
    )

    generate_btn = st.button(
        "🚀 Generate Synthetic Data",
        type="primary",
        disabled=not can_generate
    )
    st.markdown("<br>", unsafe_allow_html=True)

    if not uploaded_file:
        st.info(
            "Upload a CSV dataset to activate the engine."
        )
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
                st.caption(
                    "Comparing statistical structure between real and synthetic financial transactions."
                )

                cols_to_drop = [
                    'nameOrig',
                    'nameDest',
                    'isFlaggedFraud'
                ]

                real_numeric = df_real_raw.drop(
                    columns=[
                        'nameOrig',
                        'nameDest',
                        'step',
                        'isFlaggedFraud'
                    ],
                    errors='ignore'
                )

                real_numeric = pd.get_dummies(
                    real_numeric,
                    columns=['type'],
                    dtype=float
                )

                synth_numeric = df_synthetic.drop(
                    columns=cols_to_drop,
                    errors='ignore'
                )

    # Align columns
                for col in real_numeric.columns:
                    if col not in synth_numeric.columns:
                        synth_numeric[col] = 0.0

                synth_numeric = synth_numeric[
                    real_numeric.columns
                ]

    # ---------------------------
    # FIX 1: CLEAN EXTREME VALUES
    # ---------------------------

                monetary_cols = [
                    col for col in real_numeric.columns
                    if any(keyword in col.lower()
                    for keyword in [
                        "amount",
                        "balance",
                        "old",
                        "new"
                    ])
                ]

                for col in monetary_cols:

                    real_numeric[col] = np.log1p(
                        real_numeric[col].clip(lower=0)
                    )

                    synth_numeric[col] = np.log1p(
                        synth_numeric[col].clip(lower=0)
                    )

    # Remove infinite values
                real_numeric = real_numeric.replace(
                    [np.inf, -np.inf],
                    np.nan
                ).fillna(0)

                synth_numeric = synth_numeric.replace(
                    [np.inf, -np.inf],
                    np.nan
                ).fillna(0)

    # ---------------------------
    # FIX 2: SCALING
    # ---------------------------

                scaler = StandardScaler()

                real_scaled = scaler.fit_transform(
                    real_numeric
                )

                synth_scaled = scaler.transform(
                    synth_numeric
                )

    # ---------------------------
    # FIX 3: PCA
    # ---------------------------

                pca = PCA(
                    n_components=2,
                    random_state=42
                )

                pca_real = pca.fit_transform(
                    real_scaled
                )

                pca_synth = pca.transform(
                    synth_scaled
                )

    # Sample for visualization
                sample_size = min(
                    2000,
                    len(pca_real),
                    len(pca_synth)
                )

                real_idx = np.random.choice(
                    len(pca_real),
                    sample_size,
                    replace=False
                )

                synth_idx = np.random.choice(
                    len(pca_synth),
                    sample_size,
                    replace=False
                )

                df_pca_real = pd.DataFrame(
                    pca_real[real_idx],
                    columns=["PC1", "PC2"]
                )

                df_pca_real["Dataset"] = "Real Data"

                df_pca_synth = pd.DataFrame(
                    pca_synth[synth_idx],
                    columns=["PC1", "PC2"]
                )

                df_pca_synth["Dataset"] = "Synthetic Data"

                df_pca = pd.concat([
                    df_pca_real,
                    df_pca_synth
                ])

    # ---------------------------
    # PREMIUM CHART
    # ---------------------------

                pca_chart = (
                    alt.Chart(df_pca)
                    .mark_circle(size=55, opacity=0.35)
                    .encode(
                        x=alt.X(
                            "PC1:Q",
                            title="Principal Component 1"
                        ),
                        y=alt.Y(
                            "PC2:Q",
                            title="Principal Component 2"
                        ),
                        color=alt.Color(
                            "Dataset:N",
                            scale=alt.Scale(
                                domain=[
                                    "Real Data",
                                    "Synthetic Data"
                                ],
                                range=[
                                    "#60a5fa",
                                    "#34d399"
                                ]
                            ),
                            legend=alt.Legend(
                                orient="top-right"
                            )
                        ),
                        tooltip=[
                            "PC1",
                            "PC2",
                            "Dataset"
                        ]
                    )
                    .properties(
                        height=600
                    )
                    .interactive()
                )

                st.altair_chart(
                    pca_chart,
                    use_container_width=True
                )

                explained = (
                    pca.explained_variance_ratio_.sum()
                    * 100
                )

                st.info(
                    f"PCA explained variance: "
                    f"{explained:.2f}%"
                )
            
                
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