# 🌌 AI Financial Data Synthesizer
### 🎓 Code in Place - Final Project

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c)
![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-FF4B4B)

An AI-powered synthetic data generation engine designed for the financial and accounting sectors. This application utilizes a **Variational Autoencoder (VAE)** to learn the latent statistical patterns of real financial transactions and generate completely new, privacy-compliant datasets that are mathematically aligned with real-world behavior for secure software testing and auditing.

---
### 🎯 Project Purpose, Learning, and Application

**Why This Project?**
Financial software testing requires realistic data, but using actual client records poses massive privacy risks. This project uses Generative AI to solve this problem. Instead of masking data, it generates entirely new, secure transactional records that perfectly mimic real-world financial behavior, ensuring a robust and privacy-compliant testing environment.

**What I Learned**
Through this project, I advanced my Python skills into data science and deep learning. I mastered essential preprocessing techniques (like log transformations and One-Hot encoding) and implemented a Variational Autoencoder (VAE) architecture using PyTorch. Additionally, I learned to convert complex mathematical models into an interactive user interface using Streamlit and Altair.

**How I Applied It**
I built a complete three-layer system:
1. **Data Pipeline:** Automatically cleans irrelevant variables and scales financial amounts for high-precision model processing.
2. **AI Core:** Developed and trained the VAE model in PyTorch to sample from a latent space, generating unique transactions that match the multidimensional geometry of the original data.
3. **Web Platform:** Designed a Streamlit dashboard that allows users to upload baselines, control generation volume, validate data alignment using Principal Component Analysis (PCA), and export the synthetic datasets as CSV files.

## 📋 Project Overview

When developing financial software or conducting accounting audits, using real client data poses massive privacy and security risks. This project solves that problem by using Generative Artificial Intelligence to create realistic, synthetic financial records. 

Instead of just shuffling or masking existing data, this system runs an advanced deep learning model that understands the underlying mathematical relationships of the business, allowing users to generate high-fidelity datasets with a single click.

---

## 📊 Dataset Requirements & Structure

To activate the generative engine, the uploaded file (`dataset.csv`) must follow a specific structure so the neural network can process the numbers correctly. 

### Required Columns
Your CSV file must contain the following columns:

1. **`amount`** *(Numeric - Float)*: The monetary value of the transaction. The model applies log transformations (`np.log1p`) to handle extreme values gracefully and avoid mathematical distortion.
2. **`oldbalanceOrg`** *(Numeric - Float)*: The initial balance of the origin account before the transaction took place.
3. **`newbalanceOrig`** *(Numeric - Float)*: The final balance of the origin account after the transaction.
4. **`oldbalanceDest`** *(Numeric - Float)*: The initial balance of the destination account before receiving the funds.
5. **`newbalanceDest`** *(Numeric - Float)*: The final balance of the destination account after the transaction.
6. **`type`** *(Categorical - String)*: The type of transaction (e.g., `CASH_IN`, `CASH_OUT`, `DEBIT`, `PAYMENT`, `TRANSFER`). The application automatically converts these text categories into numbers using One-Hot Encoding (`pd.get_dummies`).

### Automatically Ignored Columns
For privacy and architectural reasons, the system safely ignores identifiers and non-numeric tracking labels such as `nameOrig`, `nameDest`, `step`, and fraud flags (`isFraud`). You do not need to remove them manually; the pipeline filters them out during execution.

---

## 🛠️ Tech Stack

- **Core / Deep Learning:** Python 3.8+, PyTorch, NumPy
- **Data Processing:** Pandas, Scikit-Learn (PCA, StandardScaler)
- **User Interface:** Streamlit, Streamlit Option Menu
- **Data Visualization:** Altair (Interactive charts)

---

## ⚙️ How to Run the Project Locally

Follow these simple steps to launch the application on your computer:

1. **Clone this repository:**
   ```bash
   git clone [https://github.com/mariomarroquin3/data_generator](https://github.com/mariomarroquin3/data_generator)
   cd data_generator
Set up a virtual environment (Recommended):

Bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
Install the dependencies:

Bash
pip install -r requirements.txt
Launch the Web Application:

Bash
streamlit run app.py
Open your browser and navigate to http://localhost:8501.

🧠 How it Works (The Short Version)
Upload: The user uploads a real transaction baseline via the sidebar.

Inference: The system loads the pre-trained neural network weights (vae_weights.pth).

Stochastic Generation: Instead of copying data, the VAE samples from a continuous probabilistic distribution in the latent space. This means every time you generate data, you get a unique, realistic simulation.

Validation: The application displays a Principal Component Analysis (PCA) plot where the synthetic data cloud perfectly aligns with the real data geometry, proving the AI truly learned the structural rules of the dataset.

Export: The user compares the macro-level averages and downloads the clean .csv file.