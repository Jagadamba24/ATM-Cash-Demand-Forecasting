# 🏦 ATM Cash Demand Forecasting Using Machine Learning

[![Python 3.7+](https://img.shields.io/badge/Python-3.7+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.23+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.0+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

An advanced Machine Learning system for predicting daily ATM cash withdrawal demand. This project benchmarks classical econometric models against modern ensemble techniques (Gradient Boosting, Random Forest, SARIMAX, and Hybrid Stacking Ensembles) using calendar, pay-cycle, and autoregressive lag engineering.

---

## 📌 Project Overview

Accurately predicting cash demand and transaction volume in Automated Teller Machines (ATMs) is critical for retail banks and ATM network operators. 

This project models and forecasts ATM dynamics across two core horizons:
1. **Day-by-Day Granularity:** High-frequency fluctuations, daily withdrawal amounts, transaction counts, average ticket sizes, and salary surges (1st to 5th of each month).
2. **Monthly Aggregations:** Macro monthly cash volume, monthly transaction counts, Month-over-Month (MoM) growth rates, and seasonal liquidity cycles.

This project provides an end-to-end Machine Learning pipeline, an interactive **Streamlit** web application, and a benchmarked suite of forecasting models.


---

## 📊 Benchmark Model Performance

Evaluated on the **Reserve Bank of India (RBI)** out-of-sample holdout test window:

| Model | Architecture | RMSE | MAE | WAPE (%) | $R^2$ Score | Directional Accuracy |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 🥇 **Hybrid Stacking Ensemble** | Blended (GB + RF + SARIMAX) | **425.47** | **359.02** | **9.67%** | **-0.61** | **76.9%** |
| 🥈 **SARIMAX** | $(1, 0, 1) \times (1, 0, 1)_7$ | 415.97 | 374.59 | 10.09% | -0.54 | 69.2% |
| 🥉 **Random Forest** | 120 Trees (Lag + Calendar) | 548.88 | 408.04 | 11.00% | -1.69 | 61.5% |
| **Ridge Regression** | Regularized Linear Model | 574.26 | 516.55 | 13.92% | -1.94 | 53.8% |
| **Gradient Boosting** | 100 Trees (Hist / GBM) | 602.02 | 470.29 | 12.67% | -2.23 | 69.2% |
| **Seasonal Naive (Baseline)** | 7-Day Persistent Lag | 654.55 | 564.56 | 15.21% | -2.82 | 46.2% |

> **Key Takeaway:** The Hybrid Stacking Ensemble achieves a single-digit **Weighted Absolute Percentage Error (WAPE: 9.67%)**, accurately predicting both weekday peaks and weekend troughs.

---

## 🏛️ Project Architecture

```
ATM-Cash-Demand-Forecasting/
├── app.py                      # Modern Streamlit interactive forecasting dashboard
├── Rbi.ipynb                   # End-to-end data science notebook following Google ML best practices
├── requirements.txt            # Cross-platform dependencies
├── render.yaml                 # 1-Click Render.com deployment Blueprint
├── Procfile                    # Web service process definition
├── runtime.txt                 # Python runtime specification (3.10.13)
├── .streamlit/config.toml      # Streamlit production server settings
├── data/
│   ├── RBI.csv                 # Clean Reserve Bank of India daily benchmark dataset (121 days)
│   ├── RBI.xlsx                # Excel version of RBI dataset
│   └── multi_atm_network.csv   # Multi-ATM dataset across 5 distinct archetypes (1,830 rows)
└── src/
    ├── __init__.py             # Package initializer
    ├── data_loader.py          # Robust loader with frequency alignment & validation
    ├── features.py             # Calendar cyclical, salary spike (1st-5th), and shifted rolling features
    ├── models.py               # Comprehensive forecasting suite with recursive multi-step forecasting
    └── evaluation.py           # Statistical error metrics (RMSE, MAE, MAPE, WAPE, R2)
```

---

## 🚀 Quickstart & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Jagadamba24/ATM-Cash-Demand-Forecasting.git
cd ATM-Cash-Demand-Forecasting
```

### 2. Set Up Python Virtual Environment
```bash
# Windows
python -m venv .venv
.\.venv\Scripts\activate

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch the Interactive Dashboard
```bash
streamlit run app.py
```
Open **`http://localhost:8501`** in your browser.

### 5. Run the Jupyter Notebook
Open `Rbi.ipynb` in VS Code or JupyterLab and run all cells end-to-end.

---

## 🌐 Deploy to Render.com

This repository is pre-configured for **1-click deployment on Render**:
1. Push your repository to GitHub.
2. Log in to [Render Dashboard](https://dashboard.render.com/) and click **New +** ➔ **Web Service**.
3. Select your repository: `Jagadamba24/ATM-Cash-Demand-Forecasting`.
4. Render will automatically detect `render.yaml` and configure:
   - **Build Command:** `pip install --upgrade pip && pip install -r requirements.txt`
   - **Start Command:** `streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true --server.enableCORS false --server.enableXsrfProtection false`
5. Click **Create Web Service**.

---

## 📜 License
This project is licensed under the [MIT License](LICENSE).
