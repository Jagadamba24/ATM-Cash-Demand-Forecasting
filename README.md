# 🏦 ATM Cash Demand Forecasting & Intelligent Replenishment Engine

[![Python 3.7+](https://img.shields.io/badge/Python-3.7+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.23+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.0+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

An enterprise-grade forecasting and cash logistics optimization system designed for retail banks, Independent ATM Deployers (IADs), and Cash-in-Transit (CIT) operators. This project combines **modern Machine Learning ensembles** with **Operations Research $(s, S)$ inventory control** to forecast daily cash demand and minimize cash replenishment costs.

---

## 🌟 What Makes This Project Unique?

In traditional data science coursework, cash demand forecasting stops at raw error metrics like RMSE or MAPE. However, in real banking operations:
> **Forecasting numbers is only half the battle. The true business value lies in translating predictions into optimal Cash-in-Transit (CIT) logistics decisions.**

This system bridges that gap with two integrated engines:
1. **Predictive Engine:** Multi-horizon Machine Learning forecasting (Gradient Boosting, Random Forest, SARIMAX, and Hybrid Stacking Ensembles) using calendar, pay-cycle, and autoregressive lag engineering.
2. **Prescriptive Engine:** An **Intelligent Dynamic $(s, S)$ Replenishment Policy** that minimizes cash holding costs and CIT armored van trips while maintaining a 99%+ customer non-stockout service level.

---

## 📐 Mathematical Formulation

### 1. Operations Research: Dynamic $(s, S)$ Inventory Control
For day $t$ with lead time $L$ (typically 1 day for armored van dispatch):
- **Safety Stock ($SS_t$):** Sized dynamically based on forecast residual uncertainty:
  $$SS_t = Z \cdot \sigma_e \cdot \sqrt{L}$$
  where $Z = 2.33$ for a 99% non-stockout service level, and $\sigma_e$ is the standard deviation of model residuals.
- **Dynamic Reorder Threshold ($s_t$):**
  $$s_t = \hat{d}_{t+1} + SS_t$$
  A replenishment order is triggered if closing cash inventory $I_t < s_t$.
- **Order-Up-To Target ($S$):** Cash is restored to full ATM capacity $C$.

### 2. Total Cash Management Cost Objective Function
$$\min \text{Total Cost} = C_{\text{holding}} + C_{\text{logistics}} + C_{\text{stockout}}$$
$$\text{Total Cost} = \sum_{t=1}^T \left( I_t \cdot \frac{r}{365} \right) + (N_{\text{refills}} \cdot C_{\text{CIT}}) + \sum_{t=1}^T \left( \max(0, d_t - I_t) \cdot P_{\text{stockout}} \right)$$
- $r$: Annual cost of capital / holding rate (default: 7.0%).
- $C_{\text{CIT}}$: Cost per armored transit trip (default: ₹2,500).
- $P_{\text{stockout}}$: Regulatory SLA and customer dissatisfaction penalty per unit of unmet cash.

---

## 📊 Benchmark Model Performance

Evaluated on the **Reserve Bank of India (RBI)** out-of-sample holdout test horizon:

| Model | Architecture | RMSE | MAE | WAPE (%) | $R^2$ Score | Directional Accuracy |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 🥇 **Hybrid Stacking Ensemble** | Blended (GB + RF + SARIMAX) | **425.47** | **359.02** | **9.67%** | **-0.61** | **76.9%** |
| 🥈 **SARIMAX** | $(1, 0, 1) \times (1, 0, 1)_7$ | 415.97 | 374.59 | 10.09% | -0.54 | 69.2% |
| 🥉 **Random Forest** | 120 Trees (Lag + Calendar) | 548.88 | 408.04 | 11.00% | -1.69 | 61.5% |
| **Ridge Regression** | Regularized Linear Model | 574.26 | 516.55 | 13.92% | -1.94 | 53.8% |
| **Gradient Boosting** | 100 Trees (Hist / GBM) | 602.02 | 470.29 | 12.67% | -2.23 | 69.2% |
| **Seasonal Naive (Baseline)** | 7-Day Persistent Lag | 654.55 | 564.56 | 15.21% | -2.82 | 46.2% |

### 💰 Financial Cost Savings Impact
Backtested on the simulated fleet network:
- **Traditional Fixed Calendar Schedule (Mon/Fri):** ₹14,596.49 total cost (4 trips, high idle cash).
- **ML Dynamic $(s, S)$ Replenishment Policy:** ₹8,934.82 total cost (2 trips, zero stockouts).
- **Net Operational Savings:** **₹5,661.67 per ATM every 14 days (38.8% Cost Reduction)**.

---

## 🏛️ Project Architecture

```
ATM-Cash-Demand-Forecasting/
├── app.py                      # Modern Streamlit interactive web dashboard
├── Rbi.ipynb                   # End-to-end data science notebook following Google ML best practices
├── requirements.txt            # Dependency specification
├── data/
│   ├── RBI.csv                 # Clean Reserve Bank of India daily benchmark dataset (121 days)
│   ├── RBI.xlsx                # Excel version of RBI dataset
│   └── multi_atm_network.csv   # Multi-ATM fleet simulation across 5 distinct archetypes (1,830 rows)
└── src/
    ├── __init__.py             # Package initializer
    ├── data_loader.py          # Robust loader with frequency alignment & validation
    ├── features.py             # Calendar cyclical, salary spike (1st-5th), and shifted rolling features
    ├── models.py               # Comprehensive forecasting suite with recursive multi-step forecasting
    ├── optimizer.py            # Operations research dynamic (s, S) cash replenishment engine
    └── evaluation.py           # Statistical metrics (RMSE, WAPE) & financial inventory cost accounting
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
*The dashboard will automatically open at `http://localhost:8501` featuring Fleet Command KPIs, Model Arena, Replenishment Optimizer, and Stress Testing.*

### 5. Run the Story-Driven Jupyter Notebook
Open `Rbi.ipynb` in your preferred Jupyter environment (VS Code, JupyterLab, or Google Colab). All cells execute from top to bottom with zero external missing file dependencies.

---

## 🏙️ Heterogeneous ATM Fleet Archetypes

The included `multi_atm_network.csv` simulates 5 distinct real-world banking environments:
1. **Commercial Tech Park (`ATM_01`):** Heavy weekday demand, sharp 1st–5th salary rush (+75%), muted weekends.
2. **Shopping Mall (`ATM_02`):** Massive weekend & festival spikes (+80%), moderate weekdays.
3. **Residential Suburb (`ATM_03`):** Consistent withdrawals, pension/utility cycle at month-start.
4. **Airport / Transit Terminal (`ATM_04`):** High velocity 24/7 withdrawals with high volatility.
5. **University Campus (`ATM_05`):** Semester cycles and academic term fluctuations.

---

## 📜 License
This project is released under the [MIT License](LICENSE).
