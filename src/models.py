"""
Forecasting models suite for ATM cash demand.
Includes:
1. Seasonal Naive Baseline (Lag-7)
2. Classical SARIMAX with stabilized parameterization
3. Random Forest Regressor
4. Gradient Boosting Regressor
5. Regularized Ridge Regression
6. Hybrid Stacking Ensemble
With recursive multi-step out-of-sample forecasting and feature importance extraction.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Any
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import statsmodels.api as sm

from src.features import create_features
from src.evaluation import evaluate_forecast


class BaselineSeasonalNaive:
    """Predicts future values based on the same day of the prior week (Lag-7)."""
    def __init__(self, lag: int = 7):
        self.lag = lag
        self.history = None

    def fit(self, train_series: pd.Series):
        self.history = list(train_series.values)

    def predict(self, steps: int) -> np.ndarray:
        preds = []
        hist = list(self.history)
        for _ in range(steps):
            pred_val = hist[-self.lag]
            preds.append(pred_val)
            hist.append(pred_val)
        return np.array(preds)


class SARIMAXModel:
    """Robust Seasonal ARIMA with convergence stabilization."""
    def __init__(self, order=(1, 0, 1), seasonal_order=(1, 0, 1, 7)):
        self.order = order
        self.seasonal_order = seasonal_order
        self.model_res = None

    def fit(self, train_series: pd.Series):
        model = sm.tsa.statespace.SARIMAX(
            train_series,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        self.model_res = model.fit(disp=False, maxiter=200)

    def predict(self, steps: int) -> np.ndarray:
        forecast_res = self.model_res.forecast(steps=steps)
        return np.array(forecast_res)


class MLRecursiveForecaster:
    """
    Recursive multi-step forecaster for tabular ML algorithms (Random Forest, Gradient Boosting, Ridge).
    Featurizes historical data, trains the estimator, and iteratively predicts future steps while
    dynamically rolling forward the autoregressive lags.
    """
    def __init__(self, estimator, name: str):
        self.estimator = estimator
        self.name = name
        self.feature_names = []
        self.train_df = None
        self.target_col = "Value"

    def fit(self, train_df: pd.DataFrame, target_col: str = "Value"):
        self.target_col = target_col
        self.train_df = train_df.copy()
        X_train, y_train, self.feature_names = create_features(train_df, target_col=target_col)
        self.estimator.fit(X_train, y_train)

    def predict(self, future_dates: pd.DatetimeIndex) -> np.ndarray:
        """
        Recursively forecasts step-by-step for the given future dates.
        """
        history_df = self.train_df.copy()
        predictions = []

        for curr_date in future_dates:
            # Create a provisional row for curr_date with placeholder target
            temp_row = pd.DataFrame({self.target_col: [np.nan]}, index=[curr_date])
            combined = pd.concat([history_df, temp_row])
            
            # Featurize
            X_all, _, _ = create_features(combined, target_col=self.target_col)
            
            if curr_date in X_all.index:
                x_curr = X_all.loc[[curr_date]]
            else:
                x_curr = X_all.iloc[[-1]]

            pred_val = float(self.estimator.predict(x_curr)[0])
            predictions.append(pred_val)

            # Update history with predicted value for subsequent steps
            history_df.loc[curr_date, self.target_col] = pred_val

        return np.array(predictions)


def train_and_evaluate_all(
    df: pd.DataFrame,
    target_col: str = "Value",
    test_size: int = 14
) -> Dict[str, Any]:
    """
    Conducts an end-to-end benchmark comparison across all models:
    1. Chronological Train/Test Split
    2. Model Training
    3. Multi-Step Horizon Forecasting
    4. Performance Metrics (RMSE, MAE, MAPE, WAPE, R2, Directional Accuracy)
    5. Feature Importance Extraction
    """
    n_total = len(df)
    train_size = n_total - test_size
    train_df = df.iloc[:train_size].copy()
    test_df = df.iloc[train_size:].copy()

    test_dates = test_df.index
    y_test = test_df[target_col].values

    # 1. Seasonal Naive Baseline
    m_baseline = BaselineSeasonalNaive(lag=7)
    m_baseline.fit(train_df[target_col])
    pred_baseline = m_baseline.predict(len(test_dates))

    # 2. SARIMAX
    m_sarimax = SARIMAXModel(order=(1, 0, 1), seasonal_order=(1, 0, 1, 7))
    m_sarimax.fit(train_df[target_col])
    pred_sarimax = m_sarimax.predict(len(test_dates))

    # 3. Random Forest Regressor
    rf = RandomForestRegressor(n_estimators=120, max_depth=8, min_samples_split=4, random_state=42)
    m_rf = MLRecursiveForecaster(rf, "Random Forest")
    m_rf.fit(train_df, target_col)
    pred_rf = m_rf.predict(test_dates)

    # 4. Gradient Boosting Regressor
    gb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.08, max_depth=4, random_state=42)
    m_gb = MLRecursiveForecaster(gb, "Gradient Boosting")
    m_gb.fit(train_df, target_col)
    pred_gb = m_gb.predict(test_dates)

    # 5. Regularized Ridge Regression
    ridge_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=10.0))
    ])
    m_ridge = MLRecursiveForecaster(ridge_pipe, "Ridge Regression")
    m_ridge.fit(train_df, target_col)
    pred_ridge = m_ridge.predict(test_dates)

    # 6. Hybrid Stacking Ensemble (Blended weights based on variance reduction)
    pred_ensemble = (
        0.40 * pred_gb +
        0.35 * pred_rf +
        0.25 * pred_sarimax
    )

    models_dict = {
        "Seasonal Naive (Lag-7)": pred_baseline,
        "SARIMAX (1,0,1)x(1,0,1,7)": pred_sarimax,
        "Ridge Regression": pred_ridge,
        "Random Forest": pred_rf,
        "Gradient Boosting": pred_gb,
        "Hybrid Ensemble": pred_ensemble
    }

    # Calculate metrics table
    metrics_records = []
    for m_name, preds in models_dict.items():
        m_eval = evaluate_forecast(y_test, preds)
        m_eval["Model"] = m_name
        metrics_records.append(m_eval)

    df_metrics = pd.DataFrame(metrics_records)
    cols = ["Model", "RMSE", "MAE", "MAPE (%)", "WAPE (%)", "R2", "Directional Accuracy (%)"]
    df_metrics = df_metrics[cols].sort_values("RMSE").reset_index(drop=True)

    # DataFrame of Predictions vs Actuals
    df_preds = pd.DataFrame({"Actual": y_test}, index=test_dates)
    for m_name, preds in models_dict.items():
        df_preds[m_name] = preds

    # Feature Importance from Random Forest and Gradient Boosting
    feature_names = m_rf.feature_names
    rf_importances = m_rf.estimator.feature_importances_
    gb_importances = m_gb.estimator.feature_importances_

    df_importance = pd.DataFrame({
        "Feature": feature_names,
        "RF_Importance": rf_importances,
        "GB_Importance": gb_importances,
        "Avg_Importance": (rf_importances + gb_importances) / 2.0
    }).sort_values("Avg_Importance", ascending=False).reset_index(drop=True)

    return {
        "metrics": df_metrics,
        "predictions": df_preds,
        "feature_importance": df_importance,
        "train_df": train_df,
        "test_df": test_df,
        "models": {
            "Baseline": m_baseline,
            "SARIMAX": m_sarimax,
            "Random Forest": m_rf,
            "Gradient Boosting": m_gb,
            "Ridge": m_ridge
        }
    }
