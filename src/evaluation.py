"""
Evaluation metrics for ATM cash demand forecasting and inventory efficiency.
Calculates RMSE, MAE, MAPE, WAPE (Weighted Absolute Percentage Error),
R-squared, and financial cost metrics.
"""

import numpy as np
import pandas as pd
from typing import Dict, Union


def evaluate_forecast(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray]
) -> Dict[str, float]:
    """
    Computes industry-standard forecasting accuracy metrics.
    
    WAPE is the retail banking benchmark because it weights errors proportionally
    to demand volume, avoiding distortion on low-withdrawal days.
    """
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)

    if len(y_t) != len(y_p):
        raise ValueError(f"Length mismatch: y_true ({len(y_t)}) vs y_pred ({len(y_p)})")

    errors = y_t - y_p
    abs_errors = np.abs(errors)
    squared_errors = errors ** 2

    # Basic metrics
    mae = float(np.mean(abs_errors))
    rmse = float(np.sqrt(np.mean(squared_errors)))

    # Percentage metrics (guard against zero division)
    non_zeros = y_t != 0
    if np.any(non_zeros):
        mape = float(np.mean(np.abs(errors[non_zeros] / y_t[non_zeros])) * 100.0)
    else:
        mape = 0.0

    sum_y_true = np.sum(y_t)
    wape = float((np.sum(abs_errors) / sum_y_true * 100.0) if sum_y_true > 0 else 0.0)

    # R-squared
    ss_tot = np.sum((y_t - np.mean(y_t)) ** 2)
    ss_res = np.sum(squared_errors)
    r2 = float(1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0)

    # Directional Accuracy (Did the model predict the correct day-over-day movement?)
    if len(y_t) > 1:
        actual_diff = np.diff(y_t)
        pred_diff = np.diff(y_p)
        correct_direction = np.sign(actual_diff) == np.sign(pred_diff)
        da = float(np.mean(correct_direction) * 100.0)
    else:
        da = 100.0

    return {
        "RMSE": round(rmse, 2),
        "MAE": round(mae, 2),
        "MAPE (%)": round(mape, 2),
        "WAPE (%)": round(wape, 2),
        "R2": round(r2, 4),
        "Directional Accuracy (%)": round(da, 2)
    }


def compute_inventory_costs(
    daily_balances: np.ndarray,
    unmet_demands: np.ndarray,
    num_refills: int,
    annual_holding_rate: float = 0.07,
    cit_trip_cost: float = 2500.0,
    stockout_penalty_per_rupee: float = 0.05
) -> Dict[str, float]:
    """
    Computes total operational cost of managing ATM cash:
    1. Opportunity/Holding Cost: Cost of idle cash locked inside ATM.
    2. Replenishment/Logistics Cost: CIT (Cash-in-Transit) van trip expenses.
    3. Stockout Penalty: Cost of customer dissatisfaction, lost fee income, and SLA penalties.
    """
    daily_holding_rate = annual_holding_rate / 365.0
    holding_cost = float(np.sum(daily_balances * daily_holding_rate))
    logistics_cost = float(num_refills * cit_trip_cost)
    stockout_penalty = float(np.sum(unmet_demands * stockout_penalty_per_rupee))
    total_cost = holding_cost + logistics_cost + stockout_penalty

    return {
        "holding_cost": round(holding_cost, 2),
        "logistics_cost": round(logistics_cost, 2),
        "stockout_penalty": round(stockout_penalty, 2),
        "total_cost": round(total_cost, 2)
    }
