"""
Data loading utilities for ATM Cash Demand Forecasting.
Supports both the RBI daily national aggregate benchmark and multi-ATM fleet datasets.
"""

import os
import pandas as pd
import numpy as np


def get_project_root() -> str:
    """Returns the absolute root path of the project."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_rbi_data(filepath: str = None) -> pd.DataFrame:
    """
    Loads and preprocesses the Reserve Bank of India (RBI) daily ATM cash withdrawal dataset.
    Ensures datetime indexing, uniform column naming ('Value'), and daily frequency.
    """
    if filepath is None:
        root = get_project_root()
        csv_candidate = os.path.join(root, "data", "RBI.csv")
        xlsx_candidate = os.path.join(root, "data", "RBI.xlsx")
        root_csv = os.path.join(root, "RBI.csv")
        root_xlsx = os.path.join(root, "RBI.xlsx")

        if os.path.exists(csv_candidate):
            filepath = csv_candidate
        elif os.path.exists(root_csv):
            filepath = root_csv
        elif os.path.exists(xlsx_candidate):
            filepath = xlsx_candidate
        elif os.path.exists(root_xlsx):
            filepath = root_xlsx
        else:
            raise FileNotFoundError("Could not find RBI dataset in data/ or root directory.")

    # Read based on file extension
    if filepath.endswith(".csv"):
        df = pd.read_csv(filepath)
    else:
        df = pd.read_excel(filepath)

    # Standardize column names
    col_mapping = {}
    for col in df.columns:
        c_lower = col.strip().lower()
        if "date" in c_lower:
            col_mapping[col] = "Date"
        elif "val" in c_lower:
            col_mapping[col] = "Value"
    df = df.rename(columns=col_mapping)

    if "Date" not in df.columns or "Value" not in df.columns:
        raise ValueError(f"Expected 'Date' and 'Value' columns, but found: {df.columns.tolist()}")

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")

    # Set DatetimeIndex with explicit daily frequency
    df = df.set_index("Date")
    df = df.asfreq("D")

    # Interpolate if any single missing day exists
    if df["Value"].isnull().any():
        df["Value"] = df["Value"].interpolate(method="time")

    return df


def load_network_data(filepath: str = None) -> pd.DataFrame:
    """
    Loads the multi-ATM network dataset containing heterogeneous ATM locations,
    capacities, transaction volumes, and holiday markers.
    """
    if filepath is None:
        root = get_project_root()
        filepath = os.path.join(root, "data", "multi_atm_network.csv")

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Multi-ATM network dataset not found at {filepath}")

    df = pd.read_csv(filepath)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def get_atm_series(df_network: pd.DataFrame, atm_id: str) -> pd.DataFrame:
    """
    Filters multi-ATM network dataset for a specific ATM ID and returns a sorted, indexed DataFrame.
    """
    df_atm = df_network[df_network["ATM_ID"] == atm_id].copy()
    df_atm = df_atm.sort_values("Date").set_index("Date")
    df_atm["Value"] = df_atm["Cash_Withdrawn"]
    return df_atm


def get_network_kpis(df_network: pd.DataFrame) -> dict:
    """
    Computes fleet-wide operational KPIs.
    """
    total_withdrawn = df_network["Cash_Withdrawn"].sum()
    total_txns = df_network["Transaction_Count"].sum()
    avg_ticket = total_withdrawn / max(1, total_txns)
    unique_atms = df_network["ATM_ID"].nunique()
    date_range = (df_network["Date"].min().strftime("%b %d, %Y"), df_network["Date"].max().strftime("%b %d, %Y"))

    return {
        "total_cash_disbursed": total_withdrawn,
        "total_transactions": total_txns,
        "avg_ticket_size": avg_ticket,
        "atm_count": unique_atms,
        "date_range": date_range,
    }
