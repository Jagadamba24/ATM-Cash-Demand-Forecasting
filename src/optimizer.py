"""
Intelligent Cash Replenishment & Inventory Optimizer for ATMs.
Transforms ML point forecasts and uncertainty intervals into an optimal
Dynamic (s, S) Replenishment Policy that minimizes holding cost, CIT logistics trips,
and cash-out stockout penalties.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional
from src.evaluation import compute_inventory_costs


class ATMReplenishmentOptimizer:
    """
    Simulates and optimizes ATM cash replenishment logistics using operations research
    and predictive inventory control.
    """

    def __init__(
        self,
        capacity: float = 2500000.0,
        lead_time_days: int = 1,
        service_level: float = 0.99,
        cit_cost_per_trip: float = 2500.0,
        annual_holding_rate: float = 0.07,
        stockout_penalty_rate: float = 0.05
    ):
        self.capacity = capacity
        self.lead_time = lead_time_days
        self.cit_cost = cit_cost_per_trip
        self.holding_rate = annual_holding_rate
        self.stockout_penalty_rate = stockout_penalty_rate

        # Standard normal inverse CDF quantile for safety stock
        # 0.95 -> 1.645, 0.98 -> 2.05, 0.99 -> 2.326
        z_lookup = {0.90: 1.282, 0.95: 1.645, 0.98: 2.054, 0.99: 2.326, 0.999: 3.090}
        self.z_score = z_lookup.get(service_level, 2.326)

    def calculate_dynamic_reorder_point(
        self,
        forecast_lead_demand: float,
        residual_std: float
    ) -> float:
        """
        Calculates dynamic reorder threshold (s):
        s = Expected Demand during Lead Time + Safety Stock
        Safety Stock = Z * sigma_residual * sqrt(lead_time)
        """
        safety_stock = self.z_score * residual_std * np.sqrt(self.lead_time)
        reorder_point = forecast_lead_demand + safety_stock
        # Ensure reorder point is bounded within reasonable fraction of capacity
        return float(min(self.capacity * 0.90, max(reorder_point, safety_stock)))

    def simulate_ml_policy(
        self,
        actual_demand: pd.Series,
        predicted_demand: pd.Series,
        residual_std: Optional[float] = None,
        initial_cash: Optional[float] = None
    ) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """
        Simulates ML-guided Dynamic (s, S) Replenishment.
        CIT replenishment is triggered only when inventory drops below the dynamic reorder point.
        """
        n = len(actual_demand)
        dates = actual_demand.index
        demands = actual_demand.values
        preds = predicted_demand.values

        if residual_std is None:
            residual_std = float(np.std(demands - preds))

        cash_balance = self.capacity if initial_cash is None else initial_cash

        balances = []
        refill_events = []
        unmet_demands = []
        reorder_thresholds = []
        refill_amounts = []

        pending_refill = False

        for t in range(n):
            # If a refill was dispatched on previous day, it arrives this morning
            if pending_refill:
                refill_amt = self.capacity - cash_balance
                cash_balance = self.capacity
                refill_events.append(1)
                refill_amounts.append(refill_amt)
                pending_refill = False
            else:
                refill_events.append(0)
                refill_amounts.append(0.0)

            # Demand consumption during the day
            actual_d = demands[t]
            if cash_balance >= actual_d:
                cash_balance -= actual_d
                unmet = 0.0
            else:
                unmet = actual_d - cash_balance
                cash_balance = 0.0

            unmet_demands.append(unmet)
            balances.append(cash_balance)

            # Calculate dynamic reorder point for next day
            next_forecast = preds[min(t + 1, n - 1)]
            s_t = self.calculate_dynamic_reorder_point(next_forecast, residual_std)
            reorder_thresholds.append(s_t)

            # Check if closing inventory warrants a replenishment order
            if cash_balance < s_t and not pending_refill:
                pending_refill = True

        df_sim = pd.DataFrame({
            "Actual_Demand": demands,
            "Predicted_Demand": preds,
            "Cash_Balance": balances,
            "Reorder_Threshold": reorder_thresholds,
            "Refill_Event": refill_events,
            "Refill_Amount": refill_amounts,
            "Unmet_Demand": unmet_demands
        }, index=dates)

        costs = compute_inventory_costs(
            daily_balances=np.array(balances),
            unmet_demands=np.array(unmet_demands),
            num_refills=sum(refill_events),
            annual_holding_rate=self.holding_rate,
            cit_trip_cost=self.cit_cost,
            stockout_penalty_per_rupee=self.stockout_penalty_rate
        )
        costs["num_refills"] = sum(refill_events)
        costs["stockout_days"] = int(np.sum(np.array(unmet_demands) > 0))

        return df_sim, costs

    def simulate_static_policy(
        self,
        actual_demand: pd.Series,
        refill_weekdays: List[int] = (0, 4),  # Monday (0) and Friday (4)
        initial_cash: Optional[float] = None
    ) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """
        Simulates traditional banking practice: Fixed calendar refills (e.g. every Monday and Friday)
        regardless of upcoming demand.
        """
        n = len(actual_demand)
        dates = actual_demand.index
        demands = actual_demand.values

        cash_balance = self.capacity if initial_cash is None else initial_cash

        balances = []
        refill_events = []
        unmet_demands = []
        refill_amounts = []

        for t in range(n):
            day_of_week = dates[t].weekday()
            
            # Static scheduled refill arrives in morning
            if day_of_week in refill_weekdays:
                refill_amt = self.capacity - cash_balance
                cash_balance = self.capacity
                refill_events.append(1)
                refill_amounts.append(refill_amt)
            else:
                refill_events.append(0)
                refill_amounts.append(0.0)

            # Demand consumption
            actual_d = demands[t]
            if cash_balance >= actual_d:
                cash_balance -= actual_d
                unmet = 0.0
            else:
                unmet = actual_d - cash_balance
                cash_balance = 0.0

            unmet_demands.append(unmet)
            balances.append(cash_balance)

        df_sim = pd.DataFrame({
            "Actual_Demand": demands,
            "Cash_Balance": balances,
            "Refill_Event": refill_events,
            "Refill_Amount": refill_amounts,
            "Unmet_Demand": unmet_demands
        }, index=dates)

        costs = compute_inventory_costs(
            daily_balances=np.array(balances),
            unmet_demands=np.array(unmet_demands),
            num_refills=sum(refill_events),
            annual_holding_rate=self.holding_rate,
            cit_trip_cost=self.cit_cost,
            stockout_penalty_per_rupee=self.stockout_penalty_rate
        )
        costs["num_refills"] = sum(refill_events)
        costs["stockout_days"] = int(np.sum(np.array(unmet_demands) > 0))

        return df_sim, costs

    def compare_policies(
        self,
        actual_demand: pd.Series,
        predicted_demand: pd.Series,
        initial_cash: Optional[float] = None
    ) -> Dict[str, any]:
        """
        Executes a rigorous comparative backtest between the ML Dynamic Policy
        and the Traditional Static Schedule. Quantifies net financial savings and stockout reduction.
        """
        df_ml, cost_ml = self.simulate_ml_policy(actual_demand, predicted_demand, initial_cash=initial_cash)
        df_static, cost_static = self.simulate_static_policy(actual_demand, initial_cash=initial_cash)

        net_savings = cost_static["total_cost"] - cost_ml["total_cost"]
        pct_savings = (net_savings / cost_static["total_cost"] * 100.0) if cost_static["total_cost"] > 0 else 0.0

        return {
            "ml_simulation": df_ml,
            "ml_costs": cost_ml,
            "static_simulation": df_static,
            "static_costs": cost_static,
            "net_savings_rupees": round(net_savings, 2),
            "percentage_savings": round(pct_savings, 2),
            "stockout_reduction_days": cost_static["stockout_days"] - cost_ml["stockout_days"]
        }
