"""
Tradability analysis module (Step 8).
Evaluates PnL, hit rate, turnover, and break-even transaction costs strictly on out-of-sample test set.
"""
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd


def evaluate_trading_signal(
    y_test_actual: np.ndarray,
    y_test_pred: np.ndarray,
    costs_in_ticks: Tuple[float, ...] = (0.5, 1.0, 2.0)
) -> Dict[str, object]:
    """
    Backtests trading strategy strictly out-of-sample:
      Signal_t = sign(y_test_pred_t)
      Realized trade return_t = Signal_t * y_test_actual_t (in ticks)
    """
    valid = ~(np.isnan(y_test_actual) | np.isnan(y_test_pred))
    y_act = y_test_actual[valid]
    y_pr = y_test_pred[valid]

    n = len(y_act)
    if n == 0:
        return {'status': 'error', 'reason': 'No valid test observations'}

    # Generate signals: +1 (long), -1 (short), 0 (neutral)
    signal = np.sign(y_pr)

    # Gross return per interval in ticks
    gross_pnl_per_interval = signal * y_act

    # Number of trades: position flips
    pos_diff = np.diff(signal, prepend=0)
    trades = np.abs(pos_diff) > 0
    n_trades = int(np.sum(trades))

    # Hit rate: fraction of active predictions where signal matches actual return sign
    active_mask = (signal != 0) & (y_act != 0)
    hit_rate = float(np.mean(signal[active_mask] == np.sign(y_act[active_mask]))) if np.sum(active_mask) > 0 else np.nan

    gross_total_ticks = float(np.sum(gross_pnl_per_interval))
    gross_avg_per_trade = float(gross_total_ticks / n_trades) if n_trades > 0 else 0.0

    # Break-even round-trip cost per trade in ticks
    break_even_cost_ticks = gross_avg_per_trade

    cost_results = {}
    for c in costs_in_ticks:
        # Total cost = n_trades * round_trip_cost
        total_cost = n_trades * c
        net_total_ticks = gross_total_ticks - total_cost
        net_avg_per_trade = (net_total_ticks / n_trades) if n_trades > 0 else 0.0
        survives = net_total_ticks > 0

        cost_results[f'cost_{c}_ticks'] = {
            'round_trip_cost_ticks': c,
            'net_total_pnl_ticks': float(net_total_ticks),
            'net_pnl_per_trade_ticks': float(net_avg_per_trade),
            'survives_costs': bool(survives)
        }

    return {
        'status': 'success',
        'n_intervals': n,
        'n_trades': n_trades,
        'turnover_trades_per_interval': float(n_trades / n),
        'hit_rate': hit_rate,
        'gross_total_pnl_ticks': gross_total_ticks,
        'gross_avg_pnl_per_trade_ticks': gross_avg_per_trade,
        'break_even_cost_ticks': float(break_even_cost_ticks),
        'cost_scenarios': cost_results
    }
