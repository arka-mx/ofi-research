"""
Market depth calculation module (Step 4).
"""
import numpy as np
import pandas as pd


def compute_instantaneous_depth(
    df: pd.DataFrame,
    bid_sz_col: str = 'bid_volume_level_1',
    ask_sz_col: str = 'ask_volume_level_1'
) -> pd.Series:
    """
    Computes best quote depth: D_k = (q^b_k + q^a_k) / 2
    """
    return (df[bid_sz_col] + df[ask_sz_col]) / 2.0


def compute_window_depth(
    df: pd.DataFrame,
    bid_sz_col: str = 'bid_volume_level_1',
    ask_sz_col: str = 'ask_volume_level_1'
) -> float:
    """
    Computes average depth AD_i over a window i as in Cont et al. (2011):
      AD_i = 1/N * sum (q^b_n + q^a_n) / 2
    """
    d_series = compute_instantaneous_depth(df, bid_sz_col, ask_sz_col)
    return float(d_series.mean())
