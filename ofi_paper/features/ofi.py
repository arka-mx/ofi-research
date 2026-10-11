"""
Order Flow Imbalance (OFI) calculation module (Step 2).
Implements event contribution e_n from Cont, Kukanov, Stoikov (2011, Section 2.1).
"""
import math
from typing import Optional, Tuple
import numpy as np
import pandas as pd


def compute_single_event_ofi(
    p_b_prev: float,
    q_b_prev: float,
    p_a_prev: float,
    q_a_prev: float,
    p_b_curr: float,
    q_b_curr: float,
    p_a_curr: float,
    q_a_curr: float,
    tol: float = 1e-9
) -> float:
    """
    Computes single event contribution e_n according to Cont et al. (2011):
      e_n = 1{Pb_n >= Pb_{n-1}} * qb_n - 1{Pb_n <= Pb_{n-1}} * qb_{n-1}
            - 1{Pa_n <= Pa_{n-1}} * qa_n + 1{Pa_n >= Pa_{n-1}} * qa_{n-1}
    """
    # Bid side
    if p_b_curr > p_b_prev + tol:
        e_bid = q_b_curr
    elif p_b_curr < p_b_prev - tol:
        e_bid = -q_b_prev
    else:
        e_bid = q_b_curr - q_b_prev

    # Ask side
    if p_a_curr < p_a_prev - tol:
        e_ask = -q_a_curr
    elif p_a_curr > p_a_prev + tol:
        e_ask = q_a_prev
    else:
        e_ask = -(q_a_curr - q_a_prev)

    return float(e_bid + e_ask)


def compute_bid_ask_contributions(
    p_b_prev: float,
    q_b_prev: float,
    p_a_prev: float,
    q_a_prev: float,
    p_b_curr: float,
    q_b_curr: float,
    p_a_curr: float,
    q_a_curr: float,
    tol: float = 1e-9
) -> Tuple[float, float]:
    """Returns separate (bid_contribution, ask_contribution)."""
    if p_b_curr > p_b_prev + tol:
        e_bid = q_b_curr
    elif p_b_curr < p_b_prev - tol:
        e_bid = -q_b_prev
    else:
        e_bid = q_b_curr - q_b_prev

    if p_a_curr < p_a_prev - tol:
        e_ask = -q_a_curr
    elif p_a_curr > p_a_prev + tol:
        e_ask = q_a_prev
    else:
        e_ask = -(q_a_curr - q_a_prev)

    return float(e_bid), float(e_ask)


def stylized_model_bid_move(
    depth: float,
    market_sells: float = 0.0,
    limit_buys: float = 0.0,
    cancels: float = 0.0
) -> int:
    """
    Paper Section 2.2: Stylized model of order book.
    Delta Pb = ceil((Lb - Cb - Ms) / D)
    """
    net_queue = limit_buys - cancels - market_sells
    # In math ceil(-15/5) = -3; ceil((7-15)/5) = ceil(-8/5) = -1; ceil((7-4-15)/5) = ceil(-12/5) = -2
    return int(math.ceil(net_queue / depth))


def compute_ofi_series(
    df: pd.DataFrame,
    bid_px_col: str = 'bid_px_1',
    bid_sz_col: str = 'bid_volume_level_1',
    ask_px_col: str = 'ask_px_1',
    ask_sz_col: str = 'ask_volume_level_1',
    valid_interval_col: str = 'is_valid_interval',
    tautology_filter: bool = False
) -> pd.Series:
    """
    Vectorized OFI calculation across consecutive snapshots.
    If tautology_filter is True: excludes price-changing events (sets OFI to 0 or NaN when price changes).
    """
    pb = df[bid_px_col].values
    qb = df[bid_sz_col].values
    pa = df[ask_px_col].values
    qa = df[ask_sz_col].values

    n = len(df)
    ofi = np.full(n, np.nan)

    if n < 2:
        return pd.Series(ofi, index=df.index)

    pb_prev = pb[:-1]
    pb_curr = pb[1:]
    qb_prev = qb[:-1]
    qb_curr = qb[1:]

    pa_prev = pa[:-1]
    pa_curr = pa[1:]
    qa_prev = qa[:-1]
    qa_curr = qa[1:]

    tol = 1e-12

    # Bid logic
    bid_higher = pb_curr > pb_prev + tol
    bid_lower = pb_curr < pb_prev - tol
    bid_equal = ~(bid_higher | bid_lower)

    e_bid = np.where(bid_higher, qb_curr, np.where(bid_lower, -qb_prev, qb_curr - qb_prev))

    # Ask logic
    ask_lower = pa_curr < pa_prev - tol
    ask_higher = pa_curr > pa_prev + tol
    ask_equal = ~(ask_lower | ask_higher)

    e_ask = np.where(ask_lower, -qa_curr, np.where(ask_higher, qa_prev, -(qa_curr - qa_prev)))

    total_e = e_bid + e_ask

    if tautology_filter:
        # Footnote 3: exclude price-changing events from OFI
        # Only intervals where neither best bid nor best ask changed
        no_px_change = bid_equal & ask_equal
        total_e = np.where(no_px_change, total_e, 0.0)

    ofi[1:] = total_e

    # If valid interval column exists, set invalid to NaN
    if valid_interval_col in df.columns:
        valid = df[valid_interval_col].values
        ofi[~valid] = np.nan

    return pd.Series(ofi, index=df.index)


def compute_multilevel_ofi(
    df: pd.DataFrame,
    num_levels: int = 10,
    valid_interval_col: str = 'is_valid_interval'
) -> pd.DataFrame:
    """
    Computes multi-level OFI for levels 1 to num_levels.
    For crypto L2 data where level k volume is cumulative up to level k:
    incremental volume at level k = vol_level_k - vol_level_{k-1} (for k > 1).
    """
    n = len(df)
    ml_ofi = pd.DataFrame(index=df.index)

    for lvl in range(1, num_levels + 1):
        if lvl == 1:
            q_b = df['bid_volume_level_1']
            q_a = df['ask_volume_level_1']
        else:
            q_b = (df[f'bid_volume_level_{lvl}'] - df[f'bid_volume_level_{lvl-1}']).clip(lower=0.0)
            q_a = (df[f'ask_volume_level_{lvl}'] - df[f'ask_volume_level_{lvl-1}']).clip(lower=0.0)

        # Distances
        p_b = df['close_price'] * (1.0 - df[f'bid_distance_level_{lvl}'] / 10000.0)
        p_a = df['close_price'] * (1.0 + df[f'ask_distance_level_{lvl}'] / 10000.0)

        temp_df = pd.DataFrame({
            'pb': p_b,
            'qb': q_b,
            'pa': p_a,
            'qa': q_a,
            'valid': df[valid_interval_col] if valid_interval_col in df.columns else True
        })

        ml_ofi[f'OFI_L{lvl}'] = compute_ofi_series(
            temp_df, 'pb', 'qb', 'pa', 'qa', 'valid'
        )

    return ml_ofi
