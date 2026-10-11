"""
Data cleaning and preparation module (Step 1).
"""
from typing import Optional, Tuple
import pandas as pd
import numpy as np


def clean_orderbook_data(
    df: pd.DataFrame,
    tick_size: float,
    spread_pct_cutoff: float = 95.0,
    max_gap_minutes: float = 5.0,
    session_start: Optional[str] = None,
    session_end: Optional[str] = None,
    drop_first_min: int = 0,
    drop_last_min: int = 0
) -> pd.DataFrame:
    """
    Cleans raw order book snapshot data:
    1. Filter session hours if specified.
    2. Remove non-positive prices and sizes, and crossed/locked books.
    3. Reconstruct bid/ask prices from mid-price and basis point distances.
    4. Apply the paper's spread filter (drop snapshots exceeding 95th percentile of spread).
    5. Compute mid-price and delta_P_k in ticks.
    6. Flag and drop jumps across gaps (> max_gap_minutes).
    """
    cleaned = df.copy()

    # Session filter if requested
    if session_start is not None and session_end is not None:
        times = cleaned['timestamp_utc'].dt.time
        t_start = pd.to_datetime(session_start).time()
        t_end = pd.to_datetime(session_end).time()
        cleaned = cleaned[(times >= t_start) & (times <= t_end)].copy()

    if drop_first_min > 0 or drop_last_min > 0:
        cleaned['date'] = cleaned['timestamp_utc'].dt.date
        cleaned_groups = []
        for _, g in cleaned.groupby('date'):
            t_min = g['timestamp_utc'].min() + pd.Timedelta(minutes=drop_first_min)
            t_max = g['timestamp_utc'].max() - pd.Timedelta(minutes=drop_last_min)
            cleaned_groups.append(g[(g['timestamp_utc'] >= t_min) & (g['timestamp_utc'] <= t_max)])
        cleaned = pd.concat(cleaned_groups, ignore_index=True)

    # Basic positive checks
    cleaned = cleaned[cleaned['close_price'] > 0].copy()
    cleaned = cleaned[(cleaned['bid_volume_level_1'] > 0) & (cleaned['ask_volume_level_1'] > 0)].copy()

    # Spread in bps
    cleaned['spread_bps'] = cleaned['bid_distance_level_1'] + cleaned['ask_distance_level_1']
    # Drop crossed/locked books
    cleaned = cleaned[cleaned['spread_bps'] > 0].copy()

    # Reconstruct Best Bid & Ask Prices
    # Distance is in basis points (1 bp = 1e-4) from mid/close price
    cleaned['bid_px_1'] = cleaned['close_price'] * (1.0 - cleaned['bid_distance_level_1'] / 10000.0)
    cleaned['ask_px_1'] = cleaned['close_price'] * (1.0 + cleaned['ask_distance_level_1'] / 10000.0)
    cleaned['mid_price'] = (cleaned['bid_px_1'] + cleaned['ask_px_1']) / 2.0
    cleaned['spread_currency'] = cleaned['ask_px_1'] - cleaned['bid_px_1']

    # Paper's spread filter: drop snapshots whose spread exceeds stock's 95th percentile
    if spread_pct_cutoff is not None and len(cleaned) > 0:
        spread_threshold = np.percentile(cleaned['spread_bps'], spread_pct_cutoff)
        cleaned['spread_filter_passed'] = cleaned['spread_bps'] <= spread_threshold
        cleaned = cleaned[cleaned['spread_filter_passed']].copy()
    else:
        cleaned['spread_filter_passed'] = True

    # Re-sort after filtering
    cleaned = cleaned.sort_values('timestamp_utc').reset_index(drop=True)

    # Detect gaps and compute price changes
    cleaned['time_delta_sec'] = cleaned['timestamp_utc'].diff().dt.total_seconds()
    # Continuous interval flag: previous row exists and gap <= max_gap_minutes * 60 + tolerance
    tolerance_sec = 30.0  # slight tolerance for jitter
    cleaned['is_valid_interval'] = (
        cleaned['time_delta_sec'].notna() &
        (cleaned['time_delta_sec'] <= (max_gap_minutes * 60.0 + tolerance_sec))
    )

    # Price difference in currency and in ticks
    cleaned['delta_mid'] = cleaned['mid_price'].diff()
    cleaned['delta_P'] = cleaned['delta_mid'] / tick_size  # in ticks

    # Set delta_P to NaN where interval was invalid/interrupted
    cleaned.loc[~cleaned['is_valid_interval'], 'delta_P'] = np.nan
    cleaned.loc[~cleaned['is_valid_interval'], 'delta_mid'] = np.nan

    return cleaned
