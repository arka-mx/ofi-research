"""
Microprice and Expected Price Calculation Module.
Implements:
1. Level-1 and Multi-Level Microprice (Stoikov 2018).
2. Microprice changes (currency and ticks) with interval gap filtering.
3. Expected price estimation (instantaneous book expectation and OFI impact model).
"""
from typing import Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd
import statsmodels.api as sm

from ofi_paper.models.impact_regression import fit_ols_white


def compute_microprice(
    df: pd.DataFrame,
    bid_px_col: str = 'bid_px_1',
    ask_px_col: str = 'ask_px_1',
    bid_sz_col: str = 'bid_volume_level_1',
    ask_sz_col: str = 'ask_volume_level_1'
) -> pd.Series:
    """
    Computes Level 1 microprice according to Stoikov (2018):
      P^{micro} = (q^b * P^a + q^a * P^b) / (q^b + q^a)
    
    Equivalently:
      P^{micro} = P^{mid} + 0.5 * Imbalance * Spread
      where Imbalance = (q^b - q^a) / (q^b + q^a), Spread = P^a - P^b
    """
    qb = df[bid_sz_col]
    qa = df[ask_sz_col]
    pb = df[bid_px_col]
    pa = df[ask_px_col]

    denom = qb + qa
    # Avoid division by zero
    zero_mask = (denom <= 0) | denom.isna()
    micro = (qb * pa + qa * pb) / np.where(zero_mask, np.nan, denom)

    # Fallback to mid-price if total volume is zero
    mid = (pb + pa) / 2.0
    micro = micro.where(~zero_mask, mid)
    return micro


def compute_multilevel_microprice(
    df: pd.DataFrame,
    levels: int = 10,
    weighting: str = 'equal'
) -> pd.Series:
    """
    Computes Multi-Level weighted microprice incorporating up to `levels` depth:
      P^{micro, L} = sum_{l=1}^L w_l * (q^b_l * P^a_l + q^a_l * P^b_l) / sum_{l=1}^L w_l * (q^b_l + q^a_l)
    
    Weights options:
      - 'equal': w_l = 1.0
      - 'linear_decay': w_l = 1.0 / l
      - 'exponential': w_l = exp(-0.5 * (l - 1))
    """
    num_sum = np.zeros(len(df), dtype=float)
    den_sum = np.zeros(len(df), dtype=float)

    close = df['close_price'].values

    for lvl in range(1, levels + 1):
        bid_vol_col = f'bid_volume_level_{lvl}'
        ask_vol_col = f'ask_volume_level_{lvl}'
        bid_dist_col = f'bid_distance_level_{lvl}'
        ask_dist_col = f'ask_distance_level_{lvl}'

        if not all(c in df.columns for c in [bid_vol_col, ask_vol_col, bid_dist_col, ask_dist_col]):
            break

        qb = df[bid_vol_col].values
        qa = df[ask_vol_col].values
        pb = close * (1.0 - df[bid_dist_col].values / 10000.0)
        pa = close * (1.0 + df[ask_dist_col].values / 10000.0)

        if weighting == 'linear_decay':
            w = 1.0 / float(lvl)
        elif weighting == 'exponential':
            w = float(np.exp(-0.5 * (lvl - 1)))
        else:
            w = 1.0

        num_sum += w * (qb * pa + qa * pb)
        den_sum += w * (qb + qa)

    zero_mask = den_sum <= 0
    micro_ml = np.where(zero_mask, np.nan, num_sum / np.where(zero_mask, 1.0, den_sum))
    return pd.Series(micro_ml, index=df.index)


def compute_microprice_change(
    df: pd.DataFrame,
    tick_size: float,
    microprice_col: str = 'microprice',
    valid_interval_col: str = 'is_valid_interval'
) -> Tuple[pd.Series, pd.Series]:
    """
    Computes discrete change in microprice across consecutive snapshots:
      delta_micro = microprice_t - microprice_{t-1} (currency)
      delta_P_micro = delta_micro / tick_size (ticks)
    
    Masks price changes as NaN if interval is interrupted or invalid.
    """
    delta_micro = df[microprice_col].diff()
    delta_p_micro = delta_micro / tick_size

    if valid_interval_col in df.columns:
        invalid = ~df[valid_interval_col]
        delta_micro = delta_micro.mask(invalid, np.nan)
        delta_p_micro = delta_p_micro.mask(invalid, np.nan)

    return delta_micro, delta_p_micro


def compute_expected_price(
    df: pd.DataFrame,
    tick_size: float,
    ofi_col: str = 'OFI',
    mid_px_col: str = 'mid_price',
    micro_px_col: str = 'microprice',
    delta_p_mid_col: str = 'delta_P',
    delta_p_micro_col: str = 'delta_P_micro'
) -> pd.DataFrame:
    """
    Computes:
    1. Instantaneous fair expected price from order book imbalance: E[P_t | book_t] = P^{micro}_t
    2. Expected price adjustment: E[Delta P_t] = P^{micro}_t - P^{mid}_t (in currency and ticks)
    3. Contemporaneous OFI impact model expected price:
       hat{P}^{exp, mid}_t = P^{mid}_{t-1} + (alpha_mid + beta_mid * OFI_t) * tick_size
       hat{P}^{exp, micro}_t = P^{micro}_{t-1} + (alpha_micro + beta_micro * OFI_t) * tick_size
    4. 1-step ahead predictive expected price (Lag 1):
       hat{P}^{exp, lag1}_t = P^{mid}_{t-1} + (alpha_1 + beta_1 * OFI_{t-1}) * tick_size
    """
    out = df.copy()

    # Ensure microprice exists
    if micro_px_col not in out.columns:
        out[micro_px_col] = compute_microprice(out)

    if delta_p_micro_col not in out.columns:
        _, out[delta_p_micro_col] = compute_microprice_change(out, tick_size, microprice_col=micro_px_col)

    # 1. Instantaneous book expectation
    out['expected_adj_currency'] = out[micro_px_col] - out[mid_px_col]
    out['expected_adj_ticks'] = out['expected_adj_currency'] / tick_size

    # 2. Fit contemporaneous impact models
    valid_mid = ~(out[ofi_col].isna() | out[delta_p_mid_col].isna())
    valid_micro = ~(out[ofi_col].isna() | out[delta_p_micro_col].isna())

    res_mid = fit_ols_white(out.loc[valid_mid, delta_p_mid_col].values,
                            out.loc[valid_mid, ofi_col].values.reshape(-1, 1))
    res_micro = fit_ols_white(out.loc[valid_micro, delta_p_micro_col].values,
                              out.loc[valid_micro, ofi_col].values.reshape(-1, 1))

    # Fitted expected changes
    out['expected_delta_p_mid'] = np.nan
    out.loc[valid_mid, 'expected_delta_p_mid'] = (
        res_mid['alpha'] + res_mid['beta'] * out.loc[valid_mid, ofi_col]
    )

    out['expected_delta_p_micro'] = np.nan
    out.loc[valid_micro, 'expected_delta_p_micro'] = (
        res_micro['alpha'] + res_micro['beta'] * out.loc[valid_micro, ofi_col]
    )

    # Expected price levels
    out['expected_price_mid_ofi'] = out[mid_px_col].shift(1) + out['expected_delta_p_mid'] * tick_size
    out['expected_price_micro_ofi'] = out[micro_px_col].shift(1) + out['expected_delta_p_micro'] * tick_size

    # 3. Fit Lag 1 predictive model
    ofi_lag1 = out[ofi_col].shift(1)
    valid_lag1 = ~(ofi_lag1.isna() | out[delta_p_mid_col].isna())
    if valid_lag1.sum() > 20:
        res_lag1 = fit_ols_white(out.loc[valid_lag1, delta_p_mid_col].values,
                                 ofi_lag1.loc[valid_lag1].values.reshape(-1, 1))
        out['expected_delta_p_lag1'] = np.nan
        out.loc[valid_lag1, 'expected_delta_p_lag1'] = (
            res_lag1['alpha'] + res_lag1['beta'] * ofi_lag1.loc[valid_lag1]
        )
        out['expected_price_lag1'] = out[mid_px_col].shift(1) + out['expected_delta_p_lag1'] * tick_size
    else:
        out['expected_delta_p_lag1'] = np.nan
        out['expected_price_lag1'] = np.nan

    return out


def compare_price_impact_mid_vs_micro(
    df: pd.DataFrame,
    tick_size: float,
    ofi_col: str = 'OFI'
) -> Dict[str, Union[float, Dict]]:
    """
    Fits and compares contemporaneous price impact regressions for both
    Mid-Price change and Microprice change on OFI:
      Delta P^{mid} = alpha_mid + beta_mid * OFI + eps
      Delta P^{micro} = alpha_micro + beta_micro * OFI + eps
    """
    # Ensure microprice and delta_P_micro exist
    clean = df.copy()
    if 'microprice' not in clean.columns:
        clean['microprice'] = compute_microprice(clean)
    if 'delta_P_micro' not in clean.columns:
        _, clean['delta_P_micro'] = compute_microprice_change(clean, tick_size, 'microprice')

    valid = ~(clean[ofi_col].isna() | clean['delta_P'].isna() | clean['delta_P_micro'].isna())
    x = clean.loc[valid, ofi_col].values.reshape(-1, 1)
    y_mid = clean.loc[valid, 'delta_P'].values
    y_micro = clean.loc[valid, 'delta_P_micro'].values

    res_mid = fit_ols_white(y_mid, x)
    res_micro = fit_ols_white(y_micro, x)

    return {
        'n_obs': int(len(y_mid)),
        'mid': res_mid,
        'micro': res_micro,
        'delta_beta': float(res_micro['beta'] - res_mid['beta']),
        'delta_r2': float(res_micro['r2'] - res_mid['r2']),
        'beta_ratio': float(res_micro['beta'] / res_mid['beta']) if res_mid['beta'] != 0 else np.nan
    }
