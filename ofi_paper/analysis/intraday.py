"""
Intraday patterns module (Step 5).
Computes ACF, intraday seasonality in price impact beta, average depth AD, and variance identity.
"""
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import acf


def compute_acf_series(
    series: pd.Series,
    nlags: int = 50,
    alpha: float = 0.05
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes ACF and confidence intervals for a series up to nlags.
    """
    clean = series.dropna().values
    if len(clean) < nlags + 5:
        nlags = max(1, len(clean) - 5)
    acf_vals, confint = acf(clean, nlags=nlags, alpha=alpha)
    return acf_vals, confint


def compute_intraday_profiles(
    df: pd.DataFrame,
    time_bin: str = '30min'
) -> pd.DataFrame:
    """
    Computes intraday profiles of beta_i, AD_i, var[delta_P], and var[OFI] by time-of-day bin.
    """
    work = df.copy()
    work['time_of_day'] = work['timestamp_utc'].dt.floor(time_bin).dt.time

    grouped_records = []
    for tod, g in work.groupby('time_of_day'):
        y = g['delta_P'].dropna()
        x = g['OFI'].dropna()

        # Align
        common_idx = g.dropna(subset=['delta_P', 'OFI']).index
        if len(common_idx) < 5:
            continue

        y_c = g.loc[common_idx, 'delta_P'].values
        x_c = g.loc[common_idx, 'OFI'].values

        var_dp = float(np.var(y_c, ddof=1))
        var_ofi = float(np.var(x_c, ddof=1))

        # OLS beta
        X_mat = sm.add_constant(x_c)
        res = sm.OLS(y_c, X_mat).fit(cov_type='HC0')
        beta_i = float(res.params[1])

        ad_i = float(((g['bid_volume_level_1'] + g['ask_volume_level_1']) / 2.0).mean())

        grouped_records.append({
            'time_of_day': str(tod),
            'n_obs': len(common_idx),
            'beta': beta_i,
            'avg_depth': ad_i,
            'var_delta_p': var_dp,
            'var_ofi': var_ofi,
            'pred_var_delta_p': (beta_i ** 2) * var_ofi
        })

    return pd.DataFrame(grouped_records)
