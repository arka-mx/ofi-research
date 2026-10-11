"""
Trades vs OFI analysis module (Step 6).
Handles trade volume scaling (Regressions 17a-c) and square-root scaling simulation.
"""
from typing import Dict, Tuple
import numpy as np
import pandas as pd
import statsmodels.api as sm


def estimate_volume_exponent(
    df: pd.DataFrame,
    delta_p_col: str = 'delta_P',
    vol_col: str = 'interval_traded_volume'
) -> Dict[str, float]:
    """
    Estimates volume exponent H from paper Section 4.2:
      log |delta_P_k| = log(theta) + H * log(VOL_k) + eps
    Uses White HC0 standard errors.
    """
    if vol_col not in df.columns:
        return {'status': 'skipped', 'reason': 'No trade volume column'}

    sub = df[[delta_p_col, vol_col]].dropna()
    # Filter non-zero price changes and positive volume
    sub = sub[(sub[delta_p_col].abs() > 0) & (sub[vol_col] > 0)]

    if len(sub) < 20:
        return {'status': 'skipped', 'reason': 'Insufficient non-zero observations'}

    log_abs_dp = np.log(sub[delta_p_col].abs().values)
    log_vol = np.log(sub[vol_col].values)

    X = sm.add_constant(log_vol)
    res = sm.OLS(log_abs_dp, X).fit(cov_type='HC0')

    log_theta = float(res.params[0])
    H_hat = float(res.params[1])
    se_H = float(res.bse[1])
    t_H = float(res.tvalues[1])
    r2 = float(res.rsquared)

    return {
        'status': 'completed',
        'n_obs': len(sub),
        'H_hat': H_hat,
        'se_H': se_H,
        't_H': t_H,
        'log_theta': log_theta,
        'r2': r2,
    }


def simulate_square_root_scaling(
    n_events_list: Tuple[int, ...] = (10, 50, 100, 500, 1000, 5000),
    n_sims: int = 10000,
    seed: int = 42
) -> pd.DataFrame:
    """
    Paper Section 4: Square-root scaling simulation.
    Confirms that when order flow consists of N random signed events:
      E[|OFI|] grows like sqrt(N), while Total Volume grows like N.
    """
    rng = np.random.default_rng(seed)
    records = []

    for N in n_events_list:
        # Sum of N Rademacher (+-1) variables
        steps = rng.choice([-1, 1], size=(n_sims, N))
        ofi_samples = np.sum(steps, axis=1)
        mean_abs_ofi = np.mean(np.abs(ofi_samples))
        total_vol = float(N)

        records.append({
            'N': N,
            'Total_Volume': total_vol,
            'Mean_Abs_OFI': float(mean_abs_ofi),
            'Theoretical_Sqrt_N': float(np.sqrt(2.0 * N / np.pi)),
            'Ratio_Abs_OFI_to_Sqrt_N': float(mean_abs_ofi / np.sqrt(N))
        })

    return pd.DataFrame(records)
