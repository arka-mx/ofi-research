"""
Depth and price impact regressions module (Step 4).
Implements Regressions (5) and (6) from Cont et al. (2011, Section 3.2).
"""
import warnings
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm


def estimate_depth_impact_relation(
    beta_series: pd.Series,
    ad_series: pd.Series,
    maxlags: int = 5
) -> Dict[str, float]:
    """
    Estimates parameters lambda and c relating price impact beta_i to average depth AD_i:
      Regression (5): log(beta_i) = alpha_L - lambda * log(AD_i) + eps_L
      Regression (6): beta_i = alpha_M + c / (AD_i^lambda) + eps_M

    Aligns series on common index.
    Uses Newey-West (HAC) standard errors.
    Tests H0: lambda = 1.
    """
    # Align on common index
    common_idx = beta_series.dropna().index.intersection(ad_series.dropna().index)
    b = beta_series.loc[common_idx].values
    ad = ad_series.loc[common_idx].values

    # Step 4 rule: Drop windows with non-positive beta before taking logs, report how many dropped
    pos_mask = (b > 0) & (ad > 0)
    n_total = len(b)
    n_dropped = int((~pos_mask).sum())

    b_pos = b[pos_mask]
    ad_pos = ad[pos_mask]

    if len(b_pos) < 5:
        return {
            'n_obs': len(b_pos),
            'n_dropped_nonpositive_beta': n_dropped,
            'lambda_hat': np.nan, 'se_lambda': np.nan, 't_lambda_0': np.nan,
            't_lambda_1': np.nan, 'p_lambda_eq_1': np.nan,
            'ci_lambda_lower': np.nan, 'ci_lambda_upper': np.nan,
            'c_hat': np.nan, 'se_c': np.nan, 't_c': np.nan,
            'ci_c_lower': np.nan, 'ci_c_upper': np.nan,
            'r2_log_reg5': np.nan, 'r2_level_reg6': np.nan,
            'corr_beta_pred_sq': np.nan, 'corr_beta_pred_restr_sq': np.nan
        }

    # Regression (5): log(beta) = a + b_log * log(AD) + eps
    # In the paper: log(beta) = a - lambda * log(AD)  ==> lambda = -b_log
    log_b = np.log(b_pos)
    log_ad = np.log(ad_pos)

    X5 = sm.add_constant(log_ad)
    lags = min(maxlags, max(1, int(len(b_pos) ** 0.25)))

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:
            res5 = sm.OLS(log_b, X5).fit(cov_type='HAC', cov_kwds={'maxlags': lags})
            alpha_L = res5.params[0]
            b_log = res5.params[1]
            se_b_log = res5.bse[1]
            lam = -b_log
            se_lam = se_b_log
            r2_5 = res5.rsquared

            # Test H0: lambda = 1 (i.e. -b_log = 1 => b_log = -1)
            t_lam_1 = (lam - 1.0) / se_lam if se_lam > 0 else np.nan
            p_lam_1 = 2.0 * (1.0 - stats.norm.cdf(abs(t_lam_1))) if not np.isnan(t_lam_1) else np.nan
            t_lam_0 = lam / se_lam if se_lam > 0 else np.nan

            ci_lam_lower = lam - 1.96 * se_lam
            ci_lam_upper = lam + 1.96 * se_lam
        except Exception:
            return {
                'n_obs': len(b_pos),
                'n_dropped_nonpositive_beta': n_dropped,
                'lambda_hat': np.nan, 'se_lambda': np.nan, 't_lambda_0': np.nan,
                't_lambda_1': np.nan, 'p_lambda_eq_1': np.nan,
                'ci_lambda_lower': np.nan, 'ci_lambda_upper': np.nan,
                'c_hat': np.nan, 'se_c': np.nan, 't_c': np.nan,
                'ci_c_lower': np.nan, 'ci_c_upper': np.nan,
                'r2_log_reg5': np.nan, 'r2_level_reg6': np.nan,
                'corr_beta_pred_sq': np.nan, 'corr_beta_pred_restr_sq': np.nan
            }

    # Regression (6): beta = alpha_M + c / (AD^lambda) + eps
    try:
        inv_ad_lam = 1.0 / (ad_pos ** lam)
        X6 = sm.add_constant(inv_ad_lam)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            res6 = sm.OLS(b_pos, X6).fit(cov_type='HAC', cov_kwds={'maxlags': lags})

        alpha_M = res6.params[0]
        c_hat = res6.params[1]
        se_c = res6.bse[1]
        t_c = c_hat / se_c if se_c > 0 else np.nan
        ci_c_lower = c_hat - 1.96 * se_c
        ci_c_upper = c_hat + 1.96 * se_c

        beta_pred = res6.fittedvalues
        corr = float(np.corrcoef(b_pos, beta_pred)[0, 1]) if len(b_pos) > 2 else np.nan
        r2_6 = res6.rsquared
    except Exception:
        c_hat, se_c, t_c = np.nan, np.nan, np.nan
        ci_c_lower, ci_c_upper = np.nan, np.nan
        corr, r2_6 = np.nan, np.nan

    # Restricted model with lambda = 1
    try:
        inv_ad_1 = 1.0 / ad_pos
        X6_restr = sm.add_constant(inv_ad_1)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            res6_restr = sm.OLS(b_pos, X6_restr).fit(cov_type='HAC', cov_kwds={'maxlags': lags})
        beta_pred_restr = res6_restr.fittedvalues
        corr_restr = float(np.corrcoef(b_pos, beta_pred_restr)[0, 1]) if len(b_pos) > 2 else np.nan
    except Exception:
        corr_restr = np.nan

    return {
        'n_obs': int(len(b_pos)),
        'n_dropped_nonpositive_beta': n_dropped,
        'lambda_hat': float(lam),
        'se_lambda': float(se_lam),
        't_lambda_0': float(t_lam_0),
        't_lambda_1': float(t_lam_1),
        'p_lambda_eq_1': float(p_lam_1),
        'ci_lambda_lower': float(ci_lam_lower),
        'ci_lambda_upper': float(ci_lam_upper),
        'c_hat': float(c_hat),
        'se_c': float(se_c),
        't_c': float(t_c),
        'ci_c_lower': float(ci_c_lower),
        'ci_c_upper': float(ci_c_upper),
        'r2_log_reg5': float(r2_5),
        'r2_level_reg6': float(r2_6) if not np.isnan(r2_6) else np.nan,
        'corr_beta_pred_sq': float(corr ** 2) if not np.isnan(corr) else np.nan,
        'corr_beta_pred_restr_sq': float(corr_restr ** 2) if not np.isnan(corr_restr) else np.nan
    }
