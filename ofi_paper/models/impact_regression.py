"""
Price impact regression module (Step 3).
Implements OLS with White HC0 standard errors and quadratic variant from Cont et al. (2011, Section 3.2).
"""
import warnings
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm


def fit_ols_white(
    y: np.ndarray,
    X: np.ndarray,
    add_constant: bool = True
) -> Dict[str, float]:
    """
    Fits OLS with White (HC0) heteroscedasticity-robust covariance.
    Returns dictionary with coefficients, robust standard errors, t-stats, p-values, R2, and residuals.
    """
    mask = ~(np.isnan(y) | np.isnan(X).any(axis=1))
    y_clean = y[mask]
    X_clean = X[mask]

    if len(y_clean) < (X_clean.shape[1] + (2 if add_constant else 1)):
        return {
            'alpha': np.nan, 'beta': np.nan, 't_alpha': np.nan, 't_beta': np.nan,
            'p_alpha': np.nan, 'p_beta': np.nan, 'r2': np.nan, 'n_obs': len(y_clean),
            'kurtosis': np.nan
        }

    # Check for zero variance in regressor
    if np.all(np.std(X_clean, axis=0) < 1e-12):
        return {
            'alpha': float(np.mean(y_clean)) if add_constant else 0.0,
            'beta': 0.0, 't_alpha': np.nan, 't_beta': np.nan,
            'p_alpha': np.nan, 'p_beta': np.nan, 'r2': 0.0, 'n_obs': len(y_clean),
            'kurtosis': np.nan
        }

    if add_constant:
        X_mat = sm.add_constant(X_clean, has_constant='add')
    else:
        X_mat = X_clean

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:
            model = sm.OLS(y_clean, X_mat)
            results = model.fit(cov_type='HC0')
        except Exception:
            return {
                'alpha': np.nan, 'beta': np.nan, 't_alpha': np.nan, 't_beta': np.nan,
                'p_alpha': np.nan, 'p_beta': np.nan, 'r2': np.nan, 'n_obs': len(y_clean),
                'kurtosis': np.nan
            }

    params = results.params
    tvalues = results.tvalues
    pvalues = results.pvalues
    r2 = results.rsquared
    resid = results.resid

    # Excess kurtosis (Fisher definition: normal = 0)
    kurt = float(stats.kurtosis(resid, fisher=True)) if len(resid) > 4 else np.nan

    out = {
        'alpha': float(params[0]) if add_constant else 0.0,
        'beta': float(params[1]) if add_constant else float(params[0]),
        't_alpha': float(tvalues[0]) if add_constant else np.nan,
        't_beta': float(tvalues[1]) if add_constant else float(tvalues[0]),
        'p_alpha': float(pvalues[0]) if add_constant else np.nan,
        'p_beta': float(pvalues[1]) if add_constant else float(pvalues[0]),
        'r2': float(r2),
        'n_obs': int(len(y_clean)),
        'kurtosis': kurt,
        'residuals': resid,
        'fitted': results.fittedvalues
    }
    return out


def fit_quadratic_ols_white(
    y: np.ndarray,
    ofi: np.ndarray
) -> Dict[str, float]:
    """
    Fits quadratic variant: delta_P = alpha + beta * OFI + gamma_Q * OFI * |OFI| + eps
    with White HC0 standard errors.
    """
    mask = ~(np.isnan(y) | np.isnan(ofi))
    y_c = y[mask]
    ofi_c = ofi[mask]

    if len(y_c) < 5 or np.std(ofi_c) < 1e-12:
        return {
            'alpha': np.nan, 'beta': np.nan, 'gamma_Q': np.nan,
            't_alpha': np.nan, 't_beta': np.nan, 't_gamma_Q': np.nan,
            'p_gamma_Q': np.nan, 'r2': np.nan, 'n_obs': len(y_c)
        }

    ofi_quad = ofi_c * np.abs(ofi_c)
    X = np.column_stack([ofi_c, ofi_quad])
    X_mat = sm.add_constant(X)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:
            model = sm.OLS(y_c, X_mat)
            results = model.fit(cov_type='HC0')
        except Exception:
            return {
                'alpha': np.nan, 'beta': np.nan, 'gamma_Q': np.nan,
                't_alpha': np.nan, 't_beta': np.nan, 't_gamma_Q': np.nan,
                'p_gamma_Q': np.nan, 'r2': np.nan, 'n_obs': len(y_c)
            }

    return {
        'alpha': float(results.params[0]),
        'beta': float(results.params[1]),
        'gamma_Q': float(results.params[2]),
        't_alpha': float(results.tvalues[0]),
        't_beta': float(results.tvalues[1]),
        't_gamma_Q': float(results.tvalues[2]),
        'p_gamma_Q': float(results.pvalues[2]),
        'r2': float(results.rsquared),
        'n_obs': int(len(y_c))
    }


def run_windowed_impact_regressions(
    df: pd.DataFrame,
    window_col: str = 'window_id',
    delta_p_col: str = 'delta_P',
    ofi_col: str = 'OFI',
    tautology_ofi_col: Optional[str] = 'OFI_tautology'
) -> pd.DataFrame:
    """
    Runs price impact regressions for every window in df.
    """
    records = []

    for wid, group in df.groupby(window_col):
        y = group[delta_p_col].values
        x = group[ofi_col].values

        lin = fit_ols_white(y, x.reshape(-1, 1), add_constant=True)
        if np.isnan(lin['beta']):
            continue

        quad = fit_quadratic_ols_white(y, x)

        rec = {
            'window_id': wid,
            'n_obs': lin['n_obs'],
            'alpha': lin['alpha'],
            't_alpha': lin['t_alpha'],
            'p_alpha': lin['p_alpha'],
            'beta': lin['beta'],
            't_beta': lin['t_beta'],
            'p_beta': lin['p_beta'],
            'r2': lin['r2'],
            'excess_kurtosis': lin['kurtosis'],
            'gamma_Q': quad['gamma_Q'],
            't_gamma_Q': quad['t_gamma_Q'],
            'p_gamma_Q': quad['p_gamma_Q'],
            'r2_quad': quad['r2'],
            'r2_increase': quad['r2'] - lin['r2'] if not np.isnan(quad['r2']) else np.nan,
            'beta_sig_5pct': (lin['p_beta'] < 0.05) if not np.isnan(lin['p_beta']) else False,
            'alpha_sig_5pct': (lin['p_alpha'] < 0.05) if not np.isnan(lin['p_alpha']) else False,
            'gamma_sig_5pct': (quad['p_gamma_Q'] < 0.05) if not np.isnan(quad['p_gamma_Q']) else False,
        }

        if tautology_ofi_col and tautology_ofi_col in group.columns:
            x_taut = group[tautology_ofi_col].values
            lin_taut = fit_ols_white(y, x_taut.reshape(-1, 1), add_constant=True)
            rec['r2_tautology'] = lin_taut['r2']
            rec['r2_drop_tautology'] = (lin['r2'] - lin_taut['r2']) if not np.isnan(lin_taut['r2']) else np.nan

        records.append(rec)

    return pd.DataFrame(records)
