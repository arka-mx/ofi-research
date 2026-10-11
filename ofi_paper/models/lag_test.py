"""
OFI predictive lag test module (Step 7 - Extension).
Evaluates whether lagged OFI predicts price changes out-of-sample.
"""
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import statsmodels.api as sm


def chronological_train_test_split(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    date_col: str = 'timestamp_utc'
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits DataFrame chronologically by date/time without shuffling.
    Strictly forbids random splits.
    """
    sorted_df = df.sort_values(date_col).reset_index(drop=True)
    split_idx = int(len(sorted_df) * train_frac)
    train_df = sorted_df.iloc[:split_idx].copy()
    test_df = sorted_df.iloc[split_idx:].copy()
    return train_df, test_df


def compute_lag_regression_metrics(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    lag: int,
    delta_p_col: str = 'delta_P',
    ofi_col: str = 'OFI'
) -> Dict[str, float]:
    """
    Regresses delta_P_t on OFI_{t-L} for lag L.
    Fits OLS with White HC0 SE on train_df.
    Evaluates out-of-sample R2 = 1 - SSE / SST (where SST uses train mean) on test_df.
    """
    # Prepare train data with lag
    df_tr = train_df[[delta_p_col, ofi_col]].copy()
    df_tr['ofi_lag'] = df_tr[ofi_col].shift(lag) if lag > 0 else df_tr[ofi_col]
    df_tr = df_tr.dropna()

    if len(df_tr) < 20:
        return {'lag': lag, 'train_n': len(df_tr), 'test_n': 0, 'r2_oos': np.nan}

    y_tr = df_tr[delta_p_col].values
    x_tr = df_tr['ofi_lag'].values

    X_tr_mat = sm.add_constant(x_tr)
    model = sm.OLS(y_tr, X_tr_mat)
    res = model.fit(cov_type='HC0')

    alpha_hat = res.params[0]
    beta_hat = res.params[1]
    t_beta = res.tvalues[1]
    p_beta = res.pvalues[1]
    r2_is = res.rsquared

    # Prepare test data with lag
    df_te = test_df[[delta_p_col, ofi_col]].copy()
    df_te['ofi_lag'] = df_te[ofi_col].shift(lag) if lag > 0 else df_te[ofi_col]
    df_te = df_te.dropna()

    if len(df_te) < 10:
        return {
            'lag': lag, 'train_n': len(df_tr), 'test_n': len(df_te),
            'beta_hat': beta_hat, 't_beta': t_beta, 'r2_is': r2_is, 'r2_oos': np.nan
        }

    y_te = df_te[delta_p_col].values
    x_te = df_te['ofi_lag'].values

    y_pred_te = alpha_hat + beta_hat * x_te

    # Out of sample R2 using training mean
    y_tr_mean = np.mean(y_tr)
    sse = np.sum((y_te - y_pred_te) ** 2)
    sst = np.sum((y_te - y_tr_mean) ** 2)
    r2_oos = 1.0 - (sse / sst) if sst > 0 else np.nan

    # Correlation between predicted and actual
    corr_oos = float(np.corrcoef(y_te, y_pred_te)[0, 1]) if len(y_te) > 2 else np.nan

    return {
        'lag': lag,
        'train_n': int(len(df_tr)),
        'test_n': int(len(df_te)),
        'alpha_hat': float(alpha_hat),
        'beta_hat': float(beta_hat),
        't_beta_white': float(t_beta),
        'p_beta': float(p_beta),
        'r2_in_sample': float(r2_is),
        'r2_oos': float(r2_oos),
        'corr_oos': float(corr_oos),
        'y_test_actual': y_te,
        'y_test_pred': y_pred_te
    }


def run_all_lags_test(
    df: pd.DataFrame,
    lags: List[int] = [0, 1, 2, 3, 4, 5, 6],
    train_frac: float = 0.70,
    delta_p_col: str = 'delta_P',
    ofi_col: str = 'OFI'
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Runs predictive test for all specified lags and checks the sanity relation:
      R2(lag 1) ~ R2(lag 0) * rho^2
    """
    train_df, test_df = chronological_train_test_split(df, train_frac)

    # Compute lag-1 autocorrelation of OFI (rho)
    ofi_clean = df[ofi_col].dropna()
    rho = float(ofi_clean.autocorr(lag=1)) if len(ofi_clean) > 10 else np.nan

    results = []
    for lag in lags:
        m = compute_lag_regression_metrics(train_df, test_df, lag, delta_p_col, ofi_col)
        results.append({
            'lag': lag,
            'train_n': m.get('train_n', 0),
            'test_n': m.get('test_n', 0),
            'alpha_hat': m.get('alpha_hat', np.nan),
            'beta_hat': m.get('beta_hat', np.nan),
            't_beta_white': m.get('t_beta_white', np.nan),
            'p_beta': m.get('p_beta', np.nan),
            'r2_in_sample': m.get('r2_in_sample', np.nan),
            'r2_oos': m.get('r2_oos', np.nan),
            'corr_oos': m.get('corr_oos', np.nan),
        })

    res_df = pd.DataFrame(results)

    # Sanity relation check
    r2_lag0 = res_df.loc[res_df['lag'] == 0, 'r2_in_sample'].values[0] if 0 in res_df['lag'].values else np.nan
    r2_lag1 = res_df.loc[res_df['lag'] == 1, 'r2_in_sample'].values[0] if 1 in res_df['lag'].values else np.nan
    theoretical_r2_lag1 = r2_lag0 * (rho ** 2) if not np.isnan(r2_lag0) and not np.isnan(rho) else np.nan

    sanity = {
        'rho_ofi_lag1': rho,
        'r2_lag0': r2_lag0,
        'r2_lag1_empirical': r2_lag1,
        'r2_lag1_theoretical': theoretical_r2_lag1
    }

    return res_df, sanity
