"""
Plotting module to reproduce paper figures and project visualizations.
Uses matplotlib only.
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#CCCCCC'
plt.rcParams['axes.linewidth'] = 0.8


def plot_figure_1_acf(
    dp_acf: np.ndarray,
    dp_conf: np.ndarray,
    ofi_acf: np.ndarray,
    ofi_conf: np.ndarray,
    symbol: str,
    output_path: Optional[Union[str, Path]] = None
) -> plt.Figure:
    """
    Figure 1: ACF of mid-price changes delta_P and OFI with 5% bounds.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    lags = np.arange(len(dp_acf))

    # Delta P
    ax1.plot(lags, dp_acf, color='#1f77b4', marker='o', markersize=3, label='ACF')
    ax1.fill_between(lags, dp_conf[:, 0] - dp_acf, dp_conf[:, 1] - dp_acf, color='#1f77b4', alpha=0.2, label='95% CI')
    ax1.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    ax1.set_title(f'ACF of Mid-Price Changes $\\Delta P_k$ ({symbol})', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Lag (5-min intervals)')
    ax1.set_ylabel('Autocorrelation')
    ax1.legend(loc='upper right')

    # OFI
    ax2.plot(lags, ofi_acf, color='#2ca02c', marker='s', markersize=3, label='ACF')
    ax2.fill_between(lags, ofi_conf[:, 0] - ofi_acf, ofi_conf[:, 1] - ofi_acf, color='#2ca02c', alpha=0.2, label='95% CI')
    ax2.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    ax2.set_title(f'ACF of Order Flow Imbalance $OFI_k$ ({symbol})', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Lag (5-min intervals)')
    ax2.legend(loc='upper right')

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    return fig


def plot_figure_2_impact_scatter(
    ofi: np.ndarray,
    delta_p: np.ndarray,
    alpha: float,
    beta: float,
    r2: float,
    symbol: str,
    window_label: str = '',
    output_path: Optional[Union[str, Path]] = None
) -> plt.Figure:
    """
    Figure 2: Scatter plot of delta_P against OFI with fitted line.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(ofi, delta_p, alpha=0.4, color='#1f77b4', edgecolors='none', s=25, label='Observations')

    # Fitted line
    x_grid = np.linspace(np.min(ofi), np.max(ofi), 200)
    y_grid = alpha + beta * x_grid
    r2_pct = r2 * 100.0
    ax.plot(x_grid, y_grid, color='#d62728', linewidth=2.0,
            label=f'OLS: $\\Delta P = {alpha:.3f} + {beta:.5f} \\cdot OFI$\n$R^2$ = {r2_pct:.1f}%')

    ax.set_title(f'Figure 2: Price Impact $\\Delta P$ vs $OFI$ — {symbol} {window_label}', fontsize=12, fontweight='bold')
    ax.set_xlabel('Order Flow Imbalance (OFI)')
    ax.set_ylabel('Mid-Price Change $\\Delta P$ (ticks)')
    ax.legend(loc='upper left', frameon=True)

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    return fig


def plot_figure_3_excess_kurtosis(
    kurtosis_values: np.ndarray,
    output_path: Optional[Union[str, Path]] = None
) -> plt.Figure:
    """
    Figure 3: Distribution of excess kurtosis of residuals across samples.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    clean_kurt = kurtosis_values[~np.isnan(kurtosis_values)]

    ax.hist(clean_kurt, bins=30, color='#3b528b', edgecolor='white', alpha=0.85, density=True)
    ax.axvline(np.mean(clean_kurt), color='#d62728', linestyle='--', linewidth=1.5,
               label=f'Mean = {np.mean(clean_kurt):.2f}')
    ax.axvline(np.median(clean_kurt), color='#2ca02c', linestyle=':', linewidth=1.5,
               label=f'Median = {np.median(clean_kurt):.2f}')

    ax.set_title('Figure 3: Distribution of Excess Kurtosis of Residuals $\\hat{\\epsilon}_k$', fontsize=12, fontweight='bold')
    ax.set_xlabel('Excess Kurtosis')
    ax.set_ylabel('Density')
    ax.legend(loc='upper right')

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    return fig


def plot_figure_4_depth_impact_loglog(
    ad_values: np.ndarray,
    beta_values: np.ndarray,
    lambda_hat: float,
    alpha_L: float,
    r2: float,
    symbol: str,
    output_path: Optional[Union[str, Path]] = None
) -> plt.Figure:
    """
    Figure 4: Log-log scatter plot of beta_i vs AD_i with fitted line.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    pos_mask = (ad_values > 0) & (beta_values > 0)
    log_ad = np.log(ad_values[pos_mask])
    log_beta = np.log(beta_values[pos_mask])

    ax.scatter(log_ad, log_beta, color='#2b83ba', alpha=0.7, edgecolors='k', s=35, label='Windows')

    if len(log_ad) > 2:
        x_grid = np.linspace(np.min(log_ad), np.max(log_ad), 100)
        y_grid = alpha_L - lambda_hat * x_grid
        r2_pct = r2 * 100.0
        ax.plot(x_grid, y_grid, color='#d7191c', linewidth=2,
                label=f'Fit: $\\log(\\beta) = {alpha_L:.2f} - {lambda_hat:.2f}\\log(AD)$\n$R^2$ = {r2_pct:.1f}%')

    ax.set_title(f'Figure 4: Log Price Impact $\\hat{{\\beta}}_i$ vs Market Depth $AD_i$ ({symbol})', fontsize=12, fontweight='bold')
    ax.set_xlabel('$\\log(AD_i)$')
    ax.set_ylabel('$\\log(\\hat{\\beta}_i)$')
    ax.legend(loc='upper right')

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    return fig


def plot_figure_5_intraday_patterns(
    intraday_df: pd.DataFrame,
    output_path: Optional[Union[str, Path]] = None
) -> plt.Figure:
    """
    Figure 5: Intraday patterns in normalized beta and AD.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    x_labels = intraday_df['time_of_day'].values
    x_idx = np.arange(len(x_labels))

    norm_beta = intraday_df['beta'] / intraday_df['beta'].mean()
    norm_ad = intraday_df['avg_depth'] / intraday_df['avg_depth'].mean()

    ax.plot(x_idx, norm_beta, color='#d62728', marker='o', linewidth=2, label='Normalized Price Impact $\\hat{\\beta}_i$')
    ax.plot(x_idx, norm_ad, color='#1f77b4', marker='s', linewidth=2, label='Normalized Market Depth $AD_i$')

    ax.set_xticks(x_idx[::4])
    ax.set_xticklabels(x_labels[::4], rotation=45)
    ax.set_title('Figure 5: Intraday Seasonality in Normalized Impact and Depth', fontsize=12, fontweight='bold')
    ax.set_ylabel('Normalized Level (Relative to Daily Mean)')
    ax.legend(loc='upper right')

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    return fig


def plot_figure_6_variance_identity(
    intraday_df: pd.DataFrame,
    output_path: Optional[Union[str, Path]] = None
) -> plt.Figure:
    """
    Figure 6: Identity var[delta_P] ~ beta^2 * var[OFI].
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    x_labels = intraday_df['time_of_day'].values
    x_idx = np.arange(len(x_labels))

    ax.plot(x_idx, intraday_df['var_delta_p'], color='#1f77b4', marker='o', linewidth=2, label='Actual $\\mathrm{var}[\\Delta P_k]_i$')
    ax.plot(x_idx, intraday_df['pred_var_delta_p'], color='#ff7f0e', marker='^', linestyle='--', linewidth=2,
            label='Predicted $\\hat{\\beta}_i^2 \\cdot \\mathrm{var}[OFI_k]_i$')

    ax.set_xticks(x_idx[::4])
    ax.set_xticklabels(x_labels[::4], rotation=45)
    ax.set_title('Figure 6: Intraday Variance Identity $\\mathrm{var}[\\Delta P] \\approx \\hat{\\beta}^2 \\mathrm{var}[OFI]$', fontsize=12, fontweight='bold')
    ax.set_ylabel('Variance')
    ax.legend(loc='upper right')

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    return fig


def plot_figure_7_prediction_oos(
    y_act_0: np.ndarray,
    y_pred_0: np.ndarray,
    r2_0: float,
    y_act_1: np.ndarray,
    y_pred_1: np.ndarray,
    r2_1: float,
    symbol: str,
    output_path: Optional[Union[str, Path]] = None
) -> plt.Figure:
    """
    Figure 7: Out-of-sample actual vs predicted delta_P for Lag 0 vs Lag 1.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    n_show = min(150, len(y_act_0))
    idx = np.arange(n_show)

    # Lag 0 (Contemporaneous)
    r2_0_pct = r2_0 * 100.0
    ax1.plot(idx, y_act_0[:n_show], color='#333333', linewidth=1.2, label='Actual $\\Delta P_t$ (ticks)')
    ax1.plot(idx, y_pred_0[:n_show], color='#1f77b4', linestyle='--', linewidth=1.5,
             label=f'Fitted (Lag 0 OFI contemporaneous) [$R^2$ = {r2_0_pct:.1f}%]')
    ax1.set_title(f'Lag 0 (Contemporaneous Benchmark) — {symbol} Out-of-Sample Window', fontsize=11, fontweight='bold')
    ax1.set_ylabel('$\\Delta P$ (ticks)')
    ax1.legend(loc='upper right')

    # Lag 1 (Predictive)
    r2_1_pct = r2_1 * 100.0
    ax2.plot(idx, y_act_1[:n_show], color='#333333', linewidth=1.2, label='Actual $\\Delta P_t$ (ticks)')
    ax2.plot(idx, y_pred_1[:n_show], color='#d62728', linestyle='--', linewidth=1.5,
             label=f'Predicted (Lag 1 $OFI_{{t-1}}$) [$R^2$ = {r2_1_pct:.1f}%]')
    ax2.set_title(f'Lag 1 (One 5-min Interval Ahead Forecast) — {symbol} Out-of-Sample Window', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Out-of-Sample Snapshot Index')
    ax2.set_ylabel('$\\Delta P$ (ticks)')
    ax2.legend(loc='upper right')

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    return fig


def plot_all_orderbooks_lag0_lag1(
    orderbook_metrics_df: pd.DataFrame,
    output_path: Optional[Union[str, Path]] = None
) -> plt.Figure:
    """
    Comprehensive figure across ALL 24 order books:
    Compares R2 at Lag 0 vs Lag 1 for every instrument.
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    symbols = orderbook_metrics_df['symbol'].values
    x = np.arange(len(symbols))
    width = 0.38

    r2_0 = orderbook_metrics_df['r2_lag0'].values * 100.0
    r2_1 = orderbook_metrics_df['r2_lag1'].values * 100.0

    rects1 = ax.bar(x - width/2, r2_0, width, label='Lag 0 (Contemporaneous $R^2$ %)', color='#1f77b4', alpha=0.9)
    rects2 = ax.bar(x + width/2, r2_1, width, label='Lag 1 (Predictive $R^2$ %)', color='#ff7f0e', alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels(symbols, rotation=45, ha='right', fontsize=10, fontweight='bold')
    ax.set_ylabel('Explained Variance $R^2$ (%)', fontsize=11)
    ax.set_title('Price Impact Regression Across All 24 Crypto Order Books: Lag 0 vs Lag 1 $R^2$', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', frameon=True)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    return fig


def plot_figure_2_impact_scatter_with_microprice(
    ofi: np.ndarray,
    delta_p_mid: np.ndarray,
    delta_p_micro: np.ndarray,
    alpha_mid: float,
    beta_mid: float,
    r2_mid: float,
    alpha_micro: float,
    beta_micro: float,
    r2_micro: float,
    symbol: str,
    window_label: str = '',
    output_path: Optional[Union[str, Path]] = None
) -> plt.Figure:
    """
    Figure 2 Extension: Price Impact Comparison on the SAME GRAPH.
    Overlays Mid-Price Change (delta P_mid) and Microprice Change (delta P_micro) against OFI.
    """
    fig, ax = plt.subplots(figsize=(9, 6.5))

    # Scatter points for both mid-price and microprice changes
    ax.scatter(ofi, delta_p_mid, alpha=0.30, color='#1f77b4', edgecolors='none', s=22,
               label=r'Mid-Price Obs ($\Delta P^{\mathrm{mid}}$)')
    ax.scatter(ofi, delta_p_micro, alpha=0.30, color='#ff7f0e', edgecolors='none', s=22,
               label=r'Microprice Obs ($\Delta P^{\mathrm{micro}}$)')

    # Regression lines
    x_grid = np.linspace(np.min(ofi), np.max(ofi), 200)
    y_mid = alpha_mid + beta_mid * x_grid
    y_micro = alpha_micro + beta_micro * x_grid

    ax.plot(x_grid, y_mid, color='#08519c', linewidth=2.2, linestyle='-',
            label=f'Mid-Price OLS: $\\beta={beta_mid:.4f}$, $R^2={r2_mid*100:.1f}\\%$')
    ax.plot(x_grid, y_micro, color='#d7301f', linewidth=2.2, linestyle='--',
            label=f'Microprice OLS: $\\beta={beta_micro:.4f}$, $R^2={r2_micro*100:.1f}\\%$')

    title = f'Price Impact on OFI: Mid-Price vs Microprice — {symbol}'
    if window_label:
        title += f' ({window_label})'
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('Order Flow Imbalance (OFI)', fontsize=11)
    ax.set_ylabel('Price Change $\\Delta P$ (ticks)', fontsize=11)
    ax.legend(loc='upper left', frameon=True, fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    return fig


def plot_expected_price_trajectory(
    df: pd.DataFrame,
    symbol: str,
    n_bars: int = 100,
    time_col: str = 'timestamp_utc',
    mid_col: str = 'mid_price',
    micro_col: str = 'microprice',
    exp_col: str = 'expected_price_mid_ofi',
    output_path: Optional[Union[str, Path]] = None
) -> plt.Figure:
    """
    Plots the price trajectory and expected price on the SAME GRAPH:
    - Actual Mid-Price
    - Microprice (Instantaneous fair price)
    - OFI Model-implied Expected Price
    With a bottom panel showing the Microprice spread adjustment (P_micro - P_mid).
    """
    sub_df = df.dropna(subset=[mid_col, micro_col, exp_col]).tail(n_bars).copy()
    times = pd.to_datetime(sub_df[time_col])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

    # Top Panel: Prices on the same graph
    ax1.plot(times, sub_df[mid_col], color='#2b83ba', linewidth=1.8, label=r'Mid-Price $P^{\mathrm{mid}}$')
    ax1.plot(times, sub_df[micro_col], color='#d7191c', linewidth=1.5, linestyle='--',
             label=r'Microprice $P^{\mathrm{micro}}$ (Instantaneous Expected Price)')
    ax1.plot(times, sub_df[exp_col], color='#2ca02c', linewidth=1.3, linestyle=':',
             label=r'Expected Price $\hat{P}^{\mathrm{exp}}_{\mathrm{OFI}}$ (Model-Implied)')

    ax1.set_title(f'Expected Price & Microprice Trajectory on the Same Graph — {symbol}', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Price (USDT)', fontsize=11)
    ax1.legend(loc='upper left', frameon=True, fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Bottom Panel: Microprice Adjustment (P_micro - P_mid)
    spread_adj = sub_df[micro_col] - sub_df[mid_col]
    ax2.plot(times, spread_adj, color='#756bb1', linewidth=1.2, label=r'$P^{\mathrm{micro}} - P^{\mathrm{mid}}$ (Spread Bias)')
    ax2.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    ax2.fill_between(times, 0, spread_adj, color='#756bb1', alpha=0.25)
    ax2.set_ylabel('Adjustment (USDT)', fontsize=10)
    ax2.set_xlabel('Timestamp (UTC)', fontsize=11)
    ax2.legend(loc='upper left', frameon=True, fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    return fig


def plot_microprice_impact_all_orderbooks(
    comparison_df: pd.DataFrame,
    output_path: Optional[Union[str, Path]] = None
) -> plt.Figure:
    """
    Bar chart comparing Price Impact R^2 between Mid-Price and Microprice across all 24 crypto order books.
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    symbols = comparison_df['Ticker'].values
    x = np.arange(len(symbols))
    width = 0.38

    r2_mid = comparison_df['R2_mid'].values * 100.0
    r2_micro = comparison_df['R2_micro'].values * 100.0

    ax.bar(x - width/2, r2_mid, width, label='Mid-Price Impact ($R^2$ %)', color='#1f77b4', alpha=0.9)
    ax.bar(x + width/2, r2_micro, width, label='Microprice Impact ($R^2$ %)', color='#e6550d', alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels(symbols, rotation=45, ha='right', fontsize=10, fontweight='bold')
    ax.set_ylabel('Explained Variance $R^2$ (%)', fontsize=11)
    ax.set_title('Price Impact Regression Across All 24 Crypto Order Books: Mid-Price vs Microprice $R^2$',
                 fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', frameon=True)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    return fig


def plot_lag0_vs_lag1_regression_side_by_side(
    df: pd.DataFrame,
    symbol: str,
    train_frac: float = 0.70,
    ofi_col: str = 'OFI',
    delta_p_col: str = 'delta_P',
    output_path: Optional[Union[str, Path]] = None
) -> plt.Figure:
    """
    Plots the Lag 0 and Lag 1 regressions SIDE-BY-SIDE:
    - Left Panel:  Lag 0 (Contemporaneous: delta P_t on OFI_t)
    - Right Panel: Lag 1 (Predictive Forecast: delta P_t on OFI_{t-1})
    """
    from ofi_paper.models.lag_test import chronological_train_test_split, compute_lag_regression_metrics

    clean = df.dropna(subset=[ofi_col, delta_p_col]).copy()
    tr, te = chronological_train_test_split(clean, train_frac=train_frac)

    m0 = compute_lag_regression_metrics(tr, te, lag=0, delta_p_col=delta_p_col, ofi_col=ofi_col)
    m1 = compute_lag_regression_metrics(tr, te, lag=1, delta_p_col=delta_p_col, ofi_col=ofi_col)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # --- LEFT PANEL: LAG 0 (Contemporaneous) ---
    x0 = te[ofi_col].values
    y0 = te[delta_p_col].values
    val0 = ~(np.isnan(x0) | np.isnan(y0))
    x0, y0 = x0[val0], y0[val0]

    ax1.scatter(x0, y0, alpha=0.35, color='#1f77b4', edgecolors='none', s=25, label='Observations')
    if len(x0) > 2 and 'beta_hat' in m0:
        x0_grid = np.linspace(np.min(x0), np.max(x0), 150)
        y0_grid = m0['alpha_hat'] + m0['beta_hat'] * x0_grid
        r2_0_pct = m0['r2_in_sample'] * 100.0
        ax1.plot(x0_grid, y0_grid, color='#08519c', linewidth=2.2,
                 label=f"Fit: $\\Delta P_t = {m0['alpha_hat']:+.3f} + ({m0['beta_hat']:.5f}) \\cdot OFI_t$\n$R^2 = {r2_0_pct:.1f}\\%$, $t_\\beta = {m0['t_beta_white']:.2f}$")

    ax1.set_title(f"Lag 0: Contemporaneous Price Impact — {symbol}\n($\\Delta P_t$ vs $OFI_t$)",
                  fontsize=12, fontweight='bold')
    ax1.set_xlabel("Order Flow Imbalance ($OFI_t$)", fontsize=11)
    ax1.set_ylabel("Mid-Price Change $\\Delta P_t$ (ticks)", fontsize=11)
    ax1.legend(loc='upper left', frameon=True, fontsize=10)
    ax1.grid(True, alpha=0.3)

    # --- RIGHT PANEL: LAG 1 (Predictive Forecast) ---
    clean['ofi_lag1'] = clean[ofi_col].shift(1)
    tr_l, te_l = chronological_train_test_split(clean.dropna(subset=['ofi_lag1', delta_p_col]), train_frac=train_frac)
    x1 = te_l['ofi_lag1'].values
    y1 = te_l[delta_p_col].values

    ax2.scatter(x1, y1, alpha=0.35, color='#ff7f0e', edgecolors='none', s=25, label='Observations')
    if len(x1) > 2 and 'beta_hat' in m1:
        x1_grid = np.linspace(np.min(x1), np.max(x1), 150)
        y1_grid = m1['alpha_hat'] + m1['beta_hat'] * x1_grid
        r2_1_pct = m1['r2_oos'] * 100.0
        ax2.plot(x1_grid, y1_grid, color='#d62728', linewidth=2.2, linestyle='--',
                 label=f"Fit: $\\Delta P_t = {m1['alpha_hat']:+.3f} + ({m1['beta_hat']:.5f}) \\cdot OFI_{{t-1}}$\n$R^2_{{\\mathrm{{OOS}}}} = {r2_1_pct:.2f}\\%$, $t_\\beta = {m1['t_beta_white']:.2f}$")

    ax2.set_title(f"Lag 1: Predictive Return Forecast — {symbol}\n($\\Delta P_t$ vs $OFI_{{t-1}}$)",
                  fontsize=12, fontweight='bold')
    ax2.set_xlabel("Lagged Order Flow Imbalance ($OFI_{t-1}$)", fontsize=11)
    ax2.set_ylabel("Mid-Price Change $\\Delta P_t$ (ticks)", fontsize=11)
    ax2.legend(loc='upper left', frameon=True, fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    return fig
