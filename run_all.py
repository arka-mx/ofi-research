"""
End-to-End Orchestrator for OFI Research (Cont, Kukanov, Stoikov 2011).
CLI interface supporting: --data, --tick, --session-start, --session-end, --cost, --train-frac, --lags
"""
import argparse
import os
from pathlib import Path
import sys
import numpy as np
import pandas as pd

from ofi_paper.data.loader import get_default_tick_sizes, load_raw_dataset, list_available_instruments
from ofi_paper.data.clean import clean_orderbook_data
from ofi_paper.features.ofi import compute_ofi_series, compute_multilevel_ofi
from ofi_paper.features.depth import compute_instantaneous_depth, compute_window_depth
from ofi_paper.features.microprice import (
    compute_microprice, compute_microprice_change, compute_expected_price, compare_price_impact_mid_vs_micro
)
from ofi_paper.models.impact_regression import (
    fit_ols_white, fit_quadratic_ols_white, run_windowed_impact_regressions
)
from ofi_paper.models.depth_regression import estimate_depth_impact_relation
from ofi_paper.models.lag_test import run_all_lags_test, chronological_train_test_split, compute_lag_regression_metrics
from ofi_paper.analysis.intraday import compute_acf_series, compute_intraday_profiles
from ofi_paper.analysis.trades_vs_ofi import estimate_volume_exponent, simulate_square_root_scaling
from ofi_paper.analysis.tradability import evaluate_trading_signal
from ofi_paper.plots.figures import (
    plot_figure_1_acf, plot_figure_2_impact_scatter, plot_figure_3_excess_kurtosis,
    plot_figure_4_depth_impact_loglog, plot_figure_5_intraday_patterns,
    plot_figure_6_variance_identity, plot_figure_7_prediction_oos,
    plot_all_orderbooks_lag0_lag1,
    plot_figure_2_impact_scatter_with_microprice,
    plot_expected_price_trajectory,
    plot_microprice_impact_all_orderbooks,
    plot_lag0_vs_lag1_regression_side_by_side
)


def parse_args():
    parser = argparse.ArgumentParser(description="Empirical Order Flow Imbalance (Cont et al. 2011)")
    parser.add_argument("--data", type=str, default="public/dex l2 5 min", help="Path to data directory or file")
    parser.add_argument("--tick", type=float, default=None, help="Tick size (if None, uses default per instrument)")
    parser.add_argument("--session-start", type=str, default=None, help="Session start time HH:MM")
    parser.add_argument("--session-end", type=str, default=None, help="Session end time HH:MM")
    parser.add_argument("--cost", type=float, default=1.0, help="Round-trip transaction cost in ticks")
    parser.add_argument("--train-frac", type=float, default=0.70, help="Chronological training split fraction")
    parser.add_argument("--lags", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6], help="Lags for predictive test")
    parser.add_argument("--symbol", type=str, default=None, help="Filter to a single ticker symbol")
    return parser.parse_args()


def main():
    args = parse_args()
    base_dir = Path(__file__).resolve().parent
    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = base_dir / data_path

    tables_dir = base_dir / "output" / "tables"
    figures_dir = base_dir / "output" / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    default_ticks = get_default_tick_sizes()

    if data_path.is_file():
        files = [data_path]
    else:
        files = sorted(list(data_path.glob("*.parquet")))
        if not files:
            files = sorted(list(data_path.glob("*.csv")))

    if args.symbol:
        files = [f for f in files if args.symbol.upper() in f.stem.upper()]

    if not files:
        print(f"No data files found in {data_path}")
        sys.exit(1)

    print(f"Processing {len(files)} instrument files...")

    all_table2_rows = []
    all_table3_rows = []
    all_orderbook_metrics = []
    all_tradability_rows = []
    all_kurtosis = []
    all_microprice_rows = []
    intraday_profiles_list = []
    rep_stock_data = {}

    for f in files:
        sym = f.stem.replace("_sample_7d", "")
        tick_sz = args.tick if args.tick is not None else default_ticks.get(sym, 0.0001)

        raw_df = load_raw_dataset(f)
        cleaned_df = clean_orderbook_data(
            raw_df,
            tick_size=tick_sz,
            spread_pct_cutoff=95.0,
            max_gap_minutes=5.0,
            session_start=args.session_start,
            session_end=args.session_end
        )

        # OFI standard and tautology-filtered
        cleaned_df['OFI'] = compute_ofi_series(cleaned_df, tautology_filter=False)
        cleaned_df['OFI_tautology'] = compute_ofi_series(cleaned_df, tautology_filter=True)
        cleaned_df['AD'] = compute_instantaneous_depth(cleaned_df)

        # Microprice calculation
        cleaned_df['microprice'] = compute_microprice(cleaned_df)
        _, cleaned_df['delta_P_micro'] = compute_microprice_change(cleaned_df, tick_size=tick_sz)

        # Group into daily windows or half-hour intervals
        cleaned_df['day_id'] = cleaned_df['timestamp_utc'].dt.date
        cleaned_df['window_id'] = (
            cleaned_df['timestamp_utc'].dt.floor('4h').astype(str)
        )

        # Step 3: Run impact regressions across windows
        win_reg = run_windowed_impact_regressions(
            cleaned_df,
            window_col='window_id',
            delta_p_col='delta_P',
            ofi_col='OFI',
            tautology_ofi_col='OFI_tautology'
        )

        # Overall full-sample regression for this stock
        y_all = cleaned_df['delta_P'].values
        x_all = cleaned_df['OFI'].values
        lin_full = fit_ols_white(y_all, x_all.reshape(-1, 1), add_constant=True)
        quad_full = fit_quadratic_ols_white(y_all, x_all)

        # Microprice impact on full sample
        y_micro = cleaned_df['delta_P_micro'].values
        valid_micro = ~(np.isnan(y_micro) | np.isnan(x_all))
        lin_micro = fit_ols_white(y_micro[valid_micro], x_all[valid_micro].reshape(-1, 1), add_constant=True)

        all_microprice_rows.append({
            'Ticker': sym,
            'Equation_Mid': f"ΔP_mid = {lin_full['alpha']:.4f} + ({lin_full['beta']:.5f})·OFI",
            'Equation_Micro': f"ΔP_micro = {lin_micro['alpha']:.4f} + ({lin_micro['beta']:.5f})·OFI",
            'alpha_mid': lin_full['alpha'],
            'beta_mid': lin_full['beta'],
            't_beta_mid': lin_full['t_beta'],
            'R2_mid': lin_full['r2'],
            'alpha_micro': lin_micro['alpha'],
            'beta_micro': lin_micro['beta'],
            't_beta_micro': lin_micro['t_beta'],
            'R2_micro': lin_micro['r2'],
            'delta_beta': lin_micro['beta'] - lin_full['beta'],
            'delta_R2': lin_micro['r2'] - lin_full['r2']
        })

        # Tautology check on full sample
        x_taut = cleaned_df['OFI_tautology'].values
        taut_full = fit_ols_white(y_all, x_taut.reshape(-1, 1), add_constant=True)

        kurt_vals = win_reg['excess_kurtosis'].dropna().values
        all_kurtosis.extend(kurt_vals)

        table2_entry = {
            'Ticker': sym,
            'Equation_Linear': f"ΔP = {lin_full['alpha']:.4f} + ({lin_full['beta']:.5f})·OFI",
            'Equation_Quad': f"ΔP = {quad_full['alpha']:.4f} + ({quad_full['beta']:.5f})·OFI + ({quad_full['gamma_Q']:.7f})·OFI|OFI|",
            'alpha': lin_full['alpha'],
            't_alpha': lin_full['t_alpha'],
            'beta': lin_full['beta'],
            't_beta': lin_full['t_beta'],
            'gamma_Q': quad_full['gamma_Q'],
            't_gamma_Q': quad_full['t_gamma_Q'],
            'R2': lin_full['r2'],
            'R2_quad': quad_full['r2'],
            'R2_drop_tautology': (lin_full['r2'] - taut_full['r2']),
            'pct_win_beta_sig': float(win_reg['beta_sig_5pct'].mean() * 100.0) if len(win_reg) > 0 else np.nan,
            'pct_win_alpha_sig': float(win_reg['alpha_sig_5pct'].mean() * 100.0) if len(win_reg) > 0 else np.nan,
            'pct_win_gamma_sig': float(win_reg['gamma_sig_5pct'].mean() * 100.0) if len(win_reg) > 0 else np.nan,
        }
        all_table2_rows.append(table2_entry)

        # Step 4: Window depth vs beta
        # Calculate window average depth AD_i
        win_ad = cleaned_df.groupby('window_id')['AD'].mean()
        win_beta = win_reg.set_index('window_id')['beta']
        common_w = win_beta.dropna().index.intersection(win_ad.dropna().index)
        aligned_win_beta = win_beta.loc[common_w]
        aligned_win_ad = win_ad.loc[common_w]
        depth_res = estimate_depth_impact_relation(aligned_win_beta, aligned_win_ad)

        eq_depth = (
            f"log(β) = {np.log(depth_res['c_hat']):.3f} - {depth_res['lambda_hat']:.3f}·log(AD) => β = {depth_res['c_hat']:.4f} / AD^{depth_res['lambda_hat']:.3f}"
            if depth_res['c_hat'] > 0 else "N/A"
        )

        table3_entry = {
            'Ticker': sym,
            'Equation_Depth_Scaling': eq_depth,
            'c_hat': depth_res['c_hat'],
            'lambda_hat': depth_res['lambda_hat'],
            't_c': depth_res['t_c'],
            't_lambda_1': depth_res['t_lambda_1'],
            'p_lambda_eq_1': depth_res['p_lambda_eq_1'],
            'ci_c_lower': depth_res['ci_c_lower'],
            'ci_c_upper': depth_res['ci_c_upper'],
            'ci_lambda_lower': depth_res['ci_lambda_lower'],
            'ci_lambda_upper': depth_res['ci_lambda_upper'],
            'R2_log': depth_res['r2_log_reg5'],
            'corr_beta_pred_sq': depth_res['corr_beta_pred_sq'],
            'n_dropped_nonpos_beta': depth_res['n_dropped_nonpositive_beta']
        }
        all_table3_rows.append(table3_entry)

        # Step 5: Intraday profile for this stock
        intra_df = compute_intraday_profiles(cleaned_df, time_bin='30min')
        intraday_profiles_list.append(intra_df)

        # Step 7: Predictive test (Lags 0 to 6)
        lag_res_df, sanity = run_all_lags_test(
            cleaned_df, lags=args.lags, train_frac=args.train_frac
        )

        r2_lag0 = lag_res_df.loc[lag_res_df['lag'] == 0, 'r2_in_sample'].values[0]
        r2_lag1 = lag_res_df.loc[lag_res_df['lag'] == 1, 'r2_oos'].values[0]
        t_lag1 = lag_res_df.loc[lag_res_df['lag'] == 1, 't_beta_white'].values[0]
        corr_lag1 = lag_res_df.loc[lag_res_df['lag'] == 1, 'corr_oos'].values[0]

        all_orderbook_metrics.append({
            'symbol': sym,
            'Equation_Lag0': f"ΔP_t = {lag_res_df.loc[lag_res_df['lag'] == 0, 'alpha_hat'].values[0]:.4f} + ({lag_res_df.loc[lag_res_df['lag'] == 0, 'beta_hat'].values[0]:.5f})·OFI_t",
            'Equation_Lag1': f"ΔP_t = {lag_res_df.loc[lag_res_df['lag'] == 1, 'alpha_hat'].values[0]:.4f} + ({lag_res_df.loc[lag_res_df['lag'] == 1, 'beta_hat'].values[0]:.5f})·OFI_{{t-1}}",
            'r2_lag0': r2_lag0,
            'r2_lag1': r2_lag1,
            't_stat_lag1': t_lag1,
            'corr_lag1': corr_lag1,
            'rho_ofi': sanity['rho_ofi_lag1'],
            'r2_lag1_theoretical': sanity['r2_lag1_theoretical']
        })

        # Step 8: Tradability check on test set for Lag 1
        tr_df, te_df = chronological_train_test_split(cleaned_df, train_frac=args.train_frac)
        m1 = compute_lag_regression_metrics(tr_df, te_df, lag=1)
        if 'y_test_pred' in m1:
            trade_res = evaluate_trading_signal(
                m1['y_test_actual'], m1['y_test_pred'],
                costs_in_ticks=(0.5, args.cost, 2.0)
            )
            c_def = trade_res['cost_scenarios'][f'cost_{args.cost}_ticks']
            all_tradability_rows.append({
                'Ticker': sym,
                'Test_N': trade_res['n_intervals'],
                'Trades': trade_res['n_trades'],
                'Turnover': trade_res['turnover_trades_per_interval'],
                'Hit_Rate': trade_res['hit_rate'],
                'Gross_Ticks': trade_res['gross_total_pnl_ticks'],
                'Gross_Ticks_Per_Trade': trade_res['gross_avg_pnl_per_trade_ticks'],
                'Break_Even_Ticks': trade_res['break_even_cost_ticks'],
                f'Net_Ticks_{args.cost}_cost': c_def['net_total_pnl_ticks'],
                'Survives_Cost': c_def['survives_costs']
            })

        # Keep BTC as representative stock for detailed plots
        if sym == 'BTC' or not rep_stock_data:
            df_with_exp = compute_expected_price(cleaned_df, tick_size=tick_sz)
            rep_stock_data = {
                'symbol': sym,
                'cleaned_df': df_with_exp,
                'win_ad': aligned_win_ad.values,
                'win_beta': aligned_win_beta.values,
                'depth_res': depth_res,
                'lag_metrics': m1,
                'lag_0_metrics': compute_lag_regression_metrics(tr_df, te_df, lag=0),
                'lin_full': lin_full,
                'lin_micro': lin_micro,
                'intraday_df': intra_df
            }

    # Save Tables
    table2_df = pd.DataFrame(all_table2_rows)
    table2_df.to_csv(tables_dir / "table_2_price_impact.csv", index=False)

    table3_df = pd.DataFrame(all_table3_rows)
    table3_df.to_csv(tables_dir / "table_3_depth_impact.csv", index=False)

    orderbooks_df = pd.DataFrame(all_orderbook_metrics)
    orderbooks_df.to_csv(tables_dir / "table_orderbooks_lag0_lag1.csv", index=False)

    tradability_df = pd.DataFrame(all_tradability_rows)
    tradability_df.to_csv(tables_dir / "table_tradability_test_set.csv", index=False)

    table_micro_df = pd.DataFrame(all_microprice_rows)
    table_micro_df.to_csv(tables_dir / "table_microprice_comparison.csv", index=False)

    # Step 6: Trade volume scaling and simulation
    vol_scale_rows = []
    for f in files:
        sym = f.stem.replace("_sample_7d", "")
        df_c = clean_orderbook_data(load_raw_dataset(f), tick_size=default_ticks.get(sym, 0.0001))
        df_c['delta_P'] = df_c['close_price'].diff() / default_ticks.get(sym, 0.0001)
        h_res = estimate_volume_exponent(df_c)
        if h_res.get('status') == 'completed':
            vol_scale_rows.append({
                'Ticker': sym,
                'H_hat': h_res['H_hat'],
                'se_H': h_res['se_H'],
                't_H': h_res['t_H'],
                'R2': h_res['r2']
            })
    table_vol_df = pd.DataFrame(vol_scale_rows)
    table_vol_df.to_csv(tables_dir / "table_volume_scaling_H.csv", index=False)

    # Square-root simulation
    sim_df = simulate_square_root_scaling()
    sim_df.to_csv(tables_dir / "table_squareroot_scaling_simulation.csv", index=False)

    print("Tables saved successfully.")

    # Save Plots
    rep = rep_stock_data
    sym_rep = rep['symbol']
    df_rep = rep['cleaned_df']

    # Figure 1: ACF
    dp_acf, dp_conf = compute_acf_series(df_rep['delta_P'], nlags=50)
    ofi_acf, ofi_conf = compute_acf_series(df_rep['OFI'], nlags=50)
    plot_figure_1_acf(dp_acf, dp_conf, ofi_acf, ofi_conf, sym_rep, figures_dir / "figure_1_acf.png")

    # Figure 2: Impact Scatter
    valid_mask = ~(df_rep['OFI'].isna() | df_rep['delta_P'].isna())
    plot_figure_2_impact_scatter(
        df_rep.loc[valid_mask, 'OFI'].values,
        df_rep.loc[valid_mask, 'delta_P'].values,
        rep['lin_full']['alpha'],
        rep['lin_full']['beta'],
        rep['lin_full']['r2'],
        sym_rep,
        window_label='Full 7-Day Sample',
        output_path=figures_dir / "figure_2_impact_scatter.png"
    )

    # Figure 2 Extension: Price Impact Mid-Price vs Microprice on the SAME GRAPH
    valid_both = ~(df_rep['OFI'].isna() | df_rep['delta_P'].isna() | df_rep['delta_P_micro'].isna())
    plot_figure_2_impact_scatter_with_microprice(
        df_rep.loc[valid_both, 'OFI'].values,
        df_rep.loc[valid_both, 'delta_P'].values,
        df_rep.loc[valid_both, 'delta_P_micro'].values,
        rep['lin_full']['alpha'],
        rep['lin_full']['beta'],
        rep['lin_full']['r2'],
        rep['lin_micro']['alpha'],
        rep['lin_micro']['beta'],
        rep['lin_micro']['r2'],
        sym_rep,
        window_label='Full 7-Day Sample',
        output_path=figures_dir / "figure_2_impact_scatter_with_microprice.png"
    )

    # Figure: Expected Price Trajectory on the SAME GRAPH
    plot_expected_price_trajectory(
        df_rep,
        sym_rep,
        n_bars=120,
        output_path=figures_dir / "figure_expected_price_trajectory.png"
    )

    # Figure: All Order Books Mid-Price vs Microprice Impact Comparison
    plot_microprice_impact_all_orderbooks(
        table_micro_df,
        figures_dir / "figure_microprice_impact_all_orderbooks.png"
    )

    # Figure 3: Kurtosis
    plot_figure_3_excess_kurtosis(np.array(all_kurtosis), figures_dir / "figure_3_excess_kurtosis.png")

    # Figure 4: Depth vs Impact log-log
    plot_figure_4_depth_impact_loglog(
        rep['win_ad'], rep['win_beta'],
        rep['depth_res']['lambda_hat'],
        np.log(rep['depth_res']['c_hat']) if rep['depth_res']['c_hat'] > 0 else 0.0,
        rep['depth_res']['r2_log_reg5'],
        sym_rep,
        output_path=figures_dir / "figure_4_depth_loglog.png"
    )

    # Figure 5 & 6: Intraday
    avg_intraday = pd.concat(intraday_profiles_list).groupby('time_of_day').mean().reset_index()
    plot_figure_5_intraday_patterns(avg_intraday, figures_dir / "figure_5_intraday_patterns.png")
    plot_figure_6_variance_identity(avg_intraday, figures_dir / "figure_6_variance_identity.png")

    # Figure 7: Prediction OOS
    m0 = rep['lag_0_metrics']
    m1 = rep['lag_metrics']
    plot_figure_7_prediction_oos(
        m0['y_test_actual'], m0['y_test_pred'], m0['r2_oos'],
        m1['y_test_actual'], m1['y_test_pred'], m1['r2_oos'],
        sym_rep,
        output_path=figures_dir / "figure_7_prediction_oos.png"
    )

    # Figure: Lag 0 vs Lag 1 Regression Side-by-Side
    plot_lag0_vs_lag1_regression_side_by_side(
        df_rep,
        sym_rep,
        train_frac=args.train_frac,
        output_path=figures_dir / "figure_lag0_vs_lag1_regression_side_by_side.png"
    )

    # Figure 8: All Order Books Lag 0 vs Lag 1
    plot_all_orderbooks_lag0_lag1(orderbooks_df, figures_dir / "figure_orderbooks_lag0_lag1.png")

    print("All figures saved successfully to output/figures/.")
    print("Execution complete.")


if __name__ == "__main__":
    main()
