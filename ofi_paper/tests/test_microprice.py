"""
Unit tests for microprice and expected price calculation.
"""
import pytest
import numpy as np
import pandas as pd

from ofi_paper.features.microprice import (
    compute_microprice,
    compute_multilevel_microprice,
    compute_microprice_change,
    compute_expected_price,
    compare_price_impact_mid_vs_micro
)


def test_microprice_level1_symmetry():
    """When bid and ask volumes are identical, microprice equals mid-price."""
    df = pd.DataFrame({
        'bid_px_1': [100.0, 200.0],
        'ask_px_1': [102.0, 204.0],
        'bid_volume_level_1': [50.0, 100.0],
        'ask_volume_level_1': [50.0, 100.0]
    })
    micro = compute_microprice(df)
    mid = (df['bid_px_1'] + df['ask_px_1']) / 2.0
    np.testing.assert_allclose(micro.values, mid.values)


def test_microprice_skew():
    """When bid volume is larger, microprice is higher (closer to ask)."""
    df = pd.DataFrame({
        'bid_px_1': [100.0],
        'ask_px_1': [102.0],
        'bid_volume_level_1': [90.0],
        'ask_volume_level_1': [10.0]
    })
    micro = compute_microprice(df).iloc[0]
    # Expected: (90 * 102 + 10 * 100) / 100 = (9180 + 1000) / 100 = 101.8
    assert micro == pytest.approx(101.8, rel=1e-5)
    assert micro > 101.0  # Above mid-price (101.0)


def test_microprice_multilevel():
    """Multi-level microprice with 2 levels."""
    df = pd.DataFrame({
        'close_price': [100.0],
        'bid_volume_level_1': [10.0],
        'ask_volume_level_1': [10.0],
        'bid_distance_level_1': [10.0],  # 10 bps => 100 * (1 - 0.0010) = 99.90
        'ask_distance_level_1': [10.0],  # 10 bps => 100 * (1 + 0.0010) = 100.10
        'bid_volume_level_2': [20.0],
        'ask_volume_level_2': [20.0],
        'bid_distance_level_2': [20.0],
        'ask_distance_level_2': [20.0],
    })
    micro_ml = compute_multilevel_microprice(df, levels=2)
    assert micro_ml.iloc[0] == pytest.approx(100.0, rel=1e-5)


def test_microprice_change_and_gap_handling():
    """Gaps should mask out microprice change as NaN."""
    df = pd.DataFrame({
        'microprice': [100.0, 101.0, 103.0],
        'is_valid_interval': [False, True, False]
    })
    delta_m, delta_p = compute_microprice_change(df, tick_size=0.5)
    assert np.isnan(delta_p.iloc[0])
    assert delta_p.iloc[1] == pytest.approx(2.0)  # (101 - 100) / 0.5 = 2.0
    assert np.isnan(delta_p.iloc[2])  # Interval is invalid


def test_expected_price_structure():
    """Verify expected price computation generates expected columns."""
    df = pd.DataFrame({
        'timestamp_utc': pd.date_range('2026-01-01', periods=50, freq='5min'),
        'close_price': np.linspace(100, 105, 50),
        'bid_px_1': np.linspace(99.9, 104.9, 50),
        'ask_px_1': np.linspace(100.1, 105.1, 50),
        'bid_volume_level_1': np.random.uniform(5, 20, 50),
        'ask_volume_level_1': np.random.uniform(5, 20, 50),
        'is_valid_interval': [True] * 50,
        'OFI': np.random.normal(0, 10, 50)
    })
    df['mid_price'] = (df['bid_px_1'] + df['ask_px_1']) / 2.0
    df['delta_P'] = df['mid_price'].diff() / 0.1
    df.loc[df.index[0], 'delta_P'] = np.nan

    res = compute_expected_price(df, tick_size=0.1)
    assert 'microprice' in res.columns
    assert 'expected_price_mid_ofi' in res.columns
    assert 'expected_price_micro_ofi' in res.columns
    assert 'expected_adj_currency' in res.columns
