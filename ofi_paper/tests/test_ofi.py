"""
Unit tests for OFI implementation, stylized model, and regression recovery.
"""
import pytest
import numpy as np
import pandas as pd
from ofi_paper.features.ofi import (
    compute_single_event_ofi,
    compute_bid_ask_contributions,
    stylized_model_bid_move,
    compute_ofi_series
)
from ofi_paper.models.impact_regression import fit_ols_white


def test_ofi_event_a():
    """
    Test a: Previous book bid 100.00 x 500, ask 100.01 x 300.
    Bid size 500 -> 700 at the same price gives e = +200.
    """
    e = compute_single_event_ofi(
        p_b_prev=100.00, q_b_prev=500,
        p_a_prev=100.01, q_a_prev=300,
        p_b_curr=100.00, q_b_curr=700,
        p_a_curr=100.01, q_a_curr=300
    )
    assert e == 200.0


def test_ofi_event_b():
    """
    Test b: Previous book bid 100.00 x 500, ask 100.01 x 300.
    Bid price drops to 99.99 with size 400 gives e = -500.
    """
    e = compute_single_event_ofi(
        p_b_prev=100.00, q_b_prev=500,
        p_a_prev=100.01, q_a_prev=300,
        p_b_curr=99.99, q_b_curr=400,
        p_a_curr=100.01, q_a_curr=300
    )
    assert e == -500.0


def test_ofi_event_c():
    """
    Test c: Previous book bid 100.00 x 500, ask 100.02 x 300.
    Bid price rises to 100.01 with size 150 gives e = +150 (ask unchanged at 100.02 x 300).
    """
    e_bid, e_ask = compute_bid_ask_contributions(
        p_b_prev=100.00, q_b_prev=500,
        p_a_prev=100.02, q_a_prev=300,
        p_b_curr=100.01, q_b_curr=150,
        p_a_curr=100.02, q_a_curr=300
    )
    assert e_bid == 150.0
    assert e_ask == 0.0
    assert (e_bid + e_ask) == 150.0


def test_ofi_event_d():
    """
    Test d: Previous book bid 100.00 x 500, ask 100.01 x 300.
    Ask size 300 -> 100 at the same price gives e = +200.
    """
    e = compute_single_event_ofi(
        p_b_prev=100.00, q_b_prev=500,
        p_a_prev=100.01, q_a_prev=300,
        p_b_curr=100.00, q_b_curr=500,
        p_a_curr=100.01, q_a_curr=100
    )
    assert e == 200.0


def test_stylized_model():
    """
    Stylized model tests:
    Depth D = 5:
    1. Market sells of 15 shares give bid move of -3 ticks.
    2. Adding 7 limit buys gives -1 tick.
    3. Adding 4 cancels as well gives -2 ticks (ceiling rounding).
    """
    # 1. Market sells of 15 shares
    m1 = stylized_model_bid_move(depth=5, market_sells=15, limit_buys=0, cancels=0)
    assert m1 == -3

    # 2. Market sells 15 + limit buys 7 => net = 7 - 15 = -8; ceil(-8/5) = -1
    m2 = stylized_model_bid_move(depth=5, market_sells=15, limit_buys=7, cancels=0)
    assert m2 == -1

    # 3. Market sells 15 + limit buys 7 + cancels 4 => net = 7 - 4 - 15 = -12; ceil(-12/5) = -2
    m3 = stylized_model_bid_move(depth=5, market_sells=15, limit_buys=7, cancels=4)
    assert m3 == -2


def test_synthetic_regression_recovery():
    """
    Synthetic-data test where true beta is known and OLS with White HC0 recovers it.
    """
    np.random.seed(42)
    n = 1000
    true_alpha = 0.05
    true_beta = 0.0025

    # Simulated OFI with heavy tails
    ofi = np.random.standard_t(df=5, size=n) * 100.0

    # Heteroscedastic noise
    noise_sigma = 0.2 + 0.001 * np.abs(ofi)
    noise = np.random.normal(0, noise_sigma, size=n)

    y = true_alpha + true_beta * ofi + noise

    res = fit_ols_white(y, ofi.reshape(-1, 1), add_constant=True)

    # Check recovery within 2 standard errors
    assert np.isclose(res['beta'], true_beta, atol=0.0003)
    assert np.isclose(res['alpha'], true_alpha, atol=0.03)
    assert res['t_beta'] > 10.0
    assert res['p_beta'] < 0.001
    assert res['r2'] > 0.30
