"""Downstream vol-targeting backtest: sizing, costs, timing and leakage."""

import numpy as np
import pytest

from src.models import portfolio_backtest as pb


def test_exposure_targets_and_caps():
    e = pb.exposure(np.array([0.05, 0.10, 0.20, 0.40]), target=0.10, cap=1.0)
    assert e == pytest.approx([1.0, 1.0, 0.5, 0.25])
    assert pb.exposure(np.array([0.0]), 0.10, 1.0)[0] == 1.0  # floor, no division by zero


def test_simulate_timing_and_cost_hand_values():
    R = np.array([[0.01, 0.02, -0.01, 0.03, 0.04, 0.05]])
    grid, expo = np.array([1, 3]), np.array([[0.5, 1.0]])
    s, days = pb.simulate(R, grid, expo, rf_daily=0.0, cost_bps=100)
    assert list(days) == [2, 3, 4, 5]
    # exposure .5 on days 2,3 (cost 1% * .5 on day 2); exposure 1 on days 4,5 (cost 1% * .5)
    assert s[0] == pytest.approx([0.5 * -0.01 - 0.005, 0.5 * 0.03, 0.04 - 0.005, 0.05])


def test_simulate_uses_only_past_exposure_choice():
    """Changing the exposure chosen at grid day g must not change returns on or before day g."""
    rng = np.random.default_rng(0)
    R = rng.normal(0, 0.01, (3, 40))
    grid = np.array([5, 10, 15, 20])
    E1 = rng.uniform(0.2, 1, (3, 4))
    E2 = E1.copy()
    E2[:, 2] = 0.1  # decision at day 15
    s1, d = pb.simulate(R, grid, E1, 0.0)
    s2, _ = pb.simulate(R, grid, E2, 0.0)
    assert np.array_equal(s1[:, d <= 15], s2[:, d <= 15])
    assert not np.allclose(s1[:, d > 15], s2[:, d > 15])


def test_zero_cost_full_exposure_equals_portfolio():
    R = np.random.default_rng(1).normal(0, 0.01, (2, 30))
    s, d = pb.simulate(R, np.array([4, 9]), np.ones((2, 2)), 0.0, cost_bps=0)
    assert s == pytest.approx(R[:, d])


def test_segment_metrics_known_series():
    strat = np.full((1, 252), 0.001)
    held = np.ones((1, 252))
    m = pb.segment_metrics(strat, held, rf_daily=0.0, target=0.10)
    assert m["ann_return"][0] == pytest.approx(0.252)
    assert m["max_dd"][0] == 0.0 and m["turnover"][0] == 0.0
    down = pb.segment_metrics(np.array([[0.1, -0.5, 0.1]]), np.ones((1, 3)), 0.0)
    assert down["max_dd"][0] == pytest.approx(-0.5)
