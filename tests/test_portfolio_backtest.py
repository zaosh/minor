"""Downstream vol-targeting backtest: sizing, costs, timing, alignment and leakage."""

import numpy as np
import pandas as pd
import pytest

from src.models import portfolio_backtest as pb


def test_exposure_targets_and_caps():
    e = pb.exposure(np.array([0.05, 0.10, 0.20, 0.40]), target=0.10, cap=1.0)
    assert e == pytest.approx([1.0, 1.0, 0.5, 0.25])
    assert pb.exposure(np.array([0.0]), 0.10, 1.0)[0] == 1.0  # floor, no division by zero


def test_simulate_hand_values_with_drift_cost():
    R = np.array([[0.01, 0.02, -0.01, 0.03, 0.04, 0.05]])
    grid, expo = np.array([1, 3]), np.array([[0.5, 1.0]])
    s, days, held = pb.simulate(R, grid, expo, rf_daily=0.0, cost_bps=100)
    assert list(days) == [2, 3, 4, 5]
    assert held[0] == pytest.approx([0.5, 0.5, 1.0, 1.0])
    # period 1: exposure .5, built from cash (cost .5 * 1%); wealth path .5 * A + .5
    w2, w3 = 0.5 * 0.99 + 0.5, 0.5 * 0.99 * 1.03 + 0.5
    assert s[0, 0] == pytest.approx(w2 * 0.995 - 1)
    assert s[0, 1] == pytest.approx(w3 / w2 - 1)
    # the position drifted to .5 * A / W before the rebalance to 1.0, cost 1% of the change
    drift = 0.5 * 0.99 * 1.03 / w3
    c2 = abs(1.0 - drift) * 0.01
    assert s[0, 2] == pytest.approx(1.04 * (1 - c2) - 1)
    assert s[0, 3] == pytest.approx(0.05)


def test_simulate_end_truncates_the_last_holding_period():
    R = np.random.default_rng(2).normal(0, 0.01, (2, 40))
    s, days, _ = pb.simulate(R, np.array([5, 10]), np.ones((2, 2)), 0.0, end=15)
    assert days[0] == 6 and days[-1] == 15 and s.shape == (2, 10)


def test_simulate_uses_only_past_exposure_choice():
    """Changing the exposure chosen at grid day g must not change returns on or before day g."""
    rng = np.random.default_rng(0)
    R = rng.normal(0, 0.01, (3, 40))
    grid = np.array([5, 10, 15, 20])
    E1 = rng.uniform(0.2, 1, (3, 4))
    E2 = E1.copy()
    E2[:, 2] = 0.1  # decision at day 15
    s1, d, _ = pb.simulate(R, grid, E1, 0.0)
    s2, _, _ = pb.simulate(R, grid, E2, 0.0)
    assert np.array_equal(s1[:, d <= 15], s2[:, d <= 15])
    assert not np.allclose(s1[:, d > 15], s2[:, d > 15])


def test_zero_cost_full_exposure_equals_portfolio():
    R = np.random.default_rng(1).normal(0, 0.01, (2, 30))
    s, d, _ = pb.simulate(R, np.array([4, 9]), np.ones((2, 2)), 0.0, cost_bps=0)
    assert s == pytest.approx(R[:, d])


def test_constant_exposure_without_cost_is_buy_and_hold_not_daily_rebalanced():
    R = np.array([[0.10, 0.10, 0.10, 0.10]])
    s, _, _ = pb.simulate(R, np.array([0]), np.array([[0.5]]), 0.0, cost_bps=0, end=3)
    # wealth .5 * 1.1^k + .5; a daily-rebalanced 50% position would return a flat 5% each day
    assert s[0] == pytest.approx([0.05, 1.105 / 1.05 - 1, 1.1655 / 1.105 - 1])


def test_segment_metrics_known_series():
    strat = np.full((1, 252), 0.001)
    held = np.ones((1, 252))
    m = pb.segment_metrics(strat, held, rf_daily=0.0, target=0.10)
    assert m["ann_return"][0] == pytest.approx(0.252)
    assert m["max_dd"][0] == 0.0 and m["turnover"][0] == 0.0
    down = pb.segment_metrics(np.array([[0.1, -0.5, 0.1]]), np.ones((1, 3)), 0.0)
    assert down["max_dd"][0] == pytest.approx(-0.5)


def test_block_vols_and_tracking_rmse_with_mask():
    rng = np.random.default_rng(3)
    x = rng.normal(0, 0.01, (2, 200))
    bv = pb.block_vols(x, block=63)
    assert bv.shape == (2, 3)  # the partial 4th block is dropped
    assert bv[0, 0] == pytest.approx(x[0, :63].std(ddof=1) * np.sqrt(252))
    mask = np.array([[True, False, True], [False, False, False]])
    r = pb.tracking_rmse(bv, 0.10, mask)
    assert r[0] == pytest.approx(np.sqrt(((bv[0, [0, 2]] - 0.10) ** 2).mean()))
    assert np.isnan(r[1])


# ---- end to end through run() on a small synthetic world ---------------------------------
@pytest.fixture
def world(monkeypatch):
    T, P = 700, 4
    idx = pd.bdate_range("2019-01-01", periods=T)
    rng = np.random.default_rng(7)
    R = rng.normal(0.0003, 0.012, (P, T))
    monkeypatch.setattr(pb, "load_inputs", lambda: (R, idx, [{}] * P))
    dates = idx[300:600:5]
    truth = np.full((len(dates), P), 0.2)

    def frame(model, vol):
        rows = []
        for p in range(P):
            rows.append(pd.DataFrame({"date": dates, "pid": p, "model": model, "target": "vol",
                                      "y_true": truth[:, p], "y_pred": vol[:, p]}))
        return pd.concat(rows)

    base = np.full((len(dates), P), 0.2)
    skew = base.copy()
    skew[:, 0] = 0.8  # pid 0 forecast to be very volatile -> small exposure
    preds = pd.concat([frame("naive", base), frame("gjr", base), frame("mdl", skew)])
    return R, idx, dates, preds


def test_run_aligns_pids_and_oracle_excluded_from_tests(world):
    _, _, dates, preds = world
    res = pb.run(preds)
    d = res["by_pid_fold"]
    m = d[(d["model"] == "mdl")].groupby("pid")["mean_exposure"].mean()
    assert m[0] == pytest.approx(0.1 / 0.8, abs=0.01) and m[1] == pytest.approx(0.5, abs=0.01)
    assert "oracle" not in res["cube"] and "unscaled" in res["cube"]
    t = pb.dm_tests(pb.loss_frame(res["cube"], res["dates"]))
    assert set(t["model"]) == {"mdl", "gjr"}  # gjr vs naive; unscaled and oracle never tested
    assert (t["p_holm"] >= t["p"] - 1e-12).all()


def test_run_forecast_at_t_cannot_change_earlier_strategy_returns(world, monkeypatch):
    R, idx, dates, preds = world
    a = pb.run(preds)["by_pid_fold"]
    late = preds.copy()
    late.loc[(late["model"] == "mdl") & (late["date"] == dates[-1]), "y_pred"] = 0.05
    b = pb.run(late)["by_pid_fold"]
    ma, mb = (x[x["model"] == "mdl"] for x in (a, b))
    first = ma["fold"].iloc[0]
    # only the final holding period (one stride) can differ; the first fold is untouched
    assert ma[ma["fold"] == first].drop(columns="model").equals(
        mb[mb["fold"] == first].drop(columns="model"))


def test_run_last_exposure_is_held_one_stride_only(world):
    R, idx, dates, preds = world
    d = pb.run(preds)["by_pid_fold"]
    n = d[d["model"] == "naive"].groupby("fold")["n_days"].first().sum()
    g = idx.get_indexer(dates)
    assert n == (g[-1] + (g[-1] - g[-2])) - g[0]


def test_dm_tests_single_holm_family_and_no_reference_rows():
    rng = np.random.default_rng(4)
    L = pd.DataFrame(rng.uniform(0, 1, (200, 5)), columns=["naive", "gjr", "a", "b", "unscaled"])
    t = pb.dm_tests(L)
    assert set(t["model"]) == {"a", "b", "gjr"} and "unscaled" not in set(t["model"])
    assert len(t) == 3 + 2  # gjr and a, b vs naive; a, b vs gjr
