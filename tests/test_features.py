import numpy as np
import pandas as pd
import pytest

from src import config
from src.features import dataset, engineer, targets
from src.models import baselines
from src.portfolios import Portfolio, build_user_portfolio, generate_portfolios, portfolio_returns

T, N = 700, 6
H = config.H


@pytest.fixture(scope="module")
def data():
    idx = pd.bdate_range("2015-01-01", periods=T)
    rng = np.random.default_rng(1)
    returns = pd.DataFrame(rng.normal(0.0004, 0.012, (T, N)), index=idx,
                           columns=[f"S{i}" for i in range(N)])
    mkt = pd.Series(rng.normal(0.0004, 0.01, T), index=idx, name="mkt_ret")
    macro = pd.DataFrame({"VIX": 15 + rng.normal(0, 1, T).cumsum() * 0.1}, index=idx)
    return returns, mkt, macro


def features_for(returns, mkt, macro, p):
    cache = engineer.pair_corr_cache(returns)
    members = [returns.columns.get_loc(t) for t in p.tickers]
    mc = {L: engineer.mean_pair_corr(c, members, returns.index) for L, c in cache.items()}
    return engineer.compute_features(portfolio_returns(returns, p), mkt, macro, mc,
                                     len(p.tickers), p.hhi)


# ---- targets: hand-worked values -------------------------------------------------------
def test_forward_vol_hand_example():
    r = pd.Series([0.01, 0.02, -0.01, 0.03, 0.0, 0.01])
    out = targets.forward_vol(r, h=3)
    expected = np.std([0.02, -0.01, 0.03], ddof=1) * np.sqrt(252)
    assert out.iloc[0] == pytest.approx(expected)
    assert out.iloc[-3:].isna().all()


def test_forward_dd_hand_example():
    r = pd.Series([0.5, -0.06, 0.02, 0.01, 0.01, 0.01, 0.01])
    flag = targets.forward_dd_flag(r, h=3, thresh=0.05)
    assert flag.iloc[0] == 1.0  # path 1, .94, .9588, .9684 -> mdd -6%
    assert flag.iloc[1] == 0.0  # path after the dip only rises
    assert flag.iloc[-3:].isna().all()


def test_next_return_and_logratio():
    r = pd.Series([0.1, 0.2, 0.3])
    assert targets.next_return(r).iloc[0] == 0.2
    lr = targets.log_ratio(pd.Series([0.2]), pd.Series([0.1]))
    assert lr.iloc[0] == pytest.approx(np.log(2))


# ---- targets: leakage -------------------------------------------------------------------
@pytest.mark.parametrize("fn", [targets.forward_vol, targets.forward_dd_flag, targets.next_return])
def test_target_uses_only_t_plus_1_to_t_plus_h(fn):
    rng = np.random.default_rng(3)
    r = pd.Series(rng.normal(0, 0.001, 200))  # calm, so the dd flag starts at 0
    t = 100
    base = fn(r).iloc[t]
    past = r.copy()
    past.iloc[: t + 1] = rng.normal(0, 0.05, t + 1)  # change everything up to and including t
    assert fn(past).iloc[t] == pytest.approx(base)
    far = r.copy()
    far.iloc[t + H + 1 :] += 0.1  # change beyond the horizon
    if fn is not targets.next_return:
        assert fn(far).iloc[t] == pytest.approx(base)
    near = r.copy()
    near.iloc[t + 1] = -0.2  # r_{t+1} is inside every target's window
    assert fn(near).iloc[t] != pytest.approx(base)


# ---- features: leakage ------------------------------------------------------------------
def test_features_and_baselines_do_not_change_when_future_changes(data):
    returns, mkt, macro = data
    p = Portfolio(("S0", "S2", "S3"), (0.5, 0.3, 0.2))
    t0 = 400
    before = features_for(returns, mkt, macro, p)
    r2, m2 = returns.copy(), mkt.copy()
    r2.iloc[t0 + 1 :] = np.random.default_rng(9).normal(0, 0.05, (T - t0 - 1, N))
    m2.iloc[t0 + 1 :] = np.random.default_rng(8).normal(0, 0.05, T - t0 - 1)
    after = features_for(r2, m2, macro, p)
    pd.testing.assert_frame_equal(before.iloc[: t0 + 1], after.iloc[: t0 + 1], check_exact=False)

    pr1, pr2 = portfolio_returns(returns, p), portfolio_returns(r2, p)
    for fn in (baselines.naive_vol, baselines.ewma_vol, baselines.hist_var, baselines.param_var):
        pd.testing.assert_series_equal(fn(pr1).iloc[: t0 + 1], fn(pr2).iloc[: t0 + 1])
    w, pos = np.array(p.weights), np.array([300, 400])
    a = baselines.lw_vol(returns[list(p.tickers)].to_numpy(), w, pos)
    b = baselines.lw_vol(r2[list(p.tickers)].to_numpy(), w, pos)
    np.testing.assert_allclose(a, b)


# ---- features: correctness --------------------------------------------------------------
def test_mean_pair_corr_matches_brute_force(data):
    returns, _, _ = data
    cache = engineer.pair_corr_cache(returns)
    members = [0, 2, 4]
    got = engineer.mean_pair_corr(cache[63], members, returns.index)
    t = 300
    win = returns.iloc[t - 62 : t + 1, members].corr().to_numpy()
    expected = (win.sum() - 3) / 6
    assert got.iloc[t] == pytest.approx(expected, abs=1e-5)


def test_rolling_max_drawdown_and_vol_hand_example():
    r = pd.Series([0.0, 0.1, -0.2, 0.05, 0.0])
    mdd = engineer.rolling_max_drawdown(r, 3)  # last 3 returns: wealth 1.1 -> .88 -> .924
    assert mdd.iloc[3] == pytest.approx(0.88 / 1.1 - 1)
    assert mdd.iloc[:3].isna().all()


def test_single_asset_pair_corr_is_one(data):
    returns, _, _ = data
    cache = engineer.pair_corr_cache(returns)
    assert (engineer.mean_pair_corr(cache[21], [1], returns.index) == 1.0).all()


def test_sharpe_matches_formula(data):
    returns, mkt, macro = data
    p = Portfolio(("S1", "S4"), (0.6, 0.4))
    f = features_for(returns, mkt, macro, p)
    r = portfolio_returns(returns, p)
    win = r.iloc[300 - 62 : 301] - config.RF_ANNUAL / 252
    assert f["sharpe_63"].iloc[300] == pytest.approx(
        win.mean() / win.std() * np.sqrt(252), rel=1e-9
    )


# ---- portfolios -------------------------------------------------------------------------
def test_generate_portfolios_properties():
    uni = [f"T{i}" for i in range(30)]
    ps = generate_portfolios(50, uni, seed=42)
    assert ps == generate_portfolios(50, uni, seed=42)
    for p in ps:
        assert 3 <= len(p.tickers) <= 15 and len(set(p.tickers)) == len(p.tickers)
        assert sum(p.weights) == pytest.approx(1.0)


def test_user_portfolio_validation():
    avail = ["A", "B", "C"]
    assert build_user_portfolio(["A", "B"], [0.5, 0.5], avail).tickers == ("A", "B")
    for bad in ([["A", "Z"], [0.5, 0.5]], [["A", "A"], [0.5, 0.5]], [["A", "B"], [0.7, 0.7]],
                [["A", "B"], [1.5, -0.5]], [["A"], [0.5, 0.5]],
                [["A", "B"], [float("nan"), 1.0]], [["A", "B"], [float("inf"), 0.0]]):
        with pytest.raises(ValueError):
            build_user_portfolio(*bad, avail)


def test_user_portfolio_drops_zero_weights():
    p = build_user_portfolio(["A", "B", "C"], [0.6, 0.0, 0.4], ["A", "B", "C"])
    assert p.tickers == ("A", "C") and p.weights == pytest.approx((0.6, 0.4))


# ---- panel ------------------------------------------------------------------------------
def test_portfolio_frame_shape_index_and_no_nan_targets(data):
    returns, mkt, macro = data
    p = Portfolio(("S0", "S1", "S5"), (0.4, 0.4, 0.2))
    pos = dataset.grid_positions(len(returns))
    df = dataset.portfolio_frame(
        7, p, returns, mkt, macro, engineer.pair_corr_cache(returns), pos
    )
    assert df.columns.is_unique
    assert len(df) == len(pos) and df["date"].is_unique
    assert df["pid"].eq(7).all()
    tcols = ["vol_fwd", "y_logratio", "dd_flag", "r_next", "lw_vol", "ewma_vol", "hist_var250"]
    assert not df[tcols].isna().any().any()
    assert not df.drop(columns=["date"]).isna().any().any()
    assert pos.max() + H < len(returns)


def test_dataset_workers_work_under_spawn(data):
    """Workers get their inputs through the Pool initializer, so spawn (macOS/Windows) works."""
    import multiprocessing as mp

    returns, mkt, macro = data
    ports = [Portfolio(("S0", "S1"), (0.5, 0.5)), Portfolio(("S2", "S3", "S4"), (0.2, 0.3, 0.5))]
    ctx = dict(returns=returns, mkt_ret=mkt, macro=macro, corr=engineer.pair_corr_cache(returns),
               pos=dataset.grid_positions(len(returns)))
    with mp.get_context("spawn").Pool(
        2, initializer=dataset._init_worker, initargs=(ctx,)
    ) as pool:
        frames = pool.map(dataset._worker, list(enumerate(ports)))
    for pid, (p, got) in enumerate(zip(ports, frames, strict=True)):
        want = dataset.portfolio_frame(
            pid, p, returns, mkt, macro, ctx["corr"], ctx["pos"]
        )
        pd.testing.assert_frame_equal(got, want)


def test_sortino_is_finite_when_no_return_is_below_the_risk_free_rate():
    idx = pd.bdate_range("2020-01-01", periods=200)
    r = pd.Series(0.01, index=idx)  # always above rf: zero downside deviation
    mkt = pd.Series(np.random.default_rng(0).normal(0, 0.01, 200), index=idx)
    corr = {L: pd.Series(0.5, index=idx) for L in (21, 63, 126)}
    f = engineer.compute_features(r, mkt, pd.DataFrame(index=idx), corr, 3, 0.4)
    assert np.isfinite(f[["sortino_21", "sortino_63", "sortino_126"]].dropna()).all().all()
