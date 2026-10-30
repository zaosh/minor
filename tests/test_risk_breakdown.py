import numpy as np
import pandas as pd
import pytest

from src import config, risk_breakdown
from src.models import baselines
from src.portfolios import Portfolio, portfolio_returns


@pytest.fixture(scope="module")
def rets():
    idx = pd.bdate_range("2018-01-01", periods=700)
    rng = np.random.default_rng(3)
    common = rng.normal(0, 0.008, (700, 1))
    r = pd.DataFrame(common + rng.normal(0, 0.01, (700, 4)) * np.array([1, 2, 1, 0.5]),
                     index=idx, columns=list("ABCD"))
    return r


P = Portfolio(("A", "B", "C"), (0.5, 0.3, 0.2))


def test_vol_contrib_sums_to_the_lw_vol_baseline(rets):
    t = 600
    c = risk_breakdown.vol_contributions(rets, P, rets.index[t])
    lw = baselines.lw_vol(rets[list(P.tickers)].to_numpy(), np.array(P.weights), np.array([t]))[0]
    assert c["vol_contribution"].sum() == pytest.approx(lw)
    assert c["share"].sum() == pytest.approx(1.0)


def test_marginal_vol_matches_a_finite_difference(rets):
    from sklearn.covariance import ledoit_wolf

    t = 600
    c = risk_breakdown.vol_contributions(rets, P, rets.index[t])
    cov = ledoit_wolf(rets[list(P.tickers)].iloc[t - 249 : t + 1].to_numpy())[0] * 252
    w = np.array(P.weights)
    sig = lambda x: np.sqrt(x @ cov @ x)  # noqa: E731
    eps = 1e-6
    fd = [(sig(w + eps * np.eye(3)[i]) - sig(w - eps * np.eye(3)[i])) / (2 * eps)
          for i in range(3)]
    np.testing.assert_allclose(c["marginal_vol"], fd, rtol=1e-5)


def test_es_contrib_sums_to_historical_es(rets):
    t = 600
    c = risk_breakdown.es_contributions(rets, P, rets.index[t])
    r = portfolio_returns(rets, P).to_numpy()
    assert c["es_contribution"].sum() == pytest.approx(baselines.hist_es_at(r, np.array([t]))[0])
    assert c["n_tail_days"].iloc[0] == 13  # 250 days, alpha 5%, quantile inclusive


def test_breakdown_ignores_data_after_as_of(rets):
    t = 500
    other = rets.copy()
    other.iloc[t + 1 :] = np.random.default_rng(9).normal(0, 0.1, other.iloc[t + 1 :].shape)
    for fn in (risk_breakdown.vol_contributions, risk_breakdown.es_contributions):
        pd.testing.assert_frame_equal(fn(rets, P, rets.index[t]), fn(other, P, rets.index[t]))


def test_breakdown_needs_enough_history(rets):
    with pytest.raises(ValueError, match="250 days"):
        risk_breakdown.vol_contributions(rets, P, rets.index[100])


def test_stress_replay_matches_portfolio_returns(rets):
    market = (1 + rets["D"]).cumprod()
    start, end = str(rets.index[100].date()), str(rets.index[160].date())
    out = risk_breakdown.stress_replay(rets, market, P, start, end)
    r = portfolio_returns(rets, P).loc[start:end].iloc[1:]
    assert out["total_return"] == pytest.approx((1 + r).prod() - 1)
    assert out["max_drawdown"] <= 0 and out["worst_day"] == pytest.approx(r.min())
    assert out["wealth"].iloc[0] == 1.0 and out["n_days"] == len(r)
    assert out["market_return"] == pytest.approx(market[end] / market[start] - 1)
    assert out["contributions"].sum() == pytest.approx(r.sum())


def test_scenarios_are_inside_the_cached_data_and_are_peak_to_trough():
    path = config.DATA_PROCESSED / "returns.parquet"
    if not path.exists():
        pytest.skip("needs cached data")
    idx = pd.read_parquet(path).index
    mkt = pd.read_parquet(config.DATA_PROCESSED / "market.parquet")[config.MARKET]
    for name, (a, b) in config.STRESS_SCENARIOS.items():
        assert idx.min() <= pd.Timestamp(a) < pd.Timestamp(b) <= idx.max(), name
        assert pd.Timestamp(a) in idx and pd.Timestamp(b) in idx, name
        assert mkt[b] < mkt[a], name  # a drawdown window, not a rally
