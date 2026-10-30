"""Train/serve parity on synthetic data, so it also runs on a fresh clone (no cached data)."""

import numpy as np
import pandas as pd
import pytest

from src import config, predict
from src.features import dataset, engineer
from src.portfolios import Portfolio


@pytest.fixture
def synthetic(monkeypatch):
    T, N = 600, 6
    idx = pd.bdate_range("2018-01-01", periods=T)
    rng = np.random.default_rng(11)
    returns = pd.DataFrame(rng.normal(0.0004, 0.012, (T, N)), index=idx,
                           columns=[f"S{i}" for i in range(N)])
    mkt_px = 100 * np.cumprod(1 + rng.normal(0.0003, 0.01, T))
    market = pd.DataFrame({config.MARKET: mkt_px}, index=idx)
    vix = pd.Series(15 + np.cumsum(rng.normal(0, 0.2, T)), index=idx)
    macro = pd.DataFrame({"VIX": vix, "VIX_chg": np.log(vix).diff()})
    data = {"returns": returns, "market": market, "macro": macro}
    monkeypatch.setattr(predict, "load_data", lambda: data)
    return data


def test_serving_features_match_training_rows(synthetic):
    returns, market, macro = (synthetic[k] for k in ("returns", "market", "macro"))
    p = Portfolio(("S1", "S3", "S4"), (0.5, 0.3, 0.2))
    mkt_ret = market[config.MARKET].pct_change().fillna(0.0).rename("mkt_ret")
    pos = dataset.grid_positions(len(returns))
    frame = dataset.portfolio_frame(0, p, returns, mkt_ret, macro,
                                    engineer.pair_corr_cache(returns), pos).set_index("date")
    date = frame.index[40]
    row = predict.portfolio_features(p, date)
    cols = [c for c in row.columns if c in frame.columns and c != "day_idx"]
    assert {"vol_21", "mean_pair_corr_63", "naive_vol", "lw_vol", "hist_var250"} <= set(cols)
    np.testing.assert_allclose(row[cols].iloc[0].to_numpy(float),
                               frame.loc[date, cols].to_numpy(float), rtol=1e-4, atol=1e-5)


def test_serving_model_features_match_the_batch_features(synthetic):
    from src.features import model_features
    from src.portfolios import portfolio_returns

    returns, market, macro = (synthetic[k] for k in ("returns", "market", "macro"))
    p = Portfolio(("S1", "S3", "S4"), (0.5, 0.3, 0.2))
    mkt_ret = market[config.MARKET].pct_change().fillna(0.0).rename("mkt_ret")
    pos = dataset.grid_positions(len(returns))
    frame = dataset.portfolio_frame(0, p, returns, mkt_ret, macro,
                                    engineer.pair_corr_cache(returns), pos)
    panel = frame.set_index(["date", "pid"])
    panel["day_idx"] = returns.index.get_indexer(panel.index.get_level_values("date"))
    r = portfolio_returns(returns, p)
    feats, _ = model_features.garch_panel_features(r.to_numpy()[None, :], panel, returns.index,
                                                   workers=1)
    full = model_features.add_model_features(panel, returns.index, feats)
    date = panel.index.get_level_values("date")[-5]
    year = model_features.refresh_year(date.year)
    dyear = returns.index[panel["day_idx"].to_numpy().astype(int)].year.to_numpy()
    fit_end = int(panel["day_idx"].to_numpy()[dyear == year].min())
    coef = model_features.har_coefficients(panel, returns.index)[year]
    row = predict.portfolio_features(p, date)
    got = predict.serving_features(r, int(returns.index.get_loc(date)), row, coef, fit_end, date)
    want = full.xs(date, level="date").iloc[0]
    for k in ("har_vol", "garch_vol21", "garch_sigma1", "garch_fallback"):
        assert got[k] == pytest.approx(float(want[k]), rel=1e-6), k
    assert got["sim_dd_raw"] == pytest.approx(float(want["sim_dd_raw"]), abs=2.5e-3)
    assert fit_end <= int(returns.index.get_loc(date))  # the fit never reaches the forecast day


def test_combine_models_matches_the_frame_combination():
    from src.models import combos

    m = {"har": {"vol_hat": 0.2, "dd_hat": 0.3, "var_hat": -0.02, "es_hat": -0.03},
         "garch": {"vol_hat": 0.1, "dd_hat": 0.2, "var_hat": -0.01, "es_hat": -0.02},
         "xgboost": {"vol_hat": 0.3, "dd_hat": 0.4, "var_hat": -0.03, "es_hat": float("nan")}}
    hg = combos.combine_models(m, ("har", "garch"))
    assert hg == pytest.approx({"vol_hat": 0.15, "dd_hat": 0.25, "var_hat": -0.015,
                                "es_hat": -0.025})
    hgx = combos.combine_models(m, ("har", "garch", "xgboost"))
    assert "es_hat" not in hgx and hgx["vol_hat"] == pytest.approx(0.2)
