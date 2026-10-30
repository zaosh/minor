import json

import numpy as np
import pandas as pd
import pytest

from src import backtest, config, predict
from src.features import targets
from src.portfolios import Portfolio, portfolio_returns

FOLDS = [{"name": "2020", "test_start": "2020-01-03", "train_end": "2019-11-28",
          "first_test_day_idx": 10},
         {"name": "2021", "test_start": "2021-01-04", "train_end": "2020-11-26",
          "first_test_day_idx": 20}]

needs_artifacts = pytest.mark.skipif(
    not (config.ARTIFACTS / "folds" / "2022" / "baselines.json").exists()
    or not (config.REPORTS / "predictions.parquet").exists()
    or not (config.DATA_PROCESSED / "portfolios.json").exists(),
    reason="needs evaluation outputs and cached data (see README)",
)


# ---- pure logic ---------------------------------------------------------------------------
def test_fold_for_date():
    assert backtest.fold_for_date(pd.Timestamp("2020-06-01"), FOLDS)["name"] == "2020"
    assert backtest.fold_for_date(pd.Timestamp("2021-01-02"), FOLDS)["name"] == "2021"
    assert backtest.fold_for_date(pd.Timestamp("2030-01-01"), FOLDS)["name"] == "2021"
    with pytest.raises(ValueError):
        backtest.fold_for_date(pd.Timestamp("2020-01-02"), FOLDS)


def _preds(fold: str, date: str) -> pd.DataFrame:
    rows = [{"date": pd.Timestamp(date), "pid": 0, "fold": fold, "model": "naive",
             "target": t, "y_true": 0.1, "y_pred": 0.2, "vol_now": 0.1, "heldout": False}
            for t in ("vol", "dd", "var")]
    return pd.DataFrame(rows)


def test_stored_backtest_snaps_to_grid_date_not_after():
    p = _preds("2020", "2020-03-06")
    out = backtest.stored_backtest(0, pd.Timestamp("2020-03-10"), FOLDS, p)
    assert out["as_of"] == pd.Timestamp("2020-03-06")
    with pytest.raises(ValueError):
        backtest.stored_backtest(0, pd.Timestamp("2020-03-01"), FOLDS, p)


def test_stored_backtest_rejects_leaky_or_misassigned_rows():
    leaky = [{**FOLDS[0], "train_end": "2020-05-01"}]  # training reaches the prediction date
    with pytest.raises(RuntimeError, match="training period"):
        backtest.stored_backtest(0, pd.Timestamp("2020-06-01"), leaky, _preds("2020", "2020-03-06"))
    with pytest.raises(RuntimeError, match="wrong fold"):
        backtest.stored_backtest(0, pd.Timestamp("2020-06-01"), FOLDS, _preds("2021", "2020-03-06"))


def test_risk_band():
    q = [0.1, 0.2, 0.3]
    assert [predict.risk_band(v, q) for v in (0.05, 0.15, 0.25, 0.5)] == [
        "Low", "Moderate", "High", "Very high"]


@pytest.fixture
def synthetic(monkeypatch):
    T = 120
    idx = pd.bdate_range("2021-01-01", periods=T)
    rng = np.random.default_rng(2)
    returns = pd.DataFrame(rng.normal(0, 0.01, (T, 3)), index=idx, columns=list("ABC"))
    data = {"returns": returns, "market": pd.DataFrame(), "macro": pd.DataFrame()}
    monkeypatch.setattr(predict, "load_data", lambda: data)
    return returns, Portfolio(("A", "B"), (0.5, 0.5))


def test_realized_matches_targets_functions(synthetic):
    returns, p = synthetic
    r = portfolio_returns(returns, p)
    t = returns.index[30]
    got = backtest.realized(p, t)
    assert got["vol_fwd"] == pytest.approx(targets.forward_vol(r).loc[t])
    assert got["dd_flag"] == targets.forward_dd_flag(r).loc[t]
    assert got["r_next"] == pytest.approx(targets.next_return(r).loc[t])
    w = (1 + r.iloc[30 : 30 + config.H + 1]).cumprod()
    w = pd.concat([pd.Series([1.0]), (1 + r.iloc[31 : 31 + config.H]).cumprod()])
    assert got["max_dd"] == pytest.approx(float((w / w.cummax() - 1).min()))
    assert (got["max_dd"] <= -config.DD_THRESH) == bool(got["dd_flag"])


def test_realized_is_pending_near_the_end(synthetic):
    returns, p = synthetic
    got = backtest.realized(p, returns.index[-5])
    assert np.isnan(got["vol_fwd"]) and np.isnan(got["max_dd"]) and got["days_after"] == 4


# ---- with real artifacts ------------------------------------------------------------------
def _stored_portfolio(pid: int):
    ports = json.load(open(config.DATA_PROCESSED / "portfolios.json"))
    return ports[pid]["tickers"], ports[pid]["weights"]


@needs_artifacts
def test_custom_backtest_matches_saved_walk_forward_predictions():
    tickers, weights = _stored_portfolio(3)
    stored = backtest.stored_backtest(3, pd.Timestamp("2022-06-15"))
    custom = backtest.custom_backtest(tickers, weights, stored["as_of"])
    assert custom["as_of"] == stored["as_of"]
    for model, vals in stored["models"].items():
        for key, want in vals.items():
            got = custom["models"][model].get(key)
            if got is None or np.isnan(want):
                continue
            assert got == pytest.approx(want, abs=1e-4), (model, key)


@needs_artifacts
def test_backtests_never_touch_the_deployed_models(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("deployed models used")

    monkeypatch.setattr(predict, "load_models", boom)
    monkeypatch.setattr(predict, "load_meta", boom)
    tickers, weights = _stored_portfolio(3)
    backtest.stored_backtest(3, pd.Timestamp("2023-03-01"))
    backtest.custom_backtest(tickers, weights, pd.Timestamp("2023-03-01"))


@needs_artifacts
def test_predict_portfolio_rejects_dates_inside_the_training_period():
    tickers, weights = _stored_portfolio(3)
    with pytest.raises(ValueError, match="Backtest"):
        predict.predict_portfolio(tickers, weights, pd.Timestamp("2022-06-10"))


@needs_artifacts
def test_custom_backtest_ignores_data_after_as_of(monkeypatch):
    tickers, weights = _stored_portfolio(3)
    as_of = pd.Timestamp("2022-06-10")
    base = backtest.custom_backtest(tickers, weights, as_of)
    p = base["portfolio"]
    real_before = backtest.realized(p, as_of)
    d = predict.load_data()
    returns = d["returns"].copy()
    after = returns.index > as_of
    returns.loc[after] = np.random.default_rng(0).normal(0, 0.05, (after.sum(), returns.shape[1]))
    market = d["market"].copy()
    market.loc[market.index > as_of, config.MARKET] *= 1.3
    monkeypatch.setattr(predict, "load_data",
                        lambda: {"returns": returns, "market": market, "macro": d["macro"]})
    changed = backtest.custom_backtest(tickers, weights, as_of)
    for model, vals in base["models"].items():
        for key, v in vals.items():
            if not np.isnan(v):
                assert changed["models"][model][key] == pytest.approx(v, abs=1e-9), (model, key)
    assert backtest.realized(p, as_of)["vol_fwd"] != pytest.approx(real_before["vol_fwd"])
