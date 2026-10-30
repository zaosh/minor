import pytest

from src import config

pytestmark = pytest.mark.skipif(
    not (config.ARTIFACTS / "meta.json").exists()
    or not (config.DATA_PROCESSED / "returns.parquet").exists(),
    reason="needs cached data and trained artifacts (see README)",
)


def test_predict_portfolio_outputs():
    from src.predict import predict_portfolio

    res = predict_portfolio(["TCS.NS", "HDFCBANK.NS", "GOLDBEES.NS"], [0.4, 0.4, 0.2])
    for name in ("rf", "xgboost", "lstm"):
        assert res["ml"][name]["vol_hat"] > 0
        assert 0 <= res["ml"][name]["dd_hat"] <= 1
    assert res["ml"]["xgboost"]["var_hat"] < 0 and res["ml"]["lstm"]["var_hat"] < 0
    assert {"naive", "ewma", "ledoit_wolf", "hist_var", "param_var"} <= set(res["baselines"])


@pytest.mark.parametrize("tickers,weights", [
    (["NOPE.NS"], [1.0]), (["TCS.NS", "INFY.NS"], [0.7, 0.7]), (["TCS.NS", "TCS.NS"], [0.5, 0.5]),
    (["TCS.NS", "INFY.NS"], [1.5, -0.5]),
])
def test_bad_portfolios_raise_value_error(tickers, weights):
    from src.predict import predict_portfolio

    with pytest.raises(ValueError):
        predict_portfolio(tickers, weights)


def test_features_match_training_panel():
    """The app's feature code must reproduce the training panel exactly (no train/serve skew)."""
    import json

    import numpy as np
    import pandas as pd

    from src.portfolios import Portfolio
    from src.predict import portfolio_features

    panel = pd.read_parquet(config.DATA_PROCESSED / "panel.parquet")
    ports = json.load(open(config.DATA_PROCESSED / "portfolios.json"))
    sub = panel.xs(3, level="pid")
    date = sub.index[200]
    port = Portfolio(tuple(ports[3]["tickers"]), tuple(ports[3]["weights"]))
    row = portfolio_features(port, date)
    cols = [c for c in row.columns if c in sub.columns and c != "day_idx"]
    np.testing.assert_allclose(row[cols].iloc[0].to_numpy(float),
                               sub.loc[date, cols].to_numpy(float), rtol=1e-4, atol=1e-5)


def test_streamlit_app_smoke_and_bad_ticker():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(config.ROOT / "app" / "streamlit_app.py"), default_timeout=120).run()
    assert not at.exception
    assert len(at.metric) == 3
    at.text_input[0].set_value("INFY, FAKETICKER").run()
    assert not at.exception
    assert any("FAKETICKER" in w.value for w in at.warning)


def _app():
    from streamlit.testing.v1 import AppTest

    return AppTest.from_file(str(config.ROOT / "app" / "streamlit_app.py"), default_timeout=120)


def test_swapping_a_ticker_does_not_inherit_its_weight():
    at = _app().run()
    picked = [t for t in at.multiselect[0].value if t != "GOLDBEES.NS"] + ["INFY.NS"]
    at.multiselect[0].set_value(picked).run()
    assert not at.exception
    wmap = at.session_state["weight_map"]
    assert wmap["INFY.NS"] == pytest.approx(100 / len(picked))  # not GOLDBEES' 15


def test_rf_has_no_var_and_app_says_so():
    at = _app().run()
    at.selectbox[0].set_value("rf").run()
    assert not at.exception
    assert any("no VaR head" in c.value for c in at.caption)
    assert at.metric[2].value == "n/a"


def test_backtest_view_stored_and_typed_portfolio():
    at = _app().run()
    view = next(r for r in at.radio if "Forecast" in r.options)
    view.set_value("Backtest").run()
    assert not at.exception, at.exception
    assert any("NOT used here" in i.value for i in at.info)
    assert len(at.dataframe) >= 1
    src = next(r for r in at.radio if "Sidebar portfolio" in r.options)
    src.set_value("Sidebar portfolio").run()
    assert not at.exception, at.exception
    assert any("Prediction date used" in c.value for c in at.caption)


def test_backtest_view_works_without_deployed_models(monkeypatch):
    from src import predict

    def missing():
        raise FileNotFoundError("meta.json")

    monkeypatch.setattr(predict, "load_meta", missing)
    at = _app().run()
    view = next(r for r in at.radio if "Forecast" in r.options)
    view.set_value("Backtest").run()
    assert not at.exception, at.exception
    assert any("trained only on earlier data" in i.value for i in at.info)
    assert not any("NOT used here" in i.value for i in at.info)


def test_risk_breakdown_view_renders_every_scenario():
    from src import config

    at = _app().run()
    view = next(r for r in at.radio if "Forecast" in r.options)
    view.set_value("Risk breakdown").run()
    assert not at.exception, at.exception
    assert any("add up to the portfolio volatility" in c.value for c in at.caption)
    scen = next(b for b in at.selectbox if b.label == "Scenario")
    for name in config.STRESS_SCENARIOS:
        scen.set_value(name).run()
        assert not at.exception, (name, at.exception)
        assert len(at.metric) == 4


def test_forecast_log_view_empty_state(monkeypatch, tmp_path):
    from src import forecast_log

    monkeypatch.setattr(forecast_log, "LOG_PATH", tmp_path / "none.csv")
    monkeypatch.setattr(forecast_log.load_log, "__defaults__", (tmp_path / "none.csv",))
    at = _app().run()
    next(r for r in at.radio if "Forecast log" in r.options).set_value("Forecast log").run()
    assert not at.exception, at.exception
    assert any("No forecasts saved" in i.value for i in at.info)


def test_save_forecast_button_writes_once(monkeypatch, tmp_path):
    from src import forecast_log

    path = tmp_path / "log.csv"
    monkeypatch.setattr(forecast_log.append_forecast, "__defaults__", (path,))
    at = _app().run()
    assert not path.exists()  # nothing is logged just by viewing
    save = next(b for b in at.button if b.label == "Save forecast to log")
    save.click().run()
    assert path.exists() and any("Saved" in s.value for s in at.success)
    n = len(forecast_log.load_log(path))
    next(b for b in at.button if b.label == "Save forecast to log").click().run()
    assert len(forecast_log.load_log(path)) == n
    assert any("already saved" in s.value for s in at.success)
