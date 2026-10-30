import numpy as np
import pandas as pd
import pytest

from src import config, forecast_log
from src.features import targets
from src.portfolios import Portfolio, portfolio_returns

H = config.H


def fake_result(as_of="2024-03-01", tickers=("A", "B"), weights=(0.6, 0.4)):
    p = Portfolio(tuple(tickers), tuple(weights))
    pred = {"vol_hat": 0.2, "dd_hat": 0.3, "var_hat": -0.02}
    return {"as_of": pd.Timestamp(as_of), "portfolio": p,
            "ml": {"rf": {**pred, "var_hat": np.nan}, "lstm": dict(pred)},
            "baselines": {"naive": dict(pred), "hist_var": {"var_hat": -0.03}}}


@pytest.fixture
def returns():
    idx = pd.bdate_range("2024-01-01", periods=80)
    rng = np.random.default_rng(4)
    return pd.DataFrame(rng.normal(0, 0.01, (80, 2)), index=idx, columns=["A", "B"])


def test_portfolio_key_is_order_invariant_and_rounded():
    a = forecast_log.portfolio_key(["B", "A"], [0.4, 0.6])
    b = forecast_log.portfolio_key(["A", "B"], [0.6, 0.4000001])
    assert a == b == '{"A":0.6,"B":0.4}'


def test_duplicate_forecast_is_skipped_regardless_of_order(tmp_path):
    path = tmp_path / "log.csv"
    assert forecast_log.append_forecast(fake_result(), "v1", path) is True
    swapped = fake_result(tickers=("B", "A"), weights=(0.4, 0.6))
    assert forecast_log.append_forecast(swapped, "v1", path) is False
    assert forecast_log.append_forecast(fake_result(), "v2", path) is True  # new model version
    assert forecast_log.append_forecast(fake_result("2024-03-04"), "v1", path) is True
    log = forecast_log.load_log(path)
    assert log["forecast_id"].nunique() == 3
    assert len(log) == 3 * 4  # 4 models per forecast


def test_csv_round_trip_types(tmp_path):
    path = tmp_path / "log.csv"
    forecast_log.append_forecast(fake_result(), "v1", path)
    log = forecast_log.load_log(path)
    assert list(log.columns) == forecast_log.COLUMNS
    assert pd.api.types.is_datetime64_any_dtype(log["as_of"])
    assert log["as_of"].iloc[0] == pd.Timestamp("2024-03-01")
    assert log["vol_hat"].dtype == float and log["var_hat"].isna().sum() == 1  # only rf lacks VaR
    assert log.loc[log["model"] == "rf", "var_hat"].isna().all()
    assert (log["vol_now"] == 0.2).all() and log["forecast_id"].str.len().eq(12).all()


def test_missing_file_gives_empty_log(tmp_path):
    assert forecast_log.load_log(tmp_path / "none.csv").empty


def _scored(returns, as_of_pos, n_total=None):
    r = returns if n_total is None else returns.iloc[:n_total]
    res = fake_result(as_of=returns.index[as_of_pos])
    log = forecast_log.forecast_rows(res, "v1")
    log["as_of"] = pd.to_datetime(log["as_of"])
    return forecast_log.score_log(log, r)


def test_completion_counts_trading_days_across_a_holiday_gap(returns):
    # a trading calendar with a 5-day hole (holiday) after the forecast date
    idx = returns.index.delete([12, 13, 14, 15, 16])
    r = returns.reindex(idx)
    pos = 10
    for n_after, status in ((H - 1, f"pending {H - 1}/{H}"), (H, "complete")):
        s = forecast_log.score_log(
            forecast_log.forecast_rows(fake_result(idx[pos]), "v1").assign(
                as_of=lambda d: pd.to_datetime(d["as_of"])), r.iloc[: pos + n_after + 1])
        assert (s["status"] == status).all()
        assert s["realized_vol"].notna().all() == (status == "complete")
        assert s["next_return"].notna().all()  # VaR is scorable after one day
    assert (idx[pos + H] - idx[pos]).days > H + 5  # the calendar gap really was there


def test_pending_and_unscorable_statuses(returns):
    s = _scored(returns, as_of_pos=len(returns) - 1)  # nothing after yet
    assert (s["status"] == "pending 0/21").all() and s["next_return"].isna().all()
    log = forecast_log.forecast_rows(fake_result("2030-01-01"), "v1")
    log["as_of"] = pd.to_datetime(log["as_of"])
    assert (forecast_log.score_log(log, returns)["status"] == "unscorable").all()


def test_realized_equals_targets_and_ignores_other_days(returns):
    pos = 20
    s = _scored(returns, pos)
    r = portfolio_returns(returns, Portfolio(("A", "B"), (0.6, 0.4)))
    assert s["realized_vol"].iloc[0] == pytest.approx(targets.forward_vol(r).iloc[pos])
    assert s["dd_occurred"].iloc[0] == targets.forward_dd_flag(r).iloc[pos]
    assert s["next_return"].iloc[0] == pytest.approx(r.iloc[pos + 1])
    # changing returns at or before t, or after t+H, leaves the realized outcome unchanged
    changed = returns.copy()
    changed.iloc[: pos + 1] = 0.5
    changed.iloc[pos + H + 1 :] = -0.5
    s2 = _scored(changed, pos)
    s2 = s2.assign(as_of=s["as_of"])
    for c in ("realized_vol", "dd_occurred", "next_return"):
        assert s2[c].iloc[0] == pytest.approx(s[c].iloc[0])


def test_summary_scores_only_what_is_scorable(returns):
    s = _scored(returns, 20)
    out = forecast_log.summarize_scores(s)
    assert out.loc["lstm", "n_vol_dd"] == 1 and out.loc["lstm", "n_var"] == 1
    assert out.loc["rf", "n_var"] == 0  # no VaR head
    assert out.loc["lstm", "vol_mae"] == pytest.approx(abs(0.2 - s["realized_vol"].iloc[0]))
