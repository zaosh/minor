import zlib

import numpy as np
import pandas as pd
import pytest

from src import config
from src.data import fetch


@pytest.fixture
def prices():
    idx = pd.bdate_range("2010-01-01", periods=300)
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        100 * np.exp(np.cumsum(rng.normal(0, 0.01, (300, 3)), axis=0)),
        index=idx, columns=["A", "B", "LATE"],
    )
    df.loc[idx[:100], "LATE"] = np.nan
    return df


def test_coverage_drops_late_listing(prices):
    cov = fetch.coverage_report(prices, start="2010-01-01")
    assert cov.loc["A", "keep"] and cov.loc["B", "keep"]
    assert not cov.loc["LATE", "keep"]


def test_no_nan_after_alignment(prices):
    prices.loc[prices.index[50], "A"] = np.nan  # single gap gets forward-filled
    r, _ = fetch.clean_returns(prices[["A", "B"]])
    assert not r.isna().any().any()


def test_jump_flagging(prices):
    prices.iloc[200, 0] *= 1.5
    _, flags = fetch.clean_returns(prices[["A", "B"]])
    assert (flags["ticker"] == "A").any()
    assert len(flags) == 2 and flags["ret"].notna().all()  # the jump and its reversal
    assert not flags["zeroed"].any()  # +/-50% or less is kept


def test_implausible_return_is_zeroed(prices):
    prices.iloc[200:, 0] *= 0.1  # e.g. an unadjusted split
    r, flags = fetch.clean_returns(prices[["A", "B"]])
    assert r["A"].abs().max() < 0.25
    assert flags["zeroed"].sum() == 1


def test_macro_calendar_lag_hides_unpublished_values():
    obs = pd.Series([1.0, 2.0], index=pd.to_datetime(["2020-01-01", "2020-02-01"]))
    idx = pd.bdate_range("2020-01-01", "2020-04-30")
    out = fetch.lag_macro(obs, "45d", idx)
    assert out.loc[:"2020-02-14"].isna().all()  # Jan value (+45d = Feb 15) not yet public
    assert out.loc["2020-02-17"] == 1.0
    assert out.loc["2020-03-17"] == 2.0  # Feb value public from Mar 17


def test_macro_trading_day_lag_is_causal():
    idx = pd.bdate_range("2020-01-01", periods=10)
    s = pd.Series(np.arange(10.0), index=idx)
    out = fetch.lag_macro(s, 1, idx)
    assert np.isnan(out.iloc[0])
    assert (out.iloc[1:].values == s.iloc[:-1].values).all()


def test_fred_without_key_degrades_to_none(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "DATA_RAW", tmp_path)
    monkeypatch.delenv("FRED_API_KEY", raising=False)

    def boom(*a, **k):
        raise OSError("offline")

    monkeypatch.setattr(pd, "read_csv", boom)
    assert fetch.fetch_fred("DGS10") is None


def test_prices_cache_hit_never_refetches(monkeypatch, tmp_path, prices):
    monkeypatch.setattr(config, "DATA_RAW", tmp_path)
    prices.to_parquet(tmp_path / "prices.parquet")
    import yfinance

    def boom(*a, **k):
        raise AssertionError("network used")

    monkeypatch.setattr(yfinance, "download", boom)
    out = fetch.fetch_prices(["A", "B"])
    assert list(out.columns) == ["A", "B"]


def _fake_fred(cut: pd.Timestamp | None = None):
    """Synthetic FRED series for every configured id; values after `cut` are shifted."""
    def fake(series_id: str, refresh: bool = False) -> pd.Series:
        monthly = series_id in ("FEDFUNDS", "CPIAUCSL")
        obs = (pd.date_range("2014-01-01", "2019-12-31", freq="MS") if monthly
               else pd.bdate_range("2014-01-01", "2019-12-31"))
        rng = np.random.default_rng(zlib.crc32(series_id.encode()))
        base = 100 + np.cumsum(rng.normal(0, 0.5, len(obs)))
        s = pd.Series(base, index=obs, name="value")
        if cut is not None:
            s[s.index > cut] += 50.0
        return s
    return fake


def test_build_macro_is_causal(monkeypatch):
    """Changing any raw FRED value after day t must not change the macro row at or before t."""
    idx = pd.bdate_range("2016-01-01", "2018-12-31")
    vix = pd.Series(15 + np.sin(np.arange(len(idx)) / 20), index=idx)
    market = pd.DataFrame({config.VIX: vix, config.MARKET: 100.0}, index=idx)
    cut = pd.Timestamp("2017-06-15")
    monkeypatch.setattr(fetch, "fetch_fred", _fake_fred(None))
    base = fetch.build_macro(idx, market)
    monkeypatch.setattr(fetch, "fetch_fred", _fake_fred(cut))
    pert = fetch.build_macro(idx, market)
    pd.testing.assert_frame_equal(base.loc[:cut], pert.loc[:cut])
    assert not base.loc[cut + pd.Timedelta(days=120):].equals(
        pert.loc[cut + pd.Timedelta(days=120):])  # the perturbation is visible later


def test_refresh_keeps_old_cache_on_failure(monkeypatch, tmp_path, prices):
    monkeypatch.setattr(config, "DATA_RAW", tmp_path)
    prices.to_parquet(tmp_path / "prices.parquet")
    old_last = prices.index.max()
    monkeypatch.setattr(fetch, "download_prices", lambda t, start=None: prices[["A"]])  # B missing
    assert fetch.refresh_cache(["A", "B"]) is False
    assert pd.read_parquet(tmp_path / "prices.parquet").index.max() == old_last
    monkeypatch.setattr(fetch, "download_prices", lambda t, start=None: prices.iloc[:-5])
    assert fetch.refresh_cache(["A", "B"]) is False  # last date went backwards
    assert len(pd.read_parquet(tmp_path / "prices.parquet")) == len(prices)


def test_refresh_replaces_cache_and_keeps_fred_on_failure(monkeypatch, tmp_path, prices):
    monkeypatch.setattr(config, "DATA_RAW", tmp_path)
    prices.iloc[:-5].to_parquet(tmp_path / "prices.parquet")
    old_fred = pd.Series([1.0, 2.0], index=pd.to_datetime(["2020-01-01", "2020-02-01"]),
                         name="value")
    old_fred.to_frame("value").to_parquet(tmp_path / "fred_DGS10.parquet")
    monkeypatch.setattr(fetch, "download_prices", lambda t, start=None: prices)
    monkeypatch.setattr(fetch, "download_fred", lambda sid: None)
    assert fetch.refresh_cache(["A", "B"]) is True
    assert len(pd.read_parquet(tmp_path / "prices.parquet")) == len(prices)
    pd.testing.assert_series_equal(
        pd.read_parquet(tmp_path / "fred_DGS10.parquet")["value"], old_fred)


def test_build_macro_has_changes_not_levels(monkeypatch):
    idx = pd.bdate_range("2016-01-01", "2018-12-31")
    vix = pd.Series(15 + np.sin(np.arange(len(idx)) / 20), index=idx)
    market = pd.DataFrame({config.VIX: vix, config.MARKET: 100.0}, index=idx)
    monkeypatch.setattr(fetch, "fetch_fred", _fake_fred(None))
    cols = set(fetch.build_macro(idx, market).columns)
    assert cols == {"DGS10_d63", "FEDFUNDS_d126", "CPI_YOY_d126", "DEXINUS_r21", "DEXINUS_r63",
                    "DCOILBRENTEU_r21", "DCOILBRENTEU_r63", "VIX", "VIX_chg"}
