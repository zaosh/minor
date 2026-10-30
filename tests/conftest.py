"""Shared fixtures."""

import numpy as np
import pandas as pd
import pytest

from src.features import dataset, engineer
from src.portfolios import Portfolio, portfolio_returns


@pytest.fixture(scope="session")
def world():
    """Synthetic 7-year market, 4 portfolios, panel with a (date, pid) index."""
    T, N = 1800, 6
    idx = pd.bdate_range("2015-01-01", periods=T)
    rng = np.random.default_rng(5)
    vol = 0.008 + 0.01 * (np.sin(np.arange(T) / 150) ** 2)  # slowly varying vol
    returns = pd.DataFrame(rng.normal(0.0003, 1, (T, N)) * vol[:, None], index=idx,
                           columns=[f"S{i}" for i in range(N)])
    mkt = returns.mean(axis=1).rename("mkt_ret")
    macro = pd.DataFrame({"VIX": 15.0 + 0 * vol}, index=idx)
    ports = [Portfolio(("S0", "S1"), (0.5, 0.5)), Portfolio(("S2", "S3", "S4"), (0.2, 0.3, 0.5)),
             Portfolio(("S0", "S5", "S3"), (0.6, 0.2, 0.2)), Portfolio(("S1", "S4"), (0.9, 0.1))]
    cache, pos = engineer.pair_corr_cache(returns), dataset.grid_positions(T)
    frames = [dataset.portfolio_frame(i, p, returns, mkt, macro, cache, pos)
              for i, p in enumerate(ports)]
    panel = pd.concat(frames).set_index(["date", "pid"]).sort_index()
    panel["day_idx"] = idx.get_indexer(panel.index.get_level_values("date"))
    R = np.stack([portfolio_returns(returns, p).to_numpy() for p in ports])
    return panel, idx, R
