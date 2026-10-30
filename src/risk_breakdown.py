"""Where a portfolio's risk comes from, and how today's weights would have fared in past stress.

Everything uses returns up to the as-of date only. Contributions are descriptive (they split
a realised-risk measure across holdings); they are not forecasts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.covariance import ledoit_wolf

from src import config
from src.portfolios import Portfolio, portfolio_returns


def _window(returns: pd.DataFrame, p: Portfolio, as_of: pd.Timestamp, window: int) -> pd.DataFrame:
    """The last `window` daily returns of the holdings up to and including `as_of`."""
    w = returns[list(p.tickers)].loc[:pd.Timestamp(as_of)].iloc[-window:]
    if len(w) < window:
        raise ValueError(f"Need {window} days of history up to {pd.Timestamp(as_of):%Y-%m-%d}.")
    return w


def vol_contributions(returns: pd.DataFrame, p: Portfolio, as_of: pd.Timestamp,
                      window: int = 250) -> pd.DataFrame:
    """Euler decomposition of annualized volatility with a Ledoit-Wolf covariance.

    marginal_i = (Sigma w)_i / sigma_p, contribution_i = w_i * marginal_i, and the
    contributions sum to sigma_p, which equals the `lw_vol` baseline for the same date.
    """
    r = _window(returns, p, as_of, window)
    cov = ledoit_wolf(r.to_numpy())[0] * 252
    w = np.array(p.weights)
    sigma = float(np.sqrt(w @ cov @ w))
    marginal = cov @ w / sigma
    contrib = w * marginal
    return pd.DataFrame({"ticker": list(p.tickers), "weight": w, "marginal_vol": marginal,
                         "vol_contribution": contrib, "share": contrib / sigma})


def es_contributions(returns: pd.DataFrame, p: Portfolio, as_of: pd.Timestamp, window: int = 250,
                     alpha: float = config.VAR_ALPHA) -> pd.DataFrame:
    """Euler decomposition of historical expected shortfall (daily return units).

    contribution_i = w_i * mean(r_i on the portfolio's worst alpha days); the contributions sum
    to the historical ES of the same window. With 250 days that is only about 12 tail days, so
    treat the split as indicative.
    """
    r = _window(returns, p, as_of, window)
    w = np.array(p.weights)
    rp = r.to_numpy() @ w
    tail = rp <= np.quantile(rp, alpha)
    contrib = w * r.to_numpy()[tail].mean(axis=0)
    return pd.DataFrame({"ticker": list(p.tickers), "weight": w, "es_contribution": contrib,
                         "share": contrib / contrib.sum(), "n_tail_days": int(tail.sum())})


def stress_replay(returns: pd.DataFrame, market: pd.Series, p: Portfolio, start: str,
                  end: str) -> dict:
    """Apply `p`'s weights (daily rebalanced, as everywhere in the project) to a past window.

    Returns the wealth path (start = 1 at the `start` close), total return, max drawdown, worst
    day, realised annualized vol, the market's return over the window, and each holding's sum of
    daily contributions. This is a replay of history, not a forecast.
    """
    r = portfolio_returns(returns, p).loc[start:end]
    if r.empty or r.index.min() > pd.Timestamp(start) + pd.Timedelta(days=7):
        raise ValueError(f"No return data for the window {start} to {end}.")
    if r.index[0] == pd.Timestamp(start):
        r = r.iloc[1:]  # the start day's own return happened before the window opens
    wealth = pd.concat([pd.Series([1.0], index=[pd.Timestamp(start)]), (1 + r).cumprod()])
    mk = market.loc[start:end]
    parts = (returns.loc[r.index, list(p.tickers)] * np.array(p.weights)).sum()
    return {"wealth": wealth, "total_return": float(wealth.iloc[-1] - 1),
            "max_drawdown": float((wealth / wealth.cummax() - 1).min()),
            "worst_day": float(r.min()), "worst_day_date": r.idxmin(),
            "realized_vol": float(r.std() * np.sqrt(252)),
            "market_return": float(mk.iloc[-1] / mk.iloc[0] - 1),
            "contributions": parts.rename("sum_of_daily_contributions"), "n_days": len(r)}
