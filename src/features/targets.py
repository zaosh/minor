"""Forward-looking targets. The value at row t uses returns t+1..t+H only."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import config

ANN = np.sqrt(252)


def max_drawdown_rows(returns: np.ndarray, n: int) -> np.ndarray:
    """Max drawdown (<= 0) of the wealth path W[i..i+n] for every start i in 0..T-n-1.

    W[j] is the cumulative wealth after return j; the path starts at W[i] (the initial peak),
    so row i depends on returns i+1..i+n only. The result has length T-n.
    """
    w = np.cumprod(1.0 + returns)
    win = np.lib.stride_tricks.sliding_window_view(w, n + 1)
    dd = win / np.maximum.accumulate(win, axis=1) - 1.0
    return dd.min(axis=1)


def forward_vol(r: pd.Series, h: int = config.H) -> pd.Series:
    """Annualized realized vol of returns t+1..t+h (NaN for the last h rows)."""
    return (r.rolling(h).std().shift(-h) * ANN).rename("vol_fwd")


def forward_dd_flag(
    r: pd.Series, h: int = config.H, thresh: float = config.DD_THRESH
) -> pd.Series:
    """1.0 if max drawdown over t..t+h (starting peak = W_t) is >= thresh, else 0.0."""
    mdd = np.full(len(r), np.nan)
    mdd[: len(r) - h] = max_drawdown_rows(r.to_numpy(), h)
    flag = np.where(np.isnan(mdd), np.nan, (mdd <= -thresh).astype(float))
    return pd.Series(flag, index=r.index, name="dd_flag")


def next_return(r: pd.Series) -> pd.Series:
    """Next-day return r_{t+1}, the realisation the 1-day VaR is evaluated against."""
    return r.shift(-1).rename("r_next")


def log_ratio(vol_fwd: pd.Series, vol_now: pd.Series, floor: float = 1e-4) -> pd.Series:
    """log(vol_fwd / vol_now): the ratio target (proposal P1)."""
    return np.log(vol_fwd.clip(lower=floor) / vol_now.clip(lower=floor)).rename("y_logratio")
