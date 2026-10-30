"""Backward-looking features. The value at row t uses data up to and including t only."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import config
from src.features.targets import max_drawdown_rows
from src.models.baselines import ewma_vol

ANN = np.sqrt(252)


def rolling_max_drawdown(r: pd.Series, window: int) -> pd.Series:
    """Max drawdown over the last `window` returns (<= 0); NaN for the first `window` rows."""
    rows = max_drawdown_rows(r.to_numpy(), window)  # row i covers returns i+1..i+window
    out = np.full(len(r), np.nan)
    out[window:] = rows  # shifted so the value at t covers returns t-window+1..t
    return pd.Series(out, index=r.index)


def pair_corr_cache(returns: pd.DataFrame, lookbacks=config.LOOKBACKS) -> dict[int, np.ndarray]:
    """Rolling pairwise correlation matrices of the whole universe: {L: array (T, N, N)}."""
    n = returns.shape[1]
    cache = {}
    for L in lookbacks:
        cov = returns.rolling(L).cov().to_numpy().reshape(len(returns), n, n)
        sd = np.sqrt(np.einsum("tii->ti", cov))
        cache[L] = (cov / (sd[:, :, None] * sd[:, None, :])).astype(np.float32)
    return cache


def mean_pair_corr(cache: np.ndarray, members: list[int], index: pd.DatetimeIndex) -> pd.Series:
    """Mean off-diagonal correlation among `members` (column positions); 1.0 if only one."""
    k = len(members)
    if k < 2:
        return pd.Series(1.0, index=index)
    sub = cache[:, members][:, :, members]
    return pd.Series((sub.sum(axis=(1, 2)) - k) / (k * (k - 1)), index=index)


def compute_features(
    port_ret: pd.Series,
    mkt_ret: pd.Series,
    macro: pd.DataFrame,
    mean_corr: dict[int, pd.Series],
    k: int,
    hhi: float,
    rf_annual: float = config.RF_ANNUAL,
    lookbacks=config.LOOKBACKS,
) -> pd.DataFrame:
    """Feature matrix for one portfolio on the dates of `port_ret` (causal)."""
    rf = rf_annual / 252
    logr = np.log1p(port_ret)
    cols: dict[str, pd.Series] = {}
    for L in lookbacks:
        roll = port_ret.rolling(L)
        sd = roll.std()
        excess = port_ret - rf
        downside = np.sqrt((excess.clip(upper=0) ** 2).rolling(L).mean())
        cols[f"vol_{L}"] = sd * ANN
        cols[f"mdd_{L}"] = rolling_max_drawdown(port_ret, L)
        cols[f"sharpe_{L}"] = excess.rolling(L).mean() / sd * ANN
        cols[f"sortino_{L}"] = excess.rolling(L).mean() / downside.clip(lower=1e-8) * ANN
        cols[f"mom_{L}"] = np.expm1(logr.rolling(L).sum())
        cols[f"skew_{L}"] = roll.skew()
        cols[f"kurt_{L}"] = roll.kurt()
        cols[f"corr_mkt_{L}"] = roll.corr(mkt_ret)
        cols[f"beta_mkt_{L}"] = roll.cov(mkt_ret) / mkt_ret.rolling(L).var()
        cols[f"mean_pair_corr_{L}"] = mean_corr[L]
    cols["ewma_vol"] = ewma_vol(port_ret)
    cols["mkt_vol_21"] = mkt_ret.rolling(21).std() * ANN
    cols["mkt_mom_21"] = np.expm1(np.log1p(mkt_ret).rolling(21).sum())
    cols["k"] = pd.Series(float(k), index=port_ret.index)
    cols["hhi"] = pd.Series(hhi, index=port_ret.index)
    feats = pd.DataFrame(cols)
    return feats.join(macro.reindex(port_ret.index))
