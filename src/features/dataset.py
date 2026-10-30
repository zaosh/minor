"""Build the modelling panel: one row per (date, portfolio id) on a shared date grid."""

from __future__ import annotations

import json
from multiprocessing import Pool

import numpy as np
import pandas as pd

from src import config
from src.features import engineer, targets
from src.models import baselines
from src.portfolios import Portfolio, generate_portfolios, portfolio_returns

WARMUP = 250  # first grid row needs 250 days of history (hist VaR / Ledoit-Wolf windows)
SEQ_CHANNELS = ["port_ret", "mkt_ret", "VIX", "VIX_chg"]

_CTX: dict = {}  # per-worker shared inputs, set by _init_worker


def grid_positions(n_days: int, h: int = config.H, stride: int = config.STRIDE) -> np.ndarray:
    """Row positions of the shared date grid: every `stride`-th day with full history and target."""
    return np.arange(WARMUP, n_days - h, stride)


def portfolio_frame(
    pid: int,
    p: Portfolio,
    returns: pd.DataFrame,
    mkt_ret: pd.Series,
    macro: pd.DataFrame,
    corr_cache: dict[int, np.ndarray],
    pos: np.ndarray,
) -> pd.DataFrame:
    """Features + causal baselines + targets for one portfolio on grid positions `pos`."""
    r = portfolio_returns(returns, p)
    members = [returns.columns.get_loc(t) for t in p.tickers]
    mean_corr = {
        L: engineer.mean_pair_corr(c, members, returns.index) for L, c in corr_cache.items()
    }
    feats = engineer.compute_features(r, mkt_ret, macro, mean_corr, len(p.tickers), p.hhi)
    base = pd.concat(
        [baselines.naive_vol(r), baselines.hist_var(r), baselines.param_var(r)],  # ewma: feature
        axis=1,
    )
    vol_fwd = targets.forward_vol(r)
    tgt = pd.concat(
        [vol_fwd, targets.log_ratio(vol_fwd, base["naive_vol"]), targets.forward_dd_flag(r),
         targets.next_return(r)],
        axis=1,
    )
    out = pd.concat([feats, base, tgt], axis=1).iloc[pos].copy()
    out["lw_vol"] = baselines.lw_vol(
        returns[list(p.tickers)].to_numpy(), np.array(p.weights), pos
    )
    out["pid"] = pid
    return out.reset_index(names="date")


def _init_worker(ctx: dict) -> None:
    """Pool initializer: store the shared inputs in each worker (works with spawn and fork)."""
    _CTX.clear()
    _CTX.update(ctx)


def _worker(args: tuple[int, Portfolio]) -> pd.DataFrame:
    pid, p = args
    c = _CTX
    return portfolio_frame(pid, p, c["returns"], c["mkt_ret"], c["macro"], c["corr"], c["pos"])


def build_panel(n_portfolios: int = config.N_PORTFOLIOS, workers: int = 8) -> pd.DataFrame:
    """Build and save panel.parquet, seq.npy and portfolios.json under data/processed."""
    d = config.DATA_PROCESSED
    returns = pd.read_parquet(d / "returns.parquet")
    market = pd.read_parquet(d / "market.parquet")
    macro = pd.read_parquet(d / "macro.parquet")
    mkt_ret = market[config.MARKET].pct_change().fillna(0.0).rename("mkt_ret")
    portfolios = generate_portfolios(n_portfolios, list(returns.columns))
    ctx = dict(
        returns=returns, mkt_ret=mkt_ret, macro=macro,
        corr=engineer.pair_corr_cache(returns), pos=grid_positions(len(returns)),
    )
    with Pool(workers, initializer=_init_worker, initargs=(ctx,)) as pool:
        frames = pool.map(_worker, list(enumerate(portfolios)))
    panel = pd.concat(frames).set_index(["date", "pid"]).sort_index()
    panel["day_idx"] = returns.index.get_indexer(panel.index.get_level_values("date"))
    panel = panel.astype("float32")
    panel.to_parquet(d / "panel.parquet")
    seq = np.stack(
        [
            np.column_stack(
                [portfolio_returns(returns, p), mkt_ret, macro["VIX"], macro["VIX_chg"]]
            )
            for p in portfolios
        ]
    ).astype("float32")
    np.save(d / "seq.npy", np.nan_to_num(seq))
    (d / "portfolios.json").write_text(
        json.dumps([{"tickers": p.tickers, "weights": p.weights} for p in portfolios])
    )
    return panel


if __name__ == "__main__":
    out = build_panel()
    print(out.shape, out.isna().sum().sum())
