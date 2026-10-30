"""Downstream test: do better volatility forecasts give better volatility-targeted portfolios?

Reads the saved walk-forward `vol` predictions (reports/predictions.parquet); nothing is retrained.
For every model and portfolio, exposure to the portfolio is set on each forecast date t from the
forecast made at t (information up to and including t) and held, buy-and-hold, over the returns of
days t+1 up to the next forecast date (the last one for one stride only)::

    exposure_t = min(cap, target_vol / vol_hat_t)        remainder earns the daily risk-free rate

At each forecast date the position is traded back to the new exposure and `COST_BPS` is charged
on the change from the *drifted* exposure (the position moves with the market between dates); the
first position is built from cash. The portfolio is the constant-weight series used everywhere
else (daily rebalanced), so its internal turnover is not charged. Costs are charged on wealth.

Primary metric (revised after code review, see HANDOVER): RMSE between the realised volatility of
non-overlapping `BLOCK_DAYS`-day blocks of the strategy and the target, over the blocks in which the
target is reachable without leverage (the unscaled portfolio's own volatility in that block is at
least the target; this mask is model-independent). The all-block RMSE and the full-fold gap are
reported as secondary, with Sharpe, max drawdown and turnover. Two reference rows are added:
`unscaled` (exposure 1) and `oracle` (exposure from the realised future volatility, an
unattainable ceiling); neither enters the tests.
The forecast-level test is a Diebold-Mariano test on the date-level loss
(exposure_t * realised_21d_vol_t - target)^2, the squared miss of the scaled 21-day volatility the
forecast was made for; all comparisons are corrected together with one Holm family.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src import config
from src.models.significance import HORIZON_STRIDE, dm_test, holm
from src.portfolios import Portfolio, portfolio_returns

TARGET_VOL = 0.10
CAP = 1.0  # no leverage
COST_BPS = 15.0
BLOCK_DAYS = 63
BASELINE_REF = ("naive", "gjr")  # comparators for the DM tests
REFERENCE = ("unscaled", "oracle")
VOL_FLOOR = 1e-4


def exposure(vol_hat: np.ndarray, target: float = TARGET_VOL, cap: float = CAP) -> np.ndarray:
    """Volatility-targeting exposure target / vol_hat, capped at `cap` (no leverage)."""
    return np.minimum(cap, target / np.maximum(np.asarray(vol_hat, dtype=float), VOL_FLOOR))


def simulate(
    returns: np.ndarray, grid: np.ndarray, expo: np.ndarray, rf_daily: float,
    cost_bps: float = COST_BPS, end: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Buy-and-hold strategy returns for exposures set on grid days.

    `returns` is (P, T) portfolio returns, `grid` the increasing day positions of the forecast
    dates and `expo` the (P, G) exposures chosen at those days. Exposure `expo[:, i]` is held over
    days grid[i]+1 .. grid[i+1]; the last one is held to `end` (default: the last data day).
    On each rebalance the position is traded to the target and the cost is charged on the
    change from the drifted exposure. Returns the (P, D) strategy returns, the (D,) day positions
    (grid[0]+1 .. end) and the (P, D) exposure at the start of each day.
    """
    P, T = returns.shape
    end = T - 1 if end is None else min(end, T - 1)
    bounds = np.append(grid, end)
    days = np.arange(grid[0] + 1, end + 1)
    strat = np.empty((P, len(days)))
    held = np.empty((P, len(days)))
    drifted = np.zeros(P)
    for i in range(len(grid)):
        lo, hi = bounds[i] + 1, bounds[i + 1]  # inclusive day positions of this holding period
        if hi < lo:
            continue
        e = expo[:, i]
        cost = np.abs(e - drifted) * cost_bps / 1e4
        risky = np.cumprod(1 + returns[:, lo:hi + 1], axis=1)
        cash = (1 + rf_daily) ** np.arange(1, hi - lo + 2)
        wealth = e[:, None] * risky + (1 - e[:, None]) * cash
        prev = np.concatenate([np.ones((P, 1)), wealth[:, :-1] * (1 - cost)[:, None]], axis=1)
        sl = slice(lo - days[0], hi - days[0] + 1)
        strat[:, sl] = wealth * (1 - cost)[:, None] / prev - 1
        held[:, sl] = e[:, None]
        drifted = e * risky[:, -1] / wealth[:, -1]
    return strat, days, held


def block_vols(strat: np.ndarray, block: int = BLOCK_DAYS) -> np.ndarray:
    """Annualised volatility of consecutive non-overlapping `block`-day blocks, (P, n_blocks)."""
    n = strat.shape[1] // block
    if n == 0:
        return np.empty((len(strat), 0))
    x = strat[:, : n * block].reshape(len(strat), n, block)
    return x.std(axis=2, ddof=1) * np.sqrt(252)


def tracking_rmse(bv: np.ndarray, target: float, mask: np.ndarray | None = None) -> np.ndarray:
    """Per-portfolio RMSE of block volatility vs target over the blocks where `mask` is True."""
    mask = np.ones_like(bv, dtype=bool) if mask is None else mask
    n = mask.sum(axis=1)
    sq = np.where(mask, (bv - target) ** 2, 0.0).sum(axis=1)
    return np.sqrt(np.divide(sq, n, out=np.full(len(bv), np.nan), where=n > 0))


def segment_metrics(strat: np.ndarray, held: np.ndarray, rf_daily: float,
                    target: float = TARGET_VOL) -> pd.DataFrame:
    """Per-portfolio metrics of one calendar segment: (P, D) strategy returns, (P, D) exposure."""
    ann = 252
    mean = strat.mean(axis=1)
    vol = strat.std(axis=1, ddof=1) * np.sqrt(ann)
    wealth = np.concatenate([np.ones((len(strat), 1)), np.cumprod(1 + strat, axis=1)], axis=1)
    mdd = (wealth / np.maximum.accumulate(wealth, axis=1) - 1).min(axis=1)
    sharpe = (mean - rf_daily) * ann / np.where(vol > 0, vol, np.nan)
    turnover = np.abs(np.diff(held, axis=1)).sum(axis=1) * ann / max(strat.shape[1], 1)
    return pd.DataFrame({
        "ann_return": mean * ann, "ann_vol": vol, "vol_gap": vol - target, "sharpe": sharpe,
        "max_dd": mdd, "turnover": turnover, "mean_exposure": held.mean(axis=1)})


def load_inputs() -> tuple[np.ndarray, pd.DatetimeIndex, list[dict]]:
    """Portfolio returns (400, T) via `portfolios.portfolio_returns`, the day index, definitions."""
    returns = pd.read_parquet(config.DATA_PROCESSED / "returns.parquet")
    ports = json.loads((config.DATA_PROCESSED / "portfolios.json").read_text())
    R = np.stack([portfolio_returns(returns, Portfolio(tuple(q["tickers"]), tuple(q["weights"])))
                  .to_numpy() for q in ports])
    return R, returns.index, ports


def vol_forecasts(preds: pd.DataFrame, n_pid: int) -> tuple[pd.DatetimeIndex, dict, np.ndarray]:
    """Wide (G, P) forecast arrays per model and the realised-vol (G, P) array."""
    v = preds[preds["target"] == "vol"]
    out, truth = {}, None
    for model, g in v.groupby("model", sort=False):
        pv = g.pivot(index="date", columns="pid", values="y_pred").sort_index()
        tv = g.pivot(index="date", columns="pid", values="y_true").sort_index()
        if list(pv.columns) != list(range(n_pid)) or pv.isna().any().any():
            raise ValueError(f"{model}: forecast pids must be 0..{n_pid - 1} with no gaps")
        out[model] = pv.to_numpy()
        if truth is None:
            dates, truth = pv.index, tv.to_numpy()
        elif not (pv.index.equals(dates) and np.allclose(tv.to_numpy(), truth)):
            raise ValueError(f"{model}: grid or realised vol differs from the other models")
    return dates, out, truth


def run(preds: pd.DataFrame, target: float = TARGET_VOL, cap: float = CAP,
        cost_bps: float = COST_BPS) -> dict:
    """Backtest every model; return per-fold metrics and the (G, P) tracking-loss arrays."""
    R, idx, ports = load_inputs()
    dates, fc, truth = vol_forecasts(preds, len(ports))
    grid = idx.get_indexer(dates)
    if (grid < 0).any():
        raise ValueError("forecast dates missing from the returns index")
    rf_daily = (1 + config.RF_ANNUAL) ** (1 / 252) - 1
    exposures = {"unscaled": np.ones((len(ports), len(grid)))}
    exposures.update({m: exposure(a, target, cap).T for m, a in fc.items()})
    exposures["oracle"] = exposure(truth, target, cap).T
    end = int(grid[-1] + (grid[-1] - grid[-2]))  # last exposure is held for one stride only
    years = idx[grid[0] + 1: end + 1].year.to_numpy()
    fold = np.where(years >= 2025, "2025+", years.astype(str))  # matches evaluate.TEST_YEARS
    blocks: dict[tuple[str, str], np.ndarray] = {}
    frames: dict[tuple[str, str], pd.DataFrame] = {}
    for model, E in exposures.items():
        strat, days, held = simulate(R, grid, E, rf_daily, cost_bps, end)
        for f in pd.unique(fold):
            m = fold == f
            df = segment_metrics(strat[:, m], held[:, m], rf_daily, target)
            df.insert(0, "n_days", int(m.sum()))
            frames[(model, f)] = df
            blocks[(model, f)] = block_vols(strat[:, m])
    rows = []
    for (model, f), df in frames.items():
        reach = blocks[("unscaled", f)] >= target
        df = df.assign(tracking_rmse=tracking_rmse(blocks[(model, f)], target),
                       tracking_rmse_reach=tracking_rmse(blocks[(model, f)], target, reach),
                       reach_share=reach.mean() if reach.size else np.nan)
        df.insert(0, "pid", np.arange(len(ports)))
        df.insert(0, "fold", f)
        df.insert(0, "model", model)
        rows.append(df)
    cube = {m: (np.asarray(E).T * truth - target) ** 2 for m, E in exposures.items()
            if m != "oracle"}
    return {"by_pid_fold": pd.concat(rows, ignore_index=True), "cube": cube, "dates": dates}


def summarize(by_pid: pd.DataFrame, pids: set[int] | None = None) -> pd.DataFrame:
    """Mean over portfolios within each fold, then mean and std across folds.

    `tracking_rmse_reach_w` weights the folds by their number of days (the 2025+ fold is longer).
    """
    d = by_pid if pids is None else by_pid[by_pid["pid"].isin(pids)]
    per_fold = d.groupby(["model", "fold"]).mean(numeric_only=True).drop(columns="pid")
    w = per_fold.groupby("model").apply(
        lambda g: np.average(g["tracking_rmse_reach"], weights=g["n_days"]), include_groups=False)
    agg = per_fold.drop(columns="n_days").groupby("model").agg(["mean", "std"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg["tracking_rmse_reach_w"] = w
    return agg.sort_values("tracking_rmse_reach_mean")


def loss_frame(cube: dict, dates: pd.DatetimeIndex, pids: set[int] | None = None) -> pd.DataFrame:
    """Date-level tracking loss per model, averaged over `pids` (default all)."""
    cols = slice(None) if pids is None else sorted(pids)
    return pd.DataFrame({m: a[:, cols].mean(axis=1) for m, a in cube.items()}, index=dates)


def dm_tests(losses: pd.DataFrame, h: int = HORIZON_STRIDE["vol"]) -> pd.DataFrame:
    """DM tests of every contender against naive and gjr, with ONE Holm family over all of them."""
    rows = []
    for ref in BASELINE_REF:
        for m in losses.columns:
            if m in REFERENCE or m == ref or (ref == "gjr" and m == "naive"):
                continue
            rows.append({"model": m, "versus": ref,
                         **dm_test(losses[m].to_numpy(), losses[ref].to_numpy(), h)})
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out["p"].to_numpy())
    return out


def main(argv: list[str] | None = None) -> None:
    """CLI: python -m src.models.portfolio_backtest [--preds PATH] [--out DIR]."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--preds", type=Path, default=config.REPORTS / "predictions.parquet")
    ap.add_argument("--out", type=Path, default=config.REPORTS / "portfolio_backtest")
    a = ap.parse_args(argv)
    preds = pd.read_parquet(a.preds, filters=[("target", "==", "vol")])
    res = run(preds)
    a.out.mkdir(parents=True, exist_ok=True)
    held = set(json.loads((config.REPORTS / "holdout_pids.json").read_text()))
    summarize(res["by_pid_fold"]).to_csv(a.out / "summary_all.csv")
    summarize(res["by_pid_fold"], held).to_csv(a.out / "summary_heldout.csv")
    dm_tests(loss_frame(res["cube"], res["dates"])).to_csv(a.out / "dm_tracking.csv", index=False)
    dm_tests(loss_frame(res["cube"], res["dates"], held)).to_csv(
        a.out / "dm_tracking_heldout.csv", index=False)
    (res["by_pid_fold"].drop(columns="pid").groupby(["model", "fold"]).mean()
     .to_csv(a.out / "by_fold.csv"))
    print(summarize(res["by_pid_fold"])[["tracking_rmse_reach_mean", "tracking_rmse_reach_w",
                                         "tracking_rmse_mean", "vol_gap_mean", "sharpe_mean",
                                         "max_dd_mean", "turnover_mean"]].round(4).to_string())


if __name__ == "__main__":
    main()
