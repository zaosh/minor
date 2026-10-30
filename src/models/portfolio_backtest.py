"""Downstream test: do better volatility forecasts give better volatility-targeted portfolios?

Reads the saved walk-forward `vol` predictions (reports/predictions.parquet); nothing is retrained.
For every model and portfolio, exposure to the portfolio is set on each forecast date t from the
forecast made at t (information up to and including t) and applied to the returns of days t+1 up
to the next forecast date::

    exposure_t = min(cap, target_vol / vol_hat_t)        remainder earns the daily risk-free rate

The cost is `COST_BPS` per unit of exposure change (the first position is built from cash).
The portfolio itself is the constant-weight series used everywhere else (daily rebalanced), so
internal turnover is not charged.

Pre-registered primary metric: tracking error of the realised strategy volatility against the
target (`vol_gap` = realised - target, summarised by RMSE and mean). Sharpe, max drawdown and
turnover are secondary. Two reference rows are added: `unscaled` (exposure 1) and `oracle`
(exposure from the realised future volatility, an unattainable upper bound, excluded from tests).
The forecast-level test is a Diebold-Mariano test on the date-level loss
(exposure_t * realised_vol_t - target)^2, which is the squared miss of the scaled 21-day volatility.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src import config
from src.models.significance import dm_test, holm

TARGET_VOL = 0.10
CAP = 1.0  # no leverage
COST_BPS = 15.0
BASELINE_REF = ("naive", "gjr")  # comparators for the DM tests
REFERENCE = ("unscaled", "oracle")
VOL_FLOOR = 1e-4


def exposure(vol_hat: np.ndarray, target: float = TARGET_VOL, cap: float = CAP) -> np.ndarray:
    """Volatility-targeting exposure target / vol_hat, capped at `cap` (no leverage)."""
    return np.minimum(cap, target / np.maximum(np.asarray(vol_hat, dtype=float), VOL_FLOOR))


def simulate(
    returns: np.ndarray, grid: np.ndarray, expo: np.ndarray, rf_daily: float,
    cost_bps: float = COST_BPS,
) -> tuple[np.ndarray, np.ndarray]:
    """Strategy daily returns for exposures set on grid days.

    `returns` is (P, T) portfolio returns, `grid` the increasing day positions of the forecast
    dates and `expo` the (P, G) exposures chosen at those days. Exposure `expo[:, i]` is held over
    days grid[i]+1 .. grid[i+1] (the last one to the end of the data). Returns the (P, D) strategy
    returns for days grid[0]+1 .. T-1 and the matching (D,) day positions.
    """
    P, T = returns.shape
    days = np.arange(grid[0] + 1, T)
    owner = np.searchsorted(grid, days, side="left") - 1  # last grid day strictly before the day
    held = expo[:, owner]
    strat = held * returns[:, days] + (1 - held) * rf_daily
    prev = np.concatenate([np.zeros((P, 1)), expo[:, :-1]], axis=1)
    change = np.abs(expo - prev) * cost_bps / 1e4
    first_day = np.searchsorted(days, grid + 1)  # day index where each exposure starts
    ok = first_day < len(days)
    strat[:, first_day[ok]] -= change[:, ok]
    return strat, days


def segment_metrics(strat: np.ndarray, held: np.ndarray, rf_daily: float,
                    target: float = TARGET_VOL) -> pd.DataFrame:
    """Per-portfolio metrics of one calendar segment: (P, D) strategy returns, (P, D) exposure."""
    ann = 252
    mean = strat.mean(axis=1)
    vol = strat.std(axis=1, ddof=1) * np.sqrt(ann)
    wealth = np.cumprod(1 + strat, axis=1)
    peak = np.maximum.accumulate(np.concatenate([np.ones((len(strat), 1)), wealth], axis=1), axis=1)
    mdd = (np.concatenate([np.ones((len(strat), 1)), wealth], axis=1) / peak - 1).min(axis=1)
    sharpe = (mean - rf_daily) * ann / np.where(vol > 0, vol, np.nan)
    turnover = np.abs(np.diff(held, axis=1)).sum(axis=1) * ann / max(strat.shape[1], 1)
    return pd.DataFrame({
        "ann_return": mean * ann, "ann_vol": vol, "vol_gap": vol - target, "sharpe": sharpe,
        "max_dd": mdd, "turnover": turnover, "mean_exposure": held.mean(axis=1)})


def load_inputs() -> tuple[np.ndarray, pd.DatetimeIndex, list[dict]]:
    """Portfolio returns (400, T), the trading-day index and the portfolio definitions."""
    returns = pd.read_parquet(config.DATA_PROCESSED / "returns.parquet")
    ports = json.loads((config.DATA_PROCESSED / "portfolios.json").read_text())
    W = np.zeros((len(ports), returns.shape[1]))
    for i, q in enumerate(ports):
        W[i, [returns.columns.get_loc(t) for t in q["tickers"]]] = q["weights"]
    return W @ returns.to_numpy().T, returns.index, ports


def vol_forecasts(preds: pd.DataFrame, n_pid: int) -> tuple[pd.DatetimeIndex, dict, np.ndarray]:
    """Wide (G, P) forecast arrays per model and the realised-vol (G, P) array."""
    v = preds[preds["target"] == "vol"]
    out, truth = {}, None
    for model, g in v.groupby("model", sort=False):
        pv = g.pivot(index="date", columns="pid", values="y_pred").sort_index()
        tv = g.pivot(index="date", columns="pid", values="y_true").sort_index()
        if pv.shape[1] != n_pid or pv.isna().any().any():
            raise ValueError(f"{model}: incomplete forecast grid")
        out[model] = pv.to_numpy()
        if truth is None:
            dates, truth = pv.index, tv.to_numpy()
        elif not (pv.index.equals(dates) and np.allclose(tv.to_numpy(), truth)):
            raise ValueError(f"{model}: grid or realised vol differs from the other models")
    return dates, out, truth


def run(preds: pd.DataFrame, target: float = TARGET_VOL, cap: float = CAP,
        cost_bps: float = COST_BPS) -> dict[str, pd.DataFrame]:
    """Backtest every model; return per-fold metrics and the date-level tracking losses."""
    R, idx, ports = load_inputs()
    dates, fc, truth = vol_forecasts(preds, len(ports))
    grid = idx.get_indexer(dates)
    if (grid < 0).any():
        raise ValueError("forecast dates missing from the returns index")
    rf_daily = (1 + config.RF_ANNUAL) ** (1 / 252) - 1
    exposures = {m: exposure(a, target, cap).T for m, a in fc.items()}
    exposures["unscaled"] = np.ones((len(ports), len(grid)))
    exposures["oracle"] = exposure(truth, target, cap).T
    years = idx[grid[0] + 1:].year.to_numpy()
    fold = np.where(years >= 2025, "2025+", years.astype(str))  # matches evaluate.TEST_YEARS
    rows, losses = [], {}
    for model, E in exposures.items():
        strat, days = simulate(R, grid, E, rf_daily, cost_bps)
        held = E[:, np.searchsorted(grid, days, side="left") - 1]
        for f in pd.unique(fold):
            m = fold == f
            df = segment_metrics(strat[:, m], held[:, m], rf_daily, target)
            df.insert(0, "pid", np.arange(len(ports)))
            df.insert(0, "fold", f)
            df.insert(0, "model", model)
            rows.append(df)
        if model not in REFERENCE or model == "unscaled":
            e_grid = E.T if model != "unscaled" else np.ones_like(truth)
            losses[model] = pd.Series(((e_grid * truth - target) ** 2).mean(axis=1), index=dates)
    return {"by_pid_fold": pd.concat(rows, ignore_index=True), "losses": pd.DataFrame(losses)}


def summarize(by_pid: pd.DataFrame, pids: set[int] | None = None) -> pd.DataFrame:
    """Mean over portfolios within each fold, then mean and std across folds."""
    d = by_pid if pids is None else by_pid[by_pid["pid"].isin(pids)]
    d = d.assign(vol_gap_sq=d["vol_gap"] ** 2)
    per_fold = d.groupby(["model", "fold"]).mean(numeric_only=True).drop(columns="pid")
    per_fold["tracking_rmse"] = np.sqrt(per_fold["vol_gap_sq"])
    per_fold = per_fold.drop(columns="vol_gap_sq")
    agg = per_fold.groupby("model").agg(["mean", "std"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    return agg.sort_values("tracking_rmse_mean")


def dm_tests(losses: pd.DataFrame, h: int = 5) -> pd.DataFrame:
    """DM tests of every model's tracking loss against the reference baselines (Holm per family)."""
    rows = []
    for ref in BASELINE_REF:
        fam = []
        for m in losses.columns:
            if m == ref or m in REFERENCE and m != "unscaled":
                continue
            r = dm_test(losses[m].to_numpy(), losses[ref].to_numpy(), h)
            fam.append({"model": m, "versus": ref, **r})
        p = holm(np.array([x["p"] for x in fam]))
        rows += [{**x, "p_holm": q} for x, q in zip(fam, p, strict=True)]
    return pd.DataFrame(rows)


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
    dm_tests(res["losses"]).to_csv(a.out / "dm_tracking.csv", index=False)
    res["by_pid_fold"].groupby(["model", "fold"]).mean(numeric_only=True).to_csv(
        a.out / "by_fold.csv")
    print(summarize(res["by_pid_fold"]).round(4).to_string())


if __name__ == "__main__":
    main()
