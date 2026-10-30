"""Round 4 (B2/B3): HAR and GARCH forecasts as model inputs, and the simulated drawdown probability.

All three inputs are causal and refreshed once per calendar year, so no panel row sees a fit that
used its own year's returns or labels:

* GARCH(1,1) features and `sim_dd_raw`: for refresh year Y the GARCH parameters and residuals use
  returns strictly before the first grid day of Y (the same fit the walk-forward GARCH baseline
  uses in test year Y), filtered forward through the year. The result is stored in
  `data/processed/model_features.parquet`, which depends on returns only (not on the holdout).
* `har_vol`: OLS coefficients of the HAR baseline on rows with day + H < ts_Y (ts_Y = first
  trading day of Y), excluding held-out portfolios. It depends on the holdout, so it is added by
  `add_model_features` inside evaluate/seeds/train_final, not stored.

Rows of years after the last refresh year (2026) use the last refresh (that of fold "2025+").
"""

from __future__ import annotations

from multiprocessing import Pool

import numpy as np
import pandas as pd

from src import config
from src.models import prereg, simdd
from src.models.baselines import ANN, HAR, FHSBaseline, GarchFitCache

_CTX: dict = {}  # per-worker shared inputs, set by _init_worker


def refresh_year(year: int, last_refresh: int = prereg.LAST_REFRESH_YEAR) -> int:
    """The refresh year whose fit serves rows of calendar `year` (later years reuse the last)."""
    return min(int(year), last_refresh)


def refresh_points(day_idx: np.ndarray, years: np.ndarray,
                   last_refresh: int = prereg.LAST_REFRESH_YEAR) -> dict[int, tuple[int, int]]:
    """{refresh year Y: (start, last)} = first and last grid day served by Y's fit.

    `start` is the first grid day of Y (the GARCH fit cut-off); `last` is the last grid day of
    Y, or of the whole panel for the last refresh year (it also serves the later years).
    """
    ry = np.minimum(years, last_refresh)
    return {int(y): (int(day_idx[ry == y].min()), int(day_idx[ry == y].max()))
            for y in np.unique(ry)}


def _init_worker(ctx: dict) -> None:
    """Pool initializer: store the shared inputs in each worker (works with spawn and fork)."""
    _CTX.clear()
    _CTX.update(ctx)


def _pid_features(pid: int) -> tuple[pd.DataFrame, list[dict]]:
    """Features of one portfolio's panel rows plus its per-year fit diagnostics."""
    c = _CTX
    rows = c["rows"][pid]
    day, ewma, ry = c["day"][rows], c["ewma"][rows], c["ry"][rows]
    cache = GarchFitCache(c["R"])
    fhs = FHSBaseline(c["R"])  # only for its EWMA residual pool on a failed fit
    out = np.full((len(rows), 4), np.nan)  # vol21, sigma1 (annualised), sim_dd_raw, fallback
    diag = []
    for y, (start, last) in c["points"].items():
        m = ry == y
        if not m.any():
            continue
        k = day[m] - start
        res = cache.get(pid, start, last)
        ok = res is not None and bool(
            np.isfinite(res["vol21"][k]).all() and (res["vol21"][k] > 0).all()
            and np.isfinite(res["sigma1"][k]).all() and (res["sigma1"][k] > 0).all())
        U = c["U"][y]
        if ok:
            p = res["params"]
            vol21, sigma1 = res["vol21"][k], res["sigma1"][k]
            dd = simdd.simulate_dd_prob(sigma1, res["mu"], p["omega"] / 1e4, p["alpha"],
                                        p["beta"], res["z"], U)
            diag.append({"pid": pid, "year": y, "converged": True, "mu": res["mu"], **p})
        else:
            e = ewma[m].astype(float)
            vol21, sigma1 = e, e / ANN
            r = simdd.ewma_recursion()
            dd = simdd.simulate_dd_prob(sigma1, r["mu"], r["omega"], r["alpha"], r["beta"],
                                        fhs._ewma_z(pid, start), U)
            diag.append({"pid": pid, "year": y, "converged": False})
        out[m] = np.column_stack([vol21, sigma1 * ANN, dd, np.full(m.sum(), 0.0 if ok else 1.0)])
    frame = pd.DataFrame(out, columns=["garch_vol21", "garch_sigma1", "sim_dd_raw",
                                       "garch_fallback"], index=c["index"][rows])
    return frame, diag


def garch_panel_features(R: np.ndarray, panel: pd.DataFrame, dates: pd.DatetimeIndex,
                         workers: int = 8, last_refresh: int = prereg.LAST_REFRESH_YEAR
                         ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """GARCH features and simulated drawdown probability for every panel row.

    `R` is (n_portfolios, T) daily portfolio returns aligned with `day_idx`. Returns the feature
    frame (float64, same index as `panel`) and a per-(pid, year) diagnostics table.
    """
    day = panel["day_idx"].to_numpy().astype(int)
    years = dates[day].year.to_numpy()
    pid = panel.index.get_level_values("pid").to_numpy()
    points = refresh_points(day, years, last_refresh)
    rows = {int(p): np.flatnonzero(pid == p) for p in np.unique(pid)}
    ctx = {"R": R.astype("float64"), "rows": rows, "day": day,
           "ry": np.minimum(years, last_refresh), "ewma": panel["ewma_vol"].to_numpy(),
           "index": panel.index, "points": points,
           "U": {y: simdd.uniform_matrix(y) for y in points}}
    pids = sorted(rows)
    if workers <= 1:
        _init_worker(ctx)
        results = [_pid_features(p) for p in pids]
    else:
        with Pool(workers, initializer=_init_worker, initargs=(ctx,)) as pool:
            results = pool.map(_pid_features, pids)
    feats = pd.concat([r[0] for r in results]).reindex(panel.index)
    diag = pd.DataFrame([d for r in results for d in r[1]])
    return feats, diag


def har_coefficients(panel: pd.DataFrame, dates: pd.DatetimeIndex, exclude_pids=None,
                     h: int = config.H, last_refresh: int = prereg.LAST_REFRESH_YEAR
                     ) -> dict[int, np.ndarray]:
    """HAR OLS coefficients per refresh year: fit on rows with day + h < ts_Y, minus `exclude`.

    A year with no earlier labelled rows gets the "no change" coefficients (har_vol = vol_21).
    """
    day = panel["day_idx"].to_numpy().astype(int)
    years = dates[day].year.to_numpy()
    pid = panel.index.get_level_values("pid").to_numpy()
    keep = ~np.isin(pid, exclude_pids) if exclude_pids is not None else np.ones(len(panel), bool)
    out = {}
    for y in np.unique(np.minimum(years, last_refresh)):
        ts = int(np.searchsorted(dates.year.to_numpy(), y))  # first trading day of year y
        train = np.flatnonzero((day + h < ts) & keep)
        if len(train) == 0:
            out[int(y)] = np.array(prereg.HAR_NO_CHANGE_COEF)
            continue
        har = HAR()
        har.fit(panel.iloc[train])
        out[int(y)] = np.asarray(har.coef_, dtype=float)
    return out


def har_feature(panel: pd.DataFrame, dates: pd.DatetimeIndex, exclude_pids=None,
                last_refresh: int = prereg.LAST_REFRESH_YEAR) -> tuple[pd.Series, dict]:
    """`har_vol` for every row (each row uses its refresh year's coefficients) and the table."""
    coefs = har_coefficients(panel, dates, exclude_pids, last_refresh=last_refresh)
    day = panel["day_idx"].to_numpy().astype(int)
    ry = np.minimum(dates[day].year.to_numpy(), last_refresh)
    har = HAR()
    out = np.full(len(panel), np.nan)
    for y, coef in coefs.items():
        m = ry == y
        har.coef_ = coef
        out[m] = np.exp(har._design(panel.iloc[np.flatnonzero(m)]) @ coef)
    return pd.Series(out, index=panel.index, name="har_vol"), coefs


def add_model_features(panel: pd.DataFrame, dates: pd.DatetimeIndex, feats: pd.DataFrame,
                       exclude_pids=None) -> pd.DataFrame:
    """Panel plus har_vol, garch_vol21, garch_sigma1, garch_fallback and sim_dd_raw.

    `feats` comes from `garch_panel_features` / model_features.parquet (restricted to the panel
    rows); `exclude_pids` are the held-out portfolios left out of the HAR coefficient fits.
    """
    joined = feats.reindex(panel.index)
    if joined.isna().any().any():
        raise ValueError("model features are missing for some panel rows; "
                         "run python -m src.features.model_features")
    har, _ = har_feature(panel, dates, exclude_pids)
    out = panel.copy()
    out["har_vol"] = har
    for c in ("garch_vol21", "garch_sigma1", "garch_fallback", "sim_dd_raw"):
        out[c] = joined[c].to_numpy()
    return out


def main(workers: int = 8) -> None:
    """Write data/processed/model_features.parquet and garch_params.parquet (new files only)."""
    d = config.DATA_PROCESSED
    panel = pd.read_parquet(d / "panel.parquet")
    dates = pd.read_parquet(d / "returns.parquet").index
    R = np.load(d / "seq.npy")[:, :, 0]
    feats, diag = garch_panel_features(R, panel, dates, workers)
    feats.to_parquet(d / "model_features.parquet")
    diag.to_parquet(d / "garch_params.parquet")
    print(feats.shape, "fallback share:", float(feats["garch_fallback"].mean()),
          "NaN:", int(feats.isna().sum().sum()))


if __name__ == "__main__":
    main()
