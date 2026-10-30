"""Statistical comparison of the walk-forward predictions (numpy/scipy/pandas only).

Portfolios share market days, so rows are far from independent. Every test here therefore works
on DATE-LEVEL series: the loss of each model on each test date, averaged over portfolios. That
leaves about 330 dates in the 2020-2025+ test period. On the stride-5 date grid a 21-day target
overlaps its neighbours (MA(4) errors), so volatility and drawdown tests use a HAC variance with
at least 4 lags; VaR and ES are one-day targets.

Caveats: the tests assume stationary loss differentials (expanding-window forecasts are not
covered by Giacomini-White theory), and 330 dates is a small sample.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from src import config
from src.models.evaluate import fz0_rows, variance_losses

ML_NAMES = ("rf", "xgboost", "lstm")
HORIZON_STRIDE = {"vol": 5, "dd": 5, "var": 1, "es": 1}  # h = ceil(21 / grid stride) for 21-day


def is_ml(model: str) -> bool:
    """True for the ML models and their conformal-calibrated variants."""
    return model.removesuffix("_cal") in ML_NAMES


def is_combo(model: str) -> bool:
    """True for the equal-weight forecast combinations (and their calibrated variants)."""
    return model.removesuffix("_cal").startswith("combo_")


def holm(p: np.ndarray) -> np.ndarray:
    """Holm step-down adjusted p-values (monotone, capped at 1), in the input order."""
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    adj = np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order])
    out = np.empty_like(p)
    out[order] = np.minimum(adj, 1.0)
    return out


def _row_loss(target: str, g: pd.DataFrame, loss: str = "se") -> np.ndarray:
    y, p = g["y_true"].to_numpy(), g["y_pred"].to_numpy()
    if target == "vol" and loss in ("qlike", "mse_var"):
        q, m, _ = variance_losses(y, p)
        return q if loss == "qlike" else m
    if target in ("vol", "dd"):
        return (y - p) ** 2  # squared error / Brier
    if target == "var":
        d = y - p
        return np.maximum(config.VAR_ALPHA * d, (config.VAR_ALPHA - 1) * d)
    return fz0_rows(y, g["y_aux"].to_numpy(), p)


def date_losses(preds: pd.DataFrame, target: str, loss: str = "se") -> pd.DataFrame:
    """Loss per test date (rows) and model (columns), averaged over portfolios.

    `loss` picks the volatility loss: "se" (squared error, the RMSE basis), "qlike" or
    "mse_var"; it is ignored for the other targets.
    """
    d = preds[preds["target"] == target]
    parts = {}
    for model, g in d.groupby("model", sort=False):
        parts[model] = pd.Series(_row_loss(target, g, loss), index=pd.to_datetime(g["date"])
                                 ).groupby(level=0).mean()
    return pd.DataFrame(parts).sort_index().dropna()


def default_lags(T: int, h: int = 1) -> int:
    """Newey-West lag count: max(h - 1, floor(4 (T/100)^(2/9)))."""
    return max(h - 1, int(np.floor(4 * (T / 100) ** (2 / 9))))


def newey_west_lrv(x: np.ndarray, lags: int) -> float:
    """Bartlett-kernel long-run variance of a series (mean removed)."""
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    T = len(x)
    lrv = float(x @ x) / T
    for k in range(1, lags + 1):
        lrv += 2 * (1 - k / (lags + 1)) * float(x[k:] @ x[:-k]) / T
    return max(lrv, 1e-300)


def dm_lrv(x: np.ndarray, h: int) -> float:
    """Diebold-Mariano (1995) variance for h-step forecasts: gamma0 + 2 * sum of the first h-1
    autocovariances (rectangular kernel, exact for MA(h-1) errors such as overlapping targets).
    Falls back to Bartlett Newey-West if the estimate is not positive."""
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    T = len(x)
    lrv = float(x @ x) / T + 2 * sum(float(x[k:] @ x[:-k]) / T for k in range(1, h))
    return lrv if lrv > 0 else newey_west_lrv(x, default_lags(T, h))


def hac_mean_test(x: np.ndarray, h: int = 1) -> dict[str, float]:
    """t-test that mean(x) = 0 with a HAC variance and the Harvey-Leybourne-Newbold small-sample
    correction; two-sided p from Student t(T-1). h = 1 uses a Bartlett Newey-West variance, h > 1
    the Diebold-Mariano truncated variance for overlapping h-step forecast errors."""
    x = np.asarray(x, dtype=float)
    T = len(x)
    lrv = dm_lrv(x, h) if h > 1 else newey_west_lrv(x, default_lags(T, h))
    stat = x.mean() / np.sqrt(lrv / T)
    hln = np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    stat *= hln
    return {"mean": float(x.mean()), "stat": float(stat),
            "p": float(2 * stats.t.sf(abs(stat), T - 1)), "T": T}


def dm_test(loss_a: np.ndarray, loss_b: np.ndarray, h: int = 1) -> dict[str, float]:
    """Diebold-Mariano test on date-level losses: mean(loss_a - loss_b) = 0. A negative mean
    (and stat) means model A has the lower loss."""
    return hac_mean_test(np.asarray(loss_a) - np.asarray(loss_b), h)


def mcs(losses: pd.DataFrame, alpha: float = 0.10, B: int = 2000, block: int = 6,
        seed: int = config.SEED) -> pd.DataFrame:
    """Hansen-Lunde-Nason model confidence set (T_max statistic, moving-block bootstrap).

    Repeatedly tests equal predictive ability of the surviving models and removes the worst
    until the test no longer rejects at `alpha`. Returns each model's MCS p-value (it is in the
    90% set when p >= alpha) in elimination order, best survivors last.
    """
    L = losses.to_numpy(dtype=float)
    T, M = L.shape
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, T, size=(B, int(np.ceil(T / block))))
    idx = ((starts[:, :, None] + np.arange(block)) % T).reshape(B, -1)[:, :T]
    boot = L[idx].mean(axis=1)  # (B, M) bootstrap loss means
    alive, out, p_prev = list(range(M)), {}, 0.0
    while len(alive) > 1:
        a = np.array(alive)
        D = L[:, a].mean(0) - L[:, a].mean()
        Db = boot[:, a] - boot[:, a].mean(axis=1, keepdims=True)
        se = Db.std(axis=0, ddof=1)
        t = D / se
        tmax = t.max()
        tboot = ((Db - D) / se).max(axis=1)
        p = float((tboot >= tmax).mean())
        worst = alive[int(np.argmax(t))]
        p_prev = max(p_prev, p)
        out[losses.columns[worst]] = p_prev
        alive.remove(worst)
    out[losses.columns[alive[0]]] = 1.0
    return pd.DataFrame({"model": list(out), "mcs_p": list(out.values())}).assign(
        in_set=lambda d: d["mcs_p"] >= alpha)


def clustered_coverage(preds: pd.DataFrame, alpha: float = config.VAR_ALPHA) -> pd.DataFrame:
    """Date-clustered check of VaR calibration, next to (not instead of) the pooled Kupiec test.

    For every model, the share of portfolios exceeding the VaR on each date is averaged and
    tested against alpha with a HAC variance over dates, for all dates and for each year.
    """
    rows = []
    v = preds[preds["target"] == "var"]
    for model, g in v.groupby("model", sort=False):
        e = (pd.Series((g["y_true"] < g["y_pred"]).astype(float).to_numpy(),
                       index=pd.to_datetime(g["date"])).groupby(level=0).mean() - alpha)
        for label, s in [("all", e)] + [(str(y), x) for y, x in e.groupby(e.index.year)]:
            r = hac_mean_test(s.to_numpy(), 1)
            rows.append({"model": model, "period": label, "n_dates": r["T"],
                         "exceed_rate": r["mean"] + alpha, "stat": r["stat"], "p": r["p"]})
    return pd.DataFrame(rows)


def dm_table(preds: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """DM tests of every ML model against (a) the best baseline of the same kind and (b) the
    reference baseline (naive vol, climatology, historical VaR).

    Raw ML models are compared with raw baselines and calibrated ones with calibrated
    baselines, so calibration never favours one side.
    """
    reference = {"vol": "naive", "dd": "climatology", "var": "hist_var"}
    rows = []
    for target in ("vol", "dd", "var", "es"):
        if not (preds["target"] == target).any():
            continue
        L = date_losses(preds, target)
        h = HORIZON_STRIDE[target]
        for m in [c for c in L.columns if is_ml(c)]:
            cal = m.endswith("_cal")
            pool = [c for c in L.columns
                    if not is_ml(c) and not is_combo(c) and c.endswith("_cal") == cal]
            if not pool:
                continue
            best = L[pool].mean().idxmin()
            refs = [("best_baseline", best)]
            ref = reference.get(target)
            if ref and (ref + ("_cal" if cal else "")) in L.columns:
                refs.append(("reference", ref + ("_cal" if cal else "")))
            for kind, other in refs:
                r = dm_test(L[m].to_numpy(), L[other].to_numpy(), h)
                rows.append({"target": target, "model": m, "vs": kind, "baseline": other,
                             "mean_loss_diff": r["mean"], "dm_stat": r["stat"], "p": r["p"],
                             "better": bool(r["mean"] < 0), "significant": bool(r["p"] < alpha),
                             "n_dates": r["T"]})
    return pd.DataFrame(rows)


def dm_losses_table(preds: pd.DataFrame, losses: tuple[str, ...], alpha: float = 0.05
                    ) -> pd.DataFrame:
    """DM tests of each raw ML model on the volatility `losses` (e.g. "qlike", "mse_var"),
    against the best raw baseline under that loss and against naive vol. Combos are not
    baselines. Same layout as `dm_table` plus a `loss` column."""
    rows = []
    for loss in losses:
        L = date_losses(preds, "vol", loss)
        pool = [c for c in L.columns if not is_ml(c) and not is_combo(c)]
        best = L[pool].mean().idxmin()
        for m in [c for c in L.columns if c in ML_NAMES]:
            for kind, other in (("best_baseline", best), ("reference", "naive")):
                r = dm_test(L[m].to_numpy(), L[other].to_numpy(), HORIZON_STRIDE["vol"])
                rows.append({"target": "vol", "loss": loss, "model": m, "vs": kind,
                             "baseline": other, "mean_loss_diff": r["mean"],
                             "dm_stat": r["stat"], "p": r["p"], "better": bool(r["mean"] < 0),
                             "significant": bool(r["p"] < alpha), "n_dates": r["T"]})
    return pd.DataFrame(rows)


def dm_regime(d: np.ndarray, stress: np.ndarray, h: int) -> dict[str, float]:
    """Regime-specific DM test: OLS of the date-level loss differential `d` on two regime
    dummies (stress, calm; no intercept) with a Newey-West Bartlett sandwich variance
    (lags = `default_lags(T, h)`) and t(T-2) p-values. Each coefficient is that regime's mean
    loss difference (negative = model A better)."""
    d = np.asarray(d, dtype=float)
    s = np.asarray(stress, dtype=bool)
    T = len(d)
    X = np.column_stack([s.astype(float), (~s).astype(float)])
    beta = np.linalg.solve(X.T @ X, X.T @ d)
    u = d - X @ beta
    g = X * u[:, None]
    lags = default_lags(T, h)
    S = g.T @ g / T
    for k in range(1, lags + 1):
        G = g[k:].T @ g[:-k] / T
        S += (1 - k / (lags + 1)) * (G + G.T)
    XtXi = np.linalg.inv(X.T @ X / T)
    V = XtXi @ S @ XtXi / T
    se = np.sqrt(np.maximum(np.diag(V), 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(se > 0, beta / se, np.where(beta == 0, 0.0, np.inf * np.sign(beta)))
    p = 2 * stats.t.sf(np.abs(t), T - 2)
    return {"stress_mean": float(beta[0]), "calm_mean": float(beta[1]),
            "stress_stat": float(t[0]), "calm_stat": float(t[1]),
            "stress_p": float(p[0]), "calm_p": float(p[1]),
            "n_stress": int(s.sum()), "n_calm": int((~s).sum()), "lags": lags}


def run_significance(preds: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """All date-level tests for a long predictions frame: DM, MCS and clustered coverage."""
    mcs_rows = []
    for target in ("vol", "dd", "var", "es"):
        if (preds["target"] == target).any():
            m = mcs(date_losses(preds, target))
            mcs_rows.append(m.assign(target=target))
    return {"dm": dm_table(preds), "mcs": pd.concat(mcs_rows, ignore_index=True),
            "coverage": clustered_coverage(preds)}
