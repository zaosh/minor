"""VaR backtests beyond Kupiec: Christoffersen independence / conditional coverage and DQ.

Hits sit on the stride-5 date grid, so "independence" means between grid dates 5 trading days
apart, not between consecutive trading days. Per-portfolio tests have only about 17 hits each:
they are descriptive (reported as the share of portfolios that reject), and the inferential
version is the date-level DQ test on the cross-sectional hit rate with a HAC variance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import xlogy

from src import config
from src.models import prereg
from src.models.evaluate import kupiec_pof
from src.models.significance import default_lags, holm

ALPHA = config.VAR_ALPHA
NAN_RESULT = {"n00": 0, "n01": 0, "n10": 0, "n11": 0, "lr_pof": np.nan, "lr_ind": np.nan,
              "lr_cc": np.nan, "p_ind": np.nan, "p_cc": np.nan}


def _transition_counts(hits: np.ndarray) -> tuple[int, int, int, int]:
    a, b = hits[:-1], hits[1:]
    return (int(np.sum((a == 0) & (b == 0))), int(np.sum((a == 0) & (b == 1))),
            int(np.sum((a == 1) & (b == 0))), int(np.sum((a == 1) & (b == 1))))


def _lr_from_counts(n00: int, n01: int, n10: int, n11: int, x: int, n: int) -> dict:
    """Christoffersen LR_ind (chi2 1) and LR_cc = LR_pof + LR_ind (chi2 2); 0 log 0 = 0."""
    pi01 = n01 / max(n00 + n01, 1)
    pi11 = n11 / max(n10 + n11, 1)
    pi = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)
    alt = (xlogy(n00, 1 - pi01) + xlogy(n01, pi01) + xlogy(n10, 1 - pi11) + xlogy(n11, pi11))
    null = xlogy(n00 + n10, 1 - pi) + xlogy(n01 + n11, pi)
    lr_ind = max(float(-2 * (null - alt)), 0.0)
    lr_pof = kupiec_pof(x, n)[0]
    lr_cc = lr_pof + lr_ind
    return {"n00": n00, "n01": n01, "n10": n10, "n11": n11, "lr_pof": lr_pof, "lr_ind": lr_ind,
            "lr_cc": lr_cc, "p_ind": float(stats.chi2.sf(lr_ind, 1)),
            "p_cc": float(stats.chi2.sf(lr_cc, 2))}


def christoffersen(hits: np.ndarray) -> dict[str, float]:
    """Christoffersen tests for one 0/1 hit series. Fewer than `prereg.MIN_HITS` hits (or a
    series shorter than 2) is not testable: the LR/p fields are NaN."""
    h = np.asarray(hits).astype(int)
    if len(h) < 2 or h.sum() < prereg.MIN_HITS:
        return dict(NAN_RESULT)
    return _lr_from_counts(*_transition_counts(h), int(h.sum()), len(h))


def christoffersen_pooled(hits_by_unit: list[np.ndarray]) -> dict[str, float]:
    """Christoffersen tests with transition counts summed over units (portfolios); the
    frequency part is the pooled Kupiec test. Descriptive: pooled tests are overpowered."""
    n00 = n01 = n10 = n11 = x = n = 0
    for h in hits_by_unit:
        h = np.asarray(h).astype(int)
        if len(h) >= 2:
            c = _transition_counts(h)
            n00, n01, n10, n11 = n00 + c[0], n01 + c[1], n10 + c[2], n11 + c[3]
        x, n = x + int(h.sum()), n + len(h)
    if x < prereg.MIN_HITS:
        return dict(NAN_RESULT)
    return _lr_from_counts(n00, n01, n10, n11, x, n)


def _lagged_design(y: np.ndarray, extra: np.ndarray, lags: int) -> tuple[np.ndarray, np.ndarray]:
    """Rows t >= lags: [1, y_{t-1..t-lags}, extra_t] and the response y_t."""
    T = len(y)
    cols = [np.ones(T - lags)] + [y[lags - k : T - k] for k in range(1, lags + 1)]
    cols.append(extra[lags:])
    return np.column_stack(cols), y[lags:]


def dq_test(hits: np.ndarray, var: np.ndarray, lags: int = prereg.DQ_LAGS) -> dict[str, float]:
    """Engle-Manganelli dynamic quantile test on one portfolio: regress Hit - alpha on a
    constant, `lags` lags of (Hit - alpha) and VaR_t; stat = b'X'Xb / (alpha (1 - alpha)) ~
    chi2(1 + lags + 1) under correct conditional coverage."""
    y = np.asarray(hits, dtype=float) - ALPHA
    X, resp = _lagged_design(y, np.asarray(var, dtype=float), lags)
    beta = np.linalg.lstsq(X, resp, rcond=None)[0]
    stat = float(beta @ (X.T @ X) @ beta / (ALPHA * (1 - ALPHA)))
    df = X.shape[1]
    return {"stat": stat, "p": float(stats.chi2.sf(stat, df)), "df": df}


def dq_date_level(e: np.ndarray, var_mean: np.ndarray,
                  lags: int = prereg.DQ_LAGS) -> dict[str, float]:
    """Date-level DQ test: e_t = mean_i Hit_it - alpha regressed on a constant, e_{t-1..t-lags}
    and the cross-sectional mean VaR_t, with a Newey-West Bartlett sandwich variance (lags =
    `default_lags(T, 1)`) and a Wald chi2(1 + lags + 1) statistic."""
    e = np.asarray(e, dtype=float)
    X, y = _lagged_design(e, np.asarray(var_mean, dtype=float), lags)
    T = len(y)
    XtXi = np.linalg.inv(X.T @ X / T)
    beta = XtXi @ (X.T @ y / T)
    g = X * (y - X @ beta)[:, None]
    nw = default_lags(T, 1)
    S = g.T @ g / T
    for k in range(1, nw + 1):
        G = g[k:].T @ g[:-k] / T
        S += (1 - k / (nw + 1)) * (G + G.T)
    V = XtXi @ S @ XtXi / T
    wald = float(beta @ np.linalg.solve(V, beta))
    df = X.shape[1]
    return {"stat": wald, "p": float(stats.chi2.sf(wald, df)), "df": df, "T": T,
            "nw_lags": nw}


def _per_unit(g: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
    """(hits, var) arrays per portfolio in date order for one model's VaR rows."""
    g = g.sort_values(["pid", "date"])
    hit = (g["y_true"].to_numpy() < g["y_pred"].to_numpy()).astype(int)
    var = g["y_pred"].to_numpy()
    _, start = np.unique(g["pid"].to_numpy(), return_index=True)
    return list(zip(np.split(hit, start[1:]), np.split(var, start[1:]), strict=True))


def var_tests_table(preds: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """All Tier A VaR tests for every VaR model (calibrated variants and combos included).

    Returns "portfolio" (share of portfolios rejecting at 5%), "pooled" (per fold, descriptive),
    and "datelevel" (inferential date-level DQ with Holm-adjusted p across models).
    """
    v = preds[preds["target"] == "var"]
    port, pooled, date_rows = [], [], []
    for model, g in v.groupby("model", sort=False):
        units = _per_unit(g)
        ch = [christoffersen(h) for h, _ in units]
        dq = [dq_test(h, q) if h.sum() >= prereg.MIN_HITS else {"p": np.nan} for h, q in units]
        pi = np.array([c["p_ind"] for c in ch])
        pc = np.array([c["p_cc"] for c in ch])
        pq = np.array([d["p"] for d in dq])
        ok = ~np.isnan(pi)
        level = prereg.TEST_LEVEL
        k_ind = k_cc = n_folds = 0
        for fold, gf in g.groupby("fold", sort=True):
            r = christoffersen_pooled([h for h, _ in _per_unit(gf)])
            pooled.append({"model": model, "fold": fold, **r})
            n_folds += 1
            k_ind += int(r["p_ind"] < level)
            k_cc += int(r["p_cc"] < level)
        port.append({"model": model, "n_portfolios": len(units), "n_testable": int(ok.sum()),
                     "n_not_testable": int((~ok).sum()),
                     "share_reject_ind": float(np.mean(pi[ok] < level)) if ok.any() else np.nan,
                     "share_reject_cc": float(np.mean(pc[ok] < level)) if ok.any() else np.nan,
                     "share_reject_dq": float(np.nanmean(pq < level)) if ok.any() else np.nan,
                     "pooled_ind_reject_folds": f"{k_ind}/{n_folds}",
                     "pooled_cc_reject_folds": f"{k_cc}/{n_folds}"})
        d = g.assign(hit=(g["y_true"] < g["y_pred"]).astype(float)).groupby("date").agg(
            hit=("hit", "mean"), var=("y_pred", "mean")).sort_index()
        r = dq_date_level((d["hit"] - ALPHA).to_numpy(), d["var"].to_numpy())
        date_rows.append({"model": model, **r})
    date_level = pd.DataFrame(date_rows)
    date_level["p_holm"] = holm(date_level["p"].to_numpy())
    return {"portfolio": pd.DataFrame(port), "pooled": pd.DataFrame(pooled),
            "datelevel": date_level}
