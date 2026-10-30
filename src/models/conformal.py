"""Online (adaptive) conformal calibration of 1-day VaR forecasts, numpy only.

A VaR forecast q (a negative return) is shifted by a scalar offset c measured in units of the
daily vol scale s = var_scale(ewma_vol):  q_cal = q - c * s.  After each calibration date the
offset follows the quantile-tracking recursion of Angelopoulos, Candes and Tibshirani (2023):

    c <- c + eta * (err - alpha),   err = share of portfolios whose return fell below q_cal.

Too many exceedances (err > alpha) push c up (a deeper VaR); too few pull it down. The offset
used for a prediction at day t depends only on dates t' with day(t') + lag <= day(t), because the
next-day return of t' is only known `lag` trading days later. Held-out portfolios never enter
`err`, so they stay out-of-sample for the calibration as well as for training.

Guarantee: summing the recursion gives sum(err_t - alpha) = (c_end - c_start) / eta, so the
long-run average miscoverage is bounded by the drift of c divided by eta*T. That says nothing
about any single year, portfolio or market regime, and the delayed feedback and the dependence
between portfolios make it approximate. It is a calibration aid, not a coverage guarantee.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import config

ETA = 0.25  # step size: set by reasoning (about 0.25 sd of correction per half-year), not tuned
LAG = 1  # trading days until a date's next-day return is known


def tracker_states(
    day: np.ndarray, err_fn, eta: float = ETA, lag: int = LAG, alpha: float = config.VAR_ALPHA
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the recursion over sorted unique calibration days.

    `err_fn(i, c)` returns the exceedance share at day[i] when the offset used there was c.
    Returns (offset_used, err, c_after): the offset applied at each day, the realised share, and
    the state after absorbing that day's error (a day is absorbed once a later day is >= lag
    trading days ahead, and all remaining days are absorbed at the end).
    """
    n = len(day)
    used, err, after = np.zeros(n), np.zeros(n), np.zeros(n)
    c, j = 0.0, 0
    for i in range(n):
        while j < i and day[j] + lag <= day[i]:
            c += eta * (err[j] - alpha)
            after[j] = c
            j += 1
        used[i] = c
        err[i] = err_fn(i, c)
    while j < n:
        c += eta * (err[j] - alpha)
        after[j] = c
        j += 1
    return used, err, after


def calibrate_var(
    day: np.ndarray, y: np.ndarray, q: np.ndarray, scale: np.ndarray, calib: np.ndarray,
    eta: float = ETA, lag: int = LAG
) -> tuple[np.ndarray, pd.DataFrame]:
    """Calibrate one model's VaR rows.

    `day`, `y` (next-day return), `q` (raw VaR), `scale` are per-row; `calib` is a boolean mask
    of rows that may feed the recursion (non-held-out portfolios). Returns the calibrated VaR
    per row and a per-day table (day_idx, offset, err, c_after) that `offset_at` can replay.
    """
    days = np.unique(day)
    rows = [np.flatnonzero((day == d) & calib) for d in days]

    def err_fn(i: int, c: float) -> float:
        r = rows[i]
        return float(np.mean(y[r] < q[r] - c * scale[r])) if len(r) else 0.0

    used, err, after = tracker_states(days, err_fn, eta, lag)
    table = pd.DataFrame({"day_idx": days, "offset": used, "err": err, "c_after": after})
    off = table.set_index("day_idx")["offset"].reindex(day).to_numpy()
    return q - off * scale, table


def offset_at(table: pd.DataFrame, day_idx: int, lag: int = LAG) -> float:
    """Offset a prediction made at `day_idx` would use: the state after every calibration day
    d with d + lag <= day_idx (0 before any). Reproduces `calibrate_var`'s offsets on its grid."""
    k = int(np.searchsorted(table["day_idx"].to_numpy() + lag, day_idx, side="right"))
    return 0.0 if k == 0 else float(table["c_after"].iloc[k - 1])


def drift_summary(table: pd.DataFrame, eta: float = ETA,
                  alpha: float = config.VAR_ALPHA) -> dict[str, float]:
    """Average exceedance share of the calibrated VaR and the identity-based drift bound.

    mean(err) - alpha equals (c_end - c_start)/(eta*T) exactly, so a small value says the
    long-run average is calibrated, nothing more.
    """
    T = len(table)
    return {"mean_err": float(table["err"].mean()),
            "drift_bound": float(table["c_after"].iloc[-1] / (eta * T)), "n_dates": T,
            "alpha": alpha}
