"""Simulated drawdown probability (Round 4, B3).

`simulate_dd_prob` turns a GARCH(1,1) fit into P(max drawdown over the next H days >= 5%): paths
are simulated from the fit's recursion with innovations bootstrapped from its pre-start
standardised residuals. One uniform matrix per refresh year is shared by every portfolio and
origin (common random numbers), so a typed portfolio gets exactly the numbers of a stored one.
`SimDDModel` wraps the precomputed probability (panel column `sim_dd_raw`) as a RiskModel, with
an optional unpenalised logistic recalibration fit on training rows only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import config
from src.models import prereg
from src.models.base import empty_preds


def uniform_matrix(year: int, n_paths: int = prereg.SIM_PATHS, h: int = config.H) -> np.ndarray:
    """The (n_paths, h) uniform draws of refresh year `year` (default_rng([SIM_SEED, year]))."""
    return np.random.default_rng([prereg.SIM_SEED, int(year)]).random((n_paths, h))


def simulate_dd_prob(sigma1: np.ndarray, mu: float, omega: float, alpha: float, beta: float,
                     z: np.ndarray, U: np.ndarray, thresh: float = config.DD_THRESH) -> np.ndarray:
    """P(max drawdown >= `thresh` within U.shape[1] days) for each origin in `sigma1`.

    `sigma1[o]` is the one-step-ahead daily sigma at origin o (return units); returns follow
    r_k = mu + sigma_k z_k and sigma_{k+1}^2 = omega + alpha (sigma_k z_k)^2 + beta sigma_k^2
    with z drawn from the residual pool `z` by the uniforms in `U` (the same draws for every
    origin). The wealth path starts at 1 (the first peak), as in `targets.max_drawdown_rows`.
    """
    sigma1 = np.atleast_1d(np.asarray(sigma1, dtype=float))
    z = np.asarray(z, dtype=float)
    draws = z[np.minimum((U * len(z)).astype(int), len(z) - 1)]  # (n_paths, h)
    n_o, n_p = len(sigma1), U.shape[0]
    sig2 = np.repeat(sigma1[:, None] ** 2, n_p, axis=1)
    w = np.ones((n_o, n_p))
    peak = np.ones((n_o, n_p))
    mdd = np.zeros((n_o, n_p))
    for k in range(U.shape[1]):
        eps = np.sqrt(sig2) * draws[:, k][None, :]
        w = w * (1.0 + mu + eps)
        peak = np.maximum(peak, w)
        mdd = np.minimum(mdd, w / peak - 1.0)
        sig2 = omega + alpha * eps**2 + beta * sig2
    return (mdd <= -thresh).mean(axis=1)


def ewma_recursion() -> dict[str, float]:
    """Parameters of the EWMA (lambda .94) variance recursion, the fallback for a failed fit."""
    return {"mu": 0.0, "omega": 0.0, "alpha": 0.06, "beta": 0.94}


def _logit(p: np.ndarray, n_paths: int = prereg.SIM_PATHS) -> np.ndarray:
    eps = 1.0 / (2 * n_paths)
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return np.log(p / (1 - p))


class SimDDModel:
    """Drawdown probability from simulated paths: `fhs_mc`, or recalibrated `fhs_mc_logit`.

    Reads the panel column `sim_dd_raw` (precomputed from causal fits, never a model input).
    With `recalibrate`, `fit` estimates dd_flag ~ a + b * logit(p) by unpenalised logistic
    regression on the training rows it is given (fold.train minus held-out portfolios).
    Drawdown head only; the other heads are NaN.
    """

    def __init__(self, name: str, recalibrate: bool):
        self.name, self.recalibrate = name, recalibrate
        self.a, self.b = 0.0, 1.0

    def fit(self, train: pd.DataFrame, val: pd.DataFrame | None = None) -> None:
        """Fit the logistic recalibration on `train` (no-op without recalibration)."""
        if not self.recalibrate:
            return
        from sklearn.linear_model import LogisticRegression

        lr = LogisticRegression(C=np.inf, max_iter=1000)  # C=inf: no regularisation
        lr.fit(_logit(train["sim_dd_raw"].to_numpy())[:, None], train["dd_flag"].to_numpy())
        self.a, self.b = float(lr.intercept_[0]), float(lr.coef_[0, 0])

    def calibrate(self, p: np.ndarray) -> np.ndarray:
        """Apply the stored calibration to raw simulated probabilities."""
        p = np.asarray(p, dtype=float)
        if not self.recalibrate:
            return p
        return 1.0 / (1.0 + np.exp(-(self.a + self.b * _logit(p))))

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """dd_hat for panel rows from their `sim_dd_raw`."""
        out = empty_preds(X.index)
        out["dd_hat"] = self.calibrate(X["sim_dd_raw"].to_numpy())
        return out
