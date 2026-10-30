"""Common interface and column groups shared by every model and baseline."""

from __future__ import annotations

from typing import Protocol

import numpy as np
import pandas as pd

from src.models import prereg

TARGET_COLS = ["vol_fwd", "y_logratio", "dd_flag", "r_next"]
BASELINE_COLS = ["naive_vol", "hist_var250", "param_var63", "lw_vol"]
META_COLS = ["day_idx"]
# Round 4 (B2): HAR and GARCH forecasts are input features; the simulated drawdown probability
# and the GARCH fallback flag ride along in the panel but are never model inputs.
MODEL_FEATURES = list(prereg.MODEL_FEATURES)
NONFEATURE_COLS = ["sim_dd_raw", "garch_fallback"]
PRED_COLS = ["vol_hat", "dd_hat", "var_hat", "es_hat"]
# evaluation target name -> (panel column holding the realisation, prediction column)
TARGETS = {"vol": ("vol_fwd", "vol_hat"), "dd": ("dd_flag", "dd_hat"), "var": ("r_next", "var_hat"),
           "es": ("r_next", "es_hat")}


def var_scale(ewma_vol) -> np.ndarray:
    """Daily vol scale used to standardise the VaR target: q = scale * f(x)."""
    return np.maximum(np.asarray(ewma_vol, dtype=float), 1e-3) / np.sqrt(252)


def feature_cols(panel: pd.DataFrame) -> list[str]:
    """Model input columns: everything except targets, non-feature baselines and metadata."""
    skip = set(TARGET_COLS + BASELINE_COLS + META_COLS + NONFEATURE_COLS)
    return [c for c in panel.columns if c not in skip]


class RiskModel(Protocol):
    """fit() on training rows; predict() returns vol_hat, dd_hat, var_hat on X's index.

    Unsupported targets are NaN. `vol_hat` is annualized vol, `dd_hat` a probability and
    `var_hat` the 5% quantile of the next-day return (a negative number) and `es_hat` the mean
    return beyond it (expected shortfall, more negative than `var_hat`).
    """

    name: str

    def fit(self, train: pd.DataFrame, val: pd.DataFrame | None = None) -> None: ...

    def predict(self, X: pd.DataFrame) -> pd.DataFrame: ...

    # Optional: models that need an inner validation block (early stopping) may define
    #     def refit(self, train: pd.DataFrame) -> None
    # which retrains on fit+val with the stopping point chosen in fit(). See evaluate.fit_model.


def empty_preds(index: pd.Index) -> pd.DataFrame:
    """A NaN prediction frame with the standard columns."""
    return pd.DataFrame(np.nan, index=index, columns=PRED_COLS)
