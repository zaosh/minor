"""Random Forest for volatility (log-ratio target) and drawdown probability; no VaR head."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from src import config
from src.models.base import empty_preds, feature_cols


class RFModel:
    """Random Forest for `vol_hat` (via log(vol_fwd/vol_21)) and `dd_hat`."""

    name = "rf"

    def __init__(self, n_estimators: int = 200, seed: int = config.SEED):
        common = dict(n_estimators=n_estimators, min_samples_leaf=2000, max_samples=0.3,
                      max_features=0.5, n_jobs=-1, random_state=seed)
        self.reg = RandomForestRegressor(**common)
        self.clf = RandomForestClassifier(**common)

    def fit(self, train: pd.DataFrame, val: pd.DataFrame | None = None) -> None:
        """Fit both forests on the training rows (val is unused: no early stopping)."""
        self.cols = feature_cols(train)
        X = train[self.cols].to_numpy(dtype=np.float32)
        self.reg.fit(X, train["y_logratio"].to_numpy())
        self.clf.fit(X, train["dd_flag"].to_numpy())

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Predict vol (back-transformed to the vol_fwd scale) and drawdown probability."""
        out = empty_preds(X.index)
        arr = X[self.cols].to_numpy(dtype=np.float32)
        out["vol_hat"] = X["naive_vol"].to_numpy() * np.exp(self.reg.predict(arr))
        out["dd_hat"] = self.clf.predict_proba(arr)[:, 1]
        return out
