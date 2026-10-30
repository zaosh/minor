"""XGBoost: volatility, drawdown probability and a quantile-loss VaR."""

from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb

from src import config
from src.models.base import empty_preds, feature_cols, var_scale


def _scale(df: pd.DataFrame) -> np.ndarray:
    """Daily vol scale (EWMA) used to standardise the VaR target: q = scale * f(x)."""
    return var_scale(df["ewma_vol"].to_numpy())


class XGBModel:
    """Gradient-boosted trees; each head's round count is the minimum of its validation curve.

    Round 2 used patience-30 early stopping, which on the noisy quantile curve (only ~50
    independent validation dates) could stop after a handful of rounds. Now every head trains
    the full `n_estimators` rounds with the validation block only recording the curve, and the
    refit uses the argmin round. `n_best`, `at_cap` and `val_curves` are kept as diagnostics.
    """

    name = "xgboost"

    def __init__(self, n_estimators: int = 600, seed: int = config.SEED):
        common = dict(n_estimators=n_estimators, learning_rate=0.03, max_depth=4, subsample=0.7,
                      colsample_bytree=0.7, min_child_weight=500, tree_method="hist",
                      early_stopping_rounds=None, random_state=seed, n_jobs=8)
        self._n_estimators = n_estimators
        self.reg = xgb.XGBRegressor(objective="reg:squarederror", **common)
        self.clf = xgb.XGBClassifier(objective="binary:logistic", **common)
        self.qr = xgb.XGBRegressor(objective="reg:quantileerror",
                                   quantile_alpha=config.VAR_ALPHA, **common)

    def _heads(self, df: pd.DataFrame) -> dict:
        return {"reg": (self.reg, df["y_logratio"]), "clf": (self.clf, df["dd_flag"]),
                "qr": (self.qr, df["r_next"] / _scale(df))}

    def fit(self, train: pd.DataFrame, val: pd.DataFrame | None = None) -> None:
        """Fit the three heads for the full round budget and record each validation curve."""
        assert val is not None, "XGBModel needs a validation block to choose the round count"
        self.cols = feature_cols(train)
        X, Xv = (d[self.cols].to_numpy(dtype=np.float32) for d in (train, val))
        self.val_curves: dict[str, list[float]] = {}
        for name, (head, y) in self._heads(train).items():
            head.set_params(n_estimators=self._n_estimators)  # undo a previous refit()
            head.fit(X, y, eval_set=[(Xv, self._heads(val)[name][1])], verbose=False)
            self.val_curves[name] = [float(v) for v in
                                     next(iter(head.evals_result()["validation_0"].values()))]
        self.n_best = {k: int(np.argmin(v)) + 1 for k, v in self.val_curves.items()}
        self.at_cap = {k: n >= self._n_estimators for k, n in self.n_best.items()}

    def refit(self, train: pd.DataFrame, rounds: dict[str, int] | None = None) -> None:
        """Refit each head on `train` for its argmin round count (no rescale for the larger set).

        `rounds` overrides the count for the named heads (used only by the deployed model).
        """
        X = train[self.cols].to_numpy(dtype=np.float32)
        use = {**self.n_best, **(rounds or {})}
        self.n_rounds = {}
        for name, (head, y) in self._heads(train).items():
            head.set_params(n_estimators=use[name])
            head.fit(X, y, verbose=False)
            self.n_rounds[name] = use[name]

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Predict vol_hat, dd_hat and var_hat (5% return quantile)."""
        out = empty_preds(X.index)
        arr = X[self.cols].to_numpy(dtype=np.float32)
        out["vol_hat"] = X["naive_vol"].to_numpy() * np.exp(self.reg.predict(arr))
        out["dd_hat"] = self.clf.predict_proba(arr)[:, 1]
        out["var_hat"] = self.qr.predict(arr) * _scale(X)
        return out
