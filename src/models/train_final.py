"""Fit RF, XGBoost and LSTM on all labelled rows and save artifacts for the app."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src import config
from src.features import model_features
from src.models import evaluate, prereg
from src.models.base import feature_cols
from src.models.lstm import STATIC_R4, LSTMModel
from src.models.rf import RFModel
from src.models.xgb import XGBModel


def fold_median_rounds(folds_dir: Path, head: str) -> int:
    """Median XGBoost round count chosen for `head` by the saved walk-forward fold models.

    The deployed VaR head uses this instead of its own single 252-day validation curve: in
    Round 4 that curve's minimum was at round 2, so the head learned almost nothing.
    """
    paths = sorted(folds_dir.glob("*/xgboost.joblib"))
    if not paths:
        raise FileNotFoundError(f"no fold models in {folds_dir}; run python -m src.models.evaluate")
    return int(np.median([joblib.load(p).n_rounds[head] for p in paths]))


def main() -> None:
    """Train the deployed models on every labelled panel row (all portfolios) and save them.

    As in the walk-forward folds: early stopping uses the last 252 days (purged from the rest),
    then XGBoost and the LSTM are refit on all rows with the stopping point that was chosen.
    """
    config.set_seeds()
    panel = pd.read_parquet(config.DATA_PROCESSED / "panel.parquet")
    dates = pd.read_parquet(config.DATA_PROCESSED / "returns.parquet").index
    feats = pd.read_parquet(config.DATA_PROCESSED / "model_features.parquet")
    panel = model_features.add_model_features(panel, dates, feats)  # all portfolios (no holdout)
    assert set(prereg.MODEL_FEATURES) <= set(panel.columns), "Round 4 model features missing"
    har_coefs = model_features.har_coefficients(panel, dates)
    points = model_features.refresh_points(
        panel["day_idx"].to_numpy().astype(int),
        dates[panel["day_idx"].to_numpy().astype(int)].year.to_numpy())
    last_y = max(points)  # the refresh that serves the newest rows (deployed-model inputs)
    day = panel["day_idx"].to_numpy().astype(int)
    val_start = day.max() - evaluate.INNER_VAL_DAYS
    val = panel[day > val_start]
    fit = panel[day + config.H < val["day_idx"].min()]
    config.ARTIFACTS.mkdir(parents=True, exist_ok=True)
    rf, xg, ls = RFModel(), XGBModel(), LSTMModel(static_cols=STATIC_R4)
    for m in (rf, ls):
        evaluate.fit_model(m, fit, val, panel)
    qr_rounds = fold_median_rounds(config.ARTIFACTS / "folds", "qr")
    xg.fit(fit, val)
    xg.refit(panel, rounds={"qr": qr_rounds})
    joblib.dump(rf, config.ARTIFACTS / "rf.joblib")
    joblib.dump(xg, config.ARTIFACTS / "xgboost.joblib")
    ls.save(config.ARTIFACTS / "lstm.pt")
    last_idx = int(panel["day_idx"].max())
    q = panel["vol_fwd"].quantile([0.25, 0.5, 0.75]).tolist()  # risk-band cut points
    meta = {
        "feature_cols": feature_cols(panel),
        "model_features": list(prereg.MODEL_FEATURES),
        "har_feature_coef": [float(c) for c in har_coefs[last_y]],
        "feature_refresh_day_idx": int(points[last_y][0]),
        "feature_refresh_date": str(dates[points[last_y][0]].date()),
        "train_end_day_idx": last_idx,
        "train_end_date": str(panel.index.get_level_values("date").max().date()),
        "label_end_date": str(dates[last_idx + config.H].date()),  # last return used by a label
        "n_fit_rows": int(len(panel)),
        "n_rounds": xg.n_rounds,
        "n_rounds_val_argmin": xg.n_best,
        "qr_rounds_source": "median of walk-forward fold models",
        "best_epochs": dict(zip(("vol", "dd", "var"), ls.best_epochs, strict=True)),
        "xgboost_at_cap": xg.at_cap,
        "vol_quartiles": q,
        "seed": config.SEED,
        "horizon": config.H,
    }
    digest = hashlib.sha1((config.ARTIFACTS / "lstm.pt").read_bytes()).hexdigest()[:8]
    meta["model_version"] = f"{meta['label_end_date']}-{digest}"
    (config.ARTIFACTS / "meta.json").write_text(json.dumps(meta, indent=2))
    print("saved", sorted(p.name for p in config.ARTIFACTS.iterdir()), len(panel), "rows")


if __name__ == "__main__":
    main()
