"""Point-in-time backtests: what the walk-forward models predicted for a past date.

Every function here uses only the per-fold models (trained on data before that year's purge
cut-off) and the saved walk-forward predictions. The deployed models are never used, because
they have seen all labelled data.
"""

from __future__ import annotations

import json
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from src import config, predict
from src.features import targets
from src.models import baselines, combos, conformal, prereg
from src.models.base import var_scale
from src.models.lstm import LSTMModel
from src.models.simdd import SimDDModel
from src.portfolios import Portfolio, build_user_portfolio, portfolio_returns

FOLD_DIR = config.ARTIFACTS / "folds"
ML_MODELS = ("rf", "xgboost", "lstm")


def load_folds(path=None) -> list[dict]:
    """Fold metadata written by `python -m src.models.evaluate` (reports/folds.json)."""
    path = path or config.REPORTS / "folds.json"
    if not path.exists():
        raise FileNotFoundError("Missing reports/folds.json; run python -m src.models.evaluate")
    return json.loads(path.read_text())


def fold_for_date(d: pd.Timestamp, folds: list[dict]) -> dict:
    """The fold whose test period covers calendar date `d` (the last fold runs to the end).

    Raises ValueError for dates before the first test period.
    """
    d = pd.Timestamp(d)
    if d < pd.Timestamp(folds[0]["test_start"]):
        raise ValueError(f"No backtest before {folds[0]['test_start']} (first test period).")
    eligible = [f for f in folds if int(f["name"]) <= d.year]
    return eligible[-1]


def _row_models(pred: pd.DataFrame) -> dict[str, dict]:
    """{model: {vol_hat, dd_hat, var_hat}} from the long prediction rows of one (pid, date)."""
    cols = {"vol": "vol_hat", "dd": "dd_hat", "var": "var_hat", "es": "es_hat"}
    out: dict[str, dict] = {}
    for r in pred.itertuples():
        out.setdefault(r.model, {})[cols[r.target]] = float(r.y_pred)
    return out


def stored_predictions(pid: int, path=None) -> pd.DataFrame:
    """All saved walk-forward predictions of one portfolio (pid filter pushed into parquet)."""
    path = path or config.REPORTS / "predictions.parquet"
    if not path.exists():
        raise FileNotFoundError("Missing reports/predictions.parquet; run the evaluation first")
    return pd.read_parquet(path, filters=[("pid", "==", int(pid))])


def stored_backtest(
    pid: int,
    as_of: pd.Timestamp,
    folds: list[dict] | None = None,
    preds: pd.DataFrame | None = None,
) -> dict:
    """Saved walk-forward predictions at the last grid date <= as_of for stored portfolio.

    Raises ValueError if no grid date exists on or before `as_of`, and RuntimeError if a row
    came from a model whose training period reaches the date (a leakage alarm).
    """
    folds = folds or load_folds()
    preds = stored_predictions(pid) if preds is None else preds
    dates = pd.DatetimeIndex(sorted(pd.to_datetime(preds["date"]).unique()))
    ok = dates[dates <= pd.Timestamp(as_of)]
    if len(ok) == 0:
        raise ValueError(f"No saved prediction on or before {pd.Timestamp(as_of):%Y-%m-%d}.")
    date = ok[-1]
    rows = preds[pd.to_datetime(preds["date"]) == date]
    fold = fold_for_date(date, folds)
    if not (rows["fold"].astype(str) == fold["name"]).all():
        raise RuntimeError("Leakage alarm: prediction came from the wrong fold")
    if not pd.Timestamp(fold["train_end"]) < date:
        raise RuntimeError("Leakage alarm: training period reaches the prediction date")
    return {"as_of": date, "fold": fold, "models": _row_models(rows)}


def stored_history(pid: int, preds: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per grid date: realized vol / next-day return and each model's predictions (wide)."""
    preds = stored_predictions(pid) if preds is None else preds
    d = preds.assign(date=pd.to_datetime(preds["date"]))
    wide = d.pivot_table(index="date", columns=["target", "model"], values="y_pred")
    real_vol = d[d["target"] == "vol"].groupby("date")["y_true"].first().rename("realized_vol")
    real_ret = d[d["target"] == "var"].groupby("date")["y_true"].first().rename("next_return")
    out = pd.concat([real_vol, real_ret], axis=1)
    for target, col in (("vol", "vol_hat"), ("var", "var_hat")):
        for model in wide[target].columns:
            out[f"{col}:{model}"] = wide[target][model]
    return out


def realized(p: Portfolio, as_of: pd.Timestamp) -> dict:
    """What actually happened after `as_of`; values are NaN while the window is incomplete."""
    r = portfolio_returns(predict.load_data()["returns"], p)
    pos = r.index.get_loc(pd.Timestamp(as_of))
    n_after = len(r) - 1 - pos
    h = config.H
    mdd = (float(targets.max_drawdown_rows(r.to_numpy()[pos : pos + h + 1], h)[0])
           if n_after >= h else np.nan)
    return {
        "vol_fwd": float(targets.forward_vol(r).iloc[pos]),
        "dd_flag": float(targets.forward_dd_flag(r).iloc[pos]),
        "max_dd": mdd,
        "r_next": float(targets.next_return(r).iloc[pos]),
        "days_after": n_after,
    }


@lru_cache(maxsize=8)
def load_fold_models(name: str) -> dict:
    """Per-fold ML models plus baseline parameters for fold `name`."""
    f = FOLD_DIR / name
    if not (f / "baselines.json").exists():
        raise FileNotFoundError(f"Missing {f}; run python -m src.models.evaluate")
    return {"rf": joblib.load(f / "rf.joblib"), "xgboost": joblib.load(f / "xgboost.joblib"),
            "lstm": LSTMModel.load(f / "lstm.pt"),
            "baselines": json.loads((f / "baselines.json").read_text())}


def custom_backtest(
    tickers: list[str], weights: list[float], as_of: pd.Timestamp, folds: list[dict] | None = None
) -> dict:
    """Predict a typed portfolio as of a past date with that year's walk-forward models.

    Features use data up to `as_of` only; GARCH parameters are fit on rows before the fold's
    first test day, exactly as in the evaluation.
    """
    folds = folds or load_folds()
    p = build_user_portfolio(tickers, weights, predict.available_tickers())
    date = predict._resolve_date(as_of)
    fold = fold_for_date(date, folds)
    models = load_fold_models(fold["name"])
    fit_end = fold["first_test_day_idx"]
    res = predict._predict_with(models, p, date, garch_fit_end=fit_end,
                                har_coef=models["baselines"]["har_coef"], feature_fit_end=fit_end)
    feats = res["features"]
    har = baselines.HAR()
    har.coef_ = np.array(models["baselines"]["har_coef"])
    hp = har.predict(feats).iloc[0]
    base = res["baselines"]
    base["har"] = {k: float(hp[k]) for k in ("vol_hat", "dd_hat", "var_hat", "es_hat")}
    base["climatology"] = {"dd_hat": float(models["baselines"]["climatology_rate"])}
    if "fhs_mc_logit" in models["baselines"]:  # fold artifacts from before Round 4 lack it
        sim = SimDDModel("fhs_mc_logit", True)
        sim.a, sim.b = models["baselines"]["fhs_mc_logit"]
        raw = SimDDModel("fhs_mc", False)
        base["fhs_mc"] = {"dd_hat": float(raw.predict(feats)["dd_hat"].iloc[0])}
        base["fhs_mc_logit"] = {"dd_hat": float(sim.predict(feats)["dd_hat"].iloc[0])}
    out = {**res["ml"], **base}
    for name, members in prereg.COMBOS.items():
        out[name] = combos.combine_models(out, members)
    out.update(calibrated(out, feats, int(feats["day_idx"].iloc[0])))
    return {"as_of": date, "fold": fold, "portfolio": p, "models": out,
            "warnings": res["warnings"]}


@lru_cache(maxsize=1)
def load_offsets(path=None) -> dict[str, pd.DataFrame]:
    """Conformal offset tables per model from reports/conformal_offsets.csv ({} if missing)."""
    path = path or config.REPORTS / "conformal_offsets.csv"
    if not path.exists():
        return {}
    t = pd.read_csv(path)
    return {m: g.drop(columns="model").reset_index(drop=True) for m, g in t.groupby("model")}


def calibrated(models: dict[str, dict], feats: pd.DataFrame, day_idx: int) -> dict[str, dict]:
    """`{model}_cal` VaR entries for every model with a VaR forecast and a saved offset table.

    The offset is replayed from the walk-forward stream at `day_idx` (only dates whose next-day
    return was already known are used), so it matches the stored `{model}_cal` rows.
    """
    scale = float(var_scale(feats["ewma_vol"].to_numpy())[0])
    out = {}
    for name, tab in load_offsets().items():
        v = models.get(name, {}).get("var_hat", np.nan)
        if pd.notna(v):
            out[f"{name}_cal"] = {"var_hat": float(v) - conformal.offset_at(tab, day_idx) * scale}
    return out


def backtest_table(models: dict[str, dict], real: dict) -> pd.DataFrame:
    """One row per model: predictions next to what happened (percent-style raw floats)."""
    rows = []
    for name, v in models.items():
        var = v.get("var_hat", np.nan)
        rows.append({
            "model": name,
            "vol_pred": v.get("vol_hat", np.nan), "vol_real": real["vol_fwd"],
            "p_drawdown": v.get("dd_hat", np.nan), "drawdown_occurred": real["dd_flag"],
            "max_dd": real["max_dd"],
            "var_pred": var, "es_pred": v.get("es_hat", np.nan), "next_return": real["r_next"],
            "var_exceeded": (bool(real["r_next"] < var) if pd.notna(var) else None),
        })
    return pd.DataFrame(rows).set_index("model")


def stored_portfolios() -> list[dict]:
    """The 400 training portfolios: id, label (top holdings), held-out flag, tickers, weights."""
    ports = json.loads((config.DATA_PROCESSED / "portfolios.json").read_text())
    hp = config.REPORTS / "holdout_pids.json"
    held = set(json.loads(hp.read_text())) if hp.exists() else set()
    out = []
    for pid, q in enumerate(ports):
        top = sorted(zip(q["tickers"], q["weights"], strict=True), key=lambda x: -x[1])[:3]
        label = ", ".join(f"{t.removesuffix('.NS')} {w:.0%}" for t, w in top)
        if len(q["tickers"]) > 3:
            label += f" (+{len(q['tickers']) - 3})"
        out.append({"pid": pid, "label": label, "heldout": pid in held,
                    "tickers": q["tickers"], "weights": q["weights"]})
    return out
