"""Stress versus calm regimes from India VIX (pre-registered rule, see HANDOVER Round 4).

A date is "stress" if VIX_t (the India VIX close on the forecast date, known at that close)
exceeds the median of daily VIX from `prereg.VIX_START` to the fold's `train_end`, fixed for the
fold's test period. A second, calendar view puts the folds in `prereg.STRESS_FOLDS` against the
rest. Regime metrics are pooled over rows (a fold can have no stress dates at all).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.models import evaluate, prereg, significance

PRIMARY = {"vol": "se", "dd": "se", "var": "se"}  # primary DM losses: vol-SE, dd-Brier, pinball
VIEWS = {"vix": "stress_vix", "calendar": "stress_cal"}


def vix_thresholds(vix: pd.Series, folds: list[dict]) -> dict[str, float]:
    """Median daily VIX from `prereg.VIX_START` to each fold's train_end (nothing later)."""
    return {f["name"]: float(vix.loc[prereg.VIX_START : f["train_end"]].median()) for f in folds}


def label_dates(fold_by_date: pd.Series, vix: pd.Series,
                thresholds: dict[str, float]) -> pd.DataFrame:
    """Per test date: fold, VIX_t, threshold and the two stress flags (`stress_vix`,
    `stress_cal`). `fold_by_date` is indexed by date. A date without VIX raises ValueError."""
    d = pd.DataFrame({"fold": fold_by_date.astype(str)})
    d["vix"] = vix.reindex(d.index)
    if d["vix"].isna().any():
        raise ValueError("India VIX is missing on some test dates")
    d["threshold"] = d["fold"].map(thresholds)
    d["stress_vix"] = d["vix"] > d["threshold"]
    d["stress_cal"] = d["fold"].isin(prereg.STRESS_FOLDS)
    return d


def regime_counts(lab: pd.DataFrame) -> dict[str, tuple[int, int]]:
    """(stress dates, calm dates) for each view."""
    return {v: (int(lab[c].sum()), int((~lab[c]).sum())) for v, c in VIEWS.items()}


def regime_metrics(preds: pd.DataFrame, lab: pd.DataFrame) -> pd.DataFrame:
    """All metrics per (view, regime, model, target), pooled over rows."""
    rows = []
    for view, col in VIEWS.items():
        flag = preds["date"].map(lab[col]).astype(bool)
        for regime, mask in (("stress", flag), ("calm", ~flag)):
            for (model, target), g in preds[mask].groupby(["model", "target"], sort=False):
                s = evaluate.score(target, g["y_true"].to_numpy(), g["y_pred"].to_numpy(),
                                   g["vol_now"].to_numpy(), g["y_aux"].to_numpy())
                rows += [{"view": view, "regime": regime, "model": model, "target": target,
                          "metric": k, "value": v, "n_rows": len(g),
                          "n_dates": g["date"].nunique()} for k, v in s.items()]
    return pd.DataFrame(rows)


def _best_baseline(L: pd.DataFrame, mask: np.ndarray) -> str:
    pool = [c for c in L.columns if not significance.is_ml(c) and not significance.is_combo(c)
            and not c.endswith("_cal")]
    return L.loc[mask, pool].mean().idxmin()


def regime_dm(preds: pd.DataFrame, lab: pd.DataFrame) -> pd.DataFrame:
    """Regime-specific DM tests of each raw ML model against the best raw baseline of the
    same regime, on the primary losses. One row per (view, regime, target, model)."""
    rows = []
    for target, loss in PRIMARY.items():
        L = significance.date_losses(preds, target, loss)
        h = significance.HORIZON_STRIDE[target]
        ml = [c for c in L.columns if c in significance.ML_NAMES]
        for view, col in VIEWS.items():
            stress = lab.loc[L.index, col].to_numpy(dtype=bool)
            best = {"stress": _best_baseline(L, stress), "calm": _best_baseline(L, ~stress)}
            for m in ml:
                for regime in ("stress", "calm"):
                    d = L[m].to_numpy() - L[best[regime]].to_numpy()
                    r = significance.dm_regime(d, stress, h)
                    rows.append({"view": view, "regime": regime, "target": target, "model": m,
                                 "baseline": best[regime], "mean_loss_diff": r[f"{regime}_mean"],
                                 "dm_stat": r[f"{regime}_stat"], "p": r[f"{regime}_p"],
                                 "n_dates": r[f"n_{regime}"], "nw_lags": r["lags"]})
    return pd.DataFrame(rows)


def regime_mcs(preds: pd.DataFrame, lab: pd.DataFrame) -> pd.DataFrame:
    """90% model confidence sets within each regime that has at least
    `prereg.MIN_REGIME_DATES` dates (the moving-block bootstrap is applied to the subset of
    dates in that regime, which are not contiguous: treat as approximate)."""
    out = []
    for target in ("vol", "dd", "var", "es"):
        L = significance.date_losses(preds, target)
        for view, col in VIEWS.items():
            flag = lab.loc[L.index, col].to_numpy(dtype=bool)
            for regime, mask in (("stress", flag), ("calm", ~flag)):
                if mask.sum() >= prereg.MIN_REGIME_DATES:
                    m = significance.mcs(L[mask], prereg.MCS_ALPHA)
                    out.append(m.assign(target=target, view=view, regime=regime,
                                        n_dates=int(mask.sum())))
    cols = ["model", "mcs_p", "in_set", "target", "view", "regime", "n_dates"]
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=cols)


def regime_coverage(preds: pd.DataFrame, lab: pd.DataFrame) -> pd.DataFrame:
    """Descriptive HAC check of VaR exceedance by regime (raw VaR models, combos included):
    the date-level hit rate minus alpha regressed on the two regime dummies."""
    alpha = evaluate.config.VAR_ALPHA
    rows = []
    v = preds[(preds["target"] == "var") & ~preds["model"].str.endswith("_cal")]
    for model, g in v.groupby("model", sort=False):
        e = (pd.Series((g["y_true"] < g["y_pred"]).astype(float).to_numpy(),
                       index=pd.to_datetime(g["date"])).groupby(level=0).mean() - alpha)
        for view, col in VIEWS.items():
            r = significance.dm_regime(e.to_numpy(), lab.loc[e.index, col].to_numpy(dtype=bool), 1)
            for regime in ("stress", "calm"):
                rows.append({"view": view, "regime": regime, "model": model,
                             "exceed_rate": r[f"{regime}_mean"] + alpha,
                             "stat": r[f"{regime}_stat"], "p": r[f"{regime}_p"],
                             "n_dates": r[f"n_{regime}"]})
    return pd.DataFrame(rows)
