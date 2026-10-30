"""Walk-forward evaluation: one set of folds and metrics for every model and baseline."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from src import config
from src.models import conformal, prereg
from src.models.base import TARGETS, RiskModel, var_scale

TEST_YEARS = (2020, 2021, 2022, 2023, 2024, 2025)
INNER_VAL_DAYS = 252
N_HOLDOUT = 80  # portfolios excluded from every training row (P8)


@dataclass
class Fold:
    """Row positions (into the panel) for one walk-forward fold."""

    name: str
    fit: np.ndarray  # training rows used for fitting
    val: np.ndarray  # last INNER_VAL_DAYS of training (early stopping), purged from `fit`
    train: np.ndarray  # fit + val + purge gap rows (everything labelled before the test start)
    test: np.ndarray


def make_folds(
    panel: pd.DataFrame, dates: pd.DatetimeIndex, test_years=TEST_YEARS, h: int = config.H
) -> list[Fold]:
    """Expanding-window folds with an H-day purge between train and test (and fit and val).

    A training row at calendar position d is allowed only if d + h < test_start, so no training
    target window reaches into the test period. The last fold runs to the end of the data.
    """
    day = panel["day_idx"].to_numpy().astype(int)
    year = panel.index.get_level_values("date").year.to_numpy()
    folds = []
    for y in test_years:
        ts = int(np.searchsorted(dates.year.to_numpy(), y))  # first trading day of year y
        is_test = (year == y) if y != test_years[-1] else (year >= y)
        train = np.flatnonzero(day + h < ts)
        if len(train) == 0:
            continue
        val_start = day[train].max() - INNER_VAL_DAYS
        val = train[day[train] > val_start]
        fit = train[day[train] + h < day[val].min()]
        folds.append(Fold(str(y), fit, val, train, np.flatnonzero(is_test)))
    return folds


# ---------------------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------------------
def rmse(y: np.ndarray, p: np.ndarray) -> float:
    """Root mean squared error."""
    return float(np.sqrt(np.mean((y - p) ** 2)))


def mae(y: np.ndarray, p: np.ndarray) -> float:
    """Mean absolute error."""
    return float(np.mean(np.abs(y - p)))


def directional_accuracy(y: np.ndarray, p: np.ndarray, now: np.ndarray) -> float:
    """Share of rows where predicted and actual vol move the same direction vs current vol.

    NaN if the model never predicts a move (e.g. the naive forecast).
    """
    pd_, ad = np.sign(p - now), np.sign(y - now)
    if np.all(pd_ == 0):
        return float("nan")
    return float(np.mean(pd_ == ad))


def variance_losses(y_vol: np.ndarray, p_vol: np.ndarray,
                    floor: float = prereg.VOL_FLOOR) -> tuple[np.ndarray, np.ndarray, int]:
    """Per-row QLIKE and MSE on variance for annualised-vol realisations and forecasts.

    The proxy is s = y_vol**2 (from the 21 daily returns) and the forecast h = p_vol**2.
    QLIKE = s/h - ln(s/h) - 1 and MSE-var = (s - h)**2 are the two losses Patton (2011) shows
    to be robust to a noisy volatility proxy. Both vols are floored at `floor`; the number of
    rows where either was floored is returned. A NaN raises ValueError.
    """
    y, p = np.asarray(y_vol, dtype=float), np.asarray(p_vol, dtype=float)
    if np.isnan(y).any() or np.isnan(p).any():
        raise ValueError("NaN in volatility realisations or forecasts")
    n_floored = int(np.sum((y < floor) | (p < floor)))
    s, h = np.maximum(y, floor) ** 2, np.maximum(p, floor) ** 2
    return s / h - np.log(s / h) - 1.0, (s - h) ** 2, n_floored


def qlike(y_vol: np.ndarray, p_vol: np.ndarray) -> float:
    """Mean QLIKE loss of a volatility forecast (see `variance_losses`)."""
    return float(np.mean(variance_losses(y_vol, p_vol)[0]))


def mse_var(y_vol: np.ndarray, p_vol: np.ndarray) -> float:
    """Mean squared error on variance of a volatility forecast (see `variance_losses`)."""
    return float(np.mean(variance_losses(y_vol, p_vol)[1]))


def pinball(y: np.ndarray, q: np.ndarray, alpha: float = config.VAR_ALPHA) -> float:
    """Mean pinball (quantile) loss of predicted quantile `q` against realisations `y`."""
    d = y - q
    return float(np.mean(np.maximum(alpha * d, (alpha - 1) * d)))


def exceedance_rate(y: np.ndarray, q: np.ndarray) -> float:
    """Fraction of realisations below the predicted VaR quantile (target: alpha)."""
    return float(np.mean(y < q))


def kupiec_pof(n_exceed: int, n: int, p: float = config.VAR_ALPHA) -> tuple[float, float]:
    """Kupiec proportion-of-failures test: returns (LR statistic, p-value), chi2 with 1 dof."""
    x = n_exceed
    pi = x / n

    def ll(prob: float) -> float:
        a = (n - x) * np.log(1 - prob) if x < n else 0.0
        b = x * np.log(prob) if x > 0 else 0.0
        return a + b

    lr = -2 * (ll(p) - ll(pi))
    return float(lr), float(stats.chi2.sf(lr, 1))


def fz0_loss(y: np.ndarray, v: np.ndarray, e: np.ndarray,
             alpha: float = config.VAR_ALPHA) -> float:
    """Mean FZ0 joint (VaR, ES) loss of Patton, Ziegel and Chen (2019); lower is better.

    `y` realised returns, `v` VaR forecasts (negative), `e` ES forecasts (negative, <= v). It is
    strictly consistent for the pair, so it is the one ES ranking score used here.
    """
    return float(np.mean(fz0_rows(y, v, e, alpha)))


def fz0_rows(y: np.ndarray, v: np.ndarray, e: np.ndarray,
             alpha: float = config.VAR_ALPHA) -> np.ndarray:
    """Per-row FZ0 loss (see `fz0_loss`)."""
    e = np.minimum(e, -1e-8)
    hit = (y <= v).astype(float)
    return -hit * (v - y) / (alpha * e) + v / e + np.log(-e) - 1.0


def es_residual(y: np.ndarray, v: np.ndarray, e: np.ndarray) -> float:
    """McNeil-Frey style check: mean of (y - e)/|e| on the VaR exceedances (0 if ES is right,
    negative if ES is too shallow). NaN when there are no exceedances."""
    hit = y <= v
    return float(np.mean((y[hit] - e[hit]) / np.abs(e[hit]))) if hit.any() else float("nan")


def es_ratio(y: np.ndarray, v: np.ndarray, e: np.ndarray) -> float:
    """Mean realised exceedance return over mean predicted ES on those rows (1 if ES is right)."""
    hit = y <= v
    return float(y[hit].mean() / e[hit].mean()) if hit.any() else float("nan")


def score(target: str, y: np.ndarray, p: np.ndarray, now: np.ndarray,
          aux: np.ndarray | None = None) -> dict[str, float]:
    """All metrics for one target on rows with a prediction.

    For the "es" target `p` is the ES forecast and `aux` the same model's VaR forecast.
    """
    if target == "es":
        return {"fz0": fz0_loss(y, aux, p), "es_resid": es_residual(y, aux, p),
                "es_ratio": es_ratio(y, aux, p)}
    if target == "vol":
        return {"rmse": rmse(y, p), "mae": mae(y, p), "dir_acc": directional_accuracy(y, p, now),
                "qlike": qlike(y, p), "mse_var": mse_var(y, p)}
    if target == "dd":
        p = np.clip(p, 1e-6, 1 - 1e-6)
        auc = roc_auc_score(y, p) if len(np.unique(y)) > 1 else float("nan")
        return {"auc": float(auc), "brier": float(brier_score_loss(y, p)),
                "logloss": float(log_loss(y, p, labels=[0, 1]))}
    exc = int(np.sum(y < p))
    lr, pval = kupiec_pof(exc, len(y))
    return {"pinball": pinball(y, p), "exceedance": exc / len(y), "kupiec_p": pval}


# ---------------------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------------------
def holdout_pids(n_portfolios: int, n: int = N_HOLDOUT, seed: int = config.SEED) -> np.ndarray:
    """Portfolio ids held out of all training rows (sorted, reproducible)."""
    return np.sort(np.random.default_rng(seed).choice(n_portfolios, size=n, replace=False))


def fit_model(
    m: RiskModel, fit: pd.DataFrame, val: pd.DataFrame, train: pd.DataFrame
) -> None:
    """Fit `m` so that it ends up estimated on all of `train`.

    Models with a `refit` method (early stopping) are fit on `fit` with `val` choosing the
    stopping point, then refit on `train` (= fit + val + purge gap) with that stopping point.
    All other models are fit directly on `train`.
    """
    if hasattr(m, "refit"):
        m.fit(fit, val)
        m.refit(train)
    else:
        m.fit(train)


def _assert_no_silent_nan(model: RiskModel, pred: pd.DataFrame) -> None:
    """A head is either fully supported (no NaN) or fully absent (all NaN); never partial."""
    for col in pred.columns:
        n_nan = int(pred[col].isna().sum())
        if 0 < n_nan < len(pred):
            raise AssertionError(f"{model.name}: {n_nan}/{len(pred)} NaN predictions in {col}")


def fold_info(panel: pd.DataFrame, dates: pd.DatetimeIndex, fold: Fold) -> dict:
    """Dates describing a fold, for the backtest views (saved to folds.json)."""
    day = panel["day_idx"].to_numpy().astype(int)
    d = panel.index.get_level_values("date")
    train_end_idx = int(day[fold.train].max())
    return {
        "name": fold.name,
        "test_start": str(d[fold.test].min().date()),
        "test_end": str(d[fold.test].max().date()),
        "train_end": str(dates[train_end_idx].date()),
        "label_end": str(dates[train_end_idx + config.H].date()),
        "first_test_day_idx": int(day[fold.test].min()),
    }


def save_fold_models(models: list[RiskModel], folder: Path,
                     fallbacks: dict[str, int] | None = None) -> None:
    """Persist the fold's fitted ML models and the small baseline parameters.

    `fallbacks` ({model: EWMA fallbacks used in this fold}) is recorded in training.json.
    """
    import joblib

    folder.mkdir(parents=True, exist_ok=True)
    base: dict = {}
    training: dict = {"fallbacks": dict(fallbacks or {})}
    for m in models:
        if m.name in ("rf", "xgboost"):
            joblib.dump(m, folder / f"{m.name}.joblib")
            if m.name == "xgboost":
                training["xgboost_rounds"] = m.n_best
                training["xgboost_at_cap"] = m.at_cap
        elif m.name == "lstm":
            m.save(folder / "lstm.pt")
            training["lstm_best_epochs"] = dict(zip(("vol", "dd", "var"), m.best_epochs,
                                                    strict=True))
        elif m.name == "har":
            base["har_coef"] = [float(c) for c in m.coef_]
        elif m.name == "climatology":
            base["climatology_rate"] = float(m.rate)
        elif m.name == "fhs_mc_logit":
            base["fhs_mc_logit"] = [float(m.a), float(m.b)]
    (folder / "baselines.json").write_text(json.dumps(base))
    (folder / "training.json").write_text(json.dumps(training))


def run_walk_forward(
    models: list[RiskModel],
    panel: pd.DataFrame,
    folds: list[Fold],
    save_dir: Path | None = None,
    holdout: np.ndarray | None = None,
) -> pd.DataFrame:
    """Fit every model on every fold and return long predictions.

    Columns: fold, date, pid, model, target, y_true, y_pred, vol_now, heldout, y_aux (the model's
    VaR on its expected-shortfall rows, NaN elsewhere). Each model's
    prediction index must equal the test index, so all models are scored on identical rows.
    Portfolios in `holdout` are removed from every training row but remain in the test rows.
    """
    pids = panel.index.get_level_values("pid").to_numpy()
    in_hold = np.isin(pids, holdout) if holdout is not None else np.zeros(len(panel), bool)
    parts = []
    for fold in folds:
        fb_before = {m.name: m.n_fallback for m in models if hasattr(m, "n_fallback")}
        keep = lambda rows: rows[~in_hold[rows]]  # noqa: E731
        train, val = panel.iloc[keep(fold.train)], panel.iloc[keep(fold.val)]
        fit, test = panel.iloc[keep(fold.fit)], panel.iloc[fold.test]
        for m in models:
            fit_model(m, fit, val, train)
            pred = m.predict(test)
            assert pred.index.equals(test.index), f"{m.name}: prediction index mismatch"
            _assert_no_silent_nan(m, pred)
            for tname, (ycol, pcol) in TARGETS.items():
                ok = pred[pcol].notna().to_numpy()
                if not ok.any():
                    continue
                d = pd.DataFrame(
                    {"fold": fold.name, "model": m.name, "target": tname,
                     "y_true": test[ycol].to_numpy()[ok], "y_pred": pred[pcol].to_numpy()[ok],
                     "vol_now": test["naive_vol"].to_numpy()[ok],
                     "heldout": in_hold[fold.test][ok],
                     "y_aux": pred["var_hat"].to_numpy()[ok] if tname == "es" else np.nan},
                    index=test.index[ok],
                )
                parts.append(d.reset_index())
        if save_dir is not None:
            fb = {m.name: m.n_fallback - fb_before[m.name]
                  for m in models if hasattr(m, "n_fallback")}
            save_fold_models(models, save_dir / fold.name, fb)
    return pd.concat(parts, ignore_index=True)


def compute_metrics(preds: pd.DataFrame) -> pd.DataFrame:
    """Per (fold, model, target) metrics in long format."""
    rows = []
    for (fold, model, target), g in preds.groupby(["fold", "model", "target"], sort=False):
        s = score(target, g["y_true"].to_numpy(), g["y_pred"].to_numpy(), g["vol_now"].to_numpy(),
                  g["y_aux"].to_numpy())
        rows += [{"fold": fold, "model": model, "target": target, "metric": k, "value": v,
                  "n_rows": len(g), "n_dates": g["date"].nunique()} for k, v in s.items()]
    return pd.DataFrame(rows)


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    """Mean ± std across folds: one row per (target, model), one column per metric.

    Kupiec p-values are not averaged (a mean of p-values is meaningless); instead
    `kupiec_reject` counts the folds with p < 0.05 ("k/n") and `exc_range` gives the lowest and
    highest yearly exceedance rate.
    """
    agg = metrics[metrics["metric"] != "kupiec_p"].groupby(["target", "model", "metric"])[
        "value"].agg(["mean", "std"])
    agg["text"] = [f"{m:.6f} ± {s:.6f}" if k[2] in ("pinball", "mse_var")
                   else f"{m:.4f} ± {s:.4f}"
                   for k, m, s in zip(agg.index, agg["mean"], agg["std"], strict=True)]
    table = agg["text"].unstack("metric")
    kp = metrics[metrics["metric"] == "kupiec_p"].groupby(["target", "model"])["value"]
    table["kupiec_reject"] = kp.agg(lambda v: f"{int((v < 0.05).sum())}/{int(v.notna().sum())}")
    ex = metrics[metrics["metric"] == "exceedance"].groupby(["target", "model"])["value"]
    table["exc_range"] = ex.agg(lambda v: f"{v.min():.1%}-{v.max():.1%}")
    return table


def add_calibrated(
    preds: pd.DataFrame, panel: pd.DataFrame, eta: float = conformal.ETA, lag: int = conformal.LAG
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Append online-conformal-calibrated VaR rows named `{model}_cal` for EVERY VaR model.

    Calibrating only the ML models would tilt the comparison, so baselines get the same
    treatment. Held-out portfolios are excluded from the calibration signal. Returns the
    extended predictions and the per-model offset tables (model, day_idx, offset, err, c_after).
    """
    var = preds[(preds["target"] == "var") & ~preds["model"].str.endswith("_cal")]
    idx = pd.MultiIndex.from_frame(var[["date", "pid"]])
    scale = var_scale(panel["ewma_vol"].reindex(idx).to_numpy())
    new, tables = [], []
    for model in var["model"].unique():
        m = (var["model"] == model).to_numpy()
        g = var[m]
        day = panel["day_idx"].reindex(idx[m]).to_numpy().astype(int)
        q_cal, tab = conformal.calibrate_var(
            day, g["y_true"].to_numpy(), g["y_pred"].to_numpy(), scale[m],
            ~g["heldout"].to_numpy().astype(bool), eta, lag)
        new.append(g.assign(model=f"{model}_cal", y_pred=q_cal))
        tables.append(tab.assign(model=model))
    return pd.concat([preds, *new], ignore_index=True), pd.concat(tables, ignore_index=True)


def default_models() -> list[RiskModel]:
    """Baselines (incl. GARCH and FHS) plus the three ML models; a missing dependency raises."""
    from src.models import baselines

    seq_path = config.DATA_PROCESSED / "seq.npy"
    port = np.load(seq_path)[:, :, 0].astype("float64")
    models = baselines.default_baselines(port)
    from src.models.lstm import STATIC_R4, LSTMModel
    from src.models.rf import RFModel
    from src.models.simdd import SimDDModel
    from src.models.xgb import XGBModel

    sim = [SimDDModel("fhs_mc", False), SimDDModel("fhs_mc_logit", True)]
    return models + sim + [RFModel(), XGBModel(), LSTMModel(static_cols=STATIC_R4)]


def load_panel_with_features(quick: bool, holdout_n: int | None = None
                             ) -> tuple[pd.DataFrame, pd.DatetimeIndex, np.ndarray]:
    """Panel (+ Round 4 model features), date index and held-out portfolio ids.

    HAR coefficients are fit without the held-out portfolios, so the holdout is chosen first.
    Quick mode keeps the first 50 portfolios and holds out 10 of them.
    """
    from src.features import model_features

    panel = pd.read_parquet(config.DATA_PROCESSED / "panel.parquet")
    dates = pd.read_parquet(config.DATA_PROCESSED / "returns.parquet").index
    if quick:
        panel = panel[panel.index.get_level_values("pid") < 50]
    n_ports = int(panel.index.get_level_values("pid").max()) + 1
    holdout = holdout_pids(n_ports, holdout_n or (10 if quick else N_HOLDOUT))
    feats = pd.read_parquet(config.DATA_PROCESSED / "model_features.parquet")
    panel = model_features.add_model_features(panel, dates, feats, holdout)
    return panel, dates, holdout


def main(quick: bool = False) -> None:
    """Run all models on all folds and write predictions, per-fold metrics and summaries.

    `quick` is a smoke test and writes to reports/quick/, never touching the full results.
    The full run also saves reports/folds.json and per-fold models under models/artifacts/folds.
    """
    from src.models import combos

    config.set_seeds()
    years = TEST_YEARS[-2:] if quick else TEST_YEARS
    panel, dates, holdout = load_panel_with_features(quick)
    assert set(prereg.MODEL_FEATURES) <= set(panel.columns), "Round 4 model features missing"
    folds = make_folds(panel, dates, years)
    out = config.REPORTS / "quick" if quick else config.REPORTS
    save_dir = out / "fold_models" if quick else config.ARTIFACTS / "folds"
    preds = run_walk_forward(default_models(), panel, folds, save_dir=save_dir, holdout=holdout)
    preds = combos.add_combos(preds)  # equal-weight combinations of the raw members (A3)
    preds, offsets = add_calibrated(preds, panel)
    metrics = compute_metrics(preds)
    out.mkdir(parents=True, exist_ok=True)
    offsets.to_csv(out / "conformal_offsets.csv", index=False)
    preds.to_parquet(out / "predictions.parquet")
    metrics.to_csv(out / "metrics_by_fold.csv", index=False)
    (out / "folds.json").write_text(json.dumps([fold_info(panel, dates, f) for f in folds]))
    (out / "holdout_pids.json").write_text(json.dumps([int(x) for x in holdout]))
    table = summarize(metrics)
    table.to_csv(out / "comparison.csv")
    held = summarize(compute_metrics(preds[preds["heldout"]]))
    held.to_csv(out / "comparison_heldout.csv")
    from src.models import significance

    for name, df in significance.run_significance(preds).items():
        df.to_csv(out / {"dm": "significance_dm.csv", "mcs": "mcs.csv",
                         "coverage": "coverage_clustered.csv"}[name], index=False)
    print(table.to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="2 folds, 50 portfolios")
    main(ap.parse_args().quick)
