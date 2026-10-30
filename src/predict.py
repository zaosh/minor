"""Predict forward risk for an arbitrary user portfolio from the saved artifacts."""

from __future__ import annotations

import json
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from src import config
from src.features import engineer
from src.models import baselines, prereg, simdd
from src.models.base import PRED_COLS
from src.models.lstm import WINDOW, LSTMModel
from src.portfolios import Portfolio, build_user_portfolio, portfolio_returns


@lru_cache(maxsize=1)
def load_data() -> dict:
    """Cached returns, market and macro tables."""
    d = config.DATA_PROCESSED
    return {
        "returns": pd.read_parquet(d / "returns.parquet"),
        "market": pd.read_parquet(d / "market.parquet"),
        "macro": pd.read_parquet(d / "macro.parquet"),
    }


@lru_cache(maxsize=1)
def load_models() -> dict:
    """Cached trained models from models/artifacts (run `python -m src.models.train_final`)."""
    a = config.ARTIFACTS
    needed = [a / "rf.joblib", a / "xgboost.joblib", a / "lstm.pt", a / "meta.json"]
    missing = [p.name for p in needed if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing {missing}; run python -m src.models.train_final")
    return {"rf": joblib.load(a / "rf.joblib"), "xgboost": joblib.load(a / "xgboost.joblib"),
            "lstm": LSTMModel.load(a / "lstm.pt")}


def available_tickers() -> list[str]:
    """Tickers a user can put in a portfolio."""
    return list(load_data()["returns"].columns)


def portfolio_features(p: Portfolio, as_of: pd.Timestamp | None = None) -> pd.DataFrame:
    """Feature + causal-baseline row for the latest date (<= as_of); same code as training."""
    d = load_data()
    returns = d["returns"] if as_of is None else d["returns"].loc[:as_of]
    r = portfolio_returns(returns, p)
    mkt = d["market"][config.MARKET].pct_change().fillna(0.0).reindex(returns.index)
    members = returns[list(p.tickers)]
    cache = engineer.pair_corr_cache(members)
    mc = {L: engineer.mean_pair_corr(c, list(range(len(p.tickers))), returns.index)
          for L, c in cache.items()}
    feats = engineer.compute_features(r, mkt, d["macro"], mc, len(p.tickers), p.hhi)
    last = feats.iloc[[-1]].copy()
    last["naive_vol"] = baselines.naive_vol(r).iloc[-1]
    last["hist_var250"] = baselines.hist_var(r).iloc[-1]
    last["param_var63"] = baselines.param_var(r).iloc[-1]
    pos = np.array([len(returns) - 1])
    last["lw_vol"] = baselines.lw_vol(members.to_numpy(), np.array(p.weights), pos)[0]
    last["day_idx"] = len(returns) - 1
    return last


def _arch_fit(r: pd.Series, fit_end: int | None, spec: str):
    """Fit the `spec` model ("garch" or "gjr") and return (result, 21-step variance forecast
    from the last row of `r`, in percent-squared units). Raises RuntimeError on non-convergence."""
    from arch import arch_model

    am = arch_model(r * 100, mean="Constant", rescale=False, **baselines._GARCH_SPECS[spec])
    res = am.fit(last_obs=fit_end, disp="off", show_warning=False)
    if res.convergence_flag != 0:
        raise RuntimeError(f"{spec} did not converge")
    start = None if fit_end is None else len(r) - 1
    v = res.forecast(horizon=config.H, start=start, reindex=False).variance.to_numpy()[-1]
    return res, v


def garch_forecast(r: pd.Series, fit_end: int | None = None) -> dict:
    """GARCH(1,1) forecast at the last row of `r` from ONE fit.

    Returns `vol` (21-day annualized), normal-innovation `var` and `es` (1-day 95%, return
    units, negative), the filtered-historical-simulation `fhs_var` and `fhs_es` (empirical
    quantile of the standardized residuals), and what the Round 4 features and the simulated
    drawdown need: `sigma1` (one-step daily sigma), `mu`, `z` (standardized residuals before the
    cut-off) and `params` (omega in percent-squared units, alpha, beta).

    By default parameters are fit on the full history of `r`. With `fit_end` they are fit on the
    first `fit_end` rows only (as in the walk-forward baseline) and the variance is then filtered
    through the rest. Raises RuntimeError if the optimizer did not converge.
    """
    res, v = _arch_fit(r, fit_end, "garch")
    sigma = float(np.sqrt(v[0]) / 100)
    z = np.asarray(res.std_resid, dtype=float)
    zf = z[np.isfinite(z)][:fit_end]
    q, es = baselines.fhs_quantiles(zf)
    mu = float(res.params["mu"]) / 100
    pr = res.params
    return {"vol": float(np.sqrt(v.mean() * 252) / 100), "var": -baselines.Z95 * sigma,
            "es": baselines.ES_NORMAL * sigma, "fhs_var": mu + sigma * q,
            "fhs_es": mu + sigma * es, "sigma1": sigma, "mu": mu, "z": zf,
            "params": {"omega": float(pr["omega"]), "alpha": float(pr["alpha[1]"]),
                       "beta": float(pr["beta[1]"])}}


def gjr_forecast(r: pd.Series, fit_end: int | None = None) -> dict[str, float]:
    """GJR-GARCH(1,1,1)-t forecast at the last row of `r` (same fit policy as garch_forecast).

    Returns `vol` (21-day annualized) and zero-mean `var` / `es` (1-day 95%, return units) from
    the fitted standardised-t quantile and shortfall. Raises RuntimeError on non-convergence.
    """
    res, v = _arch_fit(r, fit_end, "gjr")
    sigma = float(np.sqrt(v[0]) / 100)
    q, es = baselines.std_t_var_es(float(res.params["nu"]))
    return {"vol": float(np.sqrt(v.mean() * 252) / 100), "var": q * sigma, "es": es * sigma}


def serving_features(r: pd.Series, end: int, row: pd.DataFrame, har_coef, fit_end: int,
                     date: pd.Timestamp) -> dict[str, float]:
    """The Round 4 model inputs for one portfolio at row position `end` (data up to `end` only).

    `har_vol` uses the given HAR coefficients; the GARCH features and the simulated drawdown
    probability come from one GARCH(1,1) fit on the first `fit_end` rows (the refresh cut-off),
    with the EWMA fallback of the batch code (flagged in `garch_fallback`). The simulation uses
    the uniform draws of the refresh year of `date`, exactly as `model_features`.
    """
    har = baselines.HAR()
    har.coef_ = np.asarray(har_coef, dtype=float)
    out = {"har_vol": float(np.exp(har._design(row) @ har.coef_)[0])}
    U = simdd.uniform_matrix(min(date.year, prereg.LAST_REFRESH_YEAR))
    try:
        g = garch_forecast(r.iloc[: end + 1], fit_end)
        ok = bool(np.isfinite(g["vol"]) and g["vol"] > 0 and np.isfinite(g["sigma1"])
                  and g["sigma1"] > 0)
    except Exception:
        ok = False
    if ok:
        p = g["params"]
        dd = simdd.simulate_dd_prob([g["sigma1"]], g["mu"], p["omega"] / 1e4, p["alpha"],
                                    p["beta"], g["z"], U)[0]
        out.update(garch_vol21=g["vol"], garch_sigma1=g["sigma1"] * baselines.ANN,
                   garch_fallback=0.0, sim_dd_raw=float(dd))
    else:
        ewma = float(row["ewma_vol"].iloc[0])
        rec = simdd.ewma_recursion()
        z = baselines.FHSBaseline(r.to_numpy(dtype=float)[None, :])._ewma_z(
            0, min(fit_end, end + 1))  # never past the as-of row
        dd = simdd.simulate_dd_prob([ewma / baselines.ANN], rec["mu"], rec["omega"], rec["alpha"],
                                    rec["beta"], z, U)[0]
        out.update(garch_vol21=ewma, garch_sigma1=ewma, garch_fallback=1.0, sim_dd_raw=float(dd))
    return out


@lru_cache(maxsize=1)
def load_meta() -> dict:
    """Metadata of the deployed models (training end, label end, vol quartiles, version)."""
    return json.loads((config.ARTIFACTS / "meta.json").read_text())


RISK_BANDS = ("Low", "Moderate", "High", "Very high")


def risk_band(vol: float, quartiles: list[float]) -> str:
    """Band of a predicted vol against the training-set quartiles of forward vol."""
    return RISK_BANDS[int(np.searchsorted(quartiles, vol, side="right"))]


def _resolve_date(as_of: pd.Timestamp | None) -> pd.Timestamp:
    """The last trading day in the cached data that is <= as_of (default: latest)."""
    idx = load_data()["returns"].index
    return idx[-1] if as_of is None else idx[idx <= pd.Timestamp(as_of)][-1]


def _model_window(p: Portfolio, end: int) -> np.ndarray:
    """The LSTM input (WINDOW, 4): port_ret, mkt_ret, VIX, VIX_chg for rows end-WINDOW+1..end."""
    d = load_data()
    sl = slice(end - WINDOW + 1, end + 1)
    return np.column_stack([
        portfolio_returns(d["returns"], p).iloc[sl].to_numpy(),
        d["market"][config.MARKET].pct_change().fillna(0.0).reindex(d["returns"].index)
        .iloc[sl].to_numpy(),
        d["macro"]["VIX"].iloc[sl].to_numpy(),
        d["macro"]["VIX_chg"].fillna(0.0).iloc[sl].to_numpy(),
    ]).astype(np.float32)


def _predict_with(
    models: dict, p: Portfolio, as_of: pd.Timestamp | None = None,
    garch_fit_end: int | None = None, har_coef=None, feature_fit_end: int | None = None,
) -> dict:
    """Predict with the given models ({"rf","xgboost","lstm"}) using data up to `as_of`.

    With `garch_fit_end` (a row position) GARCH and GJR parameters are fit on earlier rows only
    and a failed fit falls back to EWMA, mirroring the walk-forward baselines. With `har_coef`
    and `feature_fit_end` the Round 4 model inputs (HAR and GARCH forecasts, simulated drawdown
    probability) are computed from that refresh cut-off and added to the feature row; without
    them models trained before Round 4 are served unchanged.
    """
    row = portfolio_features(p, as_of)
    d = load_data()
    warnings = []
    if not config.K_RANGE[0] <= len(p.tickers) <= config.K_RANGE[1]:
        warnings.append(f"Models were trained on {config.K_RANGE[0]}-{config.K_RANGE[1]} assets; "
                        f"{len(p.tickers)} is outside that range, so treat results with caution.")
    date = row.index[0]
    row.index = pd.MultiIndex.from_tuples([(date, 0)], names=["date", "pid"])
    end = d["returns"].index.get_loc(date)
    r_full = portfolio_returns(d["returns"], p)
    if har_coef is not None:
        for k, v in serving_features(r_full, end, row, har_coef, feature_fit_end, date).items():
            row[k] = v
    out: dict[str, dict] = {}
    for name in ("rf", "xgboost"):
        out[name] = models[name].predict(row).iloc[0][PRED_COLS].to_dict()
    static = {c: float(row[c].iloc[0]) for c in models["lstm"].static_cols}
    out["lstm"] = models["lstm"].predict_window(_model_window(p, end), static)
    naive, ewma, lw = (float(row[c].iloc[0]) for c in ("naive_vol", "ewma_vol", "lw_vol"))
    base: dict[str, dict] = {}
    for name, vol in (("naive", naive), ("ewma", ewma), ("ledoit_wolf", lw)):
        base[name] = {"vol_hat": vol, "dd_hat": float(baselines.dd_prob_from_vol([vol])[0]),
                      "var_hat": float(baselines.var_from_vol([vol])[0]),
                      "es_hat": float(baselines.es_from_vol([vol])[0])}
    r_hist = r_full.to_numpy(dtype=float)
    try:
        g = garch_forecast(r_full.iloc[: end + 1], garch_fit_end)
        base["garch"] = {"vol_hat": g["vol"],
                         "dd_hat": float(baselines.dd_prob_from_vol([g["vol"]])[0]),
                         "var_hat": g["var"], "es_hat": g["es"]}
        base["fhs"] = {"var_hat": g["fhs_var"], "es_hat": g["fhs_es"]}
    except Exception:
        if garch_fit_end is None:
            warnings.append("GARCH did not converge for this portfolio; GARCH and FHS omitted.")
        else:
            base["garch"] = {k: base["ewma"][k] for k in ("vol_hat", "dd_hat", "var_hat",
                                                          "es_hat")}
            q, es = baselines.fhs_quantiles(
                baselines.FHSBaseline(r_hist[None, :])._ewma_z(0, min(garch_fit_end, end + 1)))
            base["fhs"] = {"var_hat": q * ewma / baselines.ANN, "es_hat": es * ewma / baselines.ANN}
            warnings.append("GARCH did not converge; EWMA used in its place (also for FHS).")
    try:
        gj = gjr_forecast(r_full.iloc[: end + 1], garch_fit_end)
        base["gjr"] = {"vol_hat": gj["vol"],
                       "dd_hat": float(baselines.dd_prob_from_vol([gj["vol"]])[0]),
                       "var_hat": gj["var"], "es_hat": gj["es"]}
    except Exception:
        base["gjr"] = dict(base["ewma"])
        if garch_fit_end is None:
            warnings.append("GJR-GARCH did not converge; EWMA used in its place.")
    pos_end = np.array([end])
    base["hist_var"] = {"var_hat": float(row["hist_var250"].iloc[0]),
                        "es_hat": float(baselines.hist_es_at(r_hist, pos_end)[0])}
    base["param_var"] = {"var_hat": float(row["param_var63"].iloc[0]),
                         "es_hat": float(baselines.param_es_at(r_hist, pos_end)[0])}
    return {"as_of": date, "portfolio": p, "ml": out, "baselines": base, "warnings": warnings,
            "features": row}


def predict_portfolio(
    tickers: list[str], weights: list[float], as_of: pd.Timestamp | None = None
) -> dict:
    """Predict vol / drawdown probability / VaR for a user portfolio with the deployed models.

    Raises ValueError for invalid tickers or weights, and for an `as_of` whose labels the
    deployed models were trained on (< meta["label_end_date"]): those predictions would be
    in-sample, so use the Backtest view (per-fold models) for past dates. On label_end_date
    itself the target window (date+1..date+H) lies wholly after the last training return.
    """
    p = build_user_portfolio(tickers, weights, available_tickers())
    date, meta = _resolve_date(as_of), load_meta()
    if date < pd.Timestamp(meta["label_end_date"]):
        raise ValueError(
            f"{date:%Y-%m-%d} is inside the deployed models' training period (labels run to "
            f"{meta['label_end_date']}); use the Backtest view for past dates.")
    res = _predict_with(load_models(), p, date,
                        har_coef=meta.get("har_feature_coef"),
                        feature_fit_end=meta.get("feature_refresh_day_idx"))
    res["risk_band"] = {k: risk_band(v["vol_hat"], meta["vol_quartiles"])
                        for k, v in res["ml"].items()}
    return res


def portfolio_history(p: Portfolio) -> pd.DataFrame:
    """Wealth index, rolling 21d vol and drawdown of the portfolio for the history charts."""
    r = portfolio_returns(load_data()["returns"], p)
    wealth = (1 + r).cumprod()
    return pd.DataFrame({"wealth": wealth, "vol_21": r.rolling(21).std() * np.sqrt(252),
                         "drawdown": wealth / wealth.cummax() - 1})
