"""Turn reports/predictions.parquet into the comparison tables and interactive figures."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src import config
from src.models import evaluate

PRIMARY = {"vol": "rmse", "dd": "brier", "var": "pinball"}  # lower is better for all three
FIG_DIR = config.REPORTS / "figures"


def _md(df: pd.DataFrame, index: bool = True) -> str:
    """Minimal markdown table (avoids a tabulate dependency)."""
    d = df.reset_index() if index else df
    head = "| " + " | ".join(map(str, d.columns)) + " |\n|" + "---|" * len(d.columns) + "\n"
    body = "".join("| " + " | ".join(map(str, r)) + " |\n" for r in d.itertuples(index=False))
    return head + body


def beats_naive_table(metrics: pd.DataFrame) -> pd.DataFrame:
    """For each target/model: mean primary metric and the number of folds it beats naive in."""
    rows = []
    for target, metric in PRIMARY.items():
        m = metrics[(metrics["target"] == target) & (metrics["metric"] == metric)]
        piv = m.pivot(index="fold", columns="model", values="value")
        ref_name = {"vol": "naive", "dd": "climatology", "var": "hist_var"}[target]
        for model in piv.columns:
            # calibrated models are compared with the equally calibrated reference
            ref = piv[ref_name + "_cal"] if model.endswith("_cal") and target == "var" \
                else piv[ref_name]
            rows.append({"target": target, "metric": metric, "model": model,
                         "mean": piv[model].mean(), "std": piv[model].std(),
                         "folds_better_than_ref": int((piv[model] < ref).sum()),
                         "n_folds": int(piv[model].notna().sum())})
    return pd.DataFrame(rows)


def kupiec_reject_share(preds: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """Share of portfolios whose own Kupiec test rejects, per model (pooled tests overstate N)."""
    v = preds[preds["target"] == "var"]
    rows = []
    for (model, pid), g in v.groupby(["model", "pid"]):
        exc = int((g["y_true"] < g["y_pred"]).sum())
        p_val = evaluate.kupiec_pof(exc, len(g))[1]
        rows.append({"model": model, "pid": pid, "reject": p_val < alpha})
    return (pd.DataFrame(rows).groupby("model")["reject"].mean().rename("kupiec_reject_share")
            .reset_index())


def dd_calibration(preds: pd.DataFrame) -> pd.DataFrame:
    """Per model and fold: mean predicted drawdown probability vs the realized event rate."""
    dd = preds[preds["target"] == "dd"]
    g = dd.groupby(["model", "fold"]).agg(predicted=("y_pred", "mean"), realized=("y_true", "mean"))
    return g.round(3).reset_index()


def rf_importance() -> pd.DataFrame | None:
    """Impurity importance of the final RF volatility model, if artifacts exist."""
    import joblib

    path = config.ARTIFACTS / "rf.joblib"
    if not path.exists():
        return None
    m = joblib.load(path)
    imp = pd.Series(m.reg.feature_importances_, index=m.cols).sort_values(ascending=False)
    return imp.head(15).round(4).rename("importance").rename_axis("feature").reset_index()


def training_diagnostics() -> pd.DataFrame | None:
    """Per fold: boosting rounds chosen for each XGBoost head and the LSTM's best epochs."""
    import json

    rows = []
    for f in sorted((config.ARTIFACTS / "folds").glob("*/training.json")):
        t = json.loads(f.read_text())
        rows.append({"fold": f.parent.name,
                     **{f"xgb_{k}": v for k, v in t.get("xgboost_rounds", {}).items()},
                     "xgb_at_cap": ",".join(k for k, v in t.get("xgboost_at_cap", {}).items() if v),
                     **{f"lstm_{k}": v for k, v in t.get("lstm_best_epochs", {}).items()}})
    return pd.DataFrame(rows) if rows else None


def conformal_summary() -> pd.DataFrame | None:
    """Average exceedance of each calibrated VaR and the drift bound of the tracker identity."""
    from src.models import conformal

    path = config.REPORTS / "conformal_offsets.csv"
    if not path.exists():
        return None
    t = pd.read_csv(path)
    rows = [{"model": m, **{k: round(v, 4) for k, v in conformal.drift_summary(g).items()
                            if k not in ("alpha",)}} for m, g in t.groupby("model")]
    return pd.DataFrame(rows)


def eta_sensitivity(preds: pd.DataFrame) -> pd.DataFrame:
    """Calibrated-VaR exceedance for other step sizes (sensitivity only, never a selection)."""
    from src.models import conformal

    panel = pd.read_parquet(config.DATA_PROCESSED / "panel.parquet")
    raw = preds[(preds["target"] == "var") & ~preds["model"].str.endswith("_cal")]
    rows = []
    for eta in (0.1, conformal.ETA, 0.5):
        cal, _ = evaluate.add_calibrated(raw, panel, eta=eta)
        t = evaluate.summarize(evaluate.compute_metrics(cal[cal["model"].str.endswith("_cal")]))
        t = t.loc["var"][["exceedance", "exc_range", "kupiec_reject"]].assign(eta=eta)
        rows.append(t.reset_index())
    return pd.concat(rows).sort_values(["model", "eta"])[
        ["model", "eta", "exceedance", "exc_range", "kupiec_reject"]]


def _csv(name: str) -> pd.DataFrame | None:
    path = config.REPORTS / name
    return pd.read_csv(path) if path.exists() else None


def significance_markdown() -> str:
    """Date-level tests (DM, MCS, clustered coverage) from the evaluate outputs, if present."""
    parts = []
    dm, mcs, cov = (_csv("significance_dm.csv"), _csv("mcs.csv"), _csv("coverage_clustered.csv"))
    if dm is not None:
        d = dm.copy()
        for c in ("mean_loss_diff", "dm_stat", "p"):
            d[c] = d[c].map(lambda x: f"{x:.4g}")
        parts.append("## Diebold-Mariano tests (date-level, HAC variance)\n\nNegative mean loss "
                     "difference = the ML model has the lower loss. `significant` is p < 0.05. "
                     "~330 dates, so low power.\n\n" + _md(d, index=False) + "\n")
    if mcs is not None:
        rows = [{"target": t, "in the 90% confidence set": ", ".join(g[g["in_set"]]["model"]),
                 "eliminated": ", ".join(g[~g["in_set"]]["model"])}
                for t, g in mcs.groupby("target", sort=False)]
        parts.append("## Model confidence set (90%, Hansen-Lunde-Nason)\n\nModels that cannot be "
                     "statistically separated from the best on date-level loss.\n\n"
                     + _md(pd.DataFrame(rows), index=False) + "\n")
    if cov is not None:
        allp = cov[cov["period"] == "all"].copy()
        yearly = cov[cov["period"] != "all"].groupby("model")["p"].agg(
            lambda v: f"{int((v < 0.05).sum())}/{len(v)}").rename("years rejected (p<0.05)")
        allp = allp.merge(yearly, on="model")
        for c in ("exceed_rate", "stat", "p"):
            allp[c] = allp[c].map(lambda x: f"{x:.4g}")
        parts.append("## VaR coverage, date-clustered (HAC) test\n\nThe pooled Kupiec test treats "
                     "~20,000 rows per year as independent; this one averages the exceedance rate "
                     "over portfolios per date and tests the ~330 date-level rates.\n\n"
                     + _md(allp.drop(columns="period"), index=False) + "\n")
    return "\n".join(parts)


def _fmt_cols(d: pd.DataFrame, cols: tuple[str, ...], spec: str = "{:.4g}") -> pd.DataFrame:
    d = d.copy()
    for c in cols:
        if c in d:
            d[c] = d[c].map(lambda x: "-" if pd.isna(x) else spec.format(x))
    return d


def round4_markdown(folder=None) -> str:
    """Round 4 tables (Tier A/B outputs of `python -m src.models.round4`), if they exist.

    Everything is read from the CSV files the code wrote; nothing is typed in by hand.
    """
    from pathlib import Path

    folder = Path(folder) if folder else config.REPORTS / "round4" / "tierB"
    if not (folder / "dm_round4.csv").exists():
        return ""
    r = lambda n: pd.read_csv(folder / n)  # noqa: E731
    parts = ["# Round 4 (Tier B outputs; Round 3 numbers are in data/round3_frozen)\n"]
    vs = r("vol_summary.csv")
    parts.append("## Volatility: RMSE, MAE, QLIKE and MSE-var, six-year means\n\nQLIKE and "
                 "MSE-var are robust to the noisy 21-day volatility proxy (Patton 2011); when "
                 "the rankings disagree, ranking statements cite them. Lower is better except "
                 "`dir_acc`.\n\n" + _md(_fmt_cols(vs, tuple(vs.columns[1:6]), "{:.5f}"),
                                          index=False) + "\n")
    ml = r("ml_round3_vs_round4.csv")
    parts.append("## ML models: Round 3 beside Round 4 (HAR/GARCH features added in Round 4), "
                 "six-year means\n\n" + _md(_fmt_cols(ml, ("round3", "round4",
                                                           "diff_r4_minus_r3"), "{:.5f}"),
                                             index=False) + "\n")
    rel = r("dd_reliability.csv")
    rel = rel.groupby("model").apply(lambda g: pd.Series({
        "n": int(g["n"].sum()),
        "mean_pred": float((g["mean_pred"] * g["n"]).sum() / g["n"].sum()),
        "mean_true": float((g["mean_true"] * g["n"]).sum() / g["n"].sum())}),
        include_groups=False).reset_index()
    parts.append("## Drawdown: pooled mean predicted probability vs realised rate "
                 "(bin-level reliability is in dd_reliability.csv)\n\n"
                 + _md(_fmt_cols(rel, ("mean_pred", "mean_true"), "{:.4f}"), index=False) + "\n")
    dm = r("dm_round4.csv")
    dm = _fmt_cols(dm, ("mean_loss_diff", "dm_stat", "p_raw", "p_holm"))
    show = [c for c in ("family", "target", "loss", "model", "baseline", "vs", "mean_loss_diff",
                        "dm_stat", "p_raw", "p_holm", "n_dates") if c in dm]
    parts.append("## Diebold-Mariano tests, raw and Holm-adjusted p (within each family)\n\n"
                 "Negative mean loss difference = the first model has the lower loss. About "
                 "330 dates, so low power; headline claims use the Holm p.\n\n"
                 + _md(dm[show].fillna("-"), index=False) + "\n")
    parts.append("## Number of tests run\n\n" + _md(r("tests_count.csv"), index=False) + "\n")
    mcs = r("mcs_round4.csv")
    rows = [{"target": t, "loss": lo, "in the 90% set": ", ".join(g[g["in_set"]]["model"]),
             "eliminated": ", ".join(g[~g["in_set"]]["model"])}
            for (t, lo), g in mcs.groupby(["target", "loss"], sort=False)]
    parts.append("## Model confidence sets (90%)\n\n" + _md(pd.DataFrame(rows), index=False)
                 + "\n")
    return "\n".join(parts)


def to_markdown(
    summary: pd.DataFrame,
    extra: pd.DataFrame,
    kupiec: pd.DataFrame,
    heldout: pd.DataFrame | None = None,
    calib: pd.DataFrame | None = None,
) -> str:
    """Write the final comparison as markdown."""
    parts = ["# Model comparison (mean ± std across walk-forward folds)\n"]
    cols = {"vol": ["rmse", "mae", "dir_acc", "qlike", "mse_var"],
            "dd": ["auc", "brier", "logloss"],
            "var": ["pinball", "exceedance", "exc_range", "kupiec_reject"],
            "es": ["fz0", "es_resid", "es_ratio"]}
    for target, title in (("vol", "Volatility (vol_fwd)"), ("dd", "Drawdown >= 5% probability"),
                          ("var", "95% 1-day VaR (target exceedance 5%)"),
                          ("es", "95% 1-day expected shortfall")):
        t = summary.loc[target][cols[target]].dropna(how="all")
        parts.append(f"## {title}\n\n{_md(t)}\n")
        if target == "es":
            parts.append("`fz0` = Fissler-Ziegel/Patton joint (VaR, ES) loss, lower is better and "
                         "the only ranking score; `es_resid` = mean of (return - ES)/|ES| on the "
                         "VaR exceedances (0 if right, negative = ES too shallow); `es_ratio` "
                         "= realised/predicted average exceedance (1 if right). Only the "
                         "baselines have an ES; the ML models have no ES head. There are only "
                         "about 2.5 exceedances per portfolio-year, so these checks have low "
                         "power.\n")
        if target == "var":
            parts.append("Models ending in `_cal` are the same VaR forecasts after the online "
                         "conformal correction (applied to every VaR model, baselines included). "
                         "Pinball is shown with six decimals because the models differ in the "
                         "fourth.\n")
            parts.append("`kupiec_reject` = folds (test years) where the pooled Kupiec test "
                         "rejects at 5%; `exc_range` = lowest-highest yearly exceedance rate. "
                         "Pooled tests overstate the sample size (exceedances are dependent "
                         "across portfolios).\n")
    wins = extra.copy()
    wins["mean ± std"] = wins.apply(lambda r: f"{r['mean']:.4f} ± {r['std']:.4f}", axis=1)
    wins["folds beating reference"] = wins["folds_better_than_ref"].astype(str) + "/" + \
        wins["n_folds"].astype(str)
    parts.append("## Primary metric and folds beating the reference\n\n(reference: naive for vol, "
                 "climatology for drawdown, historical VaR for VaR; lower is better)\n\n"
                 + wins[["target", "metric", "model", "mean ± std", "folds beating reference"]]
                 
                 .pipe(_md, index=False) + "\n")
    parts.append("## VaR: share of portfolios rejected by their own Kupiec test (5% level)\n\n"
                 + kupiec
                 .pipe(_md, index=False) + "\n")
    if heldout is not None:
        parts.append("## Held-out portfolios (ids never used for training)\n")
        for target, title in (("vol", "Volatility"), ("dd", "Drawdown"), ("var", "VaR"),
                              ("es", "Expected shortfall")):
            tab = heldout.loc[target][cols[target]].dropna(how="all")
            parts.append(f"### {title}\n\n{_md(tab)}\n")
    if calib is not None:
        parts.append("## Drawdown calibration: mean predicted probability vs realized rate\n\n"
                     + _md(calib, index=False) + "\n")
    return "\n".join(parts)


def _save(fig: go.Figure, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.write_html(FIG_DIR / f"{name}.html", include_plotlyjs="cdn")


def make_figures(preds: pd.DataFrame, metrics: pd.DataFrame) -> None:
    """Write the interactive HTML figures used in the report."""
    vol = preds[preds["target"] == "vol"]

    sample = vol.sample(min(len(vol), 40000), random_state=config.SEED)
    show = [m for m in ("naive", "har", "garch", "xgboost", "lstm") if m in set(sample["model"])]
    s = sample[sample["model"].isin(show)].groupby("model").head(1500)
    fig = px.scatter(s, x="y_true", y="y_pred", facet_col="model", opacity=0.3,
                     labels={"y_true": "actual vol_fwd", "y_pred": "predicted"})
    lim = float(s["y_true"].quantile(0.99))
    fig.update_xaxes(range=[0, lim]).update_yaxes(range=[0, lim])
    _save(fig, "vol_pred_vs_actual")

    rm = metrics[(metrics["target"] == "vol") & (metrics["metric"] == "rmse")]
    _save(px.line(rm, x="fold", y="value", color="model", markers=True,
                  labels={"value": "RMSE (vol)", "fold": "test year"},
                  title="Volatility error by walk-forward fold"), "vol_rmse_by_fold")

    for year in (y for y in (2020, 2022) if y in set(pd.to_datetime(vol["date"]).dt.year)):
        d = vol[pd.to_datetime(vol["date"]).dt.year == year]
        g = d.groupby(["date", "model"])[["y_true", "y_pred"]].mean().reset_index()
        fig = go.Figure()
        actual = g[g["model"] == g["model"].iloc[0]]
        fig.add_scatter(x=actual["date"], y=actual["y_true"], name="actual", line=dict(width=3))
        for m in show:
            gm = g[g["model"] == m]
            fig.add_scatter(x=gm["date"], y=gm["y_pred"], name=m)
        fig.update_layout(title=f"Mean predicted vs actual 21-day vol across portfolios, {year}")
        _save(fig, f"stress_{year}")

    dd = preds[preds["target"] == "dd"].copy()
    dd["bin"] = pd.cut(dd["y_pred"], np.linspace(0, 1, 11), include_lowest=True)
    rel = (dd.groupby(["model", "bin"], observed=True)
             .agg(pred=("y_pred", "mean"), obs=("y_true", "mean"), n=("y_true", "size"))
             .reset_index())
    fig = px.line(rel, x="pred", y="obs", color="model", markers=True,
                  title="Drawdown probability reliability (perfect = diagonal)")
    fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash", color="grey"))
    _save(fig, "dd_reliability")

    ex = metrics[(metrics["target"] == "var") & (metrics["metric"] == "exceedance")]
    ex = ex.groupby("model")["value"].mean().reset_index()
    fig = px.bar(ex, x="model", y="value", title="VaR exceedance rate (target 5%)")
    fig.add_hline(y=config.VAR_ALPHA, line_dash="dash")
    _save(fig, "var_exceedance")


def xgb_importance() -> pd.DataFrame | None:
    """Gain importance of the final XGBoost vol model, if artifacts exist."""
    import joblib

    path = config.ARTIFACTS / "xgboost.joblib"
    if not path.exists():
        return None
    m = joblib.load(path)
    imp = pd.Series(m.reg.feature_importances_, index=m.cols).sort_values(ascending=False)
    return imp.head(15).rename("gain").rename_axis("feature").reset_index()


def main() -> None:
    """Build comparison.md and figures from the saved evaluation outputs."""
    preds = pd.read_parquet(config.REPORTS / "predictions.parquet")
    metrics = pd.read_csv(config.REPORTS / "metrics_by_fold.csv", dtype={"fold": str})
    held = None
    if "heldout" in preds.columns and preds["heldout"].any():
        held = evaluate.summarize(evaluate.compute_metrics(preds[preds["heldout"]]))
    md = to_markdown(evaluate.summarize(metrics), beats_naive_table(metrics),
                     kupiec_reject_share(preds), held, dd_calibration(preds))
    md += "\n" + significance_markdown()
    cs = conformal_summary()
    if cs is not None:
        md += ("\n## Conformal calibration: average exceedance and drift bound\n\n`mean_err` is "
               "the average share of portfolios below the calibrated VaR (target 0.05); by the "
               "tracker identity mean_err - 0.05 = `drift_bound` exactly. A small number says "
               "the long-run average is calibrated, nothing about single years.\n\n"
               + _md(cs, index=False) + "\n")
        md += ("\n## Step-size sensitivity (reported only; eta = 0.25 was fixed beforehand)\n\n"
               + _md(eta_sensitivity(preds), index=False) + "\n")
    td = training_diagnostics()
    if td is not None:
        md += ("\n## Training diagnostics per fold\n\nXGBoost rounds = argmin of each head's "
               "validation curve (`xgb_at_cap` lists heads that hit the 600-round cap); LSTM "
               "epochs = each head's best validation epoch.\n\n" + _md(td, index=False) + "\n")
    seeds = _csv("seeds/seed_spread.csv")
    if seeds is not None:
        s = seeds.copy()
        for c in ("mean", "std", "min", "max"):
            s[c] = s[c].map(lambda x: f"{x:.5f}")
        md += ("\n## Seed spread, Round 3 ML configuration (6-year means across seeds)\n\n"
               + _md(s, index=False) + "\n")
    rimp = rf_importance()
    if rimp is not None:
        md += "\n## Random Forest volatility model: top features (impurity)\n\n" + _md(
            rimp, index=False) + "\n"
    imp = xgb_importance()
    if imp is not None:
        md += "\n## XGBoost volatility model: top features (gain)\n\n" + _md(imp, index=False)
        _save(px.bar(imp[::-1], x="gain", y="feature", orientation="h",
                     title="XGBoost feature importance"), "xgb_importance")
    md += "\n" + round4_markdown()
    (config.REPORTS / "comparison.md").write_text(md + "\n")
    make_figures(preds, metrics)
    print("wrote reports/comparison.md and reports/figures/")


if __name__ == "__main__":
    main()
