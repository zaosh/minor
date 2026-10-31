"""Figures for the report and deck, drawn from the saved results in reports/.

Documentation tooling, not part of the model code: needs matplotlib and pandas
(not in requirements.txt). Run with any Python that has them:
    python docs/make_figures.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import json

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "reports"
OUT = Path(__file__).resolve().parent / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

C_ML, C_COMBO, C_TRAD, C_NAIVE, C_REF = "#D55E00", "#009E73", "#0072B2", "#7F7F7F", "#CC79A7"
KIND = {"rf": C_ML, "xgboost": C_ML, "lstm": C_ML, "combo_hg": C_COMBO, "combo_hgx": C_COMBO,
        "naive": C_NAIVE, "unscaled": C_NAIVE, "climatology": C_NAIVE, "oracle": C_REF,
        "fhs_mc_logit": C_TRAD, "fhs_mc": C_TRAD}
LABEL = {"combo_hgx": "combo HAR+GARCH+XGB", "combo_hg": "combo HAR+GARCH", "gjr": "GJR-GARCH-t",
         "garch": "GARCH(1,1)", "har": "HAR", "ewma": "EWMA", "ledoit_wolf": "Ledoit-Wolf",
         "xgboost": "XGBoost", "rf": "Random Forest", "lstm": "LSTM", "naive": "Naive",
         "fhs_mc_logit": "Simulated DD (recal.)", "fhs_mc": "Simulated DD",
         "climatology": "Base rate", "unscaled": "No scaling", "oracle": "Oracle (ceiling)",
         "hist_var": "Historical VaR", "param_var": "Parametric VaR", "fhs": "Filtered hist. sim."}
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titlesize": 9, "figure.dpi": 150, "savefig.dpi": 300,
                     "font.family": "DejaVu Sans"})


def col(m):
    return KIND.get(m, C_TRAD)


def lab(m):
    return LABEL.get(m, m)


def legend_kinds(ax, loc="lower right"):
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=C_ML, label="Machine learning"),
                       Patch(color=C_COMBO, label="Combination"),
                       Patch(color=C_TRAD, label="Traditional"),
                       Patch(color=C_NAIVE, label="Reference")], frameon=False, loc=loc,
              fontsize=7)


def pipeline():
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    ax.axis("off")
    boxes = ["Data\n45 NSE tickers\nIndia VIX, FRED", "Portfolios\n400 random,\n3-15 assets",
             "Features\nvolatility, drawdown,\ncorrelation, macro",
             "Targets (t+1..t+21)\nvolatility\nP(drawdown >= 5%)\n95% VaR",
             "Models\nRF, XGBoost, LSTM\nvs GARCH, HAR, ..."]
    w, gap = 1.5, 0.22
    for i, t in enumerate(boxes):
        x = i * (w + gap)
        ax.add_patch(FancyBboxPatch((x, 1.0), w, 1.0, boxstyle="round,pad=0.03", fc="#EAF1F8",
                                    ec="#0072B2", lw=1))
        ax.text(x + w / 2, 1.5, t, ha="center", va="center", fontsize=6.5)
        if i < len(boxes) - 1:
            ax.annotate("", xy=(x + w + gap - 0.03, 1.5), xytext=(x + w + 0.03, 1.5),
                        arrowprops=dict(arrowstyle="->", color="#444"))
    total = len(boxes) * (w + gap) - gap
    split = 3 * (w + gap) - gap
    for t, x0, x1 in [("Walk-forward evaluation: 6 expanding folds (2020-2025+),\n"
                       "21-day purge, 80 held-out portfolios", 0.0, split),
                      ("Tests: Diebold-Mariano (Holm),\nconfidence set, Kupiec,\nChristoffersen",
                       split + gap, total)]:
        ax.add_patch(FancyBboxPatch((x0, 0.0), x1 - x0, 0.7, boxstyle="round,pad=0.03",
                                    fc="#FDF1E7", ec="#D55E00", lw=1))
        ax.text((x0 + x1) / 2, 0.35, t, ha="center", va="center", fontsize=6.5)
    ax.set_xlim(-0.1, total + 0.1)
    ax.set_ylim(-0.1, 2.15)
    fig.savefig(OUT / "pipeline.png", bbox_inches="tight")
    plt.close(fig)


def folds():
    f = json.loads((R / "folds.json").read_text())
    fig, ax = plt.subplots(figsize=(3.5, 2.1))
    start = pd.Timestamp("2010-01-01")
    for i, k in enumerate(f):
        te, ts = pd.Timestamp(k["train_end"]), pd.Timestamp(k["test_start"])
        e = pd.Timestamp(k["test_end"])
        y = len(f) - i
        ax.barh(y, (te - start).days, left=0, color="#0072B2", height=0.55)
        ax.barh(y, (ts - te).days, left=(te - start).days, color="#BBBBBB", height=0.55)
        ax.barh(y, (e - ts).days, left=(ts - start).days, color="#D55E00", height=0.55)
    ax.set_yticks(range(1, len(f) + 1))
    ax.set_yticklabels([k["name"] for k in reversed(f)])
    ticks = [pd.Timestamp(f"{y}-01-01") for y in (2010, 2015, 2020, 2025)]
    ax.set_xticks([(t - start).days for t in ticks])
    ax.set_xticklabels([t.year for t in ticks])
    ax.set_ylabel("Test year")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#0072B2", label="train"), Patch(color="#BBBBBB", label="purge"),
                       Patch(color="#D55E00", label="test")], frameon=False, fontsize=7,
              loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.savefig(OUT / "folds.png", bbox_inches="tight")
    plt.close(fig)


def vol_bars():
    v = pd.read_csv(R / "round4/tierB/vol_summary.csv").set_index("model")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=False)
    for ax, m, t in zip(axes, ("qlike", "rmse"), ("QLIKE (lower is better)", "RMSE (lower is better)")):
        s = v[m].sort_values()
        ax.barh([lab(i) for i in s.index], s.values, color=[col(i) for i in s.index])
        ax.invert_yaxis()
        ax.set_title(t)
        ax.set_xlim(0, s.max() * 1.12)  # zero baseline: bar length must mean magnitude
        for y, val in enumerate(s.values):
            ax.text(val, y, f" {val:.3f}", va="center", fontsize=6.5)
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=C_ML, label="Machine learning"),
                        Patch(color=C_COMBO, label="Combination"),
                        Patch(color=C_TRAD, label="Traditional"),
                        Patch(color=C_NAIVE, label="Reference")], frameon=False, ncol=4,
               loc="lower center", fontsize=7)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(OUT / "vol_bars.png", bbox_inches="tight")
    plt.close(fig)


def vol_time():
    p = pd.read_parquet(R / "predictions.parquet", filters=[("target", "==", "vol")],
                        columns=["date", "model", "y_true", "y_pred"])
    models = ["naive", "gjr", "xgboost", "lstm"]
    g = p[p.model.isin(models)].groupby(["model", "date"])[["y_true", "y_pred"]].mean()
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    t = g.loc["naive"]["y_true"]
    ax.fill_between(t.index, t.values, color="#DDDDDD", label="Realised (mean of 400 portfolios)")
    for m in models:
        x = g.loc[m]["y_pred"]
        ax.plot(x.index, x.values, lw=0.9,
                color={"gjr": C_TRAD, "xgboost": C_ML, "lstm": "#E6A800", "naive": C_NAIVE}[m],
                ls="--" if m == "naive" else "-", label=lab(m))
    ax.set_ylabel("Annualised 21-day volatility")
    ax.legend(frameon=False, ncol=3, fontsize=7, loc="upper right")
    fig.savefig(OUT / "vol_time.png", bbox_inches="tight")
    plt.close(fig)


def dd_reliability():
    d = pd.read_csv(R / "round4/tierB/dd_reliability.csv")
    fig, ax = plt.subplots(figsize=(3.5, 3.2))
    ax.plot([0, 1], [0, 1], color="#999", lw=0.8, ls=":")
    for m, c, mk in (("fhs_mc_logit", C_TRAD, "o"), ("lstm", C_ML, "s"), ("garch", "#56B4E9", "^")):
        g = d[(d.model == m) & (d.n > 200)]
        ax.plot(g.mean_pred, g.mean_true, marker=mk, color=c, lw=1, ms=3.5, label=lab(m))
    ax.set_xlabel("Predicted probability of >= 5% drawdown")
    ax.set_ylabel("Realised frequency")
    ax.legend(frameon=False, fontsize=7)
    ax.set_aspect("equal")
    fig.savefig(OUT / "dd_reliability.png", bbox_inches="tight")
    plt.close(fig)


def var_yearly():
    m = pd.read_csv(R / "metrics_by_fold.csv")
    e = m[(m.metric == "exceedance") & (m.target == "var")]
    models = ["gjr", "garch", "fhs", "xgboost", "lstm", "naive", "hist_var"]
    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    for i, k in enumerate(models):
        v = e[e.model == k].value * 100
        ax.plot([i, i], [v.min(), v.max()], color=col(k) if k not in ("garch", "fhs", "hist_var", "gjr")
                else C_TRAD, lw=5, solid_capstyle="butt", alpha=0.8)
        ax.plot([i], [v.mean()], "k_", ms=9)
    ax.axhline(5, color="#999", ls=":", lw=1)
    ax.set_xticks(range(len(models)))
    short = {"gjr": "GJR", "garch": "GARCH", "fhs": "FHS", "xgboost": "XGB", "lstm": "LSTM",
             "naive": "Naive", "hist_var": "Hist.\nVaR"}
    ax.set_xticklabels([short[k] for k in models], fontsize=7)
    ax.set_ylabel("VaR exceedance by year (%)")
    ax.text(len(models) - 0.5, 5.15, "target 5%", fontsize=6.5, ha="right", color="#666")
    fig.savefig(OUT / "var_yearly.png", bbox_inches="tight")
    plt.close(fig)


def downstream():
    s = pd.read_csv(R / "portfolio_backtest/summary_all.csv").set_index("model")
    s = s.sort_values("tracking_rmse_reach_mean")
    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    sub = s.drop(index="unscaled")
    ax.barh([lab(i) for i in sub.index], sub.tracking_rmse_reach_mean, color=[col(i) for i in sub.index],
            xerr=sub.tracking_rmse_reach_std, error_kw=dict(lw=0.6, capsize=1.5, ecolor="#555"))
    ax.invert_yaxis()
    ax.set_xlabel("Tracking error of 63-day vol vs 10% target")
    ax.text(0.98, 0.98, f"No scaling: {s.loc['unscaled', 'tracking_rmse_reach_mean']:.3f}\n"
            "(bars: mean, whiskers: std over 6 folds)", transform=ax.transAxes, ha="right",
            va="top", fontsize=7)
    fig.savefig(OUT / "downstream.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    for fn in (pipeline, folds, vol_bars, vol_time, dd_reliability, var_yearly, downstream):
        fn()
        print("made", fn.__name__)
