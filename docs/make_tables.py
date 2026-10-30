"""LaTeX table bodies for the report, generated from the saved results (no hand-copied numbers).

Documentation tooling (needs pandas only):  python docs/make_tables.py
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "reports"
OUT = Path(__file__).resolve().parent / "report" / "tables"
OUT.mkdir(parents=True, exist_ok=True)
NAME = {"combo_hgx": "Combo HAR+GARCH+XGB", "combo_hg": "Combo HAR+GARCH", "gjr": "GJR-GARCH-t",
        "garch": "GARCH(1,1)", "har": "HAR", "ewma": "EWMA", "ledoit_wolf": "Ledoit-Wolf",
        "xgboost": "XGBoost", "rf": "Random Forest", "lstm": "LSTM", "naive": "Naive",
        "fhs_mc_logit": "Simulated DD (recal.)", "fhs_mc": "Simulated DD", "fhs": "Filtered hist. sim.",
        "climatology": "Base rate", "hist_var": "Historical VaR", "param_var": "Parametric VaR",
        "unscaled": "No scaling", "oracle": "Oracle (ceiling)", "gjr_cal": "GJR-GARCH-t (cal.)",
        "xgboost_cal": "XGBoost (cal.)", "lstm_cal": "LSTM (cal.)"}
ML = {"rf", "xgboost", "lstm"}


def mean(x) -> float:
    """First number of a 'mean +- std' string."""
    return float(str(x).split("±")[0])


def table(rows: list[list[str]], head: list[str], spec: str) -> str:
    lines = [f"\\begin{{tabular}}{{{spec}}}", "\\hline", " & ".join(head) + " \\\\", "\\hline"]
    lines += [" & ".join(r) + " \\\\" for r in rows]
    lines += ["\\hline", "\\end{tabular}"]
    return "\n".join(lines)


def name(m: str) -> str:
    return NAME.get(m, m)


def tab_vol(c: pd.DataFrame) -> str:
    v = c[c.target == "vol"].copy()
    for k in ("qlike", "rmse", "mse_var", "dir_acc"):
        v[k] = v[k].map(mean)
    v = v.sort_values("qlike")
    rows = [[name(m), f"{q:.3f}", f"{1e3 * mv:.2f}", f"{r:.4f}",
             "--" if pd.isna(d) else f"{100 * d:.1f}"]
            for m, q, mv, r, d in zip(v.model, v.qlike, v.mse_var, v.rmse, v.dir_acc, strict=True)]
    return table(rows, ["Model", "QLIKE", "MSE-var$\\times10^3$", "RMSE", "Dir.\\ acc.\\ (\\%)"],
                 "lrrrr")


def tab_dd(c: pd.DataFrame) -> str:
    d = c[c.target == "dd"].copy()
    d["b"] = d.brier.map(mean)
    d["a"] = d.auc.map(mean)
    d = d.sort_values("b")
    rows = [[name(m), f"{b:.4f}", f"{a:.3f}"] for m, b, a in zip(d.model, d.b, d.a, strict=True)]
    return table(rows, ["Model", "Brier", "ROC-AUC"], "lrr")


def tab_var(c: pd.DataFrame) -> str:
    keep = ["gjr", "combo_hgx", "garch", "fhs", "xgboost", "naive", "lstm", "hist_var",
            "gjr_cal", "xgboost_cal", "lstm_cal"]
    v = c[(c.target == "var") & c.model.isin(keep)].copy()
    v["order"] = v.model.map({m: i for i, m in enumerate(keep)})
    v = v.sort_values("order")
    rows = [[name(r.model), f"{mean(r.pinball) * 1e3:.3f}", f"{100 * mean(r.exceedance):.1f}",
             str(r.exc_range).replace("%", "").replace("-", "--"), str(r.kupiec_reject)]
            for r in v.itertuples()]
    return table(rows, ["Model", "Pinball$\\times10^3$", "Exc.\\ (\\%)", "Yearly range (\\%)",
                        "Kupiec rej."], "lrrrr")


def tab_down() -> str:
    s = pd.read_csv(R / "portfolio_backtest/summary_all.csv").set_index("model")
    s = s.sort_values("tracking_rmse_reach_mean")
    rows = [[name(m), f"{r.tracking_rmse_reach_mean:.4f}", f"{r.ann_vol_mean * 100:.1f}",
             f"{r.sharpe_mean:.2f}", f"{r.max_dd_mean * 100:.1f}", f"{r.turnover_mean:.1f}"]
            for m, r in s.iterrows()]
    return table(rows, ["Sizing forecast", "Track.\\ err.", "Vol (\\%)", "Sharpe", "Max DD (\\%)",
                        "Turn.\\ /yr"], "lrrrrr")


def tab_dm() -> str:
    d = pd.read_csv(R / "round4/tierB/dm_round4.csv")
    d = d[(d.family == "A1") & (d.loss == "qlike")]
    rows = [[name(r.model), "best baseline (GJR)" if r.vs == "best_baseline" else "naive",
             f"{r.mean_loss_diff:+.3f}", f"{r.p_raw:.3f}", f"{r.p_holm:.3f}"]
            for r in d.itertuples()]
    return table(rows, ["ML model", "versus", "$\\Delta$ QLIKE", "raw $p$", "Holm $p$"], "llrrr")


if __name__ == "__main__":
    c = pd.read_csv(R / "round4/tierB/comparison_all.csv")
    for n, t in (("vol", tab_vol(c)), ("dd", tab_dd(c)), ("var", tab_var(c)), ("down", tab_down()),
                 ("dm", tab_dm())):
        (OUT / f"{n}.tex").write_text(t + "\n")
        print("wrote", n)
