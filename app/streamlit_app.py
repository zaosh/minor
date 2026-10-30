"""Streamlit dashboard: enter a portfolio, see predicted 21-day risk vs traditional models."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from src import backtest, config, forecast_log, predict, risk_breakdown  # noqa: E402
from src.predict import available_tickers, portfolio_history, predict_portfolio  # noqa: E402

MODEL_LABELS = {"xgboost": "XGBoost", "rf": "Random Forest", "lstm": "LSTM"}
BASE_LABELS = {"naive": "Naive vol", "ewma": "EWMA", "ledoit_wolf": "Ledoit-Wolf covariance",
               "garch": "GARCH(1,1)", "gjr": "GJR-GARCH(1,1) t",
               "fhs": "Filtered hist. sim. (FHS)",
               "hist_var": "Historical VaR", "param_var": "Parametric VaR"}
DEFAULT = {"RELIANCE.NS": 25, "HDFCBANK.NS": 25, "TCS.NS": 20, "NIFTYBEES.NS": 15,
           "GOLDBEES.NS": 15}

SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # dataviz palette slots 1-3 (validated all-pairs)

st.set_page_config(page_title="Portfolio Risk Prediction", layout="wide")


def polish(fig: go.Figure, fmt: str | None = None, height: int = 340,
           hover: str = "x unified") -> go.Figure:
    """Apply the shared chart look: palette, legend on top, light grid, tight margins."""
    fig.update_layout(colorway=SERIES, height=height, margin=dict(l=0, r=0, t=48, b=0),
                      legend=dict(orientation="h", y=1.14, x=0, title_text=""), hovermode=hover)
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.2)", zeroline=False)
    if fmt:
        fig.update_yaxes(tickformat=fmt)
    return fig


@st.cache_data(show_spinner="Predicting...")
def run(tickers: tuple[str, ...], weights: tuple[float, ...]) -> dict:
    return predict_portfolio(list(tickers), list(weights))


def model_label(key: str) -> str:
    """Display name of a model key, with a suffix for conformal-calibrated variants."""
    base = key.removesuffix("_cal")
    name = BT_LABELS.get(base, base)
    return name + (" - calibrated" if key.endswith("_cal") else "")


def fmt_pct(x: float | None) -> str:
    return "n/a" if x is None or pd.isna(x) else f"{x:.1%}"


def default_weights(tickers: list[str], saved: dict[str, float]) -> list[float]:
    """Weight (%) per ticker: the saved weight if the ticker had one, else an equal share.

    Weights are looked up by ticker, never by position, so swapping a ticker cannot hand its
    weight to the ticker that replaced it.
    """
    return [saved.get(t, 100 / len(tickers)) for t in tickers]


def parse_extra(text: str, valid: set[str]) -> tuple[list[str], list[str]]:
    """Split a comma-separated string into (valid, unknown) tickers, adding .NS if missing."""
    good, bad = [], []
    for raw in [t.strip().upper() for t in text.split(",") if t.strip()]:
        t = raw if "." in raw else raw + ".NS"
        (good if t in valid else bad).append(t if t in valid else raw)
    return good, bad


def sidebar() -> tuple[list[str], tuple[float, ...], str] | None:
    """Portfolio inputs. Returns (tickers, weights, ML model key), or None if input is invalid."""
    valid = set(available_tickers())
    with st.sidebar:
        st.header("Portfolio")
        picked = st.multiselect("Tickers", sorted(valid), default=list(DEFAULT))
        extra = st.text_input("Or type tickers (comma-separated)", placeholder="INFY, SBIN")
        extra_ok, extra_bad = parse_extra(extra, valid)
        tickers = list(dict.fromkeys(picked + extra_ok))
        if extra_bad:
            st.warning(f"Ignored unknown tickers: {', '.join(extra_bad)}")
        model_key = st.selectbox("ML model", list(MODEL_LABELS), format_func=MODEL_LABELS.get)
        if not tickers:
            st.info("Pick at least one ticker.")
            return None
        saved = st.session_state.setdefault("weight_map", {t: float(w) for t, w in DEFAULT.items()})
        table = pd.DataFrame({"ticker": tickers, "weight_%": default_weights(tickers, saved)})
        edited = st.data_editor(table, hide_index=True, disabled=["ticker"],
                                key="w_" + "|".join(tickers))
        saved.update(dict(zip(edited["ticker"], edited["weight_%"].astype(float), strict=True)))
        total = float(edited["weight_%"].sum())
        st.write(f"Total weight: **{total:.1f}%**")
        normalize = st.checkbox("Normalize weights to 100%", value=True)
        if st.button("Reload data", help="Re-read the cached data files (refresh them with "
                                         "`python -m src.data.fetch --refresh`)."):
            predict.load_data.cache_clear()
            st.cache_data.clear()
            st.rerun()
    w = edited["weight_%"].astype(float)
    if w.isna().any() or (w < 0).any() or total <= 0:
        st.error("Weights must be non-negative numbers that sum to more than zero.")
        return None
    if not normalize and abs(total - 100) > 0.1:
        st.error(f"Weights sum to {total:.1f}%, not 100%. Fix them or tick 'Normalize'.")
        return None
    keep = w > 0  # zero-weight tickers are not part of the portfolio
    return (list(edited["ticker"][keep]), tuple((w[keep] / w[keep].sum()).round(6)), model_key)


def view_forecast(tickers: list[str], weights: tuple[float, ...], model_key: str) -> None:
    """Forecast view: predicted risk for the sidebar portfolio vs the traditional baselines."""
    try:
        res = run(tuple(tickers), weights)
    except FileNotFoundError as e:
        st.error(str(e))
        return
    except ValueError as e:
        st.error(f"Invalid portfolio: {e}")
        return

    for w in res["warnings"]:
        st.warning(w)
    st.caption(f"Latest delayed daily close: {res['as_of']:%d %b %Y}. Predictions are for the "
               f"following {config.H} trading days.")

    ml, base = res["ml"][model_key], res["baselines"]
    ref_key = "garch" if "garch" in base else "ewma"
    ref_name = BASE_LABELS[ref_key].split(" (")[0]
    ref = base[ref_key]
    var_delta = (None if pd.isna(ml["var_hat"])
                 else f"{-ml['var_hat'] + base['hist_var']['var_hat']:+.2%} vs historical")
    st.subheader(f"Risk band: {res['risk_band'][model_key]}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Volatility (annual)", fmt_pct(ml["vol_hat"]),
              f"{ml['vol_hat'] - ref['vol_hat']:+.1%} vs {ref_name}", delta_color="off")
    c2.metric("Chance of a 5% drawdown", fmt_pct(ml["dd_hat"]),
              f"{ml['dd_hat'] - ref['dd_hat']:+.1%} vs {ref_name}", delta_color="off")
    c3.metric("1-day 95% VaR (loss)", fmt_pct(-ml["var_hat"]), var_delta, delta_color="off")
    day_idx = int(res["features"]["day_idx"].iloc[0])
    cal = backtest.calibrated({**res["ml"], **base}, res["features"], day_idx)
    if pd.isna(ml["var_hat"]):
        st.caption(f"{MODEL_LABELS[model_key]} has no VaR head (n/a); pick XGBoost or LSTM.")
    elif f"{model_key}_cal" in cal:
        st.caption(f"Calibrated VaR (approximate): {fmt_pct(-cal[f'{model_key}_cal']['var_hat'])}. "
                   "The raw VaR is shifted by an offset learned online from this model family's "
                   "past misses in the walk-forward test. It does not guarantee coverage.")
    quartiles = ", ".join(f"{q:.0%}" for q in predict.load_meta()["vol_quartiles"])
    st.caption("Risk band: predicted volatility vs the quartiles of 21-day volatility in the "
               f"training set ({quartiles})")
    if st.button("Save forecast to log"):
        saved = forecast_log.append_forecast(res, predict.load_meta()["model_version"])
        st.success("Saved to data/forecast_log.csv" if saved else "This forecast is already saved.")

    rows = [{"model": MODEL_LABELS[k] + " (ML)", **v} for k, v in res["ml"].items()]
    rows += [{"model": BASE_LABELS[k], **v} for k, v in base.items()]
    rows += [{"model": model_label(k), **v} for k, v in cal.items()]
    cmp_df = pd.DataFrame(rows).set_index("model").astype(float)  # None -> NaN, shown as "-"
    cmp_df["var_loss"] = -cmp_df["var_hat"]
    cmp_df["es_loss"] = -cmp_df["es_hat"]

    st.subheader("Predicted volatility by model")
    vol = cmp_df["vol_hat"].dropna().drop([i for i in cmp_df.index if "calibrated" in i],
                                          errors="ignore").sort_values()
    vol_df = pd.DataFrame({"model": vol.index, "vol": vol.values,
                           "type": ["ML" if "(ML)" in m else "Traditional" for m in vol.index]})
    fig = px.bar(vol_df, x="vol", y="model", color="type", orientation="h",
                 text=vol_df["vol"].map(fmt_pct),
                 color_discrete_map={"ML": SERIES[0], "Traditional": SERIES[1]})
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_xaxes(tickformat=".0%", title_text="", gridcolor="rgba(128,128,128,0.2)")
    fig.update_yaxes(title_text="")
    st.plotly_chart(polish(fig, height=60 + 34 * len(vol_df), hover="closest"), width="stretch")
    with st.expander("All numbers (volatility, drawdown, VaR, expected shortfall)"):
        st.dataframe(
            cmp_df[["vol_hat", "dd_hat", "var_loss", "es_loss"]].rename(columns={
                "vol_hat": "Volatility", "dd_hat": "P(drawdown >= 5%)",
                "var_loss": "95% VaR (1-day loss)", "es_loss": "95% expected shortfall (1-day loss)"
            }).style.format("{:.1%}", na_rep="-"),
            width="stretch",
        )
        st.caption("Expected shortfall is the average loss on the worst 5% of days, beyond the "
                   "VaR. The ML models have no shortfall head. Calibrated rows shift a VaR by an "
                   "online correction; see the README for what that does and does not guarantee.")

    left, right = st.columns([2, 1])
    hist = portfolio_history(res["portfolio"])
    with left:
        st.subheader("Past risk (realized, not predicted)")
        since = st.slider("Show since year", 2011, hist.index.max().year - 1, 2018)
        h = hist.loc[str(since):]
        fig = go.Figure()
        fig.add_scatter(x=h.index, y=h["vol_21"], name="Rolling 21d volatility")
        fig.add_scatter(x=h.index, y=h["drawdown"].abs(), name="Drawdown (abs)", fill="tozeroy",
                        opacity=0.4)
        fig.add_scatter(x=[res["as_of"]], y=[ml["vol_hat"]], mode="markers",
                        name="Predicted vol", marker=dict(size=12, symbol="star",
                                                          line=dict(width=2, color="white")))
        st.plotly_chart(polish(fig, ".0%"), width="stretch")
    with right:
        st.subheader("Composition")
        comp = pd.DataFrame({"ticker": res["portfolio"].tickers,
                             "weight": res["portfolio"].weights})
        fig = px.pie(comp, names="ticker", values="weight", hole=0.5)
        fig.update_traces(textinfo="label+percent", marker=dict(line=dict(width=2, color="white")))
        fig.update_layout(showlegend=False, height=340, margin=dict(l=0, r=0, t=48, b=0))
        st.plotly_chart(fig, width="stretch")

    with st.expander("About the models and limitations"):
        st.markdown(
            "- ML models are trained on 400 random portfolios of NSE stocks/ETFs (3-15 assets) "
            "with walk-forward validation; baselines are naive/EWMA/Ledoit-Wolf vol, GARCH(1,1) "
            "and historical/parametric VaR.\n"
            "- Prices come from Yahoo Finance (delayed); the app shows the latest daily close "
            "in the local cache, refreshed on demand from the CLI. Portfolios are assumed "
            "rebalanced daily.\n"
            "- Training data are today's large caps (survivorship bias) and have no bond asset.\n"
            "- Forecasts can be wrong, especially in regime shifts. Not investment advice."
        )


@st.cache_data(show_spinner="Running the year's models...")
def run_custom_backtest(tickers: tuple[str, ...], weights: tuple[float, ...], d: str) -> dict:
    return backtest.custom_backtest(list(tickers), list(weights), pd.Timestamp(d))


@st.cache_data
def stored_list() -> list[dict]:
    return backtest.stored_portfolios()


@st.cache_data
def stored_hist(pid: int) -> pd.DataFrame:
    return backtest.stored_history(pid)


def _fmt_table(t: pd.DataFrame):
    return t.rename(index=model_label).rename(columns={
        "vol_pred": "Vol predicted", "vol_real": "Vol realized", "p_drawdown": "P(drawdown >= 5%)",
        "drawdown_occurred": "Drawdown occurred", "max_dd": "Max drawdown",
        "var_pred": "VaR predicted", "es_pred": "Shortfall predicted",
        "next_return": "Next-day return",
        "var_exceeded": "VaR exceeded"}).style.format(
        {"Vol predicted": "{:.1%}", "Vol realized": "{:.1%}", "P(drawdown >= 5%)": "{:.1%}",
         "Max drawdown": "{:.1%}", "VaR predicted": "{:.2%}", "Shortfall predicted": "{:.2%}",
         "Next-day return": "{:.2%}",
         "Drawdown occurred": lambda x: "pending" if pd.isna(x) else ("yes" if x else "no")},
        na_rep="-")


BT_LABELS = {**{k: v + " (ML)" for k, v in MODEL_LABELS.items()}, **BASE_LABELS,
             "har": "HAR", "climatology": "Climatology", "combo_hg": "Average: HAR + GARCH",
             "combo_hgx": "Average: HAR + GARCH + XGBoost",
             "fhs_mc": "Simulated drawdown (FHS)",
             "fhs_mc_logit": "Simulated drawdown, recalibrated"}


def view_backtest(tickers: list[str], weights: tuple[float, ...], model_key: str) -> None:
    """Backtest view: what each year's walk-forward models predicted for a past date."""
    try:
        folds = backtest.load_folds()
    except FileNotFoundError as e:
        st.error(f"{e}. A fresh clone has to run the evaluation before this view works.")
        return
    note = "Each year was predicted by models trained only on earlier data (train end shown below)."
    try:
        note += (f" The deployed models are trained up to {predict.load_meta()['label_end_date']}"
                 " and are NOT used here.")
    except FileNotFoundError:
        pass
    st.info(note)
    returns_end = predict.load_data()["returns"].index.max().date()
    source = st.radio("Portfolio", ["Stored portfolio", "Sidebar portfolio"], horizontal=True)
    d = st.date_input("As-of date", value=pd.Timestamp("2022-06-10").date(),
                      min_value=pd.Timestamp(folds[0]["test_start"]).date(), max_value=returns_end)
    d = pd.Timestamp(d)
    try:
        if source == "Stored portfolio":
            ports = stored_list()
            sel = st.selectbox("Stored portfolio", ports, format_func=lambda q: (
                f"#{q['pid']}: {q['label']}" + ("  [held out]" if q["heldout"] else "")))
            res = backtest.stored_backtest(sel["pid"], d, folds)
            tick, wts = sel["tickers"], sel["weights"]
        else:
            res = run_custom_backtest(tuple(tickers), weights, str(d.date()))
            tick, wts = list(res["portfolio"].tickers), list(res["portfolio"].weights)
            sel = None
    except (ValueError, FileNotFoundError) as e:
        st.error(str(e))
        return
    for w in res.get("warnings", []):
        st.warning(w)
    p = predict.build_user_portfolio(tick, wts, available_tickers())
    snapped, fold = res["as_of"], res["fold"]
    st.caption(f"Prediction date used: {snapped:%d %b %Y} (fold {fold['name']}, models trained on "
               f"data up to {fold['train_end']}). Stored portfolios have predictions every 5th "
               "trading day; typed portfolios use the exact date.")
    real = backtest.realized(p, snapped)
    if real["days_after"] < config.H:
        st.warning(f"Only {real['days_after']} trading days have passed since this date; "
                   "volatility and drawdown outcomes are still pending.")
    st.dataframe(_fmt_table(backtest.backtest_table(res["models"], real)), width="stretch")
    if sel is None:
        st.caption("Typed portfolios get this point-in-time view only; pick a stored portfolio "
                   "for the history charts.")
        return
    h = stored_hist(sel["pid"])
    keys = [m.split(":", 1)[1] for m in h.columns if m.startswith(("vol_hat:", "var_hat:"))]
    name = st.selectbox("Model for the charts", list(dict.fromkeys(keys)), index=0,
                        format_func=model_label)
    if f"vol_hat:{name}" in h:
        fig = go.Figure()
        fig.add_scatter(x=h.index, y=h["realized_vol"], name="Realized vol (next 21 days)")
        fig.add_scatter(x=h.index, y=h[f"vol_hat:{name}"], name=f"Predicted: {model_label(name)}")
        fig.update_layout(title="Predicted vs realized volatility, 2020-26")
        st.plotly_chart(polish(fig, ".0%"), width="stretch")
    else:
        st.caption(f"{model_label(name)} is a VaR-only model, so there is no volatility chart.")
    vcol = f"var_hat:{name}"
    if vcol in h and h[vcol].notna().any():
        exc = h[h["next_return"] < h[vcol]]
        fig = go.Figure()
        fig.add_scatter(x=h.index, y=h["next_return"], mode="markers", name="Next-day return",
                        marker=dict(size=4))
        fig.add_scatter(x=h.index, y=h[vcol], name=f"95% VaR: {model_label(name)}")
        fig.add_scatter(x=exc.index, y=exc["next_return"], mode="markers", name="Exceedance",
                        marker=dict(size=9, symbol="x", color="red"))
        fig.update_layout(title=f"VaR vs next-day return ({len(exc)} exceedances of "
                                f"{h[vcol].notna().sum()} days)")
        st.plotly_chart(polish(fig, ".1%"), width="stretch")
    else:
        st.caption(f"{model_label(name)} has no VaR head, so there is no VaR chart.")


def view_log(tickers: list[str], weights: tuple[float, ...], model_key: str) -> None:
    """Forecast log view: saved forecasts and how they turned out."""
    log = forecast_log.load_log()
    if log.empty:
        st.info("No forecasts saved yet. Use 'Save forecast to log' in the Forecast view, or "
                "`python -m src.forecast_log add --tickers .. --weights ..`.")
        return
    returns = predict.load_data()["returns"]
    scored = forecast_log.score_log(log, returns)
    first = scored.drop_duplicates("forecast_id")
    st.subheader("Saved forecasts")
    st.dataframe(first[["forecast_id", "as_of", "portfolio", "status"]], width="stretch",
                 hide_index=True)
    st.subheader("Scores so far")
    summ = forecast_log.summarize_scores(scored)
    st.dataframe(summ.style.format({"vol_mae": "{:.3f}", "dd_brier": "{:.3f}",
                                    "var_exceedance": "{:.0%}"}, na_rep="-"), width="stretch")
    st.caption("Volatility and drawdown are scored after 21 trading days, VaR after one. With a "
               "handful of forecasts these numbers are anecdotes, not evidence. The log lives in "
               "data/forecast_log.csv; deleting data/ deletes it.")
    with st.expander("All rows"):
        st.dataframe(scored, width="stretch", hide_index=True)


def view_risk(tickers: list[str], weights: tuple[float, ...], model_key: str) -> None:
    """Risk breakdown view: contributions of each holding and a replay of past stress periods."""
    p = predict.build_user_portfolio(tickers, list(weights), available_tickers())
    d = predict.load_data()
    returns = d["returns"]
    as_of = returns.index.max()
    st.caption(f"Descriptive, not predicted: based on the last 250 trading days to "
               f"{as_of:%d %b %Y} and today's weights.")
    try:
        vc = risk_breakdown.vol_contributions(returns, p, as_of)
        ec = risk_breakdown.es_contributions(returns, p, as_of)
    except ValueError as e:
        st.error(str(e))
        return
    left, right = st.columns(2)
    with left:
        st.subheader("Volatility contribution")
        fig = px.bar(vc, x="ticker", y="share", labels={"share": "share of portfolio volatility"})
        fig.add_scatter(x=vc["ticker"], y=vc["weight"], mode="markers", name="weight",
                        marker=dict(size=10, symbol="diamond"))
        st.plotly_chart(polish(fig, ".0%", hover="closest"), width="stretch")
        st.caption(f"Contributions add up to the portfolio volatility of "
                   f"{vc['vol_contribution'].sum():.1%} (Ledoit-Wolf covariance). A bar above its "
                   "diamond means the holding adds more risk than its weight.")
    with right:
        st.subheader("Tail-loss (expected shortfall) contribution")
        fig = px.bar(ec, x="ticker", y="share", labels={"share": "share of tail loss"})
        st.plotly_chart(polish(fig, ".0%", hover="closest"), width="stretch")
        st.caption(f"Based on only the worst {int(ec['n_tail_days'].iloc[0])} days of the window, "
                   f"so it is noisy. Average loss on those days: "
                   f"{-ec['es_contribution'].sum():.2%} per day.")

    st.subheader("Stress replay")
    name = st.selectbox("Scenario", list(config.STRESS_SCENARIOS))
    start, end = config.STRESS_SCENARIOS[name]
    try:
        out = risk_breakdown.stress_replay(returns, d["market"][config.MARKET], p, start, end)
    except ValueError as e:
        st.error(str(e))
        return
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Portfolio return", f"{out['total_return']:.1%}")
    c2.metric("Nifty 50 return", f"{out['market_return']:.1%}")
    c3.metric("Max drawdown", f"{out['max_drawdown']:.1%}")
    c4.metric("Worst day", f"{out['worst_day']:.1%}")
    c4.caption(f"on {out['worst_day_date']:%d %b %Y}")
    mk = d["market"][config.MARKET].loc[start:end]
    fig = go.Figure()
    fig.add_scatter(x=out["wealth"].index, y=out["wealth"], name="Your weights")
    fig.add_scatter(x=mk.index, y=mk / mk.iloc[0], name="Nifty 50")
    fig.update_layout(title=f"Growth of 1 from {start} to {end} (not a forecast)")
    st.plotly_chart(polish(fig), width="stretch")
    contrib = out["contributions"].rename("sum of daily contributions").to_frame()
    st.dataframe(contrib.style.format("{:.1%}"), width="stretch")
    st.caption("Today's weights applied to that period's returns, rebalanced daily like the rest "
               "of the project. Sum of daily contributions is arithmetic, so it differs a little "
               "from the compounded portfolio return.")


VIEWS = {"Forecast": view_forecast, "Backtest": view_backtest, "Risk breakdown": view_risk,
         "Forecast log": view_log}


def main() -> None:
    """Page header, sidebar inputs, and the selected view."""
    st.title("Portfolio Risk Prediction")
    st.caption(f"Next-{config.H}-day risk for an NSE portfolio: ML models vs traditional "
               "baselines. Educational demo, not advice.")
    inputs = sidebar()
    if inputs is None:
        return
    view = "Forecast"
    if len(VIEWS) > 1:
        view = st.radio("View", list(VIEWS), horizontal=True, label_visibility="collapsed")
    VIEWS[view](*inputs)


main()
