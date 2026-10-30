"""Build the project presentation (.pptx) from the saved results and the report figures.

Documentation tooling (needs python-pptx, pandas; not in requirements.txt):
    python docs/deck/build_deck.py
"""

from pathlib import Path

import pandas as pd
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.util import Emu, Inches, Pt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
FIG = ROOT / "docs" / "report" / "figures"
OUT = HERE / "portfolio_risk_ml.pptx"

NAVY, BLUE, ORANGE = RGBColor(0x14, 0x2B, 0x4A), RGBColor(0x00, 0x72, 0xB2), RGBColor(0xD5, 0x5E, 0x00)
GREY, LIGHT, WHITE = RGBColor(0x44, 0x44, 0x44), RGBColor(0xEA, 0xF1, 0xF8), RGBColor(255, 255, 255)

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BLANK = prs.slide_layouts[6]
W, H = prs.slide_width, prs.slide_height
count = 0


def text(slide, x, y, w, h, s, size=18, bold=False, color=GREY, align=PP_ALIGN.LEFT):
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = align
    r = p.add_run()
    r.text = s
    r.font.size, r.font.bold, r.font.color.rgb = Pt(size), bold, color
    return tb


def bullets(slide, x, y, w, h, items, size=20):
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    for i, it in enumerate(items):
        sub = it.startswith("- ")
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(9)
        r = p.add_run()
        r.text = ("    – " + it[2:]) if sub else ("•  " + it)
        r.font.size = Pt(size - 3 if sub else size)
        r.font.color.rgb = GREY
    return tb


def slide(title, notes="", section=None):
    global count
    count += 1
    s = prs.slides.add_slide(BLANK)
    bar = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, W, Inches(1.05))
    bar.fill.solid()
    bar.fill.fore_color.rgb = NAVY
    bar.line.fill.background()
    text(s, Inches(0.55), Inches(0.17), Inches(12.2), Inches(0.8), title, 30, True, WHITE)
    text(s, Inches(12.3), Inches(7.05), Inches(0.8), Inches(0.35), str(count), 11, False, GREY,
         PP_ALIGN.RIGHT)
    text(s, Inches(0.55), Inches(7.05), Inches(8), Inches(0.35),
         "Portfolio Risk Prediction Using Machine Learning", 11, False, GREY)
    if notes:
        s.notes_slide.notes_text_frame.text = notes
    return s


def picture(s, name, x, y, w, h):
    """Place an image inside the box, keeping its aspect ratio and centring it."""
    from PIL import Image
    path = FIG / name
    iw, ih = Image.open(path).size
    scale = min(w / iw, h / ih)
    pw, ph = int(iw * scale), int(ih * scale)
    s.shapes.add_picture(str(path), x + (w - pw) // 2, y + (h - ph) // 2, pw, ph)


def callout(s, x, y, w, h, msg, size=18, fill=LIGHT, edge=BLUE):
    box = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, h)
    box.fill.solid()
    box.fill.fore_color.rgb = fill
    box.line.color.rgb = edge
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(0.2)
    p = tf.paragraphs[0]
    r = p.add_run()
    r.text = msg
    r.font.size, r.font.color.rgb, r.font.bold = Pt(size), NAVY, True


def grid_table(s, x, y, w, rows, widths, size=14, row_h=0.4):
    t = s.shapes.add_table(len(rows), len(rows[0]), x, y, w, Inches(row_h * len(rows))).table
    for j, cw in enumerate(widths):
        t.columns[j].width = Emu(int(w * cw / sum(widths)))
    for i, row in enumerate(rows):
        for j, v in enumerate(row):
            c = t.cell(i, j)
            c.text = str(v)
            para = c.text_frame.paragraphs[0]
            para.alignment = PP_ALIGN.LEFT if j == 0 else PP_ALIGN.RIGHT
            para.runs[0].font.size = Pt(size)
            para.runs[0].font.bold = i == 0
            para.runs[0].font.color.rgb = WHITE if i == 0 else GREY
            c.fill.solid()
            c.fill.fore_color.rgb = NAVY if i == 0 else (LIGHT if i % 2 == 0 else WHITE)
    return t


L, TOP = Inches(0.6), Inches(1.35)
FULL = Inches(12.1)

# 1 title -------------------------------------------------------------------------------
count += 1
s = prs.slides.add_slide(BLANK)
bg = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, W, H)
bg.fill.solid()
bg.fill.fore_color.rgb = NAVY
bg.line.fill.background()
text(s, Inches(0.8), Inches(1.7), Inches(11.7), Inches(2.2),
     "Predicting Portfolio Risk with Machine Learning", 44, True, WHITE)
text(s, Inches(0.8), Inches(3.5), Inches(11.7), Inches(1.0),
     "A walk-forward comparison against GARCH-type baselines on Indian equities and ETFs", 24,
     False, RGBColor(0xBF, 0xD7, 0xEA))
text(s, Inches(0.8), Inches(5.0), Inches(11.7), Inches(1.6),
     "Presented by: [Author 1]   [Author 2]   [Author 3]   [Author 4]\n"
     "Mentor: [Mentor name]\nMinor Specialization Mini Project, Manipal Institute of Technology",
     18, False, WHITE)
s.notes_slide.notes_text_frame.text = (
    "Replace the bracketed names and the mentor before presenting. One-sentence summary: we asked "
    "whether ML predicts portfolio risk better than classic econometric models, and tested it "
    "carefully.")

# 2 problem -----------------------------------------------------------------------------
s = slide("Why portfolio risk prediction?", "Set the stakes. Investors want to know how bad things "
          "can get, not only the expected return. Three questions: how volatile, how likely a "
          "drawdown, how much can I lose in a day.")
bullets(s, L, TOP, Inches(7.2), Inches(5), [
    "Investors care about how bad a portfolio can get, not only its expected return",
    "Three questions over the next 21 trading days:",
    "- How volatile will it be? (volatility)",
    "- How likely is a drawdown of 5% or more? (probability)",
    "- How much can it lose on a bad day? (95% Value-at-Risk)",
    "Classic answers: GARCH, HAR, EWMA, historical VaR",
    "ML promises more flexibility, but is it actually better?"], 20)
callout(s, Inches(8.3), Inches(2.0), Inches(4.4), Inches(2.6),
        "Hansen & Lunde (2005): hard to beat a plain GARCH(1,1) out of sample", 22)

# 3 gap ---------------------------------------------------------------------------------
s = slide("Research gap and our contributions", "The gap: ML-vs-econometrics comparisons often use one "
          "asset, random splits or weak baselines. We fix that at portfolio level with one shared, "
          "strict evaluation.")
text(s, L, TOP, Inches(5.8), Inches(0.5), "The gap", 24, True, BLUE)
bullets(s, L, Inches(1.95), Inches(5.8), Inches(4.5), [
    "Comparisons often use a single asset",
    "Random train/test splits leak the future",
    "Weak baselines make ML look good",
    "Multiple testing is ignored"], 20)
text(s, Inches(6.9), TOP, Inches(5.8), Inches(0.5), "What we do", 24, True, ORANGE)
bullets(s, Inches(6.9), Inches(1.95), Inches(5.8), Inches(4.5), [
    "Portfolio-level model: one model for any user portfolio (400 random portfolios)",
    "Same folds, rows and metrics for every model; leakage tests",
    "Strong baselines incl. GJR-GARCH-t, HAR, simulation",
    "Holm-corrected tests, confidence sets, held-out portfolios",
    "Downstream test: do the forecasts help control risk?"], 20)

# 4 agenda ------------------------------------------------------------------------------
s = slide("Agenda")
bullets(s, Inches(1.2), Inches(1.6), Inches(10), Inches(5), [
    "1.  Data, targets and models", "2.  Evaluation design (no look-ahead)",
    "3.  Results: volatility, drawdown, VaR", "4.  Downstream test: volatility targeting",
    "5.  Discussion, limitations, conclusion"], 28)

# 5 pipeline ----------------------------------------------------------------------------
s = slide("End-to-end pipeline", "Walk through left to right. Emphasise that features at time t use "
          "data up to t and targets use t+1..t+21 only.")
picture(s, "pipeline.png", L, Inches(1.5), FULL, Inches(4.4))
text(s, L, Inches(6.1), FULL, Inches(0.8),
     "Features at t use data up to t. Targets use t+1 .. t+21 only. Scalers are fit on training rows only.",
     18, False, NAVY, PP_ALIGN.CENTER)

# 6 data --------------------------------------------------------------------------------
s = slide("Data and portfolios", "45 NSE instruments; macro lagged by publication delay; changes "
          "instead of levels because trees memorise calendar regimes through levels.")
bullets(s, L, TOP, Inches(7.4), Inches(5.3), [
    "45 liquid NSE instruments: 40 large caps + 5 ETFs (Nifty 50, Next 50, Bank, PSU Bank, gold)",
    "Daily adjusted closes, Jan 2010 to Oct 2026 (4,149 trading days), cached as parquet",
    "India VIX + FRED: US 10Y, USD/INR, Brent, Fed funds, CPI",
    "- each series lagged by its publication delay",
    "- entered as changes, not levels",
    "400 random portfolios: 3-15 assets, Dirichlet weights",
    "310,400 rows (every 5th trading day); 80 portfolios held out of all training"], 19)
callout(s, Inches(8.4), Inches(2.0), Inches(4.3), Inches(2.4),
        "Any user portfolio is scored by the same feature code path", 20)

# 7 targets -----------------------------------------------------------------------------
s = slide("Three forward-looking targets (H = 21 days)", "All three use only returns after t.")
grid_table(s, L, Inches(1.6), FULL, [
    ["Target", "Definition", "Type"],
    ["Volatility", "annualised std of returns t+1 .. t+21", "regression"],
    ["Drawdown", "1 if max drawdown over the next 21 days >= 5%", "probability"],
    ["95% VaR", "5% quantile of the next-day return", "quantile regression"]],
    [2, 7, 3], 20, 0.7)
bullets(s, L, Inches(4.8), FULL, Inches(2), [
    "Volatility model predicts log(vol_fwd / vol_21): the change from today's level",
    "VaR scored by pinball loss and exceedance rate; expected shortfall for baselines (FZ0 loss)"], 20)

# 8 features ----------------------------------------------------------------------------
s = slide("Features", "Lookbacks 21, 63, 126 days. Everything computed from data up to and including t.")
bullets(s, L, TOP, Inches(7.4), Inches(5), [
    "Lookbacks 21, 63 and 126 days, from the portfolio return series:",
    "- rolling volatility, max drawdown, Sharpe, Sortino",
    "- momentum, skewness, kurtosis",
    "- market beta, correlation, mean pairwise correlation of holdings",
    "Market and macro: India VIX (level, change), lagged changes of US yields, FX, oil, CPI",
    "Model-based inputs (HAR, two GARCH forecasts), refit once a year on earlier data"], 20)
callout(s, Inches(8.4), Inches(2.0), Inches(4.3), Inches(2.4),
        "Perturbation tests: changing future data cannot change a feature at t", 19)

# 9 models ------------------------------------------------------------------------------
s = slide("Models and baselines", "The baselines are strong on purpose: GJR-GARCH with t errors, HAR, "
          "simulation. Two equal-weight combinations were added after seeing earlier ranks: "
          "disclosed selection effect.")
text(s, L, TOP, Inches(5.8), Inches(0.5), "Machine learning", 24, True, ORANGE)
bullets(s, L, Inches(1.95), Inches(5.8), Inches(4.8), [
    "Random Forest (vol, drawdown)",
    "XGBoost (vol, drawdown, quantile VaR)",
    "LSTM: one single-layer multi-task network, PyTorch, window 21",
    "Fixed hyperparameters, not tuned; seed 42"], 20)
text(s, Inches(6.9), TOP, Inches(5.8), Inches(0.5), "Traditional + combinations", 24, True, BLUE)
bullets(s, Inches(6.9), Inches(1.95), Inches(5.8), Inches(4.8), [
    "Naive, EWMA, Ledoit-Wolf volatility",
    "HAR, GARCH(1,1), GJR-GARCH with t errors",
    "Historical, parametric and filtered-historical VaR",
    "Base rate and simulated drawdown (2,000 paths)",
    "Combos: HAR+GARCH, HAR+GARCH+XGBoost",
    "Optional conformal VaR correction"], 20)

# 10 evaluation -------------------------------------------------------------------------
s = slide("Evaluation: walk-forward, no look-ahead", "Never a random split. The purge stops a "
          "training target window from reaching into the test period.")
picture(s, "folds.png", L, Inches(1.5), Inches(5.4), Inches(4.8))
bullets(s, Inches(6.4), TOP, Inches(6.4), Inches(5.3), [
    "6 expanding-window folds: test 2020 ... 2025+",
    "21-day purge between train and test",
    "Early stopping on the last 252 training days, then refit on all training rows",
    "Same rows, folds and metrics for every model",
    "80 portfolios never used for training",
    "Mean +/- std across folds"], 20)

# 11 safeguards -------------------------------------------------------------------------
s = slide("Statistical rigour and safeguards", "119 inferential tests; Holm within pre-registered "
          "families. Pre-registration: choices written into code before the numbers existed.")
bullets(s, L, TOP, Inches(7.6), Inches(5.3), [
    "Volatility ranked by QLIKE and MSE on variance (robust to a noisy proxy), plus RMSE",
    "Diebold-Mariano tests on date-level losses (HAC variance), Holm-corrected",
    "90% model confidence set; Kupiec, Christoffersen tests for VaR",
    "Round-4 choices pre-registered in code and pinned by tests",
    "210 automated tests: leakage, purge, hand-computed metrics, train/serve parity"], 20)
callout(s, Inches(8.6), Inches(2.0), Inches(4.1), Inches(2.6),
        "About 190 independent 21-day periods since 2010: power is low, so we say so", 19)

# 12 volatility -------------------------------------------------------------------------
s = slide("Result 1: volatility ranking depends on the loss", "QLIKE: GJR first (0.281), best ML is "
          "LSTM (0.299). RMSE: combo with XGBoost first (0.0697). Fold std ~0.05 is larger than "
          "the gaps. Held-out portfolios give the same picture.")
picture(s, "vol_bars.png", L, Inches(1.35), Inches(8.0), Inches(4.6))
bullets(s, Inches(8.8), Inches(1.5), Inches(4.1), Inches(5), [
    "GJR-GARCH-t is best on QLIKE and MSE-variance",
    "The HAR+GARCH+XGB combo is best on RMSE",
    "Std across folds (~0.05 RMSE) exceeds the gaps",
    "Held-out portfolios: same picture"], 18)
callout(s, L, Inches(6.05), FULL, Inches(0.8),
        "No ML model beats GJR significantly; XGBoost and LSTM beat naive on QLIKE (Holm p = 0.015, 0.008)", 17)

# 13 over time --------------------------------------------------------------------------
s = slide("Volatility forecasts over time", "All models follow the 2020 spike; naive lags turning points.")
picture(s, "vol_time.png", L, Inches(1.4), FULL, Inches(4.8))
text(s, L, Inches(6.3), FULL, Inches(0.7),
     "Calm periods: XGBoost, HAR and the combo lead. Stress: GJR and GARCH lead. No regime test is significant.",
     17, False, NAVY, PP_ALIGN.CENTER)

# 14 significance -----------------------------------------------------------------------------
dm = pd.read_csv(ROOT / "reports/round4/tierB/dm_round4.csv")
dm = dm[(dm.family == "A1") & (dm.loss == "qlike")]
nm = {"rf": "Random Forest", "xgboost": "XGBoost", "lstm": "LSTM"}
rows = [["ML model", "versus", "QLIKE difference", "raw p", "Holm p"]]
for r in dm.itertuples():
    rows.append([nm[r.model], "GJR (best baseline)" if r.vs == "best_baseline" else "naive",
                 f"{r.mean_loss_diff:+.3f}", f"{r.p_raw:.3f}", f"{r.p_holm:.3f}"])
s = slide("Significance: ML beats naive, not the best baseline", "Negative difference favours ML. "
          "Only XGBoost and LSTM versus naive survive Holm correction.")
grid_table(s, L, Inches(1.6), FULL, rows, [3, 3.4, 2.4, 1.4, 1.4], 18, 0.55)
callout(s, L, Inches(5.5), FULL, Inches(1.0),
        "Not significant: any ML model vs GJR. Significant: XGBoost and LSTM vs naive (Holm p < 0.02).", 19)

# 15 drawdown ---------------------------------------------------------------------------
s = slide("Result 2: drawdown probability", "The simulated drawdown model has the best Brier "
          "(0.2046) and is well calibrated; edge over GARCH and base rate does not survive Holm "
          "correction (0.101, 0.052). GJR has best AUC.")
picture(s, "dd_reliability.png", L, Inches(1.4), Inches(5.3), Inches(5.2))
bullets(s, Inches(6.3), Inches(1.5), Inches(6.5), Inches(5), [
    "Brier: simulated drawdown 0.2046, RF 0.2061, LSTM 0.2062, combo 0.2068, base rate 0.2199",
    "Mean prediction 0.335 vs realised 0.318 (GARCH over-predicts: 0.452)",
    "Edge over GARCH (raw p = 0.034) and base rate (0.013) does not survive Holm (0.101, 0.052)",
    "GJR-GARCH has the best ROC-AUC (0.644)"], 19)

# 16 VaR --------------------------------------------------------------------------------
s = slide("Result 3: Value-at-Risk", "Average exceedance near 5% but 2-11% year to year. Pooled Kupiec "
          "rejects in most years but treats dependent portfolio-dates as independent; the date-"
          "clustered test does not reject. Conformal correction narrows the range.")
picture(s, "var_yearly.png", L, Inches(1.4), Inches(5.4), Inches(5.2))
bullets(s, Inches(6.3), Inches(1.5), Inches(6.5), Inches(5), [
    "Pinball loss nearly equal: GJR lowest (1.367e-3)",
    "Average exceedance 4.6-6.0%, but 2%-11% by year",
    "GARCH-type models under-state risk in 2020-21 and over-state it in 2023-25",
    "Pooled Kupiec rejects in most years; the date-clustered test does not reject any model",
    "Conformal calibration: average 4.6-4.9%, narrower range"], 19)

# 17 inputs -----------------------------------------------------------------------------
r34 = pd.read_csv(ROOT / "reports/round4/tierB/ml_round3_vs_round4.csv")
def g(t, m, k):
    x = r34[(r34.target == t) & (r34.model == m) & (r34.metric == k)].iloc[0]
    return f"{x.round3:.3f} -> {x.round4:.3f}"
s = slide("Do HAR and GARCH inputs help the ML models?", "Most metrics improved, but none of the 8 paired "
          "tests is significant (raw p 0.13-0.63). Suggestive only.")
grid_table(s, L, Inches(1.7), FULL, [
    ["Model", "Vol QLIKE", "Drawdown AUC", "Drawdown Brier"],
    ["LSTM", g("vol", "lstm", "qlike"), g("dd", "lstm", "auc"), g("dd", "lstm", "brier")],
    ["XGBoost", g("vol", "xgboost", "qlike"), g("dd", "xgboost", "auc"), g("dd", "xgboost", "brier")],
    ["Random Forest", g("vol", "rf", "qlike"), g("dd", "rf", "auc"), g("dd", "rf", "brier")]],
    [3, 3, 3, 3], 18, 0.7)
callout(s, L, Inches(4.9), FULL, Inches(1.0),
        "Better on most metrics, but none of the 8 paired tests is significant: suggestive only", 19)

# 18 downstream method ------------------------------------------------------------------
s = slide("Downstream test: volatility targeting", "Idea from the sample paper: judge forecasts by "
          "portfolio outcomes. We size exposure from each volatility forecast. This metric was "
          "fixed after a code review and is not pre-registered.")
bullets(s, L, TOP, Inches(7.6), Inches(5.3), [
    "Exposure = min(1, 10% / forecast vol); the rest in cash (6% a year)",
    "Weekly rebalance, buy-and-hold in between, no leverage",
    "15 bps cost on the change from the drifted position",
    "Main metric: RMSE of 63-day realised volatility vs the 10% target, on reachable blocks",
    "Also: Sharpe, max drawdown, turnover; Diebold-Mariano on the 21-day tracking loss"], 20)
callout(s, Inches(8.6), Inches(2.0), Inches(4.1), Inches(2.6),
        "Oracle (true future vol) and no-scaling rows bound the results", 19)

# 19 downstream results -----------------------------------------------------------------
s = slide("Downstream results", "Any forecast brings realised vol to ~10-11% vs 19% unscaled and roughly "
          "halves max drawdown. Models are not reliably ranked: DM favours the combo vs naive, the "
          "realised 63-day metric ranks EWMA/naive/GJR first.")
picture(s, "downstream.png", L, Inches(1.4), Inches(6.2), Inches(5.0))
bullets(s, Inches(7.1), Inches(1.5), Inches(5.8), Inches(5), [
    "Realised vol 10-11% vs 19% unscaled; max drawdown roughly halved",
    "EWMA, naive and GJR rank first; ML and combo are close behind, not ahead",
    "DM favours the combo over naive (Holm p = 0.0006), but the realised metric does not",
    "No model beats GJR; Sharpe 0.65-0.74 is indistinguishable"], 18)

# 19b dashboard -------------------------------------------------------------------------
s = slide("The dashboard (live demo)", "Run: streamlit run app/streamlit_app.py. Enter tickers and "
          "weights, show Forecast, then Backtest for a past date, then Risk breakdown. Add "
          "screenshots here if a live demo is not possible.")
bullets(s, L, TOP, Inches(7.6), Inches(5.3), [
    "Forecast: enter tickers and weights; get volatility, drawdown probability, VaR and a risk band "
    "next to the baselines",
    "Backtest: what each year's walk-forward models predicted for a past date vs what happened",
    "Risk breakdown: which holdings drive volatility and tail loss; replay of past market stress",
    "Forecast log: saved forecasts scored against reality after 1 and 21 trading days"], 20)
callout(s, Inches(8.6), Inches(2.0), Inches(4.1), Inches(2.6),
        "Same feature code as the evaluation: any user portfolio is scored consistently", 19)

# 20 discussion -------------------------------------------------------------------------
s = slide("Discussion: where ML helps and where it does not")
text(s, L, TOP, Inches(5.8), Inches(0.5), "Helps", 24, True, ORANGE)
bullets(s, L, Inches(1.95), Inches(5.8), Inches(4.8), [
    "Beats naive volatility significantly (XGBoost, LSTM on QLIKE)",
    "Leads in calm periods",
    "Best RMSE when combined with HAR + GARCH",
    "Model-based inputs lift ML on most metrics"], 20)
text(s, Inches(6.9), TOP, Inches(5.8), Inches(0.5), "Does not", 24, True, BLUE)
bullets(s, Inches(6.9), Inches(1.95), Inches(5.8), Inches(4.8), [
    "No ML model significantly beats GJR-GARCH-t",
    "Not better on VaR clustering or drawdown calibration",
    "No reliable edge in the downstream test",
    "Consistent with Hansen & Lunde (2005)"], 20)

# 21 limitations ------------------------------------------------------------------------
s = slide("Limitations", "Be upfront: an honest limitations slide is a strength.")
bullets(s, L, TOP, FULL, Inches(5.4), [
    "About 190 independent 21-day periods; 400 portfolios add little independent information",
    "Hyperparameters not tuned; one seed per model in the final configuration",
    "Yahoo data are delayed and sometimes wrong; survivorship bias; no bond asset; US macro only",
    "Portfolios are rebalanced daily; downstream test ignores internal turnover",
    "Some choices (combination members) made after seeing earlier ranks",
    "Deployed app models are weakly trained; all reported numbers come from walk-forward fold models",
    "Educational project, not investment advice"], 19)

# 22 conclusion -------------------------------------------------------------------------
s = slide("Conclusion", "Rigour of comparison over number of models.")
bullets(s, L, TOP, FULL, Inches(4.2), [
    "ML predicts portfolio risk about as well as strong econometric models and better than naive",
    "No ML model significantly beats GJR-GARCH-t after correction",
    "Most dependable gains: combining ML with HAR + GARCH, and simulated drawdowns",
    "Any forecast helps control realised risk; the models are not reliably ranked"], 21)
callout(s, L, Inches(5.5), FULL, Inches(1.0),
        "Takeaway: ML complements, rather than replaces, econometric risk models", 22, RGBColor(0xFD, 0xF1, 0xE7), ORANGE)

# 23 future -----------------------------------------------------------------------------
s = slide("Future work")
bullets(s, L, TOP, FULL, Inches(5), [
    "Tuning and multiple seeds with the HAR/GARCH inputs",
    "Other horizons and assets: bonds, international equities",
    "An expected-shortfall head for the ML models",
    "Regime-aware training and online updating",
    "Dashboard: live forecast log scored against realised outcomes"], 22)

# 24 thanks -----------------------------------------------------------------------------
count += 1
s = prs.slides.add_slide(BLANK)
bg = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, W, H)
bg.fill.solid()
bg.fill.fore_color.rgb = NAVY
bg.line.fill.background()
text(s, Inches(0.8), Inches(2.8), Inches(11.7), Inches(1.2), "Thank you", 54, True, WHITE,
     PP_ALIGN.CENTER)
text(s, Inches(0.8), Inches(4.2), Inches(11.7), Inches(0.8), "Questions?", 28, False,
     RGBColor(0xBF, 0xD7, 0xEA), PP_ALIGN.CENTER)

prs.save(OUT)
print("saved", OUT, count, "slides")
