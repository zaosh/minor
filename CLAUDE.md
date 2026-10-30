# CLAUDE.md — Portfolio Risk Prediction Using ML

Minor project (Data Science). Predict a portfolio's future risk with ML, compare against traditional models, and expose it in a Streamlit dashboard.

## Project goal
Given a user-defined portfolio (tickers + weights), predict forward risk over a 21-trading-day horizon and compare ML models (RF, XGBoost, LSTM) against traditional baselines (historical VaR, GARCH, naive volatility).

## Locked decisions (change only with team approval)
- **Horizon H:** 21 trading days.
- **Targets (computed from the portfolio return series, strictly forward-looking):**
  1. `vol_fwd`: annualized realized volatility over the next H days (regression).
  2. `dd_prob`: 1 if max drawdown over the next H days >= 5%, else 0 (classification, output probability).
  3. `var95_fwd`: 95% 1-day VaR estimated via quantile regression (alpha=0.05); evaluate with quantile loss and exceedance rate.
- **Modeling unit:** portfolio-level. Train on many synthetic portfolios (random weights, 3–15 assets) drawn from the universe, so the model generalizes to any user portfolio. Features are computed from the portfolio return series plus market/macro features.
- **Deep learning framework:** PyTorch (not Keras).
- **Universe:** ~30–50 liquid tickers (large-cap equities, sector ETFs, broad indices, gold, bonds). Defined in `config.py`.
- **Date range:** 2010-01-01 to latest. Cache locally as parquet.
- **Language/version:** Python 3.11+.

## Non-negotiable rules
1. **No look-ahead leakage.** Features at time t use data up to and including t only. Targets use t+1..t+H only.
2. **Time-based splits only.** Walk-forward validation (expanding window). Never random train/test split. Leave an H-day gap between train and validation to avoid target overlap.
3. **Fit scalers/transformers on train only**, then apply to val/test.
4. **Fixed seeds** (`SEED=42`) for numpy, sklearn, xgboost, torch.
5. **Same splits and metrics for every model** so comparison is fair. All models and baselines go through `models/evaluate.py`.
6. **Cache data.** yfinance is delayed and rate-limited; never refetch in a loop. FRED needs a free API key via env var `FRED_API_KEY` (never commit keys).
7. **Type hints and docstrings** on public functions. Keep functions small and testable.
8. **Write tests for leakage and feature correctness** before moving to the next stage.
9. Ask before adding new dependencies beyond the stack below.

## Stack
- Data: `yfinance`, `fredapi`, `pandas`, `numpy`, `pyarrow`
- Features/stats: `scipy`, `arch` (GARCH baseline)
- ML: `scikit-learn`, `xgboost`, `torch`
- App: `streamlit`, `plotly`
- Dev: `pytest`, `ruff`, `python-dotenv`

## Repo structure
```
data/            raw/ and processed/ parquet cache (gitignored)
src/
  config.py      universe, dates, H, seed, paths
  data/fetch.py  yfinance + FRED download and caching
  features/engineer.py   rolling vol, drawdown, Sharpe/Sortino, correlations, momentum, macro
  features/targets.py    forward targets (vol_fwd, dd_prob, var95_fwd)
  portfolios.py  synthetic portfolio generator + user-portfolio builder
  models/baselines.py    historical VaR, parametric VaR, GARCH, naive vol
  models/rf.py  models/xgb.py  models/lstm.py
  models/evaluate.py     walk-forward CV, metrics, VaR backtest
app/streamlit_app.py
notebooks/       EDA only
tests/
reports/         figures, tables, final comparison
HANDOVER.md      running status log
```

## Features (computed over lookbacks 21, 63, 126 days)
Rolling std of returns, max drawdown, Sharpe, Sortino, mean pairwise correlation (and avg correlation to market), momentum (returns over lookbacks), skew/kurtosis, plus macro: VIX level and change, interest rate (e.g., 10Y yield/FFR), inflation (CPI YoY, forward-filled with release lag to avoid leakage).

## Metrics
- Regression (`vol_fwd`): RMSE, MAE, directional accuracy (did predicted vol move the same direction as actual vs current).
- Classification (`dd_prob`): ROC-AUC, Brier score, log loss.
- VaR: quantile (pinball) loss, exceedance rate vs 5%, Kupiec POF test.
- Report mean ± std across walk-forward folds.

## Commands
```
pip install -r requirements.txt
python -m src.data.fetch          # download + cache data
pytest -q
python -m src.models.evaluate     # run all models + baselines, write reports/
streamlit run app/streamlit_app.py
ruff check .
```

## Workflow with Claude Code
- Work stage by stage per HANDOVER.md. Finish and test one stage before starting the next.
- After each stage, update HANDOVER.md (done, decisions, next step, open issues).
- Prefer plan first, then implement. Show the plan for any stage touching evaluation or targets.
- If something in this file seems wrong or unworkable, flag it instead of silently changing it.

## Scope control
If time is short: simplify LSTM to a single layer with a short sequence window, and keep the evaluation rigorous. Rigor of comparison matters more than number of models.
