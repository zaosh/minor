# Portfolio Risk Prediction Using Machine Learning

Minor project (Data Science). Predicts a portfolio's risk over the next **21 trading days** for
NSE stocks/ETFs and compares ML models (Random Forest, XGBoost, one multi-task PyTorch LSTM)
against traditional baselines (naive/EWMA/Ledoit-Wolf volatility, HAR, GARCH(1,1),
GJR-GARCH(1,1) with Student-t errors, filtered historical simulation, historical and parametric
VaR, drawdown base rate and a simulated-drawdown model), plus equal-weight combinations of ML and
traditional forecasts.

Targets: annualized realized volatility, probability of a >= 5% drawdown, and 95% 1-day VaR
(plus 95% expected shortfall for the baselines).

## Run it
Python 3.12 (3.11+ should work).
```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # 16 top-level pins; plain torch picks the default wheel
cp .env.example .env              # optional: add FRED_API_KEY (the public CSV is used without it)
python -m src.data.fetch          # download + cache prices and macro (data/raw, data/processed)
python -m src.features.dataset    # build the modelling panel (400 synthetic portfolios)
python -m src.features.model_features  # yearly-refit GARCH features + simulated drawdown (~2 min)
python -m src.models.evaluate     # walk-forward evaluation, per-fold models, tests (about 11 min)
python -m src.models.train_final  # fit the deployed models for the app; run BEFORE report
python -m src.models.report       # reports/comparison.md + reports/figures/*.html
streamlit run app/streamlit_app.py
pytest -q && ruff check .
```
- `uv` works too: `uv venv --python 3.12 .venv && uv pip install -r requirements.txt`.
- CPU-only machines: `pip install torch --index-url https://download.pytorch.org/whl/cpu` first.
  `requirements-lock-linux-cu126.txt` is the exact Linux/CUDA 12.6 freeze the results came from.
- `python -m src.models.evaluate --quick` is a smoke test (2 folds, 50 portfolios). It writes to
  `reports/quick/` and never touches the full results.
- `python -m src.models.seeds --seeds 1 2 --out reports/<new folder>` re-runs only the ML models
  with other seeds (about 10 min per seed); it measures seed noise and never touches the main
  results (seed 42). `reports/seeds/` holds seeds 1 and 2 for the Round 3 ML configuration; the
  Round 4 configuration below has not been re-seeded.
- `python -m src.models.round4 tierA ...` / `tierB ...` compute the Round 4 extra metrics and tests
  (QLIKE, VaR independence tests, combinations, regimes, Round 3 vs Round 4) into
  `reports/round4/`. They refuse to overwrite an existing output folder.
- **A fresh clone must run `model_features`, `evaluate` and `train_final` before the app's
  Backtest view and forecasts work.** `data/`, `models/artifacts/` and
  `reports/predictions.parquet` are gitignored.

## The app
Four views (radio at the top):
- **Forecast**: enter tickers and weights; get predicted volatility, drawdown probability, VaR and a
  Low / Moderate / High / Very high risk band (predicted vol against training-set quartiles),
  next to the baselines, including expected shortfall and an approximate calibrated VaR.
  "Save forecast to log" stores it.
- **Backtest**: what each year's walk-forward models predicted for a past date (2020 onwards),
  next to what happened. Works for the 400 stored portfolios (with history charts) and for the
  sidebar portfolio (point-in-time table). It uses models trained only on earlier data; the
  deployed models are never used for past dates.
- **Risk breakdown**: which holdings drive the portfolio's volatility (Euler split with a
  Ledoit-Wolf covariance, last 250 days) and its tail loss (split of the historical expected
  shortfall, only ~13 tail days), plus a replay of today's weights through four past Nifty
  peak-to-trough windows (2013 taper tantrum, 2018 IL&FS, COVID 2020, 2021-22 sell-off). It is
  descriptive and not a forecast.
- **Forecast log**: saved forecasts scored against reality: VaR after 1 trading day, volatility and
  drawdown after 21. Stored in `data/forecast_log.csv`; **deleting `data/` deletes the log.** CLI:
  `python -m src.forecast_log add --tickers TCS.NS INFY.NS --weights 0.5 0.5` and `... score`.

Data is the latest *delayed* daily close in the local cache, refreshed on demand with
`python -m src.data.fetch --refresh` (falls back to the old cache if any download fails), then
rebuild the panel. The app's "Reload data" button re-reads the files. It is not a real-time feed.

## Method in brief
- **Data:** 45 NSE tickers (40 large caps + 5 index/sector/gold ETFs), 2010 to present, adjusted
  close from Yahoo Finance; India VIX; FRED US 10Y, USD/INR, Brent, Fed funds, CPI YoY lagged by
  their publication delay (10 calendar days for FX/oil, which publish weekly). Rates, inflation,
  FX and oil enter as *changes* (not levels, which let trees memorise calendar regimes).
- **Training unit:** 400 random portfolios (3-15 assets, Dirichlet weights) on a shared date grid
  (every 5th day). Features come from the portfolio return series (rolling vol, drawdown,
  Sharpe/Sortino, momentum, skew/kurtosis, market beta/correlation, mean pairwise correlation over
  21/63/126 days) plus macro. Any user portfolio is scored with the same code.
- **Targets:** `vol_fwd` (modelled as `log(vol_fwd/vol_21)`, so models predict the *change* from
  today's volatility), `dd_flag`, next-day return for the quantile VaR.
- **Models:** RF (vol, drawdown), XGBoost (vol, drawdown, quantile VaR), and ONE single-layer
  multi-task LSTM (window 21, three heads). Hyperparameters were set once by reasoning and not
  tuned; one seed. The ML models have no expected-shortfall head. Since Round 4 the ML models
  also get three model-based inputs: the HAR forecast and two GARCH(1,1) forecasts (21-day and
  next-day), re-estimated once a year on data before that year.
- **Traditional baselines:** HAR (OLS on log vol over 21/63/126 days), GARCH(1,1) and
  GJR-GARCH(1,1)-t (asymmetric, fat-tailed), EWMA, Ledoit-Wolf. Drawdown probability also comes
  from 2,000 simulated GARCH paths per portfolio and date (`fhs_mc`, with a logistic recalibration
  `fhs_mc_logit`). `combo_hg` averages HAR and GARCH; `combo_hgx` averages HAR, GARCH and XGBoost.
- **VaR/ES baselines:** filtered historical simulation (GARCH(1,1) filter with the empirical
  quantile of its standardized residuals), plus normal, Student-t (GJR), historical and parametric
  versions. Expected shortfall is scored with the joint FZ0 loss.
- **Conformal calibration:** an online correction (quantile tracking, step 0.25 chosen by
  reasoning, one-day feedback delay) shifts every VaR forecast, baselines included, so a model
  that misses too often gets deeper and one that is too cautious gets shallower. Held-out
  portfolios never feed it. It only calibrates the average miscoverage over time; it gives no
  guarantee for a single year or regime.
- **Validation:** 6 expanding-window folds testing 2020, ..., 2025+, with a 21-day purge between
  train and test. XGBoost trains 600 rounds per head and takes the validation-curve minimum; the
  LSTM keeps, for each head, the weights of that head's best validation epoch (last 252 training
  days). Both are then refit on all training rows for those round/epoch counts; scalers are fit
  on training rows only; seed 42. 80 portfolios (seed 42) are excluded from every training row and
  scored separately. Every model and baseline goes through `src/models/evaluate.py` on identical
  rows. The deployed XGBoost VaR head (app only) uses the median round count of the six fold
  models (388), because the single last-year validation curve of the deployed fit picked 2.
- **Losses:** volatility is ranked by QLIKE and MSE on variance (Patton 2011, robust to a noisy
  volatility proxy), with RMSE, MAE and directional accuracy also reported.
- **Significance:** date-level Diebold-Mariano tests (HAC variance) with Holm correction within
  each pre-registered family of tests, a 90% model confidence set, a date-clustered VaR coverage
  test and Christoffersen independence / conditional-coverage tests, on the ~330 test dates (rows
  are not independent). All Round 4 choices were written down before any Round 4 number existed
  (`src/models/prereg.py`, pinned by `tests/test_prereg.py`).
- **Tests:** leakage tests (changing future data cannot change features, baselines or backtests at
  t; targets use only t+1..t+21), hand-computed metrics, fold purge checks, train/serve parity
  (also on synthetic data), forecast-log scoring.

## Results (Round 4; mean over 6 test years; full tables in `reports/comparison.md` and `reports/round4/`)
| Target | Result | Notes |
|---|---|---|
| Volatility, QLIKE (main ranking loss) | GJR 0.281, combo_hgx 0.298, LSTM 0.299, XGBoost 0.306, GARCH 0.309, RF 0.338, HAR 0.361, naive 0.399 | No ML model beats the best baseline (GJR) significantly (raw DM p 0.36-0.55). XGBoost and the LSTM do beat naive volatility significantly (Holm-adjusted p 0.015 and 0.008; RF 0.074). |
| Volatility, MSE on variance | GJR 0.00497, LSTM 0.00518, GARCH 0.00527, combo_hgx 0.00529, XGBoost 0.00547 | No ML model beats GJR or naive significantly on this loss. |
| Volatility, RMSE | combo_hgx 0.0697, XGBoost 0.0710, LSTM 0.0718, RF 0.0724, HAR 0.0728, GJR 0.0734, GARCH 0.0748, naive 0.0834 | Std across folds about 0.04-0.05. The ranking depends on the loss: the combination is best on RMSE, GJR on QLIKE and MSE-variance. Directional accuracy: LSTM 68%, the rest 62-67%. |
| Drawdown, Brier | fhs_mc_logit 0.2046, fhs_mc 0.2050, RF 0.2061, LSTM 0.2062, combo_hgx 0.2068, XGBoost 0.2101, base rate 0.2199, GARCH 0.2225 | The simulated-drawdown model is best and well calibrated (mean prediction 0.335 vs realised 0.318; GARCH 0.452), but its edge over GARCH (raw p 0.034) and the base rate (0.013) does not survive Holm (0.10, 0.052). ML models are equally well calibrated (0.33). |
| Drawdown, ROC-AUC | GJR 0.644, combo_hgx 0.634, LSTM 0.632, GARCH 0.632, fhs_mc 0.629, XGBoost 0.616, RF 0.607 | GJR ranks risky periods best. |
| 95% VaR | Pinball: GJR 0.001367, combo_hgx 0.001384, GARCH 0.001392, FHS 0.001394, XGBoost 0.001411, LSTM 0.001457 | Average exceedance 4.6%-6.3% but about 2%-11% across years. The pooled Kupiec test rejects in most years for most models, but it treats ~20,000 portfolio-days per year as independent. The date-clustered test does not reject the average rate for any model (p 0.27-0.95). |
| VaR after calibration | Average exceedance 4.6%-4.9% | Yearly range narrows (e.g. GJR 2.9%-7.2%, XGBoost 2.6%-7.6%); pinball unchanged or slightly worse. |
| VaR clustering (Christoffersen independence, per portfolio) | 1%-3% of portfolios reject for most models including XGBoost (Ledoit-Wolf 3.3%, historical VaR 4.8%); LSTM 6% (2.5% calibrated) | In Round 3 the LSTM rejected in 49% of portfolios. |
| Expected shortfall | Best FZ0: GJR, then EWMA, combo_hg, FHS, GARCH (within noise) | GJR (t errors) and FHS get the tail size about right (realised/predicted 1.02-1.03; historical 0.97); normal-based models understate it by 11-14%. Few exceedances, low power. |
| Model confidence set (90%) | Every model is in the set for every target and loss, except naive under QLIKE | ~330 dates cannot separate the models. |
| Calm vs stress (India VIX above its training median) | Stress (128 dates), vol RMSE: GJR 0.108, GARCH 0.113, combo_hgx 0.114, LSTM 0.114. Calm (205 dates): XGBoost 0.048, combo_hgx 0.049, HAR 0.049 | Traditional models lead in stress, ML in calm, but no regime test is significant and regimes are confounded with test year. |
| Held-out portfolios | Same picture | On the 80 portfolios never used for training: vol QLIKE GJR 0.288, LSTM 0.305, XGBoost 0.312, naive 0.409; drawdown Brier fhs_mc 0.202, RF 0.204. |

**Round 3 -> Round 4 (ML models, after adding the HAR/GARCH inputs).** Most metrics improved,
e.g. LSTM QLIKE 0.321 -> 0.299, vol RMSE 0.0746 -> 0.0718, drawdown AUC 0.606 -> 0.632; XGBoost
QLIKE 0.317 -> 0.306, AUC 0.584 -> 0.616; RF AUC 0.582 -> 0.607. None of the 8 Round 4 vs Round 3
tests is significant (raw p 0.13-0.63). Full table: `reports/round4/tierB/ml_round3_vs_round4.csv`.

Takeaway: ML is competitive with, not better than, strong econometric models. GJR-GARCH is the
best single volatility model on the losses that are robust to a noisy volatility proxy;
combining XGBoost with HAR and GARCH gives the lowest RMSE; the ML models significantly beat
naive volatility but not the best traditional model; and the simulated-drawdown model gives the
best-calibrated drawdown probabilities. This matches the published finding that HAR/GARCH-type
models are hard to beat at a one-month horizon with daily data.

## Limitations
- Only about 190 independent 21-day periods exist since 2010, and portfolios share market days, so
  fold-to-fold variance is large and 400 portfolios add little independent information.
- Hyperparameters were not tuned; one seed per model. For the Round 3 ML configuration, seeds 1
  and 2 moved the LSTM noticeably (vol RMSE 0.070-0.075, Brier 0.208-0.217) while XGBoost and RF
  barely moved; the Round 4 configuration has not been re-seeded. Conformal step size 0.25 was
  fixed beforehand; 0.1 and 0.5 are reported only as sensitivity.
- The deployed LSTM's volatility and drawdown heads stop at epoch 1 (as in Round 3); the app's
  models are for demonstration, and all reported numbers come from the walk-forward fold models.
- The pre-registered date-level DQ test is oversized in simulation (12-40% rejections at a
  nominal 5%), so its results are reported as descriptive only.
- The DM, confidence-set and coverage tests assume stationary loss differences, which expanding
  windows do not guarantee, and ~330 dates give them low power.
- Yahoo data is delayed and occasionally wrong (bad prints are zeroed and logged in
  `reports/return_jumps.csv`); today's large caps imply survivorship bias; no bond asset (Indian
  gilt ETFs lack history to 2010); portfolios are rebalanced daily; US macro series only.
- Pooled Kupiec tests overstate sample size because exceedances are dependent across portfolios;
  the date-clustered test is the better guide. The ES checks have very few exceedance dates.
- Forecast-log scores from a handful of forecasts are anecdotes, not evidence.
- Educational project, not investment advice.
