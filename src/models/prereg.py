"""Round 4 pre-registered choices (fixed 2026-10-29, before any Tier A or Tier B number existed).

These constants mirror the table in HANDOVER.md ("Round 4 pre-registration"). Tier A reads them;
tests/test_prereg.py pins them and the Round 3 ML hyperparameters, so a later edit has to be a
visible, deliberate change and not a quiet tuning step.
"""

from __future__ import annotations

# A1: volatility proxy and losses (Patton 2011). sigma2_hat = vol_fwd**2, h = vol_hat**2.
VOL_FLOOR = 1e-4  # both annualised vols are floored here; floored rows are counted
LOSS_KINDS = ("se", "qlike", "mse_var")  # vol-SE (RMSE basis), QLIKE, MSE on variance

# A2: Christoffersen / DQ tests
DQ_LAGS = 4
TEST_LEVEL = 0.05
MIN_HITS = 2  # fewer hits than this: a portfolio's independence test is "not testable"

# A3: equal-weight combinations. Nothing is estimated. Members were chosen after Round 3 ranks
# were known (disclosed: mild selection effect possible).
COMBOS: dict[str, tuple[str, ...]] = {
    "combo_hg": ("har", "garch"),
    "combo_hgx": ("har", "garch", "xgboost"),
}

# A4: VIX regimes. Stress = VIX_t above the median of daily VIX from VIX_START to the fold's
# train_end. Calendar view: folds in STRESS_FOLDS against the rest.
VIX_START = "2010-01-05"
STRESS_FOLDS = ("2020", "2022")
MIN_REGIME_DATES = 50  # MCS per regime only above this many dates
EXPECTED_REGIME_COUNTS = {"vix": (128, 205), "calendar": (99, 234)}  # (stress, calm) dates

# G4: Holm adjustment within each family of tests; MCS level
MCS_ALPHA = 0.10

# G3: ML hyperparameters stay exactly as in Round 3.
RF_PARAMS = {"n_estimators": 200, "min_samples_leaf": 2000, "max_samples": 0.3,
             "max_features": 0.5}
XGB_PARAMS = {"n_estimators": 600, "learning_rate": 0.03, "max_depth": 4, "subsample": 0.7,
              "colsample_bytree": 0.7, "min_child_weight": 500}
LSTM_PARAMS = {"hidden": 32, "window": 21, "max_epochs": 30, "patience": 5, "batch_size": 1024,
               "lr": 1e-3}

# Tier B (pre-registered 2026-10-29 before any Tier B number existed; see HANDOVER)
GJR_SPEC = {"p": 1, "o": 1, "q": 1, "dist": "t"}  # B1: GJR-GARCH(1,1,1) with Student t
MODEL_FEATURES = ("har_vol", "garch_vol21", "garch_sigma1")  # B2a
HAR_FIRST_YEAR = 2011  # B2b/B2c: refresh years run HAR_FIRST_YEAR..LAST_REFRESH_YEAR
LAST_REFRESH_YEAR = 2025
HAR_NO_CHANGE_COEF = (0.0, 1.0, 0.0, 0.0)  # B2b: har_vol = vol_21 before any labels exist
SIM_PATHS = 2000  # B3a
SIM_SEED = 42  # B3a: default_rng([SIM_SEED, refresh_year])
RELIABILITY_BINS = 10  # B3b
