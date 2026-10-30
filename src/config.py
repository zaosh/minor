"""Project-wide constants: universe, dates, horizon, seed, paths."""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")  # FRED_API_KEY etc.; real environment variables take precedence
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
ARTIFACTS = ROOT / "models" / "artifacts"

SEED = 42
H = 21  # forward horizon in trading days
LOOKBACKS = (21, 63, 126)
START_DATE = "2010-01-01"
RF_ANNUAL = 0.06  # constant annual risk-free rate (INR) for Sharpe/Sortino
DD_THRESH = 0.05  # drawdown event threshold for dd_prob
VAR_ALPHA = 0.05

N_PORTFOLIOS = 400
K_RANGE = (3, 15)
STRIDE = 5  # sample every 5th trading day to reduce target overlap

MARKET = "^NSEI"

# Market-stress windows for the app's replay (Nifty 50 peak to trough in the cached data).
STRESS_SCENARIOS = {
    "COVID-19 crash (Jan-Mar 2020)": ("2020-01-14", "2020-03-23"),
    "Rate-hike sell-off (Oct 2021 - Jun 2022)": ("2021-10-18", "2022-06-17"),
    "Taper tantrum (May-Aug 2013)": ("2013-05-17", "2013-08-28"),
    "IL&FS credit scare (Aug-Oct 2018)": ("2018-08-28", "2018-10-26"),
}
VIX = "^INDIAVIX"

# {ticker: sector/asset class}. Approved by the team 2026-10-23; all have data from 2010-01-04.
UNIVERSE: dict[str, str] = {
    "TCS.NS": "IT",
    "INFY.NS": "IT",
    "WIPRO.NS": "IT",
    "HCLTECH.NS": "IT",
    "TECHM.NS": "IT",
    "HDFCBANK.NS": "Financials",
    "ICICIBANK.NS": "Financials",
    "SBIN.NS": "Financials",
    "AXISBANK.NS": "Financials",
    "KOTAKBANK.NS": "Financials",
    "INDUSINDBK.NS": "Financials",
    "BAJFINANCE.NS": "Financials",
    "RELIANCE.NS": "Energy",
    "ONGC.NS": "Energy",
    "BPCL.NS": "Energy",
    "NTPC.NS": "Energy",
    "POWERGRID.NS": "Energy",
    "ITC.NS": "Consumer",
    "HINDUNILVR.NS": "Consumer",
    "NESTLEIND.NS": "Consumer",
    "BRITANNIA.NS": "Consumer",
    "ASIANPAINT.NS": "Consumer",
    "TITAN.NS": "Consumer",
    "BHARTIARTL.NS": "Consumer",
    "MARUTI.NS": "Auto",
    "M&M.NS": "Auto",
    "BAJAJ-AUTO.NS": "Auto",
    "EICHERMOT.NS": "Auto",
    "HEROMOTOCO.NS": "Auto",
    "LT.NS": "Industrials",
    "ULTRACEMCO.NS": "Industrials",
    "GRASIM.NS": "Industrials",
    "TATASTEEL.NS": "Industrials",
    "JSWSTEEL.NS": "Industrials",
    "HINDALCO.NS": "Industrials",
    "ADANIPORTS.NS": "Industrials",
    "SUNPHARMA.NS": "Pharma",
    "DRREDDY.NS": "Pharma",
    "CIPLA.NS": "Pharma",
    "DIVISLAB.NS": "Pharma",
    "NIFTYBEES.NS": "ETF: Broad index",
    "JUNIORBEES.NS": "ETF: Next 50 index",
    "BANKBEES.NS": "ETF: Bank index",
    "PSUBNKBEES.NS": "ETF: PSU bank index",
    "GOLDBEES.NS": "ETF: Gold",
}

# {FRED series id: availability lag}; ints are NSE trading days, "35d"/"45d" are calendar days.
# DEXINUS/DCOILBRENTEU are published weekly in practice, so a 1-day lag would not be available live.
FRED_SERIES: dict[str, int | str] = {
    "DGS10": 1,
    "DEXINUS": "10d",
    "DCOILBRENTEU": "10d",
    "FEDFUNDS": "35d",
    "CPIAUCSL": "45d",
}


def fred_api_key() -> str | None:
    """Return the FRED API key from the environment, or None if unset."""
    return os.environ.get("FRED_API_KEY") or None


def set_seeds(seed: int = SEED) -> None:
    """Seed python, numpy and (if installed) torch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
