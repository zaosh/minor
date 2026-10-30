"""Append-only log of saved forecasts, scored against what actually happened.

The log lives in data/forecast_log.csv (gitignored: deleting data/ deletes the log). Rows are
long format, one per (forecast, model). A forecast is saved only on request (the app's "Save
forecast" button or the CLI), never on every page rerun.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src import config
from src.features import targets
from src.portfolios import Portfolio, portfolio_returns

LOG_PATH = config.ROOT / "data" / "forecast_log.csv"
COLUMNS = ["forecast_id", "logged_at_utc", "as_of", "portfolio", "model", "kind", "vol_hat",
           "dd_hat", "var_hat", "vol_now", "model_version"]
PRED = ["vol_hat", "dd_hat", "var_hat"]


def portfolio_key(tickers: tuple[str, ...] | list[str], weights: tuple[float, ...] | list[float]
                  ) -> str:
    """Canonical JSON of a portfolio: sorted tickers, weights rounded to 4 decimals."""
    return json.dumps({t: round(float(w), 4) for t, w in sorted(zip(tickers, weights,
                                                                      strict=True))},
                      separators=(",", ":"))


def forecast_id(as_of: pd.Timestamp, portfolio: str, model_version: str) -> str:
    """Stable 12-character id of a forecast; saving the same forecast twice is a no-op."""
    raw = f"{pd.Timestamp(as_of):%Y-%m-%d}|{portfolio}|{model_version}"
    return hashlib.sha1(raw.encode()).hexdigest()[:12]


def forecast_rows(res: dict, model_version: str) -> pd.DataFrame:
    """Long-format log rows (one per model) from a `predict_portfolio` result."""
    p = res["portfolio"]
    key = portfolio_key(p.tickers, p.weights)
    fid = forecast_id(res["as_of"], key, model_version)
    now = datetime.now(UTC).isoformat(timespec="seconds")
    vol_now = res["baselines"]["naive"]["vol_hat"]
    rows = []
    for kind, group in (("ml", res["ml"]), ("baseline", res["baselines"])):
        for model, v in group.items():
            rows.append({"forecast_id": fid, "logged_at_utc": now,
                         "as_of": pd.Timestamp(res["as_of"]).strftime("%Y-%m-%d"),
                         "portfolio": key, "model": model, "kind": kind,
                         **{c: v.get(c, np.nan) for c in PRED}, "vol_now": vol_now,
                         "model_version": model_version})
    return pd.DataFrame(rows, columns=COLUMNS)


def load_log(path: Path = LOG_PATH) -> pd.DataFrame:
    """Read the log with proper types (empty frame if the file does not exist)."""
    if not Path(path).exists():
        return pd.DataFrame(columns=COLUMNS).astype({c: float for c in PRED + ["vol_now"]})
    df = pd.read_csv(path, dtype={"forecast_id": str, "portfolio": str, "model": str,
                                  "kind": str, "model_version": str})
    df["as_of"] = pd.to_datetime(df["as_of"])
    df["logged_at_utc"] = pd.to_datetime(df["logged_at_utc"], utc=True)
    return df


def append_forecast(res: dict, model_version: str, path: Path = LOG_PATH) -> bool:
    """Append a forecast to the log; returns False (and writes nothing) if already saved."""
    rows = forecast_rows(res, model_version)
    path = Path(path)
    if path.exists() and rows["forecast_id"].iloc[0] in set(load_log(path)["forecast_id"]):
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(path, mode="a", header=not path.exists(), index=False)
    return True


def _outcome(returns: pd.DataFrame, key: str, as_of: pd.Timestamp, h: int) -> dict:
    """Realized outcome of one (portfolio, as_of) with its completion status."""
    nan = {"realized_vol": np.nan, "dd_occurred": np.nan, "next_return": np.nan}
    w = json.loads(key)
    idx = returns.index
    if as_of not in idx or not set(w) <= set(returns.columns):
        return {**nan, "n_after": -1, "status": "unscorable"}
    p = Portfolio(tuple(w), tuple(w.values()))
    r = portfolio_returns(returns, p)
    pos = idx.get_loc(as_of)
    n_after = len(idx) - 1 - pos
    out = {**nan, "n_after": n_after}
    if n_after >= 1:
        out["next_return"] = float(targets.next_return(r).iloc[pos])
    if n_after >= h:
        out["realized_vol"] = float(targets.forward_vol(r, h).iloc[pos])
        out["dd_occurred"] = float(targets.forward_dd_flag(r, h).iloc[pos])
    out["status"] = "complete" if n_after >= h else f"pending {max(n_after, 0)}/{h}"
    return out


def score_log(log: pd.DataFrame, returns: pd.DataFrame, h: int = config.H) -> pd.DataFrame:
    """Attach realized outcomes and errors to each log row.

    VaR is scorable once one trading day has passed; volatility and drawdown once `h` trading
    days of the returns index have passed (so holidays do not count as days).
    """
    if log.empty:
        return log.assign(status=[], n_after=[], realized_vol=[], dd_occurred=[], next_return=[])
    cache: dict[tuple[str, pd.Timestamp], dict] = {}
    for key, as_of in log[["portfolio", "as_of"]].drop_duplicates().itertuples(index=False):
        cache[(key, as_of)] = _outcome(returns, key, as_of, h)
    out = pd.DataFrame([cache[(k, a)] for k, a in zip(log["portfolio"], log["as_of"],
                                                       strict=True)], index=log.index)
    s = pd.concat([log, out], axis=1)
    s["vol_abs_err"] = (s["vol_hat"] - s["realized_vol"]).abs()
    s["dd_brier"] = (s["dd_hat"] - s["dd_occurred"]) ** 2
    s["var_exceeded"] = np.where(s["var_hat"].notna() & s["next_return"].notna(),
                                 s["next_return"] < s["var_hat"], np.nan)
    return s


def summarize_scores(scored: pd.DataFrame) -> pd.DataFrame:
    """Per model: number of scored forecasts and average errors (only what is scorable)."""
    if scored.empty:
        return pd.DataFrame()
    g = scored.groupby("model")
    return pd.DataFrame({
        "n_vol_dd": g["realized_vol"].count(),
        "vol_mae": g["vol_abs_err"].mean(),
        "dd_brier": g["dd_brier"].mean(),
        "n_var": g["var_exceeded"].count(),
        "var_exceedance": g["var_exceeded"].mean(),
    })


def _cmd_add(args: argparse.Namespace) -> None:
    from src import predict

    res = predict.predict_portfolio(args.tickers, args.weights)
    saved = append_forecast(res, predict.load_meta()["model_version"], args.path)
    print(f"{'saved' if saved else 'already in the log'}: as of {res['as_of']:%Y-%m-%d}")


def _cmd_score(args: argparse.Namespace) -> None:
    returns = pd.read_parquet(config.DATA_PROCESSED / "returns.parquet")
    scored = score_log(load_log(args.path), returns)
    if scored.empty:
        print("log is empty")
        return
    print(scored.drop_duplicates("forecast_id")[["forecast_id", "as_of", "status"]].to_string())
    print(summarize_scores(scored).round(4).to_string())


def main() -> None:
    """CLI: `add --tickers .. --weights ..` saves today's forecast; `score` scores the log."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--path", type=Path, default=LOG_PATH)
    sub = ap.add_subparsers(dest="cmd", required=True)
    add = sub.add_parser("add")
    add.add_argument("--tickers", nargs="+", required=True)
    add.add_argument("--weights", nargs="+", type=float, required=True)
    add.set_defaults(fn=_cmd_add)
    sub.add_parser("score").set_defaults(fn=_cmd_score)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
