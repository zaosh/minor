"""Seed-noise check: refit the ML models with other seeds on the same folds and holdout.

The headline results use seed 42 (CLAUDE.md). This is an extra robustness analysis: it re-runs
only RF, XGBoost and the LSTM (baselines are deterministic) with other seeds, applies the same
conformal calibration, and writes the per-seed, per-fold metrics and the across-seed spread of
the 6-year means to reports/seeds/. It never touches the main results.

    python -m src.models.seeds --seeds 1 2 3 4 --out reports/seeds_r4   # about 10 min per seed

Round 4: the models are the Round 4 configuration (HAR/GARCH features, LSTM static inputs from
`lstm.STATIC_R4`). `--out` must be a folder that does not hold seed results yet, so the Round 3
seed results (reports/seeds, data/round3_frozen) can never be overwritten.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.models import evaluate
from src.models.lstm import STATIC_R4, LSTMModel
from src.models.rf import RFModel
from src.models.xgb import XGBModel


def seed_models(seed: int) -> list:
    """The three ML models built with `seed`."""
    return [RFModel(seed=seed), XGBModel(seed=seed),
            LSTMModel(seed=seed, static_cols=STATIC_R4)]


def run_seeds(seeds: list[int], panel: pd.DataFrame, dates: pd.DatetimeIndex,
              holdout, years: tuple[int, ...] = evaluate.TEST_YEARS) -> pd.DataFrame:
    """Per-fold metrics for each seed (long format with a `seed` column). `panel` must already
    carry the Round 4 model features (see `evaluate.load_panel_with_features`)."""
    folds = evaluate.make_folds(panel, dates, years)
    parts = []
    for seed in seeds:
        preds = evaluate.run_walk_forward(seed_models(seed), panel, folds, holdout=holdout)
        preds, _ = evaluate.add_calibrated(preds, panel)
        parts.append(evaluate.compute_metrics(preds).assign(seed=seed))
    return pd.concat(parts, ignore_index=True)


def spread(metrics: pd.DataFrame) -> pd.DataFrame:
    """Across-seed mean and std of each model's 6-year mean metric."""
    m = metrics[metrics["metric"] != "kupiec_p"]
    per_seed = m.groupby(["target", "model", "metric", "seed"])["value"].mean()
    out = per_seed.groupby(["target", "model", "metric"]).agg(["mean", "std", "min", "max"])
    return out.reset_index()


def main(seeds: list[int], out: Path) -> None:
    """Run the seeds and write seed_metrics.csv and seed_spread.csv into the new folder `out`."""
    if (out / "seed_metrics.csv").exists():
        raise FileExistsError(f"{out} already holds seed results; choose a new --out folder")
    panel, dates, holdout = evaluate.load_panel_with_features(quick=False)
    metrics = run_seeds(seeds, panel, dates, holdout)
    out.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(out / "seed_metrics.csv", index=False)
    spread(metrics).to_csv(out / "seed_spread.csv", index=False)
    print(spread(metrics).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--out", type=Path, required=True, help="NEW output folder")
    args = ap.parse_args()
    main(args.seeds, args.out)
