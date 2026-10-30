"""Equal-weight forecast combinations built post hoc from stored predictions (nothing fitted).

Volatility: mean of `vol_hat` (a mean of vols is not the vol of the mean variance; Jensen is
noted, not corrected). Drawdown: linear pool, the mean of each member's probability. VaR: mean
of the quantiles. ES: mean of `es_hat`, only when every member has an ES forecast; the ES row
carries the combo's own VaR (mean of the members' VaR) in `y_aux`, so ES <= VaR still holds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.models import prereg

KEY = ["date", "pid", "fold"]


def combine(preds: pd.DataFrame, members: tuple[str, ...], name: str) -> pd.DataFrame:
    """Rows for model `name`: the equal-weight mean of `members` for every target they all
    cover. Members must have identical (date, pid, fold) rows and realisations per target;
    otherwise ValueError. `y_true`, `vol_now` and `heldout` come from the first member."""
    out = []
    for target, g in preds[preds["model"].isin(members)].groupby("target", sort=False):
        parts = {m: g[g["model"] == m].sort_values(["date", "pid"]) for m in members}
        if any(len(p) == 0 for p in parts.values()):
            continue  # not every member covers this target (e.g. ES for the ML models)
        first = parts[members[0]]
        for m, p in parts.items():
            same = (len(p) == len(first) and np.array_equal(p[KEY].to_numpy(),
                                                           first[KEY].to_numpy()))
            if not same or not np.array_equal(p["y_true"].to_numpy(), first["y_true"].to_numpy()):
                raise ValueError(f"combo {name}: member {m} rows differ from {members[0]} "
                                 f"for target {target}")
        row = first.copy()
        row["model"] = name
        row["y_pred"] = np.mean([p["y_pred"].to_numpy() for p in parts.values()], axis=0)
        if target == "es":
            row["y_aux"] = np.mean([p["y_aux"].to_numpy() for p in parts.values()], axis=0)
        out.append(row)
    return pd.concat(out, ignore_index=True)


def add_combos(preds: pd.DataFrame,
               specs: dict[str, tuple[str, ...]] = prereg.COMBOS) -> pd.DataFrame:
    """Append the pre-registered combinations to a predictions frame (raw members only)."""
    return pd.concat([preds, *[combine(preds, m, n) for n, m in specs.items()]],
                     ignore_index=True)


def combine_models(models: dict[str, dict], members: tuple[str, ...]) -> dict[str, float]:
    """The same equal-weight combination for one portfolio-date: {model: {head: value}} in,
    {head: mean} out for every head ({vol,dd,var,es}_hat) that ALL members cover."""
    out = {}
    for head in ("vol_hat", "dd_hat", "var_hat", "es_hat"):
        vals = [models.get(m, {}).get(head, np.nan) for m in members]
        if all(pd.notna(v) for v in vals):
            out[head] = float(np.mean(vals))
    return out
