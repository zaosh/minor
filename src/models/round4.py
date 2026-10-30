"""Round 4 Tier A: extra metrics and tests computed once from SAVED predictions (no refit).

    python -m src.models.round4 tierA --preds data/round3_frozen/reports/predictions.parquet \
        --expect-sha256 <sha> --out reports/round4/tierA

Refuses to overwrite an existing output folder and to run on predictions whose sha256 differs
from the pre-registered one. A bug fix goes to a NEW folder; keep both outputs and log the fix.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src import config
from src.models import combos, evaluate, prereg, regimes, significance, var_tests

FROZEN = config.ROOT / "data" / "round3_frozen" / "reports"
CAL_KEYS = ["date", "pid", "model"]


def sha256(path: Path) -> str:
    """sha256 of a file, read in chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def add_combo_calibration(preds: pd.DataFrame, panel: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Extend frozen predictions (raw + stored `_cal` rows) with the combos and their `_cal`.

    Calibration is recomputed for every raw VaR model through `evaluate.add_calibrated` (so the
    combos get exactly the same eta, lag and held-out handling) and the recomputed `_cal` rows
    of the existing models are compared with the stored ones as a determinism check. Stored
    rows are kept for the existing models. Returns (predictions, check) with the max abs diff.
    """
    raw = preds[~preds["model"].str.endswith("_cal")]
    stored_cal = preds[preds["model"].str.endswith("_cal")]
    raw_ext = combos.add_combos(raw)
    with_cal, _ = evaluate.add_calibrated(raw_ext, panel)
    new_cal = with_cal[with_cal["model"].str.endswith("_cal")]
    old = new_cal[~new_cal["model"].str.startswith("combo_")]
    a = old.sort_values(CAL_KEYS)["y_pred"].to_numpy()
    b = stored_cal.sort_values(CAL_KEYS)["y_pred"].to_numpy()
    same_rows = len(a) == len(b)
    diff = float(np.max(np.abs(a - b))) if same_rows else float("nan")
    combo_cal = new_cal[new_cal["model"].str.startswith("combo_")]
    return pd.concat([raw_ext, stored_cal, combo_cal], ignore_index=True), {
        "n_rows_equal": bool(same_rows), "max_abs_diff_existing_cal": diff}


def check_against_frozen(metrics: pd.DataFrame, frozen_metrics: pd.DataFrame) -> dict:
    """Max abs difference between recomputed and frozen per-fold metrics on shared keys."""
    keys = ["fold", "model", "target", "metric"]
    m = metrics.merge(frozen_metrics, on=keys, suffixes=("", "_frozen"))
    d = (m["value"] - m["value_frozen"]).abs()
    both_nan = m["value"].isna() & m["value_frozen"].isna()
    return {"n_compared": int(len(m)), "n_frozen": int(len(frozen_metrics)),
            "max_abs_diff": float(d[~both_nan].max()), "n_nan_both": int(both_nan.sum())}


def vol_summary(metrics: pd.DataFrame, floored: dict[str, int]) -> pd.DataFrame:
    """Six-year mean of each volatility metric per model, with rank under each metric."""
    v = metrics[metrics["target"] == "vol"].pivot_table(
        index="model", columns="metric", values="value", aggfunc="mean")
    cols = [c for c in ("rmse", "mae", "dir_acc", "qlike", "mse_var") if c in v]
    v = v[cols]
    for c in ("rmse", "mae", "qlike", "mse_var"):
        v[f"rank_{c}"] = v[c].rank().astype(int)
    v["rank_dir_acc"] = v["dir_acc"].rank(ascending=False, na_option="bottom").astype(int)
    v["floored_rows"] = pd.Series(floored)
    return v.sort_values("qlike")


def _best(L: pd.DataFrame, exclude_ml: bool = True) -> str:
    pool = [c for c in L.columns if not significance.is_combo(c) and not c.endswith("_cal")
            and not (exclude_ml and significance.is_ml(c))]
    return L[pool].mean().idxmin()


def combo_dm(preds: pd.DataFrame) -> pd.DataFrame:
    """The 13 pre-registered DM tests on raw combos (A3c): hgx vs hg, hgx vs best single
    baseline, hg vs best single baseline on vol-SE, vol-QLIKE, dd-Brier, var-pinball, plus hg
    vs the best ES baseline on FZ0."""
    rows = []
    specs = [("vol", "se"), ("vol", "qlike"), ("dd", "se"), ("var", "se"), ("es", "se")]
    for target, loss in specs:
        L = significance.date_losses(preds, target, loss)
        h = significance.HORIZON_STRIDE[target]
        best = _best(L)
        pairs = [("combo_hg", best)] if target == "es" else [
            ("combo_hgx", "combo_hg"), ("combo_hgx", best), ("combo_hg", best)]
        for a, b in pairs:
            r = significance.dm_test(L[a].to_numpy(), L[b].to_numpy(), h)
            rows.append({"family": "A3", "target": target, "loss": loss, "model": a,
                         "vs": "combo" if b.startswith("combo_") else "best_baseline",
                         "baseline": b, "mean_loss_diff": r["mean"], "dm_stat": r["stat"],
                         "p": r["p"], "n_dates": r["T"]})
    return pd.DataFrame(rows)


def with_holm(df: pd.DataFrame) -> pd.DataFrame:
    """Add `p_raw` and Holm-adjusted `p_holm` (within the whole frame, one family)."""
    df = df.copy()
    df["p_raw"] = df["p"]
    df["p_holm"] = significance.holm(df["p"].to_numpy())
    return df.drop(columns="p")


def mcs_all(preds: pd.DataFrame) -> pd.DataFrame:
    """90% MCS on date-level losses: vol (SE, QLIKE, MSE-var), dd, var, es; combos included."""
    out = []
    for target, loss in (("vol", "se"), ("vol", "qlike"), ("vol", "mse_var"), ("dd", "se"),
                         ("var", "se"), ("es", "se")):
        m = significance.mcs(significance.date_losses(preds, target, loss), prereg.MCS_ALPHA)
        out.append(m.assign(target=target, loss=loss))
    return pd.concat(out, ignore_index=True)


def tier_a(preds_path: Path, expect_sha: str, out: Path, folds_path: Path, panel_path: Path,
           frozen_metrics_path: Path, macro_path: Path | None = None,
           expected_counts: dict | None = prereg.EXPECTED_REGIME_COUNTS) -> dict:
    """Compute every Tier A output and write them to `out` (which must not exist).

    `expected_counts` is the pre-registered (stress, calm) date counts per regime view; the run
    stops on a mismatch (None skips the check, for synthetic test data only)."""
    if out.exists():
        raise FileExistsError(f"{out} exists; write a fix to a NEW folder and log it")
    got = sha256(preds_path)
    if got != expect_sha:
        raise ValueError(f"predictions sha256 {got} != expected {expect_sha}")
    config.set_seeds()
    preds = pd.read_parquet(preds_path)
    folds = json.loads(folds_path.read_text())
    panel = pd.read_parquet(panel_path, columns=["ewma_vol", "day_idx"])
    vix = pd.read_parquet(macro_path or config.DATA_PROCESSED / "macro.parquet")["VIX"]

    # regime design counts first: stop on a mismatch before anything is written
    dates = preds.drop_duplicates("date").set_index("date")["fold"].sort_index()
    thr = regimes.vix_thresholds(vix, folds)
    lab = regimes.label_dates(dates, vix, thr)
    counts = regimes.regime_counts(lab)
    if expected_counts is not None and counts != expected_counts:
        raise ValueError(f"regime counts {counts} != pre-registered {expected_counts}")

    full, cal_check = add_combo_calibration(preds, panel)
    metrics = evaluate.compute_metrics(full)
    frozen_check = check_against_frozen(metrics, pd.read_csv(frozen_metrics_path,
                                                             dtype={"fold": str}))
    vol = full[full["target"] == "vol"]
    floored = {m: int(evaluate.variance_losses(g["y_true"].to_numpy(),
                                               g["y_pred"].to_numpy())[2])
               for m, g in vol.groupby("model", sort=False)}

    a1 = significance.dm_losses_table(full, ("qlike", "mse_var")).assign(family="A1")
    a3 = combo_dm(full)
    a4 = regimes.regime_dm(full, lab).assign(family="A4")
    dm = pd.concat([with_holm(a1), with_holm(a3), with_holm(a4)], ignore_index=True)
    vt = var_tests.var_tests_table(full)
    mcs_tbl = mcs_all(full)
    reg_mcs = regimes.regime_mcs(full, lab)
    names = ["har", "garch", "xgboost", "combo_hg", "combo_hgx"]
    sel = full[full["model"].str.removesuffix("_cal").isin(names)]
    combo_tbl = evaluate.summarize(evaluate.compute_metrics(sel))
    combo_held = evaluate.summarize(evaluate.compute_metrics(sel[sel["heldout"]]))
    counts_tbl = pd.DataFrame([
        {"family": "A1 (DM)", "n_tests": int((dm["family"] == "A1").sum())},
        {"family": "A2 (date-level DQ)", "n_tests": int(len(vt["datelevel"]))},
        {"family": "A3 (DM)", "n_tests": int((dm["family"] == "A3").sum())},
        {"family": "A4 (regime DM)", "n_tests": int((dm["family"] == "A4").sum())}])
    counts_tbl.loc[len(counts_tbl)] = ["total inferential tests", int(counts_tbl["n_tests"].sum())]
    counts_tbl.loc[len(counts_tbl)] = ["MCS sets (overall)", int(mcs_tbl[["target", "loss"]]
                                                                .drop_duplicates().shape[0])]
    counts_tbl.loc[len(counts_tbl)] = ["MCS sets (per regime)", int(reg_mcs[
        ["target", "view", "regime"]].drop_duplicates().shape[0])]

    out.mkdir(parents=True, exist_ok=False)
    metrics[metrics["metric"].isin(["qlike", "mse_var"])].to_csv(out / "metrics_extra.csv",
                                                                 index=False)
    vol_summary(metrics, floored).to_csv(out / "vol_summary.csv")
    dm.to_csv(out / "dm_round4.csv", index=False)
    mcs_tbl.to_csv(out / "mcs_round4.csv", index=False)
    reg_mcs.to_csv(out / "regime_mcs.csv", index=False)
    for k, v in vt.items():
        v.to_csv(out / f"var_tests_{k}.csv", index=False)
    combo_tbl.to_csv(out / "combo_metrics.csv")
    combo_held.to_csv(out / "combo_metrics_heldout.csv")
    regimes.regime_metrics(full, lab).to_csv(out / "regime_metrics.csv", index=False)
    regimes.regime_coverage(full, lab).to_csv(out / "regime_coverage.csv", index=False)
    lab.to_csv(out / "regime_labels.csv")
    counts_tbl.to_csv(out / "tests_count.csv", index=False)
    manifest = {
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "input_predictions": str(preds_path), "input_sha256": got,
        "python": platform.python_version(),
        "packages": {p: __import__(p).__version__ for p in ("numpy", "pandas", "scipy")},
        "prereg": {"combos": {k: list(v) for k, v in prereg.COMBOS.items()},
                   "vol_floor": prereg.VOL_FLOOR, "dq_lags": prereg.DQ_LAGS,
                   "vix_start": prereg.VIX_START, "stress_folds": list(prereg.STRESS_FOLDS),
                   "min_regime_dates": prereg.MIN_REGIME_DATES, "mcs_alpha": prereg.MCS_ALPHA},
        "vix_thresholds": thr, "regime_counts": {k: list(v) for k, v in counts.items()},
        "floored_vol_rows": floored, "cal_determinism_check": cal_check,
        "recompute_vs_frozen_metrics": frozen_check,
        "known_limitations": [
            "date-level DQ Wald is oversized under clustered hits in simulation "
            "(tests/test_round4_tier_a.py): p-values are descriptive",
            "per-portfolio Christoffersen/DQ have about 17 hits each: descriptive",
            "regime MCS bootstraps non-contiguous dates: approximate"],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def _pair_dm(preds: pd.DataFrame, rows: list[tuple[str, str, str, str]], family: str
             ) -> pd.DataFrame:
    """DM tests for explicit (target, loss, model, baseline) pairs on one predictions frame."""
    out = []
    for target, loss, a, b in rows:
        L = significance.date_losses(preds, target, loss)
        r = significance.dm_test(L[a].to_numpy(), L[b].to_numpy(),
                                 significance.HORIZON_STRIDE[target])
        out.append({"family": family, "target": target, "loss": loss, "model": a, "baseline": b,
                    "mean_loss_diff": r["mean"], "dm_stat": r["stat"], "p": r["p"],
                    "n_dates": r["T"]})
    return pd.DataFrame(out)


def b2_dm(new: pd.DataFrame, old: pd.DataFrame) -> pd.DataFrame:
    """Round 4 vs Round 3 for each ML model on its primary losses (vol-SE, dd-Brier,
    var-pinball): the model with the HAR/GARCH features against the same model without."""
    rows = []
    for target in ("vol", "dd", "var"):
        Ln = significance.date_losses(new, target)
        Lo = significance.date_losses(old, target)
        for m in significance.ML_NAMES:
            if m not in Ln.columns or m not in Lo.columns:
                continue
            both = Ln.index.intersection(Lo.index)
            r = significance.dm_test(Ln.loc[both, m].to_numpy(), Lo.loc[both, m].to_numpy(),
                                     significance.HORIZON_STRIDE[target])
            rows.append({"family": "B2", "target": target, "loss": "se", "model": m,
                         "baseline": f"{m}_round3", "mean_loss_diff": r["mean"],
                         "dm_stat": r["stat"], "p": r["p"], "n_dates": r["T"]})
    return pd.DataFrame(rows)


def dd_reliability(preds: pd.DataFrame, bins: int = prereg.RELIABILITY_BINS) -> pd.DataFrame:
    """Reliability table of every drawdown model: equal-width probability bins with the count,
    mean predicted probability and realised frequency (pooled over folds and portfolios)."""
    d = preds[(preds["target"] == "dd") & ~preds["model"].str.endswith("_cal")].copy()
    d["bin"] = np.minimum((d["y_pred"] * bins).astype(int), bins - 1)
    g = d.groupby(["model", "bin"]).agg(n=("y_true", "size"), mean_pred=("y_pred", "mean"),
                                        mean_true=("y_true", "mean"))
    return g.reset_index()


def ml_round3_vs_round4(new_metrics: pd.DataFrame, old_metrics: pd.DataFrame) -> pd.DataFrame:
    """Six-year mean of every ML metric, Round 3 beside Round 4 (same fold keys)."""
    keep = lambda m: m[m["model"].str.removesuffix("_cal").isin(significance.ML_NAMES)  # noqa
                       & (m["metric"] != "kupiec_p")]
    n = keep(new_metrics).groupby(["target", "model", "metric"])["value"].mean().rename("round4")
    o = keep(old_metrics).groupby(["target", "model", "metric"])["value"].mean().rename("round3")
    t = pd.concat([o, n], axis=1).reset_index()
    t["diff_r4_minus_r3"] = t["round4"] - t["round3"]
    return t


def fallback_counts(folds_dir: Path) -> dict:
    """EWMA fallbacks per fold and model from the per-fold training.json files, if present."""
    out = {}
    for f in sorted(folds_dir.glob("*/training.json")):
        out[f.parent.name] = json.loads(f.read_text()).get("fallbacks", {})
    return out


def tier_b(preds_path: Path, compare_path: Path, out: Path, folds_path: Path,
           macro_path: Path | None = None, folds_dir: Path | None = None,
           expected_counts: dict | None = prereg.EXPECTED_REGIME_COUNTS) -> dict:
    """Tier B outputs from the Round 4 walk-forward predictions (combos and `_cal` rows already
    included by evaluate), with Round 3 predictions as the comparison. `out` must not exist."""
    if out.exists():
        raise FileExistsError(f"{out} exists; write a fix to a NEW folder and log it")
    config.set_seeds()
    full = pd.read_parquet(preds_path)
    old = pd.read_parquet(compare_path)
    need = {"gjr", "fhs_mc", "fhs_mc_logit", "combo_hg", "combo_hgx"}
    if not need <= set(full["model"]):
        raise ValueError(f"predictions lack {sorted(need - set(full['model']))}")
    folds = json.loads(folds_path.read_text())
    vix = pd.read_parquet(macro_path or config.DATA_PROCESSED / "macro.parquet")["VIX"]
    dates = full.drop_duplicates("date").set_index("date")["fold"].sort_index()
    thr = regimes.vix_thresholds(vix, folds)
    lab = regimes.label_dates(dates, vix, thr)
    counts = regimes.regime_counts(lab)
    if expected_counts is not None and counts != expected_counts:
        raise ValueError(f"regime counts {counts} != pre-registered {expected_counts}")

    metrics = evaluate.compute_metrics(full)
    old_metrics = evaluate.compute_metrics(old)
    vol = full[full["target"] == "vol"]
    floored = {m: int(evaluate.variance_losses(g["y_true"].to_numpy(),
                                               g["y_pred"].to_numpy())[2])
               for m, g in vol.groupby("model", sort=False)}
    fam = {
        "A1": significance.dm_losses_table(full, ("qlike", "mse_var")).assign(family="A1"),
        "A3": combo_dm(full),
        "A4": regimes.regime_dm(full, lab).assign(family="A4"),
        "R3rerun": significance.dm_table(full).assign(family="R3rerun"),
        "B1": _pair_dm(full, [("vol", "se", "gjr", "garch"), ("vol", "qlike", "gjr", "garch"),
                              ("var", "se", "gjr", "garch"), ("es", "se", "gjr", "garch")], "B1"),
        "B2": b2_dm(full, old),
        "B3": _pair_dm(full, [("dd", "se", "fhs_mc", "garch"),
                              ("dd", "se", "fhs_mc_logit", "garch"),
                              ("dd", "se", "fhs_mc_logit", "climatology"),
                              ("dd", "se", "fhs_mc_logit", "fhs_mc")], "B3"),
    }
    dm = pd.concat([with_holm(df) for df in fam.values()], ignore_index=True)
    vt = var_tests.var_tests_table(full)
    mcs_tbl = mcs_all(full)
    reg_mcs = regimes.regime_mcs(full, lab)
    counts_tbl = pd.DataFrame([{"family": k, "n_tests": int(len(v))} for k, v in fam.items()])
    counts_tbl.loc[len(counts_tbl)] = ["A2 (date-level DQ)", int(len(vt["datelevel"]))]
    counts_tbl.loc[len(counts_tbl)] = ["total inferential tests", int(counts_tbl["n_tests"].sum())]
    counts_tbl.loc[len(counts_tbl)] = ["MCS sets (overall)", int(
        mcs_tbl[["target", "loss"]].drop_duplicates().shape[0])]
    counts_tbl.loc[len(counts_tbl)] = ["MCS sets (per regime)", int(
        reg_mcs[["target", "view", "regime"]].drop_duplicates().shape[0])]

    out.mkdir(parents=True, exist_ok=False)
    metrics[metrics["metric"].isin(["qlike", "mse_var"])].to_csv(out / "metrics_extra.csv",
                                                                 index=False)
    vol_summary(metrics, floored).to_csv(out / "vol_summary.csv")
    dm.to_csv(out / "dm_round4.csv", index=False)
    mcs_tbl.to_csv(out / "mcs_round4.csv", index=False)
    reg_mcs.to_csv(out / "regime_mcs.csv", index=False)
    for k, v in vt.items():
        v.to_csv(out / f"var_tests_{k}.csv", index=False)
    names = ["har", "garch", "gjr", "xgboost", "combo_hg", "combo_hgx"]
    sel = full[full["model"].str.removesuffix("_cal").isin(names)]
    evaluate.summarize(evaluate.compute_metrics(sel)).to_csv(out / "combo_metrics.csv")
    evaluate.summarize(evaluate.compute_metrics(sel[sel["heldout"]])).to_csv(
        out / "combo_metrics_heldout.csv")
    evaluate.summarize(metrics).to_csv(out / "comparison_all.csv")
    regimes.regime_metrics(full, lab).to_csv(out / "regime_metrics.csv", index=False)
    regimes.regime_coverage(full, lab).to_csv(out / "regime_coverage.csv", index=False)
    lab.to_csv(out / "regime_labels.csv")
    dd_reliability(full).to_csv(out / "dd_reliability.csv", index=False)
    ml_round3_vs_round4(metrics, old_metrics).to_csv(out / "ml_round3_vs_round4.csv", index=False)
    counts_tbl.to_csv(out / "tests_count.csv", index=False)
    manifest = {
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "input_predictions": str(preds_path), "input_sha256": sha256(preds_path),
        "compare_predictions": str(compare_path), "compare_sha256": sha256(compare_path),
        "python": platform.python_version(),
        "packages": {p: __import__(p).__version__ for p in ("numpy", "pandas", "scipy", "arch")},
        "vix_thresholds": thr, "regime_counts": {k: list(v) for k, v in counts.items()},
        "floored_vol_rows": floored,
        "fallbacks_by_fold": fallback_counts(folds_dir or config.ARTIFACTS / "folds"),
        "known_limitations": [
            "date-level DQ Wald is oversized under clustered hits in simulation: descriptive",
            "per-portfolio Christoffersen/DQ have about 17 hits each: descriptive",
            "regime MCS bootstraps non-contiguous dates: approximate",
            "B2 is a second look at the same test years (Round 4 ML is the headline by "
            "pre-registration)"],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main() -> None:
    """CLI entry point (`tierA` subcommand)."""
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("tierA")
    a.add_argument("--preds", type=Path, default=FROZEN / "predictions.parquet")
    a.add_argument("--expect-sha256", required=True)
    a.add_argument("--out", type=Path, default=config.REPORTS / "round4" / "tierA")
    a.add_argument("--folds", type=Path, default=FROZEN / "folds.json")
    a.add_argument("--panel", type=Path, default=config.DATA_PROCESSED / "panel.parquet")
    a.add_argument("--frozen-metrics", type=Path, default=FROZEN / "metrics_by_fold.csv")
    b = sub.add_parser("tierB")
    b.add_argument("--preds", type=Path, default=config.REPORTS / "predictions.parquet")
    b.add_argument("--compare", type=Path, default=FROZEN / "predictions.parquet")
    b.add_argument("--out", type=Path, default=config.REPORTS / "round4" / "tierB")
    b.add_argument("--folds", type=Path, default=config.REPORTS / "folds.json")
    args = ap.parse_args()
    if args.cmd == "tierB":
        m = tier_b(args.preds, args.compare, args.out, args.folds)
        print(json.dumps({k: m[k] for k in ("input_sha256", "compare_sha256", "regime_counts",
                                            "fallbacks_by_fold")}, indent=2))
        return
    m = tier_a(args.preds, args.expect_sha256, args.out, args.folds, args.panel,
               args.frozen_metrics)
    print(json.dumps({k: m[k] for k in ("input_sha256", "regime_counts", "cal_determinism_check",
                                        "recompute_vs_frozen_metrics")}, indent=2))


if __name__ == "__main__":
    main()
