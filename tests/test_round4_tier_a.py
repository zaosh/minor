"""Round 4 Tier A: variance losses, VaR independence tests, combinations, VIX regimes."""

import math

import numpy as np
import pandas as pd
import pytest

from src import config
from src.models import combos, evaluate, prereg, regimes, significance, var_tests

FROZEN = config.ROOT / "data" / "round3_frozen" / "reports"


# ---- A1: variance losses ------------------------------------------------------------------
def test_variance_losses_hand_values():
    q, m, n = evaluate.variance_losses(np.array([0.2]), np.array([0.1]))
    assert q[0] == pytest.approx(4 - math.log(4) - 1) and q[0] == pytest.approx(1.6137, abs=1e-4)
    assert m[0] == pytest.approx((0.04 - 0.01) ** 2) and n == 0
    q0, m0, _ = evaluate.variance_losses(np.array([0.3]), np.array([0.3]))
    assert q0[0] == pytest.approx(0.0, abs=1e-12) and m0[0] == pytest.approx(0.0, abs=1e-12)


def test_variance_losses_floor_counted_and_nan_raises():
    q, m, n = evaluate.variance_losses(np.array([5e-5, 0.2]), np.array([0.1, 1e-5]))
    assert n == 2 and np.isfinite(q).all() and np.isfinite(m).all()
    with pytest.raises(ValueError):
        evaluate.variance_losses(np.array([0.2, np.nan]), np.array([0.1, 0.1]))


def test_qlike_and_msevar_are_robust_to_a_noisy_proxy_but_vol_mse_is_not():
    rng = np.random.default_rng(0)
    sigma = 0.2
    proxy_var = sigma**2 * rng.chisquare(20, 400_000) / 20  # unbiased noisy variance proxy
    y = np.sqrt(proxy_var)
    grid = sigma * np.linspace(0.9, 1.1, 41)
    ql = [evaluate.qlike(y, np.full_like(y, g)) for g in grid]
    mv = [evaluate.mse_var(y, np.full_like(y, g)) for g in grid]
    vm = [np.mean((y - g) ** 2) for g in grid]
    assert grid[int(np.argmin(ql))] == pytest.approx(sigma, abs=0.0012)
    assert grid[int(np.argmin(mv))] == pytest.approx(sigma, abs=0.0012)
    # vol-level MSE prefers a lower forecast (Jensen: about 0.0025 below sigma at 20 dof)
    assert grid[int(np.argmin(vm))] <= sigma - 0.001


def test_score_vol_adds_qlike_and_msevar_and_keeps_old_metrics():
    y = np.array([0.1, 0.2, 0.15])
    s = evaluate.score("vol", y, y * 1.1, y, None)
    assert {"rmse", "mae", "dir_acc", "qlike", "mse_var"} <= set(s)
    assert s["rmse"] == pytest.approx(evaluate.rmse(y, y * 1.1))


def test_share_rejecting_ignores_untestable_portfolios():
    p = np.array([0.01, 0.20, np.nan, 0.03, np.nan])
    assert var_tests.share_rejecting(p, 0.05) == pytest.approx(2 / 3)  # not 2/5
    assert math.isnan(var_tests.share_rejecting(np.array([np.nan, np.nan]), 0.05))


def test_holm_hand_values():
    assert significance.holm(np.array([0.01, 0.04, 0.03])) == pytest.approx([0.03, 0.06, 0.06])
    assert significance.holm(np.array([0.5])) == pytest.approx([0.5])


@pytest.mark.skipif(not (FROZEN / "predictions.parquet").exists(), reason="frozen copy missing")
def test_recomputed_metrics_match_frozen_metrics():
    p = pd.read_parquet(FROZEN / "predictions.parquet",
                        filters=[("target", "==", "vol"), ("fold", "==", "2021")])
    p = p[p["model"].isin(["naive", "har", "xgboost"])]
    got = evaluate.compute_metrics(p)
    old = pd.read_csv(FROZEN / "metrics_by_fold.csv", dtype={"fold": str})
    m = got.merge(old, on=["fold", "model", "target", "metric"], suffixes=("", "_old"))
    m = m[m["metric"].isin(["rmse", "mae", "dir_acc"])]
    assert len(m) == 9
    np.testing.assert_allclose(m["value"], m["value_old"], rtol=0, atol=1e-12, equal_nan=True)


# ---- A2: Christoffersen and DQ ------------------------------------------------------------
def _ll(n_a, p_a, n_b, p_b):
    t = 0.0
    for n, p in ((n_a, 1 - p_a), (n_b, p_a)):
        t += n * math.log(p) if n else 0.0
    for n, p in ((0, 1 - p_b), (n_b, p_b)):
        t += n * math.log(p) if n else 0.0
    return t


def test_christoffersen_hand_counts():
    hits = np.array([0, 0, 1, 1, 0, 0, 0, 1, 0, 0])
    r = var_tests.christoffersen(hits)
    assert (r["n00"], r["n01"], r["n10"], r["n11"]) == (4, 2, 2, 1)
    p01, p11, p = 2 / 6, 1 / 3, 3 / 9
    alt = 4 * math.log(1 - p01) + 2 * math.log(p01) + 2 * math.log(1 - p11) + math.log(p11)
    null = 6 * math.log(1 - p) + 3 * math.log(p)
    assert r["lr_ind"] == pytest.approx(-2 * (null - alt))
    assert r["lr_cc"] == pytest.approx(r["lr_pof"] + r["lr_ind"])  # identity
    assert r["lr_pof"] == pytest.approx(evaluate.kupiec_pof(3, 10)[0])


def test_christoffersen_no_consecutive_hits_is_finite_and_few_hits_not_testable():
    r = var_tests.christoffersen(np.array([0, 1, 0, 0, 1, 0, 0, 0]))
    assert r["n11"] == 0 and np.isfinite(r["lr_ind"]) and np.isfinite(r["p_cc"])
    assert np.isnan(var_tests.christoffersen(np.array([0, 0, 1, 0]))["lr_ind"])


def test_christoffersen_size_and_power():
    rng = np.random.default_rng(1)
    rej = np.mean([var_tests.christoffersen((rng.random(2000) < 0.05).astype(int))["p_ind"] < 0.05
                   for _ in range(400)])
    assert 0.02 <= rej <= 0.09
    pw = []
    for _ in range(100):
        h, prev = [], 0
        for u in rng.random(2000):
            prev = int(u < (0.4 if prev else 0.05))
            h.append(prev)
        pw.append(var_tests.christoffersen(np.array(h))["p_ind"] < 0.05)
    assert np.mean(pw) > 0.8


def test_dq_per_portfolio_size_and_power():
    rng = np.random.default_rng(2)
    T = 2000
    rej = np.mean([var_tests.dq_test((rng.random(T) < 0.05).astype(int), rng.normal(size=T))["p"]
                   < 0.05 for _ in range(300)])
    assert 0.02 <= rej <= 0.10
    h, prev = [], 0
    for u in rng.random(T):
        prev = int(u < (0.4 if prev else 0.05))
        h.append(prev)
    assert var_tests.dq_test(np.array(h), rng.normal(size=T))["p"] < 0.01


def test_dq_date_level_is_oversized_with_clustered_hits_known_limitation():
    """The registered date-level DQ Wald (Newey-West, chi2(6)) over-rejects under the null when
    hits cluster across portfolios (simulated size about 0.12-0.40 at a nominal 0.05, found in
    Round 4). Its p-values are therefore descriptive only. If this ever fails because the size
    became nominal, the limitation note in HANDOVER and the report must be revisited."""
    rng = np.random.default_rng(3)
    T, N, rho, a = 300, 20, 0.5, 0.05
    from scipy import stats

    crit = stats.norm.ppf(a)
    rej = []
    for _ in range(200):
        z = rng.normal(size=(T, 1))
        hits = (np.sqrt(rho) * z + np.sqrt(1 - rho) * rng.normal(size=(T, N)) < crit)
        r = var_tests.dq_date_level(hits.mean(axis=1) - a, rng.normal(size=T))
        assert np.isfinite(r["stat"]) and r["df"] == 6
        rej.append(r["p"] < 0.05)
    assert np.mean(rej) > 0.10


# ---- A3: combinations ---------------------------------------------------------------------
def _toy_preds():
    rows = []
    dates = pd.to_datetime(["2020-01-03", "2020-01-08"])
    vals = {"har": (0.10, 0.2, -0.020, -0.030), "garch": (0.20, 0.4, -0.010, -0.020),
            "xgboost": (0.30, 0.6, -0.030, np.nan)}
    for model, (v, d, q, e) in vals.items():
        for di, date in enumerate(dates):
            for pid in (0, 1):
                k = 1 + di + 0.1 * pid
                for target, pred in (("vol", v * k), ("dd", d), ("var", q * k), ("es", e * k)):
                    if target == "es" and np.isnan(e):
                        continue
                    aux = q * k if target == "es" else np.nan
                    rows.append({"date": date, "pid": pid, "fold": "2020", "model": model,
                                 "target": target, "y_true": 0.5 if target != "dd" else 1.0,
                                 "y_pred": pred, "vol_now": 0.1, "heldout": pid == 1,
                                 "y_aux": aux})
    return pd.DataFrame(rows)


def test_combos_are_equal_weight_means_and_es_needs_all_members():
    p = combos.add_combos(_toy_preds())
    g = p[(p["model"] == "combo_hgx") & (p["target"] == "vol") & (p["pid"] == 0)].sort_values(
        "date")
    assert g["y_pred"].to_numpy() == pytest.approx([0.2 * 1, 0.2 * 2])  # mean(.1,.2,.3)*k
    assert not ((p["model"] == "combo_hgx") & (p["target"] == "es")).any()
    es = p[(p["model"] == "combo_hg") & (p["target"] == "es")]
    assert len(es) == 4 and (es["y_pred"] <= es["y_aux"]).all()
    v = p[(p["model"] == "combo_hg") & (p["target"] == "var")].sort_values(["date", "pid"])
    assert es.sort_values(["date", "pid"])["y_aux"].to_numpy() == pytest.approx(
        v["y_pred"].to_numpy())  # ES row carries the combo's own VaR
    dd = p[(p["model"] == "combo_hgx") & (p["target"] == "dd")]
    assert dd["y_pred"].to_numpy() == pytest.approx(np.full(len(dd), 0.4))
    assert dd["y_true"].eq(1.0).all() and dd.groupby("pid")["heldout"].first().tolist() == [
        False, True]


def test_combo_of_one_member_equals_that_member_and_mismatch_raises():
    p = _toy_preds()
    one = combos.combine(p, ("har",), "solo")
    har = p[p["model"] == "har"].sort_values(["target", "date", "pid"])
    got = one.sort_values(["target", "date", "pid"])
    assert got["y_pred"].to_numpy() == pytest.approx(har["y_pred"].to_numpy())
    bad = p.drop(p[(p["model"] == "garch") & (p["target"] == "vol")].index[0])
    with pytest.raises(ValueError):
        combos.add_combos(bad)


# ---- A4: VIX regimes ----------------------------------------------------------------------
def _vix():
    idx = pd.bdate_range("2010-01-05", periods=40)
    return pd.Series(np.arange(10.0, 50.0), index=idx)


def test_threshold_is_the_training_median_and_ignores_the_future():
    vix = _vix()
    folds = [{"name": "A", "train_end": str(vix.index[9].date())}]
    thr = regimes.vix_thresholds(vix, folds)
    assert thr["A"] == pytest.approx(np.median(np.arange(10.0, 20.0)))  # 14.5
    vix2 = vix.copy()
    vix2.iloc[10:] = 999.0
    assert regimes.vix_thresholds(vix2, folds)["A"] == thr["A"]


def test_label_dates_hand_example():
    vix = _vix()
    dates = vix.index[[12, 14, 20, 30]]
    lab = regimes.label_dates(pd.Series(["A"] * 4, index=dates), vix, {"A": 14.5})
    assert lab["stress_vix"].tolist() == [True, True, True, True]  # vix 22,24,30,40 > 14.5
    lab = regimes.label_dates(pd.Series(["A"] * 2, index=vix.index[[3, 9]]), vix, {"A": 14.5})
    assert lab["stress_vix"].tolist() == [False, True]  # vix 13 and 19 against 14.5


def test_dm_regime_recovers_group_means_exactly():
    rng = np.random.default_rng(4)
    stress = rng.random(200) < 0.4
    d = 0.3 * stress
    r = significance.dm_regime(d, stress, h=5)
    assert r["stress_mean"] == pytest.approx(0.3, abs=1e-12)
    assert r["calm_mean"] == pytest.approx(0.0, abs=1e-12)
    assert r["n_stress"] == int(stress.sum()) and r["n_calm"] == int((~stress).sum())


def test_dm_regime_size_is_close_to_nominal():
    rng = np.random.default_rng(5)
    rej = []
    for _ in range(300):
        stress = np.repeat(rng.random(67) < 0.4, 5)[:333]
        r = significance.dm_regime(rng.normal(size=333), stress, h=1)
        rej += [r["stress_p"] < 0.05, r["calm_p"] < 0.05]
    assert 0.01 <= np.mean(rej) <= 0.12


@pytest.mark.skipif(not (FROZEN / "predictions.parquet").exists(), reason="frozen copy missing")
def test_design_counts_match_the_preregistered_counts():
    import json

    folds = json.loads((FROZEN / "folds.json").read_text())
    p = pd.read_parquet(FROZEN / "predictions.parquet", columns=["date", "fold", "target"],
                        filters=[("target", "==", "vol"), ("model", "==", "naive")])
    dates = p.drop_duplicates("date").set_index("date")["fold"].sort_index()
    macro = pd.read_parquet(config.DATA_PROCESSED / "macro.parquet")["VIX"]
    lab = regimes.label_dates(dates, macro, regimes.vix_thresholds(macro, folds))
    assert regimes.regime_counts(lab) == {
        "vix": prereg.EXPECTED_REGIME_COUNTS["vix"],
        "calendar": prereg.EXPECTED_REGIME_COUNTS["calendar"]}


# ---- the tierA CLI on synthetic predictions -----------------------------------------------
def _synthetic_world(tmp_path):
    from src.models import round4

    rng = np.random.default_rng(11)
    idx = pd.bdate_range("2010-01-05", "2021-12-31")
    grid = idx[(idx >= "2020-01-03")][::5][:60]
    n_pid = 6
    panel = pd.DataFrame(
        {"ewma_vol": 0.15, "day_idx": idx.get_indexer(np.repeat(grid, n_pid))},
        index=pd.MultiIndex.from_product([grid, range(n_pid)], names=["date", "pid"]))
    base = pd.DataFrame({"date": np.repeat(grid, n_pid), "pid": np.tile(range(n_pid), len(grid)),
                         "fold": np.where(np.repeat(grid, n_pid) < pd.Timestamp("2021-01-01"),
                                          "2020", "2021")})
    base["heldout"] = base["pid"] >= 4
    vol_t = rng.uniform(0.1, 0.3, len(base))
    ret = rng.normal(0, 0.01, len(base))
    dd_t = (rng.random(len(base)) < 0.3).astype(float)
    models = ["naive", "ewma", "ledoit_wolf", "har", "garch", "rf", "xgboost", "lstm"]
    rows = []
    for m in models:
        rows.append(base.assign(model=m, target="vol", y_true=vol_t,
                                y_pred=vol_t * rng.uniform(0.7, 1.3, len(base)),
                                vol_now=0.15, y_aux=np.nan))
        rows.append(base.assign(model=m, target="dd", y_true=dd_t,
                                y_pred=rng.uniform(0.1, 0.6, len(base)), vol_now=0.15,
                                y_aux=np.nan))
    for m in ["naive", "ewma", "ledoit_wolf", "har", "garch", "xgboost", "lstm", "hist_var",
              "param_var", "fhs"]:
        q = -rng.uniform(0.01, 0.02, len(base))
        rows.append(base.assign(model=m, target="var", y_true=ret, y_pred=q, vol_now=0.15,
                                y_aux=np.nan))
        if m not in ("xgboost", "lstm"):
            rows.append(base.assign(model=m, target="es", y_true=ret, y_pred=q * 1.3,
                                    vol_now=0.15, y_aux=q))
    preds = pd.concat(rows, ignore_index=True)
    preds["y_true"] = preds["y_true"].astype("float32")
    from src.models.evaluate import add_calibrated

    preds, _ = add_calibrated(preds, panel)
    pp, fp, mp, vp, metp = (tmp_path / n for n in
                            ("p.parquet", "folds.json", "panel.parquet", "macro.parquet",
                             "metrics.csv"))
    preds.to_parquet(pp)
    panel.to_parquet(mp)
    pd.DataFrame({"VIX": rng.uniform(12, 30, len(idx))}, index=idx).to_parquet(vp)
    import json

    fp.write_text(json.dumps([{"name": "2020", "train_end": "2019-11-28"},
                              {"name": "2021", "train_end": "2020-11-26"}]))
    evaluate.compute_metrics(preds).to_csv(metp, index=False)
    return round4, pp, fp, mp, vp, metp, round4.sha256(pp)


def test_tier_a_end_to_end_guards_and_outputs(tmp_path):
    r4, pp, fp, mp, vp, metp, sha = _synthetic_world(tmp_path)
    out = tmp_path / "out"
    with pytest.raises(ValueError):  # wrong sha: nothing is written
        r4.tier_a(pp, "0" * 64, out, fp, mp, metp, vp, None)
    assert not out.exists()
    man = r4.tier_a(pp, sha, out, fp, mp, metp, vp, None)
    for f in ("metrics_extra.csv", "vol_summary.csv", "dm_round4.csv", "mcs_round4.csv",
              "var_tests_portfolio.csv", "var_tests_pooled.csv", "var_tests_datelevel.csv",
              "combo_metrics.csv", "regime_metrics.csv", "regime_coverage.csv",
              "tests_count.csv", "manifest.json"):
        assert (out / f).exists(), f
    assert man["recompute_vs_frozen_metrics"]["max_abs_diff"] < 1e-12
    assert man["cal_determinism_check"]["max_abs_diff_existing_cal"] < 1e-12
    dm = pd.read_csv(out / "dm_round4.csv")
    assert (dm["p_holm"] >= dm["p_raw"] - 1e-12).all() and (dm["p_holm"] <= 1).all()
    assert (dm["family"] == "A3").sum() == 13 and (dm["family"] == "A1").sum() == 12
    assert (dm["family"] == "A4").sum() == 32
    vt = pd.read_csv(out / "var_tests_datelevel.csv")
    assert len(vt) == 24  # 10 raw + 10 calibrated + 2 combos + 2 calibrated combos
    with pytest.raises(FileExistsError):  # never overwrite
        r4.tier_a(pp, sha, out, fp, mp, metp, vp, None)
    with pytest.raises(ValueError):  # regime counts differ from the pre-registered ones
        r4.tier_a(pp, sha, tmp_path / "out2", fp, mp, metp, vp)
    assert not (tmp_path / "out2").exists()
