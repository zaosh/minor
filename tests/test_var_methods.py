"""FHS, expected shortfall, conformal calibration and the date-level significance tests."""

import numpy as np
import pandas as pd
import pytest

from src import config
from src.models import baselines, conformal, evaluate, significance

ALPHA = config.VAR_ALPHA


def _test_rows(world, year=2021, pids=None):
    panel, idx, R = world
    f = evaluate.make_folds(panel, idx, test_years=(year,))[0]
    test = panel.iloc[f.test]
    return test, R


# ---- expected shortfall producers -------------------------------------------------------
def test_hist_es_hand_computed():
    r = np.array([-5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0])
    # window 5 at t=4 is [-5..-1]; its 40% quantile is -3.4, so the tail is {-5, -4}
    got = baselines.hist_es_at(r, np.array([4]), window=5, alpha=0.4)
    assert got[0] == pytest.approx(-4.5)


def test_hist_es_uses_the_same_window_and_quantile_as_hist_var(world):
    panel, idx, R = world
    r = R[0]
    t = 800
    var = pd.Series(r).rolling(250).quantile(ALPHA).iloc[t]
    es = baselines.hist_es_at(r, np.array([t]))[0]
    w = r[t - 249 : t + 1]
    assert var == pytest.approx(np.quantile(w, ALPHA))
    assert es == pytest.approx(w[w <= var].mean()) and es <= var


def test_param_es_formula_and_normal_constant():
    assert baselines.ES_NORMAL == pytest.approx(-2.0627, abs=1e-3)
    r = np.random.default_rng(0).normal(0.001, 0.01, 300)
    t = 200
    w = r[t - 62 : t + 1]
    want = w.mean() + w.std(ddof=1) * baselines.ES_NORMAL
    assert baselines.param_es_at(r, np.array([t]))[0] == pytest.approx(want)


def test_es_is_never_above_var_for_all_baselines(world):
    panel, idx, R = world
    f = evaluate.make_folds(panel, idx, test_years=(2021,))[0]
    test = panel.iloc[f.test]
    for m in baselines.default_baselines(R):
        m.fit(panel.iloc[f.train])
        p = m.predict(test)
        ok = p["es_hat"].notna() & p["var_hat"].notna()
        if ok.any():
            assert (p.loc[ok, "es_hat"] <= p.loc[ok, "var_hat"] + 1e-12).all(), m.name


def test_fz0_hand_value_and_minimised_at_the_truth():
    # y=-3 <= v=-2: -(v-y)/(a e) + v/e + log(-e) - 1 with a=.05, e=-4
    y, v, e = np.array([-3.0]), np.array([-2.0]), np.array([-4.0])
    want = -(1.0) / (0.05 * -4.0) + (-2.0 / -4.0) + np.log(4.0) - 1
    assert evaluate.fz0_loss(y, v, e) == pytest.approx(want)
    rng = np.random.default_rng(1)
    y = rng.standard_normal(200_000)
    v0, e0 = -1.6449, -2.0627
    best = evaluate.fz0_loss(y, np.full_like(y, v0), np.full_like(y, e0))
    for k in (0.7, 1.4):
        worse = evaluate.fz0_loss(y, np.full_like(y, v0 * k), np.full_like(y, e0 * k))
        assert worse > best


def test_es_residual_is_zero_for_a_correct_model_and_negative_for_a_shallow_one():
    y = np.random.default_rng(2).standard_normal(400_000)
    v = np.full_like(y, -1.6449)
    assert evaluate.es_residual(y, v, np.full_like(y, -2.0627)) == pytest.approx(0.0, abs=0.01)
    assert evaluate.es_residual(y, v, np.full_like(y, -1.7)) < -0.1
    assert evaluate.es_ratio(y, v, np.full_like(y, -2.0627)) == pytest.approx(1.0, abs=0.01)


def test_compute_metrics_scores_es_with_the_same_models_var(world):
    panel, idx, R = world
    folds = evaluate.make_folds(panel, idx, test_years=(2021,))
    preds = evaluate.run_walk_forward(baselines.default_baselines(R), panel, folds)
    es = preds[preds["target"] == "es"]
    assert {"fhs", "hist_var", "garch"} <= set(es["model"])
    var = preds[preds["target"] == "var"].set_index(["model", "date", "pid"])["y_pred"]
    chk = es.set_index(["model", "date", "pid"])
    np.testing.assert_allclose(chk["y_aux"], var.reindex(chk.index))
    m = evaluate.compute_metrics(preds)
    assert {"fz0", "es_resid", "es_ratio"} <= set(m[m["target"] == "es"]["metric"])


# ---- FHS --------------------------------------------------------------------------------
def test_fhs_params_ignore_returns_from_the_first_test_day_on(world):
    panel, idx, R = world
    start = 1200
    R2 = R.copy()
    R2[:, start:] = np.random.default_rng(3).normal(0, 0.05, R[:, start:].shape)
    a = baselines.GarchFitCache(R).get(0, start, start + 20)
    b = baselines.GarchFitCache(R2).get(0, start, start + 20)
    np.testing.assert_allclose(a["z"], b["z"])
    assert a["mu"] == b["mu"] and len(a["z"]) == start


def test_fhs_matches_a_manual_arch_computation(world):
    from arch import arch_model

    test, R = _test_rows(world)
    pid = int(test.index.get_level_values("pid")[0])
    sub = test[test.index.get_level_values("pid") == pid]
    start, day = int(test["day_idx"].min()), int(sub["day_idx"].iloc[3])
    res = arch_model(R[pid] * 100, mean="Constant", vol="GARCH", p=1, q=1, rescale=False).fit(
        last_obs=start, disp="off", show_warning=False)
    z = np.asarray(res.std_resid)
    z = z[np.isfinite(z)][:start]
    f = res.forecast(horizon=1, start=start, reindex=False)
    sigma = np.sqrt(f.variance.iloc[day - start, 0]) / 100
    want = res.params["mu"] / 100 + sigma * np.quantile(z, ALPHA)
    got = baselines.FHSBaseline(R).predict(sub)["var_hat"].iloc[3]
    assert got == pytest.approx(want, rel=1e-6)


def test_fhs_is_causal_and_aligned(world):
    test, R = _test_rows(world)
    t0 = int(np.median(test["day_idx"]))
    R2 = R.copy()
    R2[:, t0 + 1 :] = np.random.default_rng(1).normal(0, 0.05, R[:, t0 + 1 :].shape)
    a = baselines.FHSBaseline(R).predict(test)
    b = baselines.FHSBaseline(R2).predict(test)
    early = (test["day_idx"] <= t0).to_numpy()
    pd.testing.assert_frame_equal(a[early], b[early])
    assert a.index.equals(test.index) and (a["var_hat"] < 0).all() and a["vol_hat"].isna().all()


def test_fhs_fallback_is_finite_and_counted(world, monkeypatch):
    test, R = _test_rows(world)
    cache = baselines.GarchFitCache(R)
    monkeypatch.setattr(cache, "_fit", lambda *a: None)
    m = baselines.FHSBaseline(R, cache)
    out = m.predict(test)
    assert m.n_fallback == test.index.get_level_values("pid").nunique()
    assert np.isfinite(out[["var_hat", "es_hat"]]).all().all()
    assert (out["es_hat"] <= out["var_hat"]).all()


def test_garch_and_fhs_share_one_fit_per_portfolio(world, monkeypatch):
    test, R = _test_rows(world)
    cache = baselines.GarchFitCache(R)
    calls = []
    real = cache._fit
    monkeypatch.setattr(cache, "_fit", lambda *a: calls.append(a) or real(*a))
    baselines.GarchBaseline(R, cache).predict(test)
    baselines.FHSBaseline(R, cache).predict(test)
    assert len(calls) == test.index.get_level_values("pid").nunique()


# ---- conformal --------------------------------------------------------------------------
def test_tracker_recursion_hand_computed():
    errs = [0.2, 0.0, 0.1]
    used, err, after = conformal.tracker_states(
        np.array([0, 5, 10]), lambda i, c: errs[i], eta=0.5, lag=1, alpha=0.05)
    np.testing.assert_allclose(used, [0.0, 0.075, 0.05])
    np.testing.assert_allclose(after, [0.075, 0.05, 0.075])


def _toy(n_days=40, n_pid=30, seed=0, shallow=0.8):
    rng = np.random.default_rng(seed)
    day = np.repeat(np.arange(n_days) * 5, n_pid)
    y = rng.standard_normal(len(day)) * 0.01
    q = np.full(len(day), -1.645 * 0.01 * shallow)  # too shallow VaR -> too many exceedances
    scale = np.full(len(day), 0.01)
    calib = np.ones(len(day), bool)
    return day, y, q, scale, calib


def test_calibration_reduces_the_miscoverage_of_a_shallow_var():
    day, y, q, scale, calib = _toy(n_days=120)
    q_cal, tab = conformal.calibrate_var(day, y, q, scale, calib)
    assert (y < q).mean() > (y < q_cal).mean() > ALPHA - 0.02
    assert tab["offset"].iloc[0] == 0.0 and tab["offset"].iloc[-1] > 0


@pytest.mark.parametrize("lag", [1, 21])
def test_offsets_ignore_labels_not_yet_available(lag):
    day, y, q, scale, calib = _toy()
    base = conformal.calibrate_var(day, y, q, scale, calib, lag=lag)[1]
    d0 = int(np.unique(day)[10])
    y2 = y.copy()
    y2[day == d0] = -0.5  # a huge loss on one date
    new = conformal.calibrate_var(day, y2, q, scale, calib, lag=lag)[1]
    same = base["day_idx"] < d0 + lag  # not yet knowable at those dates
    pd.testing.assert_series_equal(base.loc[same, "offset"], new.loc[same, "offset"])
    assert not np.allclose(base.loc[~same, "offset"], new.loc[~same, "offset"])


def test_heldout_labels_do_not_move_offsets():
    day, y, q, scale, calib = _toy()
    held = np.arange(len(day)) % 3 == 0
    a = conformal.calibrate_var(day, y, q, scale, ~held)[1]
    y2 = y.copy()
    y2[held] = -0.5
    b = conformal.calibrate_var(day, y2, q, scale, ~held)[1]
    pd.testing.assert_frame_equal(a, b)


def test_offset_at_replays_the_stream_and_the_drift_identity_holds():
    day, y, q, scale, calib = _toy(seed=4)
    for lag in (1, 21):
        _, tab = conformal.calibrate_var(day, y, q, scale, calib, lag=lag)
        for d, off in zip(tab["day_idx"], tab["offset"], strict=True):
            assert conformal.offset_at(tab, int(d), lag) == pytest.approx(off)
        s = conformal.drift_summary(tab)
        assert s["mean_err"] - ALPHA == pytest.approx(s["drift_bound"])


def test_add_calibrated_keeps_rows_identical_and_covers_every_var_model(world):
    panel, idx, R = world
    folds = evaluate.make_folds(panel, idx, test_years=(2020, 2021))
    preds = evaluate.run_walk_forward(baselines.default_baselines(R), panel, folds,
                                      holdout=np.array([3]))
    out, tabs = evaluate.add_calibrated(preds, panel)
    var = out[out["target"] == "var"]
    raw = sorted(m for m in var["model"].unique() if not m.endswith("_cal"))
    assert sorted(var["model"].unique()) == sorted(raw + [m + "_cal" for m in raw])
    assert set(tabs["model"]) == set(raw)
    for m in raw:
        a = var[var["model"] == m].reset_index(drop=True)
        b = var[var["model"] == m + "_cal"].reset_index(drop=True)
        pd.testing.assert_frame_equal(a[["fold", "date", "pid", "y_true", "heldout"]],
                                      b[["fold", "date", "pid", "y_true", "heldout"]])
    assert out["model"].isin(preds["model"]).sum() >= len(preds)


# ---- significance -----------------------------------------------------------------------
def test_newey_west_lrv_hand_computed():
    x = np.array([1.0, -1.0, 1.0, -1.0])
    # mean 0, gamma0 = 1, gamma1 = -3/4; lags=1 Bartlett weight 1/2 -> 1 + 2*0.5*(-0.75)
    assert significance.newey_west_lrv(x, 1) == pytest.approx(0.25)
    assert significance.newey_west_lrv(x, 0) == pytest.approx(1.0)


def test_hln_factor_and_p_value():
    x = np.random.default_rng(0).normal(0.5, 1.0, 100)
    r1, r5 = significance.hac_mean_test(x, 1), significance.hac_mean_test(x, 5)
    lrv = significance.newey_west_lrv(x, significance.default_lags(100, 1))
    raw = x.mean() / np.sqrt(lrv / 100)
    assert r1["stat"] == pytest.approx(raw * np.sqrt((100 + 1 - 2) / 100))
    assert 0 <= r1["p"] < 1e-3 and r5["T"] == 100


def test_dm_null_size_with_ma4_errors():
    rng = np.random.default_rng(7)
    rej, n = 0, 400
    for _ in range(n):
        e = rng.standard_normal(340 + 4)
        d = np.convolve(e, np.ones(5) / 5, mode="valid")  # MA(4) loss differential, mean 0
        rej += significance.dm_test(d, np.zeros_like(d), h=5)["p"] < 0.05
    assert 0.02 <= rej / n <= 0.09


def test_dm_detects_a_shift():
    rng = np.random.default_rng(8)
    d = rng.standard_normal(300) * 0.5
    r = significance.dm_test(d - 0.3, d, h=1)
    assert r["mean"] == pytest.approx(-0.3) and r["p"] < 0.01


def _preds(rows):
    return pd.DataFrame(rows)


def test_date_losses_average_over_portfolios():
    dates = pd.to_datetime(["2021-01-04"] * 2 + ["2021-01-11"] * 2)
    p = _preds({"target": "vol", "model": "a", "date": dates, "pid": [0, 1, 0, 1],
                "y_true": [1.0, 1.0, 1.0, 1.0], "y_pred": [0.0, 2.0, 1.0, 3.0]})
    out = significance.date_losses(p, "vol")
    np.testing.assert_allclose(out["a"], [1.0, 2.0])  # (1+1)/2 and (0+4)/2


def test_mcs_keeps_equal_models_and_drops_a_worse_one_reproducibly():
    rng = np.random.default_rng(11)
    T = 300
    base = rng.standard_normal(T) * 0.3
    L = pd.DataFrame({"a": 1 + base + rng.standard_normal(T) * 0.05,
                      "b": 1 + base + rng.standard_normal(T) * 0.05,
                      "bad": 1.6 + base + rng.standard_normal(T) * 0.05})
    r = significance.mcs(L)
    assert r.set_index("model").loc["bad", "in_set"] == False  # noqa: E712
    assert r.set_index("model").loc[["a", "b"], "in_set"].all()
    pd.testing.assert_frame_equal(r, significance.mcs(L))


def test_clustered_coverage_flags_a_consistently_shallow_var():
    rng = np.random.default_rng(12)
    dates = np.repeat(pd.bdate_range("2021-01-01", periods=250), 20)
    y = rng.standard_normal(len(dates))
    rows = []
    for name, q in (("good", -1.645), ("shallow", -1.0)):
        rows.append(pd.DataFrame({"target": "var", "model": name, "date": dates, "pid": 0,
                                  "y_true": y, "y_pred": q}))
    out = significance.clustered_coverage(pd.concat(rows))
    allp = out[out["period"] == "all"].set_index("model")
    assert allp.loc["good", "p"] > 0.05 and allp.loc["shallow", "p"] < 1e-4
    assert allp.loc["shallow", "exceed_rate"] == pytest.approx(0.159, abs=0.02)


def test_run_significance_runs_end_to_end_on_baselines(world):
    panel, idx, R = world
    folds = evaluate.make_folds(panel, idx, test_years=(2020, 2021))
    preds = evaluate.run_walk_forward(baselines.default_baselines(R), panel, folds)
    out = significance.run_significance(preds)
    assert {"vol", "dd", "var", "es"} <= set(out["mcs"]["target"])
    assert len(out["coverage"]) > 0
