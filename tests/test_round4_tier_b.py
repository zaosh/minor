"""Round 4 Tier B: GJR baseline, HAR/GARCH model features, simulated drawdown probability."""

import numpy as np
import pandas as pd
import pytest
from scipy import integrate, stats

from src import config
from src.features import model_features, targets
from src.models import baselines, evaluate, prereg, simdd
from src.models.base import MODEL_FEATURES

ALPHA = config.VAR_ALPHA


@pytest.fixture(scope="module")
def feats(world):
    panel, idx, R = world
    f, diag = model_features.garch_panel_features(R, panel, idx, workers=1)
    return f, diag


@pytest.fixture(scope="module")
def panel_f(world, feats):
    panel, idx, _ = world
    return model_features.add_model_features(panel, idx, feats[0], None)


def _fold(world, year=2020, years=(2020, 2021)):
    panel, idx, R = world
    folds = evaluate.make_folds(panel, idx, test_years=years)
    return next(f for f in folds if f.name == str(year))


# ---- B1: GarchFitCache is unchanged for the default spec ---------------------------------------
GOLDEN = {  # captured on the synthetic world BEFORE the Round 4 refactor (start 1305, last 1775)
    0: (-4.383019609801976e-05, 0.13874620013653016, 0.008625879305926933, 0.1540602590186378,
        -8.329955235058748),
    1: (0.0003326721645284348, 0.1362471071712594, 0.008579552332504728, 0.12235573790918598,
        16.145569628911126),
    2: (0.00018185062667414492, 0.13858345416396847, 0.00864346197008622, 0.13910846888313852,
        -1.4250867440547044),
    3: (-0.00011238240491918544, 0.19095147831716258, 0.011918053004170514, 0.2204073002050432,
        -0.0949769826542628),
}


def test_garch_fit_cache_default_spec_is_unchanged_since_round3(world):
    panel, idx, R = world
    f = evaluate.make_folds(panel, idx, test_years=(2020,))[0]
    test = panel.iloc[f.test]
    start, last = int(test["day_idx"].min()), int(test["day_idx"].max())
    assert (start, last) == (1305, 1775)
    cache = baselines.GarchFitCache(R.astype("float64"))
    for pid, (mu, v0, s0, v_last, zsum) in GOLDEN.items():
        r = cache.get(pid, start, last)
        assert r["mu"] == pytest.approx(mu, rel=1e-8)
        assert r["vol21"][0] == pytest.approx(v0, rel=1e-8)
        assert r["sigma1"][0] == pytest.approx(s0, rel=1e-8)
        assert r["vol21"][-1] == pytest.approx(v_last, rel=1e-8)
        assert len(r["z"]) == start and r["z"].sum() == pytest.approx(zsum, rel=1e-8)
        assert set(r["params"]) == {"omega", "alpha", "beta"}


# ---- B1: GJR-GARCH-t ---------------------------------------------------------------------------
def test_std_t_var_es_matches_numerical_integration_and_the_normal_limit():
    for nu in (4.5, 8.0, 30.0):
        q, es = baselines.std_t_var_es(nu)
        c = np.sqrt((nu - 2) / nu)
        dens = lambda x, nu=nu, c=c: stats.t.pdf(x / c, nu) / c  # noqa: E731  unit-variance t
        assert integrate.quad(dens, -np.inf, q)[0] == pytest.approx(ALPHA, abs=1e-8)
        tail = integrate.quad(lambda x, dens=dens: x * dens(x), -np.inf, q)[0] / ALPHA
        assert es == pytest.approx(tail, abs=1e-8) and es < q < 0
    q, es = baselines.std_t_var_es(1e6)
    assert q == pytest.approx(-baselines.Z95, abs=1e-3)
    assert es == pytest.approx(baselines.ES_NORMAL, abs=1e-3)


def _test_rows(world, year=2021):
    panel, idx, R = world
    f = _fold(world, year, (year,))
    return panel.iloc[f.test], R


def test_gjr_matches_a_manual_arch_computation(world):
    from arch import arch_model

    test, R = _test_rows(world)
    pid = int(test.index.get_level_values("pid")[0])
    sub = test[test.index.get_level_values("pid") == pid]
    start, day = int(test["day_idx"].min()), int(sub["day_idx"].iloc[3])
    res = arch_model(R[pid] * 100, mean="Constant", vol="GARCH", p=1, o=1, q=1, dist="t",
                     rescale=False).fit(last_obs=start, disp="off", show_warning=False)
    f = res.forecast(horizon=config.H, start=start, reindex=False)
    v = f.variance.iloc[day - start].to_numpy()
    q, es = baselines.std_t_var_es(float(res.params["nu"]))
    got = baselines.GJRBaseline(R).predict(sub).iloc[3]
    assert got["vol_hat"] == pytest.approx(np.sqrt(v.mean() * 252) / 100, rel=1e-6)
    assert got["var_hat"] == pytest.approx(q * np.sqrt(v[0]) / 100, rel=1e-6)
    assert got["es_hat"] == pytest.approx(es * np.sqrt(v[0]) / 100, rel=1e-6)
    assert got["dd_hat"] == pytest.approx(baselines.dd_prob_from_vol([got["vol_hat"]])[0])


def test_gjr_is_causal_aligned_and_es_below_var(world):
    test, R = _test_rows(world)
    t0 = int(np.median(test["day_idx"]))
    R2 = R.copy()
    R2[:, t0 + 1 :] = np.random.default_rng(1).normal(0, 0.05, R[:, t0 + 1 :].shape)
    a = baselines.GJRBaseline(R).predict(test)
    b = baselines.GJRBaseline(R2).predict(test)
    early = (test["day_idx"] <= t0).to_numpy()
    pd.testing.assert_frame_equal(a[early], b[early])
    assert a.index.equals(test.index) and a.notna().all().all()
    assert (a["var_hat"] < 0).all() and (a["es_hat"] < a["var_hat"]).all()


def test_gjr_fallback_is_finite_and_counted(world, monkeypatch):
    test, R = _test_rows(world)
    cache = baselines.GarchFitCache(R, "gjr")
    monkeypatch.setattr(cache, "_fit", lambda *a: None)
    m = baselines.GJRBaseline(R, cache)
    out = m.predict(test)
    assert m.n_fallback == test.index.get_level_values("pid").nunique()
    assert np.isfinite(out).all().all() and (out["es_hat"] <= out["var_hat"]).all()


def test_default_baselines_include_gjr_and_run_through_the_runner(world):
    panel, idx, R = world
    folds = evaluate.make_folds(panel, idx, test_years=(2020,))
    models = baselines.default_baselines(R)
    assert "gjr" in [m.name for m in models]
    preds = evaluate.run_walk_forward([m for m in models if m.name in ("garch", "gjr")], panel,
                                      folds)
    assert {"vol", "dd", "var", "es"} <= set(preds[preds["model"] == "gjr"]["target"])


# ---- B2: model features -------------------------------------------------------------------------
def test_garch_features_ignore_returns_after_t(world, feats):
    panel, idx, R = world
    t0 = int(np.median(panel["day_idx"]))
    R2 = R.copy()
    R2[:, t0 + 1 :] = np.random.default_rng(2).normal(0, 0.05, R[:, t0 + 1 :].shape)
    f2, _ = model_features.garch_panel_features(R2, panel, idx, workers=1)
    early = (panel["day_idx"] <= t0).to_numpy()
    pd.testing.assert_frame_equal(feats[0][early], f2[early])  # incl. sim_dd_raw
    assert not np.allclose(feats[0][~early]["garch_vol21"], f2[~early]["garch_vol21"])


def test_har_feature_ignores_labels_that_reach_into_its_year_and_heldout_labels(world, feats):
    panel, idx, _ = world
    base, _ = model_features.har_feature(panel, idx, None)
    Y = 2019
    ts = int(np.searchsorted(idx.year.to_numpy(), Y))
    bad = panel.copy()
    late = (bad["day_idx"] + config.H >= ts).to_numpy()
    bad.loc[late, "vol_fwd"] = 7.0  # labels that reach into year Y (or later)
    got, _ = model_features.har_feature(bad, idx, None)
    yrs = idx[panel["day_idx"].to_numpy().astype(int)].year.to_numpy()
    np.testing.assert_array_equal(got[yrs <= Y], base[yrs <= Y])
    assert not np.allclose(got[yrs > Y], base[yrs > Y])
    held = np.array([1])
    h0, _ = model_features.har_feature(panel, idx, held)
    pid = panel.index.get_level_values("pid").to_numpy()
    bad2 = panel.copy()
    bad2.loc[pid == 1, "vol_fwd"] = 7.0
    h1, _ = model_features.har_feature(bad2, idx, held)
    np.testing.assert_array_equal(h0, h1)


def test_features_equal_the_baseline_predictions_on_test_rows(world, panel_f):
    panel, idx, R = world
    for year in (2020, 2021):
        f = _fold(world, year)
        test = panel_f.iloc[f.test]
        har = baselines.HAR()
        har.fit(panel.iloc[f.train])
        np.testing.assert_array_equal(test["har_vol"].to_numpy(),
                                      har.predict(test)["vol_hat"].to_numpy())
        g = baselines.GarchBaseline(R).predict(test)
        np.testing.assert_array_equal(test["garch_vol21"].to_numpy(), g["vol_hat"].to_numpy())
        np.testing.assert_allclose(test["garch_sigma1"].to_numpy(), -g["var_hat"] / baselines.Z95
                                   * baselines.ANN, rtol=1e-12)


def test_first_year_rows_have_no_change_har_feature(world, panel_f):
    panel, idx, _ = world
    coefs = model_features.har_coefficients(panel, idx)
    first = min(coefs)
    np.testing.assert_array_equal(coefs[first], np.array(prereg.HAR_NO_CHANGE_COEF))
    yrs = idx[panel["day_idx"].to_numpy().astype(int)].year.to_numpy()
    m = yrs == first
    np.testing.assert_allclose(panel_f.loc[m, "har_vol"], panel.loc[m, "vol_21"].clip(lower=1e-4),
                               rtol=1e-6)


def test_garch_features_are_positive_flagged_and_complete(panel_f, feats):
    assert panel_f[list(MODEL_FEATURES)].gt(0).all().all()
    assert panel_f[["garch_fallback", "sim_dd_raw"]].notna().all().all()
    assert panel_f["sim_dd_raw"].between(0, 1).all()
    assert set(feats[1]["converged"]) <= {True, False}


def test_garch_failure_falls_back_to_ewma_features(world, monkeypatch):
    panel, idx, R = world
    monkeypatch.setattr(baselines.GarchFitCache, "_fit", lambda *a: None)
    f, diag = model_features.garch_panel_features(R, panel, idx, workers=1)
    assert (f["garch_fallback"] == 1.0).all() and not diag["converged"].any()
    np.testing.assert_allclose(f["garch_vol21"], panel["ewma_vol"].astype(float))
    np.testing.assert_allclose(f["garch_sigma1"], panel["ewma_vol"].astype(float))
    assert f["sim_dd_raw"].between(0, 1).all()


def test_garch_features_work_with_worker_processes(world, feats):
    panel, idx, R = world
    f2, _ = model_features.garch_panel_features(R, panel, idx, workers=2)
    pd.testing.assert_frame_equal(feats[0], f2)


def test_missing_model_features_raise(world):
    panel, idx, _ = world
    with pytest.raises(ValueError, match="model_features"):
        model_features.add_model_features(panel, idx, pd.DataFrame(columns=["garch_vol21"]))


def test_test_labels_never_reach_the_ml_predictions_with_model_features(world, feats):
    from src.models import lstm, rf, xgb
    from tests.test_models import seq_from

    panel, idx, R = world
    f = _fold(world, 2020)  # tests year 2020 only: later years would use 2020 labels (by design)
    seq = seq_from(R, panel, idx)

    def run(frame):
        pf = model_features.add_model_features(frame, idx, feats[0], None)
        outs = []
        for m in (rf.RFModel(n_estimators=10), xgb.XGBModel(n_estimators=30),
                  lstm.LSTMModel(seq=seq, max_epochs=2, static_cols=lstm.STATIC_R4)):
            evaluate.fit_model(m, pf.iloc[f.fit], pf.iloc[f.val], pf.iloc[f.train])
            outs.append(m.predict(pf.iloc[f.test]))
            assert set(MODEL_FEATURES) <= set(getattr(m, "cols", MODEL_FEATURES))
        return outs

    a = run(panel)
    bad = panel.copy()
    for c in ("vol_fwd", "y_logratio", "dd_flag", "r_next"):
        bad.iloc[f.test, bad.columns.get_loc(c)] = np.random.default_rng(0).normal(
            size=len(f.test))
    for x, y in zip(a, run(bad), strict=True):
        pd.testing.assert_frame_equal(x, y)


def test_lstm_static_inputs_follow_the_configured_columns(world, panel_f):
    from src.models import lstm
    from tests.test_models import seq_from

    panel, idx, R = world
    f = _fold(world, 2020, (2020,))
    m = lstm.LSTMModel(seq=seq_from(R, panel, idx), max_epochs=2, static_cols=lstm.STATIC_R4)
    m.fit(panel_f.iloc[f.fit], panel_f.iloc[f.val])
    assert m.net.head[0].in_features == m.hidden + len(lstm.STATIC_R4) == m.hidden + 5
    assert m.predict(panel_f.iloc[f.test])[["vol_hat", "dd_hat", "var_hat"]].notna().all().all()
    with pytest.raises(KeyError):  # no silent fallback to the two Round 3 inputs
        lstm.LSTMModel(seq=seq_from(R, panel, idx), max_epochs=1,
                       static_cols=lstm.STATIC_R4).fit(panel.iloc[f.fit], panel.iloc[f.val])


# ---- B3: simulated drawdown probability ---------------------------------------------------------
def _path_mdd(returns):
    """Max drawdown (<= 0) of a wealth path that starts at 1, via the target's own function."""
    return float(targets.max_drawdown_rows(np.r_[0.0, returns], config.H)[0])


def test_simulation_matches_the_target_drawdown_path_by_path():
    U = simdd.uniform_matrix(2020, 50)
    z = np.array([-2.0, -0.5, 0.3, 1.1, 0.7])
    s, mu = 0.012, 0.0004
    p = simdd.simulate_dd_prob([s], mu, 0.0, 0.0, 1.0, z, U, thresh=0.05)[0]  # constant sigma
    draws = z[(U * len(z)).astype(int)]
    want = np.mean([_path_mdd(mu + s * d) <= -0.05 for d in draws])
    assert p == pytest.approx(want, abs=1e-12)
    assert 0.0 < p < 1.0  # the test is informative


def test_degenerate_residual_pool_gives_zero_or_one():
    U = simdd.uniform_matrix(2020, 20)
    down = simdd.simulate_dd_prob([0.01], 0.0, 0.0, 0.0, 1.0, np.array([-1.0]), U)[0]
    up = simdd.simulate_dd_prob([0.01], 0.0, 0.0, 0.0, 1.0, np.array([1.0]), U)[0]
    assert _path_mdd(np.full(config.H, -0.01)) <= -0.05 and down == 1.0 and up == 0.0


def test_probability_is_monotone_in_the_starting_sigma_with_common_random_numbers():
    z = np.random.default_rng(1).standard_normal(5000)
    U = simdd.uniform_matrix(2021)
    sig = np.linspace(0.004, 0.03, 12)
    p = simdd.simulate_dd_prob(sig, 0.0003, 2e-6, 0.08, 0.9, z, U)
    assert (np.diff(p) >= 0).all() and p[0] < p[-1]


def test_gaussian_constant_variance_matches_the_lookup_within_monte_carlo_error():
    z = np.random.default_rng(3).standard_normal(20000)
    U = simdd.uniform_matrix(2022)
    sigma_ann = np.array([0.15, 0.30, 0.50])
    p = simdd.simulate_dd_prob(sigma_ann / np.sqrt(252), 0.0, 0.0, 0.0, 1.0, z, U)
    np.testing.assert_allclose(p, baselines.dd_prob_from_vol(sigma_ann), atol=0.04)


def test_simulation_is_deterministic_and_uniforms_depend_on_year_only():
    a = simdd.uniform_matrix(2021)
    np.testing.assert_array_equal(a, simdd.uniform_matrix(2021))
    assert not np.array_equal(a, simdd.uniform_matrix(2022))
    assert a.shape == (prereg.SIM_PATHS, config.H)


def test_sim_dd_calibrator_sees_only_its_training_rows(world, panel_f):
    panel, idx, _ = world
    f = _fold(world, 2020, (2020,))
    m = simdd.SimDDModel("fhs_mc_logit", True)
    evaluate.fit_model(m, panel_f.iloc[f.fit], panel_f.iloc[f.val], panel_f.iloc[f.train])
    bad = panel_f.copy()
    outside = np.setdiff1d(np.arange(len(bad)), f.train)  # test rows and labels reaching them
    bad.iloc[outside, bad.columns.get_loc("dd_flag")] = 1.0 - bad.iloc[outside]["dd_flag"]
    m2 = simdd.SimDDModel("fhs_mc_logit", True)
    evaluate.fit_model(m2, bad.iloc[f.fit], bad.iloc[f.val], bad.iloc[f.train])
    assert (m.a, m.b) == (m2.a, m2.b)


def test_recalibration_keeps_the_ranking_when_the_slope_is_positive(world, panel_f):
    panel, idx, _ = world
    f = _fold(world, 2021, (2020, 2021))
    raw, cal = simdd.SimDDModel("fhs_mc", False), simdd.SimDDModel("fhs_mc_logit", True)
    for m in (raw, cal):
        evaluate.fit_model(m, panel_f.iloc[f.fit], panel_f.iloc[f.val], panel_f.iloc[f.train])
    test = panel_f.iloc[f.test]
    assert cal.b > 0
    a, b = raw.predict(test)["dd_hat"], cal.predict(test)["dd_hat"]
    assert stats.spearmanr(a, b)[0] == pytest.approx(1.0, abs=1e-12)
    assert raw.predict(test)[["vol_hat", "var_hat", "es_hat"]].isna().all().all()


def test_sim_models_run_through_the_runner_and_save_their_coefficients(world, panel_f, tmp_path):
    import json

    panel, idx, R = world
    folds = evaluate.make_folds(panel_f, idx, test_years=(2020,))
    models = [simdd.SimDDModel("fhs_mc", False), simdd.SimDDModel("fhs_mc_logit", True),
              baselines.Climatology()]
    preds = evaluate.run_walk_forward(models, panel_f, folds, save_dir=tmp_path,
                                      holdout=np.array([2]))
    assert set(preds[preds["model"] == "fhs_mc_logit"]["target"]) == {"dd"}
    saved = json.loads((tmp_path / "2020" / "baselines.json").read_text())
    assert saved["fhs_mc_logit"] == [models[1].a, models[1].b]
    assert json.loads((tmp_path / "2020" / "training.json").read_text())["fallbacks"] == {}


def test_seeds_refuse_to_overwrite_existing_results(tmp_path):
    from src.models import seeds

    (tmp_path / "seed_metrics.csv").write_text("x")
    with pytest.raises(FileExistsError, match="--out"):
        seeds.main([1], tmp_path)


# ---- tier B report ------------------------------------------------------------------------------
def _tier_b_inputs(tmp_path):
    from src.models import combos, round4
    from tests.test_round4_tier_a import _synthetic_world

    _, pp, fp, mp, vp, _, _ = _synthetic_world(tmp_path)
    old = pd.read_parquet(pp)
    raw = old[~old["model"].str.endswith("_cal")]
    rng = np.random.default_rng(4)

    def clone(src, name, targets):
        d = raw[(raw["model"] == src) & raw["target"].isin(targets)].copy()
        d["model"] = name
        d["y_pred"] = d["y_pred"] * rng.uniform(0.95, 1.05, len(d))
        return d

    extra = [clone("garch", "gjr", ["vol", "dd", "var", "es"]), clone("har", "fhs_mc", ["dd"]),
             clone("har", "fhs_mc_logit", ["dd"]), clone("har", "climatology", ["dd"])]
    full = combos.add_combos(pd.concat([raw, *extra], ignore_index=True))
    panel = pd.read_parquet(mp)
    full, _ = evaluate.add_calibrated(full, panel)
    new = tmp_path / "new.parquet"
    full.to_parquet(new)
    return round4, new, pp, fp, vp


def test_tier_b_end_to_end_counts_guards_and_outputs(tmp_path):
    round4, new, old, fp, vp = _tier_b_inputs(tmp_path)
    out = tmp_path / "tb"
    man = round4.tier_b(new, old, out, fp, vp, tmp_path / "nofolds", None)
    for f in ("dm_round4.csv", "mcs_round4.csv", "dd_reliability.csv", "ml_round3_vs_round4.csv",
              "tests_count.csv", "var_tests_datelevel.csv", "vol_summary.csv", "manifest.json",
              "comparison_all.csv", "regime_metrics.csv"):
        assert (out / f).exists(), f
    dm = pd.read_csv(out / "dm_round4.csv")
    n = dm.groupby("family").size()
    assert n["B1"] == 4 and n["B2"] == 8 and n["B3"] == 4
    assert n["A1"] == 12 and n["A3"] == 13 and n["A4"] == 32
    assert (dm["p_holm"] >= dm["p_raw"] - 1e-12).all()
    rel = pd.read_csv(out / "dd_reliability.csv")
    assert {"fhs_mc", "fhs_mc_logit", "garch"} <= set(rel["model"])
    assert rel.groupby("model")["n"].sum().nunique() == 1  # every model on the same rows
    cmp3 = pd.read_csv(out / "ml_round3_vs_round4.csv")
    assert (cmp3["round3"] == cmp3["round4"]).all()  # identical inputs for the ML rows here
    assert man["compare_sha256"] == round4.sha256(old)
    with pytest.raises(FileExistsError):
        round4.tier_b(new, old, out, fp, vp, tmp_path / "nofolds", None)
    with pytest.raises(ValueError, match="pre-registered"):  # design counts are checked
        round4.tier_b(new, old, tmp_path / "tb2", fp, vp, tmp_path / "nofolds")
    assert not (tmp_path / "tb2").exists()
    with pytest.raises(ValueError, match="lack"):  # predictions without the Round 4 models
        round4.tier_b(old, old, tmp_path / "tb3", fp, vp, tmp_path / "nofolds", None)


def test_round4_markdown_renders_tier_b_tables_from_files(tmp_path):
    from src.models import report

    round4, new, old, fp, vp = _tier_b_inputs(tmp_path)
    assert report.round4_markdown(tmp_path / "missing") == ""
    round4.tier_b(new, old, tmp_path / "tb", fp, vp, tmp_path / "nofolds", None)
    md = report.round4_markdown(tmp_path / "tb")
    for s in ("QLIKE", "Round 3 beside Round 4", "Holm", "Number of tests run",
              "Model confidence sets"):
        assert s in md, s
