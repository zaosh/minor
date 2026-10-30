import numpy as np
import pandas as pd
import pytest

from src import config
from src.models import baselines, evaluate

H = config.H


# ---- folds ------------------------------------------------------------------------------
def test_folds_are_time_ordered_and_purged(world):
    panel, idx, _ = world
    folds = evaluate.make_folds(panel, idx, test_years=(2019, 2020, 2021))
    day = panel["day_idx"].to_numpy()
    for f in folds:
        ts = day[f.test].min()
        assert (day[f.train] + H < idx.searchsorted(pd.Timestamp(f"{f.name}-01-01"))).all()
        assert day[f.train].max() < ts
        assert day[f.fit].max() + H < day[f.val].min()  # purge between fit and inner val
        assert len(set(f.train) & set(f.test)) == 0
        assert set(f.fit) | set(f.val) <= set(f.train)
    assert [len(f.train) for f in folds] == sorted(len(f.train) for f in folds)  # expanding


def test_last_fold_runs_to_end(world):
    panel, idx, _ = world
    folds = evaluate.make_folds(panel, idx, test_years=(2020, 2021))
    assert folds[-1].test.max() == len(panel) - 1


# ---- metrics ----------------------------------------------------------------------------
def test_basic_metrics_hand_computed():
    y, p = np.array([1.0, 2.0, 3.0]), np.array([1.0, 3.0, 5.0])
    assert evaluate.rmse(y, p) == pytest.approx(np.sqrt(5 / 3))
    assert evaluate.mae(y, p) == pytest.approx(1.0)


def test_directional_accuracy():
    now = np.array([1.0, 1.0, 1.0, 1.0])
    y = np.array([2.0, 0.5, 2.0, 0.5])
    p = np.array([1.5, 0.8, 0.9, 1.2])  # right, right, wrong, wrong
    assert evaluate.directional_accuracy(y, p, now) == 0.5
    assert np.isnan(evaluate.directional_accuracy(y, now, now))  # never predicts a move


def test_pinball_and_exceedance():
    y, q = np.array([-0.03, 0.01]), np.array([-0.02, -0.02])
    # row0: d=-0.01 -> (0.05-1)*-0.01 = .0095 ; row1: d=.03 -> .05*.03=.0015
    assert evaluate.pinball(y, q) == pytest.approx((0.0095 + 0.0015) / 2)
    assert evaluate.exceedance_rate(y, q) == 0.5


def test_kupiec_known_values():
    lr, p = evaluate.kupiec_pof(5, 100, 0.05)  # exactly at the nominal rate
    assert lr == pytest.approx(0.0, abs=1e-9) and p == pytest.approx(1.0)
    lr, p = evaluate.kupiec_pof(10, 100, 0.05)
    expected = -2 * (90 * np.log(0.95) + 10 * np.log(0.05)
                     - 90 * np.log(0.9) - 10 * np.log(0.1))
    assert lr == pytest.approx(expected) and lr == pytest.approx(4.1308, abs=1e-3)
    assert p == pytest.approx(0.0421, abs=1e-3)
    assert evaluate.kupiec_pof(0, 100, 0.05)[1] < 0.05  # zero exceedances is rejected too


def test_classification_scores():
    s = evaluate.score("dd", np.array([0, 1, 0, 1.0]), np.array([0.1, 0.9, 0.2, 0.8]), None)
    assert s["auc"] == 1.0
    assert s["brier"] == pytest.approx(np.mean([0.01, 0.01, 0.04, 0.04]))


# ---- baselines --------------------------------------------------------------------------
def test_dd_lookup_monotone_and_bounded():
    s = np.linspace(0.05, 1.5, 40)
    p = baselines.dd_prob_from_vol(s)
    assert (np.diff(p) >= 0).all() and p.min() >= 0 and p.max() <= 1


def test_test_labels_never_affect_predictions(world):
    """Corrupting every label in the test rows must not change any baseline's predictions."""
    panel, idx, R = world
    f = evaluate.make_folds(panel, idx, test_years=(2020,))[0]
    test = panel.iloc[f.test]
    corrupted = test.copy()
    for c in ("vol_fwd", "y_logratio", "dd_flag", "r_next"):
        corrupted[c] = 123.0
    for m in baselines.default_baselines(R):
        m.fit(panel.iloc[f.train])
        pd.testing.assert_frame_equal(m.predict(test), m.predict(corrupted), obj=m.name)


def test_har_fit_ignores_labels_outside_its_training_rows(world):
    """Complements the test above: the fit itself only reads the rows it is given."""
    panel, idx, _ = world
    f = evaluate.make_folds(panel, idx, test_years=(2020,))[0]
    a, b = baselines.HAR(), baselines.HAR()
    a.fit(panel.iloc[f.fit])
    modified = panel.copy()
    modified.iloc[f.test, modified.columns.get_loc("vol_fwd")] = 9.0
    b.fit(modified.iloc[f.fit])
    np.testing.assert_allclose(a.coef_, b.coef_)
    # and it is sensitive to its own rows, so the check above can fail
    modified.iloc[f.fit, modified.columns.get_loc("vol_fwd")] *= 2.0
    c = baselines.HAR()
    c.fit(modified.iloc[f.fit])
    assert not np.allclose(a.coef_, c.coef_)


class _PartialNaN:
    name = "partial"

    def fit(self, train, val=None):
        pass

    def predict(self, X):
        out = baselines.VolColumnBaseline("x", "naive_vol").predict(X)
        out.iloc[0, out.columns.get_loc("vol_hat")] = np.nan
        return out


def test_run_walk_forward_rejects_partial_nan_predictions(world):
    panel, idx, _ = world
    folds = evaluate.make_folds(panel, idx, test_years=(2020,))
    with pytest.raises(AssertionError, match="NaN predictions"):
        evaluate.run_walk_forward([_PartialNaN()], panel, folds)


def test_garch_is_causal_and_aligned(world):
    panel, idx, R = world
    f = evaluate.make_folds(panel, idx, test_years=(2021,))[0]
    test = panel.iloc[f.test]
    t0 = int(np.median(test["day_idx"]))
    R2 = R.copy()
    R2[:, t0 + 1 :] = np.random.default_rng(1).normal(0, 0.05, R[:, t0 + 1 :].shape)
    a = baselines.GarchBaseline(R).predict(test)
    b = baselines.GarchBaseline(R2).predict(test)
    early = (test["day_idx"] <= t0).to_numpy()
    pd.testing.assert_frame_equal(a[early], b[early])
    assert a.index.equals(test.index) and a["vol_hat"].notna().all()
    assert (a["var_hat"] < 0).all()


def test_all_baselines_share_the_test_index_and_run_end_to_end(world, tmp_path):
    panel, idx, R = world
    folds = evaluate.make_folds(panel, idx, test_years=(2020, 2021))
    models = baselines.default_baselines(R)
    preds = evaluate.run_walk_forward(models, panel, folds)
    metrics = evaluate.compute_metrics(preds)
    table = evaluate.summarize(metrics)
    assert {"naive", "har", "garch", "hist_var"} <= set(preds["model"])
    # every model scored on the same rows within a target
    rows = preds[preds["target"] == "vol"].groupby("model").size()
    assert rows.nunique() == 1
    assert "rmse" in table.columns and len(table) > 0


# ---- refit, held-out portfolios, fold artifacts ----------------------------------------------
class _Recorder:
    """Records which rows fit/refit saw."""

    name = "rec"

    def __init__(self, with_refit: bool):
        self.calls = []
        if with_refit:
            self.refit = lambda train: self.calls.append(("refit", len(train)))

    def fit(self, train, val=None):
        self.calls.append(("fit", len(train), None if val is None else len(val)))

    def predict(self, X):
        return baselines.VolColumnBaseline("x", "naive_vol").predict(X)


def test_fit_model_refits_on_train_only_when_supported(world):
    panel, idx, _ = world
    f = evaluate.make_folds(panel, idx, test_years=(2020,))[0]
    fit, val, train = (panel.iloc[r] for r in (f.fit, f.val, f.train))
    a, b = _Recorder(True), _Recorder(False)
    evaluate.fit_model(a, fit, val, train)
    evaluate.fit_model(b, fit, val, train)
    assert a.calls == [("fit", len(fit), len(val)), ("refit", len(train))]
    assert b.calls == [("fit", len(train), None)]


def test_holdout_portfolios_never_reach_training(world):
    panel, idx, _ = world
    folds = evaluate.make_folds(panel, idx, test_years=(2020,))
    m = _Recorder(False)
    seen = []
    m.fit = lambda train, val=None: seen.append(set(train.index.get_level_values("pid")))
    preds = evaluate.run_walk_forward([m], panel, folds, holdout=np.array([1, 3]))
    assert seen[0] == {0, 2}
    assert set(preds.loc[preds["heldout"], "pid"]) == {1, 3}  # but they are still scored
    assert set(preds["pid"]) == {0, 1, 2, 3}


def test_holdout_pids_reproducible_and_sized():
    a, b = evaluate.holdout_pids(400), evaluate.holdout_pids(400)
    assert (a == b).all() and len(set(a)) == evaluate.N_HOLDOUT


def test_fold_info_dates(world):
    panel, idx, _ = world
    f = evaluate.make_folds(panel, idx, test_years=(2020,))[0]
    info = evaluate.fold_info(panel, idx, f)
    assert pd.Timestamp(info["train_end"]) < pd.Timestamp(info["test_start"])
    assert pd.Timestamp(info["label_end"]) < pd.Timestamp(info["test_start"])  # purge holds
    assert idx[info["first_test_day_idx"]] == pd.Timestamp(info["test_start"])


def test_summarize_reports_reject_counts_not_mean_p():
    rows = [{"fold": str(y), "model": "m", "target": "var", "metric": k, "value": v}
            for y, (p, e) in enumerate([(0.01, 0.08), (0.2, 0.05), (0.03, 0.02)])
            for k, v in (("kupiec_p", p), ("exceedance", e), ("pinball", 0.001))]
    t = evaluate.summarize(pd.DataFrame(rows)).loc[("var", "m")]
    assert t["kupiec_reject"] == "2/3" and t["exc_range"] == "2.0%-8.0%"
