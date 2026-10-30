import numpy as np
import pandas as pd
import pytest
import torch

from src.models import evaluate, lstm, rf, xgb
from src.models.base import feature_cols


@pytest.fixture(scope="module")
def split(world):
    panel, idx, R = world
    f = evaluate.make_folds(panel, idx, test_years=(2020,))[0]
    return panel, idx, R, f


def seq_from(R, panel, idx):
    """Rebuild a seq.npy-style array (P, T, 4): port_ret, mkt_ret, VIX, VIX_chg."""
    rng = np.random.default_rng(0)
    mkt = rng.normal(0, 0.01, R.shape[1])
    vix = np.full(R.shape[1], 15.0) + rng.normal(0, 0.1, R.shape[1]).cumsum()
    dv = np.diff(np.log(np.abs(vix) + 1), prepend=0.0)
    rows = [np.stack([R[p], mkt, vix, dv], axis=1) for p in range(len(R))]
    return np.stack(rows).astype("float32")


def make(kind, seq):
    if kind == "rf":
        return rf.RFModel(n_estimators=20)
    if kind == "xgb":
        return xgb.XGBModel(n_estimators=40)
    return lstm.LSTMModel(seq=seq, max_epochs=2)


def test_features_exclude_targets_baselines_and_meta(world):
    panel, _, _ = world
    cols = set(feature_cols(panel))
    for banned in ("vol_fwd", "y_logratio", "dd_flag", "r_next", "lw_vol", "naive_vol",
                   "hist_var250", "param_var63", "day_idx"):
        assert banned not in cols
    assert "ewma_vol" in cols  # causal, allowed as a feature (proposal P3)
    from src.models.base import MODEL_FEATURES, NONFEATURE_COLS

    # Round 4 (B2): the three HAR/GARCH forecasts are inputs; the simulated drawdown
    # probability and the fallback flag are carried in the panel but never fed to a model.
    wide = panel.assign(**{c: 1.0 for c in MODEL_FEATURES + NONFEATURE_COLS})
    got = set(feature_cols(wide))
    assert set(MODEL_FEATURES) <= got and not set(NONFEATURE_COLS) & got


@pytest.mark.parametrize("kind", ["rf", "xgb", "lstm"])
def test_fit_predict_shapes_and_ranges(split, kind):
    panel, idx, R, f = split
    m = make(kind, seq_from(R, panel, idx))
    m.fit(panel.iloc[f.fit], panel.iloc[f.val])
    pred = m.predict(panel.iloc[f.test])
    assert pred.index.equals(panel.iloc[f.test].index)
    assert pred["vol_hat"].notna().all() and (pred["vol_hat"] > 0).all()
    assert pred["dd_hat"].between(0, 1).all()
    if kind == "rf":
        assert pred["var_hat"].isna().all()  # no quantile loss in sklearn RF
    else:
        assert pred["var_hat"].notna().all()


@pytest.mark.parametrize("kind", ["rf", "xgb"])
def test_deterministic_with_seed(split, kind):
    panel, idx, R, f = split
    a, b = make(kind, None), make(kind, None)
    for m in (a, b):
        m.fit(panel.iloc[f.fit], panel.iloc[f.val])
    test = panel.iloc[f.test]
    pd.testing.assert_frame_equal(a.predict(test), b.predict(test))


@pytest.mark.parametrize("kind", ["rf", "xgb"])
def test_predictions_ignore_test_labels(split, kind):
    panel, idx, R, f = split
    m = make(kind, None)
    m.fit(panel.iloc[f.fit], panel.iloc[f.val])
    test = panel.iloc[f.test]
    corrupted = test.copy()
    for c in ("vol_fwd", "y_logratio", "dd_flag", "r_next"):
        corrupted[c] = 123.0
    pd.testing.assert_frame_equal(m.predict(test), m.predict(corrupted))


def test_lstm_window_ends_at_t():
    day = torch.tensor([30, 100])
    w = lstm.window_indices(day)
    assert w.shape == (2, lstm.WINDOW)
    assert (w[:, -1] == day).all() and (w[:, 0] == day - (lstm.WINDOW - 1)).all()


def test_lstm_scaler_uses_train_period_only(split):
    panel, idx, R, f = split
    seq = seq_from(R, panel, idx)
    train, val = panel.iloc[f.fit], panel.iloc[f.val]
    a = make("lstm", seq)
    a.fit(train, val)
    seq2 = seq.copy()
    last = int(train["day_idx"].max())
    seq2[:, last + 1 :] = 5.0  # change everything after the training period
    b = make("lstm", seq2)
    b.fit(train, val)
    assert torch.allclose(a.mu, b.mu) and torch.allclose(a.sd, b.sd)


def test_lstm_prediction_depends_only_on_window(split):
    panel, idx, R, f = split
    seq = seq_from(R, panel, idx)
    m = make("lstm", seq)
    m.fit(panel.iloc[f.fit], panel.iloc[f.val])
    test = panel.iloc[f.test].iloc[:50]
    base = m.predict(test)
    t = int(test["day_idx"].max())
    m.X = m.X.clone()
    m.X[:, t + 1 :] = 9.0  # future of every test row's window end
    pd.testing.assert_frame_equal(base, m.predict(test))


def test_xgb_refit_uses_chosen_rounds_on_all_train_rows(split):
    panel, idx, R, f = split
    m = make("xgb", None)
    m.fit(panel.iloc[f.fit], panel.iloc[f.val])
    chosen = dict(m.n_best)
    m.refit(panel.iloc[f.train])
    assert m.n_rounds == chosen
    assert m.reg.get_booster().num_boosted_rounds() == chosen["reg"]
    assert m.predict(panel.iloc[f.test])["vol_hat"].gt(0).all()


def test_xgb_refit_round_override_applies_to_named_head_only(split):
    panel, idx, R, f = split
    m = make("xgb", None)
    m.fit(panel.iloc[f.fit], panel.iloc[f.val])
    chosen = dict(m.n_best)
    m.refit(panel.iloc[f.train], rounds={"qr": 7})
    assert m.n_rounds == {**chosen, "qr": 7}
    assert m.n_best == chosen  # the validation choice is kept for the record
    assert m.qr.get_booster().num_boosted_rounds() == 7
    assert m.reg.get_booster().num_boosted_rounds() == chosen["reg"]


def test_fold_median_rounds_reads_saved_fold_models(tmp_path):
    from types import SimpleNamespace

    import joblib

    from src.models.train_final import fold_median_rounds

    for year, qr in [("2020", 182), ("2021", 24), ("2022", 155), ("2023", 595)]:
        (tmp_path / year).mkdir()
        joblib.dump(SimpleNamespace(n_rounds={"qr": qr}), tmp_path / year / "xgboost.joblib")
    assert fold_median_rounds(tmp_path, "qr") == 168  # median of 24, 155, 182, 595
    with pytest.raises(FileNotFoundError):
        fold_median_rounds(tmp_path / "missing", "qr")


def test_lstm_refit_runs_best_epochs_and_refits_scalers(split):
    panel, idx, R, f = split
    seq = seq_from(R, panel, idx)
    m = make("lstm", seq)
    m.fit(panel.iloc[f.fit], panel.iloc[f.val])
    assert len(m.best_epochs) == 3
    assert all(1 <= e <= m.max_epochs for e in m.best_epochs)
    mu_before = m.mu.clone()
    m.refit(panel.iloc[f.train])
    assert not torch.allclose(m.mu, mu_before)  # recomputed on the larger train set
    pred = m.predict(panel.iloc[f.test])
    assert pred["vol_hat"].notna().all() and pred["dd_hat"].between(0, 1).all()


def test_fold_models_round_trip(split, tmp_path):
    from src.models import baselines

    panel, idx, R, f = split
    seq = seq_from(R, panel, idx)
    models = [make("rf", None), make("xgb", None), make("lstm", seq), baselines.HAR(),
              baselines.Climatology()]
    for m in models:
        evaluate.fit_model(m, panel.iloc[f.fit], panel.iloc[f.val], panel.iloc[f.train])
    evaluate.save_fold_models(models, tmp_path)
    import joblib

    test = panel.iloc[f.test]
    pd.testing.assert_frame_equal(joblib.load(tmp_path / "xgboost.joblib").predict(test),
                                  models[1].predict(test))
    loaded = lstm.LSTMModel.load(tmp_path / "lstm.pt")
    row = test.iloc[0]
    win = seq[int(test.index.get_level_values("pid")[0]),
              int(row["day_idx"]) - lstm.WINDOW + 1 : int(row["day_idx"]) + 1]
    out = loaded.predict_window(win, {"naive_vol": float(row["naive_vol"]),
                                      "ewma_vol": float(row["ewma_vol"])})
    assert out["vol_hat"] > 0
    import json

    assert set(json.loads((tmp_path / "baselines.json").read_text())) == {"har_coef",
                                                                          "climatology_rate"}


def test_xgb_can_be_refit_across_consecutive_folds(split):
    """The walk-forward loop reuses model objects: fit -> refit -> fit -> refit must work."""
    panel, idx, R, f = split
    m = make("xgb", None)
    for _ in range(2):
        m.fit(panel.iloc[f.fit], panel.iloc[f.val])
        assert m.reg.get_params()["n_estimators"] == m._n_estimators  # refit's cut undone
        m.refit(panel.iloc[f.train])


def test_lstm_raises_clearly_when_validation_loss_is_nan(split):
    panel, idx, R, f = split
    m = make("lstm", seq_from(R, panel, idx))
    m._epoch_losses = lambda va: np.full(3, np.nan)
    with pytest.raises(RuntimeError, match="never improved"):
        m.fit(panel.iloc[f.fit], panel.iloc[f.val])


def test_var_scale_is_shared_and_floored():
    from src.models.base import var_scale

    out = var_scale([0.0, 0.252 * 4])
    np.testing.assert_allclose(out, [1e-3 / np.sqrt(252), 1.008 / np.sqrt(252)])


def test_xgb_n_best_is_argmin_of_full_validation_curve(split):
    panel, idx, R, f = split
    m = make("xgb", None)
    m.fit(panel.iloc[f.fit], panel.iloc[f.val])
    for head, curve in m.val_curves.items():
        assert len(curve) == m._n_estimators  # no early stopping: the whole curve is recorded
        assert m.n_best[head] == int(np.argmin(curve)) + 1
        assert m.at_cap[head] == (m.n_best[head] >= m._n_estimators)


def test_lstm_per_head_epochs_and_snapshots(split):
    """Each head uses the weights of its own best epoch; refit reproduces those epochs."""
    panel, idx, R, f = split
    m = make("lstm", seq_from(R, panel, idx))
    m.max_epochs = 4
    m.fit(panel.iloc[f.fit], panel.iloc[f.val])
    assert len(m.states) == 3 and all(1 <= e <= 4 for e in m.best_epochs)
    m.refit(panel.iloc[f.train])
    # heads sharing a best epoch share one snapshot object; different epochs differ
    for a in range(3):
        for b in range(a + 1, 3):
            assert (m.states[a] is m.states[b]) == (m.best_epochs[a] == m.best_epochs[b])
    pred = m.predict(panel.iloc[f.test])
    assert pred[["vol_hat", "dd_hat", "var_hat"]].notna().all().all()


def test_lstm_loads_old_single_state_checkpoint(split, tmp_path):
    panel, idx, R, f = split
    m = make("lstm", seq_from(R, panel, idx))
    m.fit(panel.iloc[f.fit], panel.iloc[f.val])
    m.save(tmp_path / "new.pt")
    ck = torch.load(tmp_path / "new.pt", map_location="cpu", weights_only=False)
    ck["state"] = ck.pop("states")[0]
    ck.pop("best_epochs")
    ck.pop("static_cols")  # pre-Round-4 files had no such key: two static inputs
    torch.save(ck, tmp_path / "old.pt")
    old = lstm.LSTMModel.load(tmp_path / "old.pt")
    assert len(old.states) == 3 and old.states[0] is old.states[2]
    a, b = m.predict_window, old.predict_window
    w = np.random.default_rng(0).normal(0, 0.01, (lstm.WINDOW, 4))
    st = {"naive_vol": 0.2, "ewma_vol": 0.2}
    assert old.static_cols == lstm.STATIC_R3
    assert a(w, st)["vol_hat"] > 0 and b(w, st)["vol_hat"] > 0
