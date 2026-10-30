"""Round 4 pre-registration: the constants and the Round 3 ML hyperparameters must not move."""

import numpy as np
import pytest

from src.models import prereg


def test_round3_rf_hyperparameters_unchanged():
    from src.models.rf import RFModel

    m = RFModel()
    for k, v in prereg.RF_PARAMS.items():
        assert getattr(m.reg, k) == v, k
        assert getattr(m.clf, k) == v, k


def test_round3_xgb_hyperparameters_unchanged():
    from src.models.xgb import XGBModel

    m = XGBModel()
    for head in (m.reg, m.clf, m.qr):
        p = head.get_params()
        for k, v in prereg.XGB_PARAMS.items():
            assert p[k] == v, k
        assert p["early_stopping_rounds"] is None


def test_round3_lstm_hyperparameters_unchanged():
    from src.models import lstm

    m = lstm.LSTMModel(seq=np.zeros((1, 1, 4), dtype=np.float32))
    p = prereg.LSTM_PARAMS
    assert (m.hidden, m.max_epochs, m.patience, m.batch_size) == (
        p["hidden"], p["max_epochs"], p["patience"], p["batch_size"])
    assert m.lr == pytest.approx(p["lr"]) and lstm.WINDOW == p["window"]


def test_preregistered_choices():
    assert prereg.COMBOS == {"combo_hg": ("har", "garch"),
                             "combo_hgx": ("har", "garch", "xgboost")}
    assert prereg.VOL_FLOOR == 1e-4 and prereg.DQ_LAGS == 4 and prereg.MCS_ALPHA == 0.10
    assert prereg.STRESS_FOLDS == ("2020", "2022") and prereg.VIX_START == "2010-01-05"
    assert prereg.EXPECTED_REGIME_COUNTS == {"vix": (128, 205), "calendar": (99, 234)}


def test_tier_b_preregistered_choices():
    assert prereg.GJR_SPEC == {"p": 1, "o": 1, "q": 1, "dist": "t"}
    assert prereg.MODEL_FEATURES == ("har_vol", "garch_vol21", "garch_sigma1")
    assert (prereg.HAR_FIRST_YEAR, prereg.LAST_REFRESH_YEAR) == (2011, 2025)
    assert prereg.HAR_NO_CHANGE_COEF == (0.0, 1.0, 0.0, 0.0)
    assert (prereg.SIM_PATHS, prereg.SIM_SEED, prereg.RELIABILITY_BINS) == (2000, 42, 10)


def test_lstm_static_inputs_are_the_registered_columns():
    from src.models import lstm

    assert lstm.STATIC_R3 == ("naive_vol", "ewma_vol")
    assert lstm.STATIC_R4 == ("naive_vol", "ewma_vol", "har_vol", "garch_vol21", "garch_sigma1")
