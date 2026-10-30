import pandas as pd

from src import config
from src.models import report


def _metrics():
    rows = []
    for fold in ("2020", "2021"):
        for model, v in (("hist_var", 0.0015), ("hist_var_cal", 0.0016), ("xgboost_cal", 0.0014),
                         ("xgboost", 0.0016)):
            rows.append({"fold": fold, "model": model, "target": "var", "metric": "pinball",
                         "value": v})
        for model, v in (("naive", 0.08), ("rf", 0.07)):
            rows.append({"fold": fold, "model": model, "target": "vol", "metric": "rmse",
                         "value": v})
        for model, v in (("climatology", 0.22), ("rf", 0.21)):
            rows.append({"fold": fold, "model": model, "target": "dd", "metric": "brier",
                         "value": v})
    return pd.DataFrame(rows)


def test_calibrated_models_are_compared_with_the_calibrated_reference():
    t = report.beats_naive_table(_metrics()).set_index(["target", "model"])
    # raw xgboost (0.0016) does not beat raw hist_var (0.0015); calibrated xgboost (0.0014)
    # beats calibrated hist_var (0.0016), not the raw one it would also beat
    assert t.loc[("var", "xgboost"), "folds_better_than_ref"] == 0
    assert t.loc[("var", "xgboost_cal"), "folds_better_than_ref"] == 2
    assert t.loc[("var", "hist_var_cal"), "folds_better_than_ref"] == 0


def test_significance_markdown_renders_saved_tables(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "REPORTS", tmp_path)
    assert report.significance_markdown() == ""  # nothing saved yet
    pd.DataFrame({"target": ["vol"], "model": ["rf"], "vs": ["reference"], "baseline": ["naive"],
                  "mean_loss_diff": [-0.001], "dm_stat": [-2.1], "p": [0.04], "better": [True],
                  "significant": [True], "n_dates": [330]}).to_csv(
        tmp_path / "significance_dm.csv", index=False)
    pd.DataFrame({"model": ["rf", "naive"], "mcs_p": [1.0, 0.01], "in_set": [True, False],
                  "target": "vol"}).to_csv(tmp_path / "mcs.csv", index=False)
    pd.DataFrame({"model": ["rf"] * 3, "period": ["all", "2020", "2021"], "n_dates": [330, 50, 50],
                  "exceed_rate": [0.055] * 3, "stat": [1.0] * 3, "p": [0.3, 0.01, 0.5]}).to_csv(
        tmp_path / "coverage_clustered.csv", index=False)
    md = report.significance_markdown()
    assert "Diebold-Mariano" in md and "Model confidence set" in md
    assert "1/2" in md  # one of two yearly tests rejected
