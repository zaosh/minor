"""Causal (parameter-free) risk baselines, usable as panel columns and as app comparisons.

Every function at row t uses data up to and including t only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.covariance import ledoit_wolf

from src import config
from src.models import prereg

ANN = np.sqrt(252)
Z95 = 1.6448536269514722
# E[Z | Z <= -Z95] for a standard normal = -phi(Z95) / alpha
ES_NORMAL = float(-np.exp(-0.5 * Z95**2) / np.sqrt(2 * np.pi) / config.VAR_ALPHA)


def naive_vol(r: pd.Series, window: int = 21) -> pd.Series:
    """Current realized vol (annualized): the 'tomorrow looks like today' forecast."""
    return (r.rolling(window).std() * ANN).rename("naive_vol")


def ewma_vol(r: pd.Series, lam: float = 0.94, min_periods: int = 30) -> pd.Series:
    """RiskMetrics EWMA vol (annualized); no fitted parameters."""
    var = (r**2).ewm(alpha=1 - lam, adjust=False, min_periods=min_periods).mean()
    return (np.sqrt(var) * ANN).rename("ewma_vol")


def hist_var(r: pd.Series, window: int = 250, alpha: float = config.VAR_ALPHA) -> pd.Series:
    """Historical-simulation 1-day VaR: the alpha-quantile of the last `window` returns."""
    return r.rolling(window).quantile(alpha).rename("hist_var250")


def param_var(r: pd.Series, window: int = 63, z: float = Z95) -> pd.Series:
    """Parametric (normal) 1-day VaR as a return quantile: mean - z * std over `window`."""
    return (r.rolling(window).mean() - z * r.rolling(window).std()).rename("param_var63")


def hist_es_at(r: np.ndarray, days: np.ndarray, window: int = 250,
               alpha: float = config.VAR_ALPHA) -> np.ndarray:
    """Historical expected shortfall at day t: mean of the last `window` returns at or below
    their alpha-quantile (the same window and quantile as `hist_var`). Uses rows <= t only."""
    out = np.full(len(days), np.nan)
    for n, t in enumerate(days):
        if t + 1 < window:
            continue
        w = r[t + 1 - window : t + 1]
        out[n] = w[w <= np.quantile(w, alpha)].mean()
    return out


def param_es_at(r: np.ndarray, days: np.ndarray, window: int = 63) -> np.ndarray:
    """Normal expected shortfall at day t: mean63 + std63 * E[Z | Z <= -z] (rows <= t only)."""
    s = pd.Series(r)
    mu, sd = s.rolling(window).mean().to_numpy(), s.rolling(window).std().to_numpy()
    return (mu + sd * ES_NORMAL)[days]


def es_from_vol(sigma_ann: np.ndarray | pd.Series) -> np.ndarray:
    """Zero-mean normal 1-day 5% expected shortfall (negative) from annualized vol."""
    return ES_NORMAL * np.asarray(sigma_ann, dtype=float) / ANN


def fhs_quantiles(z: np.ndarray, alpha: float = config.VAR_ALPHA) -> tuple[float, float]:
    """Empirical alpha-quantile of standardized residuals and the mean of those at or below it."""
    z = z[np.isfinite(z)]
    q = float(np.quantile(z, alpha))
    return q, float(z[z <= q].mean())


def lw_vol(
    member_rets: np.ndarray, weights: np.ndarray, idx: np.ndarray, window: int = 250
) -> np.ndarray:
    """Covariance-route vol sqrt(w' Sigma w * 252) with a Ledoit-Wolf Sigma on trailing returns.

    `member_rets` is (T, k); for each position t in `idx` the window is rows t-window+1..t.
    """
    out = np.full(len(idx), np.nan)
    for n, t in enumerate(idx):
        if t + 1 < window:
            continue
        cov, _ = ledoit_wolf(member_rets[t + 1 - window : t + 1])
        out[n] = np.sqrt(weights @ cov @ weights * 252)
    return out


# ---------------------------------------------------------------------------------------
# Model-style baselines (implement src.models.base.RiskModel)
# ---------------------------------------------------------------------------------------
_DD_GRID = np.linspace(0.02, 2.0, 100)
_DD_TABLE: np.ndarray | None = None


def dd_prob_from_vol(sigma_ann: np.ndarray | pd.Series) -> np.ndarray:
    """P(max drawdown over H days >= DD_THRESH) for zero-drift Gaussian returns with given vol.

    Monte Carlo (seed 42) on a vol grid, cached, then interpolated: monotone in sigma.
    """
    global _DD_TABLE
    if _DD_TABLE is None:
        rng = np.random.default_rng(config.SEED)
        z = rng.standard_normal((4000, config.H))
        table = []
        for s in _DD_GRID:
            w = np.concatenate(
                [np.ones((4000, 1)), np.cumprod(1 + s / np.sqrt(252) * z, axis=1)], axis=1
            )
            mdd = (w / np.maximum.accumulate(w, axis=1) - 1).min(axis=1)
            table.append((mdd <= -config.DD_THRESH).mean())
        _DD_TABLE = np.maximum.accumulate(np.array(table))
    return np.interp(np.asarray(sigma_ann, dtype=float), _DD_GRID, _DD_TABLE)


def var_from_vol(sigma_ann: np.ndarray | pd.Series) -> np.ndarray:
    """Zero-mean normal 1-day 5% VaR (a negative return quantile) from annualized vol."""
    return -Z95 * np.asarray(sigma_ann, dtype=float) / ANN


def _frame(index: pd.Index) -> pd.DataFrame:
    from src.models.base import empty_preds

    return empty_preds(index)


class VolColumnBaseline:
    """Baseline that reads a precomputed causal vol column (naive_vol, ewma_vol, lw_vol)."""

    def __init__(self, name: str, column: str):
        self.name, self.column = name, column

    def fit(self, train: pd.DataFrame, val: pd.DataFrame | None = None) -> None:
        """Nothing to fit."""

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        out = _frame(X.index)
        out["vol_hat"] = X[self.column].to_numpy()
        out["dd_hat"] = dd_prob_from_vol(out["vol_hat"])
        out["var_hat"] = var_from_vol(out["vol_hat"])
        out["es_hat"] = es_from_vol(out["vol_hat"])
        return out


class VarColumnBaseline:
    """VaR baseline reading a precomputed column (hist_var250 / param_var63).

    With `port_returns` ((n_portfolios, T), aligned with `day_idx`) it also reports the matching
    expected shortfall: `es_fn(returns_of_one_portfolio, days)`.
    """

    def __init__(self, name: str, column: str, port_returns: np.ndarray | None = None,
                 es_fn=None):
        self.name, self.column, self.R, self.es_fn = name, column, port_returns, es_fn

    def fit(self, train: pd.DataFrame, val: pd.DataFrame | None = None) -> None:
        """Nothing to fit."""

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        out = _frame(X.index)
        out["var_hat"] = X[self.column].to_numpy()
        if self.R is not None:
            pids = X.index.get_level_values("pid").to_numpy()
            days = X["day_idx"].to_numpy().astype(int)
            for pid in np.unique(pids):
                m = pids == pid
                out.loc[m, "es_hat"] = self.es_fn(self.R[pid].astype("float64"), days[m])
        return out


class Climatology:
    """Predicts the training base rate of drawdown events for every row."""

    name = "climatology"

    def fit(self, train: pd.DataFrame, val: pd.DataFrame | None = None) -> None:
        self.rate = float(train["dd_flag"].mean())

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        out = _frame(X.index)
        out["dd_hat"] = self.rate
        return out


class HAR:
    """HAR-style OLS: log vol_fwd ~ log vol_21 + log vol_63 + log vol_126, fit per fold."""

    name = "har"
    COLS = ("vol_21", "vol_63", "vol_126")

    def _design(self, X: pd.DataFrame) -> np.ndarray:
        logs = np.log(X[list(self.COLS)].clip(lower=1e-4).to_numpy(dtype=float))
        return np.column_stack([np.ones(len(X)), logs])

    def fit(self, train: pd.DataFrame, val: pd.DataFrame | None = None) -> None:
        y = np.log(train["vol_fwd"].clip(lower=1e-4).to_numpy(dtype=float))
        self.coef_, *_ = np.linalg.lstsq(self._design(train), y, rcond=None)

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        out = _frame(X.index)
        out["vol_hat"] = np.exp(self._design(X) @ self.coef_)
        out["dd_hat"] = dd_prob_from_vol(out["vol_hat"])
        out["var_hat"] = var_from_vol(out["vol_hat"])
        out["es_hat"] = es_from_vol(out["vol_hat"])
        return out


_GARCH_SPECS = {
    "garch": dict(vol="GARCH", p=1, q=1),
    "gjr": dict(vol="GARCH", p=prereg.GJR_SPEC["p"], o=prereg.GJR_SPEC["o"],
                q=prereg.GJR_SPEC["q"], dist=prereg.GJR_SPEC["dist"]),
}


def std_t_var_es(nu: float, alpha: float = config.VAR_ALPHA) -> tuple[float, float]:
    """Alpha-quantile and expected shortfall (both negative) of a UNIT-VARIANCE Student t.

    With t_q the alpha-quantile of the ordinary t(nu): ES = -(f(t_q) / alpha) * (nu + t_q^2) /
    (nu - 1), and both are rescaled by sqrt((nu - 2) / nu) so that the variance is one.
    """
    from scipy import stats

    tq = float(stats.t.ppf(alpha, nu))
    c = float(np.sqrt((nu - 2.0) / nu))
    es = -float(stats.t.pdf(tq, nu)) / alpha * (nu + tq**2) / (nu - 1.0)
    return tq * c, es * c


class GarchFitCache:
    """Per-portfolio GARCH(1,1) fits shared by the GARCH and FHS baselines (arch runs once).

    `port_returns` is (n_portfolios, T) daily portfolio returns aligned with `day_idx`. A fit for
    (pid, start) uses returns strictly before day `start` for the parameters and the standardized
    residuals; conditional variances are then filtered through day `last`, so the forecast at
    day t uses returns up to t only. `spec="gjr"` fits the GJR-GARCH(1,1,1) with Student t
    errors instead (Round 4, B1); the default spec is unchanged since Round 3.
    """

    def __init__(self, port_returns: np.ndarray, spec: str = "garch"):
        self.R = port_returns
        self.spec = spec
        self._cache: dict[tuple[int, int, int], dict | None] = {}

    def get(self, pid: int, start: int, last: int) -> dict | None:
        """Fit result or None on non-convergence: keys mu, vol21, sigma1 (indexed by day -
        start, returns units), z (standardized residuals before `start`), params (omega in
        returns-in-percent squared units, alpha, beta, and for the t spec gamma and nu)."""
        key = (pid, start, last)
        if key not in self._cache:
            self._cache[key] = self._fit(pid, start, last)
        return self._cache[key]

    def _fit(self, pid: int, start: int, last: int) -> dict | None:
        from arch import arch_model

        r = self.R[pid] * 100.0
        am = arch_model(r, mean="Constant", rescale=False, **_GARCH_SPECS[self.spec])
        try:
            res = am.fit(last_obs=start, disp="off", show_warning=False)
            if res.convergence_flag != 0:
                return None
            f = res.forecast(horizon=config.H, start=start, reindex=False)
        except Exception:
            return None
        v = f.variance.iloc[: last - start + 1].to_numpy()  # rows are origins start..last
        z = np.asarray(res.std_resid, dtype=float)
        pr = res.params
        params = {"omega": float(pr["omega"]), "alpha": float(pr["alpha[1]"]),
                  "beta": float(pr["beta[1]"])}
        if self.spec == "gjr":
            params.update(gamma=float(pr["gamma[1]"]), nu=float(pr["nu"]))
        return {"mu": float(pr["mu"]) / 100.0,
                "vol21": np.sqrt(v.mean(axis=1) * 252) / 100.0,
                "sigma1": np.sqrt(v[:, 0]) / 100.0, "z": z[np.isfinite(z)][:start],
                "params": params}


class GarchBaseline:
    """Per-portfolio GARCH(1,1), parameters fit on returns before the first test day.

    Parameters are fixed at the estimate; conditional variances are then filtered through the
    test period, so a forecast at day t uses returns up to t only. Non-convergence falls back
    to EWMA vol and is counted in `n_fallback`. VaR and ES assume normal innovations.
    """

    name = "garch"

    def __init__(self, port_returns: np.ndarray, cache: GarchFitCache | None = None):
        self.R = port_returns
        self.cache = cache or GarchFitCache(port_returns)
        self.n_fallback = 0

    def fit(self, train: pd.DataFrame, val: pd.DataFrame | None = None) -> None:
        """Parameters are fit lazily in predict() so the cut-off is the first test day."""

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        out = _frame(X.index)
        pids = X.index.get_level_values("pid").to_numpy()
        days = X["day_idx"].to_numpy().astype(int)
        start = int(days.min())  # parameters use returns strictly before this day
        for pid in np.unique(pids):
            m = pids == pid
            res = self.cache.get(int(pid), start, int(days[m].max()))
            if res is None:
                self.n_fallback += 1
                out.loc[m, "vol_hat"] = X.loc[m, "ewma_vol"].to_numpy()
                out.loc[m, "var_hat"] = var_from_vol(X.loc[m, "ewma_vol"])
                out.loc[m, "es_hat"] = es_from_vol(X.loc[m, "ewma_vol"])
            else:
                k = days[m] - start
                out.loc[m, "vol_hat"] = res["vol21"][k]
                out.loc[m, "var_hat"] = -Z95 * res["sigma1"][k]
                out.loc[m, "es_hat"] = ES_NORMAL * res["sigma1"][k]
        out["dd_hat"] = dd_prob_from_vol(out["vol_hat"])
        return out


class FHSBaseline:
    """Filtered historical simulation: GARCH filter plus the empirical residual quantile.

    var_hat = mu + sigma_{t+1|t} * q(z), es_hat = mu + sigma_{t+1|t} * mean(z | z <= q(z)), where
    z are the standardized residuals of returns BEFORE the first test day (same cut-off as the
    GARCH parameters). A failed GARCH fit falls back to the EWMA filter: sigma is the EWMA
    vol and z_s = r_s / sigma^EWMA_{s-1} over the same earlier returns. VaR/ES only.
    """

    name = "fhs"
    EWMA_WARMUP = 30

    def __init__(self, port_returns: np.ndarray, cache: GarchFitCache | None = None):
        self.R = port_returns
        self.cache = cache or GarchFitCache(port_returns)
        self.n_fallback = 0

    def fit(self, train: pd.DataFrame, val: pd.DataFrame | None = None) -> None:
        """Parameters and residual quantiles are fit lazily in predict() (first test day)."""

    def _ewma_z(self, pid: int, start: int) -> np.ndarray:
        r = pd.Series(self.R[pid][:start].astype("float64"))
        sig = np.sqrt((r**2).ewm(alpha=0.06, adjust=False, min_periods=self.EWMA_WARMUP).mean())
        return (r / sig.shift(1)).to_numpy()[self.EWMA_WARMUP + 1 :]

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        out = _frame(X.index)
        pids = X.index.get_level_values("pid").to_numpy()
        days = X["day_idx"].to_numpy().astype(int)
        start = int(days.min())
        for pid in np.unique(pids):
            m = pids == pid
            res = self.cache.get(int(pid), start, int(days[m].max()))
            if res is None:
                self.n_fallback += 1
                q, es = fhs_quantiles(self._ewma_z(int(pid), start))
                sigma, mu = X.loc[m, "ewma_vol"].to_numpy() / ANN, 0.0
            else:
                q, es = fhs_quantiles(res["z"])
                sigma, mu = res["sigma1"][days[m] - start], res["mu"]
            out.loc[m, "var_hat"] = mu + sigma * q
            out.loc[m, "es_hat"] = mu + sigma * es
        return out


class GJRBaseline:
    """Per-portfolio GJR-GARCH(1,1,1) with Student t errors (Round 4, B1).

    Same fit policy as `GarchBaseline`: parameters from returns strictly before the first test
    day, then fixed; the variance is filtered through the test period and the 21-step forecast
    is analytic. VaR/ES are zero-mean with the fitted standardised-t quantile and shortfall;
    drawdown probability uses the Gaussian lookup on the vol forecast. A failed fit falls back
    to EWMA vol (normal VaR/ES) and is counted in `n_fallback`.
    """

    name = "gjr"

    def __init__(self, port_returns: np.ndarray, cache: GarchFitCache | None = None):
        self.R = port_returns
        self.cache = cache or GarchFitCache(port_returns, "gjr")
        self.n_fallback = 0

    def fit(self, train: pd.DataFrame, val: pd.DataFrame | None = None) -> None:
        """Parameters are fit lazily in predict() so the cut-off is the first test day."""

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        out = _frame(X.index)
        pids = X.index.get_level_values("pid").to_numpy()
        days = X["day_idx"].to_numpy().astype(int)
        start = int(days.min())
        for pid in np.unique(pids):
            m = pids == pid
            res = self.cache.get(int(pid), start, int(days[m].max()))
            if res is None:
                self.n_fallback += 1
                out.loc[m, "vol_hat"] = X.loc[m, "ewma_vol"].to_numpy()
                out.loc[m, "var_hat"] = var_from_vol(X.loc[m, "ewma_vol"])
                out.loc[m, "es_hat"] = es_from_vol(X.loc[m, "ewma_vol"])
            else:
                k = days[m] - start
                q, es = std_t_var_es(res["params"]["nu"])
                out.loc[m, "vol_hat"] = res["vol21"][k]
                out.loc[m, "var_hat"] = q * res["sigma1"][k]
                out.loc[m, "es_hat"] = es * res["sigma1"][k]
        out["dd_hat"] = dd_prob_from_vol(out["vol_hat"])
        return out


def default_baselines(port_returns: np.ndarray | None = None) -> list:
    """All baselines in the comparison; ES, GARCH and FHS need the portfolio returns."""
    models: list = [
        VolColumnBaseline("naive", "naive_vol"),
        VolColumnBaseline("ewma", "ewma_vol"),
        VolColumnBaseline("ledoit_wolf", "lw_vol"),
        HAR(),
        Climatology(),
        VarColumnBaseline("hist_var", "hist_var250", port_returns, hist_es_at),
        VarColumnBaseline("param_var", "param_var63", port_returns, param_es_at),
    ]
    if port_returns is not None:
        cache = GarchFitCache(port_returns)
        models += [GarchBaseline(port_returns, cache), FHSBaseline(port_returns, cache),
                   GJRBaseline(port_returns, GarchFitCache(port_returns, "gjr"))]
    return models
