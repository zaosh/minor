"""Synthetic portfolio generator and user-portfolio builder.

Portfolios hold constant weights, i.e. they are rebalanced daily to target weights.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src import config


@dataclass(frozen=True)
class Portfolio:
    """A long-only portfolio: tickers and weights summing to 1."""

    tickers: tuple[str, ...]
    weights: tuple[float, ...]

    @property
    def hhi(self) -> float:
        """Herfindahl concentration index (sum of squared weights)."""
        return float(np.sum(np.square(self.weights)))


def generate_portfolios(
    n: int,
    universe: list[str],
    seed: int = config.SEED,
    k_range: tuple[int, int] = config.K_RANGE,
) -> list[Portfolio]:
    """Draw `n` portfolios: k ~ U{k_range}, tickers without replacement, weights ~ Dirichlet(1)."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        k = int(rng.integers(k_range[0], k_range[1] + 1))
        picks = rng.choice(len(universe), size=k, replace=False)
        w = rng.dirichlet(np.ones(k))
        out.append(Portfolio(tuple(universe[i] for i in picks), tuple(float(x) for x in w)))
    return out


def portfolio_returns(returns: pd.DataFrame, p: Portfolio) -> pd.Series:
    """Daily portfolio return series (constant weights)."""
    return (returns[list(p.tickers)] * np.array(p.weights)).sum(axis=1).rename("port_ret")


def build_user_portfolio(
    tickers: list[str], weights: list[float], available: list[str], tol: float = 1e-3
) -> Portfolio:
    """Validate user input and return a Portfolio.

    Zero-weight tickers are dropped. Raises ValueError on unknown/duplicate tickers, non-finite or
    negative weights, or weights not summing to 1.
    """
    if len(tickers) != len(weights) or not tickers:
        raise ValueError("Provide the same, non-zero number of tickers and weights.")
    if len(set(tickers)) != len(tickers):
        raise ValueError("Duplicate tickers.")
    unknown = [t for t in tickers if t not in available]
    if unknown:
        raise ValueError(f"Unknown tickers: {unknown}")
    if not np.all(np.isfinite(weights)):
        raise ValueError("Weights must be finite numbers.")
    if any(w < 0 for w in weights):
        raise ValueError("Weights must be non-negative.")
    if abs(sum(weights) - 1.0) > tol:
        raise ValueError(f"Weights must sum to 1 (got {sum(weights):.4f}).")
    kept = [(t, w) for t, w in zip(tickers, weights, strict=True) if w > 0]
    total = sum(w for _, w in kept)
    return Portfolio(tuple(t for t, _ in kept), tuple(float(w / total) for _, w in kept))
