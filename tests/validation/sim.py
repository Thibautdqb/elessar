"""Shared helpers of the validation suite."""

from __future__ import annotations

import numpy as np
import pandas as pd


def upper(rate: float, n: int, z: float = 3.0) -> float:
    """``rate`` plus ``z`` binomial standard errors: what a correct procedure stays under in ``n`` trials."""
    return rate + z * np.sqrt(rate * (1 - rate) / n)


def lower(rate: float, n: int, z: float = 3.0) -> float:
    return rate - z * np.sqrt(rate * (1 - rate) / n)


def factor_data(rng: np.random.Generator, k: int, per: int, n: int, loading: tuple[float, float] = (0.45, 0.8),
                factor_r: float = 0.0, junk: int = 0) -> pd.DataFrame:
    """Items driven by ``k`` latent factors (``per`` items each), plus ``junk`` items that load on nothing."""
    lam = rng.uniform(*loading, k * per)
    phi = np.full((k, k), factor_r) + (1 - factor_r) * np.eye(k)
    f = rng.multivariate_normal(np.zeros(k), phi, n)
    load = np.zeros((k * per, k))
    for j in range(k):
        load[j * per:(j + 1) * per, j] = lam[j * per:(j + 1) * per]
    x = f @ load.T + rng.normal(size=(n, k * per)) * np.sqrt(1 - lam**2)
    if junk:
        x = np.column_stack([x, rng.normal(size=(n, junk))])
    return pd.DataFrame(x, columns=[f"item{i}" for i in range(x.shape[1])])
