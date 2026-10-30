"""Time series: seasonality, trend under autocorrelation, stationarity verdicts."""

from __future__ import annotations

import zlib

import numpy as np
import pandas as pd
import pytest

from elessar import StoneConfig
from elessar.timeseries import analyze_series, mann_kendall, stationarity

from .sim import upper

pytestmark = pytest.mark.validation
CFG = StoneConfig()


def _ar1(rng: np.random.Generator, n: int, phi: float) -> np.ndarray:
    e = rng.normal(size=n)
    y = np.empty(n)
    y[0] = e[0] / np.sqrt(1 - phi**2)
    for i in range(1, n):
        y[i] = phi * y[i - 1] + e[i]
    return y


@pytest.mark.parametrize("case", ["white noise", "AR(1) 0.7", "weekly cycle", "weekly cycle, weak", "monthly, yearly cycle"])
def test_seasonality_detection(case: str, record) -> None:  # noqa: ANN001
    rng = np.random.default_rng(zlib.crc32(case.encode()))
    runs = 200
    hits, right_period = 0, 0
    freq, period = ("month", 12) if case.startswith("monthly") else ("day", 7)
    for _ in range(runs):
        n = 120 if freq == "month" else 365
        k = np.arange(n)
        if case == "white noise":
            y = rng.normal(size=n)
        elif case == "AR(1) 0.7":
            y = _ar1(rng, n, 0.7)
        else:
            amp = 0.6 if case.endswith("weak") else 2.0
            y = amp * np.sin(2 * np.pi * k / period) + _ar1(rng, n, 0.3)
        out = analyze_series(pd.Series(y), freq, CFG)
        hits += out["period"] is not None
        right_period += out["period"] == period
    seasonal = "cycle" in case
    record(case=case, runs=runs, frequency=freq, detected=hits / runs, right_period=right_period / runs,
           truth="seasonal" if seasonal else "no seasonality")
    if seasonal and not case.endswith("weak"):
        assert right_period / runs > 0.95
    if not seasonal:
        assert hits / runs < upper(0.05, runs)


def _arma(rng: np.random.Generator, n: int, ar: tuple = (), ma: tuple = (), burn: int = 300) -> np.ndarray:
    e = rng.normal(size=n + burn)
    y = np.zeros(n + burn)
    for t in range(n + burn):
        y[t] = e[t] + sum(a * y[t - 1 - i] for i, a in enumerate(ar) if t > i) \
            + sum(m * e[t - 1 - i] for i, m in enumerate(ma) if t > i)
    return y[burn:]


def _long_memory(rng: np.random.Generator, n: int, hurst: float = 0.8) -> np.ndarray:
    from scipy.linalg import cholesky, toeplitz

    k = np.arange(n)
    gamma = 0.5 * (np.abs(k + 1) ** (2 * hurst) - 2 * k ** (2 * hurst) + np.abs(k - 1) ** (2 * hurst))
    return cholesky(toeplitz(gamma), lower=True) @ rng.normal(size=n)


NOISES = {
    "white noise": lambda rng, n: rng.normal(size=n),
    "AR(1) 0.5": lambda rng, n: _ar1(rng, n, 0.5),
    "AR(1) 0.8": lambda rng, n: _ar1(rng, n, 0.8),
    "AR(2) 0.5, 0.3": lambda rng, n: _arma(rng, n, (0.5, 0.3)),
    "MA(1) 0.7": lambda rng, n: _arma(rng, n, (), (0.7,)),
    "ARMA(1,1) 0.7, 0.4": lambda rng, n: _arma(rng, n, (0.7,), (0.4,)),
    "long memory (Hurst 0.8)": lambda rng, n: _long_memory(rng, n),
}


@pytest.mark.parametrize("noise", list(NOISES))
def test_mann_kendall_keeps_its_type_i_error_under_autocorrelation(noise: str, record) -> None:  # noqa: ANN001
    """No trend, autocorrelated noise: the share of series in which a trend is reported at 5%.

    The correction keeps the uncorrected 26-55% of false trends to 2-10% on short-memory noise. Known limit:
    long memory is hardly distinguishable from a trend on 200 points, and no correction based on a
    short-memory model brings it back to 5% (about 25%: measured, not asserted)."""
    rng = np.random.default_rng(zlib.crc32(noise.encode()))
    runs, n = 600, 200
    corrected = naive = 0
    for _ in range(runs):
        mk = mann_kendall(NOISES[noise](rng, n))
        corrected += mk["p_value"] < 0.05
        naive += mk["p_value_naive"] < 0.05
    record(noise=noise, runs=runs, n=n, type_i_corrected=corrected / runs, type_i_uncorrected=naive / runs)
    if not noise.startswith("long memory"):  # claim: at most 10% (twice the nominal level; AR(2) is the worst)
        assert corrected / runs < upper(0.10, runs)


def test_mann_kendall_power_on_a_trend(record) -> None:  # noqa: ANN001
    rng = np.random.default_rng(55)
    runs, n = 500, 200
    found = sum(mann_kendall(0.005 * np.arange(n) + _ar1(rng, n, 0.5))["trend"] == "increasing" for _ in range(runs))
    record(runs=runs, n=n, trend="0.005 per step under AR(1) 0.5 noise (1 sd of the series over the period)",
           power=found / runs)
    assert found / runs > 0.4  # regression guard


@pytest.mark.parametrize("kind", ["white noise", "AR(1) 0.5", "random walk", "linear trend + noise"])
def test_stationarity_verdicts(kind: str, record) -> None:  # noqa: ANN001
    rng = np.random.default_rng(zlib.crc32(kind.encode()))
    runs, n = 300, 300
    expected = {"white noise": "stationary", "AR(1) 0.5": "stationary", "random walk": "unit root",
                "linear trend + noise": "trend-stationary"}[kind]
    verdicts = []
    for _ in range(runs):
        if kind == "white noise":
            y = rng.normal(size=n)
        elif kind == "AR(1) 0.5":
            y = _ar1(rng, n, 0.5)
        elif kind == "random walk":
            y = np.cumsum(rng.normal(size=n))
        else:
            y = 0.02 * np.arange(n) + rng.normal(size=n)
        verdicts.append(stationarity(y)["verdict"])
    share = pd.Series(verdicts).value_counts(normalize=True).to_dict()
    opposite = "stationary" if expected == "unit root" else "unit root"
    record(kind=kind, runs=runs, n=n, expected=expected, verdicts=share)
    # ADF and KPSS read together leave honest "inconclusive" cases (a fifth of the random walks at n = 300);
    # what must stay rare is the opposite verdict
    assert share.get(expected, 0) > 0.6 and share.get(opposite, 0) < 0.06
