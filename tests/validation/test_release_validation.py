"""Release calibration: independent seed streams, known truth and numerical identities."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from elessar import StoneConfig
from elessar.correlations import compute_correlations
from elessar.stats import _median_ci, numeric_stats

pytestmark = pytest.mark.validation
CFG = StoneConfig(n_jobs=1)
STREAM = int(os.environ.get("ELESSAR_VALIDATION_SEED_OFFSET", "0"))
RUNS = 500


def ar(rng, n, lag=1):
    values = np.empty(n)
    values[:lag] = rng.normal(size=lag) / np.sqrt(1 - .9**2)
    for i in range(lag, n):
        values[i] = .9 * values[i - lag] + rng.normal()
    return values


def states(rng, n):
    values = np.empty(n, dtype=int)
    values[0] = rng.integers(2)
    for i in range(1, n):
        values[i] = values[i - 1] ^ int(rng.random() < .05)
    return values


def evaluate(rng, design, linked=False):
    n = 600
    serial = {"time": pd.Series(pd.date_range("2020-01-01", periods=n)), "entity": None, "differences": None}
    if design == "numeric_target":
        x = ar(rng, n, lag=2)
        y = .6 * x**2 + rng.normal(size=n) if linked else ar(rng, n, lag=2)
        df = pd.DataFrame({"x": x, "y": y})
        types = dict.fromkeys(df.columns, "numeric_continuous")
        out = compute_correlations(df, types, "y", CFG, serial=serial)
        return float(out["target"].set_index("feature").loc["x", "p_value"])
    y = ar(rng, n)
    missing = y > 0 if linked else states(rng, n).astype(bool)
    if design == "missing_categorical":
        y = (2 * states(rng, n) + rng.integers(2, size=n)).astype(str)
    df = pd.DataFrame({"y": y, "partial": np.where(missing, np.nan, rng.normal(size=n))})
    types = {"y": "categorical" if design == "missing_categorical" else "numeric_continuous", "partial": "numeric_continuous"}
    out = compute_correlations(df, types, "y", CFG, serial=serial)
    m = out["missingness"].set_index(["column", "depends_on"]).loc[("partial", "y"), "p_value"]
    t = out["missing_target"].set_index("feature").loc["partial", "p_value"]
    assert m == t
    return float(t)


def summarize(p):
    values = np.asarray(p)
    count = int((values < .05).sum())
    ci = stats.binomtest(count, len(values)).proportion_ci(.95, method="wilson")
    return {"runs": len(values), "finite": int(np.isfinite(values).sum()), "rejections": count,
            "rate": count / len(values), "wilson95": [float(ci.low), float(ci.high)]}


@pytest.mark.parametrize("design", ["numeric_target", "missing_numeric", "missing_categorical"])
def test_release_serial_null_calibration(design, record):
    seed = 642001 + ["numeric_target", "missing_numeric", "missing_categorical"].index(design) + STREAM * 1_000_000
    rng = np.random.default_rng(seed)
    measured = summarize([evaluate(rng, design) for _ in range(RUNS)])
    upper = .05 + 3 * np.sqrt(.05 * .95 / RUNS)
    record(design=design, seed=seed, seed_stream=STREAM, truth="independent strictly stationary processes",
           alpha=.05, upper_bound=float(upper), **measured)
    assert measured["finite"] == RUNS
    assert measured["rate"] < upper


@pytest.mark.parametrize("design", ["numeric_target", "missing_numeric"])
def test_release_serial_power(design, record):
    seed = 643001 + ["numeric_target", "missing_numeric"].index(design) + STREAM * 1_000_000
    rng = np.random.default_rng(seed)
    measured = summarize([evaluate(rng, design, linked=True) for _ in range(200)])
    record(design=design, seed=seed, seed_stream=STREAM, truth="strong nonlinear target or informative missingness", **measured)
    assert measured["finite"] == 200 and measured["rate"] > .8


def test_exact_median_coverage_across_sample_sizes(record):
    coverage = []
    for n in range(6, 2001):
        lo, hi = _median_ci(np.arange(n, dtype=float))
        coverage.append(float(stats.binom.cdf(int(hi), n, .5) - stats.binom.cdf(int(lo), n, .5)))
    record(sample_sizes=len(coverage), smallest_n=6, largest_n=2000, method="exact binomial order statistics",
           minimum_coverage=min(coverage), nominal_coverage=.95)
    assert min(coverage) >= .95 - 1e-14


def test_numeric_moments_over_500_rescalings(record):
    seed = 644001 + STREAM * 1_000_000
    rng = np.random.default_rng(seed)
    largest_error = 0.
    for _ in range(RUNS):
        n = int(rng.integers(20, 130))
        x = rng.normal(size=n)
        scale = float(10. ** rng.uniform(-300, 300))
        out = numeric_stats(pd.Series(x * scale))
        expected = float(np.std(x, ddof=1))
        error = abs(out["std"] / scale / expected - 1)
        largest_error = max(largest_error, error)
        assert np.isfinite(out["std"]) and error < 1e-12
        assert out["skewness"] == pytest.approx(stats.skew(x, bias=False), abs=1e-12)
        assert out["kurtosis"] == pytest.approx(stats.kurtosis(x, bias=False), abs=1e-12)
    record(runs=RUNS, seed=seed, seed_stream=STREAM, truth="scale identities of sample statistics",
           scale_exponents=[-300, 300], largest_relative_std_error=largest_error)
