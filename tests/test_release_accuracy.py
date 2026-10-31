"""Acceptance invariants for the remaining scientifically consequential defects."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from elessar import StoneConfig
from elessar.correlations import compute_correlations
from elessar.correlations import correlation_ratio_test
from elessar.selection import feature_selection
from elessar.stats import _median_ci, numeric_stats
from elessar.distributions import _discrete_gof, fit_distributions
from elessar.serial_inference import serial_layout, serial_test

CFG = StoneConfig(n_jobs=1)


def temporal_frame():
    rng = np.random.default_rng(641001)
    n = 240
    x, y = np.zeros(n), np.zeros(n)
    for i in range(1, n):
        x[i] = .9 * x[i - 1] + rng.normal()
        y[i] = .9 * y[i - 1] + rng.normal()
    missing = rng.random(n) < .3
    df = pd.DataFrame({"x": x, "y": y, "partial": np.where(missing, np.nan, rng.normal(size=n))})
    serial = {"time": pd.Series(pd.date_range("2020-01-01", periods=n)), "entity": None, "differences": None}
    return df, dict.fromkeys(df.columns, "numeric_continuous"), serial


def test_numeric_target_shape_uses_temporal_calibration():
    df, types, serial = temporal_frame()
    out = compute_correlations(df, types, target="y", config=CFG, serial=serial)
    tested = out["serial_inference"]
    rows = tested[(tested["output"] == "target") & (tested["col_a"] == "x")]
    assert len(rows) == 1 and rows.iloc[0]["test"].startswith("symmetric truncated")
    assert out["target"].set_index("feature").loc["x", "p_value"] == rows.iloc[0]["p_value"]


@pytest.mark.parametrize("output", ["missingness", "missing_target"])
def test_missingness_serial_tests_use_the_actual_temporal_grid(output):
    df, types, serial = temporal_frame()
    out = compute_correlations(df, types, target="y", config=CFG, serial=serial)
    assert output in set(out["serial_inference"]["output"])
    rows = out[output]
    assert not rows.empty and rows["test"].str.contains("truncated time-shift").all()


def test_legitimate_near_deterministic_predictor_is_flagged_without_automatic_exclusion():
    rng = np.random.default_rng(641002)
    df = pd.DataFrame({"signal": rng.normal(size=240), "noise": rng.normal(size=240)})
    df["y"] = 3 * df.signal + .1 * rng.normal(size=len(df))
    out = feature_selection(df, "y", dict.fromkeys(df.columns, "numeric_continuous"), {}, CFG)
    assert "signal" in out["keep"]
    assert "signal" not in out["excluded_leaks"]
    assert "signal" in out["leakage_suspects"]
    assert out["consensus_evaluation"]["score"] > .9


def test_median_interval_meets_its_nominal_finite_sample_coverage():
    n = 21
    lo, hi = _median_ci(np.arange(n, dtype=float))
    # For continuous IID observations, the number below the population median is Binomial(n,.5).
    coverage = stats.binom.cdf(int(hi), n, .5) - stats.binom.cdf(int(lo), n, .5)
    assert coverage >= .95


def test_finite_large_values_have_representable_standard_deviation_and_shape():
    x = np.array([-.8, -.3, .2, .9])
    actual = numeric_stats(pd.Series(x * 1e200))
    assert np.isfinite(actual["std"])
    assert actual["std"] / 1e200 == pytest.approx(np.std(x, ddof=1))
    assert actual["skewness"] == pytest.approx(stats.skew(x, bias=False))
    assert actual["kurtosis"] == pytest.approx(stats.kurtosis(x, bias=False))


def test_variance_calculation_preserves_small_spread_around_a_large_offset():
    origin, step = np.ldexp(1., 600), np.ldexp(1., 560)
    values = origin + np.array([-3., -1., 1., 3.]) * step
    actual = numeric_stats(pd.Series(values))
    assert actual["std"] / (step * np.sqrt(20 / 3)) == pytest.approx(1., rel=1e-13)


def test_harmonic_mean_of_finite_subnormal_values_does_not_underflow_to_zero():
    tiny = np.nextafter(0., 1.)
    actual = numeric_stats(pd.Series([tiny, tiny, 1., 2.]))
    assert actual["harmonic_mean"] == 2 * tiny


@pytest.mark.parametrize("value", [1e200, 1e-300, np.nextafter(0., 1.)])
def test_constant_finite_values_preserve_the_mean_at_extreme_scales(value):
    actual = numeric_stats(pd.Series([value] * 8))
    assert actual["mean"] == value and actual["std"] == 0 and actual["range"] == 0


def test_integer_cdf_gap_includes_the_positions_before_observed_jumps():
    values = np.array([1., 1., 1., 2., 2., 2., 2., 4., 4., 4.])
    grid = np.arange(0, 20)
    ecdf = (values[:, None] <= grid).mean(axis=0)
    expected = np.max(np.abs(ecdf - stats.poisson.cdf(grid, 2.)))
    observed = _discrete_gof(values, stats.poisson, (2.,), 0)[0]
    assert observed == pytest.approx(expected)


def test_fitted_model_diagnostics_do_not_claim_an_unadjusted_pvalue_is_calibrated():
    values = np.random.default_rng(641003).normal(size=120)
    fit = fit_distributions(values, discrete=False, config=StoneConfig(candidates_continuous=("norm",)))["fits"][0]
    assert np.isnan(fit["ks_pvalue"])
    assert fit["gof_status"] == "uncalibrated fitted-model diagnostic"
    assert np.isfinite(fit["diagnostic_pvalue"])


@pytest.mark.parametrize("kind", ["auc", "target_numeric"])
def test_new_serial_statistics_match_independent_scipy_references(kind):
    rng = np.random.default_rng(641004)
    n, radius = 120, 30
    x = rng.integers(2, size=n).astype(bool) if kind == "auc" else rng.integers(5, size=n).astype(float)
    y = rng.normal(size=n)
    df = pd.DataFrame({"x": x, "y": y})
    grid = {"time": pd.Series(pd.date_range("2020-01-01", periods=n)), "entity": None, "differences": None}
    result = serial_test(serial_layout(df, grid, CFG), df.x, df.y, kind, CFG)

    def statistic(a, b):
        if kind == "auc":
            positive = a.astype(bool)
            u = stats.mannwhitneyu(b[positive], b[~positive]).statistic
            return float(2 * u / (positive.sum() * (~positive).sum()) - 1)
        bins = pd.qcut(pd.Series(a).rank(method="average"), q=10, labels=False, duplicates="drop")
        eta = correlation_ratio_test(bins, pd.Series(b))[0]
        return max(abs(float(stats.spearmanr(a, b).statistic)), eta if np.isfinite(eta) else 0.)

    center = slice(radius, n - radius)
    observed = statistic(x[center], y[center])
    counts = [0, 0]
    for shift in range(-radius, radius + 1):
        w = slice(radius + shift, n - radius + shift)
        counts[0] += abs(statistic(x[center], y[w])) >= abs(observed) - 1e-12
        counts[1] += abs(statistic(x[w], y[center])) >= abs(observed) - 1e-12
    assert result["statistic"] == pytest.approx(observed, abs=1e-12)
    assert result["p_value"] == min(1., max(counts) / (radius + 1))


@pytest.mark.parametrize("quarantine", [False, True])
def test_exact_alias_quarantine_is_an_explicit_overridable_policy(quarantine):
    rng = np.random.default_rng(641005)
    df = pd.DataFrame({"x": rng.normal(size=180), "noise": rng.normal(size=180)})
    df["y"] = 3 * df.x + 1
    cfg = StoneConfig(n_jobs=1, exclude_target_aliases=quarantine)
    out = feature_selection(df, "y", dict.fromkeys(df.columns, "numeric_continuous"), {}, cfg)
    assert "x" in out["target_aliases"]
    assert ("x" in out["excluded_leaks"]) == quarantine
    assert ("x" in out["training_features"]) != quarantine


def test_provenance_exclusions_override_association_heuristics():
    rng = np.random.default_rng(641006)
    df = pd.DataFrame(rng.normal(size=(180, 3)), columns=["known_post_outcome", "available", "y"])
    cfg = StoneConfig(n_jobs=1, exclude_features=("known_post_outcome",))
    out = feature_selection(df, "y", dict.fromkeys(df.columns, "numeric_continuous"), {}, cfg)
    assert "known_post_outcome" in out["excluded_leaks"] and "known_post_outcome" not in out["training_features"]


@pytest.mark.parametrize("excluded", [("typo",), ("y",)])
def test_invalid_provenance_exclusions_fail_explicitly(excluded):
    df = pd.DataFrame({"x": np.arange(40.), "y": np.arange(40.)})
    with pytest.raises(ValueError, match="exclude_features"):
        feature_selection(df, "y", dict.fromkeys(df.columns, "numeric_continuous"), {},
                          StoneConfig(exclude_features=excluded))
