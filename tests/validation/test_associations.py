"""Associations: type I error of every pair test, Benjamini-Hochberg, confidence intervals, mutual information."""

from __future__ import annotations

import zlib

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from elessar import StoneConfig, see_corr
from elessar.correlations import (auc_binary, bh_adjust, correlation_ratio_test, cramers_v_test, fisher_ci,
                                  mutual_information)

from .sim import upper

pytestmark = pytest.mark.validation

ALPHA = 0.05
SHAPES = {
    "normal": lambda rng, n: rng.normal(size=n),
    "lognormal": lambda rng, n: rng.lognormal(0, 1, n),
    "student t3": lambda rng, n: rng.standard_t(3, n),
    "uniform": lambda rng, n: rng.uniform(size=n),
    "zero-inflated": lambda rng, n: np.where(rng.random(n) < 0.4, 0, rng.exponential(1, n)),
    "discrete 0-9": lambda rng, n: rng.integers(0, 10, n).astype(float),
}


@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_numeric_pairs_keep_their_type_i_error(shape: str, record) -> None:  # noqa: ANN001
    """Independent columns of one shape: the share of pairs rejected at 5% (raw p) must stay near 5%."""
    rng = np.random.default_rng(zlib.crc32(shape.encode()))
    rejected, total, methods = 0, 0, set()
    for _ in range(4):  # 4 tables of 30 independent columns: 1,740 pairs
        df = pd.DataFrame({f"x{i}": SHAPES[shape](rng, 200) for i in range(30)})
        pairs = see_corr(df)["pairs"]
        rejected += int((pairs["p_value"] < ALPHA).sum())
        total += len(pairs)
        methods |= set(pairs["method"])
    rate = rejected / total
    # the pairs of one table share columns, so they are not independent trials: the bound counts tables of
    # 30 columns as 15 independent pairs each
    record(shape=shape, pairs=total, type_i_rate=rate, methods=sorted(methods))
    assert rate < upper(ALPHA, 4 * 15 * 4)


def test_categorical_and_mixed_tests_keep_their_type_i_error(record) -> None:  # noqa: ANN001
    rng = np.random.default_rng(31)
    runs = 2_000
    chi, kw, auc = [], [], []
    for _ in range(runs):
        a = pd.Series(rng.choice(list("abcd"), 300, p=[0.4, 0.3, 0.2, 0.1]))
        b = pd.Series(rng.choice(list("xyz"), 300))
        v = pd.Series(rng.lognormal(0, 1, 300))
        chi.append(cramers_v_test(a, b)[1] < ALPHA)
        kw.append(correlation_ratio_test(a, v)[1] < ALPHA)
        auc.append(auc_binary(v, pd.Series(rng.choice(["yes", "no"], 300, p=[0.2, 0.8])))[1] < ALPHA)
    rates = {"chi-square (cat-cat)": np.mean(chi), "Kruskal-Wallis (cat-num)": np.mean(kw),
             "Mann-Whitney AUC (binary target)": np.mean(auc)}
    record(runs=runs, **rates)
    for rate in rates.values():
        assert rate < upper(ALPHA, runs)


def test_benjamini_hochberg_controls_the_false_discovery_rate(record) -> None:  # noqa: ANN001
    rng = np.random.default_rng(32)
    fdr, power = [], []
    for _ in range(300):
        m, m1 = 200, 20
        z = np.r_[rng.normal(3, 1, m1), rng.normal(0, 1, m - m1)]
        p = 2 * stats.norm.sf(np.abs(z))
        found = bh_adjust(p) < ALPHA
        fdr.append(found[m1:].sum() / max(found.sum(), 1))
        power.append(found[:m1].mean())
    # reference: the step-up procedure written out
    p = rng.uniform(size=500) ** 2
    order = np.argsort(p)
    ref = np.minimum.accumulate((p[order] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    expected = np.empty_like(p)
    expected[order] = np.minimum(ref, 1)
    gap = float(np.max(np.abs(bh_adjust(p) - expected)))
    record(runs=300, tests_per_run=200, true_effects=20, fdr=np.mean(fdr), power=np.mean(power), max_gap_to_reference=gap)
    assert gap < 1e-12
    assert np.mean(fdr) < ALPHA * 180 / 200 + 0.01  # BH guarantees m0/m * alpha


@pytest.mark.parametrize("rho", [0.0, 0.3, 0.7])
def test_correlation_intervals_cover_the_true_value(rho: float, record) -> None:  # noqa: ANN001
    rng = np.random.default_rng(int(rho * 100) + 33)
    runs, n = 2_000, 80
    cover_p = cover_s = 0
    cov = [[1, rho], [rho, 1]]
    rho_s = 6 / np.pi * np.arcsin(rho / 2)  # Spearman's rho of a bivariate normal
    for _ in range(runs):
        x, y = rng.multivariate_normal([0, 0], cov, n).T
        lo, hi = fisher_ci(stats.pearsonr(x, y)[0], n)
        cover_p += lo <= rho <= hi
        lo, hi = fisher_ci(stats.spearmanr(x, y)[0], n, spearman=True)
        cover_s += lo <= rho_s <= hi
    record(rho=rho, n=n, runs=runs, pearson_coverage=cover_p / runs, spearman_coverage=cover_s / runs)
    for c in (cover_p / runs, cover_s / runs):
        assert c > 1 - upper(ALPHA, runs)


def test_mutual_information_is_zero_without_information(record) -> None:  # noqa: ANN001
    rng = np.random.default_rng(34)
    n = 2_000
    y = pd.Series(rng.integers(0, 2, n))
    constant = pd.Series(np.ones(n))
    order = y.sort_values().index  # a file sorted by the target
    noise = [mutual_information(pd.Series(rng.normal(size=n)), y, True, False) for _ in range(200)]
    signal = pd.Series(y * 1.0 + rng.normal(size=n))
    mi = mutual_information(signal, y, True, False)
    shuffled = mutual_information(signal.loc[order].reset_index(drop=True), y.loc[order].reset_index(drop=True),
                                  True, False)
    const = mutual_information(constant.loc[order].reset_index(drop=True), y.loc[order].reset_index(drop=True),
                               True, False)
    record(constant_column_bits=const, noise_mean_bits=np.mean(noise), noise_p99_bits=np.percentile(noise, 99),
           signal_bits=mi, signal_bits_sorted_file=shuffled)
    assert const == 0 and np.percentile(noise, 99) < 0.01 and abs(mi - shuffled) < 1e-12
    assert mi > 10 * np.percentile(noise, 99)  # one standard deviation between the classes: about 0.14 bit


def test_numeric_test_choice_keeps_power(record) -> None:  # noqa: ANN001
    """With an effect, the test chosen from the shapes keeps the power of the better-suited one."""
    rng = np.random.default_rng(35)
    runs, n = 400, 200
    hits = {"linear, normal (r .15)": 0, "monotone, lognormal": 0}
    for _ in range(runs):
        x = rng.normal(size=n)
        df = pd.DataFrame({"x": x, "y": 0.15 * x + np.sqrt(1 - 0.0225) * rng.normal(size=n),
                           "u": np.exp(x), "v": np.exp(0.25 * x + rng.normal(size=n))})
        pairs = see_corr(df, config=StoneConfig()).get("pairs").set_index(["col_a", "col_b"])
        hits["linear, normal (r .15)"] += pairs.loc[("x", "y"), "p_value"] < ALPHA
        hits["monotone, lognormal"] += pairs.loc[("u", "v"), "p_value"] < ALPHA
    power = {k: v / runs for k, v in hits.items()}
    record(runs=runs, n=n, **power)
    assert power["linear, normal (r .15)"] > 0.45 and power["monotone, lognormal"] > 0.85


def _ar1(rng: np.random.Generator, n: int, phi: float) -> np.ndarray:
    e = rng.normal(size=n)
    y = np.empty(n)
    y[0] = e[0] / np.sqrt(max(1 - phi**2, 1e-6))
    for i in range(1, n):
        y[i] = phi * y[i - 1] + e[i]
    return y


SERIES = {
    "white noise": lambda rng, n: rng.normal(size=n),
    "AR(1) 0.5": lambda rng, n: _ar1(rng, n, 0.5),
    "AR(1) 0.9": lambda rng, n: _ar1(rng, n, 0.9),
    "random walk": lambda rng, n: np.cumsum(rng.normal(size=n)),
}


@pytest.mark.parametrize("kind", list(SERIES) + ["panel of random walks"])
def test_time_ordered_rows_keep_the_type_i_error(kind: str, record) -> None:  # noqa: ANN001
    """Independent autocorrelated columns of a dated table, rows shuffled in the file: share of pairs rejected.

    Tested in levels as if the rows were independent, two AR(1) 0.9 series were rejected in about half the runs
    and two random walks in 86%."""
    rng = np.random.default_rng(zlib.crc32(kind.encode()))
    runs, rejected, total = 120, 0, 0
    for _ in range(runs):
        if kind.startswith("panel"):
            df = pd.DataFrame({"month": np.tile(pd.date_range("2000-01-01", periods=100, freq="MS"), 4),
                               "shop": np.repeat(list("abcd"), 100)})
            for c in ("u", "v", "w"):
                df[c] = np.concatenate([np.cumsum(rng.normal(size=100)) for _ in range(4)])
        else:
            df = pd.DataFrame({"day": pd.date_range("2020-01-01", periods=300)})
            for c in ("u", "v", "w"):
                df[c] = SERIES[kind](rng, 300)
        pairs = see_corr(df.sample(frac=1, random_state=0))["pairs"]
        num = pairs[pairs["kind"] == "num-num"]
        rejected += int((num["p_value"] < ALPHA).sum())
        total += len(num)
    rate = rejected / total
    record(series=kind, runs=runs, pairs=total, type_i_rate=rate)
    assert rate < upper(ALPHA, runs * 2)  # the 3 pairs of a table share columns: counted as 2 per table


def test_time_ordered_rows_keep_their_power(record) -> None:  # noqa: ANN001
    rng = np.random.default_rng(36)
    runs, hits = 150, {"AR(1) 0.5, innovations correlated 0.25": 0, "random walks sharing a common walk": 0}
    for _ in range(runs):
        e1 = rng.normal(size=300)
        e2 = 0.25 * e1 + np.sqrt(1 - 0.0625) * rng.normal(size=300)
        a, b = np.zeros(300), np.zeros(300)
        for i in range(1, 300):
            a[i], b[i] = 0.5 * a[i - 1] + e1[i], 0.5 * b[i - 1] + e2[i]
        common = np.cumsum(rng.normal(size=300))
        df = pd.DataFrame({"day": pd.date_range("2020-01-01", periods=300), "a": a, "b": b,
                           "x": common + 0.5 * rng.normal(size=300), "y": 0.8 * common + 0.3 * np.cumsum(rng.normal(size=300))})
        p = see_corr(df)["pairs"].set_index(["col_a", "col_b"])["p_value"]
        hits["AR(1) 0.5, innovations correlated 0.25"] += p[("a", "b")] < ALPHA
        hits["random walks sharing a common walk"] += p[("x", "y")] < ALPHA
    power = {k: v / runs for k, v in hits.items()}
    record(runs=runs, n=300, **power)
    assert min(power.values()) > 0.85


@pytest.mark.parametrize("effect", ["none", "linear r .15", "U shape 0.3 x2", "monotone exp"])
def test_target_link_sees_monotone_and_u_shaped_effects(effect: str, record) -> None:  # noqa: ANN001
    """Numeric feature, numeric target: Spearman and a Kruskal-Wallis across deciles, Bonferroni."""
    from elessar.correlations import target_associations

    rng = np.random.default_rng(zlib.crc32(effect.encode()))
    runs, n, hits = 400, 300, 0
    for _ in range(runs):
        x = rng.normal(size=n)
        y = {"none": 0 * x, "linear r .15": 0.15 * x, "U shape 0.3 x2": 0.3 * x**2,
             "monotone exp": np.exp(0.3 * x)}[effect] + rng.normal(size=n)
        t = target_associations(pd.DataFrame({"x": x, "y": y}), "y", {"x": "numeric_continuous", "y": "numeric_continuous"})
        hits += t["p_value"].iloc[0] < ALPHA
    rate = hits / runs
    record(effect=effect, runs=runs, n=n, rejected=rate)
    if effect == "none":
        assert rate < upper(ALPHA, runs)
    elif effect == "U shape 0.3 x2":
        assert rate > 0.9  # Spearman alone found 5%
