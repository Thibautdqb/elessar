from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from elessar import StoneConfig, see_dist
from elessar.distributions import KS_NOTE, fit_distributions, normality_tests, shape_description


@pytest.mark.parametrize(
    ("name", "make"),
    [
        ("norm", lambda r: r.normal(5, 2, 3_000)),
        ("lognorm", lambda r: r.lognormal(1, 0.7, 3_000)),
        ("expon", lambda r: r.exponential(3, 3_000)),
        ("uniform", lambda r: r.uniform(-2, 7, 3_000)),
        ("poisson", lambda r: r.poisson(4, 3_000)),
        ("nbinom", lambda r: r.negative_binomial(2, 0.2, 3_000)),
    ],
)
def test_best_fit_recovers_generator(name: str, make, rng: np.random.Generator) -> None:
    res = fit_distributions(make(rng))
    assert res["best"] == name
    assert len(res["fits"]) == 3
    best = res["fits"][0]
    assert {"aic", "bic", "ks_stat", "ks_pvalue", "params", "loglik"} <= set(best)
    assert res["fits"][0]["aic"] <= res["fits"][1]["aic"]
    assert res["ks_note"] == KS_NOTE
    assert "best fit" in res["description"]


def test_fit_handles_nonpositive_support(rng: np.random.Generator) -> None:
    x = rng.gamma(2, 1, 2_000) - 1.0  # has negative values
    res = fit_distributions(x)
    shifted = [f for f in res["fits"] if f["distribution"] in ("gamma", "lognorm", "weibull_min")]
    assert all("shifted" in f["note"] for f in shifted)


def test_fit_degenerate() -> None:
    assert fit_distributions(np.ones(50))["fits"] == []
    assert fit_distributions(np.array([1.0, 2.0]))["best"] is None
    assert fit_distributions(np.array([np.inf, np.nan, 1.0]))["fits"] == []


def test_candidate_selection_and_failures(rng: np.random.Generator) -> None:
    cfg = StoneConfig(candidates_continuous=("norm", "not_a_dist"))
    res = fit_distributions(rng.normal(size=500), config=cfg)
    assert res["best"] == "norm"
    assert "not_a_dist" in res["failed"]


def test_normality(rng: np.random.Generator) -> None:
    assert normality_tests(rng.normal(size=1_000))["verdict"] == "normal"
    res = normality_tests(rng.exponential(size=1_000))
    assert res["verdict"] == "not normal" and res["n_reject"] >= 3
    assert normality_tests(np.arange(5.0))["verdict"] == "insufficient data"
    assert normality_tests(np.ones(30))["verdict"] == "constant"
    big = normality_tests(rng.normal(size=30_000), StoneConfig(fit_sample_size=6_000))
    assert big["n"] == 30_000 and "shapiro_p" in big


def test_shape_description(rng: np.random.Generator) -> None:
    assert shape_description(rng.normal(size=2_000)) == "roughly symmetric"
    assert shape_description(rng.lognormal(0, 1, 2_000)).startswith("heavily right-skewed")
    assert shape_description(-rng.lognormal(0, 1, 2_000)).startswith("heavily left-skewed")


def test_see_dist_frame(messy_df: pd.DataFrame) -> None:
    out = see_dist(messy_df)
    assert "income" in out.index and "city" not in out.index
    assert out.loc["income", "best_fit"] == "lognorm"
    assert out.loc["visits", "best_fit"] in ("poisson", "binom", "nbinom")
    assert out.loc["amount_txt", "normality"] == "not normal"
