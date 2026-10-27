from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from elessar import see_stats
from elessar.stats import categorical_stats, column_stats, common_stats, datetime_stats, numeric_stats, text_stats


def test_common_stats_counts() -> None:
    s = pd.Series(["a", "b", "a", None, "a"])
    st = common_stats(s)
    assert st["count"] == 4 and st["n_missing"] == 1
    assert st["pct_missing"] == pytest.approx(20.0)
    assert st["n_unique"] == 2
    assert st["modes"] == ["a"] and st["mode_freq"] == 3
    assert st["top_values"][0] == {"value": "a", "count": 3, "pct": 75.0}


def test_numeric_stats_values(rng: np.random.Generator) -> None:
    x = pd.Series(np.r_[rng.normal(10, 2, 1_000), [100.0, np.inf, np.nan]])
    st = numeric_stats(x)
    finite = x[np.isfinite(x)]
    assert st["n_inf"] == 1
    assert st["mean"] == pytest.approx(finite.mean())
    assert st["median"] == pytest.approx(finite.median())
    assert st["p25"] < st["median"] < st["p75"]
    assert st["n_outliers_iqr"] >= 1 and st["n_outliers_z"] >= 1 and st["n_outliers_mz"] >= 1
    assert st["ci95_mean_low"] < st["mean"] < st["ci95_mean_high"]
    assert st["geometric_mean"] is not None and st["harmonic_mean"] is not None
    assert st["n_modes_kde"] == 1


def test_numeric_stats_shape_flags(rng: np.random.Generator) -> None:
    bimodal = pd.Series(np.r_[rng.normal(0, 1, 2_000), rng.normal(8, 1, 2_000)])
    st = numeric_stats(bimodal)
    assert st["n_modes_kde"] == 2
    assert st["bimodality_coefficient"] > 0.555
    heaped = pd.Series(np.r_[rng.integers(0, 100, 500), np.full(500, 50)])
    assert numeric_stats(heaped)["heaping_index"] > 2
    assert numeric_stats(pd.Series(np.arange(10.0)))["monotonic_increasing"]
    assert numeric_stats(pd.Series([1.25, 2.5, 3.75]))["n_decimals"] == 2
    assert numeric_stats(pd.Series([-1.0, 2.0]))["geometric_mean"] is None


def test_numeric_stats_degenerate() -> None:
    assert numeric_stats(pd.Series([np.inf, -np.inf])) == {"n_inf": 2}
    one = numeric_stats(pd.Series([3.0]))
    assert one["mean"] == 3.0 and one["ci95_mean_low"] is None


def test_categorical_stats() -> None:
    s = pd.Series(["Paris"] * 50 + ["paris "] * 5 + ["Lyon"] * 44 + ["Brest"])
    st = categorical_stats(s)
    assert st["cardinality"] == 4
    assert 0 < st["normalized_entropy"] < 1
    assert st["imbalance_ratio"] == 50
    assert st["rare_categories"] == []  # 1% is not < 1%
    assert st["n_inconsistent_groups"] == 1
    assert sorted(st["inconsistent_labels"][0]) == ["Paris", "paris "]


def test_datetime_stats() -> None:
    idx = pd.date_range("2024-01-01", periods=100, freq="D")
    s = pd.Series(idx.append(pd.DatetimeIndex(["2024-01-01", "2099-01-01"])))
    st = datetime_stats(s)
    assert st["min"] == "2024-01-01" and st["n_duplicated_timestamps"] == 1
    assert st["n_future"] == 1
    assert st["largest_gaps"][0]["end"] == "2099-01-01"
    assert "by_weekday" in st and "by_hour" not in st
    assert pd.Series(pd.date_range("2024-01-01", periods=10, freq="D")).pipe(datetime_stats)["inferred_freq"] == "D"


def test_text_stats() -> None:
    s = pd.Series(["Hello world, data is GREAT!", "data again 123", "", "   "])
    st = text_stats(s)
    assert st["n_empty"] == 1 and st["n_whitespace_only"] == 1
    assert st["top_tokens"][0]["token"] == "data"
    assert st["pct_digits"] > 0 and st["pct_uppercase"] > 0


def test_column_stats_dispatch() -> None:
    assert "top_domains" in column_stats(pd.Series([f"u{i}@x.com" for i in range(30)]))
    assert "python_kinds" in column_stats(pd.Series([[1], {"a": 1}, "x"] * 5))
    assert "n_unparsable" in column_stats(pd.Series(["1,5", "2,5", "3,25"] * 10))
    assert "pct_true" in column_stats(pd.Series(["yes", "no"] * 10))


def test_see_stats_frame(messy_df: pd.DataFrame) -> None:
    out = see_stats(messy_df)
    assert list(out.index) == list(messy_df.columns)
    assert out.loc["income", "pct_missing"] > 0
    assert out.loc["income_copy", "n_inf"] == 2  # the inf row is also duplicated
    assert "top_values" not in out.columns


def test_significance_extras(rng: np.random.Generator) -> None:
    x = pd.Series(rng.normal(50, 10, 2_000))
    st = numeric_stats(x)
    assert st["ci95_median_low"] < st["median"] < st["ci95_median_high"]
    assert abs(st["autocorr_lag1"]) < 0.1 and st["autocorr_lag1_p"] > 0.001
    walk = numeric_stats(pd.Series(np.cumsum(rng.normal(size=2_000))))
    assert walk["autocorr_lag1"] > 0.9 and walk["autocorr_lag1_p"] < 1e-10
    b = column_stats(pd.Series(["yes"] * 30 + ["no"] * 70))
    assert b["ci95_pct_true_low"] < 30 < b["ci95_pct_true_high"]
    assert categorical_stats(pd.Series(["a", "b", "c"] * 100))["uniform_chi2_p"] == pytest.approx(1.0)
    assert categorical_stats(pd.Series(["a"] * 280 + ["b"] * 20))["uniform_chi2_p"] < 1e-10


def test_heaping_ignores_mixed_precision_and_zeros(rng: np.random.Generator) -> None:
    fares = pd.Series(np.r_[rng.choice([7.25, 8.05, 26.0, 13.0], 300), rng.gamma(2, 10, 300).round(4)])
    assert numeric_stats(fares)["heaping_index"] is None
    gains = pd.Series(np.r_[np.zeros(900), rng.integers(100, 10_000, 100)])
    assert numeric_stats(gains)["heaping_index"] < 2
    ages = pd.Series(np.r_[rng.integers(18, 80, 500), [0.5, 30.5, 2.5]])
    assert numeric_stats(ages)["heaping_index"] is not None


def test_datetime_trend() -> None:
    days = np.arange(365)
    rng = np.random.default_rng(0)
    growing = pd.Series(pd.Timestamp("2024-01-01") + pd.to_timedelta(rng.choice(days, 3_000, p=days / days.sum()), unit="D"))
    st = datetime_stats(growing)
    assert st["counts_resolution"] == "week"
    assert st["counts_trend_tau"] > 0.5 and st["counts_trend_p"] < 1e-6


def test_caps_floors_and_zero_inflation(rng: np.random.Generator) -> None:
    values = np.minimum(rng.lognormal(12, 0.5, 5_000).round(-2), 500_001)
    st = numeric_stats(pd.Series(values))
    assert st["capped_max"] and st["pct_at_max"] > 1 and not st["capped_min"]
    assert numeric_stats(pd.Series(rng.normal(size=2_000)))["capped_max"] is False
    rounded = pd.Series(np.r_[rng.integers(100, 5_000, 2_000) * 100.0, np.full(50, 500_001.0)])
    assert numeric_stats(rounded)["heaping_index"] < 2  # precision is hundreds; the cap is not heaping


def test_zero_inflation_alert(rng: np.random.Generator) -> None:
    from elessar import see_column

    gains = pd.Series(np.r_[np.zeros(900), rng.integers(100, 10_000, 100)], name="gain")
    assert any("zero-inflated" in a["message"] for a in see_column(gains)["alerts"])
