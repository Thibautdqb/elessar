from __future__ import annotations

import uuid

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


@pytest.fixture
def messy_df(rng: np.random.Generator) -> pd.DataFrame:
    """Small DataFrame mixing every semantic type and the usual data problems."""
    n = 300
    income = rng.lognormal(10, 0.5, n)
    df = pd.DataFrame({
        "id": np.arange(n),
        "uuid": [str(uuid.UUID(int=int(rng.integers(2**62)) << 32)) for _ in range(n)],
        "age": rng.normal(40, 10, n).round(),
        "income": income,
        "income_copy": income * 2,
        "visits": rng.poisson(4, n),
        "city": rng.choice(["Paris", "Lyon", "Nice", "paris "], n, p=[0.5, 0.3, 0.15, 0.05]),
        "level": rng.choice(["low", "medium", "high"], n),
        "flag": rng.choice(["yes", "no"], n),
        "when": pd.Timestamp("2024-01-01") + pd.to_timedelta(rng.integers(0, 365, n), unit="D"),
        "when_str": (pd.Timestamp("2024-01-01") + pd.to_timedelta(rng.integers(0, 365, n), unit="D")).strftime("%Y-%m-%d"),
        "amount_txt": [f"{v:,.2f}".replace(",", " ").replace(".", ",") for v in rng.gamma(2, 500, n)],
        "constant": "x",
        "all_nan": np.nan,
        "listy": [[1, 2] if i % 2 else {"a": i} for i in range(n)],
        "mixed": [i if i % 3 else f"s{i}" for i in range(n)],
        "target": rng.integers(0, 2, n),
    })
    df.loc[rng.choice(n, 90, replace=False), "income"] = np.nan
    df.loc[rng.choice(n, 30, replace=False), "age"] = np.nan
    df.loc[rng.choice(n, 20, replace=False), "level"] = None
    df.loc[0, "age"] = -5
    df.loc[1, "income_copy"] = np.inf
    return pd.concat([df, df.iloc[:5]], ignore_index=True)


def edge_frames() -> dict[str, pd.DataFrame]:
    """DataFrames that must never crash elessar."""
    big_card = pd.DataFrame({"s": [f"value_{i}" for i in range(5_000)], "x": np.arange(5_000) % 7})
    return {
        "empty_0x0": pd.DataFrame(),
        "no_rows": pd.DataFrame({"a": pd.Series([], dtype=float), "b": pd.Series([], dtype=object)}),
        "single_row": pd.DataFrame({"a": [1.0], "b": ["x"], "c": [pd.Timestamp("2020-01-01")]}),
        "all_nan": pd.DataFrame({"a": [np.nan] * 5, "b": [None] * 5}),
        "inf": pd.DataFrame({"a": [1.0, np.inf, -np.inf, 2.0, np.nan, 3.0] * 5}),
        "only_inf": pd.DataFrame({"a": [np.inf, -np.inf, np.inf]}),
        "mixed": pd.DataFrame({"a": [1, "a", 2.5, None, True, [1], {"k": 1}] * 3}),
        "unhashable": pd.DataFrame({"a": [[1, 2], [3], [1, 2]] * 4, "b": [{"x": 1}, {"y": 2}, {"x": 1}] * 4}),
        "huge_cardinality": big_card,
        "dup_columns": pd.DataFrame([[1, 2, 3], [4, 5, 6]], columns=["a", "a", 7]),
        "nullable": pd.DataFrame({
            "i": pd.array([1, None, 3, 4] * 5, dtype="Int64"),
            "b": pd.array([True, None, False, True] * 5, dtype="boolean"),
            "s": pd.array(["x", None, "y", "z"] * 5, dtype="string"),
        }),
        "tz_and_timedelta": pd.DataFrame({
            "t": pd.date_range("2024-01-01", periods=20, freq="h", tz="Europe/Paris"),
            "d": pd.to_timedelta(np.arange(20), unit="min"),
        }),
        "category_dtype": pd.DataFrame({"c": pd.Categorical(["a", "b", "a", "c"] * 5)}),
        "complex": pd.DataFrame({"z": np.array([1 + 2j, 3 - 1j] * 5)}),
        "two_rows_constant_num": pd.DataFrame({"a": [5, 5], "b": [1.5, 2.5]}),
    }


@pytest.fixture(params=list(edge_frames()))
def edge_df(request: pytest.FixtureRequest) -> pd.DataFrame:
    return edge_frames()[request.param]
