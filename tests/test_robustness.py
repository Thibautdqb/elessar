"""elessar must run on any DataFrame: any column names, dtypes, index and size."""

from __future__ import annotations

import json
import uuid
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import elessar
from elessar.correlations import linear_dependencies, variance_inflation

HOSTILE_NAMES = [
    "", " ", None, np.nan, 0, -1, 3.5, True, ("a", "b"), "a", "a", "a.1", "count", "values", "index", "level_0",
    "shape", "T", "col/with/slash", "../../etc", "new\nline", "tab\tname", "pipe|name", "quote'\"", "émoji 🚀",
    "x" * 300, "%s {0} {}", "NaN", "_s", "target", "p_value", "C:\\path", "*", ".", "-", "1e5",
    pd.Timestamp("2020-01-01"), "$x$", "price ($)", "\\frac{1}{2}",
]


def _column(rng: np.random.Generator, kind: str, n: int) -> object:
    makers = {
        "float": lambda: np.where(rng.random(n) < 0.2, np.nan, rng.normal(size=n)),
        "int": lambda: rng.integers(-5, 50, n),
        "nullable_int": lambda: pd.array(np.where(rng.random(n) < 0.3, None, rng.integers(0, 9, n)), dtype="Int64"),
        "bool": lambda: rng.random(n) < 0.3,
        "nullable_bool": lambda: pd.array(np.where(rng.random(n) < 0.3, None, rng.random(n) < 0.5), dtype="boolean"),
        "cat": lambda: pd.Series(rng.choice(["a", "$5", "$x^2$", None], n), dtype=object),
        "category": lambda: pd.Categorical(rng.choice(["x", "y"], n)),
        "string": lambda: pd.array(rng.choice(["foo", "bar", None, ""], n), dtype="string"),
        "text": lambda: [" ".join(rng.choice(["alpha", "beta", "gamma", "delta"], 6)) for _ in range(n)],
        "date": lambda: pd.Timestamp("2020-01-01") + pd.to_timedelta(rng.integers(0, 1000, n), unit="D"),
        "tzdate": lambda: pd.date_range("2021-01-01", periods=n, freq="h", tz="UTC"),
        "datestr": lambda: (pd.Timestamp("2020-01-01") + pd.to_timedelta(rng.integers(0, 99, n), unit="D")).strftime("%d/%m/%Y"),
        "numstr": lambda: [f"{v:.2f}%" for v in rng.uniform(0, 100, n)],
        "timedelta": lambda: pd.to_timedelta(rng.integers(0, 100, n), unit="s"),
        "mixed": lambda: [[1, "a", 2.5, None, True][i % 5] for i in range(n)],
        "containers": lambda: [[1, 2] if i % 2 else {"k": i} for i in range(n)],
        "const": lambda: ["same"] * n,
        "allnan": lambda: [np.nan] * n,
        "inf": lambda: np.where(rng.random(n) < 0.1, np.inf, rng.normal(size=n)),
        "uuid": lambda: [str(uuid.UUID(int=int(rng.integers(2**62)))) for _ in range(n)],
        "complex": lambda: np.array([1 + 2j] * n),
        "bytes": lambda: [b"\x00\xff"] * n,
        "period": lambda: pd.period_range("2020-01", periods=n, freq="M"),
        "interval": lambda: pd.arrays.IntervalArray.from_breaks(np.arange(n + 1)),
        "sparse": lambda: pd.arrays.SparseArray(np.where(rng.random(n) < 0.9, 0, 1.0)),
        "objnum": lambda: pd.Series(rng.normal(size=n)).astype(object),
        "decimal": lambda: [Decimal("1.5") if i % 2 else Decimal("2.25") for i in range(n)],
        "huge": lambda: rng.normal(size=n) * 1e300,
        "tiny": lambda: rng.normal(size=n) * 1e-300,
        "bigint": lambda: [2**70 + i for i in range(n)],
    }
    return makers[kind]()


KINDS = sorted(
    ["float", "int", "nullable_int", "bool", "nullable_bool", "cat", "category", "string", "text", "date", "tzdate",
     "datestr", "numstr", "timedelta", "mixed", "containers", "const", "allnan", "inf", "uuid", "complex", "bytes",
     "period", "interval", "sparse", "objnum", "decimal", "huge", "tiny", "bigint"]
)


def random_frame(seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = int(rng.choice([0, 1, 2, 3, 7, 40, 200]))
    k = int(rng.integers(1, 8))
    names = [HOSTILE_NAMES[i] for i in rng.choice(len(HOSTILE_NAMES), k)]
    cols = [pd.Series(_column(rng, str(kind), n)).reset_index(drop=True) for kind in rng.choice(KINDS, k)]
    df = pd.concat(cols, axis=1)
    df.columns = pd.Index(names, dtype=object, tupleize_cols=False)
    index_kind = int(rng.integers(0, 5))
    if n and index_kind == 1:
        df.index = rng.integers(0, 3, n)  # duplicated labels
    elif n and index_kind == 2:
        df.index = pd.date_range("2020", periods=n, freq="D")
    elif n and index_kind == 3:
        df.index = pd.MultiIndex.from_arrays([np.arange(n) % 2, np.arange(n)])
    elif n and index_kind == 4:
        df.index = [f"r{i}" for i in range(n)]
    return df


@pytest.mark.parametrize("seed", range(40))
def test_any_dataframe(seed: int, tmp_path: Path) -> None:
    df = random_frame(seed)
    target = df.columns[seed % df.shape[1]] if seed % 3 == 0 else None
    report = elessar.see(df, target=target, verbose=False, plots=seed % 4 == 0, out_dir=tmp_path / "plots")
    assert report.render(width=60)
    json.dumps(report.to_dict())
    report.to_markdown(tmp_path / "md" / "report.md")
    report.to_csv(tmp_path / "csv" / "summary.csv")
    failures = [a["message"] for a in report.alerts if "failed" in a["message"] or "plot skipped" in a["message"]]
    assert failures == []


@pytest.mark.parametrize("label", [("a", "b"), 0, 3.5, np.nan, pd.Timestamp("2020-01-01"), "x|y", ""])
def test_target_given_by_original_label(label: object) -> None:
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"other": rng.normal(size=60), "g": rng.choice(["u", "v"], 60)})
    df.columns = pd.Index(["other", label], dtype=object, tupleize_cols=False)
    report = elessar.see(df, target=label, verbose=False, plots=False)
    assert report.target in report.columns
    assert not report.correlations["target"].empty


def test_none_label_is_reachable_as_string() -> None:
    df = pd.DataFrame({"g": list("abab") * 10, "x": np.arange(40.0)})
    df.columns = pd.Index([None, "x"], dtype=object)  # target=None means "no target"
    assert elessar.see(df, target="None", verbose=False, plots=False).target == "None"


def test_unknown_target_raises() -> None:
    with pytest.raises(KeyError):
        elessar.see(pd.DataFrame({"a": [1, 2]}), target=("no", "pe"), verbose=False, plots=False)


def test_specific_regressions(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    n = 120
    df = pd.DataFrame({
        "sparse": pd.arrays.SparseArray(np.where(rng.random(n) < 0.9, 0, 1.0)),
        "huge": rng.normal(size=n) * 1e300,
        "bytes": [b"\x00\xff", b"ok"] * (n // 2),
        "x": rng.normal(size=n),
        "y": rng.normal(size=n),
    }, index=rng.integers(0, 5, n))  # duplicated index
    report = elessar.see(df, target="x", verbose=False, out_dir=tmp_path)
    assert not [a for a in report.alerts if "failed" in a["message"] or "plot skipped" in a["message"]]
    assert report.columns["huge"]["stats"]["n_modes_kde"] is not None
    assert report.columns["sparse"]["stats"]["count"] == n


def test_linear_dependency_equations() -> None:
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=200), rng.normal(size=200)
    df = pd.DataFrame({"x-1": a, "y+2": b, "z": 2 * a - 3 * b + 5, "w": rng.normal(size=200)})
    df["half w"] = df["w"] / 2
    eqs = linear_dependencies(df, variance_inflation(df))["equation"].tolist()
    assert "z = 2·`x-1` - 3·`y+2` + 5" in eqs
    assert "w = 2·half w" in eqs
    assert linear_dependencies(df[["w", "x-1"]], variance_inflation(df[["w", "x-1"]])).empty
