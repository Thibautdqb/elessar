from __future__ import annotations

import numpy as np
import pandas as pd

from elessar import see_quality
from elessar.quality import SEVERITIES, duplicate_columns, duplicate_rows, missing_patterns, placeholder_values


def test_duplicates() -> None:
    df = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"], "c": [1, 1, 2], "d": [[1], [1], [2]]})
    dr = duplicate_rows(df)
    assert dr["count"] == 1 and dr["example_indices"] == [1]
    assert duplicate_columns(df) == [["a", "c"]]


def test_missing_patterns() -> None:
    df = pd.DataFrame({"a": [np.nan, np.nan, 1, 2, np.nan], "b": [np.nan, np.nan, 1, 2, np.nan], "c": range(5)})
    mp = missing_patterns(df)
    assert mp["co_missing"][0]["jaccard"] == 1.0
    assert mp["n_rows_many_missing"] == 3
    assert mp["top_patterns"][0] == {"columns": ["a", "b"], "rows": 3}


def test_placeholders() -> None:
    s = pd.Series(["a", "NA", "b", " null ", "?", "c"])
    found = placeholder_values(s, "categorical")
    assert set(found) == {"'NA'", "' null '", "'?'"}
    num = pd.Series([1.0, 2.0, 3.0, -999.0, 4.0])
    assert placeholder_values(num, "numeric_continuous") == {"-999": 1}


def test_see_quality(messy_df: pd.DataFrame) -> None:
    q = see_quality(messy_df)
    assert q["duplicate_rows"]["count"] == 5
    assert "all_nan" in q["empty_columns"] and "constant" in q["constant_columns"]
    assert "income" in q["high_missing_columns"]
    assert "mixed" in q["mixed_type_columns"] and "listy" not in q["mixed_type_columns"]
    assert "city" in q["whitespace_columns"]
    assert any("negative ages" in m for m in q["suspicious_values"]["age"])
    assert {"id", "uuid"} <= set(q["id_columns"])
    assert any({p["col_a"], p["col_b"]} == {"income", "income_copy"} for p in q["redundant_pairs"])
    assert any(o["suggested_dtype"] == "category" for o in q["dtype_optimizations"])
    assert q["potential_saving_bytes"] > 0
    sev = [a["severity"] for a in q["alerts"]]
    assert set(sev) <= set(SEVERITIES)
    assert sev == sorted(sev, key=SEVERITIES.index)
    assert all({"severity", "column", "message", "kind"} == set(a) for a in q["alerts"])


def test_suspicious_dates() -> None:
    df = pd.DataFrame({"birth_date": pd.to_datetime(["1990-01-01", "2199-01-01", "1970-01-01", "1850-05-05"])})
    msgs = " ".join(see_quality(df)["suspicious_values"]["birth_date"])
    assert "1970-01-01" in msgs and "before 1900" in msgs  # births in the future: see test_roles


def test_correlation_alerts(rng: np.random.Generator) -> None:
    n = 600
    a = rng.normal(size=n)
    group = rng.choice(["x", "y"], n)
    df = pd.DataFrame({
        "a": a, "b": rng.normal(size=n), "a_plus_b": None, "group": group,
        "partly": pd.Series(rng.normal(size=n)).where(~((group == "y") & (rng.random(n) < 0.7))),
    })
    df["a_plus_b"] = df["a"] + df["b"]
    alerts = see_quality(df)["alerts"]
    msgs = {(al["column"], al["message"].split(" ")[0]) for al in alerts}
    assert any("a_plus_b = 1·a + 1·b" in al["message"] for al in alerts)  # equation recovered
    assert ("partly", "missingness") in msgs
    from elessar.quality import correlation_alerts

    target_df = pd.DataFrame({"t": rng.integers(0, 2, n)})
    target_df["copy"] = target_df["t"].map({0: "no", 1: "yes"})
    from elessar import see_corr

    leak = correlation_alerts(see_corr(target_df, target="t"), "t")
    assert any(al["severity"] == "high" and "leakage" in al["message"] for al in leak)


def test_alert_grouping_and_duplicate_dedup(rng: np.random.Generator) -> None:
    from elessar import see

    df = pd.DataFrame({f"skew{i}": rng.lognormal(0, 1.5, 400) for i in range(4)})
    df["copy"] = df["skew0"]
    alerts = see(df, plots=False, verbose=False).alerts
    grouped = [a for a in alerts if a["kind"] == "skewed"]
    assert len(grouped) == 1 and grouped[0]["column"] is None and grouped[0]["message"].startswith("5 columns")
    about_copy = [a["message"] for a in alerts if a["column"] in ("copy", "skew0")]
    assert sum("identical" in m for m in about_copy) == 1
    assert not any("linear dependency" in m or "redundant" in m for m in about_copy)
