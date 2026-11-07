"""1.2.1: findings that contradicted the data or each other, found by profiling a table with planted problems."""
from __future__ import annotations

import numpy as np
import pandas as pd

from elessar import StoneConfig, see, see_relations
from elessar.correlations import product_dependencies
from elessar.quality import correlation_alerts, suspicious_values
from elessar.roles import birth_report


def _messages(v, column=None):
    return [a["message"] for a in v.alerts if column is None or a["column"] == column]


def test_a_column_filled_only_for_one_class_is_reported_as_leakage() -> None:
    rng = np.random.default_rng(0)
    n = 600
    y = rng.integers(0, 2, n)
    df = pd.DataFrame({"x": rng.normal(size=n) + y, "reason": np.where(y == 1, rng.choice(["a", "b"], n), None),
                       "y": y})
    v = see(df, target="y", plots=False, verbose=False)
    leaks = [a for a in v.alerts if a["column"] == "reason" and a.get("kind") == "leakage"]
    assert leaks and leaks[0]["severity"] == "high" and "filled" in leaks[0]["message"]


def test_a_very_strong_single_feature_asks_for_its_provenance() -> None:
    rng = np.random.default_rng(1)
    n = 1_000
    y = rng.integers(0, 2, n)
    df = pd.DataFrame({"strong": y * 3.4 + rng.normal(size=n), "weak": rng.normal(size=n) + 0.2 * y, "y": y})
    v = see(df, target="y", plots=False, verbose=False)
    auc = float(v.correlations["target"].set_index("feature").loc["strong", "auc"])
    assert 0.95 < auc < 0.99  # strong, not deterministic: below the leakage threshold
    medium = [a for a in v.alerts if a.get("kind") == "leakage" and a["severity"] == "medium" and a["column"] is None]
    assert medium and "strong" in medium[0]["message"] and "weak" not in medium[0]["message"]


def test_leak_tiers_follow_the_config() -> None:
    target = pd.DataFrame({"feature": ["a"], "feature_type": ["numeric_continuous"], "method": ["eta"],
                           "association": [0.8], "n": [100], "test": ["mann-whitney"], "p_value": [0.0],
                           "p_adjusted": [0.0], "significant": [True], "auc": [0.97], "mutual_info": [0.5],
                           "leakage_suspect": [False]})
    assert any(a.get("kind") == "leakage" for a in correlation_alerts({"target": target}, "y"))
    quiet = correlation_alerts({"target": target}, "y", StoneConfig(leakage_warning=0.99))
    assert not any(a.get("kind") == "leakage" for a in quiet)


def test_a_sentinel_is_neither_censoring_nor_a_negative_quantity() -> None:
    rng = np.random.default_rng(2)
    income = rng.normal(30_000, 5_000, 2_000).round(-2)
    income[:150] = -999
    v = see(pd.DataFrame({"annual_income": income, "x": rng.normal(size=2_000)}), plots=False, verbose=False)
    msgs = _messages(v, "annual_income")
    assert any("placeholder" in m and "-999" in m for m in msgs)
    assert not any("floored" in m or "censored" in m for m in msgs)
    assert not any("negative" in m for m in msgs)
    income[200] = -5  # a real negative value is still reported, alone
    assert suspicious_values(pd.Series(income, name="annual_income"), "numeric_continuous") == [
        "1 negative value in a quantity that should be >= 0"]


def test_default_dates_are_not_counted_again_as_births_on_1_january() -> None:
    rng = np.random.default_rng(3)
    births = pd.Timestamp("1950-01-01") + pd.to_timedelta(rng.integers(0, 365 * 50, 3_000), unit="D")
    births = births.where(rng.random(3_000) > 0.05, pd.Timestamp("1970-01-01"))
    df = pd.DataFrame({"birth_date": births, "visit": pd.Timestamp("2024-06-01")})
    checks = [m for _, m in birth_report(df, "birth_date", {"birth_date": "datetime", "visit": "datetime"})["checks"]]
    assert not any("1 January" in m for m in checks)


def test_dictionary_shows_examples_when_no_value_repeats() -> None:
    df = pd.DataFrame({"email": [f"user{i}@mail.example" for i in range(300)], "plan": ["a", "b", "c"] * 100})
    d = see(df, plots=False, verbose=False).dictionary().set_index("column")
    assert d.loc["email", "values"].startswith("300 distinct values, e.g. ") and "%" not in d.loc["email", "values"]
    assert "(33%)" in d.loc["plan", "values"]


def test_personal_data_columns_are_flagged() -> None:
    df = pd.DataFrame({"email": [f"user{i}@mail.example" for i in range(200)], "x": np.arange(200) % 7})
    assert any("personal data" in m for m in _messages(see(df, plots=False, verbose=False), "email"))


def test_exact_products_and_ratios_are_found_once_up_to_rounding() -> None:
    rng = np.random.default_rng(4)
    n = 500
    price = rng.lognormal(3, 0.5, n).round(2)
    qty = rng.integers(1, 40, n).astype(float)
    df = pd.DataFrame({"price": price, "qty": qty, "total": (price * qty).round(2), "noise": rng.normal(size=n),
                       "other": rng.lognormal(2, 1, n).round(1)})
    deps = product_dependencies(df)
    assert len(deps) == 1 and deps.iloc[0]["kind"] == "product" and deps.iloc[0]["column"] == "total"
    unit = pd.DataFrame({"total": (price * qty).round(2), "qty": qty, "noise": rng.normal(size=n)})
    unit["unit_price"] = (unit.total / unit.qty).round(4)
    found = product_dependencies(unit)
    assert len(found) == 1 and set(found.iloc[0]["columns"]) == {"total", "qty", "unit_price"}


def test_no_product_among_independent_columns() -> None:
    rng = np.random.default_rng(5)
    for _ in range(20):
        df = pd.DataFrame({f"c{i}": rng.integers(0, 12, 300).astype(float) for i in range(6)})
        df["d"] = rng.lognormal(2, 1, 300).round(2)
        assert product_dependencies(df).empty


def test_keys_are_suggested_by_name_when_chance_cannot_tell() -> None:
    rng = np.random.default_rng(6)
    customers = pd.DataFrame({"customer_id": np.arange(1, 501)})  # fills its range: the shift test is blind
    orders = pd.DataFrame({"order_id": np.arange(1, 3_001), "customer_id": rng.integers(1, 501, 3_000)})
    orders.loc[:9, "customer_id"] = 900  # 10 orphans
    r = see_relations({"customers": customers, "orders": orders}, verbose=False)
    k = r.keys.set_index(["child", "child_column"]).loc[("orders", "customer_id")]
    assert k["parent"] == "customers" and k["origin"] == "suggested" and k["n_orphans"] == 10
    assert any("same column name" in a["message"] for a in r.alerts)
    assert "1 key" in r.render(100) and "1 keys" not in r.render(100)


def test_a_tables_own_identifier_is_not_a_key_to_another_table() -> None:
    n = 2_000
    client = pd.DataFrame({"client_id": np.arange(1, n + 1) * 3})
    disp = pd.DataFrame({"disp_id": np.arange(1, n + 1) * 3, "client_id": np.arange(1, n + 1) * 3})  # ids in step
    r = see_relations({"client": client, "disp": disp}, verbose=False)
    pairs = set(zip(r.keys["child"] + "." + r.keys["child_column"], r.keys["parent"] + "." + r.keys["parent_column"]))
    assert pairs == {("disp.client_id", "client.client_id")}
    details = pd.DataFrame({"customer_id": np.arange(1, 301)})  # an extension table keeps its identifier as a key
    customers = pd.DataFrame({"customer_id": np.arange(1, 401)})
    r = see_relations({"customer_details": details, "customers": customers}, verbose=False)
    assert list(r.keys["parent"]) == ["customers"]
