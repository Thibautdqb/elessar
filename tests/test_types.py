from __future__ import annotations

import uuid

import numpy as np
import pandas as pd
import pytest

from elessar import see_types
from elessar.types import infer_type, ordinal_order, parse_number, parse_numeric

rng = np.random.default_rng(1)
N = 200
WORDS = "data quality matters because every model learns from what we feed it today".split()

CASES: dict[str, tuple[pd.Series, str]] = {
    "empty": (pd.Series([np.nan] * 10), "empty"),
    "constant": (pd.Series(["a"] * 10), "constant"),
    "bool": (pd.Series([True, False] * 10), "boolean"),
    "bool01": (pd.Series([0, 1, 1] * 10), "boolean"),
    "yes_no": (pd.Series(["yes", "No", "YES "] * 10), "boolean"),
    "oui_non": (pd.Series(["oui", "non"] * 10), "boolean"),
    "true_false_str": (pd.Series(["true", "false"] * 10), "boolean"),
    "id_int": (pd.Series(np.arange(100), name="row_number"), "id"),
    "id_name": (pd.Series(rng.permutation(50) * 37, name="customer_id"), "id"),
    "uuid": (pd.Series([str(uuid.uuid4()) for _ in range(50)]), "id"),
    "continuous": (pd.Series(rng.normal(size=N)), "numeric_continuous"),
    "discrete": (pd.Series(rng.poisson(20, N)), "numeric_discrete"),
    "categorical": (pd.Series(rng.choice(["red", "green", "blue", "black"], N)), "categorical"),
    "ordinal_labels": (pd.Series(rng.choice(["low", "medium", "high"], N)), "ordinal-candidate"),
    "ordinal_ints": (pd.Series(rng.integers(1, 6, N)), "ordinal-candidate"),
    "datetime": (pd.Series(pd.date_range("2020-01-01", periods=N, freq="D")), "datetime"),
    "datetime_string": (pd.Series(pd.date_range("2020-01-01", periods=N, freq="D").strftime("%d/%m/%Y")), "datetime_string"),
    "numeric_string_fr": (pd.Series([f"{v:,.1f}".replace(",", " ").replace(".", ",") for v in rng.gamma(2, 900, N)]), "numeric_string"),
    "numeric_string_pct": (pd.Series([f"{v:.1f}%" for v in rng.uniform(0, 100, N)]), "numeric_string"),
    "numeric_string_usd": (pd.Series([f"${v:.2f}" for v in rng.uniform(0, 100, N)]), "numeric_string"),
    "text": (pd.Series([" ".join(rng.choice(WORDS, 8)) for _ in range(N)]), "text"),
    "email": (pd.Series([f"user{i}@example.com" for i in range(N)]), "email"),
    "url": (pd.Series([f"https://example.com/page/{i}?q=1" for i in range(N)]), "url"),
    "phone": (pd.Series([f"+33 6 {rng.integers(10, 99)} {rng.integers(10, 99)} {rng.integers(10, 99)} 00" for _ in range(N)]), "phone"),
    "postal_hint": (pd.Series([f"{rng.integers(1000, 99999):05d}" for _ in range(N)], name="zip_code"), "postal_code"),
    "postal_uk": (pd.Series(rng.choice(["SW1A 1AA", "EC1A 1BB", "W1A 0AX", "M1 1AE"], N)), "postal_code"),
    "ipv4": (pd.Series([f"192.168.{rng.integers(256)}.{rng.integers(256)}" for _ in range(N)]), "ip_address"),
    "json_str": (pd.Series([f'{{"a": {i}, "b": [1, 2]}}' for i in range(N)]), "json_like"),
    "json_obj": (pd.Series([{"a": i} if i % 2 else [i] for i in range(N)]), "json_like"),
    "mixed": (pd.Series([i if i % 2 else f"x{i}" for i in range(N)]), "mixed"),
    "ordered_category": (pd.Series(pd.Categorical(["s", "m", "l"] * 10, categories=["s", "m", "l"], ordered=True)), "ordinal-candidate"),
}


@pytest.mark.parametrize("case", list(CASES))
def test_semantic_type(case: str) -> None:
    s, expected = CASES[case]
    info = infer_type(s)
    assert info["semantic_type"] == expected, info["reason"]
    assert 0 <= info["confidence"] <= 1
    assert info["pandas_dtype"] == str(s.dtype)


def test_suggestions() -> None:
    num = infer_type(CASES["numeric_string_fr"][0])
    assert num["suggestions"][0]["flag"] == "should-be-numeric"
    dates = infer_type(CASES["datetime_string"][0])
    assert dates["suggestions"][0]["flag"] == "should-be-datetime"
    cat = infer_type(CASES["categorical"][0])
    assert any(sug["flag"] == "should-be-category" for sug in cat["suggestions"])
    yes = infer_type(CASES["yes_no"][0])
    assert "map(" in yes["suggestions"][0]["code"]


@pytest.mark.parametrize(
    ("text", "value"),
    [("1 234,5", 1234.5), ("12%", 12.0), ("$3.2", 3.2), ("1,234.56", 1234.56), ("1.234,56", 1234.56),
     ("3,5", 3.5), ("-7", -7.0), ("1e3", 1000.0), ("(15)", -15.0), ("12 €", 12.0)],
)
def test_parse_number(text: str, value: float) -> None:
    assert parse_number(text) == pytest.approx(value)


def test_parse_number_rejects_text() -> None:
    assert np.isnan(parse_number("hello"))
    assert np.isnan(parse_number("12 34 56"))
    assert parse_numeric(pd.Series(["1,5", None, "x"])).isna().tolist() == [False, True, True]


def test_ordinal_order() -> None:
    assert ordinal_order(["High", "low", "Medium"]) == ["low", "Medium", "High"]
    assert ordinal_order(["cat", "dog"]) is None


def test_see_types_frame(messy_df: pd.DataFrame) -> None:
    out = see_types(messy_df)
    assert list(out.index) == list(messy_df.columns)
    assert {"semantic_type", "pandas_dtype", "confidence", "reason", "flags", "suggestion"} <= set(out.columns)
    assert out.loc["all_nan", "semantic_type"] == "empty"
    assert out.loc["amount_txt", "semantic_type"] == "numeric_string"
    assert out.loc["listy", "semantic_type"] == "json_like"
    assert out.loc["mixed", "semantic_type"] == "mixed"


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        (rng.poisson(0.4, 300), "numeric_discrete"),  # counts (Parch-like) are not scales
        (rng.poisson(4, 300), "numeric_discrete"),
        (rng.choice([1, 2, 3], 300, p=[0.25, 0.2, 0.55]), "ordinal-candidate"),  # passenger class
        (rng.integers(2_000, 20_000, 3_000), "numeric_continuous"),  # prices in whole dollars
        ((rng.normal(4_000, 600, 300) / 25).round() * 25, "numeric_continuous"),  # body mass rounded to 25 g
        (rng.integers(20, 70, 300), "numeric_discrete"),  # hours per week
    ],
)
def test_integer_semantics(values: np.ndarray, expected: str) -> None:
    assert infer_type(pd.Series(values))["semantic_type"] == expected


@pytest.mark.parametrize(
    "labels",
    [["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
     ["Thur", "Fri", "Sat", "Sun"], ["Fair", "Good", "Very Good", "Premium", "Ideal"], ["Q1", "Q2", "Q3", "Q4"]],
)
def test_ordered_vocabularies(labels: list[str]) -> None:
    info = infer_type(pd.Series(rng.choice(labels, 400)))
    assert info["semantic_type"] == "ordinal-candidate"
    assert info["order"] == labels


def test_short_codes_are_categorical() -> None:
    tickets = pd.Series([f"A/{i % 700}" if i % 3 else str(300_000 + i % 650) for i in range(900)])
    assert infer_type(tickets)["semantic_type"] == "categorical"


def test_numbers_with_units() -> None:
    from elessar.types import parse_numeric, split_unit

    assert split_unit("90 min") == (90.0, "min") and split_unit("2 Seasons") == (2.0, "season")
    assert split_unit("1,234 km") == (1234.0, "km") and split_unit("100 m") == (100.0, "m")
    for code in ("5A", "3rd", "4T/2010", "Paris"):
        assert split_unit(code)[1] is None
    weight = infer_type(pd.Series([f"{w:.1f} kg" for w in rng.normal(70, 10, 200)], name="w"))
    assert weight["semantic_type"] == "numeric_string" and weight["units"] == {"kg": 200}
    mixed = pd.Series(rng.choice(["90 min", "101 min", "1 Season", "3 Seasons"], 200, p=[0.35, 0.35, 0.15, 0.15]), name="duration")
    info = infer_type(mixed)
    assert info["semantic_type"] == "numeric_string" and set(info["units"]) == {"min", "season"}
    assert info["suggestions"][0]["flag"] == "mixed-units"
    values = parse_numeric(mixed)
    assert set(values.dropna().unique()) <= {90.0, 101.0}  # only the main unit is kept: units are not comparable
    df = pd.DataFrame({"duration": mixed})
    exec(info["suggestions"][0]["code"])  # the suggested fix runs as-is
    assert set(df["duration_unit"].str.strip()) <= {"min", "Season", "Seasons"}


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([", ".join(rng.choice(["Comedy", "Drama", "Action", "Horror", "Docs"], rng.integers(1, 4), replace=False))
          for _ in range(300)], "multi_label"),
        (["|".join(rng.choice(list("ABCDEFG"), rng.integers(1, 4), replace=False)) for _ in range(300)], "multi_label"),
        (list(rng.choice(["Paris, France", "Lyon, France", "Austin, USA", "Boston, USA"], 300)), "categorical"),
        (list(rng.choice(["Washed / Wet", "Natural / Dry", "Pulped natural / honey", "Other"], 300)), "categorical"),
        ([f"Company {i % 40} Co., Ltd" if i % 3 else f"Firm {i % 25}" for i in range(300)], "categorical"),
        ([f"//cdn.example.com/images/pic{i}.jpg" for i in range(300)], "url"),
        (list(rng.choice([f"Event {i} Men's Basketball" for i in range(80)], 3_000)), "categorical"),
    ],
)
def test_lists_urls_and_labels(values: list[str], expected: str) -> None:
    assert infer_type(pd.Series(values))["semantic_type"] == expected


def test_identifier_name_hints() -> None:
    assert infer_type(pd.Series(rng.integers(0, 12, 500), name="key"))["semantic_type"] != "categorical"
    volcano = pd.Series(rng.choice(rng.integers(200_000, 400_000, 900), 10_000), name="volcano_number")
    info = infer_type(volcano)
    assert info["semantic_type"] == "categorical" and "identifier" in info["reason"]
    assert infer_type(pd.Series(rng.integers(0, 9, 500), name="number"))["semantic_type"] != "categorical"


def test_account_numbers_are_identifiers_not_amounts() -> None:
    from elessar import see_types

    rng = np.random.default_rng(3)
    df = pd.DataFrame({"account": rng.choice(rng.integers(10_000_000, 99_999_999, 300), 2000).astype(float),
                       "amount": rng.lognormal(5, 1, 2000)})
    t = see_types(df)["semantic_type"]
    assert t["account"] in ("id", "categorical") and t["amount"] == "numeric_continuous"


# ----------------------------------------------------------------------------------- integers that are codes
def _ints(name: str, values: list[int], n: int = 2000, p: list[float] | None = None) -> pd.Series:
    return pd.Series(np.random.default_rng(7).choice(values, n, p=p), name=name)


@pytest.mark.parametrize("name, values", [
    ("region", [11, 24, 27, 28, 32, 44, 52, 53, 75, 76, 84, 93, 94]),
    ("store", list(range(1, 46))),
    ("Dept", [1, 2, 3, 5, 7, 8, 13, 16, 21, 33, 40, 52, 67, 72, 80, 91, 95, 99]),
    ("payment_type", [1, 2, 3, 4]),
    ("mcc", [5411, 5812, 5541, 5912, 4111]),
    ("zip", [75001, 75011, 69003, 13008, 31000]),
    ("sexe", [1, 2]),
    ("cluster", [0, 1, 2, 3, 4]),
    ("marital_status", [1, 2, 3, 4, 5]),
    ("codeRegion", [11, 24, 27]),
    ("merchant", [5411, 5812, 5541, 5912, 4111, 5311, 5999, 7011]),  # no name hint: fixed-width codes
])
def test_integer_codes_are_categories(name: str, values: list[int]) -> None:
    info = infer_type(_ints(name, values))
    assert info["semantic_type"] == "categorical" and info["codes"] in ("name", "values")
    assert info["certainty"] == "probable"  # what an integer means cannot be proven from its values
    assert info["path"][-1] == "integers that label categories (codes)?"
    assert any(s["flag"] == "integer-codes" for s in info["suggestions"])


@pytest.mark.parametrize("name, values", [
    ("SibSp", [0, 1, 2, 3, 4, 5, 8]),
    ("cylinders", [3, 4, 5, 6, 8]),
    ("year", list(range(1990, 2021))),
    ("olympics", [1896, 1900, 1904, 1908, 1912, 1920, 1924]),  # years, not codes
    ("price", [199, 249, 299, 349]),
    ("tiers", [100, 250, 500, 750, 1000]),  # rounded amounts
    ("store_count", list(range(1, 46))),  # the name says it counts
    ("category_one_defects", [0, 1, 2, 3, 5, 8]),
    ("pf_identity_sex", [0, 5, 10]),  # a graded score, the code word in the middle of the name
    ("code", list(range(40, 120))),  # a test called Code: "code" alone says nothing
])
def test_quantities_stay_numeric(name: str, values: list[int]) -> None:
    assert infer_type(_ints(name, values))["semantic_type"] != "categorical"


def test_column_types_override_the_tree() -> None:
    from elessar import StoneConfig

    s = _ints("store", list(range(1, 46)))
    info = infer_type(s, StoneConfig(column_types={"store": "numeric_discrete"}))
    assert info["semantic_type"] == "numeric_discrete" and info["certainty"] == "certain"
    assert info["path"] == ["set by the user (StoneConfig.column_types)?"]
    with pytest.raises(ValueError, match="choose one of"):
        infer_type(s, StoneConfig(column_types={"store": "banana"}))
