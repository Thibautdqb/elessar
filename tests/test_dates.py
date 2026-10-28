from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from elessar import parse_dates, see_types
from elessar.dates import detect

POSITIVE = {
    "yymmdd int": (pd.Series([930101, 951231, 970514, 930227] * 5), "1993-01-01", "compact YYMMDD"),
    "yymmdd text with '-'": (pd.Series(["930101", "951231", "-", "970514"] * 5), "1993-01-01", "compact YYMMDD"),
    "yyyymmdd": (pd.Series([20200105, 20211231, 20190301] * 5), "2020-01-05", "compact YYYYMMDD"),
    "ddmmyyyy": (pd.Series(["05012020", "31122021", "15032019"] * 5), "2020-01-05", "compact DDMMYYYY"),
    "yyyymm": (pd.Series([202001, 202112, 201903] * 5), "2020-01-01", "compact YYYYMM"),
    "birth number": (pd.Series(["706213", "450204", "406009", "561201"] * 5), "1970-12-13", "birth numbers"),
    "epoch s": (pd.Series(np.arange(1_600_000_000, 1_600_000_000 + 86_400 * 20, 86_400), name="created_at"),
                "2020-09-13", "Unix timestamps in s"),
    "epoch ms": (pd.Series(np.arange(1_600_000_000_000, 1_600_000_000_000 + 86_400_000 * 20, 86_400_000)),
                 "2020-09-13", "Unix timestamps in ms"),
    "excel serial": (pd.Series([43831, 43832, 44000, 44500], name="order_date"), "2020-01-01", "Excel"),
    "dd/mm/yyyy": (pd.Series(["05/01/2020", "31/12/2021", "15/03/2019"] * 5), "2020-01-05", "DD/MM/YYYY"),
    "mm/dd/yyyy": (pd.Series(["01/05/2020", "12/31/2021", "03/15/2019"] * 5), "2020-01-05", "MM/DD/YYYY"),
    "dd.mm.yy": (pd.Series(["05.01.20", "31.12.21", "15.03.19"] * 5), "2020-01-05", "DD.MM.YY"),
    "french": (pd.Series(["5 janvier 2020", "31 décembre 2021", "15 mars 2019"] * 5), "2020-01-05", "month names"),
    "german": (pd.Series(["5. März 2020", "31. Dezember 2021", "1. Mai 2019"] * 5), "2020-03-05", "month names"),
    "czech": (pd.Series(["5. ledna 2020", "12. března 2021", "1. prosince 2019"] * 5), "2020-01-05", "month names"),
    "english": (pd.Series(["Jan 5, 2020", "December 31, 2021", "15 March 2019"] * 5), "2020-01-05", "dates"),
    "iso with zones": (pd.Series(["2020-01-05T10:00:00Z", "2021-12-31T23:59:59+02:00", "2019-03-15 08:00"] * 5),
                       "2020-01-05", "dates"),
    "quarters": (pd.Series(["2020Q1", "2020Q2", "Q3 2021", "2021-Q4"] * 5), "2020-01-01", "quarters"),
    "iso weeks": (pd.Series(["2020-W05", "2020-W06", "2021W52"] * 5), "2020-01-27", "ISO weeks"),
    "year-month": (pd.Series(["2020-01", "2021-12", "2019-03"] * 5), "2020-01-01", "dates"),
}


@pytest.mark.parametrize("case", sorted(POSITIVE))
def test_every_format(case: str) -> None:
    s, first, label = POSITIVE[case]
    fmt = detect(s, s.name)
    assert fmt is not None and label in fmt.label, case
    parsed = parse_dates(s, fmt)
    assert parsed.dropna().iloc[0].normalize() == pd.Timestamp(first), case
    assert parsed.notna().sum() == s.isin(["-"]).__invert__().sum()
    assert see_types(s.to_frame("x" if s.name is None else s.name))["semantic_type"].iloc[0] == "datetime_string"


@pytest.mark.parametrize("values", [
    [123456, 999999, 450000, 120000] * 5,  # amounts
    list(range(100000, 100040)),  # identifiers
    ["12.5", "3.20", "1.5", "100.25"] * 5,  # prices
    ["T1", "Q3", "A12"] * 5,  # codes
    ["75001", "69002", "13008"] * 5,  # postal codes
    list(np.random.default_rng(0).normal(size=50)),
    list(np.random.default_rng(1).integers(1_000_000_000, 4_000_000_000, 50)),  # big numbers, no date in the name
])
def test_not_dates(values: list) -> None:
    assert detect(pd.Series(values)) is None


def test_ambiguity_century_and_code() -> None:
    fmt = detect(pd.Series(["01/05/2020", "02/03/2021", "11/12/2019"] * 5))
    assert fmt.format == "%d/%m/%Y" and fmt.ambiguous == ["%m/%d/%Y"]
    births = parse_dates(pd.Series(["110304", "450101", "991231", "050607"] * 3, name="birth"))
    assert births.max() <= pd.Timestamp.now() and births.min().year == 1945 and births.max().year == 2011
    df = pd.DataFrame({"date": [930101, 951231, 970514, 930227] * 5})
    info = see_types(df).iloc[0]
    assert "format='%y%m%d'" not in info["suggestion"] or "parse_dates" in info["suggestion"]
    assert parse_dates(pd.Series(["2020-01-05", None, "garbage"])).isna().tolist() == [False, True, True]
