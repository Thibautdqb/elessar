"""Checks decided against chance, not against a threshold tuned on example datasets.

Each check below compares what it sees with what chance gives on the same rows, and is measured here on
tables whose truth is known: tables where nothing is wrong (the share of tables flagged must stay near the
significance level) and tables where the defect is planted (the share found).
"""

from __future__ import annotations

import zlib

import numpy as np
import pandas as pd
import pytest

from elessar.quality import duplicate_rows

from .sim import lower, upper

pytestmark = pytest.mark.validation
ALPHA = 0.05


def _flagged(df: pd.DataFrame, ids: list[str]) -> bool:
    c = duplicate_rows(df, ids).get("chance_ignoring_ids")
    return c is not None and c["p_value"] < ALPHA and c["observed"] > c["expected"]


NULL_TABLES = {
    # few columns with few values: rows coincide often, no record is entered twice
    "type x date, 900 rows": lambda rng: pd.DataFrame({"type": rng.choice(["a", "b", "c"], 900, p=[.74, .16, .1]),
                                                        "date": rng.integers(0, 600, 900)}),
    "3 categories, 2,000 rows": lambda rng: pd.DataFrame({f"c{i}": rng.integers(0, 8, 2_000) for i in range(3)}),
    "rounded amount x category, 1,500 rows": lambda rng: pd.DataFrame(
        {"amount": rng.lognormal(5, 1, 1_500).round(-2), "k": rng.choice(list("xyz"), 1_500)}),
}


@pytest.mark.parametrize("table", sorted(NULL_TABLES))
def test_chance_duplicates_keep_their_type_i_error(table: str, record) -> None:  # noqa: ANN001
    """Independent columns, no record entered twice: the share of tables flagged must stay near 5%."""
    rng = np.random.default_rng(zlib.crc32(table.encode()))
    runs = 200
    flagged = 0
    for _ in range(runs):
        df = NULL_TABLES[table](rng)
        df.insert(0, "id", np.arange(len(df)))
        flagged += _flagged(df, ["id"])
    rate = flagged / runs
    record(table=table, runs=runs, flagged=rate)
    assert rate < upper(ALPHA, runs)  # before: every one of these tables raised a high or medium alert


@pytest.mark.parametrize("copies", [3, 10])
def test_records_entered_twice_are_found(copies: int, record) -> None:  # noqa: ANN001
    """A few records copied under a new id into tables of 900 rows with 4 varied columns."""
    rng = np.random.default_rng(copies)
    runs = 200
    found = 0
    for _ in range(runs):
        df = pd.DataFrame({"amount": rng.lognormal(8, 1, 900).round(), "duration": rng.choice([12, 24, 36, 48, 60], 900),
                           "date": rng.integers(0, 2_000, 900), "status": rng.choice(list("ABCD"), 900)})
        df = pd.concat([df, df.sample(copies, random_state=int(rng.integers(1 << 30)))], ignore_index=True)
        df.insert(0, "id", rng.permutation(len(df)))
        found += _flagged(df, ["id"])
    rate = found / runs
    record(copies=copies, runs=runs, found=rate)
    assert rate > lower(0.99, runs)


def test_dependent_columns_are_not_flagged(record) -> None:  # noqa: ANN001
    """Columns that depend on each other coincide more often than independent ones: the card type depends on
    the issue period (gold cards only in the last year), no record is entered twice. Shuffling each column on
    its own flagged 97% of these tables; the draws keep each column's strongest link (Chow-Liu tree)."""
    rng = np.random.default_rng(77)
    runs = 200
    flagged = 0
    for _ in range(runs):
        date = rng.integers(0, 600, 900)
        late = date >= 480
        kind = np.where(late, rng.choice(["classic", "gold"], 900), rng.choice(["classic", "junior"], 900, p=[.8, .2]))
        df = pd.DataFrame({"id": np.arange(900), "type": kind, "date": date})
        flagged += _flagged(df, ["id"])
    rate = flagged / runs
    record(runs=runs, flagged=rate)
    assert rate < upper(ALPHA, runs)


def test_links_outside_the_tree_are_not_flagged(record) -> None:  # noqa: ANN001
    """A hidden segment drives three columns at once: a dependence no tree of pairs holds exactly."""
    rng = np.random.default_rng(78)
    runs = 200
    flagged = 0
    for _ in range(runs):
        z = rng.integers(0, 4, 1_500)
        df = pd.DataFrame({"id": np.arange(1_500), "a": z * 3 + rng.integers(0, 3, 1_500),
                           "b": z * 2 + rng.integers(0, 2, 1_500), "c": z * 4 + rng.integers(0, 4, 1_500)})
        flagged += _flagged(df, ["id"])
    rate = flagged / runs
    record(runs=runs, flagged=rate)
    assert rate < upper(ALPHA, runs)


def test_few_copies_among_many_coincidences_limit(record) -> None:  # noqa: ANN001
    """Known limit, measured and not claimed: 3 copies among rows that already coincide often by chance (a
    category, a day of 50 and a rounded amount) cannot be told from those coincidences."""
    rng = np.random.default_rng(79)
    runs = 200
    found = 0
    for _ in range(runs):
        df = pd.DataFrame({"k": rng.choice(list("xyz"), 900), "d": rng.integers(0, 50, 900),
                           "amt": rng.lognormal(6, 1, 900).round(-1)})
        df = pd.concat([df, df.sample(3, random_state=int(rng.integers(1 << 30)))], ignore_index=True)
        df.insert(0, "id", np.arange(len(df)))
        found += _flagged(df, ["id"])
    rate = found / runs
    record(runs=runs, found=rate)
    assert 0 <= rate <= 1  # reported in VALIDATION.md
