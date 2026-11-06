from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from elessar import see_relations
from elessar import relations as rel


def _bank(rng: np.random.Generator) -> dict[str, pd.DataFrame]:
    account = pd.DataFrame({"account_id": rng.permutation(np.arange(1, 3_001) * 3)[:1_000],
                            "opened": pd.Timestamp("1995-01-01") + pd.to_timedelta(rng.integers(0, 700, 1_000), unit="D")})
    chosen = rng.choice(account.account_id, 300, replace=False)
    loan = pd.DataFrame({"loan_id": np.arange(300), "account_id": chosen,
                         "granted": account.set_index("account_id").loc[chosen, "opened"].to_numpy() + pd.Timedelta(days=30)})
    trans = pd.DataFrame({"account_id": rng.choice(account.account_id, 20_000), "amount": rng.lognormal(6, 1, 20_000)})
    return {"account": account, "loan": loan, "trans": trans}


def test_declared_keys_are_checked_exactly() -> None:
    t = _bank(np.random.default_rng(1))
    t["loan"].loc[:4, "account_id"] = 999_999  # 5 orphans
    t["loan"].loc[5:6, "account_id"] = np.nan  # 2 without key
    t["loan"].loc[10, "granted"] = pd.Timestamp("1990-01-01")  # before its account
    r = see_relations(t, keys=["loan.account_id -> account.account_id", ("trans", "account_id", "account", "account_id")],
                      dates={"account": "opened", "loan": "granted"}, suggest=False, verbose=False)
    loan = r.keys.set_index("child").loc["loan"]
    assert loan.n_orphans == 5 and loan.n_child_missing == 2 and loan.cardinality == "1:1" and loan.parent_unique
    assert loan.n_child_before_parent == 1 and loan.child_before_parent_examples == [10]
    assert loan.n_parents_without_child == 1_000 - 293
    trans = r.keys.set_index("child").loc["trans"]
    assert trans.cardinality == "1:n" and trans.n_orphans == 0 and pd.isna(trans.n_child_before_parent)
    assert {a["severity"] for a in r.alerts if a["column"] == "loan.account_id"} == {"high"}
    assert "loan.account_id -> account.account_id" in r.render(200)


def test_a_parent_that_is_not_a_key() -> None:
    parent = pd.DataFrame({"id": [1, 1, 2, 3]})
    child = pd.DataFrame({"pid": [1, 2, 3, 3]})
    r = see_relations({"p": parent, "c": child}, keys=["c.pid -> p.id"], suggest=False, verbose=False)
    assert not r.keys.loc[0, "parent_unique"] and any("not a key" in a["message"] for a in r.alerts)


def test_streamed_child_equals_in_memory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    t = _bank(np.random.default_rng(2))
    t["trans"].loc[[3, 15_000], "account_id"] = -1
    path = tmp_path / "trans.csv"
    t["trans"].to_csv(path, index=False)
    original = rel._chunks
    monkeypatch.setattr(rel, "_chunks", lambda *a, **k: original(*a, **{**k, "rows": 1_234}))  # many blocks
    on_disk = see_relations({**t, "trans": str(path)}, keys=["trans.account_id -> account.account_id"],
                            suggest=False, verbose=False).keys.iloc[0]
    in_memory = see_relations(t, keys=["trans.account_id -> account.account_id"], suggest=False, verbose=False).keys.iloc[0]
    for k in ("n_orphans", "n_child_rows", "n_parents_without_child", "children_min", "children_median", "children_max"):
        assert on_disk[k] == in_memory[k], k
    assert on_disk.n_orphans == 2 and on_disk.orphan_examples == [-1]


def test_suggested_keys() -> None:
    t = _bank(np.random.default_rng(3))
    r = see_relations(t, verbose=False)
    found = {(k.child, k.child_column, k.parent, k.parent_column) for k in r.keys.itertuples()}
    assert ("loan", "account_id", "account", "account_id") in found and ("trans", "account_id", "account", "account_id") in found
    assert all(k.origin == "suggested" for k in r.keys.itertuples())
    with pytest.raises(KeyError):
        see_relations(t, keys=["loan.account_id -> nope.id"], verbose=False)
    with pytest.raises(ValueError):
        see_relations(t, keys=["loan account_id"], verbose=False)
