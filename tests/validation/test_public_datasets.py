"""Known datasets: what elessar finds against what is known of them (seaborn-data, downloaded once and cached).

Counts are checked against pandas computed here, never against numbers copied from elessar; the other facts
are the documented content of the datasets.
"""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path

import pandas as pd
import pytest

from elessar import see

pytestmark = pytest.mark.validation
SOURCE = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/"
CACHE = Path(os.environ.get("ELESSAR_VALIDATION_DATA", Path.home() / ".cache" / "elessar-validation"))


def _path(name: str) -> Path:
    local = CACHE / f"seaborn_{name}.csv"
    if not local.exists():
        try:
            local.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(f"{SOURCE}{name}.csv", local)
        except OSError as exc:
            pytest.skip(f"cannot download {name}: {exc}")
    return local


@pytest.mark.parametrize("name", ["titanic", "iris", "tips", "penguins", "mpg", "flights", "healthexp", "diamonds",
                                  "planets", "taxis"])
def test_counts_match_pandas(name: str, record) -> None:  # noqa: ANN001
    path = _path(name)
    truth = pd.read_csv(path)
    o = see(path, plots=False, verbose=False).overview
    expected = {"rows": len(truth), "columns": truth.shape[1], "rows_with_missing": int(truth.isna().any(axis=1).sum()),
                "duplicate_rows": int(truth.duplicated().sum())}
    found = {"rows": o["n_rows"], "columns": o["n_columns"], "rows_with_missing": o["n_rows_with_missing"],
             "duplicate_rows": o["n_duplicate_rows"]}
    record(dataset=name, source=f"{SOURCE}{name}.csv", expected=expected, found=found, types=o["semantic_types"])
    assert found == expected


def test_known_facts(record) -> None:  # noqa: ANN001
    facts = {}
    flights = see(_path("flights"), plots=False, verbose=False).timeseries
    main = flights["analyses"][flights["main"]]
    facts["flights: monthly airline passengers 1949-1960"] = {
        "kind": flights["kind"], "time": flights["time_column"], "frequency": flights["frequency"],
        "period": main["period"], "seasonality": main["seasonality"], "trend": main["trend"]}
    assert (flights["kind"], flights["frequency"], main["period"], main["trend"]) == ("single series", "month", 12,
                                                                                       "increasing")
    health = see(_path("healthexp"), plots=False, verbose=False).timeseries
    facts["healthexp: one series per country"] = {"kind": health["kind"], "entity": health.get("entity")}
    assert health["kind"] == "panel" and health["entity"] == "Country"
    titanic = see(_path("titanic"), target="survived", plots=False, verbose=False)
    tgt = titanic.correlations["target"].set_index("feature")
    leaks = sorted(tgt.index[tgt["leakage_suspect"].astype(bool)])
    facts["titanic: 'alive' is the target written as yes/no"] = {"leakage_suspects": leaks,
                                                                 "kept": titanic.selection["keep"]}
    # sex, who (man / woman / child) and adult_male carry the same information: one of them is enough, and the
    # trees pick one of a redundant group, so sex alone may stay borderline
    assert "alive" in leaks and "alive" not in titanic.selection["keep"]
    assert {"sex", "who", "adult_male"} & set(titanic.selection["keep"])
    for name, target, strongest in (("iris", "species", {"petal_length", "petal_width"}),
                                    ("tips", "tip", {"total_bill"}), ("diamonds", "price", {"carat", "x"})):
        v = see(_path(name), target=target, plots=False, verbose=False)
        top = list(v.correlations["target"]["feature"][:2])
        facts[f"{name}: strongest link with {target}"] = {"top_two": top, "kept": v.selection["keep"]}
        assert strongest & set(top) and strongest <= set(v.selection["keep"])
    record(**facts)
