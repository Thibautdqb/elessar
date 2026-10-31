"""Factor analysis: number of factors on simulated structures, and the known structure of public datasets."""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from elessar import StoneConfig
from elessar.factor import _solution, factor_analysis

from .sim import factor_data, lower

pytestmark = pytest.mark.validation
CFG = StoneConfig()
SOURCE_COMMIT = "1dcc2bf5f955cc1224a3e1307256e1fe86b68dae"
RDATASETS = f"https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/{SOURCE_COMMIT}/csv/"
CACHE = Path(os.environ.get("ELESSAR_VALIDATION_DATA", Path.home() / ".cache" / "elessar-validation"))


def _dataset(path: str) -> pd.DataFrame:
    local = CACHE / (SOURCE_COMMIT[:12] + "_" + path.replace("/", "_"))
    if not local.exists():
        try:
            local.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(RDATASETS + path, local)
        except OSError as exc:
            pytest.skip(f"cannot download {path}: {exc}")
    return pd.read_csv(local)


def test_number_of_factors_on_simulated_structures(record) -> None:  # noqa: ANN001
    rng = np.random.default_rng(231)
    rows = []
    for _ in range(300):
        k, per, n = int(rng.integers(1, 5)), int(rng.integers(3, 7)), int(rng.choice([150, 300, 800]))
        junk = int(rng.integers(1, 4)) if rng.random() < 0.5 else 0
        df = factor_data(rng, k, per, n, factor_r=float(rng.uniform(0, 0.3)), junk=junk)
        first = _solution(df, None)
        fa = factor_analysis(df, {c: "numeric_continuous" for c in df}, CFG)
        items = set(fa.get("items") or [])
        true_items = {f"item{i}" for i in range(k * per)}
        rows.append({"k": k, "parallel": first["k_parallel"], "kaiser": first["k_kaiser"],
                     "variance 60%": first["k_variance"], "scree": first["k_scree"],
                     "elessar": fa.get("n_factors") or 0, "junk_removed": not (items - true_items),
                     "true_items_kept": len(items & true_items) / len(true_items)})
    t = pd.DataFrame(rows)
    right = {c: float((t[c] == t["k"]).mean()) for c in ("parallel", "kaiser", "variance 60%", "scree", "elessar")}
    over = {c: float((t[c] > t["k"]).mean()) for c in right}
    record(structures=len(t), right_number=right, overestimated=over,
           junk_items_removed=float(t["junk_removed"].mean()), true_items_kept=float(t["true_items_kept"].mean()),
           design="1-4 factors of 3-6 items, loadings .45-.80, factor correlation 0-.3, n 150/300/800, "
                  "0-3 items loading on nothing")
    assert right["elessar"] > lower(0.98, len(t)) and over["elessar"] < 0.02


@pytest.mark.parametrize("name, path, columns, k", [
    ("Holzinger-Swineford, 9 tests", "sem/HS.data.csv",
     ["visual", "cubes", "flags", "paragrap", "sentence", "wordm", "addition", "counting", "straight"], 3),
    ("Holzinger-Swineford, 24 tests", "sem/HS.data.csv",
     ["visual", "cubes", "paper", "flags", "general", "paragrap", "sentence", "wordc", "wordm", "addition", "code",
      "counting", "straight", "wordr", "numberr", "figurer", "object", "numberf", "figurew", "deduct", "numeric",
      "problemr", "series", "arithmet"], 4),
    ("Big Five questionnaire (BFI, 25 items)", "psych/bfi.csv", [f"{t}{i}" for t in "ACENO" for i in range(1, 6)], 5),
])
def test_reference_datasets_keep_their_known_structure(name: str, path: str, columns: list[str], k: int,
                                                        record) -> None:  # noqa: ANN001
    df = _dataset(path)[columns]
    fa = factor_analysis(df, {c: "numeric_discrete" for c in df}, CFG)
    groups = {f: sorted(items) for f, items in fa["factors"].items()}
    record(dataset=name, source=RDATASETS + path, rows=len(df), items=len(columns), expected_factors=k,
           factors=fa["n_factors"], kmo=fa["kmo"], items_kept=len(fa["items"]), removed=[r["item"] for r in fa["removed"]],
           groups=groups, alphas={f: round(r["reliability"], 3) for f, r in fa["reliability"].items()})
    assert fa["n_factors"] == k
    if path == "psych/bfi.csv":  # each factor gathers the items of one trait
        traits = [{c[0] for c in items} for items in fa["factors"].values()]
        assert all(len(t) == 1 for t in traits) and {t.pop() for t in traits} == set("ACENO")
