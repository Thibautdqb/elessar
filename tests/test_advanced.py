from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from elessar import StoneConfig, see
from elessar.multivariate import kmo, multivariate, pca_analysis
from elessar.outliers import column_outliers
from elessar.preparation import BKTree, composite, edit_distance, group_categories, magnitude_mix, misspellings
from elessar.selection import feature_selection
from elessar.timeseries import adf_test, analyze_series, detect

CFG = StoneConfig()


# ------------------------------------------------------------------------------------------- multivariate
def _latent(n: int = 600, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    f1, f2 = rng.normal(size=n), rng.normal(size=n)
    cols = {f"a{i}": f1 + 0.3 * rng.normal(size=n) for i in range(3)}
    cols.update({f"b{i}": f2 + 0.3 * rng.normal(size=n) for i in range(3)})
    return pd.DataFrame(cols)


def test_pca_and_factors_find_two_latent_factors() -> None:
    df = _latent()
    mv = multivariate(df, {c: "numeric_continuous" for c in df}, None, CFG)
    pca, fa = mv["pca"], mv["factor"]
    assert pca["verdict"] == "recommended" and pca["recommended"] == 2 and pca["k_parallel"] == 2
    assert pca["kmo"] > 0.6 and pca["bartlett_p"] < 1e-10
    assert fa["verdict"] == "recommended" and fa["n_factors"] == 2
    groups = sorted(sorted(g) for g in fa["defined_by"].values())
    assert groups == [["a0", "a1", "a2"], ["b0", "b1", "b2"]]
    assert sorted(sorted(g["columns"]) for g in mv["groups"]) == groups


def test_pca_not_useful_on_independent_columns() -> None:
    rng = np.random.default_rng(1)
    df = pd.DataFrame(rng.normal(size=(500, 5)), columns=list("abcde"))
    assert pca_analysis(df, CFG)["verdict"] == "not useful"
    overall, per_var = kmo(np.eye(4))
    assert overall == 0.0 and len(per_var) == 4


def test_multivariate_outliers_find_planted_rows() -> None:
    df = _latent(800)
    df.loc[[5, 50], ["a0", "b0"]] = [[8, -8], [-8, 8]]  # consistent with no factor
    mo = multivariate(df, {c: "numeric_continuous" for c in df}, None, CFG)["outliers"]
    assert mo["verdict"] == "computed" and mo["reliable"]
    assert {5, 50} <= set(mo["top"]["row"].head(5))


# ------------------------------------------------------------------------------------------- outliers
def test_outlier_methods_and_actions() -> None:
    rng = np.random.default_rng(2)
    x = pd.Series(np.r_[rng.normal(50, 5, 500), [400.0]])
    r = column_outliers(x, CFG)
    assert r["counts"]["3 sigma"] >= 1 and r["n_consensus"] >= 1
    assert r["points"].iloc[0]["value"] == 400 and r["points"].iloc[0]["n_methods"] == 3
    assert r["action"] == "remove"  # one value 70 robust sd away: an error
    tail = pd.Series(rng.lognormal(0, 1, 1000))
    assert column_outliers(tail, CFG)["action"] == "reassign"
    assert column_outliers(pd.Series(rng.normal(size=300)), CFG)["n_consensus"] <= 3


# ------------------------------------------------------------------------------------------- time series
def _daily(n: int = 1000, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = pd.date_range("2020-01-01", periods=n, freq="D")
    y = 10 + 0.01 * np.arange(n) + 3 * np.sin(2 * np.pi * np.arange(n) / 7) + rng.normal(0, 0.5, n)
    return pd.DataFrame({"day": t, "sales": y, "noise": rng.normal(size=n)})


def test_detects_series_seasonality_and_folds() -> None:
    df = _daily()
    types = {"day": "datetime", "sales": "numeric_continuous", "noise": "numeric_continuous"}
    info = detect(df, types)
    assert info["kind"] == "single series" and info["frequency"] == "day" and info["regularity"] > 0.99
    v = see(df, target="sales", plots=False, verbose=False)
    ts = v.timeseries
    main = ts["analyses"]["sales"]
    assert ts["main"] == "sales" and main["period"] == 7 and main["seasonality"] == "strong"
    assert main["trend"] == "increasing"
    assert ts["folds"] and all(f["n_train"] > 0 for f in ts["folds"])
    assert any(s["step"] == "Seasonality" for s in ts["treatment"])
    assert not any(a["message"].startswith("rows are ordered") for a in v.alerts)


def test_panel_event_log_and_year_month() -> None:
    rng = np.random.default_rng(4)
    t = pd.date_range("2021-01-01", periods=60, freq="MS")
    panel = pd.DataFrame({"month": np.tile(t, 3), "store": np.repeat(["a", "b", "c"], 60), "v": rng.normal(size=180)})
    info = detect(panel, {"month": "datetime", "store": "categorical", "v": "numeric_continuous"})
    assert info["kind"] == "panel" and info["entity"] == "store"
    events = pd.DataFrame({"at": pd.Timestamp("2022-01-01") + pd.to_timedelta(rng.integers(0, 10**7, 400), unit="s")})
    assert detect(events, {"at": "datetime"})["kind"] == "event log"
    ym = pd.DataFrame({"year": np.repeat(np.arange(2000, 2010), 12), "month": np.tile(["Jan", "Feb", "Mar", "Apr", "May",
                       "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], 10), "v": rng.normal(size=120)})
    info = detect(ym, {"year": "numeric_discrete", "month": "ordinal-candidate", "v": "numeric_continuous"})
    assert info["time_column"] == "year + month" and info["kind"] == "single series" and info["frequency"] == "month"
    assert detect(pd.DataFrame({"x": range(50)}), {"x": "numeric_discrete"}) is None


def test_adf() -> None:
    rng = np.random.default_rng(5)
    assert adf_test(rng.normal(size=500))["stationary"]
    assert not adf_test(np.cumsum(rng.normal(size=500)))["stationary"]
    assert analyze_series(pd.Series(np.ones(30)), "day", CFG)["verdict"] == "too short or constant"


# ------------------------------------------------------------------------------------------- selection
def test_feature_selection_separates_signal_from_noise() -> None:
    rng = np.random.default_rng(6)
    n = 800
    df = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.normal(size=n), "c": rng.choice(list("abc"), n)})
    for i in range(4):
        df[f"noise{i}"] = rng.normal(size=n)
    df["y"] = 3 * df["x1"] - 2 * df["x2"] + df["c"].map({"a": 0, "b": 2, "c": 4}) + rng.normal(size=n)
    types = {c: "numeric_continuous" for c in df}
    types["c"] = "categorical"
    v = see(df, target="y", plots=False, verbose=False)
    sel = v.selection
    assert set(sel["keep"]) == {"x1", "x2", "c"}
    assert set(sel["drop"]) | set(sel["borderline"]) == {f"noise{i}" for i in range(4)}
    assert sel["scores"]["lasso"] > 0.8 and sel["task"] == "regression"
    df["label"] = (df["y"] > df["y"].median()).map({True: "hi", False: "lo"})
    out = feature_selection(df.drop(columns="y"), "label", {**types, "label": "boolean"}, {}, CFG)
    assert out["task"] == "binary classification" and {"x1", "x2", "c"} <= set(out["keep"])


# ------------------------------------------------------------------------------------------- preparation
def test_bktree_and_misspellings() -> None:
    words = ["paris", "lyon", "nantes", "marseille", "toulouse"]
    tree = BKTree(words)
    assert tree.search("pariss", 1) == [(1, "paris")] and edit_distance("kitten", "sitting") == 3
    s = pd.Series(["Paris"] * 50 + ["Lyon"] * 40 + ["Pariss", "paris ", "Lyonn", "Nîmes"] + ["Nimes"] * 20
                  + ["ISLAND"] * 5 + ["INLAND"] * 60)
    fixes = misspellings(s)
    assert fixes == {"Pariss": "Paris", "paris ": "Paris", "Lyonn": "Lyon", "Nîmes": "Nimes"}
    codes = pd.Series([f"CA {i}" for i in range(200)])
    assert misspellings(codes) == {}


def test_category_grouping_composite_and_scales() -> None:
    rng = np.random.default_rng(7)
    lab = rng.choice(list("abcdefgh"), 4000)
    rate = pd.Series(lab).map({"a": .1, "b": .1, "c": .1, "d": .5, "e": .5, "f": .9, "g": .9, "h": .9})
    y = pd.Series(np.where(rng.random(4000) < rate, "yes", "no"))
    groups = group_categories(pd.Series(lab), y, False, CFG)
    assert sorted(map(sorted, groups)) == [["a", "b", "c"], ["d", "e"], ["f", "g", "h"]]
    comp = composite(pd.Series([f"{c}, {k}" for c, k in zip(rng.choice(["Paris", "Lyon"], 300),
                                                            rng.choice(["France", "FR"], 300))]))
    assert comp["separator"] == ", " and comp["parts"] == 2
    assert composite(pd.Series([f"{w} and {w}, then more" for w in rng.choice(["x", "y", "z"], 300)])) is None
    assert composite(pd.Series(["1920x1080", "1280x720"] * 50))["parts"] == 2
    mixed = pd.Series(np.r_[rng.normal(70, 8, 300), rng.normal(70_000, 8_000, 60)])
    assert magnitude_mix(mixed)["ratio"] == 1000
    assert magnitude_mix(pd.Series(rng.lognormal(3, 2, 2000))) is None


def test_report_has_every_new_section(tmp_path: Path) -> None:
    df = _latent(400)
    df["t"] = pd.date_range("2020", periods=400, freq="D")
    df["city"] = np.random.default_rng(8).choice(["Paris", "Lyon"], 400)
    df.loc[[10, 20], "city"] = "Pariss"
    df.loc[3, "a0"] = 99.0
    df["y"] = df["a0"] + df["b1"]
    v = see(df, target="y", out_dir=tmp_path, verbose=False)
    html = v.recap["html"].read_text()
    for title in ("Feature selection", "Time series", "Outliers", "Multivariate statistics", "Preparation plan",
                  "Dimensionality reduction", "Validation folds", "Outlier points"):
        assert title in html, title
    assert (tmp_path / "outliers.csv").exists()
    kinds = {r["kind"] for r in v.plot_index}
    assert {"selection", "pca", "outliers", "timeseries", "acf"} <= kinds
    actions = {s["action"] for s in v.preparation["steps"]}
    assert {"Misspelling", "Decomposition", "Dimensionality reduction"} <= actions
