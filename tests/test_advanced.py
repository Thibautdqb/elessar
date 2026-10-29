from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from elessar import StoneConfig, see
from elessar.factor import alpha_label, cronbach_alpha, factor_analysis, scree_elbow
from elessar.multivariate import kmo, multivariate, pca_analysis
from elessar.outliers import column_outliers
from elessar.preparation import BKTree, composite, edit_distance, group_categories, magnitude_mix, misspellings
from elessar.selection import feature_selection
from elessar.timeseries import (adf_test, analyze_series, bartlett_se, detect, kpss_test, mann_kendall,
                                stationarity)

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


def _survey(n: int = 400, seed: int = 3) -> pd.DataFrame:
    """Two Likert scales of four items (one reversed), an unrelated item, an item loading on both."""
    rng = np.random.default_rng(seed)
    f1, f2 = rng.normal(size=n), rng.normal(size=n)

    def likert(z: np.ndarray) -> np.ndarray:
        return np.clip(np.round(3 + 1.1 * z), 1, 5)

    df = pd.DataFrame({**{f"sat{i}": likert(f1 + 0.6 * rng.normal(size=n)) for i in range(4)},
                       **{f"trust{i}": likert(f2 + 0.6 * rng.normal(size=n)) for i in range(3)},
                       "trust_rev": likert(-f2 + 0.6 * rng.normal(size=n)),
                       "noise": likert(rng.normal(size=n)),
                       "mixed": likert(0.7 * f1 + 0.7 * f2 + 0.5 * rng.normal(size=n))})
    df.loc[:5, "noise"] = -999
    return df


def test_factor_analysis_follows_the_nine_steps() -> None:
    df = _survey()
    fa = factor_analysis(df, {c: "numeric_discrete" for c in df}, CFG)
    assert fa["declared_missing"] == {"noise": [-999]} and fa["same_scale"] and fa["sample_ok"]
    assert fa["kmo"] > 0.5 and fa["bartlett_p"] < 0.05
    # the unrelated item goes for its communality, then the cross-loading one; one at a time
    assert [r["item"] for r in fa["removed"]] == ["noise", "mixed"]
    assert fa["removed"][0]["why"].startswith("communality") and "cross-loading" in fa["removed"][1]["why"]
    sol = fa["solution"]
    assert sol["k_kaiser"] == sol["k_variance"] == sol["k_scree"] == fa["n_factors"] == 2
    assert sorted(sorted(m) for m in fa["factors"].values()) == [["sat0", "sat1", "sat2", "sat3"],
                                                                 ["trust0", "trust1", "trust2", "trust_rev"]]
    assert min(fa["communality"].values()) >= 0.5
    trust = next(f for f, m in fa["factors"].items() if "trust0" in m)
    rel = fa["reliability"][trust]
    assert rel["reversed"] == ["trust_rev"] and rel["alpha"] > 0.8 and rel["label"] in ("very good", "excellent")
    assert not rel["proposals"]
    assert set(fa["descriptives"]) == set(fa["scores"].columns) and len(fa["descriptives"]) == 2
    d = next(iter(fa["descriptives"].values()))
    assert {"mean", "median", "mode", "std"} <= set(d) and 1 <= d["mean"] <= 5
    assert "6 - items['trust_rev']" in fa["code"] and "MEAN.4" in fa["code"]
    assert [st["step"] for st in fa["steps"]] == sorted(st["step"] for st in fa["steps"])
    assert fa["verdict"] == "recommended"


def test_factor_analysis_ranks_likert_labels_and_standardises_mixed_scales() -> None:
    rng = np.random.default_rng(5)
    n = 300
    f = rng.normal(size=n)
    labels = np.array(["strongly disagree", "disagree", "neutral", "agree", "strongly agree"])
    df = pd.DataFrame({"q1": labels[np.clip(np.round(2 + f + 0.5 * rng.normal(size=n)), 0, 4).astype(int)],
                       "spending": 2000 + 800 * f + 300 * rng.normal(size=n),
                       "score": 50 + 10 * f + 4 * rng.normal(size=n)})
    types = {"q1": "ordinal-candidate", "spending": "numeric_continuous", "score": "numeric_continuous"}
    fa = factor_analysis(df, types, CFG)
    assert fa["kinds"]["q1"].startswith("Likert labels") and not fa["same_scale"]
    assert fa["n_factors"] == 1 and next(iter(fa["reliability"].values()))["standardised"]
    assert "z-scores" in fa["code"]


def test_factor_analysis_not_applicable_or_not_useful() -> None:
    rng = np.random.default_rng(1)
    two = pd.DataFrame(rng.normal(size=(200, 2)), columns=["a", "b"])
    assert factor_analysis(two, {c: "numeric_continuous" for c in two}, CFG)["verdict"] == "not applicable"
    noise = pd.DataFrame(rng.normal(size=(500, 5)), columns=list("abcde"))
    assert factor_analysis(noise, {c: "numeric_continuous" for c in noise}, CFG)["verdict"] == "not useful"
    small = _survey(n=60)
    fa = factor_analysis(small, {c: "numeric_discrete" for c in small}, CFG)
    assert not fa["sample_ok"] and fa["verdict"] == "indicative"


def test_cronbach_alpha_and_scree_elbow() -> None:
    items = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [2, 3, 4, 5, 6], "c": [1, 2, 3, 4, 5]})
    assert cronbach_alpha(items) == pytest.approx(1.0)
    # the textbook formula on a small example: k / (k - 1) * (1 - sum var_i / var_total)
    x = pd.DataFrame({"a": [3, 4, 2, 5, 4, 3], "b": [2, 4, 3, 5, 3, 2], "c": [3, 5, 2, 4, 4, 3]})
    expected = 3 / 2 * (1 - x.var().sum() / x.sum(axis=1).var())
    assert cronbach_alpha(x) == pytest.approx(expected)
    assert [alpha_label(a) for a in (0.95, 0.85, 0.75, 0.65, 0.55, 0.3)] == [
        "excellent", "very good", "good", "acceptable", "poor", "unacceptable"]
    assert scree_elbow(np.array([3.5, 2.4, 0.4, 0.3, 0.2, 0.2])) == 2
    assert scree_elbow(np.array([4.0, 0.5, 0.4, 0.3])) == 1


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


def test_adf_kpss_and_combined_verdict() -> None:
    rng = np.random.default_rng(5)
    noise = rng.normal(size=500)
    walk = np.cumsum(rng.normal(size=500))
    trend = 0.02 * np.arange(500) + rng.normal(size=500)
    assert adf_test(noise)["reject_unit_root"] and not adf_test(walk)["reject_unit_root"]
    assert not kpss_test(noise)["reject_stationarity"] and kpss_test(walk)["reject_stationarity"]
    assert stationarity(noise)["verdict"] == "stationary"
    assert stationarity(walk)["verdict"] == "unit root"
    assert stationarity(trend)["verdict"] == "trend-stationary"
    assert analyze_series(pd.Series(np.ones(30)), "day", CFG)["verdict"] == "too short or constant"


def test_mann_kendall_corrects_for_autocorrelation() -> None:
    rng = np.random.default_rng(6)
    false_positive = 0
    for _ in range(40):  # AR(1) with phi 0.8 and no trend: the naive test rejects far too often
        e = rng.normal(size=300)
        y = np.zeros(300)
        for i in range(1, 300):
            y[i] = 0.8 * y[i - 1] + e[i]
        mk = mann_kendall(y)
        assert mk["variance_inflation"] >= 1 and mk["p_value"] >= mk["p_value_naive"]
        false_positive += mk["p_value"] < 0.05
    assert false_positive <= 12  # the naive version rejects in most of these series
    up = mann_kendall(0.05 * np.arange(200) + rng.normal(size=200))
    assert up["trend"] == "increasing" and abs(up["slope"] - 0.05) < 0.01


def test_bartlett_band_widens_with_lag() -> None:
    r = np.array([1.0, 0.8, 0.6, 0.4])
    se = bartlett_se(r, 100)
    assert se[1] == pytest.approx(0.1) and se[2] > se[1] and se[3] > se[2]


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


def test_factor_analysis_questionnaire_demographics_and_redundancy() -> None:
    df = _survey(n=500, seed=7).drop(columns=["noise", "mixed"])
    rng = np.random.default_rng(7)
    df["age"] = rng.integers(18, 80, len(df))  # describes the respondent: left out
    df["education"] = rng.integers(1, 5, len(df))
    df["sat_total"] = df[[f"sat{i}" for i in range(4)]].sum(axis=1)  # not on the questionnaire's 1-5 scale
    fa = factor_analysis(df, {c: "numeric_discrete" for c in df}, CFG)
    assert set(fa["kinds"]).isdisjoint({"age", "education", "sat_total"}) and fa["left_out"] == ["sat_total"]
    assert fa["same_scale"] and fa["n_factors"] == 2
    # measures on different scales: an exact combination of others makes the matrix singular and is removed
    f1, f2 = rng.normal(size=(2, 400))
    x = pd.DataFrame({f"a{i}": f1 + 0.4 * rng.normal(size=400) for i in range(3)}
                     | {f"b{i}": 10 * f2 + 4 * rng.normal(size=400) for i in range(3)})
    x["a_total"] = x[["a0", "a1", "a2"]].sum(axis=1)
    fa = factor_analysis(x, {c: "numeric_continuous" for c in x}, CFG)
    assert len(fa["redundant"]) == 1 and fa["determinant"] > 1e-5 and fa["n_factors"] == 2


def test_factor_analysis_prefers_fewer_factors_when_many_items_misbehave() -> None:
    rng = np.random.default_rng(11)
    n = 600
    energy, tension = rng.normal(size=n), rng.normal(size=n)
    items = {}
    for name, f, sign in (("active", energy, 1), ("alert", energy, 1), ("sleepy", energy, -1), ("tired", energy, -1),
                          ("calm", tension, -1), ("relaxed", tension, -1), ("nervous", tension, 1), ("jittery", tension, 1)):
        items[name] = np.clip(np.round(1.5 + sign * f + 0.7 * rng.normal(size=n)), 0, 3)
    df = pd.DataFrame(items)
    fa = factor_analysis(df, {c: "numeric_discrete" for c in df}, CFG)
    assert fa["n_factors"] == 2 and not fa["removed"]
    groups = sorted(sorted(m) for m in fa["factors"].values())
    assert groups == [["active", "alert", "sleepy", "tired"], ["calm", "jittery", "nervous", "relaxed"]]
    for rel in fa["reliability"].values():  # the minority of each bipolar factor is reversed
        assert len(rel["reversed"]) == 2


def test_see_types_shows_the_decision_path() -> None:
    from elessar import see_types

    out = see_types(pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "x"]}))
    assert {"certainty", "path"} <= set(out.columns)
    assert out.loc["a", "path"].startswith("any non-missing value?") and out.loc["a", "certainty"] == "certain"
