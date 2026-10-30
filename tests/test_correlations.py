from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from elessar import see_corr
from elessar.correlations import (
    PAIR_COLUMNS,
    TARGET_COLUMNS,
    auc_binary,
    bh_adjust,
    corr_pvalue,
    correlation_ratio,
    correlation_ratio_test,
    cramers_v,
    cramers_v_test,
    fisher_ci,
    missingness_dependence,
    mutual_information,
    variance_inflation,
)


def test_cramers_v() -> None:
    a = pd.Series(list("aabbcc") * 50)
    assert cramers_v(a, a) == pytest.approx(1.0, abs=1e-6)
    rng = np.random.default_rng(0)
    b = pd.Series(rng.choice(list("xy"), 300))
    v, p, n = cramers_v_test(a, b)
    assert v < 0.15 and p > 0.01 and n == 300
    assert cramers_v_test(a, a)[1] < 1e-50
    assert np.isnan(cramers_v(pd.Series(["a"] * 5), pd.Series(list("abcab"))))


def test_correlation_ratio() -> None:
    cats = pd.Series(["a"] * 50 + ["b"] * 50)
    eta, p, n = correlation_ratio_test(cats, pd.Series([0.0] * 50 + [1.0] * 50))
    assert eta == pytest.approx(1.0) and p < 1e-10 and n == 100
    assert correlation_ratio(cats, pd.Series([1.0, 2.0] * 50)) == pytest.approx(0.0)
    assert np.isnan(correlation_ratio(cats, pd.Series([3.0] * 100)))


def test_significance_helpers(rng: np.random.Generator) -> None:
    x = rng.normal(size=200)
    y = x + rng.normal(size=200)
    r, p = stats.pearsonr(x, y)
    assert corr_pvalue(r, 200) == pytest.approx(p, rel=1e-6)
    lo, hi = fisher_ci(r, 200)
    assert lo < r < hi and -1 < lo and hi < 1
    assert np.isnan(fisher_ci(0.5, 3)[0])
    adj = bh_adjust([0.01, 0.04, np.nan, 0.03])
    assert np.isnan(adj[2]) and np.all(adj[[0, 1, 3]] >= [0.01, 0.04, 0.03])
    assert adj[0] == pytest.approx(0.03)


def test_auc_and_mutual_information(rng: np.random.Generator) -> None:
    labels = pd.Series(np.r_[np.zeros(300), np.ones(100)].astype(int))
    perfect = pd.Series(labels.to_numpy(dtype=float))
    auc, p = auc_binary(perfect, labels)
    assert auc == pytest.approx(1.0) and p < 1e-20
    noise = pd.Series(rng.normal(size=400))
    assert abs(auc_binary(noise, labels)[0] - 0.5) < 0.1
    mi_same = mutual_information(labels, labels, False, False)
    assert mi_same == pytest.approx(stats.entropy([0.75, 0.25], base=2), rel=1e-6)
    assert mutual_information(noise, labels, True, False) < 0.05


def test_vif(rng: np.random.Generator) -> None:
    x = rng.normal(size=(500, 3))
    df = pd.DataFrame(x, columns=["a", "b", "c"]).assign(d=lambda d: d.a + d.b, e=lambda d: d.c + 0.1 * rng.normal(size=500))
    vif = variance_inflation(df)["vif"]
    assert np.isinf(vif["d"]) and vif["e"] > 50 and vif["c"] > 50
    assert variance_inflation(df[["a"]]).empty


def test_see_corr(messy_df: pd.DataFrame) -> None:
    out = see_corr(messy_df)
    for key in ("pearson", "pearson_p", "spearman", "spearman_p", "kendall", "kendall_p", "n_numeric", "vif",
                "cramers_v", "cramers_v_p", "eta", "eta_p", "pairs", "top_pairs", "association", "association_p",
                "missingness"):
        assert key in out, key
    assert out["pearson"].loc["income", "income_copy"] == pytest.approx(1.0)
    assert out["pearson_p"].loc["income", "income_copy"] < 1e-10
    pairs = out["pairs"]
    assert list(pairs.columns) == PAIR_COLUMNS
    assert pairs["abs_value"].is_monotonic_decreasing
    assert (pairs["p_adjusted"].dropna() >= pairs["p_value"].dropna() - 1e-15).all()
    top = out["top_pairs"]
    assert (top["abs_value"] >= 0.5).all() and top.iloc[0]["significant"]
    assert "id" not in out["pearson"].columns
    assoc = out["association"]
    assert set(assoc.columns) == set(out["pearson"].columns) | set(out["cramers_v"].columns)
    assert np.allclose(assoc.to_numpy(), assoc.to_numpy().T, equal_nan=True)


def test_missingness_dependence(rng: np.random.Generator) -> None:
    n = 1_000
    group = rng.choice(["a", "b"], n)
    x = rng.normal(size=n)
    y = pd.Series(rng.normal(size=n)).where(~((group == "b") & (rng.random(n) < 0.6)))
    mcar = pd.Series(rng.normal(size=n)).where(rng.random(n) > 0.2)
    df = pd.DataFrame({"group": group, "x": x, "y": y, "mcar": mcar})
    types = {"group": "categorical", "x": "numeric_continuous", "y": "numeric_continuous", "mcar": "numeric_continuous"}
    dep = missingness_dependence(df, types)
    top = dep.iloc[0]
    assert top["column"] == "y" and top["depends_on"] == "group" and top["significant"]
    assert not dep[(dep["column"] == "mcar")]["significant"].any()


def test_target_and_leakage(rng: np.random.Generator) -> None:
    n = 500
    y = rng.normal(size=n)
    df = pd.DataFrame({
        "y": y,
        "leak": y * 3 + 1,
        "signal": y + rng.normal(size=n),
        "noise": rng.normal(size=n),
        "group": np.where(y > 0, "pos", "neg"),
    })
    out = see_corr(df, target="y")
    t = out["target"]
    assert list(t.columns) == TARGET_COLUMNS
    assert t.iloc[0]["feature"] == "leak" and bool(t.iloc[0]["leakage_suspect"])
    by = t.set_index("feature")
    assert by.loc["group", "method"] == "eta" and by.loc["group", "test"] == "kruskal-wallis"
    assert by.loc["signal", "significant"] and not by.loc["noise", "significant"]
    assert by.loc["signal", "mutual_info"] > by.loc["noise", "mutual_info"]
    cls = see_corr(df.assign(y=df["group"]).drop(columns="group"), target="y")["target"].set_index("feature")
    assert set(cls["test"]) == {"mann-whitney"}
    assert cls.loc["leak", "auc"] == pytest.approx(1.0) and cls.loc["leak", "leakage_suspect"]
    assert 0.35 < cls.loc["noise", "auc"] < 0.65


def test_missing_vs_target(rng: np.random.Generator) -> None:
    n = 800
    y = rng.integers(0, 2, n)
    informative = pd.Series(rng.normal(size=n)).where(~((y == 1) & (rng.random(n) < 0.7)))
    random = pd.Series(rng.normal(size=n)).where(rng.random(n) > 0.3)
    mt = see_corr(pd.DataFrame({"y": y, "inf": informative, "rnd": random}), target="y")["missing_target"]
    by = mt.set_index("feature")
    assert by.loc["inf", "significant"] and not by.loc["rnd", "significant"]
    assert "when missing" in by.loc["inf", "detail"]


def test_corr_edge_cases() -> None:
    assert see_corr(pd.DataFrame())["top_pairs"].empty
    assert "pearson" not in see_corr(pd.DataFrame({"a": [1.0, 2.0, 3.0]}))
    with pytest.raises(KeyError):
        see_corr(pd.DataFrame({"a": [1]}), target="missing")
    text_target = pd.DataFrame({"t": [f"some long free text number {i} here" for i in range(30)], "x": range(30)})
    assert see_corr(text_target, target="t")["target"].empty


def test_mutual_information_ignores_row_order() -> None:
    """Equal values share a bin: a column without information gets none, even in a file sorted by the target."""
    from elessar.correlations import mutual_information

    rng = np.random.default_rng(0)
    n = 2000
    y = pd.Series(np.sort(rng.normal(size=n)))  # rows sorted by the target
    assert mutual_information(pd.Series(np.ones(n)), y, True, True) == 0.0
    coin = pd.Series(np.r_[np.zeros(n // 2), np.ones(n // 2)][rng.permutation(n)])
    assert mutual_information(coin, y, True, True) < 0.01
    x = pd.Series(rng.normal(size=n))
    y2 = x + 0.5 * rng.normal(size=n)
    perm = rng.permutation(n)
    shuffled = mutual_information(x.iloc[perm].reset_index(drop=True), y2.iloc[perm].reset_index(drop=True), True, True)
    assert mutual_information(x, y2, True, True) == pytest.approx(shuffled) and shuffled > 0.5


def test_pair_test_is_chosen_from_column_shapes_and_keeps_its_error_rate() -> None:
    rng = np.random.default_rng(14)
    # 40 independent skewed columns: 780 pairs with no association. Keeping the stronger of Pearson and
    # Spearman after the fact rejected about 9.5% of them at 5%
    noise = pd.DataFrame(rng.lognormal(0, 1, (200, 40)), columns=[f"x{i}" for i in range(40)])
    pairs = see_corr(noise)["pairs"]
    assert len(pairs) == 780 and (pairs["method"] == "spearman").all()
    assert (pairs["p_value"] < 0.05).mean() < 0.07
    x = rng.normal(size=500)
    df = pd.DataFrame({"a": x, "b": x + rng.normal(size=500), "c": np.exp(x + rng.normal(size=500))})
    out = see_corr(df)
    assert out["numeric_shape"].loc["a", "pearson"] and not out["numeric_shape"].loc["c", "pearson"]
    ab = out["pairs"].set_index(["col_a", "col_b"]).loc[("a", "b")]
    assert ab["method"] == "pearson" and ab["other_value"] == pytest.approx(out["spearman"].loc["a", "b"])
    ac = out["pairs"].set_index(["col_a", "col_b"]).loc[("a", "c")]
    assert ac["method"] == "spearman" and ac["method_reason"] == "skewed or heavy-tailed: c"


def test_rows_in_time_order_are_tested_on_their_effective_size_and_differences() -> None:
    rng = np.random.default_rng(41)
    n = 400

    def ar(phi: float) -> np.ndarray:
        e = rng.normal(size=n)
        y = np.zeros(n)
        for i in range(1, n):
            y[i] = phi * y[i - 1] + e[i]
        return y

    common = np.cumsum(rng.normal(size=n))
    df = pd.DataFrame({"day": pd.date_range("2020-01-01", periods=n), "a": ar(0.9), "b": ar(0.9),
                       "w1": np.cumsum(rng.normal(size=n)), "w2": np.cumsum(rng.normal(size=n)),
                       "x": common + 0.5 * rng.normal(size=n)})
    df["linked"] = 0.8 * common + 0.3 * np.cumsum(rng.normal(size=n))
    out = see_corr(df.sample(frac=1, random_state=0))  # the order of the file does not matter
    serial = out["serial"].set_index("column")
    assert serial.loc[["w1", "w2", "x", "linked"], "differenced"].all() and not serial.loc[["a", "b"], "differenced"].any()
    pairs = out["pairs"].set_index(["col_a", "col_b"])
    # two independent AR(1) 0.9 series and two independent random walks used to look strongly associated
    assert not pairs.loc[("a", "b"), "significant"] and not pairs.loc[("w1", "w2"), "significant"]
    assert out["n_effective"].loc["a", "b"] < 0.2 * n
    assert pairs.loc[("x", "linked"), "significant"]  # a real link between two random walks remains
    panel = pd.DataFrame({"month": np.tile(pd.date_range("2000-01-01", periods=120, freq="MS"), 4),
                          "shop": np.repeat(list("abcd"), 120),
                          "u": np.concatenate([np.cumsum(rng.normal(size=120)) for _ in range(4)]),
                          "v": np.concatenate([np.cumsum(rng.normal(size=120)) for _ in range(4)])})
    out = see_corr(panel)
    assert "serial" in out and "error" not in out  # differences within each shop
    assert not out["pairs"].set_index(["col_a", "col_b"]).loc[("u", "v"), "significant"]


def test_target_link_sees_a_u_shape() -> None:
    rng = np.random.default_rng(42)
    x = rng.normal(size=400)
    df = pd.DataFrame({"x": x, "noise": rng.normal(size=400), "y": 0.4 * x**2 + rng.normal(size=400)})
    t = see_corr(df, target="y")["target"].set_index("feature")
    # Spearman alone misses it (no monotone trend); the Kruskal-Wallis across deciles sees it
    assert abs(stats.spearmanr(df["x"], df["y"])[0]) < 0.1 and t.loc["x", "significant"]
    assert not t.loc["noise", "significant"] and "Bonferroni" in t.loc["x", "test"]
