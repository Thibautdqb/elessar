"""The preparation plan: what to clean and how to transform each column, with the code.

Cleaning: invalid values, wrong formats, misspellings, duplicates, consistency, missing data,
outliers (remove or reassign). Transformation: bucketing, decomposition, aggregation,
combination, rescaling, normalisation, transformation of skewed columns, encoding, dimensionality
reduction. Every step comes from a measurement made elsewhere in the run, never from the name
of a column alone.

Two small algorithms live here:

- misspelt labels are found with a **BK-tree** (a metric tree on the edit distance): each rare
  label is looked up among the frequent ones within 1 or 2 edits;
- categories are **grouped with a small binary tree** on the target rate (or mean) of each label,
  splitting only where the two sides differ significantly: labels in one leaf can be merged.
"""

from __future__ import annotations

import ast
import re
import unicodedata
import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from ._utils import hashable, quiet, truncate
from .config import StoneConfig
from .types import CATEGORY_TYPES, DATETIME_TYPES, MULTI_LABEL, NUMERIC_TYPES, as_numeric

CLEANING = ("Invalid values", "Wrong formats", "Misspelling", "Duplicates", "Consistency", "Missing data",
            "Remove outliers", "Reassign outliers")
TRANSFORMATION = ("Bucketing", "Decomposition", "Aggregation", "Combination", "Transformation", "Rescale",
                  "Normalization", "Encode", "Dimensionality reduction")  # power transforms need the unscaled values


# ------------------------------------------------------------------------------------------- BK-tree
def edit_distance(a: str, b: str) -> int:
    """Levenshtein distance (insertions, deletions, substitutions)."""
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


class BKTree:
    """Burkhard-Keller tree: finds every word within ``k`` edits without comparing to all of them."""

    def __init__(self, words: list[str]) -> None:
        self.root: tuple[str, dict[int, Any]] | None = None
        for w in words:
            self.add(w)

    def add(self, word: str) -> None:
        if self.root is None:
            self.root = (word, {})
            return
        node = self.root
        while True:
            d = edit_distance(word, node[0])
            if d == 0:
                return
            if d not in node[1]:
                node[1][d] = (word, {})
                return
            node = node[1][d]

    def search(self, word: str, k: int) -> list[tuple[int, str]]:
        out, stack = [], [self.root] if self.root else []
        while stack:
            w, children = stack.pop()
            d = edit_distance(word, w)
            if d <= k:
                out.append((d, w))
            stack.extend(child for dist, child in children.items() if d - k <= dist <= d + k)
        return sorted(out)


def _key(label: str) -> str:
    text = unicodedata.normalize("NFKD", label).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", text).strip().lower()


def misspellings(s: pd.Series, max_labels: int = 3_000) -> dict[str, str]:
    """Rare labels that are 1-2 edits from a label at least 5 times more frequent: {variant: correction}."""
    vc = hashable(s.dropna()).astype(str).value_counts()
    if len(vc) < 3 or len(vc) > max_labels or len(vc) > 0.5 * vc.sum():
        return {}  # mostly unique values are codes or names, not a vocabulary
    keys = {}
    for label, n in vc.items():  # labels equal up to case / accents / spaces map to the most frequent spelling
        keys.setdefault(_key(label), []).append((n, label))
    fixes = {}
    for group in keys.values():
        if len(group) > 1:
            best = max(group)[1]
            fixes.update({lab: best for _, lab in group if lab != best})
    canon = {k: max(g)[1] for k, g in keys.items()}
    count = {k: sum(n for n, _ in g) for k, g in keys.items()}
    frequent = [k for k in canon if count[k] >= 5 and len(k) >= 4 and not any(ch.isdigit() for ch in k)]
    tree = BKTree(frequent)
    for k, label in canon.items():
        if len(k) < 4 or any(ch.isdigit() for ch in k):
            continue  # 'CA 2314' vs 'CA 2343' are two codes, not a typo
        limit = 1 if len(k) < 8 else 2
        if count[k] > 3:
            continue  # a typo is rarely repeated identically: 5 'ISLAND' next to 6,551 'INLAND' is a real label
        for d, other in tree.search(k, limit):
            if d and count[other] >= 5 * count[k] and not (k.replace(" ", "").isdigit() or other.isdigit()):
                fixes[label] = canon[other]
                for _, variant in keys[k]:
                    fixes[variant] = canon[other]
                break
    return fixes


# ------------------------------------------------------------------------------------------- category grouping
def group_categories(s: pd.Series, y: pd.Series, regression: bool, cfg: StoneConfig) -> list[list[str]] | None:
    """Merge labels that behave alike towards the target with a small binary tree.

    Labels are ordered by their target mean (the rate of the minority class for a binary target).
    The tree splits that order in two where the two sides differ most (Welch t-test), and only
    if the difference is significant after a Bonferroni correction for the cut points tried;
    each side then splits again, at most three levels deep. The leaves are the groups.
    """
    frame = pd.DataFrame({"x": hashable(s).astype("string"), "y": y}).dropna()
    if regression:
        frame["y"] = pd.to_numeric(frame["y"], errors="coerce")
    else:
        labels = frame["y"].astype("string")
        top = labels.value_counts()
        if len(top) != 2:
            return None
        frame["y"] = (labels == top.index[-1]).astype(float)
    frame = frame.dropna()
    agg = frame.groupby("x")["y"].agg(["size", "sum", lambda v: float((v**2).sum())])
    agg.columns = ["n", "s", "ss"]
    agg = agg[agg["n"] >= 5]
    if len(agg) < 4:
        return None
    agg = agg.assign(mean=agg["s"] / agg["n"]).sort_values("mean")
    total = agg["n"].sum()

    def side(rows: pd.DataFrame) -> tuple[float, float, int]:
        n, sm, ss = rows["n"].sum(), rows["s"].sum(), rows["ss"].sum()
        mean = sm / n
        var = max((ss - n * mean**2) / max(n - 1, 1), 1e-12)
        return mean, np.sqrt(var), int(n)

    def split(rows: pd.DataFrame, depth: int) -> list[list[str]]:
        if depth == 3 or len(rows) < 2:
            return [list(map(str, rows.index))]
        best = None
        for k in range(1, len(rows)):
            left, right = rows.iloc[:k], rows.iloc[k:]
            if min(left["n"].sum(), right["n"].sum()) < 0.05 * total:
                continue
            (m1, s1, n1), (m2, s2, n2) = side(left), side(right)
            res = stats.ttest_ind_from_stats(m1, s1, n1, m2, s2, n2, equal_var=False)
            if best is None or abs(res.statistic) > best[0]:
                best = (abs(res.statistic), res.pvalue, k)
        if best is None or min(1.0, best[1] * (len(rows) - 1)) >= cfg.alpha:
            return [list(map(str, rows.index))]
        return split(rows.iloc[:best[2]], depth + 1) + split(rows.iloc[best[2]:], depth + 1)

    groups = split(agg, 0)
    return groups if 1 < len(groups) < len(agg) else None


_FIT_ON_TRAIN = ("Learnt from the target on every row: for a model, learn it again on the training rows only, "
                 "or the validation score is optimistic")
_FIT_COMMENT = "# learnt from the target on all rows: for a model, learn it on the training rows only"


def tree_bins(x: pd.Series, y: pd.Series, regression: bool, cfg: StoneConfig) -> list[float] | None:
    """Cut points of a numeric column that best separate the target (supervised bucketing)."""
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

    frame = pd.DataFrame({"x": x.replace([np.inf, -np.inf], np.nan), "y": y}).dropna()
    if len(frame) < 50 or frame["x"].nunique() < 8 or not np.isfinite(frame["x"].astype("float32")).all():
        return None
    model = (DecisionTreeRegressor if regression else DecisionTreeClassifier)(
        max_leaf_nodes=5, min_samples_leaf=max(20, len(frame) // 20), random_state=cfg.random_state)
    target = pd.to_numeric(frame["y"], errors="coerce") if regression else hashable(frame["y"]).astype(str)
    if regression and target.isna().any():
        return None
    model.fit(frame[["x"]].to_numpy(dtype="float64"), target.to_numpy())
    cuts = sorted(float(t) for t, f in zip(model.tree_.threshold, model.tree_.feature) if f >= 0)
    return cuts or None


# ------------------------------------------------------------------------------------------- detectors
def magnitude_mix(x: pd.Series) -> dict[str, Any] | None:
    """Two clusters of values about 100x or 1000x apart (g and kg, % and fraction): mixed scales."""
    v = x.replace([np.inf, -np.inf], np.nan).dropna()
    v = v[v > 0]
    if len(v) < 30 or v.nunique() < 10:
        return None
    lg = np.sort(np.log10(v.to_numpy(dtype="float64")))
    best = None
    for q in np.linspace(0.02, 0.98, 49):  # Otsu split in log space
        cut = np.quantile(lg, q)
        a, b = lg[lg <= cut], lg[lg > cut]
        if len(a) < 0.02 * len(lg) or len(b) < 0.02 * len(lg):
            continue
        gap = b.mean() - a.mean()
        spread = max(a.std(), b.std())
        if gap >= 1.8 and spread < 0.45 and (best is None or gap > best[0]):
            best = (gap, cut, len(a) / len(lg))
    if best is None:
        return None
    gap, cut, share = best
    ratio = 10 ** round(gap)
    return {"ratio": int(ratio), "cut": float(10**cut), "share_low": 100 * share, "share_high": 100 * (1 - share)}


_SPLITTERS = (", ", " - ", " / ", "/", "|", ";", ":", " x ", "x", "-")


def composite(s: pd.Series) -> dict[str, Any] | None:
    """Every value is k parts joined by the same separator ('Paris, France', '1920x1080'): decompose."""
    text = s.dropna().astype(str).str.strip()
    text = text[text != ""]
    if len(text) < 20:
        return None
    sample_ = text.sample(min(len(text), 2_000), random_state=0)
    for sep in _SPLITTERS:
        parts = sample_.str.split(re.escape(sep), regex=True)
        k = parts.str.len()
        mode = int(k.mode().iloc[0])
        if mode < 2 or mode > 5 or (k == mode).mean() < 0.95:
            continue
        cols = [parts.str[i].str.strip() for i in range(mode)]
        if any((c == "").mean() > 0.05 for c in cols):
            continue
        if sep in ("x", "-") and not all(c.str.fullmatch(r"\d+(\.\d+)?").mean() > 0.95 for c in cols):
            continue  # a bare 'x' or '-' only counts between numbers (1920x1080, 10-20)
        numeric = [bool(pd.to_numeric(c, errors="coerce").notna().mean() > 0.95) for c in cols]
        distinct = [int(c.nunique()) for c in cols]
        if min(distinct) < 2:
            continue  # a part that never changes is decoration, not a data item
        if sep in (", ", "/", " / ", ";", "|") and all(d > 0.5 * len(sample_) for d in distinct):
            continue  # free text with commas, not a fixed structure
        return {"separator": sep, "parts": mode, "numeric_parts": numeric, "distinct_per_part": distinct,
                "example": str(sample_.iloc[0])}
    return None


# ------------------------------------------------------------------------------------------- the plan
def _safe(fn: Any, *args: Any) -> Any:
    """Run one detector; a detector that fails on odd data just finds nothing."""
    try:
        return fn(*args)
    except Exception:  # noqa: BLE001
        return None


def _literal(key: str) -> Any:
    """A placeholder as recorded by the quality checks ('NA' quoted, numbers formatted) back to a value."""
    if re.fullmatch(r"-?\d{1,3}(,\d{3})+(\.\d+)?", key.strip()):  # 99,999 is a formatted number, not a tuple
        number = float(key.replace(",", ""))
        return int(number) if number.is_integer() else number
    try:
        return ast.literal_eval(key)
    except (ValueError, SyntaxError):
        try:
            return float(key.replace(",", "").replace(" ", ""))
        except ValueError:
            return key


def _step(kind: str, action: str, column: str | None, why: str, code: str = "") -> dict[str, Any]:
    return {"kind": kind, "action": action, "column": column, "why": why, "code": code}


def plan(
    df: pd.DataFrame,
    details: dict[str, dict[str, Any]],
    quality: dict[str, Any],
    correlations: dict[str, pd.DataFrame],
    target: str | None,
    cfg: StoneConfig,
    outliers: dict[str, Any] | None = None,
    multivariate: dict[str, Any] | None = None,
    timeseries: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Every cleaning and transformation step worth doing, ordered, with the reason and the code."""
    steps: list[dict[str, Any]] = []
    types = {c: d.get("semantic_type") for c, d in details.items()}
    q = quality
    y = df[target] if target is not None else None
    t_type = types.get(target) if target else None
    regression = bool(target) and (t_type in NUMERIC_TYPES and df[target].nunique() > 10)
    found: dict[str, Any] = {"misspellings": {}, "category_groups": {}, "bins": {}, "magnitude_mix": {}, "composite": {}}

    # ---- cleaning
    for col, ph in q.get("placeholder_values", {}).items():
        values = [_literal(k) for k in list(ph)[:6]]
        steps.append(_step("cleaning", "Invalid values", col,
                           "placeholders used as missing: " + ", ".join(f"{v!r} ×{n:,}" for v, n in zip(values, ph.values())),
                           f"df[{col!r}] = df[{col!r}].replace({values!r}, np.nan)"))
    for col, sus in q.get("suspicious_values", {}).items():
        steps.append(_step("cleaning", "Invalid values", col, f"impossible values: {truncate(sus, 120)}",
                           f"df.loc[<invalid condition on {col!r}>, {col!r}] = np.nan"))
    for col, d in details.items():
        if (d.get("stats") or {}).get("n_inf"):
            steps.append(_step("cleaning", "Invalid values", col, f"{d['stats']['n_inf']:,} infinite values",
                               f"df[{col!r}] = df[{col!r}].replace([np.inf, -np.inf], np.nan)"))
    for col in q.get("whitespace_columns", {}):
        steps.append(_step("cleaning", "Wrong formats", col, "leading / trailing spaces", f"df[{col!r}] = df[{col!r}].str.strip()"))
    for col, sugs in q.get("conversion_suggestions", {}).items():
        for sug in sugs:
            steps.append(_step("cleaning", "Wrong formats", col, sug.get("flag", "wrong type") +
                               (f" ({sug['note']})" if sug.get("note") else ""), sug.get("code", "")))
    for col, kinds in q.get("mixed_type_columns", {}).items():
        steps.append(_step("cleaning", "Wrong formats", col, f"mixed Python types {kinds}",
                           f"df[{col!r}] = pd.to_numeric(df[{col!r}], errors='coerce')  # or .astype(str)"))
    with quiet():
        for col in df.columns:
            if types.get(col) in CATEGORY_TYPES and types.get(col) != "boolean":
                fixes = _safe(misspellings, df[col])
                if fixes:
                    found["misspellings"][col] = fixes
                    shown = dict(list(fixes.items())[:8])
                    steps.append(_step("cleaning", "Misspelling", col,
                                       f"{len(fixes)} label variant{'s' if len(fixes) > 1 else ''} of more frequent "
                                       f"labels (case, accents, spaces, 1-2 typos): " +
                                       ", ".join(f"{a!r} → {b!r}" for a, b in list(shown.items())[:4]),
                                       f"df[{col!r}] = df[{col!r}].replace({shown!r})"))
    dr = q.get("duplicate_rows") or {}
    if dr.get("count"):
        steps.append(_step("cleaning", "Duplicates", None, f"{dr['count']:,} duplicate rows ({dr['pct']:.1f}%)",
                           "df = df.drop_duplicates()"))
    for group in q.get("duplicate_columns", []):
        steps.append(_step("cleaning", "Duplicates", group[1], f"identical to {group[0]!r}",
                           f"df = df.drop(columns={group[1:]!r})"))
    with quiet():
        for col, d in details.items():
            units = d.get("units") or {}
            if len(units) > 1:
                steps.append(_step("cleaning", "Consistency", col, f"mixed units {list(units)[:4]}: values not comparable",
                                   next((s["code"] for s in d.get("suggestions", []) if s.get("flag") == "mixed-units"), "")))
            if types.get(col) in NUMERIC_TYPES:
                mix = _safe(lambda c=col: magnitude_mix(as_numeric(df[c], types[c])))
                if mix:
                    found["magnitude_mix"][col] = mix
                    steps.append(_step("cleaning", "Consistency", col,
                                       f"two scales ~{mix['ratio']}x apart ({mix['share_low']:.0f}% below "
                                       f"{mix['cut']:.3g}, {mix['share_high']:.0f}% above): mixed units or a "
                                       "fraction / percentage mix",
                                       f"big = df[{col!r}] > {mix['cut']:.4g}\ndf.loc[big, {col!r}] /= {mix['ratio']}"))
    dropped: set[str] = set()  # columns a step removes: no later step may use them
    miss = correlations.get("missingness")
    mnar = set(miss[miss["significant"]]["column"]) if miss is not None and not miss.empty else set()
    for col, d in details.items():
        st = d.get("stats") or {}
        pct = st.get("pct_missing") or 0
        if not pct or col == target:
            continue
        stype = types.get(col)
        if pct >= cfg.high_missing_pct:
            dropped.add(col)
            how, code = "mostly missing: drop it, or keep only a missing indicator", \
                f"df[{col!r} + '_missing'] = df[{col!r}].isna()\ndf = df.drop(columns=[{col!r}])"
        elif stype in NUMERIC_TYPES:
            skewed = abs(st.get("skewness") or 0) > 1
            stat = "median" if skewed or st.get("pct_outliers_iqr", 0) > 5 else "mean"
            how = f"impute with the {stat}" + (" (skewed or outliers)" if stat == "median" else "")
            code = f"df[{col!r}] = df[{col!r}].fillna(df[{col!r}].{stat}())"
        elif stype == "boolean":
            how = "preserve unknown values as nullable 0/1, rather than treating them as True"
            code = ("from elessar.types import as_boolean\n"
                    f"df[{col!r}] = as_boolean(df[{col!r}]).astype('Int64')")
        elif stype in DATETIME_TYPES:
            how, code = "interpolate in time order, or leave missing", f"df[{col!r}] = df[{col!r}].ffill()"
        else:
            how = "use an explicit 'missing' category"
            code = (f"if isinstance(df[{col!r}].dtype, pd.CategoricalDtype):\n"
                    f"    df[{col!r}] = df[{col!r}].cat.add_categories('missing')\n"
                    f"df[{col!r}] = df[{col!r}].fillna('missing')")
        if col in mnar:
            how += "; missingness depends on other columns (not at random): add a missing indicator"
            code = f"df[{col!r} + '_missing'] = df[{col!r}].isna()\n" + code
        steps.append(_step("cleaning", "Missing data", col, f"{pct:.1f}% missing: {how}", code))
    for col, r in ((outliers or {}).get("columns") or {}).items():
        if col == target or r["action"] == "keep":
            continue
        if any(sg.get("flag") == "mixed-units" for sg in details.get(col, {}).get("suggestions", [])):
            continue  # values in different units: split value and unit first, outliers mean nothing before
        x = f"elessar.types.parse_numeric(df[{col!r}])" if types.get(col) == "numeric_string" else f"df[{col!r}]"
        if r["action"] == "remove":
            steps.append(_step("cleaning", "Remove outliers", col, r["why"],
                               "from elessar.outliers import column_outliers\n"
                               f"x = {x}\ncheck = column_outliers(x)\n"
                               "if check is not None and check['action'] == 'remove':\n"
                               "    df = df.loc[~check['remove_mask']].copy()"))
        else:
            steps.append(_step("cleaning", "Reassign outliers", col, r["why"],
                               "from elessar.outliers import column_outliers\n"
                               f"x = {x}\ncheck = column_outliers(x)\n"
                               "if check is not None:\n"
                               f"    df[{col!r}] = x.where(~check['consensus_mask'], "
                               "x.clip(check['lower_fence'], check['upper_fence']))"))

    # ---- transformation
    with quiet():
        for col, d in details.items():
            if col == target:
                continue
            stype, st = types.get(col), d.get("stats") or {}
            if stype in DATETIME_TYPES and d.get("role") == "date of birth":
                ref = (d.get("birth") or {}).get("reference", "")
                steps.append(_step("transformation", "Decomposition", col,
                                   f"a date of birth: model the age {ref}, not the calendar parts of the birth",
                                   f"born = elessar.parse_dates(df[{col!r}])\n"
                                   f"df[{col!r} + '_age'] = (pd.Timestamp('today') - born).dt.days / 365.2425  "
                                   "# or: (df['<event date>'] - born)\n"
                                   f"df[{col!r} + '_band'] = pd.cut(df[{col!r} + '_age'], [0, 18, 25, 35, 45, 55, 65, 120])"))
            elif stype in DATETIME_TYPES:
                steps.append(_step("transformation", "Decomposition", col,
                                   "a date is not a feature: decompose into calendar parts",
                                   f"d = elessar.parse_dates(df[{col!r}])\n"
                                   f"df[{col!r} + '_year'], df[{col!r} + '_month'] = d.dt.year, d.dt.month\n"
                                   f"df[{col!r} + '_dow'], df[{col!r} + '_hour'] = d.dt.day_of_week, d.dt.hour"))
            elif stype == MULTI_LABEL:
                sep = st.get("separator", ", ")
                steps.append(_step("transformation", "Decomposition", col, f"{st.get('n_labels', '?')} labels per list: "
                                   "one 0/1 column per label (keep the frequent ones)",
                                   f"df = df.join(df[{col!r}].str.get_dummies(sep={sep!r}).add_prefix({col!r} + '='))"))
            elif stype in CATEGORY_TYPES or stype == "text":
                comp = _safe(composite, df[col])
                if comp:
                    found["composite"][col] = comp
                    sep, k = comp["separator"], comp["parts"]
                    steps.append(_step("transformation", "Decomposition", col,
                                       f"several data items in one cell ({k} parts joined by {sep!r}, e.g. "
                                       f"{comp['example']!r}): split them",
                                       f"df[[{', '.join(repr(f'{col}_{i + 1}') for i in range(k))}]] = "
                                       f"df[{col!r}].str.split({sep!r}, n={k - 1}, expand=True)"))
            if stype in NUMERIC_TYPES and target is not None and stype != "boolean":
                cuts = _safe(lambda c=col, t=stype: tree_bins(as_numeric(df[c], t), y, regression, cfg)) if (
                    st.get("heaping_index") or (st.get("n_modes_kde") or 1) > 1 or abs(st.get("skewness") or 0) > 2) else None
                if cuts:
                    found["bins"][col] = cuts
                    edges = ", ".join(["-np.inf", *(repr(round(c, 6)) for c in cuts), "np.inf"])
                    steps.append(_step("transformation", "Bucketing", col,
                                       f"irregular shape (modes, heaping or heavy skew): {len(cuts) + 1} bins chosen by "
                                       f"a decision tree on {target!r}. {_FIT_ON_TRAIN}",
                                       f"{_FIT_COMMENT}\n"
                                       f"df[{col!r} + '_bin'] = pd.cut(df[{col!r}], [{edges}])"))
            if stype in CATEGORY_TYPES and stype != "boolean" and target is not None and (st.get("cardinality") or 0) >= 4:
                groups = _safe(group_categories, df[col], y, regression, cfg)
                if groups:
                    found["category_groups"][col] = groups
                    mapping = {lab: f"group_{i + 1}" for i, g in enumerate(groups) for lab in g}
                    shown = " | ".join("{" + ", ".join(truncate(x, 14) for x in g[:4]) + ("…" if len(g) > 4 else "") + "}"
                                       for g in groups[:4])
                    steps.append(_step("transformation", "Bucketing", col,
                                       f"{st['cardinality']} labels behave as {len(groups)} groups towards {target!r} "
                                       f"(significance-tested binary tree): {shown}. {_FIT_ON_TRAIN}",
                                       f"{_FIT_COMMENT}\n"
                                       f"df[{col!r} + '_group'] = df[{col!r}].map({dict(list(mapping.items())[:12])!r})"
                                       + ("  # ...truncated" if len(mapping) > 12 else "")))
    # transform first (a power transform needs the original, positive values), then put the columns on one scale
    numeric_cols = [c for c in df.columns if types.get(c) in NUMERIC_TYPES and c != target and types.get(c) != "boolean"
                    and c not in dropped]
    spreads = {}
    for c in numeric_cols:
        st = details[c].get("stats") or {}
        if st.get("iqr") or st.get("std"):
            spreads[c] = float(st.get("iqr") or st.get("std"))
    for c in numeric_cols:
        st = details[c].get("stats") or {}
        skew = st.get("skewness")
        if skew is None or not np.isfinite(skew) or abs(skew) < 1:
            continue
        x = as_numeric(df[c], types[c]).replace([np.inf, -np.inf], np.nan).dropna()
        if x.nunique() < 10:
            continue
        with quiet(), warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore")
            if (x > 0).all():
                res = _safe(stats.boxcox, x.to_numpy(dtype="float64"))
                kind = "Box-Cox"
            else:
                res = _safe(stats.yeojohnson, x.to_numpy(dtype="float64"))
                kind = "Yeo-Johnson"
            if res is None or not np.isfinite(res[1]) or abs(res[1]) > 5:
                continue  # an extreme power is numerically unstable and means nothing to a reader
            tx, lam = res
            after = float(stats.skew(tx))
        if kind == "Box-Cox":
            # fit lambda on the observed values, then apply it: boxcox() on a column with NaN returns NaN everywhere
            name, code = f"Box-Cox (lambda {lam:.2f})", (f"from scipy.stats import boxcox\n"
                                                        f"_, lam = boxcox(df[{c!r}].dropna())  # fit (on training rows only for a model)\n"
                                                        f"df[{c!r}] = boxcox(df[{c!r}], lmbda=lam)  # missing values stay missing")
            if abs(lam) < 0.15:
                name, code = "log", f"df[{c!r}] = np.log(df[{c!r}])"
        else:
            name, code = f"Yeo-Johnson (lambda {lam:.2f})", (f"from sklearn.preprocessing import PowerTransformer\n"
                                                           f"df[[{c!r}]] = PowerTransformer().fit_transform(df[[{c!r}]])")
        if np.isfinite(after) and abs(after) < abs(skew) * 0.6:
            steps.append(_step("transformation", "Transformation", c, f"skewness {skew:.2f} → {after:.2f} after {name}", code))
    rescaled = False
    if len(spreads) >= 2:
        hi, lo = max(spreads.values()), min(v for v in spreads.values() if v > 0) if any(spreads.values()) else 1
        if lo > 0 and hi / lo >= 10:
            with_outliers = [c for c in spreads if ((outliers or {}).get("columns", {}).get(c) or {}).get("n_consensus")]
            others = [c for c in spreads if c not in with_outliers]
            lines = []
            if with_outliers:
                lines += ["from sklearn.preprocessing import RobustScaler",
                          f"df[{with_outliers!r}] = RobustScaler().fit_transform(df[{with_outliers!r}])  # outliers: median and IQR"]
            if others:
                lines += ["from sklearn.preprocessing import StandardScaler",
                          f"df[{others!r}] = StandardScaler().fit_transform(df[{others!r}])"]
            steps.append(_step("transformation", "Rescale", None,
                               f"numeric scales differ by x{hi / lo:,.0f} ({max(spreads, key=spreads.get)} vs "
                               f"{min(spreads, key=spreads.get)}): distance-based models, regularised models and "
                               "neural nets need common scales", "\n".join(lines)))
            rescaled = True
    sym = [c for c in numeric_cols if abs((details[c].get("stats") or {}).get("skewness") or 0) < 1]
    if len(sym) >= 2 and not rescaled:  # after a rescale every numeric column is already on a common scale
        steps.append(_step("transformation", "Normalization", None,
                           f"{len(sym)} roughly symmetric columns: z-score them (mean 0, sd 1) for linear and distance "
                           "models", f"cols = {sym[:12]!r}\ndf[cols] = (df[cols] - df[cols].mean()) / df[cols].std()"))
    for col, d in details.items():
        if col == target:
            continue
        stype, st = types.get(col), d.get("stats") or {}
        k = st.get("cardinality") or st.get("n_unique") or 0
        if stype == "boolean":
            code = ("from elessar.types import as_boolean\n"
                    f"df[{col!r}] = as_boolean(df[{col!r}]).astype('Int64')")
            why = "boolean: nullable 0/1; unknown values remain missing"
        elif stype == "ordinal-candidate":
            why, code = "ordered levels: map them to ranks", f"order = [...]  # low to high\ndf[{col!r}] = df[{col!r}].map({{v: i for i, v in enumerate(order)}})"
        elif stype in CATEGORY_TYPES and k <= 10:
            why, code = f"{k} labels: one-hot", f"df = pd.get_dummies(df, columns=[{col!r}], dummy_na=True)"
        elif stype in CATEGORY_TYPES:
            why = f"{k} labels: too many for one-hot; target encoding (cross-fitted) or frequency encoding" + (
                f". {_FIT_ON_TRAIN}" if target else "")
            if target:
                # Binary encoders also support multiclass one-vs-rest on sklearn 1.3.
                # Explicit target types avoid treating integer regression targets as classes.
                multi = not regression and hashable(df[target].dropna()).nunique() > 2
                start = ("from sklearn.preprocessing import TargetEncoder\n"
                         "# fit on labelled training rows only; transform validation/test separately\n"
                         f"known = df[{target!r}].notna()\n")
                if multi:
                    code = (start + "encoded = pd.DataFrame(index=df.index)\n"
                            f"classes = sorted(df.loc[known, {target!r}].unique(), key=str)\n"
                            "for i, label in enumerate(classes):\n"
                            f"    enc = TargetEncoder(target_type='binary', random_state={cfg.random_state})\n"
                            "    values = np.full((len(df), 1), np.nan)\n"
                            f"    values[known] = enc.fit_transform(df.loc[known, [{col!r}]], "
                            f"df.loc[known, {target!r}].eq(label).astype(int))\n"
                            "    if (~known).any():\n"
                            f"        values[~known] = enc.transform(df.loc[~known, [{col!r}]])\n"
                            f"    encoded[{col!r} + '__target_' + str(i)] = values[:, 0]\n"
                            f"df = pd.concat([df.drop(columns=[{col!r}]), encoded], axis=1)")
                else:
                    kind = "continuous" if regression else "binary"
                    code = (start + f"enc = TargetEncoder(target_type={kind!r}, random_state={cfg.random_state})\n"
                            "values = np.full((len(df), 1), np.nan)\n"
                            f"values[known] = enc.fit_transform(df.loc[known, [{col!r}]], df.loc[known, {target!r}])\n"
                            "if (~known).any():\n"
                            f"    values[~known] = enc.transform(df.loc[~known, [{col!r}]])\n"
                            f"df[[{col!r}]] = values")
            else:
                code = f"df[{col!r}] = df[{col!r}].map(df[{col!r}].value_counts(normalize=True))"
        elif stype == "text":
            why, code = "free text: TF-IDF features", (f"from sklearn.feature_extraction.text import TfidfVectorizer\n"
                                                     f"tfidf = TfidfVectorizer(min_df=5).fit_transform(df[{col!r}].fillna(''))")
        else:
            continue
        steps.append(_step("transformation", "Encode", col, why, code))
    mv = multivariate or {}
    for g in mv.get("groups") or []:
        others = [c for c in g["columns"] if c != g["representative"]]
        steps.append(_step("transformation", "Combination", ", ".join(g["columns"]),
                           f"{len(g['columns'])} columns move together (mean |Spearman| {g['mean_abs_spearman']:.2f}): "
                           f"keep {g['representative']!r}, or combine them into one score",
                           f"cols = {g['columns']!r}\ndf['combined'] = ((df[cols] - df[cols].mean()) / df[cols].std()).mean(axis=1)"
                           f"\n# or: df = df.drop(columns={others!r})"))
    deps = correlations.get("linear_dependencies")
    if deps is not None and not deps.empty:
        for r in deps.itertuples(index=False):
            steps.append(_step("transformation", "Combination", None, f"exact relation {r.equation}: one column is redundant",
                               "# drop the derived column, it adds no information"))
    pca, fa = mv.get("pca") or {}, mv.get("factor") or {}
    if pca.get("verdict") == "recommended":
        steps.append(_step("transformation", "Dimensionality reduction", None, "PCA: " + pca["reason"], pca["code"]))
    if fa.get("verdict") == "recommended":
        steps.append(_step("transformation", "Dimensionality reduction", None, "Factor analysis: " + fa["reason"], fa["code"]))
    ts = timeseries or {}
    for st in ts.get("treatment") or []:
        if st["step"] == "Aggregation":
            steps.append(_step("transformation", "Aggregation", ts.get("time_column"), st["advice"], st["code"]))
    if ts.get("kind") == "panel":
        e = ts["entity"]
        steps.append(_step("transformation", "Aggregation", e, f"panel data: one row per {e} with summary features",
                           f"per_{e} = df.groupby({e!r}).agg(['mean', 'std', 'min', 'max', 'last'])"))
    steps = [st for st in steps if st["column"] not in dropped or st["action"] == "Missing data"]
    order = {a: i for i, a in enumerate(CLEANING + TRANSFORMATION)}
    steps.sort(key=lambda s: (s["kind"] != "cleaning", order.get(s["action"], 99)))
    return {"steps": steps, **found,
            "counts": {a: sum(s["action"] == a for s in steps) for a in CLEANING + TRANSFORMATION}}


__all__ = ["BKTree", "CLEANING", "TRANSFORMATION", "composite", "edit_distance", "group_categories", "magnitude_mix",
           "misspellings", "plan", "tree_bins"]
