"""Data quality checks and actionable alerts."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

from ._utils import fmt_bytes, fmt_num, fmt_p, hashable, is_text_dtype, prepare, quiet, sample, truncate
from .config import StoneConfig
from .correlations import compute_correlations
from .types import CATEGORY_TYPES, DATETIME_TYPES, NUMERIC_TYPES, as_datetime, as_numeric, infer_type, python_kinds

SEVERITIES: tuple[str, ...] = ("high", "medium", "info")
_HINT_AGE = re.compile(r"(^|_)age($|_)|(^|_)age_?years")
# a unit other than years written in the name: the 0-120 range of a human age in years does not apply
_NOT_YEARS = re.compile(r"(^|_)(days?|months?|weeks?|hours?|hrs?|minutes?|mins?|seconds?|secs?|ms|jours?|mois|semaines?|"
                        r"heures?)($|_)")
_HINT_NONNEG = re.compile(
    r"price|amount|cost|salary|income|revenue|qty|quantity|count|weight|height|duration|distance|size|area|fare|fee"
)
_KIND_FAMILY = {"bool": "number", "list": "container", "dict": "container"}
_DEFAULT_DATES = (pd.Timestamp("1970-01-01"), pd.Timestamp("1900-01-01"), pd.Timestamp("1899-12-30"))


def alert(severity: str, column: str | None, message: str, kind: str | None = None) -> dict[str, Any]:
    """One alert; ``kind`` tags repetitive informational findings so they can be grouped."""
    return {"severity": severity, "column": column, "message": message, "kind": kind}


GROUPED_KINDS: dict[str, str] = {
    "outliers": "have >= {outlier_alert_pct:g}% outliers by the 1.5xIQR rule",
    "skewed": "are heavily skewed (|skewness| > 2)",
    "heaping": "show heaping (excess of values ending in 0 or 5)",
    "zero_inflated": "are zero-inflated (>= 30% zeros)",
    "high_cardinality": "have high cardinality (> {max_categories} categories)",
    "rare_categories": "have rare categories (< {rare_category_pct:g}% of rows each)",
    "imbalanced": "are imbalanced (one value >= 90% of rows)",
    "multi_label": "hold a list of labels in each cell (one column per label: .str.get_dummies(sep))",
    "integer_codes": "hold integers that label categories (region, store, payment type...): analysed as categories; "
                     "set StoneConfig(column_types=...) for any that are quantities",
}


def group_alerts(alerts: list[dict[str, Any]], config: StoneConfig | None = None, min_count: int = 3) -> list[dict[str, Any]]:
    """Merge informational alerts of the same kind found in ``min_count`` or more columns into one alert."""
    cfg = config or StoneConfig()
    by_kind: dict[str, list[dict[str, Any]]] = {}
    for a in alerts:
        if a.get("kind") in GROUPED_KINDS and a["severity"] == "info":
            by_kind.setdefault(a["kind"], []).append(a)
    merged = {k for k, group in by_kind.items() if len(group) >= min_count}
    out = [a for a in alerts if a.get("kind") not in merged]
    for kind in merged:
        cols = [str(a["column"]) for a in by_kind[kind]]
        listed = ", ".join(cols[:10]) + (f" (+{len(cols) - 10} more)" if len(cols) > 10 else "")
        what = GROUPED_KINDS[kind].format(**vars(cfg))
        out.append(alert("info", None, f"{len(cols)} columns {what}: {listed} -> details per column", kind))
    return out


def sort_alerts(alerts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rank = {s: i for i, s in enumerate(SEVERITIES)}
    return sorted(alerts, key=lambda a: (rank.get(a["severity"], 9), str(a["column"] or "")))


def _hashable_frame(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({c: hashable(df[c]) for c in df.columns}, index=df.index)


def _duplicated(df: pd.DataFrame) -> pd.Series:
    try:
        return _hashable_frame(df).duplicated(keep="first")
    except TypeError:  # values that hash but do not compare (e.g. mixed exotic objects)
        return _hashable_frame(df).astype(str).duplicated(keep="first")


def _strata(codes: np.ndarray) -> np.ndarray:
    """Codes of a column (numbered in the order of the values) cut into at most sqrt(n) groups of near
    values: the groups a dependent column is shuffled within. A column with that few values keeps one
    group per value."""
    n = len(codes)
    k = max(1, int(np.sqrt(n)))
    if codes.max() + 1 <= k:
        return codes
    ranks = pd.Series(codes).rank(method="dense").to_numpy()
    return np.minimum((ranks - 1) * k // ranks.max(), k - 1).astype(np.int64)


def _mutual_information(a: np.ndarray, b: np.ndarray) -> float:
    joint = pd.crosstab(a, b).to_numpy(dtype=float)
    p = joint / joint.sum()
    pa, pb = p.sum(axis=1, keepdims=True), p.sum(axis=0, keepdims=True)
    nz = p > 0
    return float((p[nz] * np.log(p[nz] / (pa @ pb)[nz])).sum())


def _dependence_tree(strata: np.ndarray, root: int = 0) -> list[tuple[int, int | None]]:
    """Columns in the order they are drawn, each with the column it is drawn given (None for ``root``):
    the maximum spanning tree of pairwise mutual information (Chow-Liu), the closest tree of pairwise
    links to the table's joint distribution."""
    k = strata.shape[1]
    mi = np.zeros((k, k))
    for i in range(k):
        for j in range(i + 1, k):
            mi[i, j] = mi[j, i] = _mutual_information(strata[:, i], strata[:, j])
    order: list[tuple[int, int | None]] = [(root, None)]
    best = mi[root].copy()
    parent = np.full(k, root)
    done = {root}
    while len(done) < k:
        j = max((c for c in range(k) if c not in done), key=lambda c: best[c])
        order.append((j, int(parent[j])))
        done.add(j)
        for c in range(k):
            if c not in done and mi[j, c] > best[c]:
                best[c], parent[c] = mi[j, c], j
    return order


def chance_duplicates(df: pd.DataFrame, n_perm: int = 49, max_rows: int = 100_000,
                      random_state: int = 42) -> dict[str, Any]:
    """How many duplicate rows chance alone gives with these columns' values (conditional permutation test).

    Tables are drawn that keep every column's values and the strongest link of each column to another
    (a Chow-Liu tree: each column is shuffled within the groups of the column it depends on most), and
    nothing else: two rows then coincide only because their values are common, or commonly go together,
    never because one row was copied. Few columns with few values (a card type and an issue date)
    coincide often by chance; a coincidence over many varied columns does not. Returns the observed count,
    the mean count over the draws and the one-sided p-value ``(1 + #{drawn >= observed}) / (1 + n_perm)``,
    all on the same rows (at most ``max_rows``).
    """
    rows = df if len(df) <= max_rows else df.sample(max_rows, random_state=random_state)

    def ordered_codes(s: pd.Series) -> np.ndarray:  # codes in the order of the values: groups of near values
        v = hashable(s)
        try:
            return pd.factorize(v, sort=True, use_na_sentinel=False)[0]
        except TypeError:  # values that do not compare: order of appearance
            return pd.factorize(v, use_na_sentinel=False)[0]

    codes = np.column_stack([ordered_codes(rows[c]) for c in rows.columns])
    strata = np.column_stack([_strata(c) for c in codes.T])
    # the most varied column first: the others are drawn within its groups, never the reverse (drawing
    # a varied column within the values of a 3-level one would keep a two-column table exactly as it is)
    root = int(np.argmax(codes.max(axis=0)))
    order = _dependence_tree(strata, root) if codes.shape[1] > 1 else [(0, None)]

    def count(m: np.ndarray) -> int:
        return int(pd.DataFrame(m).duplicated().sum())

    rng = np.random.default_rng(random_state)

    def draw() -> np.ndarray:
        out = np.empty_like(codes)
        drawn_strata = np.empty_like(strata)
        for col, parent in order:
            if parent is None:
                perm = rng.permutation(len(codes))
            else:  # rows whose parent fell in group g receive, in random order, the values of group g
                perm = np.empty(len(codes), dtype=np.int64)
                src = pd.Series(np.arange(len(codes))).groupby(strata[:, parent]).indices
                dst = pd.Series(np.arange(len(codes))).groupby(drawn_strata[:, parent]).indices
                for g, where in dst.items():
                    perm[where] = rng.permutation(src[g])
            out[:, col], drawn_strata[:, col] = codes[perm, col], strata[perm, col]
        return out

    observed = count(codes)
    drawn = np.array([count(draw()) for _ in range(n_perm)])
    return {"observed": observed, "expected": float(drawn.mean()), "n_rows": int(len(rows)), "n_perm": n_perm,
            "n_columns": int(codes.shape[1]),
            "p_value": float((1 + (drawn >= observed).sum()) / (1 + n_perm))}


def duplicate_rows(df: pd.DataFrame, id_columns: list[str] | None = None, random_state: int = 42) -> dict[str, Any]:
    """Count of fully duplicated rows (beyond the first occurrence) and example indices.

    With ``id_columns``, also the rows that only differ by their identifiers: the same record
    entered twice under two ids, which a plain duplicate check never sees. Those are compared with the
    count chance gives (:func:`chance_duplicates`): a few low-variety columns share values without any
    record being entered twice.
    """
    if df.empty or df.shape[1] == 0:
        return {"count": 0, "pct": 0.0, "example_indices": []}
    dup = _duplicated(df)
    n = int(dup.sum())
    out = {"count": n, "pct": 100.0 * n / len(df), "example_indices": [_idx(i) for i in df.index[dup][:10]]}
    ids = [c for c in (id_columns or []) if c in df.columns]
    if ids and len(ids) < df.shape[1]:
        key = _duplicated(df.drop(columns=ids))
        n_key = int(key.sum())
        out.update(count_ignoring_ids=n_key, pct_ignoring_ids=100.0 * n_key / len(df), id_columns=ids,
                   examples_ignoring_ids=[_idx(i) for i in df.index[key & ~dup][:10]])
        if n_key > int(dup.sum()):
            out["chance_ignoring_ids"] = chance_duplicates(df.drop(columns=ids), random_state=random_state)
    return out


def _idx(i: Any) -> Any:
    return i.item() if isinstance(i, np.generic) else i if isinstance(i, (int, str)) else str(i)


def duplicate_columns(df: pd.DataFrame) -> list[list[str]]:
    """Groups of columns with identical content."""
    buckets: dict[tuple[str, int], list[str]] = {}
    for col in df.columns:
        s = hashable(df[col])
        try:
            key = int(pd.util.hash_pandas_object(s, index=False).sum())
        except TypeError:
            key = int(pd.util.hash_pandas_object(s.astype(str), index=False).sum())
        buckets.setdefault((str(df[col].dtype), key), []).append(col)
    groups = []
    for cols in buckets.values():
        while len(cols) > 1:
            first, rest = cols[0], cols[1:]
            same = [c for c in rest if df[c].equals(df[first])]
            if same:
                groups.append([first, *same])
            cols = [c for c in rest if c not in same]
    return groups


def missing_patterns(df: pd.DataFrame, config: StoneConfig | None = None) -> dict[str, Any]:
    """Columns that go missing together, rows with many missing values, and frequent patterns."""
    cfg = config or StoneConfig()
    miss = df.isna()
    out: dict[str, Any] = {"co_missing": [], "n_rows_many_missing": 0, "rows_many_missing_examples": [], "top_patterns": []}
    if df.empty:
        return out
    cols = [c for c in df.columns if miss[c].sum() >= 2][:100]
    if len(cols) >= 2:
        m = sample(miss[cols], cfg.sample_size, cfg.random_state).to_numpy(dtype="float64")
        inter = m.T @ m
        sizes = np.diag(inter)
        union = sizes[:, None] + sizes[None, :] - inter
        with quiet():
            jac = np.where(union > 0, inter / union, 0.0)
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                if jac[i, j] >= cfg.co_missing_jaccard:
                    out["co_missing"].append({"col_a": cols[i], "col_b": cols[j], "jaccard": float(jac[i, j])})
    if df.shape[1] >= 2:
        row_pct = 100.0 * miss.mean(axis=1)
        many = row_pct >= cfg.row_missing_pct
        out["n_rows_many_missing"] = int(many.sum())
        out["rows_many_missing_examples"] = [_idx(i) for i in df.index[many][:10]]
    any_missing = miss.any(axis=1)
    if any_missing.any():
        cols_missing = miss.columns[miss.any()]
        sub = sample(miss.loc[any_missing, cols_missing], cfg.sample_size, cfg.random_state)
        vc = sub.value_counts().head(5)
        scale = any_missing.sum() / len(sub)
        out["top_patterns"] = [
            {"columns": [c for c, m in zip(cols_missing, key if isinstance(key, tuple) else (key,)) if m],
             "rows": int(round(v * scale))}
            for key, v in vc.items()
        ]
    return out


def _strings(s: pd.Series) -> pd.Series:
    """Non-missing string cells of a column (fast path for pandas string dtypes)."""
    strs = s.dropna()
    if isinstance(s.dtype, pd.StringDtype):
        return strs
    return strs[strs.map(lambda v: isinstance(v, str))].astype(str)


def whitespace_issues(s: pd.Series) -> int:
    if not is_text_dtype(s):
        return 0
    strs = _strings(s)
    return int((strs != strs.str.strip()).sum())


def placeholder_values(s: pd.Series, stype: str, config: StoneConfig | None = None) -> dict[str, int]:
    """Values that are probably sentinels for 'missing' (``"NA"``, ``"?"``, ``-999``...)."""
    cfg = config or StoneConfig()
    found: dict[str, int] = {}
    if is_text_dtype(s):
        strs = _strings(s)
        hits = strs[strs.str.strip().str.lower().isin(cfg.placeholder_strings)]
        found.update({repr(k): int(v) for k, v in hits.value_counts().items()})
    if stype in NUMERIC_TYPES or stype == "numeric_string":
        x = as_numeric(s).dropna()
        if len(x) and x.nunique() > 2:
            lo, hi = x.min(), x.max()
            for p in cfg.placeholder_numbers:
                if p in (lo, hi):
                    count = int((x == p).sum())
                    rest = x[x != p]
                    far = rest.empty or (p > rest.max() * 1.5 if p > 0 else p < rest.min() * 1.5 - 1)
                    if count and far:
                        found[fmt_num(p)] = count
    return found


def suspicious_values(s: pd.Series, stype: str) -> list[str]:
    """Domain sanity checks driven by the column name (ages, prices, birth dates, default dates)."""
    name = re.sub(r"([a-z])([A-Z])", r"\1_\2", str(s.name)).lower()
    msgs = []
    if stype in NUMERIC_TYPES or stype == "numeric_string":
        x = as_numeric(s).dropna()
        if _HINT_AGE.search(name):
            neg = int((x < 0).sum())
            old = 0 if _NOT_YEARS.search(name) else int((x > 120).sum())
            if neg:
                msgs.append(f"{neg} negative ages")
            if old:
                msgs.append(f"{old} ages above 120")
        elif _HINT_NONNEG.search(name) and (x < 0).any():
            msgs.append(f"{int((x < 0).sum())} negative values in a quantity that should be >= 0")
    if stype in DATETIME_TYPES:
        d = as_datetime(s).dropna()
        for default in _DEFAULT_DATES:
            n = int((d == default).sum())
            if n:
                msgs.append(f"{n} default-looking dates ({default.date()})")
        old = int((d < pd.Timestamp("1900-01-01")).sum())
        if old:
            msgs.append(f"{old} dates before 1900")
    return msgs


def dtype_optimization(s: pd.Series, stype: str) -> dict[str, Any] | None:
    """Cheaper dtype for the column with the estimated memory saving (None if no real win)."""
    current = int(s.memory_usage(deep=True, index=False))
    col = repr(str(s.name))
    suggestion = None
    with quiet():
        if stype == "datetime_string":  # dates stored as text or as numbers: a datetime column, not a smaller int
            new = as_datetime(s)
            suggestion = ("datetime64", new, f"df[{col}] = elessar.parse_dates(df[{col}])")
        elif pd.api.types.is_integer_dtype(s.dtype) and not pd.api.types.is_bool_dtype(s.dtype) and s.notna().all():
            kind = "unsigned" if len(s) and s.min() >= 0 else "integer"
            new = pd.to_numeric(s, downcast=kind)
            if new.dtype != s.dtype:
                suggestion = (str(new.dtype), new, f"df[{col}] = pd.to_numeric(df[{col}], downcast={kind!r})")
        elif s.dtype == np.float64 and len(s):
            new = s.astype("float32")
            if np.allclose(new.astype("float64"), s, rtol=1e-6, equal_nan=True):
                suggestion = ("float32", new, f"df[{col}] = df[{col}].astype('float32')")
        elif is_text_dtype(s) and stype in CATEGORY_TYPES:
            nonnull = max(int(s.notna().sum()), 1)
            if hashable(s).nunique() / nonnull <= 0.5:
                new = hashable(s).astype("category")
                suggestion = ("category", new, f"df[{col}] = df[{col}].astype('category')")
        elif is_text_dtype(s) and stype == "numeric_string":
            new = as_numeric(s)
            suggestion = ("float64", new, f"df[{col}] = elessar.types.parse_numeric(df[{col}])")
    if suggestion is None:
        return None
    dtype, new, code = suggestion
    new_bytes = int(new.memory_usage(deep=True, index=False))
    saving = current - new_bytes
    if saving <= 0 or saving < 0.1 * current:
        return None
    return {
        "column": str(s.name),
        "current_dtype": str(s.dtype),
        "suggested_dtype": dtype,
        "current_bytes": current,
        "new_bytes": new_bytes,
        "saving_bytes": saving,
        "saving_pct": 100.0 * saving / current if current else 0.0,
        "code": code,
    }


def redundant_pairs(
    correlations: dict[str, pd.DataFrame], config: StoneConfig | None = None, target: str | None = None
) -> list[dict[str, Any]]:
    """Feature pairs whose association exceeds ``high_corr`` (pairs with the target are not redundancy)."""
    cfg = config or StoneConfig()
    pairs = correlations.get("top_pairs")
    if pairs is None or pairs.empty:
        return []
    strong = pairs[(pairs["abs_value"] >= cfg.high_corr) & (pairs["col_a"] != target) & (pairs["col_b"] != target)]
    return [
        {"col_a": r.col_a, "col_b": r.col_b, "method": r.method, "value": float(r.value)}
        for r in strong.itertuples(index=False)
    ]


_COLUMN_KEYS = ("high_missing_columns", "quasi_constant_columns", "mixed_type_columns", "whitespace_columns",
                "placeholder_values", "suspicious_values", "conversion_suggestions")


def column_quality(s: pd.Series, info: dict[str, Any], config: StoneConfig | None = None) -> dict[str, Any]:
    """Quality checks of one column given its type info; only the findings are returned."""
    cfg = config or StoneConfig()
    stype = info["semantic_type"]
    found: dict[str, Any] = {}
    n = len(s)
    pct_missing = 100.0 * s.isna().sum() / n if n else 0.0
    if pct_missing >= cfg.medium_missing_pct and stype != "empty":
        found["high_missing_columns"] = pct_missing
    nonnull = hashable(s.dropna())
    if stype not in ("empty", "constant") and len(nonnull):
        top_pct = 100.0 * nonnull.value_counts().iloc[0] / len(nonnull)
        if top_pct >= cfg.quasi_constant_pct:
            found["quasi_constant_columns"] = top_pct
    kinds = python_kinds(sample(s, cfg.sample_size, cfg.random_state))
    if len({_KIND_FAMILY.get(k, k) for k in kinds}) > 1:
        found["mixed_type_columns"] = kinds
    checks = (("whitespace_columns", lambda: whitespace_issues(s)),
              ("placeholder_values", lambda: placeholder_values(s, stype, cfg)),
              ("suspicious_values", lambda: suspicious_values(s, stype)))
    for key, check in checks:
        res = check()
        if res:
            found[key] = res
    opt = dtype_optimization(s, stype) if stype not in ("empty", "constant") else None
    if opt:
        found["dtype_optimization"] = opt
    if info.get("suggestions"):
        found["conversion_suggestions"] = info["suggestions"]
    return found


KEY_TYPES = ("categorical", "id", "boolean", "ordinal-candidate")


def functional_dependencies(df: pd.DataFrame, types: dict[str, str], config: StoneConfig | None = None,
                            exclude: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
    """Columns that are attributes of a grouping key: one value per key, on every row (``X = f(K)``).

    A district's unemployment rate repeated on each of its clients is an attribute of the district: its
    row-level statistics weigh each district by its number of rows. A dependence is kept only when
    chance cannot explain it: if ``X`` were drawn independently of ``K``, every group of ``K`` would hold
    a single value of ``X`` with probability ``prod over groups of sum_v p_v ** size`` (``p_v`` the shares
    of ``X``'s values), which must fall below ``alpha``. With a key that rarely repeats this probability
    stays high and nothing is claimed. Keys are labels (categories, identifiers), never measurements; each
    column is given its coarsest key (fewest groups).
    """
    cfg = config or StoneConfig()
    rows = sample(df, cfg.sample_size, cfg.random_state)
    n = len(rows)
    if n < 20:
        return []
    codes = {}
    for c in rows.columns:
        if types.get(c) in ("empty", "constant") or c in exclude:
            continue
        try:
            k = pd.factorize(hashable(rows[c]), use_na_sentinel=False)[0]
        except TypeError:
            continue
        if k.max() >= 1:
            codes[c] = k
    n_values = {c: int(k.max()) + 1 for c, k in codes.items()}
    # a key is a label (a category or an identifier), never a measurement: 77 districts are also told apart by
    # their population, which does not make the population a key
    keys = sorted((c for c in codes if n_values[c] <= n // 2 and types.get(c) in KEY_TYPES), key=lambda c: n_values[c])
    found: dict[Any, dict[str, Any]] = {}
    log_alpha = np.log(cfg.alpha)
    for key in keys:
        order = np.argsort(codes[key], kind="stable")
        k_sorted = codes[key][order]
        same_group = k_sorted[1:] == k_sorted[:-1]
        sizes = np.bincount(codes[key])
        sizes = sizes[sizes >= 2]
        for col, x in codes.items():
            if col == key or col in found or n_values[col] > n_values[key]:
                continue
            x_sorted = x[order]
            if np.any(same_group & (x_sorted[1:] != x_sorted[:-1])):
                continue
            p = np.bincount(x) / n
            m, count = np.unique(sizes, return_counts=True)
            log_p = float(np.sum(count * np.log(np.array([np.sum(p ** s) for s in m]))))
            if log_p < log_alpha:
                found[col] = {"column": col, "determined_by": key, "n_groups": n_values[key],
                              "log10_p_chance": log_p / np.log(10)}
    return list(found.values())


def entity_level_censoring(df: pd.DataFrame, dep: dict[str, Any], stype: str) -> dict[str, Any]:
    """The pile-up test at a column's min and max, redone on one value per group of its key."""
    from .stats import _pile_up

    per_entity = as_numeric(df.drop_duplicates(dep["determined_by"])[dep["column"]], stype).dropna().to_numpy()
    if per_entity.size == 0:
        return {}
    out = {**_pile_up(per_entity, float(per_entity.max()), "max"), **_pile_up(per_entity, float(per_entity.min()), "min")}
    return {k: v for k, v in out.items() if k.startswith("capped_")}


def check_quality(
    df: pd.DataFrame,
    types: dict[str, dict[str, Any]],
    correlations: dict[str, pd.DataFrame],
    config: StoneConfig | None = None,
    target: str | None = None,
    per_column: dict[str, dict[str, Any]] | None = None,
    exact_duplicates: dict[str, Any] | None = None,
    exact_rows_many_missing: int | None = None,
) -> dict[str, Any]:
    """Run every dataset-level check given precomputed type info and correlations.

    ``per_column`` holds :func:`column_quality` results already computed (e.g. in parallel);
    missing columns are checked here. ``exact_duplicates`` (``{count, pct}`` over the full table
    when ``df`` is a sample) replaces the duplicate count measured on ``df``, and ``exact_rows_many_missing``
    the number of rows with at least ``row_missing_pct`` % of missing cells.
    """
    cfg = config or StoneConfig()
    out: dict[str, Any] = {
        "duplicate_rows": duplicate_rows(df, [c for c, info in types.items() if info.get("semantic_type") == "id"],
                                         cfg.random_state),
        "duplicate_columns": duplicate_columns(df),
        "empty_columns": [],
        "constant_columns": [],
        "quasi_constant_columns": {},
        "high_missing_columns": {},
        "missing_patterns": missing_patterns(df, cfg),
        "mixed_type_columns": {},
        "whitespace_columns": {},
        "placeholder_values": {},
        "suspicious_values": {},
        "id_columns": [],
        "redundant_pairs": redundant_pairs(correlations, cfg, target),
        "dtype_optimizations": [],
        "conversion_suggestions": {},
    }
    per_column = per_column or {}
    for col in df.columns:
        info = types[col]
        stype = info["semantic_type"]
        if stype == "empty":
            out["empty_columns"].append(col)
        elif stype == "constant":
            out["constant_columns"].append(col)
        elif stype == "id":
            out["id_columns"].append(col)
        try:
            found = per_column[col] if col in per_column else column_quality(df[col], info, cfg)
        except Exception as exc:  # noqa: BLE001 - quality checks must not stop on one column
            out.setdefault("errors", {})[col] = repr(exc)
            continue
        if "error" in found:
            out.setdefault("errors", {})[col] = found["error"]
        for key in _COLUMN_KEYS:
            if key in found:
                out[key][col] = found[key]
        if "dtype_optimization" in found:
            out["dtype_optimizations"].append(found["dtype_optimization"])
    if exact_rows_many_missing is not None:  # the examples stay those found in the sample
        out["missing_patterns"]["n_rows_many_missing"] = exact_rows_many_missing
    if exact_duplicates is not None:
        sample = out["duplicate_rows"]
        exact = {**exact_duplicates, "exact_over_all_rows": not exact_duplicates.get("not_counted")}
        if "count_ignoring_ids" not in exact:  # the pass did not see identifier names: keep the sample's figure
            exact.update({k: v for k, v in sample.items() if k.endswith("ignoring_ids") or k == "id_columns"})
            exact["ignoring_ids_on_sample"] = "count_ignoring_ids" in sample
        out["duplicate_rows"] = {**sample, **exact}
    out["memory_bytes"] = int(df.memory_usage(deep=True, index=True).sum()) if df.shape[1] else 0
    out["potential_saving_bytes"] = int(sum(o["saving_bytes"] for o in out["dtype_optimizations"]))
    out["alerts"] = sort_alerts(quality_alerts(out, cfg))
    return out


def _co_missing_groups(pairs: list[dict[str, Any]]) -> list[list[str]]:
    """Connected groups of columns linked by co-missing pairs."""
    groups: list[set[str]] = []
    for pair in pairs:
        linked = [g for g in groups if pair["col_a"] in g or pair["col_b"] in g]
        merged = {pair["col_a"], pair["col_b"]}.union(*linked)
        groups = [g for g in groups if g not in linked] + [merged]
    return [sorted(g) for g in groups]


def quality_alerts(q: dict[str, Any], config: StoneConfig | None = None) -> list[dict[str, Any]]:
    """Turn the quality dict into ``{severity, column, message}`` alerts."""
    cfg = config or StoneConfig()
    alerts = []
    dr = q["duplicate_rows"]
    if dr["count"]:
        examples = f", e.g. index {dr['example_indices'][:3]}" if dr.get("example_indices") else ""
        alerts.append(alert("medium", None, f"{dr['count']:,} duplicate rows ({dr['pct']:.1f}%){examples} "
                            "-> df.drop_duplicates()"))
    if dr.get("not_counted"):
        alerts.append(alert("info", None, f"duplicate rows not counted over all rows (more rows than memory allows "
                            f"hashing); {dr['count']:,} in the sample", kind="duplicates"))
    extra = (dr.get("count_ignoring_ids") or 0) - dr["count"]
    chance = dr.get("chance_ignoring_ids")
    by_chance = chance is not None and (chance["p_value"] >= cfg.alpha or chance["observed"] <= chance["expected"])
    if extra > 0 and by_chance:
        ids = dr.get("id_columns") or []
        alerts.append(alert("info", None,
                            f"{dr['count_ignoring_ids']:,} rows share all their values once the identifier "
                            f"column{'s' if len(ids) > 1 else ''} {', '.join(map(str, ids))} "
                            f"{'are' if len(ids) > 1 else 'is'} ignored, as many as chance gives with these columns "
                            f"(~{chance['expected']:,.0f} expected on {chance['n_rows']:,} rows when the columns are "
                            f"shuffled keeping their links, observed {chance['observed']:,}, p {chance['p_value']:.2f}): not a sign of "
                            "records entered twice", kind="duplicates_ids"))
    elif extra > 0:
        ids = dr.get("id_columns") or []
        where = " in the sample" if dr.get("ignoring_ids_on_sample") else ""
        ex = dr.get("examples_ignoring_ids") or []
        beyond = (f" (chance gives ~{chance['expected']:,.0f} on {chance['n_rows']:,} rows, observed "
                  f"{chance['observed']:,}, permutation p {chance['p_value']:.2f})" if chance else "")
        alerts.append(alert("high" if dr["pct_ignoring_ids"] >= 1 else "medium", None,
                            f"{dr['count_ignoring_ids']:,} rows ({dr['pct_ignoring_ids']:.1f}%){where} are duplicates once "
                            f"the identifier column{'s' if len(ids) > 1 else ''} {', '.join(map(str, ids))} "
                            f"{'are' if len(ids) > 1 else 'is'} ignored: the same record under several ids"
                            + beyond + (f", e.g. index {ex[:3]}" if ex else "")
                            + f" -> df.drop_duplicates(subset=[c for c in df.columns if c not in {ids!r}])",
                            kind="duplicates_ids"))
    for group in q["duplicate_columns"]:
        for other in group[1:]:
            alerts.append(alert("medium", other, f"identical to column '{group[0]}' -> drop one of them"))
    for col in q["empty_columns"]:
        alerts.append(alert("high", col, "column is entirely missing -> drop it"))
    for col in q["constant_columns"]:
        alerts.append(alert("medium", col, "constant column carries no information -> drop it"))
    for col, pct in q["quasi_constant_columns"].items():
        alerts.append(alert("info", col, f"quasi-constant: {pct:.1f}% of values are identical"))
    for col, pct in q["high_missing_columns"].items():
        sev = "high" if pct >= cfg.high_missing_pct else "medium"
        alerts.append(alert(sev, col, f"{pct:.1f}% missing -> drop, or impute and add a missing-indicator"))
    mp = q["missing_patterns"]
    for group in _co_missing_groups(mp["co_missing"]):
        alerts.append(alert("info", group[0], f"columns go missing together: {', '.join(group)} "
                            "-> likely a common cause (same source / form section)"))
    if mp["n_rows_many_missing"]:
        alerts.append(alert("medium", None, f"{mp['n_rows_many_missing']:,} rows have >= {cfg.row_missing_pct:.0f}% "
                            "of their values missing"))
    for col, kinds in q["mixed_type_columns"].items():
        mix = ", ".join(f"{k}={v}" for k, v in kinds.items())
        alerts.append(alert("high", col, f"mixed Python types ({mix}) -> normalise to a single type"))
    for col, n in q["whitespace_columns"].items():
        alerts.append(alert("medium", col, f"{n:,} values with leading/trailing whitespace -> .str.strip()"))
    for col, found in q["placeholder_values"].items():
        vals = ", ".join(f"{k} ({v})" for k, v in list(found.items())[:5])
        alerts.append(alert("medium", col, f"placeholder values used as missing: {vals} -> replace with NaN"))
    for col, msgs in q["suspicious_values"].items():
        for m in msgs:
            alerts.append(alert("high", col, f"suspicious: {m}"))
    for col in q["id_columns"]:
        alerts.append(alert("info", col, "identifier-like column -> exclude from modelling and statistics"))
    partners: dict[str, list[str]] = {}
    copies = [set(group) for group in q["duplicate_columns"]]
    for pair in q["redundant_pairs"]:
        if any({pair["col_a"], pair["col_b"]} <= group for group in copies):
            continue  # identical columns: already reported as such
        partners.setdefault(pair["col_b"], []).append(f"'{pair['col_a']}' ({pair['method']} {pair['value']:+.3f})")
    for col, others in partners.items():
        alerts.append(alert("medium", col, f"redundant with {', '.join(others)} -> keep only one"))
    for col, suggestions in q["conversion_suggestions"].items():
        for sug in suggestions:
            if sug["flag"] == "mixed-units":
                alerts.append(alert("medium", col, f"mixed units ({sug['note']}): these numbers are not comparable, "
                                    f"statistics use the main unit only -> split value and unit: "
                                    f"{truncate(sug['code'], 110)}"))
            elif sug["flag"] == "multi-label":
                alerts.append(alert("info", col, f"each cell holds a list of labels -> one column per label: "
                                    f"{truncate(sug['code'], 110)}", "multi_label"))
            elif sug["flag"] == "integer-codes":
                alerts.append(alert("info", col, "integers used as category codes, analysed as a category (probable: "
                                    f"{sug['note'].split(';')[0]}) -> {sug['code']}; if they are quantities: "
                                    f"StoneConfig(column_types={{{col!r}: 'numeric_discrete'}})", "integer_codes"))
            elif sug["flag"] != "should-be-category":
                alerts.append(alert("medium", col, f"{sug['flag']}: {truncate(sug['code'], 110)}"))
    opts = q["dtype_optimizations"]
    if opts:
        changes = ", ".join(f"{o['column']} -> {o['suggested_dtype']}" for o in opts[:8])
        more = f" (+{len(opts) - 8} more)" if len(opts) > 8 else ""
        alerts.append(alert("info", None, f"dtype optimizations save ~{fmt_bytes(q['potential_saving_bytes'])}: "
                            f"{changes}{more} (see quality['dtype_optimizations'] for code)"))
    for col, err in q.get("errors", {}).items():
        alerts.append(alert("high", col, f"quality checks failed: {truncate(err, 100)}"))
    return alerts


def column_alerts(col: str, detail: dict[str, Any], config: StoneConfig | None = None) -> list[dict[str, Any]]:
    """Alerts derived from one column's statistics and distribution analysis."""
    cfg = config or StoneConfig()
    st, stype = detail.get("stats", {}), detail.get("semantic_type")
    out = []
    for step, err in detail.get("errors", {}).items():
        out.append(alert("high", col, f"{step} failed: {truncate(err, 100)}"))
    if st.get("n_inf"):
        out.append(alert("high", col, f"{st['n_inf']:,} infinite values -> replace with NaN or clip"))
    numeric = stype in NUMERIC_TYPES or stype == "numeric_string"
    if numeric and (st.get("pct_outliers_iqr") or 0) >= cfg.outlier_alert_pct and st.get("iqr"):
        out.append(alert("info", col, f"{st['pct_outliers_iqr']:.1f}% outliers by the 1.5xIQR rule "
                         f"(outside [{fmt_num(st['iqr_lower'])}, {fmt_num(st['iqr_upper'])}])", "outliers"))
    skew = st.get("skewness")
    if numeric and skew is not None and np.isfinite(skew) and abs(skew) > 2:
        hint = " -> consider a log transform" if (st.get("min") or 0) > 0 and skew > 0 else ""
        out.append(alert("info", col, f"heavily skewed (skewness {skew:.2f}){hint}", "skewed"))
    for side, word in (("max", "capped at"), ("min", "floored at")):
        if st.get(f"capped_{side}") and detail.get("determined_by") is not None \
                and not detail.get("entity_capped", {}).get(f"capped_{side}", True):
            continue  # one group of the key repeated on many rows, not many values censored (see dependencies)
        if st.get(f"capped_{side}") and not (side == "min" and st[side] == 0):
            out.append(alert("medium", col, f"{word} {fmt_num(st[side])}: {st[f'pct_at_{side}']:.1f}% of values sit exactly "
                             "there (censored data) -> treat as censored, or flag and model separately"))
    if numeric and st.get("count") and st.get("min", -1) >= 0 and 100 * st.get("n_zeros", 0) / st["count"] >= 30 \
            and (st.get("n_unique") or 0) > 10:
        pct = 100 * st["n_zeros"] / st["count"]
        out.append(alert("info", col, f"zero-inflated: {pct:.0f}% zeros -> consider a two-part (hurdle) model or "
                         "an is-zero indicator", "zero_inflated"))
    if numeric and (st.get("heaping_index") or 0) >= 2:
        out.append(alert("info", col, f"heaping: {st['pct_round_values']:.0f}% of values end in 0 or 5 "
                         "(expected ~20%) -> rounded / self-reported values?", "heaping"))
    tau, tp = st.get("counts_trend_tau"), st.get("counts_trend_p")
    if tau is not None and tp < cfg.alpha and abs(tau) >= 0.3:
        way = "increase" if tau > 0 else "decrease"
        out.append(alert("info", col, f"rows per {st['counts_resolution']} {way} over time (Kendall tau {tau:.2f}, "
                         f"p={fmt_p(tp)})"))
    if st.get("n_unparsable"):
        out.append(alert("medium", col, f"{st['n_unparsable']:,} values could not be parsed as numbers"))
    if stype == "categorical" and st.get("n_unique", 0) > cfg.max_categories:
        out.append(alert("info", col, f"high cardinality ({st['n_unique']:,} categories) -> group rare levels "
                         "or use target/frequency encoding", "high_cardinality"))
    if stype in CATEGORY_TYPES and st.get("top_values") and 90 <= st["top_values"][0]["pct"] < cfg.quasi_constant_pct:
        out.append(alert("info", col, f"imbalanced: '{st['top_values'][0]['value']}' is "
                         f"{st['top_values'][0]['pct']:.1f}% of values", "imbalanced"))
    if st.get("n_inconsistent_groups"):
        ex = " / ".join(repr(v) for v in st["inconsistent_labels"][0][:3])
        out.append(alert("medium", col, f"{st['n_inconsistent_groups']} label groups differ only by case/spaces/"
                         f"accents, e.g. {ex} -> normalise labels"))
    if stype in CATEGORY_TYPES and st.get("n_rare_categories") and 2 < st.get("n_unique", 0) <= cfg.max_categories:
        out.append(alert("info", col, f"{st['n_rare_categories']} rare categories (< {cfg.rare_category_pct}% each, "
                         f"{st['pct_rows_rare']:.1f}% of rows)", "rare_categories"))
    if st.get("n_future"):
        out.append(alert("medium", col, f"{st['n_future']:,} dates are in the future"))
    if stype in DATETIME_TYPES and st.get("n_unparsed"):
        out.append(alert("medium", col, f"{st['n_unparsed']:,} values could not be parsed as dates"))
    empties = (st.get("n_empty") or 0) + (st.get("n_whitespace_only") or 0)
    if empties:
        out.append(alert("medium", col, f"{empties:,} empty or whitespace-only strings -> treat as missing?"))
    return out


def row_order_alert(details: dict[str, dict[str, Any]], config: StoneConfig | None = None) -> list[dict[str, Any]]:
    """One alert when numeric columns are strongly autocorrelated in row order (sorted file / time series)."""
    cfg = config or StoneConfig()
    hits = []
    for col, d in details.items():
        st = d.get("stats", {})
        r1, p1 = st.get("autocorr_lag1"), st.get("autocorr_lag1_p")
        monotonic = st.get("monotonic_increasing") or st.get("monotonic_decreasing")
        if d.get("semantic_type") in NUMERIC_TYPES and r1 is not None and abs(r1) >= 0.3 and p1 < cfg.alpha and not monotonic:
            hits.append((abs(r1), f"{col} {r1:+.2f}"))
    if not hits:
        return []
    listed = ", ".join(h[1] for h in sorted(hits, reverse=True)[:6]) + (f" (+{len(hits) - 6} more)" if len(hits) > 6 else "")
    return [alert("medium", None, f"rows are ordered (sorted file or time series): lag-1 autocorrelation {listed} "
                  "-> shuffle before a random split, or use time-aware validation")]


def correlation_alerts(
    correlations: dict[str, pd.DataFrame],
    target: str | None = None,
    config: StoneConfig | None = None,
    duplicate_columns: list[list[str]] | None = None,
) -> list[dict[str, Any]]:
    """Alerts from the association tests: multicollinearity, non-random missingness, target issues.

    Linear dependencies that only restate identical columns (``duplicate_columns``) are skipped.
    """
    cfg = config or StoneConfig()
    out = []
    limits = correlations.get("inference_limits")
    if limits is not None and not limits.empty:
        outputs = ", ".join(dict.fromkeys(limits["output"]))
        out.append(alert("high", None, "serial inference conditions/limits for " + outputs
                         + ": TTS requires strict stationarity of at least one process; unsupported pairs have no p-value. "
                         "Other IID/heuristic outputs remain unvalidated; see inference_limits and serial_inference",
                         "serial_inference_limit"))
    vif = correlations.get("vif")
    deps = correlations.get("linear_dependencies")
    explained: set[str] = set()
    if deps is not None:
        copies = [set(group) for group in duplicate_columns or []]
        for r in deps.itertuples(index=False):
            explained.update(r.columns)
            if any(set(r.columns) <= group for group in copies):
                continue
            out.append(alert("high", r.column, f"exact linear dependency: {truncate(r.equation, 160)} -> drop one of "
                             "these columns (singular design matrix)"))
    if vif is not None:
        high = vif[(vif["vif"] >= cfg.vif_threshold) & ~vif.index.isin(list(explained))]["vif"]
        if len(high):
            listed = ", ".join(f"{c} ({'inf' if np.isinf(v) else f'{v:.0f}'})" for c, v in high.head(8).items())
            more = f" (+{len(high) - 8} more)" if len(high) > 8 else ""
            out.append(alert("medium", None, f"multicollinearity, VIF >= {cfg.vif_threshold:g}: {listed}{more} -> drop "
                             "or combine for linear models"))
    miss = correlations.get("missingness")
    if miss is not None and not miss.empty:
        strength = np.where(miss["test"].str.startswith("mann"), (miss["effect"] - 0.5).abs() * 2, miss["effect"])
        strong = miss[miss["significant"] & (strength >= 0.2) & (miss["pct_missing"] >= 1)]
        firsts = strong.drop_duplicates("column")
        if len(firsts) == 1:
            r = firsts.iloc[0]
            out.append(alert("info", r.column, f"missingness depends on '{r.depends_on}': {r.detail} "
                             f"(p_adj={fmt_p(r.p_adjusted)}) -> not missing completely at random"))
        elif len(firsts) > 1:
            listed = ", ".join(f"{r.column} (on {r.depends_on})" for r in firsts.head(8).itertuples(index=False))
            more = f" (+{len(firsts) - 8} more)" if len(firsts) > 8 else ""
            out.append(alert("info", None, f"missing values are not random in {len(firsts)} columns: {listed}{more} "
                             "-> see correlations['missingness']; impute conditionally"))
    mt = correlations.get("missing_target")
    if mt is not None and not mt.empty:
        for r in mt[mt["significant"]].itertuples(index=False):
            out.append(alert("medium", r.feature, f"missingness is informative about '{target}': {r.detail} "
                             f"(p_adj={fmt_p(r.p_adjusted)}) -> add a missing-indicator feature"))
    assoc = correlations.get("target")
    if assoc is not None and not assoc.empty:
        for r in assoc[assoc["leakage_suspect"]].itertuples(index=False):
            auc = f", AUC {r.auc:.3f}" if pd.notna(r.auc) else ""
            out.append(alert("high", r.feature, f"possible target leakage: {r.method} = {r.association:.3f}{auc} "
                             f"with '{target}'"))
    return out


def see_quality(df: pd.DataFrame | pd.Series, config: StoneConfig | None = None) -> dict[str, Any]:
    """All data-quality checks plus their ``alerts`` (list of ``{severity, column, message}``)."""
    cfg = config or StoneConfig()
    df = prepare(df)
    types = {}
    for col in df.columns:
        try:
            types[col] = infer_type(df[col], cfg)
        except Exception as exc:  # noqa: BLE001
            types[col] = {"semantic_type": "mixed", "suggestions": [], "reason": repr(exc)}
    corr = compute_correlations(df, {c: t["semantic_type"] for c, t in types.items()}, None, cfg)
    out = check_quality(df, types, corr, cfg)
    out["missingness_dependence"] = corr["missingness"]
    out["alerts"] = sort_alerts(out["alerts"] + correlation_alerts(corr, None, cfg, out["duplicate_columns"]))
    return out


def advanced_alerts(extra: dict[str, Any], prep: dict[str, Any], target: str | None) -> list[dict[str, Any]]:
    """Alerts from the multivariate, outlier, time-series, selection analyses and the preparation plan."""
    out = []
    ts = extra.get("timeseries")
    if ts:
        main = (ts.get("analyses") or {}).get(ts.get("main"), {})
        extra_txt = ""
        if main.get("period"):
            extra_txt += f", {main['seasonality']} seasonality of period {main['period']}"
        verdict = (main.get("stationarity") or {}).get("verdict")
        if verdict and verdict not in ("stationary", "insufficient data"):
            extra_txt += f", {verdict}"
        what = {"single series": "a time series", "panel": f"a panel of time series (one per {ts.get('entity')})",
                "event log": "timestamped events"}[ts["kind"]]
        out.append(alert("info", ts["time_column"], f"{what}, {ts['frequency']} steps{extra_txt} -> time-ordered "
                         "validation, lags and windows (see Time series)", kind="timeseries"))
    mv = extra.get("multivariate") or {}
    pca, fa = mv.get("pca") or {}, mv.get("factor") or {}
    if pca.get("verdict") == "recommended":
        out.append(alert("info", None, f"PCA recommended: {pca['reason']}", kind="pca"))
    if fa.get("verdict") == "recommended":
        out.append(alert("info", None, f"factor analysis: {fa['reason']}", kind="factor"))
    mo = mv.get("outliers") or {}
    if mo.get("n_both") and mo.get("reliable", True):
        out.append(alert("info", None, f"{mo['n_both']:,} rows are outliers on several columns at once (robust "
                         "Mahalanobis + isolation forest) -> inspect them (see Outliers)", kind="multivariate_outliers"))
    for col, fixes in prep.get("misspellings", {}).items():
        typos = {a: b for a, b in fixes.items() if _key_label(a) != _key_label(b)}
        if typos:
            shown = ", ".join(f"'{a}' -> '{b}'" for a, b in list(typos.items())[:3])
            out.append(alert("info", col, f"possible misspellings of frequent labels (check them): {shown}"
                             + (f" (+{len(typos) - 3} more)" if len(typos) > 3 else ""), kind="misspelling"))
    for col, mix in prep.get("magnitude_mix", {}).items():
        out.append(alert("medium", col, f"values on two scales ~{mix['ratio']}x apart ({mix['share_high']:.0f}% above "
                         f"{mix['cut']:.3g}): mixed units or fractions vs percentages", kind="scale_mix"))
    for col, comp in prep.get("composite", {}).items():
        out.append(alert("info", col, f"{comp['parts']} data items per cell joined by '{comp['separator']}' "
                         f"(e.g. '{comp['example']}') -> split into {comp['parts']} columns", kind="composite"))
    sel = extra.get("selection")
    if sel and sel.get("drop"):
        out.append(alert("info", target, f"feature selection ({sel['n_methods']} methods): drop "
                         + ", ".join(sel["drop"][:8]) + (f" (+{len(sel['drop']) - 8})" if len(sel["drop"]) > 8 else "")
                         + f"; keep {len(sel['keep'])}", kind="selection"))
    return out


def _key_label(text: str) -> str:
    import unicodedata

    return " ".join(unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode().lower().split())


def duplicates_text(dr: dict[str, Any]) -> str:
    """One line for the overview: exact count, the count ignoring identifiers, and where it was measured."""
    text = f"{dr['count']:,} ({dr['pct']:.1f}%)"
    if dr.get("not_counted"):
        text += " in the sample (too many rows to hash them all)"
    if dr.get("count_ignoring_ids", 0) > dr["count"]:
        ids = ", ".join(map(str, dr.get("id_columns") or []))
        text += (f"; {dr['count_ignoring_ids']:,} ({dr['pct_ignoring_ids']:.1f}%) ignoring {ids}"
                 + (" (sample)" if dr.get("ignoring_ids_on_sample") else ""))
    return text
