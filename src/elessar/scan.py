"""One streaming pass over a table too big to profile in one piece, guided by the types decided
beforehand (:mod:`elessar.progressive`).

Each chunk is reduced to a small, mergeable summary; summaries are merged pairwise, so chunks can
be reduced in any order and in parallel. The pass yields:

- exact figures over every row: row count, missing values, min / max / mean / std / zeros /
  negatives / infinities of numeric columns (Chan's parallel variance) - computed on the values
  the plan says they are (numbers written as text are parsed, dates read with their format) -,
  duplicate rows (64-bit row hashes), with and without the identifier columns, value frequencies
  of low-cardinality columns;
- **type checks** on every row: values that do not fit the planned type, with examples and their
  row number; and the uniqueness of every identifier, which no sample can prove;
- distinct counts, exact below ``KMV_K`` distinct values and estimated above (k minimum values
  sketch, relative error about 1.6%);
- a uniform random sample of ``k`` rows without replacement (the ``k`` smallest random keys), in
  the original row order and indexed by original row number: the row-level profile runs on it;
- for a time-indexed table, the **series aggregated per period over every row** (rows, and per
  value column its count, sum and sum of squares), so temporal statistics never rest on a thinned
  sample; for a panel, the number of active entities per period and **whole entities drawn at
  random** (an entity is kept when the hash of its key falls below a threshold that halves when
  the kept rows exceed the budget: a Bernoulli sample of entities, identical in any chunk order).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np
import pandas as pd

from ._utils import dedupe_names, hashable, quiet
from .config import PLACEHOLDER_STRINGS
from .types import NUMERIC_TYPES, as_numeric

KMV_K = 4096
VALUE_COUNT_CAP = 1000
VIOLATION_EXAMPLES = 5
_HASH_SPACE = float(2**64)
_MIX = np.uint64(1_000_003)  # combines the column hashes of a row into one row hash
_PLACEHOLDERS = frozenset(PLACEHOLDER_STRINGS) | {"nat"}  # values that mean "missing", not a type error
_PERIOD = {"second": "s", "minute": "min", "hour": "h", "day": "D"}
_CALENDAR = {"week": "W", "month": "M", "quarter": "Q", "year": "Y"}


@dataclass
class ScanPlan:
    """What the full pass needs to know about the columns (from progressive typing)."""

    types: dict[str, dict[str, Any]] = field(default_factory=dict)  # column -> type decision
    time_column: str | None = None
    time_unit: str | None = None
    value_columns: list[str] = field(default_factory=list)
    entity: str | None = None
    entity_budget: int = 0

    @property
    def ids(self) -> list[str]:
        return [c for c, d in self.types.items() if d.get("semantic_type") == "id"]


@dataclass
class ColumnPart:
    n_missing: int = 0
    numeric: bool = True  # every chunk so far gave numbers (or nothing)
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0
    min: Any = None
    max: Any = None
    n_zeros: int = 0
    n_negative: int = 0
    n_inf: int = 0
    datetime: bool = True
    hashes: np.ndarray = field(default_factory=lambda: np.empty(0, dtype="uint64"))
    counts: pd.Series | None = field(default_factory=lambda: pd.Series(dtype="int64"))
    dtypes: set[str] = field(default_factory=set)
    n_violations: int = 0
    violations: list[tuple[int, str]] = field(default_factory=list)  # (row position key, value)


@dataclass
class Part:
    n_rows: int = 0
    columns: dict[str, ColumnPart] = field(default_factory=dict)
    sample: pd.DataFrame | None = None  # with the private column _pos = chunk id << 32 | row in chunk
    keys: np.ndarray = field(default_factory=lambda: np.empty(0))
    chunk_rows: dict[int, int] = field(default_factory=dict)
    row_hashes: list[np.ndarray] | None = field(default_factory=list)
    key_hashes: list[np.ndarray] | None = None  # rows without their identifier columns
    id_hashes: dict[str, list[np.ndarray]] = field(default_factory=dict)
    id_columns: list[str] = field(default_factory=list)
    n_hashed: int = 0
    periods: pd.DataFrame | None = None  # per period: rows, <col>__n, <col>__sum, <col>__sq
    pairs: np.ndarray | None = None  # unique (period, entity hash) pairs, for active entities per period
    entity_tau: float = 1.0
    entity_rows: pd.DataFrame | None = None


@dataclass
class FullScan:
    """Exact figures over every row, plus the samples the profile is computed on."""

    n_rows: int
    columns: pd.DataFrame
    duplicate_rows: dict[str, Any] | None
    sample: pd.DataFrame
    n_chunks: int = 0
    engine: str = ""
    violations: dict[str, dict[str, Any]] = field(default_factory=dict)
    id_duplicates: dict[str, int | None] = field(default_factory=dict)
    series: pd.DataFrame | None = None
    time_unit: str | None = None
    panel_rows: pd.DataFrame | None = None
    panel_fraction: float | None = None

    @property
    def sampled(self) -> bool:
        return len(self.sample) < self.n_rows


def _hash(values: pd.Series) -> np.ndarray:
    return pd.util.hash_pandas_object(hashable(values), index=False).to_numpy(dtype="uint64")


def period_start(t: pd.Series, unit: str) -> pd.Series:
    """Start of the calendar period of each timestamp."""
    if unit in _PERIOD:
        return t.dt.floor(_PERIOD[unit])
    return t.dt.to_period(_CALENDAR.get(unit, "D")).dt.start_time


def _placeholder(values: pd.Series) -> pd.Series:
    return values.astype(str).str.strip().str.lower().isin(_PLACEHOLDERS)


def _view(s: pd.Series, decision: dict[str, Any] | None) -> tuple[str, pd.Series | None]:
    """The column as the plan says it is: ("number", floats), ("date", datetimes) or ("raw", None)."""
    stype = (decision or {}).get("semantic_type")
    if stype in ("datetime", "datetime_string"):
        from .dates import parse_dates

        return "date", parse_dates(s, decision.get("date_format"))
    if stype in NUMERIC_TYPES or stype in ("numeric_string", "boolean") or (
            stype in ("id", "categorical", "ordinal-candidate") and pd.api.types.is_numeric_dtype(s.dtype)):
        return "number", as_numeric(s, stype)
    return "raw", None


def _numeric_values(s: pd.Series) -> np.ndarray | None:
    if pd.api.types.is_bool_dtype(s.dtype) or pd.api.types.is_numeric_dtype(s.dtype):
        return s.dropna().to_numpy(dtype="float64")
    if s.isna().all():
        return np.empty(0)
    return None


def _unique(s: pd.Series) -> bool:
    try:
        return bool(s.notna().sum() > 0 and hashable(s).nunique() == s.notna().sum())
    except TypeError:
        return False


def reduce_chunk(chunk: pd.DataFrame, chunk_id: int, k: int, seed: int = 42, hash_rows: bool = True,
                 max_key: float | None = None, plan: ScanPlan | None = None) -> Part:
    """Summarise chunk number ``chunk_id`` (chunks are numbered in table order, from 0).

    ``max_key`` drops sample candidates whose random key is above it: when the table size is
    roughly known, rows with a key far above ``k / n_rows`` cannot be among the ``k`` kept.
    """
    plan = plan or ScanPlan()
    part = Part(n_rows=len(chunk), chunk_rows={chunk_id: len(chunk)})
    chunk = chunk.set_axis(dedupe_names(list(chunk.columns)), axis=1)
    if plan.types:
        ids = [c for c in plan.ids if c in chunk.columns]
    else:  # no plan: row identifiers by name and uniqueness within the chunk
        from .types import _HINT_ID

        ids = [c for c in chunk.columns if _HINT_ID.search(str(c)) and _unique(chunk[c])]
    row_hash: np.ndarray | None = None
    key_hash: np.ndarray | None = None
    with quiet():
        for name in chunk.columns:
            s = chunk[name]
            col = ColumnPart(dtypes={str(s.dtype)})
            notna = s.notna()
            col.n_missing = int((~notna).sum())
            kind, view = _view(s, plan.types.get(name))
            if view is not None:
                bad = notna & view.isna()
                if bad.any():
                    bad &= ~_placeholder(s)
                    col.n_violations = int(bad.sum())
                    rows = np.flatnonzero(bad.to_numpy())[:VIOLATION_EXAMPLES]
                    col.violations = [((chunk_id << 32) + int(r), str(s.iloc[r])[:40]) for r in rows]
            if kind == "date":
                col.numeric = False
                d = view.dropna()
                if len(d):
                    col.min, col.max = d.min(), d.max()
            else:
                values = view.dropna().to_numpy(dtype="float64") if kind == "number" else _numeric_values(s)
                if values is None:
                    col.numeric = False
                else:
                    finite = values[np.isfinite(values)]
                    col.n_inf = int(len(values) - len(finite))
                    col.count = int(len(finite))
                    if col.count:
                        col.mean = float(finite.mean())
                        col.m2 = float(((finite - col.mean) ** 2).sum())
                        col.min, col.max = float(finite.min()), float(finite.max())
                        col.n_zeros = int((finite == 0).sum())
                        col.n_negative = int((finite < 0).sum())
                col.datetime = pd.api.types.is_datetime64_any_dtype(s.dtype) or not notna.any()
                if col.datetime and notna.any() and not col.numeric:
                    col.min, col.max = s.min(), s.max()
            hashed = _hash(s)
            row_hash = hashed if row_hash is None else (row_hash * _MIX) ^ hashed
            if name not in ids:
                key_hash = hashed if key_hash is None else (key_hash * _MIX) ^ hashed
            present = notna.to_numpy()
            if present.any():
                distinct = np.sort(pd.unique(hashed[present]))
                col.hashes = distinct[:KMV_K]
                col.counts = hashable(s[present]).value_counts() if len(distinct) <= VALUE_COUNT_CAP else None
                if name in ids and hash_rows and plan.types:
                    part.id_hashes[name] = [hashed[present]]
            part.columns[name] = col
        if hash_rows and row_hash is not None:
            part.row_hashes = [row_hash]
            part.n_hashed = len(chunk)
            if ids and key_hash is not None:
                part.key_hashes, part.id_columns = [key_hash], ids
        else:
            part.row_hashes = None
        if plan.time_column is not None:
            _reduce_time(chunk, chunk_id, part, plan)
    rng = np.random.default_rng([seed, chunk_id])
    keys = rng.random(len(chunk))
    keep = np.arange(len(chunk)) if max_key is None else np.flatnonzero(keys <= max_key)
    if len(keep) > k:
        keep = keep[np.argpartition(keys[keep], k)[:k]]
    part.sample = chunk.iloc[keep].assign(_pos=(np.int64(chunk_id) << 32) + keep.astype("int64"))
    part.keys = keys[keep]
    return part


def _reduce_time(chunk: pd.DataFrame, chunk_id: int, part: Part, plan: ScanPlan) -> None:
    from .timeseries import _time_columns

    t = _time_columns(chunk, {c: d.get("semantic_type") for c, d in plan.types.items()}).get(plan.time_column)
    if t is None:
        return
    if getattr(t.dt, "tz", None) is not None:
        t = t.dt.tz_convert(None)
    period = period_start(t, plan.time_unit or "day")
    frame = pd.DataFrame({"period": period, "rows": 1}, index=chunk.index)
    for c in plan.value_columns:
        if c in chunk.columns:
            x = as_numeric(chunk[c], plan.types.get(c, {}).get("semantic_type"))
            x = x.where(np.isfinite(x))
            frame[f"{c}__n"] = x.notna().astype("int64")
            frame[f"{c}__sum"] = x.fillna(0.0)
            frame[f"{c}__sq"] = (x * x).fillna(0.0)
    part.periods = frame.dropna(subset=["period"]).groupby("period").sum()
    if plan.entity and plan.entity in chunk.columns:
        ent = _hash(chunk[plan.entity])
        ok = period.notna().to_numpy()
        pairs = np.column_stack([period[ok].astype("int64").to_numpy(), ent[ok].view("int64")])
        part.pairs = np.unique(pairs, axis=0)
        chosen = (ent.astype("float64") / _HASH_SPACE) < part.entity_tau
        if chosen.any():
            rows = chunk.loc[chosen].copy()
            rows["_pos"] = (np.int64(chunk_id) << 32) + np.flatnonzero(chosen).astype("int64")
            rows["_ehash"] = ent[chosen]
            part.entity_rows = rows
            _thin_entities(part, plan.entity_budget)


def _thin_entities(part: Part, budget: int) -> None:
    """Halve the entity threshold until the kept rows fit the budget (whole entities only)."""
    while part.entity_rows is not None and budget and len(part.entity_rows) > budget and part.entity_tau > 1e-6:
        part.entity_tau /= 2
        keep = (part.entity_rows["_ehash"].to_numpy().astype("float64") / _HASH_SPACE) < part.entity_tau
        part.entity_rows = part.entity_rows.loc[keep]


def _merge_column(a: ColumnPart, b: ColumnPart) -> ColumnPart:
    out = ColumnPart(n_missing=a.n_missing + b.n_missing, numeric=a.numeric and b.numeric,
                     datetime=a.datetime and b.datetime, dtypes=a.dtypes | b.dtypes,
                     n_violations=a.n_violations + b.n_violations,
                     violations=sorted(a.violations + b.violations)[:VIOLATION_EXAMPLES])
    if out.numeric:
        n = a.count + b.count
        out.count = n
        if n:
            delta = b.mean - a.mean
            out.mean = a.mean + delta * b.count / n
            out.m2 = a.m2 + b.m2 + delta * delta * a.count * b.count / n
        out.n_zeros, out.n_negative, out.n_inf = a.n_zeros + b.n_zeros, a.n_negative + b.n_negative, a.n_inf + b.n_inf
    mins = [m for m in (a.min, b.min) if m is not None and not pd.isna(m)]
    maxs = [m for m in (a.max, b.max) if m is not None and not pd.isna(m)]
    try:
        out.min = min(mins) if mins else None
        out.max = max(maxs) if maxs else None
    except TypeError:  # e.g. time-zone aware and naive datetimes in different chunks
        out.min = out.max = None
    out.hashes = np.sort(pd.unique(np.concatenate([a.hashes, b.hashes])))[:KMV_K]
    if a.counts is None or b.counts is None:
        out.counts = None
    else:
        merged = a.counts.add(b.counts, fill_value=0).astype("int64") if len(a.counts) and len(b.counts) else (
            a.counts if len(a.counts) else b.counts)
        out.counts = merged if len(merged) <= VALUE_COUNT_CAP else None
    return out


def merge(a: Part, b: Part, k: int, max_hashed_rows: int, entity_budget: int = 0) -> Part:
    """Combine two summaries (associative; the order of the chunks does not matter)."""
    out = Part(n_rows=a.n_rows + b.n_rows, chunk_rows={**a.chunk_rows, **b.chunk_rows})
    for name in list(a.columns) + [c for c in b.columns if c not in a.columns]:
        ca, cb = a.columns.get(name), b.columns.get(name)
        if ca is None or cb is None:  # a column absent from some chunks counts as missing there
            only = ca or cb
            absent = (b if ca is not None else a).n_rows
            out.columns[name] = _merge_column(only, ColumnPart(n_missing=absent))
        else:
            out.columns[name] = _merge_column(ca, cb)
    parts = [(p.sample, p.keys) for p in (a, b) if p.sample is not None and len(p.sample)]
    full = [p.keys.max() for p in (a, b) if len(p.keys) >= k]
    if full:  # a row whose key is above the k-th smallest key seen so far can never be kept
        cut = min(full)
        parts = [(smp[keys <= cut], keys[keys <= cut]) for smp, keys in parts]
    frames = [smp for smp, _ in parts if len(smp)]
    if frames:
        with quiet():
            sample = pd.concat(frames, ignore_index=True, sort=False) if len(frames) > 1 else frames[0]
        keys = np.concatenate([keys for smp, keys in parts if len(smp)])
        if len(keys) > k:
            keep = np.argpartition(keys, k)[:k]
            sample, keys = sample.iloc[keep], keys[keep]
        out.sample, out.keys = sample, keys
    if a.row_hashes is not None and b.row_hashes is not None and a.n_hashed + b.n_hashed <= max_hashed_rows:
        out.row_hashes = a.row_hashes + b.row_hashes
        out.n_hashed = a.n_hashed + b.n_hashed
        if a.key_hashes is not None and b.key_hashes is not None and a.id_columns == b.id_columns:
            out.key_hashes, out.id_columns = a.key_hashes + b.key_hashes, a.id_columns
        out.id_hashes = {c: a.id_hashes.get(c, []) + b.id_hashes.get(c, []) for c in set(a.id_hashes) | set(b.id_hashes)}
    else:
        out.row_hashes = None
    periods = [p.periods for p in (a, b) if p.periods is not None]
    if periods:
        out.periods = periods[0] if len(periods) == 1 else periods[0].add(periods[1], fill_value=0)
    pairs = [p.pairs for p in (a, b) if p.pairs is not None]
    if pairs and sum(len(x) for x in pairs) <= 5_000_000:
        out.pairs = np.unique(np.concatenate(pairs), axis=0) if len(pairs) > 1 else pairs[0]
    out.entity_tau = min(a.entity_tau, b.entity_tau)
    rows = [p.entity_rows for p in (a, b) if p.entity_rows is not None and len(p.entity_rows)]
    if rows:
        with quiet():
            both = pd.concat(rows, sort=False) if len(rows) > 1 else rows[0]
        keep = (both["_ehash"].to_numpy().astype("float64") / _HASH_SPACE) < out.entity_tau
        out.entity_rows = both.loc[keep]
        _thin_entities(out, entity_budget)
    return out


def _distinct(hashes: np.ndarray) -> tuple[int, bool]:
    """(distinct count, exact?) from the k smallest hashes."""
    if len(hashes) < KMV_K:
        return len(hashes), True
    kth = float(hashes[KMV_K - 1]) + 1.0
    return int(round((KMV_K - 1) * _HASH_SPACE / kth)), False


def _n_duplicates(hashes: list[np.ndarray]) -> int:
    h = np.sort(np.concatenate(hashes)) if hashes else np.empty(0, dtype="uint64")
    return int((h[1:] == h[:-1]).sum())  # values equal to an earlier one


def _row_numbers(part: Part, pos: np.ndarray) -> np.ndarray:
    sizes = np.zeros(max(part.chunk_rows) + 1, dtype="int64")
    sizes[list(part.chunk_rows)] = list(part.chunk_rows.values())
    offsets = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    pos = np.asarray(pos, dtype="int64")
    return offsets[pos >> 32] + (pos & 0xFFFFFFFF)


def finalize(part: Part, n_chunks: int = 0, engine: str = "", plan: ScanPlan | None = None) -> FullScan:
    rows = []
    n = part.n_rows
    violations: dict[str, dict[str, Any]] = {}
    for name, c in part.columns.items():
        distinct, exact = _distinct(c.hashes)
        row: dict[str, Any] = {
            "column": name, "count": n - c.n_missing, "n_missing": c.n_missing,
            "pct_missing": 100.0 * c.n_missing / n if n else 0.0,
            "n_distinct": distinct, "distinct_exact": exact,
            "dtypes": ", ".join(sorted(c.dtypes)),
            "mean": None, "std": None, "min": None, "max": None,
            "n_zeros": None, "n_negative": None, "n_inf": None, "top_value": None, "top_count": None,
        }
        if c.numeric and c.count:
            row.update(mean=c.mean, std=float(np.sqrt(c.m2 / (c.count - 1))) if c.count > 1 else 0.0,
                       min=c.min, max=c.max, n_zeros=c.n_zeros, n_negative=c.n_negative, n_inf=c.n_inf)
        elif c.min is not None:
            row.update(min=c.min, max=c.max)
        if c.counts is not None and len(c.counts):
            top = c.counts.sort_values(ascending=False, kind="stable")
            row.update(top_value=top.index[0], top_count=int(top.iloc[0]))
        rows.append(row)
        if c.n_violations:
            where = _row_numbers(part, np.array([p for p, _ in c.violations]))
            violations[name] = {"count": c.n_violations, "pct": 100.0 * c.n_violations / max(n - c.n_missing, 1),
                                "examples": [(int(r), v) for r, (_, v) in zip(where, c.violations)]}
    columns = pd.DataFrame(rows).set_index("column") if rows else pd.DataFrame()
    dup = None
    if part.row_hashes is not None:
        n_dup = _n_duplicates(part.row_hashes)
        dup = {"count": n_dup, "pct": 100.0 * n_dup / n if n else 0.0}
        if part.key_hashes:
            n_key = _n_duplicates(part.key_hashes)
            dup.update(count_ignoring_ids=n_key, pct_ignoring_ids=100.0 * n_key / n if n else 0.0,
                       id_columns=part.id_columns)
    id_dups: dict[str, int | None] = {}
    for c in (plan.ids if plan else []):
        id_dups[c] = _n_duplicates(part.id_hashes[c]) if part.row_hashes is not None and c in part.id_hashes else None
    sample = part.sample if part.sample is not None else pd.DataFrame()
    if len(sample):
        rownum = _row_numbers(part, sample["_pos"].to_numpy())
        order = np.argsort(rownum, kind="stable")
        sample = sample.drop(columns="_pos").iloc[order].set_axis(pd.Index(rownum[order]), axis=0)
    series = None
    if part.periods is not None and len(part.periods):
        p = part.periods.sort_index()
        series = pd.DataFrame({"rows": p["rows"]}, index=p.index)
        for c in (plan.value_columns if plan else []):
            if f"{c}__n" in p:
                series[c] = p[f"{c}__sum"] / p[f"{c}__n"].replace(0, np.nan)
        if part.pairs is not None:
            per = pd.Series(part.pairs[:, 0]).value_counts()
            series["active entities"] = per.reindex(series.index.astype("int64")).to_numpy()
        series.index.name = None
    panel_rows = None
    if part.entity_rows is not None and len(part.entity_rows):
        er = part.entity_rows
        rownum = _row_numbers(part, er["_pos"].to_numpy())
        order = np.argsort(rownum, kind="stable")
        panel_rows = er.drop(columns=["_pos", "_ehash"]).iloc[order].set_axis(pd.Index(rownum[order]), axis=0)
    return FullScan(n_rows=n, columns=columns, duplicate_rows=dup, sample=sample, n_chunks=n_chunks, engine=engine,
                    violations=violations, id_duplicates=id_dups, series=series,
                    time_unit=plan.time_unit if plan else None, panel_rows=panel_rows,
                    panel_fraction=part.entity_tau if panel_rows is not None else None)


def scan_chunks(chunks: Iterable[pd.DataFrame], k: int, seed: int = 42, max_hashed_rows: int = 30_000_000,
                engine: str = "pandas", plan: ScanPlan | None = None) -> FullScan:
    """Stream ``chunks`` (pandas DataFrames) one after the other through :func:`reduce_chunk`."""
    total: Part | None = None
    n_seen = 0
    n_chunks = 0
    budget = plan.entity_budget if plan else 0
    for i, chunk in enumerate(chunks):
        part = reduce_chunk(chunk, i, k, seed, hash_rows=n_seen + len(chunk) <= max_hashed_rows, plan=plan)
        n_seen += len(chunk)
        total = part if total is None else merge(total, part, k, max_hashed_rows, budget)
        n_chunks += 1
    return finalize(total or Part(), n_chunks, engine, plan)


__all__ = ["FullScan", "KMV_K", "Part", "ScanPlan", "finalize", "merge", "period_start", "reduce_chunk", "scan_chunks"]
