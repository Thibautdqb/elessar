"""One streaming pass over a table too big to profile in one piece.

Each chunk is reduced to a small, mergeable summary; summaries are merged pairwise, so chunks can
be reduced in any order and in parallel (dask tasks, or a plain loop). The pass yields:

- exact figures over every row: row count, missing values, min / max / mean / std / zeros /
  negatives / infinities of numeric columns (Chan's parallel variance), duplicate rows (64-bit row
  hashes), value frequencies of low-cardinality columns;
- distinct counts, exact below ``KMV_K`` distinct values and estimated above (k minimum values
  sketch, relative error about 1.6%);
- a uniform random sample of ``k`` rows without replacement (the ``k`` smallest random keys), in
  the original row order and indexed by original row number. The full profile runs on it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np
import pandas as pd

from ._utils import dedupe_names, hashable, quiet
from .types import _HINT_ID as ID_NAME

KMV_K = 4096
VALUE_COUNT_CAP = 1000
_HASH_SPACE = float(2**64)
_MIX = np.uint64(1_000_003)  # combines the column hashes of a row into one row hash


@dataclass
class ColumnPart:
    n_missing: int = 0
    numeric: bool = True  # every chunk so far had a numeric (or all-missing) dtype
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


@dataclass
class Part:
    n_rows: int = 0
    columns: dict[str, ColumnPart] = field(default_factory=dict)
    sample: pd.DataFrame | None = None  # with the private column _pos = chunk id << 32 | row in chunk
    keys: np.ndarray = field(default_factory=lambda: np.empty(0))
    chunk_rows: dict[int, int] = field(default_factory=dict)
    row_hashes: list[np.ndarray] | None = field(default_factory=list)
    key_hashes: list[np.ndarray] | None = None  # rows without their identifier columns
    id_columns: list[str] = field(default_factory=list)
    n_hashed: int = 0


@dataclass
class FullScan:
    """Exact figures over every row, plus the uniform sample the profile is computed on."""

    n_rows: int
    columns: pd.DataFrame
    duplicate_rows: dict[str, Any] | None
    sample: pd.DataFrame
    n_chunks: int = 0
    engine: str = ""

    @property
    def sampled(self) -> bool:
        return len(self.sample) < self.n_rows


def _hash(values: pd.Series) -> np.ndarray:
    return pd.util.hash_pandas_object(hashable(values), index=False).to_numpy(dtype="uint64")


def _numeric_values(s: pd.Series) -> np.ndarray | None:
    if pd.api.types.is_bool_dtype(s.dtype):
        return s.dropna().to_numpy(dtype="float64")
    if pd.api.types.is_numeric_dtype(s.dtype):
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
                 max_key: float | None = None) -> Part:
    """Summarise chunk number ``chunk_id`` (chunks are numbered in table order, from 0).

    ``max_key`` drops sample candidates whose random key is above it: when the table size is
    roughly known, rows with a key far above ``k / n_rows`` cannot be among the ``k`` kept.
    """
    part = Part(n_rows=len(chunk), chunk_rows={chunk_id: len(chunk)})
    chunk = chunk.set_axis(dedupe_names(list(chunk.columns)), axis=1)
    row_hash: np.ndarray | None = None
    key_hash: np.ndarray | None = None
    # row identifiers: an id-like name and one value per row (a repeated account number is a grouping key)
    ids = [c for c in chunk.columns if ID_NAME.search(str(c)) and _unique(chunk[c])]
    with quiet():
        for name in chunk.columns:
            s = chunk[name]
            col = ColumnPart(dtypes={str(s.dtype)})
            notna = s.notna()
            col.n_missing = int((~notna).sum())
            values = _numeric_values(s)
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
            part.columns[name] = col
        if hash_rows and row_hash is not None:
            part.row_hashes = [row_hash]
            part.n_hashed = len(chunk)
            if ids and key_hash is not None:
                part.key_hashes, part.id_columns = [key_hash], ids
        else:
            part.row_hashes = None
    rng = np.random.default_rng([seed, chunk_id])
    keys = rng.random(len(chunk))
    keep = np.arange(len(chunk)) if max_key is None else np.flatnonzero(keys <= max_key)
    if len(keep) > k:
        keep = keep[np.argpartition(keys[keep], k)[:k]]
    part.sample = chunk.iloc[keep].assign(_pos=(np.int64(chunk_id) << 32) + keep.astype("int64"))
    part.keys = keys[keep]
    return part


def _merge_column(a: ColumnPart, b: ColumnPart) -> ColumnPart:
    out = ColumnPart(n_missing=a.n_missing + b.n_missing, numeric=a.numeric and b.numeric,
                     datetime=a.datetime and b.datetime, dtypes=a.dtypes | b.dtypes)
    if out.numeric:
        n = a.count + b.count
        out.count = n
        if n:
            delta = b.mean - a.mean
            out.mean = a.mean + delta * b.count / n
            out.m2 = a.m2 + b.m2 + delta * delta * a.count * b.count / n
        out.n_zeros, out.n_negative, out.n_inf = a.n_zeros + b.n_zeros, a.n_negative + b.n_negative, a.n_inf + b.n_inf
    if out.numeric or out.datetime:
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


def merge(a: Part, b: Part, k: int, max_hashed_rows: int) -> Part:
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
    else:
        out.row_hashes = None
    return out


def _distinct(hashes: np.ndarray) -> tuple[int, bool]:
    """(distinct count, exact?) from the k smallest hashes."""
    if len(hashes) < KMV_K:
        return len(hashes), True
    kth = float(hashes[KMV_K - 1]) + 1.0
    return int(round((KMV_K - 1) * _HASH_SPACE / kth)), False


def finalize(part: Part, n_chunks: int = 0, engine: str = "") -> FullScan:
    rows = []
    n = part.n_rows
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
        elif c.datetime and c.min is not None:
            row.update(min=c.min, max=c.max)
        if c.counts is not None and len(c.counts):
            top = c.counts.sort_values(ascending=False, kind="stable")
            row.update(top_value=top.index[0], top_count=int(top.iloc[0]))
        rows.append(row)
    columns = pd.DataFrame(rows).set_index("column") if rows else pd.DataFrame()
    dup = None
    if part.row_hashes is not None:
        hashes = np.concatenate(part.row_hashes) if part.row_hashes else np.empty(0, dtype="uint64")
        hashes = np.sort(hashes)
        n_dup = int((hashes[1:] == hashes[:-1]).sum())  # rows equal to an earlier row
        dup = {"count": n_dup, "pct": 100.0 * n_dup / n if n else 0.0}
        if part.key_hashes:
            keys = np.sort(np.concatenate(part.key_hashes))
            n_key = int((keys[1:] == keys[:-1]).sum())
            dup.update(count_ignoring_ids=n_key, pct_ignoring_ids=100.0 * n_key / n if n else 0.0,
                       id_columns=part.id_columns)
    sample = part.sample if part.sample is not None else pd.DataFrame()
    if len(sample):
        sizes = np.zeros(max(part.chunk_rows) + 1, dtype="int64")
        sizes[list(part.chunk_rows)] = list(part.chunk_rows.values())
        offsets = np.concatenate([[0], np.cumsum(sizes)[:-1]])
        pos = sample["_pos"].to_numpy(dtype="int64")
        rownum = offsets[pos >> 32] + (pos & 0xFFFFFFFF)
        order = np.argsort(rownum, kind="stable")
        sample = sample.drop(columns="_pos").iloc[order].set_axis(pd.Index(rownum[order]), axis=0)
    return FullScan(n_rows=n, columns=columns, duplicate_rows=dup, sample=sample, n_chunks=n_chunks, engine=engine)


def scan_chunks(chunks: Iterable[pd.DataFrame], k: int, seed: int = 42, max_hashed_rows: int = 30_000_000,
                engine: str = "pandas") -> FullScan:
    """Stream ``chunks`` (pandas DataFrames) one after the other through :func:`reduce_chunk`."""
    total: Part | None = None
    n_seen = 0
    n_chunks = 0
    for i, chunk in enumerate(chunks):
        part = reduce_chunk(chunk, i, k, seed, hash_rows=n_seen + len(chunk) <= max_hashed_rows)
        n_seen += len(chunk)
        total = part if total is None else merge(total, part, k, max_hashed_rows)
        n_chunks += 1
    return finalize(total or Part(), n_chunks, engine)
