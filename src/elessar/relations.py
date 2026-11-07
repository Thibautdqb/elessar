"""Several tables at once: keys between them, checked exactly, and dates that contradict a key.

``see_relations({"account": accounts, "loan": loans, "trans": "trans.csv"}, keys=["loan.account_id ->
account.account_id"], dates={"account": "date", "loan": "date"})`` checks, for every key:

- that the parent column is a key (unique) and how many children each parent has (1:1 or 1:n);
- the orphans: child values with no parent row, and children without a key;
- the parents with no child;
- with a date on both sides, the children dated before their parent (a loan granted before its
  account was opened): an event cannot precede the row it depends on.

Keys that are not declared are suggested among integer label columns (identifiers, categories; never
measurements): a child column whose distinct values fall in a unique parent column far more often than
the same values shifted by 1, 2, ... 199 within the parent's range do (identifiers of different tables are
often dense in the same low range: a shift keeps that overlap and breaks a real key). A range the parent fills densely (77
districts numbered 1 to 77) cannot be told from chance; neither can text keys (no chance model fits them).
Those are suggested only by the naming convention: the same specific column name in both tables
(``orders.customer_id`` and ``customers.customer_id``, never a generic ``id`` or ``code``), unique on the parent
side, with most child values found there. The report says the evidence is the name, not a test.
"""

from __future__ import annotations

import re
import shutil
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterator, TextIO

import numpy as np
import pandas as pd

from ._utils import quiet, truncate
from .config import StoneConfig
from .dates import parse_dates

_ARROW = re.compile(r"^\s*([^.\s]+)\.(.+?)\s*->\s*([^.\s]+)\.(.+?)\s*$")


@dataclass
class Relations:
    """What :func:`see_relations` found.

    Attributes:
        keys: one row per key (declared or suggested): child table and column, parent table and column,
            whether the parent column is unique, orphans, children without a key, parents without a
            child, children per parent, cardinality, children dated before their parent.
        alerts: ``{severity, column, message}`` (``column`` is ``"table.column"``).
        exact: table -> whether keys were looked for on every row (False: on a sample of a table too big for
            memory; the checks of every key are always exact).
    """

    keys: pd.DataFrame
    alerts: list[dict[str, Any]] = field(default_factory=list)
    exact: dict[str, bool] = field(default_factory=dict)

    def render(self, width: int | None = None) -> str:
        w = width or min(max(shutil.get_terminal_size((120, 40)).columns, 60), 160)
        out = ["═" * w, f" elessar relations  ·  {len(self.exact)} table{'s' * (len(self.exact) != 1)}, "
               f"{len(self.keys)} key{'s' * (len(self.keys) != 1)}", "═" * w, ""]
        for r in self.keys.itertuples(index=False):
            out.append(truncate(f"▸ {r.child}.{r.child_column} -> {r.parent}.{r.parent_column}  ·  {r.origin}"
                                f"  ·  {r.cardinality}", w))
            parts = [f"parent unique: {'yes' if r.parent_unique else f'no ({r.n_parent_duplicates:,} repeated values)'}",
                     f"orphans {r.n_orphans:,} ({r.pct_orphans:.1f}%)", f"children without key {r.n_child_missing:,}",
                     f"parents without child {r.n_parents_without_child:,} of {r.n_parents:,}",
                     f"children per parent {r.children_min:g} / {r.children_median:g} / {r.children_max:g} (min / median / max)"]
            if r.n_child_before_parent is not None and pd.notna(r.n_child_before_parent):
                parts.append(f"children dated before their parent {int(r.n_child_before_parent):,}")
            if r.origin == "suggested" and getattr(r, "by_name", False) is True:
                parts.append(f"{r.pct_included:.1f}% of its values in {r.parent}.{r.parent_column} (same column name)")
            elif r.origin == "suggested":
                parts.append(f"{r.pct_included:.1f}% of its values in {r.parent}.{r.parent_column} vs "
                             f"{100 * r.chance_rate:.1f}% once shifted (p {r.p_value:.3f})")
            out.append(truncate("    " + " · ".join(parts), w))
        if self.alerts:
            out += ["", "ALERTS", "──────"]
            out += [truncate(f"  {a['severity'].upper():<7} {a['column']}  {a['message']}", w) for a in self.alerts]
        not_exact = [t for t, e in self.exact.items() if not e]
        if not_exact:
            out += ["", truncate(f"  keys looked for on a sample of: {', '.join(not_exact)} (checks over every row)", w)]
        return "\n".join(out)

    def print(self, file: TextIO | None = None, width: int | None = None) -> None:
        print(self.render(width), file=file or sys.stdout)

    def __str__(self) -> str:
        return self.render()


def _parse_key(spec: Any) -> tuple[str, str, str, str]:
    if isinstance(spec, (tuple, list)) and len(spec) == 4:
        return tuple(map(str, spec))  # type: ignore[return-value]
    m = _ARROW.match(str(spec))
    if not m:
        raise ValueError(f"a key is 'child.column -> parent.column' or (child, column, parent, column), got {spec!r}")
    return m.group(1), m.group(2), m.group(3), m.group(4)


def _load(data: Any, cols: list[str] | None, cfg: StoneConfig, load_options: dict[str, Any],
          full: bool) -> tuple[pd.DataFrame, bool]:
    """``full``: these columns, every row; else every column, a uniform sample of a table too big for memory."""
    from .sources import load

    if isinstance(data, pd.DataFrame):
        return (data[cols] if cols else data), True
    src = load(data, replace(cfg, max_rows=10**12) if full else cfg, columns=cols, **load_options)
    return src.df, src.full is None


def _chunks(data: Any, cols: list[str], cfg: StoneConfig, load_options: dict[str, Any],
            rows: int = 2_000_000) -> Iterator[pd.DataFrame]:
    """The columns ``cols`` of every row, block by block: a child table of any size never sits whole in memory
    (CSV-like files read in blocks of ``rows``, Parquet by row group; other sources are read whole)."""
    if isinstance(data, (str, Path)) and not str(data).startswith(("http://", "https://")):
        from .readers import _csv_kwargs, detect_format

        path = str(data)
        fmt = detect_format(path)
        if fmt == "csv":
            kw = _csv_kwargs(path, {**load_options, "columns": cols})
            kw.pop("low_memory", None)
            yield from pd.read_csv(path, chunksize=rows, **kw)
            return
        if fmt == "parquet":
            import pyarrow.parquet as pq

            for batch in pq.ParquetFile(path).iter_batches(batch_size=rows, columns=cols):
                yield batch.to_pandas()
            return
    yield _load(data, cols, cfg, load_options, full=True)[0]


def _check(child: Any, ccol: str, parent: pd.DataFrame, pcol: str, cdate_col: str | None,
           pdate: pd.Series | None, cfg: StoneConfig, load_options: dict[str, Any]) -> dict[str, Any]:
    """Every check of one key, the child read block by block (exact over every row)."""
    from .dates import detect

    p = parent[pcol]
    p_known = p.dropna()
    n_dup = int(p_known.duplicated().sum())
    parents = pd.Index(p_known.unique())
    parent_date = None
    if pdate is not None and cdate_col is not None and n_dup == 0:
        parent_date = pd.Series(pdate.to_numpy(), index=p.to_numpy())
        parent_date = parent_date[parent_date.index.notna()]
    counts = pd.Series(0, index=parents, dtype="int64")
    n_rows = n_missing = n_orphans = n_early = 0
    orphan_examples: list[Any] = []
    early_examples: list[int] = []
    date_format = None
    cols = [ccol] + ([cdate_col] if parent_date is not None and cdate_col != ccol else [])
    for chunk in _chunks(child, cols, cfg, load_options):
        c = chunk[ccol]
        present = c.dropna()
        inside = present.isin(parents)
        n_rows += len(c)
        n_missing += int(c.isna().sum())
        n_orphans += int((~inside).sum())
        if len(orphan_examples) < 5:
            orphan_examples += [v for v in present[~inside].drop_duplicates().head(5).tolist() if v not in orphan_examples]
        counts = counts.add(present[inside].value_counts(), fill_value=0)
        if parent_date is not None:
            with quiet():
                date_format = date_format or detect(chunk[cdate_col], cdate_col)
                when_child = parse_dates(chunk[cdate_col], date_format)
            early = (when_child < c.map(parent_date)).fillna(False).to_numpy()
            n_early += int(early.sum())
            early_examples += [n_rows - len(c) + int(i) for i in np.flatnonzero(early)[: 5 - len(early_examples)]]
    out = {"parent_unique": n_dup == 0, "n_parent_duplicates": n_dup, "n_child_rows": n_rows,
           "n_child_missing": n_missing, "n_orphans": n_orphans,
           "pct_orphans": 100.0 * n_orphans / (n_rows - n_missing) if n_rows > n_missing else 0.0,
           "orphan_examples": orphan_examples[:5], "n_parents": int(len(parents)),
           "n_parents_without_child": int((counts == 0).sum()),
           "children_min": float(counts.min()) if len(counts) else 0.0,
           "children_median": float(counts.median()) if len(counts) else 0.0,
           "children_max": float(counts.max()) if len(counts) else 0.0}
    out["cardinality"] = "1:1" if out["children_max"] <= 1 else "1:n"
    out["n_child_before_parent"] = n_early if parent_date is not None else None
    out["child_before_parent_examples"] = early_examples
    return out


def _suggest(frames: dict[str, pd.DataFrame], declared: set[tuple[str, str, str, str]], cfg: StoneConfig,
             n_shifts: int = 199, max_values: int = 1_000_000) -> list[dict[str, Any]]:
    """Integer label columns whose values fall in a unique integer label column of another table beyond chance.

    Chance is measured on the data, not assumed: the same values shifted by s = 1..``n_shifts`` (circularly,
    within the parent's range) are looked up too. Identifiers of different tables are often dense in the same
    low range, so they overlap a lot without any link; a shift keeps that overlap, while it breaks a real key
    (which matches values exactly). A parent that fills its whole range keeps every shifted value: nothing can
    be told, nothing is suggested. The p-value ``(1 + #{shifted inclusion >= observed}) / (1 + #shifts)`` is
    adjusted over every pair tested (Benjamini-Hochberg).
    """
    from .quality import KEY_TYPES
    from .types import infer_type

    def integers(s: pd.Series) -> np.ndarray | None:
        if not (pd.api.types.is_integer_dtype(s.dtype) or pd.api.types.is_float_dtype(s.dtype)):
            return None
        v = s.dropna().to_numpy()
        if v.size == 0 or not np.all(np.isfinite(v)) or not np.all(v == np.round(v)):
            return None
        return np.unique(v.astype(np.int64))

    def is_label(s: pd.Series) -> bool:  # a key is a label (identifier, category), never a measurement
        with quiet():
            sub = s if len(s) <= 20_000 else s.sample(20_000, random_state=cfg.random_state)
            return infer_type(sub, cfg, sampled=len(s) > 20_000)["semantic_type"] in KEY_TYPES

    # a column already in a declared key references that table, or is the key others reference: nothing to guess
    engaged = {(t, c) for ct, cc, pt, pc in declared for t, c in ((ct, cc), (pt, pc))}
    values = {}
    for t, df in frames.items():
        for c in df.columns:
            v = integers(df[c])
            if v is not None and v.size >= 2 and is_label(df[c]):
                values[(t, c)] = v
    parents = {k: v for k, v in values.items() if frames[k[0]][k[1]].dropna().is_unique and v.size <= max_values}
    tests = []
    shifts = np.arange(1, n_shifts + 1)
    for (pt, pc), pv in parents.items():
        for (ct, cc), cv in values.items():
            if ct == pt or (ct, cc) in engaged or _points_away(ct, cc, pt, pc):
                continue
            observed = float(np.isin(cv, pv).mean())
            if observed <= 0.5:
                continue
            lo, span = int(pv.min()), int(pv.max() - pv.min() + 1)
            inside = (cv >= lo) & (cv < lo + span)
            steps = shifts[shifts < span]
            if steps.size == 0:
                continue
            # circular shifts within the parent's range: a parent filling its range keeps every shifted value
            shifted = np.array([np.isin(lo + (cv[inside] - lo + s) % span, pv).sum() / cv.size for s in steps])
            tests.append({"child": ct, "child_column": cc, "parent": pt, "parent_column": pc,
                          "pct_included": 100.0 * observed, "chance_rate": float(shifted.mean()),
                          "p_value": float((1 + np.sum(shifted >= observed)) / (1 + steps.size))})
    if not tests:
        return []
    from .correlations import bh_adjust

    adjusted = bh_adjust(np.array([t["p_value"] for t in tests]))
    keep = [dict(t, p_value=float(q)) for t, q in zip(tests, adjusted) if q < cfg.alpha]
    best: dict[tuple[str, str], dict[str, Any]] = {}  # one parent per child column: the most included
    for t in keep:
        key = (t["child"], t["child_column"])
        if key not in best or (t["pct_included"], -t["p_value"]) > (best[key]["pct_included"], -best[key]["p_value"]):
            best[key] = t
    return list(best.values())


_GENERIC_NAMES = {"id", "key", "code", "name", "index", "idx", "no", "num", "number", "pk", "uid", "uuid", "ref", "type",
                  "label", "value", "date", "year"}


def _stem(name: str) -> str:
    return re.sub(r"[^0-9a-z]+", "_", str(name).lower()).strip("_")


def _own_key(table: str, column: str) -> bool:
    """Is ``column`` named as the identifier of ``table`` itself (``disp.disp_id``, ``customers.customer_id``)?"""
    base = re.sub(r"_?(id|key|code|no|num|number)$", "", _stem(column))
    t = _stem(table)
    return bool(base) and (t.startswith(base) or base.startswith(t))


def _points_away(ct: str, cc: str, pt: str, pc: str) -> bool:
    """A table's own identifier is referenced, it does not reference: ``disp.disp_id`` is not a key to
    ``client.client_id`` (ids allocated in step look alike), nor ``client.client_id`` to ``disp.client_id`` (the
    reverse of the real key). Only an extension table sharing the parent's identifier (``customer_details.
    customer_id -> customers.customer_id``) keeps its own identifier as a key."""
    return _own_key(ct, cc) and not (_own_key(pt, pc) and _stem(cc) == _stem(pc))


def _suggest_by_name(frames: dict[str, pd.DataFrame], taken: set[tuple[str, str]]) -> list[dict[str, Any]]:
    """Keys proposed by the naming convention alone: the same specific column name in two tables, unique in one
    of them (the parent), with more than half of the other's distinct values found there."""
    columns: dict[str, list[tuple[str, str]]] = {}
    for t, df in frames.items():
        for c in df.columns:
            if _stem(c) and _stem(c) not in _GENERIC_NAMES:
                columns.setdefault(_stem(c), []).append((t, c))
    out = []
    for places in columns.values():
        for pt, pc in places:
            parent = frames[pt][pc].dropna()
            if parent.empty or not parent.is_unique:
                continue
            for ct, cc in places:
                if ct == pt or (ct, cc) in taken:
                    continue
                child = pd.Series(frames[ct][cc].dropna().unique())
                if child.empty:
                    continue
                if child.dtype != parent.dtype:  # 18.0 and 18, '18' and 18: compare as text of the integer if possible
                    child, values = child.astype(str).str.replace(r"\.0$", "", regex=True), \
                        parent.astype(str).str.replace(r"\.0$", "", regex=True)
                else:
                    values = parent
                share = float(child.isin(set(values)).mean())
                if share <= 0.5:
                    continue
                if _points_away(ct, cc, pt, pc):
                    continue
                if frames[ct][cc].dropna().is_unique:
                    if not _own_key(pt, pc):
                        continue  # both sides unique and the parent not named after its table: no direction
                    back = float(pd.Series(values.unique()).isin(set(child)).mean())
                    if back > share or (back == share and len(_stem(pt)) >= len(_stem(ct))):
                        continue  # 1:1: the parent holds every child value (or, on a tie, has the shorter name)
                out.append({"child": ct, "child_column": cc, "parent": pt, "parent_column": pc,
                            "pct_included": 100.0 * share, "chance_rate": np.nan, "p_value": np.nan, "by_name": True})
                taken.add((ct, cc))
    return out


def see_relations(tables: dict[str, Any], keys: list[Any] | None = None, dates: dict[str, str] | None = None,
                  config: StoneConfig | None = None, suggest: bool = True, verbose: bool = True,
                  **load_options: Any) -> Relations:
    """Check the keys between several tables, and the dates that contradict them.

    ``tables``: name -> DataFrame, or anything :func:`elessar.load` reads. Only the key and date columns
    are read for the checks, every row of them: a child table is read block by block (CSV, Parquet), so its
    size is never a limit; a parent table is read whole (its key and date columns). ``keys``: ``"child.column -> parent.column"`` strings or ``(child, column, parent, column)``
    tuples, checked exactly. ``dates``: table -> the column dating its rows (an opening date, an event
    date): a child dated before its parent is reported. With ``suggest``, integer keys that are not
    declared are proposed when chance cannot explain them (see the module docstring); they are checked
    like declared ones and marked ``suggested``. ``load_options`` (``sep``, ``encoding``, ...) apply to
    every file.
    """
    cfg = config or StoneConfig()
    dates = dict(dates or {})
    declared = [_parse_key(k) for k in (keys or [])]
    for ct, cc, pt, pc in declared:
        for t in (ct, pt):
            if t not in tables:
                raise KeyError(f"table {t!r} of key {ct}.{cc} -> {pt}.{pc} is not in tables ({', '.join(tables)})")
    unknown = set(dates) - set(tables)
    if unknown:
        raise KeyError(f"dates given for unknown tables: {', '.join(sorted(unknown))}")
    found = [dict(child=ct, child_column=cc, parent=pt, parent_column=pc, origin="declared") for ct, cc, pt, pc in declared]
    sampled: list[str] = []
    if suggest:  # every column may be a key: the whole tables, sampled when too big for memory
        candidates = {}
        for t, data in tables.items():
            candidates[t], whole = _load(data, None, cfg, load_options, full=False)
            if not whole:
                sampled.append(t)
        found += [dict(t, origin="suggested") for t in _suggest(candidates, set(declared), cfg)]
        taken = {(k["child"], k["child_column"]) for k in found} | {(k["parent"], k["parent_column"]) for k in found}
        found += [dict(t, origin="suggested") for t in _suggest_by_name(candidates, taken)]
    parents: dict[str, set[str]] = {}
    for k in found:
        parents.setdefault(k["parent"], set()).add(k["parent_column"])
    for t, c in dates.items():
        if any(k["parent"] == t for k in found):
            parents.setdefault(t, set()).add(c)
    # parents are read whole (their key and date columns); children are streamed in _check
    frames = {t: _load(tables[t], sorted(cols), cfg, load_options, full=True)[0] for t, cols in parents.items()}
    for k in found:
        if k["parent_column"] not in frames[k["parent"]].columns:
            raise KeyError(f"column {k['parent_column']!r} not in table {k['parent']!r}")
    parsed: dict[str, pd.Series] = {}
    with quiet():
        for t in frames:
            if t in dates:
                if dates[t] not in frames[t].columns:
                    raise KeyError(f"date column {dates[t]!r} not in table {t!r}")
                parsed[t] = parse_dates(frames[t][dates[t]])
    rows, alerts = [], []
    for k in found:
        ct, cc, pt, pc = k["child"], k["child_column"], k["parent"], k["parent_column"]
        try:
            checked = _check(tables[ct], cc, frames[pt], pc, dates.get(ct), parsed.get(pt), cfg, load_options)
        except (KeyError, ValueError) as exc:  # a column missing from the child file
            raise KeyError(f"key {ct}.{cc} -> {pt}.{pc}: {exc}") from exc
        r = {**k, **checked}
        rows.append(r)
        name = f"{ct}.{cc}"
        if not r["parent_unique"]:
            alerts.append({"severity": "high", "column": f"{pt}.{pc}", "message": f"not a key: {r['n_parent_duplicates']:,} "
                           f"repeated values, so {name} cannot point to a single row"})
        if r["n_orphans"]:
            alerts.append({"severity": "high" if r["origin"] == "declared" else "medium", "column": name,
                           "message": f"{r['n_orphans']:,} rows ({r['pct_orphans']:.1f}%) point to no row of {pt}.{pc}, "
                                      f"e.g. {r['orphan_examples'][:3]}"})
        if r["n_child_before_parent"]:
            alerts.append({"severity": "high", "column": name,
                           "message": f"{r['n_child_before_parent']:,} rows dated ({dates[ct]}) before their {pt} row "
                                      f"({dates[pt]}): an event cannot precede the row it depends on, e.g. rows "
                                      f"{r['child_before_parent_examples'][:3]}"})
        if r["origin"] == "suggested" and r.get("by_name"):
            alerts.append({"severity": "info", "column": name,
                           "message": f"probably a key to {pt}.{pc}: same column name and {r['pct_included']:.1f}% of its "
                                      "distinct values are there (a naming convention: chance cannot be measured on these "
                                      "values) -> declare it in keys= to confirm"})
        elif r["origin"] == "suggested":
            alerts.append({"severity": "info", "column": name,
                           "message": f"probably a key to {pt}.{pc}: {r['pct_included']:.1f}% of its values are there, "
                                      f"{100 * r['chance_rate']:.1f}% once its values are shifted (BH-adjusted p {r['p_value']:.3f})"
                                      " -> declare it in keys= to confirm"})
    for t in sampled:
        alerts.append({"severity": "info", "column": t, "message": "too big for memory: keys looked for on a sample of "
                                                                   "it (every key found is then checked on every row)"})
    out = Relations(keys=pd.DataFrame(rows), alerts=alerts, exact={t: t not in sampled for t in tables})
    if verbose:
        out.print()
    return out
