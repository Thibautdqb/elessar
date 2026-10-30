"""Turn anything tabular into a pandas DataFrame to profile, streaming what does not fit.

Accepted: pandas DataFrame / Series, dict, list of records, NumPy arrays, polars, pyarrow,
dask, any object implementing the DataFrame interchange protocol, and paths (local or URL,
a glob pattern or a folder of files): CSV / TSV / TXT (delimiter and encoding sniffed), JSON,
JSON lines, Parquet, Feather / Arrow, ORC, Excel / ODS, pickle, HDF5, SAS, Stata, SPSS, SQLite,
XML, compressed or not.

A table with more than ``StoneConfig.max_rows`` rows, or too big for the available memory, is
never loaded whole. Its column types are first decided progressively on as few rows as the
evidence allows (:mod:`elessar.progressive`); one full pass then reads every row in chunks, in
parallel (dask splits CSV files, Parquet splits by row group), guided by those types
(:mod:`elessar.scan`): exact figures and type checks over every row, a uniform random sample for
the row-level profile, and for time-indexed tables the exact per-period series and whole
entities of a panel. File formats themselves are read by :mod:`elessar.readers`.
"""

from __future__ import annotations

import io
import os
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

import numpy as np
import pandas as pd

from ._utils import fmt_bytes, quiet
from .config import StoneConfig
from .parallel import TaskError, n_spawn_workers, n_workers, run_spawned, run_tasks, shared
from .progressive import CsvProbe, FrameProbe, HeadProbe, ParquetProbe, Probe, TypePlan, progressive_types
from .readers import (COMPRESSIONS, STREAMABLE, LoadError, _chunk_iter, _csv_kwargs, _expand, _head_bytes,  # noqa: F401
                      _is_url, _read_csv, _read_whole, _sqlite_table, _suffixes, bad_lines, counting_bad_lines,
                      _is_web, READ_PEAK_PER_BYTE, arrow_batches, arrow_rows, complete_lines, detect_format, has_header, sniff_csv, sql_name, url_info)
from .scan import FullScan, Part, ScanPlan, finalize, merge, reduce_chunk, scan_chunks
from .timeseries import aggregation_unit, finest_unit


@dataclass
class Source:
    """A table ready to profile.

    ``df`` holds every row, or a uniform random sample of them when ``full`` is set; ``full``
    then carries the exact figures over all rows (and, for a time-indexed table, the exact series
    per period and whole entities of a panel), and ``plan`` the column types decided before the
    pass, progressively (:mod:`elessar.progressive`). ``origin`` describes where it came from.
    """

    df: pd.DataFrame
    name: str | None = None
    origin: dict[str, Any] = field(default_factory=dict)
    full: FullScan | None = None
    plan: TypePlan | None = None  # how the types of a streamed table were decided

    @property
    def n_rows(self) -> int:
        return self.full.n_rows if self.full is not None else len(self.df)


# ------------------------------------------------------------------------------------------- helpers
def has_dask() -> bool:
    try:
        import dask.dataframe  # noqa: F401
    except Exception:  # noqa: BLE001 - any import problem means "no dask"
        return False
    return True


def available_memory() -> int:
    """Bytes of RAM available now (best effort; 4 GiB when unknown)."""
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    try:
        return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError, AttributeError):
        return 4 * 1024**3


def hash_cap(cfg: StoneConfig) -> int:
    """Rows whose hashes fit for exact duplicate counting (16 bytes per row, a quarter of free memory)."""
    return int(cfg.max_hashed_rows) if cfg.max_hashed_rows else max(1_000_000, int(available_memory() * 0.25 / 16))


# ------------------------------------------------------------------------------------------- planning
def _estimate(files: list[str], fmt: str, opts: dict[str, Any], cfg: StoneConfig) -> dict[str, Any]:
    """Rows and in-memory bytes the files would take, from a small head (cheap)."""
    first = files[0]
    if _is_web(first):  # the size the server announces; the body is only streamed when reading
        sizes = [url_info(f)["size"] for f in files]
        size = None if None in sizes else sum(sizes)
    else:
        size = sum(os.path.getsize(f) for f in files if not _is_url(f) and os.path.exists(f))
    est: dict[str, Any] = {"size_bytes": size, "rows": None, "memory": None}
    if size is None and fmt in ("csv", "jsonl"):
        est["unknown_size"] = True  # read in chunks: safe whatever the size, and exact for a small file
    if fmt == "parquet":
        try:
            import pyarrow.parquet as pq

            meta = [pq.ParquetFile(f).metadata for f in files]
            est["rows"] = int(sum(m.num_rows for m in meta))
            head = next(pq.ParquetFile(first).iter_batches(batch_size=5_000)).to_pandas()
            per_row = head.memory_usage(deep=True, index=False).sum() / max(len(head), 1)
            est["memory"] = int(per_row * est["rows"])
        except Exception:  # noqa: BLE001 - leave unknown
            pass
    elif fmt in ("csv", "jsonl"):
        raw = _head_bytes(first, 1 << 20)
        lines = raw.count(b"\n")
        compressed = any(s in COMPRESSIONS for s in _suffixes(first))
        if lines > 1 and size and not compressed:
            est["rows"] = int(size / (len(raw) / lines))
            try:
                kw = _csv_kwargs(first, opts) if fmt == "csv" else {}
                part = io.BytesIO(complete_lines(raw)) if _is_web(first) else first  # never download it whole
                head = (_read_csv(part, kw, nrows=min(lines, 20_000)) if fmt == "csv"
                        else pd.read_json(part, lines=True, nrows=min(lines, 20_000)))
                per_row = head.memory_usage(deep=True, index=False).sum() / max(len(head), 1)
                est["memory"] = int(per_row * est["rows"])
            except Exception:  # noqa: BLE001
                pass
        elif compressed and size:
            est["memory"] = size * 12  # typical text compression ratio x in-memory overhead
    elif fmt in ("feather", "orc") and not _is_url(first):
        try:
            est["rows"] = sum(arrow_rows(f, fmt) for f in files)
            head = next(arrow_batches(first, fmt, opts.get("columns"))).slice(0, 5_000).to_pandas()
            est["memory"] = int(head.memory_usage(deep=True, index=False).sum() / max(len(head), 1) * est["rows"])
        except Exception:  # noqa: BLE001 - leave unknown
            pass
    elif fmt in READ_PEAK_PER_BYTE and size:
        est["read_peak"] = int(size * READ_PEAK_PER_BYTE[fmt])  # read whole: what reading it takes, at most
    elif fmt == "sqlite":
        try:
            with sqlite3.connect(f"file:{first}?mode=ro", uri=True) as con:
                table = _sqlite_table(con, opts.get("table"))
                est["rows"] = int(con.execute(f"SELECT COUNT(*) FROM {sql_name(table)}").fetchone()[0])
        except Exception:  # noqa: BLE001
            pass
    return est


def _too_big(est: dict[str, Any], cfg: StoneConfig) -> bool:
    budget = cfg.memory_fraction * available_memory()
    rows, memory = est.get("rows"), est.get("memory")
    if est.get("unknown_size"):
        return True
    return bool((rows is not None and rows > cfg.max_rows) or (memory is not None and memory > budget))


# ------------------------------------------------------------------------------------------- readers
# ------------------------------------------------------------------------------------------- dask
def _dask_tasks(files: list[str], fmt: str, opts: dict[str, Any], cfg: StoneConfig,
                info: dict[str, Any]) -> list[Callable[[], pd.DataFrame]] | None:
    """Independent chunk loaders for dask, or None when this input cannot be split that way."""
    if fmt == "csv":
        if any(s in COMPRESSIONS for f in files for s in _suffixes(f)) or any(_is_url(f) for f in files):
            return None  # compressed or remote text cannot be split into byte ranges here
        from dask.bytes import read_bytes

        kw = _csv_kwargs(files[0], opts)
        info.update(sep=kw["sep"], encoding=kw["encoding"], headerless="header" in kw)
        kw.pop("low_memory")
        enc = kw["encoding"]
        if "header" in kw:
            return None  # blocks would need the header line: stream instead
        total = sum(os.path.getsize(f) for f in files if os.path.exists(f))
        blocksize = int(min(128 << 20, max(8 << 20, total / (2 * (os.cpu_count() or 1)))))  # ~2 blocks per core
        _, blocks = read_bytes(files, delimiter=b"\n", blocksize=blocksize, sample=False)
        tasks: list[Callable[[], pd.DataFrame]] = []
        for path, file_blocks in zip(files, blocks):
            header = _head_bytes(path, 1 << 20).split(b"\n", 1)[0] + b"\n"
            for i, block in enumerate(file_blocks):
                tasks.append(_CsvBlock(block, None if i == 0 else header, kw, enc))
        return tasks
    return None


def _parquet_tasks(files: list[str], opts: dict[str, Any]) -> list[Callable[[], pd.DataFrame]] | None:
    """One loader per Parquet row group: they are independent, so every core can read one."""
    try:
        import pyarrow.parquet as pq

        tasks: list[Callable[[], pd.DataFrame]] = [_RowGroup(f, i, opts.get("columns"))
                                                   for f in files for i in range(pq.ParquetFile(f).num_row_groups)]
    except Exception:  # noqa: BLE001 - remote file, unreadable metadata: stream it instead
        return None
    return tasks


class _RowGroup:
    def __init__(self, path: str, index: int, columns: list[str] | None) -> None:
        self.path, self.index, self.columns = path, index, columns

    def __call__(self) -> pd.DataFrame:
        import pyarrow.parquet as pq

        return pq.ParquetFile(self.path).read_row_group(self.index, columns=self.columns).to_pandas()


class _CsvBlock:
    def __init__(self, block: Any, header: bytes | None, kw: dict[str, Any], encoding: str) -> None:
        self.block, self.header, self.kw, self.encoding = block, header, kw, encoding

    def __call__(self) -> pd.DataFrame:
        data = self.block.compute(scheduler="sync")
        if self.header is not None:
            data = self.header + data
        kw = {**self.kw, "encoding": "utf-8" if self.encoding == "utf-8-sig" and self.header is not None else self.encoding}
        # numbered lines would count from the start of the block, not of the file
        return counting_bad_lines(lambda: pd.read_csv(io.BytesIO(data), **kw, on_bad_lines="warn"), numbered=False)


class _Compute:
    def __init__(self, delayed: Any) -> None:
        self.delayed = delayed

    def __call__(self) -> pd.DataFrame:
        return self.delayed.compute(scheduler="sync")


def _reduce_task(i: int) -> Part:
    st = shared()
    return reduce_chunk(st["tasks"][i](), i, st["k"], st["seed"], max_key=st["max_key"], plan=st["plan"])


def _reduce_one(task: Callable[[], pd.DataFrame], i: int, k: int, seed: int, max_key: float | None,
                plan: ScanPlan | None) -> Part:
    """The same reduction for a spawned process: everything it needs travels with the task."""
    return reduce_chunk(task(), i, k, seed, max_key=max_key, plan=plan)


def scan_parallel(tasks: list[Callable[[], pd.DataFrame]], k: int, cfg: StoneConfig, engine: str = "dask",
                  est_rows: int | None = None, plan: ScanPlan | None = None) -> FullScan:
    """Reduce independent chunk loaders on every core, then merge the summaries.

    With ``est_rows`` (a rough row count), each chunk only sends back the sample candidates that
    can plausibly be among the ``k`` kept; if the estimate was too high to fill the sample, the
    pass runs again keeping every candidate, so the sample stays exactly uniform.
    """
    workers = n_workers(cfg, 1 << 62, len(tasks))
    spawned = n_spawn_workers(cfg, len(tasks)) if workers == 1 else 1
    max_key = min(1.0, 1.5 * k / est_rows + 6 * np.sqrt(k) / est_rows) if est_rows else None
    while True:
        if spawned > 1:  # no safe fork (macOS, Windows): self-contained chunks on fresh processes
            parts = run_spawned(_reduce_one, [(t, i, k, cfg.random_state, max_key, plan) for i, t in enumerate(tasks)],
                                spawned)
        else:
            state = {"tasks": tasks, "k": k, "seed": cfg.random_state, "max_key": max_key, "plan": plan}
            parts = run_tasks(_reduce_task, range(len(tasks)), workers, state)
        failed = [p for p in parts if isinstance(p, TaskError)]
        if failed:
            raise LoadError(f"reading a chunk failed: {failed[0].error}")
        total = parts[0]
        for part in parts[1:]:
            total = merge(total, part, k, hash_cap(cfg), plan.entity_budget if plan else 0)
        n_sample = 0 if total.sample is None else len(total.sample)
        if max_key is None or max_key >= 1.0 or n_sample >= min(k, total.n_rows):
            used = max(workers, spawned)
            return finalize(total, len(tasks), engine if used == 1 else f"{engine}, {used} processes", plan)
        max_key = None


# ------------------------------------------------------------------------------------------- objects
def _from_object(obj: Any, opts: dict[str, Any], info: dict[str, Any]) -> pd.DataFrame:
    """In-memory objects of other libraries -> pandas."""
    module = type(obj).__module__.split(".")[0]
    info.setdefault("kind", f"{module}.{type(obj).__name__}")
    if isinstance(obj, pd.DataFrame):
        return obj
    if isinstance(obj, pd.Series):
        return obj.to_frame(name="value" if obj.name is None else obj.name)
    if isinstance(obj, np.ndarray):
        if obj.dtype.names:
            return pd.DataFrame.from_records(obj)
        if obj.ndim == 1:
            return pd.DataFrame({"value": obj})
        return pd.DataFrame(obj.reshape(obj.shape[0], -1))
    if isinstance(obj, (dict, list, tuple)):
        try:
            return pd.DataFrame(obj)
        except ValueError:  # e.g. a dict of scalars
            return pd.DataFrame([obj]) if isinstance(obj, dict) else pd.DataFrame({"value": list(obj)})
    if module == "polars":
        if hasattr(obj, "collect"):
            obj = obj.collect()
        try:
            return obj.to_pandas()
        except Exception:  # noqa: BLE001 - polars without pyarrow
            return pd.DataFrame(obj.to_dict(as_series=False))
    if module == "pyarrow":
        if hasattr(obj, "read_all"):
            obj = obj.read_all()
        if hasattr(obj, "to_table"):
            obj = obj.to_table()
        return obj.to_pandas()
    for method in ("to_pandas", "toPandas", "to_df", "df"):
        fn = getattr(obj, method, None)
        if callable(fn):
            out = fn()
            if isinstance(out, pd.DataFrame):
                return out
    if hasattr(obj, "__dataframe__"):
        return pd.api.interchange.from_dataframe(obj)
    raise LoadError(f"cannot turn a {type(obj).__name__} into a table")


def _df_tasks(df: pd.DataFrame, size: int) -> list[Callable[[], pd.DataFrame]]:
    return [_Slice(df, i, size) for i in range(0, len(df), size)]


class _Slice:
    def __init__(self, df: pd.DataFrame, start: int, size: int) -> None:
        self.df, self.start, self.size = df, start, size

    def __call__(self) -> pd.DataFrame:
        return self.df.iloc[self.start:self.start + self.size]


def _scan_frame(df: pd.DataFrame, cfg: StoneConfig, info: dict[str, Any]) -> Source:
    """A DataFrame bigger than ``max_rows``: progressive typing, guided exact pass, and a sample taken
    from it (original dtypes and index)."""
    plan = progressive_types(FrameProbe(df), cfg)
    sp = scan_plan(plan, cfg)
    full = scan_parallel(_df_tasks(df, cfg.chunk_rows), cfg.max_rows, cfg, "in memory", len(df), sp)
    sample = df.iloc[full.sample.index.to_numpy()]
    if full.panel_rows is not None:
        full.panel_rows = df.iloc[full.panel_rows.index.to_numpy()]
    full.sample = pd.DataFrame(index=sample.index[:0])  # do not keep a second copy
    return Source(sample, origin={**info, "engine": full.engine}, full=full, plan=plan)


def scan_plan(plan: TypePlan, cfg: StoneConfig) -> ScanPlan:
    """What the full pass computes, from the types decided beforehand."""
    sp = ScanPlan(types=plan.types)
    ts = plan.timeseries
    if ts is None:
        return sp
    axis = ts["time_column"].split(" + ")
    numeric = [c for c, d in plan.types.items() if c not in axis and c != ts.get("entity")
               and d["semantic_type"] in ("numeric_continuous", "numeric_discrete", "numeric_string", "boolean")]
    sp.time_column, sp.value_columns = ts["time_column"], numeric[:12]
    if ts["kind"] == "event log":
        sp.time_unit = aggregation_unit(ts["start"], ts["end"])
    else:
        sp.time_unit = finest_unit(ts["frequency"], ts["start"], ts["end"])
    if ts["kind"] == "panel":
        sp.entity, sp.entity_budget = ts["entity"], cfg.max_rows
    return sp


def _streamed(probe: Probe, run_pass: Callable[[ScanPlan], FullScan], cfg: StoneConfig, info: dict[str, Any],
              name: str | None) -> Source:
    """Progressive typing on a probe, then the full pass guided by the decided types."""
    plan = progressive_types(probe, cfg)
    full = run_pass(scan_plan(plan, cfg))
    src = _from_scan(full, name, info)
    if src.full is not None:
        src.plan = plan
    return src


# ------------------------------------------------------------------------------------------- entry point
def load(
    source: Any,
    config: StoneConfig | None = None,
    *,
    sheet: str | int | None = None,
    table: str | None = None,
    sep: str | None = None,
    encoding: str | None = None,
    columns: list[str] | None = None,
    nrows: int | None = None,
    engine: str = "auto",
) -> Source:
    """Turn ``source`` (a path, URL, glob, folder or any table object) into a :class:`Source`.

    ``sheet``: Excel sheet (default: the first one with data). ``table``: SQLite table or HDF5
    key (default: the biggest table). ``sep`` / ``encoding``: override the sniffed CSV dialect.
    ``columns``: read only these. ``nrows``: read only the first rows. ``engine``: ``"auto"`` uses
    dask for big inputs when installed, ``"pandas"`` never does, ``"dask"`` requires it.
    """
    cfg = config or StoneConfig()
    started = time.perf_counter()
    if engine == "dask" and not has_dask():
        raise LoadError("engine='dask' needs dask -> pip install 'dask[dataframe]'")
    opts = {"sheet": sheet, "table": table, "sep": sep, "encoding": encoding, "columns": columns, "nrows": nrows}
    if isinstance(source, Source):
        return source
    if isinstance(source, (str, os.PathLike)):
        src = _load_path(os.fspath(source), cfg, opts, engine)
    else:
        info: dict[str, Any] = {}
        if type(source).__module__.split(".")[0] == "dask" and hasattr(source, "to_delayed"):
            info["kind"] = "dask DataFrame"
            tasks: list[Callable[[], pd.DataFrame]] = [_Compute(p) for p in source.to_delayed()]
            first = tasks[0]
            src = _streamed(HeadProbe(lambda n: first().head(n)),
                            lambda sp: scan_parallel(tasks, cfg.max_rows, cfg, "dask", None, sp), cfg, info, None)
        else:
            frame = _from_object(source, opts, info)
            if nrows is not None:
                frame = frame.head(nrows)
            if columns:
                frame = frame[columns]
            src = _scan_frame(frame, cfg, info) if len(frame) > cfg.max_rows else Source(frame, origin=info)
    src.origin["load_seconds"] = round(time.perf_counter() - started, 3)
    return src


def _from_scan(full: FullScan, name: str | None, info: dict[str, Any]) -> Source:
    info = {**info, "engine": full.engine, "chunks": full.n_chunks}
    if full.bad_lines["count"]:
        info["bad_lines"] = full.bad_lines
    if not full.sampled:  # it all fitted in the sample: profile every row, no approximation
        return Source(full.sample, name=name, origin=info)
    return Source(full.sample, name=name, origin=info, full=full)


def _load_path(path: str, cfg: StoneConfig, opts: dict[str, Any], engine: str) -> Source:
    files = _expand(path)
    fmt = detect_format(files[0])
    name = Path(urlparse(path).path).name if _is_url(path) else (Path(path).name or path)
    info: dict[str, Any] = {"kind": f"{fmt} file" if len(files) == 1 else f"{len(files)} {fmt} files",
                            "format": fmt, "path": path, "n_files": len(files)}
    if _is_web(files[0]) and fmt not in ("xml", "json") and url_info(files[0])["type"] == "text/html":
        raise LoadError(f"{path} returns a web page (text/html), not a table -> use the link to the raw file")
    est = _estimate(files, fmt, opts, cfg) if opts.get("nrows") is None else {"size_bytes": None}
    info["size_bytes"] = est.get("size_bytes")
    stream = fmt in STREAMABLE and opts.get("nrows") is None and _too_big(est, cfg)
    if stream:
        tasks, how = None, "dask"
        if fmt == "parquet" and engine != "pandas":
            tasks, how = _parquet_tasks(files, opts), "row groups"
        elif engine != "pandas" and has_dask():
            try:
                tasks = _dask_tasks(files, fmt, opts, cfg, info)
            except Exception:  # noqa: BLE001 - fall back to sequential chunks
                tasks = None

        def run_pass(sp: ScanPlan) -> FullScan:
            if tasks and len(tasks) > 1:
                try:
                    return scan_parallel(tasks, cfg.max_rows, cfg, how, est.get("rows"), sp)
                except Exception:  # noqa: BLE001 - e.g. a quoted newline cut by a block boundary
                    pass
            chunks = (chunk for f in files for chunk in _chunk_iter(f, fmt, opts, cfg, info))
            return scan_chunks(chunks, cfg.max_rows, cfg.random_state, hash_cap(cfg), "pandas chunks", sp)

        return _streamed(_probe(files, fmt, opts, cfg, info), run_pass, cfg, info, name)
    peak, free = est.get("read_peak"), available_memory()
    if peak and peak > free:  # this format is read whole: stop with advice rather than be killed by the system
        raise LoadError(f"reading this {fmt} file whole would take about {fmt_bytes(peak)} of memory, "
                        f"{fmt_bytes(free)} are available -> convert it to Parquet or CSV (read in chunks), "
                        "or split it")
    frames = [_read_whole(f, fmt, opts, info) for f in files]
    skipped = bad_lines(frames)
    if skipped["count"]:
        info["bad_lines"] = skipped
    with quiet():
        df = frames[0] if len(frames) == 1 else pd.concat(frames, ignore_index=True, sort=False)
    if opts.get("columns") and fmt not in ("csv", "parquet", "feather", "orc"):
        df = df[opts["columns"]]
    if opts.get("nrows") is not None:
        df = df.head(opts["nrows"])
    if len(df) > cfg.max_rows:
        src = _scan_frame(df, cfg, info)
        src.name = name
        return src
    return Source(df, name=name, origin=info)


def _probe(files: list[str], fmt: str, opts: dict[str, Any], cfg: StoneConfig, info: dict[str, Any]) -> Probe:
    """Random access when the format allows it (uncompressed local text, Parquet), else the head only."""
    try:
        if fmt == "parquet":
            return ParquetProbe(files, opts.get("columns"))
        if fmt == "csv" and len(files) == 1 and not _is_url(files[0]) and not any(
                x in COMPRESSIONS for x in _suffixes(files[0])):
            kw = _csv_kwargs(files[0], opts)
            return CsvProbe(files[0], kw)
    except Exception:  # noqa: BLE001 - fall back to reading the head
        pass
    return HeadProbe(lambda n: next(iter(_chunk_iter(files[0], fmt, {**opts}, cfg, info))).head(n))


def describe_origin(src: Source) -> str:
    """One line: where the table came from and how it was read."""
    o = src.origin
    parts = [o.get("kind", "DataFrame")]
    if o.get("size_bytes"):
        parts.append(fmt_bytes(o["size_bytes"]))
    if o.get("sheet") is not None:
        n_sheets = len(o.get("sheets") or [])
        parts.append(f"sheet {o['sheet']!r}" + (f" of {n_sheets}" if n_sheets > 1 else ""))
    if o.get("table"):
        parts.append(f"table {o['table']!r}")
    if o.get("sep") and o.get("format") == "csv":
        parts.append("delimiter " + {"\t": "tab", ",": "comma", ";": "semicolon", "|": "pipe"}.get(o["sep"], repr(o["sep"])))
    if o.get("headerless"):
        parts.append("no header row")
    if o.get("encoding") and o["encoding"] not in ("utf-8", "utf-8-sig"):
        parts.append(f"encoding {o['encoding']}")
    if src.full is not None:
        parts.append(f"streamed in {src.full.n_chunks} chunks ({src.full.engine})")
    if o.get("load_seconds") is not None:
        parts.append(f"read in {o['load_seconds']:.1f}s")
    return " · ".join(parts)


__all__ = ["LoadError", "Source", "describe_origin", "detect_format", "load", "sniff_csv"]
