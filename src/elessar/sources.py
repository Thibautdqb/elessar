"""Turn anything tabular into a pandas DataFrame to profile, streaming what does not fit.

Accepted: pandas DataFrame / Series, dict, list of records, NumPy arrays, polars, pyarrow,
dask, any object implementing the DataFrame interchange protocol, and paths (local or URL,
a glob pattern or a folder of files): CSV / TSV / TXT (delimiter and encoding sniffed), JSON,
JSON lines, Parquet, Feather / Arrow, ORC, Excel / ODS, pickle, HDF5, SAS, Stata, SPSS, SQLite,
XML, compressed or not.

A table with more than ``StoneConfig.max_rows`` rows, or too big for the available memory, is
never loaded whole: it is streamed once in chunks (see :mod:`elessar.scan`), in parallel with
dask when installed, to get exact counts and a uniform random sample the profile runs on.
"""

from __future__ import annotations

import csv
import glob
import io
import os
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator
from urllib.parse import urlparse

import numpy as np
import pandas as pd

from ._utils import fmt_bytes, quiet
from .config import StoneConfig
from .parallel import TaskError, n_workers, run_tasks, shared
from .scan import FullScan, Part, finalize, merge, reduce_chunk, scan_chunks

COMPRESSIONS = (".gz", ".bz2", ".zip", ".xz", ".zst", ".tar")
FORMATS = {
    ".csv": "csv", ".tsv": "csv", ".tab": "csv", ".txt": "csv", ".dat": "csv", ".psv": "csv", ".data": "csv",
    ".json": "json", ".jsonl": "jsonl", ".ndjson": "jsonl",
    ".parquet": "parquet", ".pq": "parquet", ".parq": "parquet",
    ".feather": "feather", ".arrow": "feather", ".ipc": "feather",
    ".orc": "orc",
    ".xlsx": "excel", ".xlsm": "excel", ".xls": "excel", ".xlsb": "excel", ".ods": "excel",
    ".pkl": "pickle", ".pickle": "pickle",
    ".h5": "hdf", ".hdf": "hdf", ".hdf5": "hdf",
    ".sas7bdat": "sas", ".xpt": "sas",
    ".dta": "stata",
    ".sav": "spss", ".zsav": "spss",
    ".sqlite": "sqlite", ".sqlite3": "sqlite", ".db": "sqlite",
    ".xml": "xml",
}
STREAMABLE = {"csv", "jsonl", "parquet", "sas", "stata", "sqlite"}
EXTRAS = {"parquet": "pyarrow", "feather": "pyarrow", "orc": "pyarrow", "excel": "openpyxl", "hdf": "tables",
          "spss": "pyreadstat", "xml": "lxml"}
_DEFAULT_SEP = {".tsv": "\t", ".tab": "\t", ".psv": "|"}


@dataclass
class Source:
    """A table ready to profile.

    ``df`` holds every row, or a uniform random sample of them when ``full`` is set; ``full``
    then carries the exact figures over all rows. ``origin`` describes where it came from.
    """

    df: pd.DataFrame
    name: str | None = None
    origin: dict[str, Any] = field(default_factory=dict)
    full: FullScan | None = None

    @property
    def n_rows(self) -> int:
        return self.full.n_rows if self.full is not None else len(self.df)


class LoadError(ValueError):
    """The input could not be turned into a table (unknown format, missing optional package...)."""


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


def _is_url(text: str) -> bool:
    return urlparse(text).scheme in ("http", "https", "ftp", "s3", "gs", "gcs", "az", "abfs", "hf")


def _suffixes(path: str) -> list[str]:
    name = urlparse(path).path if _is_url(path) else path
    return [s.lower() for s in Path(name).suffixes]


def detect_format(path: str) -> str:
    suffixes = [s for s in _suffixes(path) if s not in COMPRESSIONS]
    if suffixes and suffixes[-1] in FORMATS:
        return FORMATS[suffixes[-1]]
    if not _is_url(path) and os.path.isfile(path):
        with open(path, "rb") as fh:
            head = fh.read(16)
        if head.startswith(b"PAR1"):
            return "parquet"
        if head.startswith(b"SQLite format 3"):
            return "sqlite"
        if head.startswith(b"ARROW1"):
            return "feather"
        if head.startswith(b"PK\x03\x04") and not _suffixes(path)[-1:] == [".zip"]:
            return "excel"
        if head.lstrip().startswith((b"{", b"[")):
            return "json"
    return "csv"


def _missing_extra(fmt: str, exc: Exception) -> LoadError:
    pkg = EXTRAS.get(fmt)
    hint = f" -> pip install {pkg}" if pkg else ""
    return LoadError(f"cannot read {fmt} data ({exc}){hint}")


def _head_bytes(path: str, n: int = 1 << 16) -> bytes:
    """First ``n`` decompressed bytes of a local or remote file."""
    try:
        from pandas.io.common import get_handle

        with get_handle(path, "rb", compression="infer", is_text=False) as handles:
            return handles.handle.read(n)
    except Exception:  # noqa: BLE001
        if _is_url(path):
            return b""
        with open(path, "rb") as fh:
            return fh.read(n)


def sniff_csv(path: str, sep: str | None = None, encoding: str | None = None) -> dict[str, Any]:
    """Delimiter and encoding of a delimited text file, from its first 64 kB."""
    head = _head_bytes(path)
    if encoding is None:
        for enc in ("utf-8-sig", "cp1252", "latin-1"):
            try:
                head.decode(enc)
            except UnicodeDecodeError:
                if enc == "utf-8-sig" and len(head) >= (1 << 16) - 4:
                    try:  # a multi-byte character cut by the 64 kB boundary is not a decoding error
                        head[:-4].decode(enc)
                    except UnicodeDecodeError:
                        continue
                    encoding = enc
                    break
                continue
            encoding = enc
            break
    text = head.decode(encoding or "utf-8", errors="replace")
    if sep is None:
        ext = next((s for s in reversed(_suffixes(path)) if s not in COMPRESSIONS), "")
        lines = [ln for ln in text.splitlines()[:50] if ln.strip()]
        if len(lines) > 1 and not text.endswith("\n"):
            lines = lines[:-1]  # the last line may be cut
        try:
            sep = csv.Sniffer().sniff("\n".join(lines), delimiters=",;\t|").delimiter
        except csv.Error:
            sep = _DEFAULT_SEP.get(ext)
        if sep is None:
            counts = {d: lines[0].count(d) for d in ",;\t|"} if lines else {}
            sep = max(counts, key=counts.get) if counts and max(counts.values()) else ","
    return {"sep": sep, "encoding": encoding or "utf-8"}


def has_header(path: str, sep: str, encoding: str) -> bool:
    """False when the first line is data: most numeric columns are 'named' by a number."""
    try:
        head = pd.read_csv(path, sep=sep, encoding=encoding, nrows=200)
    except Exception:  # noqa: BLE001 - let the real read report the problem
        return True
    numeric = [c for c in head.columns if pd.api.types.is_numeric_dtype(head[c].dtype)]
    named_by_number = [c for c in numeric if pd.notna(pd.to_numeric(str(c).strip(), errors="coerce"))]
    return not (len(named_by_number) >= 2 and len(named_by_number) >= 0.6 * len(numeric))


def _csv_kwargs(path: str, opts: dict[str, Any]) -> dict[str, Any]:
    sniffed = sniff_csv(path, opts.get("sep"), opts.get("encoding"))
    kw: dict[str, Any] = {"sep": sniffed["sep"], "encoding": sniffed["encoding"], "low_memory": False}
    if not has_header(path, kw["sep"], kw["encoding"]):
        kw["header"] = None
    if opts.get("columns"):
        kw["usecols"] = opts["columns"]
    return kw


def _read_csv(path: str, kw: dict[str, Any], **more: Any) -> Any:
    try:
        return pd.read_csv(path, **kw, **more)
    except UnicodeDecodeError:
        return pd.read_csv(path, **{**kw, "encoding": "latin-1"}, **more)
    except pd.errors.ParserError:  # ragged lines: keep what can be read, report the rest
        return pd.read_csv(path, **{k: v for k, v in kw.items() if k != "low_memory"}, engine="python",
                           on_bad_lines="warn", **more)


# ------------------------------------------------------------------------------------------- planning
def _expand(path: str) -> list[str]:
    """Files behind a path, glob pattern or folder (local), or the URL itself."""
    if _is_url(path):
        return [path]
    p = Path(path).expanduser()
    if p.is_file():
        return [str(p)]
    if p.is_dir():
        files = sorted(str(f) for f in p.rglob("*") if f.is_file() and not f.name.startswith((".", "_"))
                       and detect_format(str(f)) in FORMATS.values() and any(s in FORMATS for s in _suffixes(str(f))))
        if not files:
            raise LoadError(f"no readable table in folder {path}")
        fmts = {detect_format(f) for f in files}
        if len(fmts) > 1:  # keep the most common format
            main = max(fmts, key=lambda fm: sum(detect_format(f) == fm for f in files))
            files = [f for f in files if detect_format(f) == main]
        return files
    matches = sorted(glob.glob(str(p), recursive=True))
    files = [m for m in matches if os.path.isfile(m)]
    if files:
        return files
    raise LoadError(f"no such file: {path}")


def _estimate(files: list[str], fmt: str, opts: dict[str, Any], cfg: StoneConfig) -> dict[str, Any]:
    """Rows and in-memory bytes the files would take, from a small head (cheap)."""
    size = sum(os.path.getsize(f) for f in files if not _is_url(f) and os.path.exists(f))
    est: dict[str, Any] = {"size_bytes": size, "rows": None, "memory": None}
    first = files[0]
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
                head = (_read_csv(first, kw, nrows=min(lines, 20_000)) if fmt == "csv"
                        else pd.read_json(first, lines=True, nrows=min(lines, 20_000)))
                per_row = head.memory_usage(deep=True, index=False).sum() / max(len(head), 1)
                est["memory"] = int(per_row * est["rows"])
            except Exception:  # noqa: BLE001
                pass
        elif compressed and size:
            est["memory"] = size * 12  # typical text compression ratio x in-memory overhead
    elif fmt == "sqlite":
        try:
            with sqlite3.connect(f"file:{first}?mode=ro", uri=True) as con:
                table = opts.get("table") or _sqlite_table(con)
                est["rows"] = int(con.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])
        except Exception:  # noqa: BLE001
            pass
    return est


def _too_big(est: dict[str, Any], cfg: StoneConfig) -> bool:
    budget = cfg.memory_fraction * available_memory()
    rows, memory = est.get("rows"), est.get("memory")
    return bool((rows is not None and rows > cfg.max_rows) or (memory is not None and memory > budget))


# ------------------------------------------------------------------------------------------- readers
def _sqlite_table(con: sqlite3.Connection) -> str:
    names = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type IN ('table','view') "
                                        "AND name NOT LIKE 'sqlite_%' ORDER BY name")]
    if not names:
        raise LoadError("the SQLite database has no table")
    sizes = {}
    for n in names:
        try:
            sizes[n] = con.execute(f'SELECT COUNT(*) FROM "{n}"').fetchone()[0]
        except sqlite3.Error:
            sizes[n] = -1
    return max(names, key=lambda n: sizes[n])  # the biggest table


def _read_whole(path: str, fmt: str, opts: dict[str, Any], info: dict[str, Any]) -> pd.DataFrame:
    nrows = opts.get("nrows")
    try:
        if fmt == "csv":
            kw = _csv_kwargs(path, opts)
            info.update(sep=kw["sep"], encoding=kw["encoding"], headerless="header" in kw)
            return _read_csv(path, kw, nrows=nrows)
        if fmt == "json":
            try:
                return pd.read_json(path)
            except ValueError:
                return pd.read_json(path, lines=True)
        if fmt == "jsonl":
            return pd.read_json(path, lines=True, nrows=nrows)
        if fmt == "parquet":
            return pd.read_parquet(path, columns=opts.get("columns"))
        if fmt == "feather":
            return pd.read_feather(path, columns=opts.get("columns"))
        if fmt == "orc":
            return pd.read_orc(path, columns=opts.get("columns"))
        if fmt == "excel":
            return _read_excel(path, opts, info)
        if fmt == "pickle":
            obj = pd.read_pickle(path)
            return obj if isinstance(obj, pd.DataFrame) else _from_object(obj, opts, info)
        if fmt == "hdf":
            return pd.read_hdf(path, key=opts.get("table"))
        if fmt == "sas":
            return pd.read_sas(path)
        if fmt == "stata":
            return pd.read_stata(path)
        if fmt == "spss":
            return pd.read_spss(path)
        if fmt == "sqlite":
            with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as con:
                table = opts.get("table") or _sqlite_table(con)
                info["table"] = table
                return pd.read_sql_query(f'SELECT * FROM "{table}"', con)
        if fmt == "xml":
            try:
                return pd.read_xml(path)
            except ImportError:
                return pd.read_xml(path, parser="etree")
    except ImportError as exc:
        raise _missing_extra(fmt, exc) from exc
    raise LoadError(f"unsupported format {fmt!r}")


def _read_excel(path: str, opts: dict[str, Any], info: dict[str, Any]) -> pd.DataFrame:
    engine = None
    try:
        import python_calamine  # noqa: F401 - much faster than openpyxl when installed

        engine = "calamine"
    except ImportError:
        pass
    sheets = pd.read_excel(path, sheet_name=None if opts.get("sheet") is None else [opts["sheet"]], engine=engine,
                           nrows=opts.get("nrows"))
    info["sheets"] = list(sheets)
    name = opts.get("sheet")
    if name is None:  # the first sheet that has data
        name = next((n for n, d in sheets.items() if d.shape[0] and d.shape[1]), next(iter(sheets)))
    info["sheet"] = name
    return sheets[name]


def _chunk_iter(path: str, fmt: str, opts: dict[str, Any], cfg: StoneConfig, info: dict[str, Any]) -> Iterator[pd.DataFrame]:
    """Chunks of a streamable file, read one after the other."""
    size = cfg.chunk_rows
    if fmt == "csv":
        kw = _csv_kwargs(path, opts)
        info.update(sep=kw["sep"], encoding=kw["encoding"], headerless="header" in kw)
        kw.pop("low_memory")
        yield from _read_csv(path, kw, chunksize=size)
    elif fmt == "jsonl":
        yield from pd.read_json(path, lines=True, chunksize=size)
    elif fmt == "parquet":
        import pyarrow.parquet as pq

        for batch in pq.ParquetFile(path).iter_batches(batch_size=size, columns=opts.get("columns")):
            yield batch.to_pandas()
    elif fmt == "sas":
        with pd.read_sas(path, chunksize=size) as reader:
            yield from reader
    elif fmt == "stata":
        with pd.read_stata(path, chunksize=size) as reader:
            yield from reader
    elif fmt == "sqlite":
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as con:
            table = opts.get("table") or _sqlite_table(con)
            info["table"] = table
            yield from pd.read_sql_query(f'SELECT * FROM "{table}"', con, chunksize=size)
    else:
        raise LoadError(f"{fmt} cannot be streamed")


# ------------------------------------------------------------------------------------------- dask
def _dask_tasks(files: list[str], fmt: str, opts: dict[str, Any], cfg: StoneConfig,
                info: dict[str, Any]) -> list[Callable[[], pd.DataFrame]] | None:
    """Independent chunk loaders for dask, or None when this input cannot be split that way."""
    if fmt == "csv":
        if any(s in COMPRESSIONS for f in files for s in _suffixes(f)):
            return None  # compressed text cannot be split into byte ranges
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
        return pd.read_csv(io.BytesIO(data), **kw)


class _Compute:
    def __init__(self, delayed: Any) -> None:
        self.delayed = delayed

    def __call__(self) -> pd.DataFrame:
        return self.delayed.compute(scheduler="sync")


def _reduce_task(i: int) -> Part:
    st = shared()
    return reduce_chunk(st["tasks"][i](), i, st["k"], st["seed"], max_key=st["max_key"])


def scan_parallel(tasks: list[Callable[[], pd.DataFrame]], k: int, cfg: StoneConfig, engine: str = "dask",
                  est_rows: int | None = None) -> FullScan:
    """Reduce independent chunk loaders on every core, then merge the summaries.

    With ``est_rows`` (a rough row count), each chunk only sends back the sample candidates that
    can plausibly be among the ``k`` kept; if the estimate was too high to fill the sample, the
    pass runs again keeping every candidate, so the sample stays exactly uniform.
    """
    workers = n_workers(cfg, 1 << 62, len(tasks))
    max_key = min(1.0, 1.5 * k / est_rows + 6 * np.sqrt(k) / est_rows) if est_rows else None
    while True:
        state = {"tasks": tasks, "k": k, "seed": cfg.random_state, "max_key": max_key}
        parts = run_tasks(_reduce_task, range(len(tasks)), workers, state)
        failed = [p for p in parts if isinstance(p, TaskError)]
        if failed:
            raise LoadError(f"reading a chunk failed: {failed[0].error}")
        total = parts[0]
        for part in parts[1:]:
            total = merge(total, part, k, cfg.max_hashed_rows)
        n_sample = 0 if total.sample is None else len(total.sample)
        if max_key is None or max_key >= 1.0 or n_sample >= min(k, total.n_rows):
            return finalize(total, len(tasks), engine if workers == 1 else f"{engine}, {workers} processes")
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
    """A DataFrame bigger than ``max_rows``: exact pass + a sample taken from it (original dtypes and index)."""
    full = scan_parallel(_df_tasks(df, cfg.chunk_rows), cfg.max_rows, cfg, "in memory", len(df))
    sample = df.iloc[full.sample.index.to_numpy()]
    full.sample = pd.DataFrame(index=sample.index[:0])  # do not keep a second copy
    return Source(sample, origin={**info, "engine": full.engine}, full=full)


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
        raise LoadError("engine='dask' needs dask -> pip install 'elessar[big]'")
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
            full = scan_parallel(tasks, cfg.max_rows, cfg)
            src = _from_scan(full, None, info)
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
    if not full.sampled:  # it all fitted in the sample: profile every row, no approximation
        return Source(full.sample, name=name, origin=info)
    return Source(full.sample, name=name, origin=info, full=full)


def _load_path(path: str, cfg: StoneConfig, opts: dict[str, Any], engine: str) -> Source:
    files = _expand(path)
    fmt = detect_format(files[0])
    name = Path(urlparse(path).path).name if _is_url(path) else (Path(path).name or path)
    info: dict[str, Any] = {"kind": f"{fmt} file" if len(files) == 1 else f"{len(files)} {fmt} files",
                            "format": fmt, "path": path, "n_files": len(files)}
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
        full = None
        if tasks and len(tasks) > 1:
            try:
                full = scan_parallel(tasks, cfg.max_rows, cfg, how, est.get("rows"))
            except Exception:  # noqa: BLE001 - e.g. a quoted newline cut by a block boundary
                full = None
        if full is None:
            chunks = (chunk for f in files for chunk in _chunk_iter(f, fmt, opts, cfg, info))
            full = scan_chunks(chunks, cfg.max_rows, cfg.random_state, cfg.max_hashed_rows, "pandas chunks")
        return _from_scan(full, name, info)
    frames = [_read_whole(f, fmt, opts, info) for f in files]
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
