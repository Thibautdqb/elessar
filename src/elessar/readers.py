"""Reading tables from files: format detection, CSV dialect sniffing, one reader per format.

Formats are recognised by extension, or by content when the extension says nothing (Parquet,
SQLite and Arrow magic bytes, JSON). Delimited text has its delimiter, encoding and header row
sniffed from its first bytes. :mod:`elessar.sources` decides whether a file is read whole or
streamed.
"""

from __future__ import annotations

import csv
import glob
import os
import sqlite3
from pathlib import Path
from collections.abc import Iterator
from typing import Any
from urllib.parse import urlparse

import pandas as pd

from .config import StoneConfig

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


class LoadError(ValueError):
    """The input could not be turned into a table (unknown format, missing optional package...)."""

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
            if isinstance(obj, pd.DataFrame):
                return obj
            from .sources import _from_object  # a pickled polars / NumPy / ... object

            return _from_object(obj, opts, info)
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


__all__ = ["COMPRESSIONS", "FORMATS", "LoadError", "STREAMABLE", "detect_format", "has_header", "sniff_csv"]
