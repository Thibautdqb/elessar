"""Reading tables from files: format detection, CSV dialect sniffing, one reader per format.

Formats are recognised by extension, or by content when the extension says nothing (Parquet,
SQLite and Arrow magic bytes, JSON). Delimited text has its delimiter, encoding and header row
sniffed from its first bytes. :mod:`elessar.sources` decides whether a file is read whole or
streamed.
"""

from __future__ import annotations

import bz2
import csv
import glob
import gzip
import io
import lzma
import os
import pickle
import re
import sqlite3
import warnings
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
STREAMABLE = {"csv", "jsonl", "parquet", "feather", "orc", "sas", "stata", "sqlite"}
# memory taken while reading a file that cannot be read in parts, per byte of file: measured with pandas 3.1 on a
# mixed table (numbers, repeated labels, unique strings); hdf and spss are not measured, set from their layout
READ_PEAK_PER_BYTE = {"excel": 15.0, "json": 10.0, "xml": 10.0, "pickle": 1.1, "hdf": 1.5, "spss": 3.0}
EXTRAS = {"parquet": "pyarrow", "feather": "pyarrow", "orc": "pyarrow", "excel": "openpyxl", "hdf": "tables",
          "spss": "pyreadstat", "xml": "lxml"}
_DEFAULT_SEP = {".tsv": "\t", ".tab": "\t", ".psv": "|"}


class LoadError(ValueError):
    """The input could not be turned into a table (unknown format, missing optional package...)."""

def _is_url(text: str) -> bool:
    return urlparse(text).scheme in ("http", "https", "ftp", "s3", "gs", "gcs", "az", "abfs", "hf")

def _is_web(path: str) -> bool:
    """http(s) and ftp: pandas downloads these whole into memory before reading a single row."""
    return urlparse(str(path)).scheme in ("http", "https", "ftp")

def _request(path: str, method: str = "GET", headers: dict[str, str] | None = None) -> Any:
    import urllib.request

    req = urllib.request.Request(path, method=method, headers={"User-Agent": "elessar", **(headers or {})})
    return urllib.request.urlopen(req, timeout=60)

def url_stream(path: str) -> Any:
    """A web file as a binary stream read while it downloads, decompressed as its name says; None when the
    compression needs the whole file (zip, tar) or a missing package (zstandard)."""
    resp = _request(path)
    sfx = _suffixes(path)
    if ".gz" in sfx:
        return gzip.GzipFile(fileobj=resp)
    if ".bz2" in sfx:
        return bz2.BZ2File(resp)
    if ".xz" in sfx:
        return lzma.LZMAFile(resp)
    if ".zst" in sfx:
        try:
            import zstandard

            return zstandard.ZstdDecompressor().stream_reader(resp)
        except ImportError:
            pass
    if any(s in sfx for s in (".zip", ".tar", ".zst")):
        resp.close()
        return None
    return resp

def url_info(path: str) -> dict[str, Any]:
    """Size in bytes (None when the server does not say) and content type of a web file, without its body."""
    import urllib.error

    def ask(method: str, headers: dict[str, str] | None = None) -> tuple[Any, str]:
        with _request(path, method, headers) as r:
            if method == "HEAD":
                return r.headers.get("Content-Length"), r.headers.get("Content-Type", "")
            total = (r.headers.get("Content-Range") or "").rpartition("/")[2] if r.status == 206 else None
            return total, r.headers.get("Content-Type", "")

    try:
        try:
            size, ctype = ask("HEAD")
        except urllib.error.HTTPError as exc:
            if exc.code in (404, 410):
                raise
            size, ctype = ask("GET", {"Range": "bytes=0-0"})  # HEAD refused: the total comes in Content-Range
    except urllib.error.HTTPError as exc:
        raise LoadError(f"cannot open {path}: HTTP {exc.code} {exc.reason}") from None
    except OSError:  # no answer to either: size unknown, the file is then read in chunks
        return {"size": None, "type": ""}
    return {"size": int(size) if size and str(size).isdigit() else None, "type": ctype.split(";")[0].strip().lower()}

def _web_or_path(path: str) -> Any:
    return (url_stream(path) if _is_web(path) else None) or path

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
    if _is_web(path):
        try:
            stream = url_stream(path)
        except OSError:
            return b""
        if stream is None:
            return b""
        with stream:
            return stream.read(n)
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

def complete_lines(raw: bytes) -> bytes:
    """The head of a file without its last, possibly cut, line."""
    return raw[:raw.rfind(b"\n") + 1] if b"\n" in raw else raw

def has_header(path: str, sep: str, encoding: str) -> bool:
    """False when the first line is data: most numeric columns are 'named' by a number."""
    try:
        head = pd.read_csv(io.BytesIO(complete_lines(_head_bytes(path, 1 << 20))) if _is_web(path) else path,
                           sep=sep, encoding=encoding, nrows=200)
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

BAD_LINE_EXAMPLES = 5


def counting_bad_lines(read: Any, numbered: bool = True) -> pd.DataFrame:
    """Run ``read()`` (a pandas reader called with ``on_bad_lines="warn"``) and record the lines it skipped
    in ``frame.attrs``: ``bad_lines`` (count) and ``bad_line_numbers`` (the first ones, 1-based in the file;
    left empty when ``numbered`` is false, for a block whose numbers would be relative to the block)."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        df = read()
    lines = [int(n) for w in caught if issubclass(w.category, pd.errors.ParserWarning)
             for n in re.findall(r"Skipping line (\d+)", str(w.message))]
    df.attrs["bad_lines"] = len(lines)
    df.attrs["bad_line_numbers"] = sorted(lines)[:BAD_LINE_EXAMPLES] if numbered else []
    return df


def bad_lines(frames: Any) -> dict[str, Any]:
    """Skipped lines of several frames read by :func:`counting_bad_lines`: ``{count, examples}``."""
    count, examples = 0, []
    for f in frames:
        count += f.attrs.get("bad_lines", 0)
        examples += f.attrs.get("bad_line_numbers", [])
    return {"count": count, "examples": sorted(examples)[:BAD_LINE_EXAMPLES]}


def _read_csv(path: str, kw: dict[str, Any], **more: Any) -> Any:
    """``pd.read_csv`` that skips the lines with more fields than the header and counts them (see
    :func:`counting_bad_lines`); with ``chunksize``, an iterator of such frames."""
    def read(args: dict[str, Any]) -> Any:
        try:
            return pd.read_csv(_web_or_path(path), **args, **more)
        except pd.errors.ParserError:  # e.g. an unclosed quote: the python engine is more forgiving
            return pd.read_csv(_web_or_path(path), **{k: v for k, v in args.items() if k != "low_memory"},
                               engine="python", **more)

    kw = {**kw, "on_bad_lines": "warn"}
    if more.get("chunksize"):
        return _counted_chunks(lambda: read(kw), lambda: read({**kw, "encoding": "latin-1"}))
    try:
        return counting_bad_lines(lambda: read(kw))
    except UnicodeDecodeError:
        return counting_bad_lines(lambda: read({**kw, "encoding": "latin-1"}))


def _counted_chunks(open_reader: Any, fallback: Any) -> Iterator[pd.DataFrame]:
    try:
        reader = open_reader()
    except UnicodeDecodeError:
        reader = fallback()
    with reader:
        while True:
            try:
                yield counting_bad_lines(lambda: next(reader))
            except StopIteration:
                return

def _expand(path: str) -> list[str]:
    """Files behind a path, glob pattern or folder (local), or the URL itself."""
    if _is_url(path):
        return [path]
    p = Path(path).expanduser()
    if p.is_file():
        return [str(p)]
    if p.is_dir():
        found = sorted(str(f) for f in p.rglob("*") if f.is_file() and not f.name.startswith((".", "_")))
        return _same_format(found, f"folder {path}")
    matches = sorted(glob.glob(str(p), recursive=True))
    files = [m for m in matches if os.path.isfile(m)]
    if files:
        return files if not glob.has_magic(str(p)) else _same_format(files, f"pattern {path}")
    raise LoadError(f"no such file: {path}")

def _same_format(found: list[str], where: str) -> list[str]:
    """The readable tables among several files, in their most common format. Pickles are left out: loading one
    runs any code it holds, so only a pickle named on its own is read."""
    files = [f for f in found if detect_format(f) in FORMATS.values() and any(s in FORMATS for s in _suffixes(f))]
    safe = [f for f in files if detect_format(f) != "pickle"]
    if not safe:
        if files:
            raise LoadError(f"only pickle files in {where}: loading a pickle runs the code it holds, so elessar reads "
                            "one only when its own path is given -> pass the file path if you trust it")
        raise LoadError(f"no readable table in {where}")
    fmts = {detect_format(f) for f in safe}
    if len(fmts) > 1:  # keep the most common format
        main = max(sorted(fmts), key=lambda fm: sum(detect_format(f) == fm for f in safe))
        safe = [f for f in safe if detect_format(f) == main]
    return safe

def sql_name(name: str) -> str:
    """A table name as an SQL identifier: quoted, with its own double quotes doubled."""
    return '"' + str(name).replace('"', '""') + '"'

# What a pickle of a pandas or NumPy table may ask for (pandas 2.2 to 3.x, NumPy 1.x and 2.x): their classes and
# the functions that rebuild them, nothing that runs code. Anything else is refused.
_PICKLE_SAFE = frozenset(
    [("builtins", n) for n in ("bytearray", "slice", "complex", "frozenset", "set")]
    + [("__builtin__", n) for n in ("bytearray", "slice", "complex", "frozenset", "set")]
    + [("_codecs", "encode"), ("collections", "OrderedDict")]
    + [("datetime", n) for n in ("date", "datetime", "time", "timedelta", "timezone")]
    + [("decimal", "Decimal"), ("zoneinfo", "ZoneInfo"), ("pytz", "_p"), ("pytz", "_UTC")]
    + [(m, n) for m in ("numpy", "numpy.core.multiarray", "numpy._core.multiarray", "numpy.core.numeric",
                        "numpy._core.numeric") for n in ("dtype", "ndarray", "_reconstruct", "_frombuffer", "scalar")]
    + [(m, n) for m in ("pandas", "pandas.arrays", "pandas.core.frame", "pandas.core.series")
       for n in ("DataFrame", "Series", "Index", "RangeIndex", "MultiIndex", "DatetimeIndex", "TimedeltaIndex",
                 "PeriodIndex", "IntervalIndex", "CategoricalIndex", "Categorical", "CategoricalDtype",
                 "DatetimeTZDtype", "PeriodDtype", "IntervalDtype", "ArrowDtype", "StringDtype", "BooleanDtype",
                 "Int8Dtype", "Int16Dtype", "Int32Dtype", "Int64Dtype", "UInt8Dtype", "UInt16Dtype", "UInt32Dtype",
                 "UInt64Dtype", "Float32Dtype", "Float64Dtype", "NA", "NaT", "Timestamp", "Timedelta", "Period",
                 "Interval", "ArrowExtensionArray", "ArrowStringArray", "BooleanArray", "DatetimeArray",
                 "FloatingArray", "IntegerArray", "IntervalArray", "PeriodArray", "StringArray", "TimedeltaArray")]
    + [("pandas.core.indexes.base", "Index"), ("pandas.core.indexes.base", "_new_Index"),
       ("pandas.core.indexes.range", "RangeIndex"), ("pandas.core.indexes.multi", "MultiIndex"),
       ("pandas.core.indexes.category", "CategoricalIndex"), ("pandas.core.indexes.period", "PeriodIndex"),
       ("pandas.core.indexes.datetimes", "DatetimeIndex"), ("pandas.core.indexes.datetimes", "_new_DatetimeIndex"),
       ("pandas.core.indexes.timedeltas", "TimedeltaIndex"), ("pandas.core.indexes.timedeltas", "_new_TimedeltaIndex"),
       ("pandas.core.indexes.interval", "IntervalIndex"), ("pandas.core.indexes.interval", "_new_IntervalIndex"),
       ("pandas.core.internals.managers", "BlockManager"), ("pandas.core.internals.managers", "SingleBlockManager"),
       ("pandas._libs.internals", "_unpickle_block"), ("pandas._libs.arrays", "__pyx_unpickle_NDArrayBacked"),
       ("pandas._libs.missing", "NA"), ("pandas._libs.tslibs.timestamps", "_unpickle_timestamp"),
       ("pandas._libs.tslibs.timedeltas", "_timedelta_unpickle"), ("pandas._libs.tslibs.nattype", "__nat_unpickle"),
       ("pandas._libs.interval", "Interval"), ("pandas.core.arrays.categorical", "Categorical"),
       ("pandas.core.arrays.datetimes", "DatetimeArray"), ("pandas.core.arrays.timedeltas", "TimedeltaArray"),
       ("pandas.core.arrays.period", "PeriodArray"), ("pandas.core.arrays.interval", "IntervalArray"),
       ("pandas.core.arrays.boolean", "BooleanArray"), ("pandas.core.arrays.boolean", "BooleanDtype"),
       ("pandas.core.arrays.integer", "IntegerArray"), ("pandas.core.arrays.floating", "FloatingArray"),
       ("pandas.core.arrays.string_", "StringArray"), ("pandas.core.arrays.string_", "StringDtype"),
       ("pandas.core.arrays.string_arrow", "ArrowStringArray"), ("pandas.core.arrays.arrow.array", "ArrowExtensionArray")]
    + [("pandas.core.arrays.integer", f"{u}Int{b}Dtype") for u in ("", "U") for b in (8, 16, 32, 64)]
    + [("pandas.core.arrays.floating", f"Float{b}Dtype") for b in (32, 64)]
    + [("pandas.core.dtypes.dtypes", n) for n in ("CategoricalDtype", "DatetimeTZDtype", "PeriodDtype", "IntervalDtype",
                                                   "ArrowDtype")]
    + [("pyarrow.lib", n) for n in ("_restore_array", "py_buffer", "type_for_alias")]
)


class _TableUnpickler(pickle.Unpickler):
    """Rebuilds pandas and NumPy objects only (the "restricting globals" recipe of the pickle documentation)."""

    def find_class(self, module: str, name: str) -> Any:
        if (module, name) in (("builtins", "getattr"), ("__builtin__", "getattr")):
            return _pickle_getattr
        if (module, name) in _PICKLE_SAFE:
            return super().find_class(module, name)
        if module == "pandas._libs.tslibs.offsets":  # the frequency of a period or date index (MonthEnd...)
            found = super().find_class(module, name)
            if isinstance(found, type) and issubclass(found, pd.DateOffset):
                return found
        raise pickle.UnpicklingError(f"{module}.{name}")


def _pickle_getattr(obj: Any, name: str) -> Any:
    from zoneinfo import ZoneInfo

    if obj is ZoneInfo and name == "_unpickle":  # how a ZoneInfo time zone rebuilds itself
        return ZoneInfo._unpickle
    raise pickle.UnpicklingError(f"getattr({getattr(obj, '__name__', type(obj).__name__)}, {name!r})")


def _check_arrow(obj: Any) -> None:
    """pyarrow rebuilds its arrays from raw buffers without checking them: validate every one of them in full."""
    frames = [obj] if isinstance(obj, pd.DataFrame) else [obj.to_frame()] if isinstance(obj, pd.Series) else []
    for df in frames:
        parts = [df.iloc[:, i] for i in range(df.shape[1])]
        for idx in (df.index, df.columns):
            parts += [idx.get_level_values(i) for i in range(idx.nlevels)]
        for part in parts:
            arr = getattr(getattr(part, "array", None), "_pa_array", None)
            if arr is not None:
                arr.validate(full=True)


def read_remote_pickle(path: str) -> Any:
    """A pickle fetched from a URL, rebuilt only if it holds pandas or NumPy objects: a pickle runs any code it
    holds, and a URL is not a file one has looked at."""
    from pandas.io.common import get_handle

    stream = url_stream(path) if _is_web(path) else None
    if stream is None and any(s in _suffixes(path) for s in (".zip", ".tar")):
        raise LoadError(f"{path}: a zipped pickle is not read from a URL -> download it, check it, pass its path")
    try:
        if stream is not None:
            with stream:
                data = stream.read()
        else:
            with get_handle(path, "rb", compression="infer", is_text=False) as handles:
                data = handles.handle.read()
        obj = _TableUnpickler(io.BytesIO(data)).load()
        _check_arrow(obj)
    except ImportError as exc:  # e.g. time zones pickled by pandas 2 with pytz
        raise LoadError(f"{path}: this pickle needs {exc.name or exc}, which is not installed -> pip install "
                        f"{exc.name or 'it'}") from None
    except (pickle.UnpicklingError, ValueError, TypeError, AttributeError, EOFError) as exc:
        raise LoadError(f"{path}: this pickle holds more than pandas or NumPy data ({exc}); a pickle can run code, so "
                        "elessar does not load it from a URL -> download it, check where it comes from, and pass "
                        "its path") from None
    return obj


def _sqlite_table(con: sqlite3.Connection, wanted: str | None = None) -> str:
    """``wanted`` when the database has it, else the biggest table."""
    names = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type IN ('table','view') "
                                        "AND name NOT LIKE 'sqlite_%' ORDER BY name")]
    if wanted is not None:
        if wanted not in names:
            raise LoadError(f"no table {wanted!r} in the SQLite database (tables: {', '.join(map(repr, names))})")
        return wanted
    if not names:
        raise LoadError("the SQLite database has no table")
    sizes = {}
    for n in names:
        try:
            sizes[n] = con.execute(f"SELECT COUNT(*) FROM {sql_name(n)}").fetchone()[0]
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
            return pd.read_json(_web_or_path(path), lines=True, nrows=nrows)
        if fmt == "parquet":
            return pd.read_parquet(path, columns=opts.get("columns"))
        if fmt == "feather":
            return pd.read_feather(path, columns=opts.get("columns"))
        if fmt == "orc":
            return pd.read_orc(path, columns=opts.get("columns"))
        if fmt == "excel":
            return _read_excel(path, opts, info)
        if fmt == "pickle":
            # a pickle named by its local path is trusted like pd.read_pickle; one fetched from a URL is rebuilt
            # by a restricted unpickler that only knows how to rebuild pandas and NumPy objects
            obj = read_remote_pickle(path) if _is_url(path) else pd.read_pickle(path)
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
                table = _sqlite_table(con, opts.get("table"))
                info["table"] = table
                return pd.read_sql_query(f"SELECT * FROM {sql_name(table)}", con)
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

def arrow_batches(path: str, fmt: str, columns: list[str] | None = None) -> Iterator[Any]:
    """The record batches (feather) or stripes (ORC) of a file, one at a time."""
    import pyarrow as pa

    if fmt == "orc":
        import pyarrow.orc as orc

        f = orc.ORCFile(path)
        for i in range(f.nstripes):
            yield f.read_stripe(i, columns=columns)
        return
    source = pa.memory_map(path)
    try:
        reader = pa.ipc.open_file(source)
        batches = (reader.get_batch(i) for i in range(reader.num_record_batches))
    except pa.ArrowInvalid:  # the stream variant of the format
        batches = iter(pa.ipc.open_stream(pa.memory_map(path)))
    for b in batches:
        yield b.select(columns) if columns else b

def arrow_rows(path: str, fmt: str) -> int:
    """Rows of a feather or ORC file, from its metadata."""
    if fmt == "orc":
        import pyarrow.orc as orc

        return int(orc.ORCFile(path).nrows)
    return int(sum(b.num_rows for b in arrow_batches(path, fmt)))  # memory-mapped: batches are not read

def _chunk_iter(path: str, fmt: str, opts: dict[str, Any], cfg: StoneConfig, info: dict[str, Any]) -> Iterator[pd.DataFrame]:
    """Chunks of a streamable file, read one after the other."""
    size = cfg.chunk_rows
    if fmt == "csv":
        kw = _csv_kwargs(path, opts)
        info.update(sep=kw["sep"], encoding=kw["encoding"], headerless="header" in kw)
        kw.pop("low_memory")
        yield from _read_csv(path, kw, chunksize=size)
    elif fmt == "jsonl":
        yield from pd.read_json(_web_or_path(path), lines=True, chunksize=size)
    elif fmt == "parquet":
        import pyarrow.parquet as pq

        for batch in pq.ParquetFile(path).iter_batches(batch_size=size, columns=opts.get("columns")):
            yield batch.to_pandas()
    elif fmt in ("feather", "orc"):
        for batch in arrow_batches(path, fmt, opts.get("columns")):
            for start in range(0, batch.num_rows, size):
                yield batch.slice(start, size).to_pandas()
    elif fmt == "sas":
        with pd.read_sas(path, chunksize=size) as reader:
            yield from reader
    elif fmt == "stata":
        with pd.read_stata(path, chunksize=size) as reader:
            yield from reader
    elif fmt == "sqlite":
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as con:
            table = _sqlite_table(con, opts.get("table"))
            info["table"] = table
            yield from pd.read_sql_query(f"SELECT * FROM {sql_name(table)}", con, chunksize=size)
    else:
        raise LoadError(f"{fmt} cannot be streamed")


__all__ = ["COMPRESSIONS", "FORMATS", "LoadError", "STREAMABLE", "detect_format", "has_header", "sniff_csv"]
