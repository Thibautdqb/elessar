from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from elessar import StoneConfig, load, see
from elessar.scan import KMV_K, scan_chunks
from elessar.sources import LoadError, detect_format, has_dask, sniff_csv


@pytest.fixture
def frame() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 5_000
    df = pd.DataFrame({
        "num": rng.normal(10, 2, n),
        "count": rng.poisson(3, n),
        "city": rng.choice(["Paris", "Lyon", "Nantes"], n),
        "id": np.arange(n),
    })
    df.loc[rng.random(n) < 0.1, "num"] = np.nan
    return pd.concat([df, df.iloc[:37]], ignore_index=True)  # 37 duplicate rows


def _has(module: str) -> bool:
    try:
        __import__(module)
    except ImportError:
        return False
    return True


WRITERS = {
    "t.csv": lambda d, p: d.to_csv(p, index=False),
    "t.tsv": lambda d, p: d.to_csv(p, sep="\t", index=False),
    "semi.txt": lambda d, p: d.to_csv(p, sep=";", index=False),
    "pipe.dat": lambda d, p: d.to_csv(p, sep="|", index=False),
    "t.csv.gz": lambda d, p: d.to_csv(p, index=False),
    "t.csv.zip": lambda d, p: d.to_csv(p, index=False),
    "latin.csv": lambda d, p: d.assign(city=d["city"].replace("Nantes", "Nîmes")).to_csv(p, index=False, encoding="cp1252"),
    "t.json": lambda d, p: d.to_json(p),
    "records.json": lambda d, p: d.to_json(p, orient="records"),
    "t.jsonl": lambda d, p: d.to_json(p, orient="records", lines=True),
    "t.pkl": lambda d, p: d.to_pickle(p),
    "t.sqlite": lambda d, p: d.to_sql("rows", sqlite3.connect(p), index=False),
}
if _has("pyarrow"):
    WRITERS.update({"t.parquet": lambda d, p: d.to_parquet(p), "t.feather": lambda d, p: d.to_feather(p)})
if _has("openpyxl"):
    WRITERS["t.xlsx"] = lambda d, p: d.to_excel(p, index=False)


@pytest.mark.parametrize("name", sorted(WRITERS))
def test_every_format_reads_back(frame: pd.DataFrame, tmp_path: Path, name: str) -> None:
    small = frame.head(300)
    path = tmp_path / name
    WRITERS[name](small, path)
    src = load(path)
    assert src.df.shape == small.shape, name
    assert list(map(str, src.df.columns)) == list(small.columns)
    assert src.full is None and src.name == name
    np.testing.assert_allclose(pd.to_numeric(src.df["count"]).to_numpy(), small["count"].to_numpy())


def test_sniffing(tmp_path: Path) -> None:
    (tmp_path / "a.csv").write_text("a;b;c\n1;2,5;x\n3;4,5;y\n", encoding="utf-8")
    assert sniff_csv(str(tmp_path / "a.csv"))["sep"] == ";"
    (tmp_path / "b.csv").write_bytes("name,ville\nA,Nîmes\n".encode("cp1252"))
    assert sniff_csv(str(tmp_path / "b.csv"))["encoding"] == "cp1252"
    (tmp_path / "mystery").write_bytes(b"PAR1....")
    assert detect_format(str(tmp_path / "mystery")) == "parquet"


def test_objects(frame: pd.DataFrame) -> None:
    arr = frame[["num", "count"]].to_numpy()
    assert load(arr).df.shape == arr.shape
    assert load(np.arange(10)).df.columns.tolist() == ["value"]
    assert load({"a": [1, 2], "b": ["x", "y"]}).df.shape == (2, 2)
    assert load([{"a": 1}, {"a": 2, "b": 3}]).df.shape == (2, 2)
    assert load(frame["city"]).df.shape == (len(frame), 1)
    if _has("polars") and _has("pyarrow"):
        import polars as pl

        assert load(pl.from_pandas(frame)).df.shape == frame.shape
        assert load(pl.from_pandas(frame).lazy()).df.shape == frame.shape
    if _has("pyarrow"):
        import pyarrow as pa

        assert load(pa.Table.from_pandas(frame)).df.shape == frame.shape
    if has_dask():
        import dask.dataframe as dd

        src = load(dd.from_pandas(frame, npartitions=4), StoneConfig(max_rows=1_000))
        assert src.full is not None and src.full.n_rows == len(frame) and len(src.df) == 1_000
        assert load(dd.from_pandas(frame, npartitions=4)).df.shape == frame.shape  # small: every row, no sample
    with pytest.raises(LoadError):
        load(object())


def test_glob_and_folder(frame: pd.DataFrame, tmp_path: Path) -> None:
    for i in range(3):
        frame.iloc[i * 100:(i + 1) * 100].to_csv(tmp_path / f"part{i}.csv", index=False)
    assert len(load(tmp_path / "part*.csv").df) == 300
    assert len(load(tmp_path).df) == 300
    with pytest.raises(LoadError):
        load(tmp_path / "nothing*.csv")


def _check_exact(src, frame: pd.DataFrame) -> None:  # noqa: ANN001
    full = src.full
    assert full is not None and full.n_rows == len(frame)
    cols = full.columns
    assert cols.loc["num", "n_missing"] == frame["num"].isna().sum()
    assert cols.loc["num", "mean"] == pytest.approx(frame["num"].mean(), rel=1e-12)
    assert cols.loc["num", "std"] == pytest.approx(frame["num"].std(), rel=1e-9)
    assert cols.loc["count", "max"] == frame["count"].max()
    assert cols.loc["city", "n_distinct"] == 3 and cols.loc["city", "distinct_exact"]
    assert cols.loc["city", "top_count"] == frame["city"].value_counts().iloc[0]
    assert full.duplicate_rows["count"] == frame.duplicated().sum() == 37
    n_ids = frame["id"].nunique()
    if n_ids >= KMV_K:
        assert abs(cols.loc["id", "n_distinct"] - n_ids) / n_ids < 0.06
    sample = src.df
    assert len(sample) == 2_000 and sample.index.is_monotonic_increasing and sample.index.is_unique
    assert sample.index.max() < len(frame)
    pd.testing.assert_frame_equal(sample, frame.loc[sample.index], check_dtype=False)  # real rows, original index


def test_streaming_is_exact_and_samples_real_rows(frame: pd.DataFrame, tmp_path: Path) -> None:
    cfg = StoneConfig(max_rows=2_000, chunk_rows=700)
    path = tmp_path / "big.csv"
    frame.to_csv(path, index=False)
    _check_exact(load(path, cfg, engine="pandas"), frame)
    if has_dask():
        _check_exact(load(path, cfg, engine="dask"), frame)
    _check_exact(load(frame, cfg), frame)  # an in-memory frame bigger than max_rows


def test_sample_does_not_depend_on_chunking(frame: pd.DataFrame) -> None:
    a = scan_chunks([frame], k=500)
    b = scan_chunks([frame.iloc[i:i + 1000] for i in range(0, len(frame), 1000)], k=500)
    assert len(a.sample) == len(b.sample) == 500
    assert a.columns.drop(columns="dtypes").equals(b.columns.drop(columns="dtypes")) or (
        np.allclose(a.columns["mean"].dropna().astype(float), b.columns["mean"].dropna().astype(float)))


def test_see_takes_a_path_and_reports_the_sample(frame: pd.DataFrame, tmp_path: Path) -> None:
    path = tmp_path / "data.csv"
    frame.to_csv(path, index=False)
    v = see(path, config=StoneConfig(max_rows=1_500), verbose=False, out_dir=tmp_path / "out")
    o = v.overview
    assert o["sampled"] and o["n_rows"] == len(frame) and o["n_rows_profiled"] == 1_500
    assert o["n_duplicate_rows"] == 37 and v.full is not None and v.name == "data.csv"
    assert any(a.get("kind") == "sampled" for a in v.alerts)
    html = v.recap["html"].read_text()
    assert "Exact figures over all" in html and "Sampled profile" in html


def test_duplicates_ignoring_identifiers_exact_and_in_sample(tmp_path: Path) -> None:
    rng = np.random.default_rng(9)
    base = pd.DataFrame({"amount": rng.normal(0, 1000, 3000).round(2), "account": rng.integers(0, 50, 3000)})
    copies = base.sample(150, random_state=1)
    df = pd.concat([base, copies], ignore_index=True)
    df.insert(0, "trans_id", rng.permutation(len(df)) + 10_000)  # every row has its own id
    path = tmp_path / "trans.csv"
    df.to_csv(path, index=False)
    small = see(path, plots=False, verbose=False)
    dr = small.quality["duplicate_rows"]
    assert dr["count"] == 0 and dr["count_ignoring_ids"] == 150 and dr["id_columns"] == ["trans_id"]
    assert any(a.get("kind") == "duplicates_ids" for a in small.alerts)
    big = see(path, plots=False, verbose=False, config=StoneConfig(max_rows=1_000, chunk_rows=700))
    dr = big.quality["duplicate_rows"]
    assert big.full is not None and dr["exact_over_all_rows"] and dr["count_ignoring_ids"] == 150
    assert "ignoring trans_id" in big.overview["duplicates_text"]
    capped = see(path, plots=False, verbose=False, config=StoneConfig(max_rows=1_000, chunk_rows=700, max_hashed_rows=1_000),
                 engine="pandas")
    assert capped.quality["duplicate_rows"].get("not_counted") and "in the sample" in capped.overview["duplicates_text"]


def test_rows_with_missing_are_counted_over_every_row(tmp_path: Path) -> None:
    rng = np.random.default_rng(8)
    n = 6_000
    df = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n), "c": rng.integers(0, 9, n).astype(float)})
    for col in df:
        df.loc[rng.choice(n, 300, replace=False), col] = np.nan
    df.loc[rng.choice(n, 120, replace=False), ["a", "b"]] = np.nan
    truth = int(df.isna().any(axis=1).sum())
    many = int((df.isna().mean(axis=1) >= StoneConfig().row_missing_pct / 100).sum())
    path = tmp_path / "gaps.csv"
    df.to_csv(path, index=False)
    v = see(path, plots=False, verbose=False, config=StoneConfig(max_rows=1_000, chunk_rows=700))
    assert v.overview["sampled"] and v.overview["n_rows_with_missing"] == truth  # not the count in the 1,000-row sample
    assert v.quality["missing_patterns"]["n_rows_many_missing"] == many
    # a column missing from some chunks leaves every row of those chunks with a missing value
    a = scan_chunks([df.iloc[:3000], df.iloc[3000:].drop(columns="c")], k=500)
    assert a.n_rows_with_missing == int(df.iloc[:3000].isna().any(axis=1).sum()) + 3000


def test_ragged_lines_are_skipped_counted_and_reported(tmp_path: Path) -> None:
    lines = ["a,b,c"] + [f"{i},{i % 7},{i % 3}" for i in range(12_000)]
    for k in (50, 6_000, 9_001):  # three lines with two extra fields (file lines 51, 6001, 9002)
        lines[k] += ",9,9"
    path = tmp_path / "ragged.csv"
    path.write_text("\n".join(lines) + "\n")
    runs = {"whole": {}, "chunks": {"config": StoneConfig(max_rows=1_000, chunk_rows=2_500), "engine": "pandas"}}
    if has_dask():
        runs["dask"] = {"config": StoneConfig(max_rows=1_000, chunk_rows=2_500), "engine": "dask"}
    for how, kw in runs.items():  # the chunked readers used to stop with a ParserError
        v = see(path, plots=False, verbose=False, **kw)
        assert v.overview["n_rows"] == 12_000 - 3, how
        found = [a["message"] for a in v.alerts if a.get("kind") == "bad_lines"]
        assert found and found[0].startswith("3 lines") and "(line 51, 6001, 9002)" in found[0], (how, found)
    if has_dask():  # a dask block counts its skipped lines but cannot number them in the file
        from dask.bytes import read_bytes

        from elessar.sources import _CsvBlock

        _, blocks = read_bytes([str(path)], delimiter=b"\n", blocksize=40_000, sample=False)
        header = lines[0].encode() + b"\n"
        frames = [_CsvBlock(b, None if i == 0 else header, {"sep": ","}, "utf-8")() for i, b in enumerate(blocks[0])]
        assert len(frames) > 1 and sum(f.attrs["bad_lines"] for f in frames) == 3
        assert not any(f.attrs["bad_line_numbers"] for f in frames)


def test_sqlite_table_names_are_quoted_and_checked(tmp_path: Path) -> None:
    path = tmp_path / "q.db"
    name = 'my "big" table'
    with sqlite3.connect(path) as con:
        con.execute('CREATE TABLE "my ""big"" table" (a INT, b TEXT)')
        con.executemany('INSERT INTO "my ""big"" table" VALUES (?, ?)', [(i, str(i % 5)) for i in range(3_000)])
        con.execute("CREATE TABLE small (a INT)")
        con.execute("INSERT INTO small VALUES (1)")
    # the biggest table, found although its name holds quotes (its count used to fail, and 'small' was read)
    assert load(path).origin["table"] == name and len(load(path).df) == 3_000
    assert len(load(path, table=name).df) == 3_000
    streamed = load(path, StoneConfig(max_rows=500, chunk_rows=700), table=name)
    assert streamed.full is not None and streamed.full.n_rows == 3_000
    with pytest.raises(LoadError, match="no table 'nope'"):
        load(path, table="nope")


_RAN: list[str] = []


def _mark(what: str) -> None:
    _RAN.append(what)


class _Runs:
    """Unpickling this calls ``_mark``: a stand-in for the code a pickle can run."""

    def __reduce__(self):  # noqa: ANN204
        return (_mark, ("unpickled",))


def test_pickles_are_read_only_when_named(tmp_path: Path) -> None:
    import pickle

    pd.DataFrame({"a": range(50), "b": range(50)}).to_csv(tmp_path / "a.csv", index=False)
    (tmp_path / "z.pkl").write_bytes(pickle.dumps(_Runs()))
    _RAN.clear()
    for scan in (tmp_path, tmp_path / "*"):  # a folder or a pattern reads the CSV and leaves the pickle alone
        src = load(scan)
        assert src.origin["kind"] == "csv file" and len(src.df) == 50
    with pytest.raises(LoadError, match="only pickle files"):
        load(tmp_path / "*.pkl")
    assert _RAN == []
    with pytest.raises(LoadError):  # named on its own: read, as pd.read_pickle would (this one holds no table)
        load(tmp_path / "z.pkl")
    assert _RAN == ["unpickled"]


@pytest.fixture
def web(tmp_path: Path):  # noqa: ANN201
    """A local HTTP server on ``tmp_path``: yields (base URL, bodies sent in full per path, whether to announce
    sizes). A body counts once the server wrote its last byte, i.e. the client read it all."""
    import functools
    import http.server
    import threading

    sent: dict[str, int] = {}
    opts = {"length": True}

    class Handler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *args: object) -> None:
            pass

        def send_header(self, key: str, value: str) -> None:
            if key != "Content-Length" or opts["length"]:
                super().send_header(key, value)

        def copyfile(self, source, outputfile) -> None:  # noqa: ANN001
            try:
                while chunk := source.read(64 << 10):
                    outputfile.write(chunk)
            except OSError:  # the client stopped reading: it only wanted the head
                return
            path = self.path.lstrip("/")
            sent[path] = sent.get(path, 0) + 1

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0),
                                             functools.partial(Handler, directory=str(tmp_path)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}", sent, opts
    finally:
        server.shutdown()


def test_web_files_are_streamed_not_downloaded_whole(web, tmp_path: Path) -> None:  # noqa: ANN001
    base, sent, opts = web
    rng = np.random.default_rng(5)
    n = 500_000  # 12 MB: well beyond what the local socket buffers absorb from a client that stops reading
    df = pd.DataFrame({"a": rng.normal(size=n), "b": rng.integers(0, 50, n), "c": rng.choice(["x", "y"], n)})
    df.to_csv(tmp_path / "big.csv", index=False)
    size = (tmp_path / "big.csv").stat().st_size
    cfg = StoneConfig(max_rows=2_000, chunk_rows=20_000)
    src = load(f"{base}/big.csv", cfg)
    assert src.full is not None and src.full.n_rows == n and src.origin["size_bytes"] == size
    # sniffing, header and size estimate read the head only: the body goes through once (it used to go four times)
    assert sent["big.csv"] == 1
    opts["length"] = False  # a server that does not give the size: read in chunks, exact either way
    sent.clear()
    assert load(f"{base}/big.csv", cfg).full.n_rows == n
    df.head(300).to_csv(tmp_path / "small.csv", index=False)
    small = load(f"{base}/small.csv", cfg)
    assert small.full is None and len(small.df) == 300  # all of it fitted in the sample: profiled whole
    (tmp_path / "page.html").write_text("<!doctype html><html><body>download</body></html>")
    with pytest.raises(LoadError, match="web page"):
        load(f"{base}/page.html")
    with pytest.raises(LoadError, match="HTTP 404"):
        load(f"{base}/missing.csv")


def test_feather_and_orc_are_streamed_and_whole_reads_are_guarded(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    import pyarrow as pa
    import pyarrow.feather as feather

    from elessar import sources

    rng = np.random.default_rng(6)
    n = 30_000
    df = pd.DataFrame({"a": rng.normal(size=n), "b": rng.integers(0, 9, n), "c": rng.choice(["x", "y", "z"], n)})
    feather.write_feather(pa.Table.from_pandas(df, preserve_index=False), tmp_path / "t.feather", chunksize=7_000)
    names = ["t.feather"]
    try:  # pyarrow has no ORC support on Windows
        import pyarrow.orc  # noqa: F401

        df.to_orc(tmp_path / "t.orc")
        names.append("t.orc")
    except ImportError:
        pass
    cfg = StoneConfig(max_rows=2_000, chunk_rows=3_000)
    for name in names:  # these used to be read whole whatever their size
        src = load(tmp_path / name, cfg)
        assert src.full is not None and src.full.n_rows == n, name
        assert src.full.engine == "pandas chunks" and src.full.n_chunks >= 10  # read part by part, never whole
        assert src.full.columns.loc["a", "mean"] == pytest.approx(df["a"].mean())
        assert src.full.columns.loc["c", "n_distinct"] == 3
    assert load(tmp_path / "t.feather").full is None  # small enough: read whole as before
    df.head(2_000).to_excel(tmp_path / "t.xlsx", index=False)
    assert len(load(tmp_path / "t.xlsx").df) == 2_000
    monkeypatch.setattr(sources, "available_memory", lambda: 64 << 10)  # 64 KiB free: reading it whole would not fit
    with pytest.raises(LoadError, match="convert it to Parquet or CSV"):
        load(tmp_path / "t.xlsx")


class _Gadget:
    """Starts from an allowed class and walks out through getattr (to __init__, then its __globals__...): the usual
    way out of a naive unpickling whitelist."""

    def __reduce__(self):  # noqa: ANN204
        return (getattr, (pd.DataFrame, "__init__"))


def test_pickles_from_a_url_are_rebuilt_only_from_table_parts(web, tmp_path: Path) -> None:  # noqa: ANN001
    import pickle

    base, _, _ = web
    n = 40
    rng = np.random.default_rng(13)
    df = pd.DataFrame({"i": np.arange(n), "f": rng.normal(size=n), "s": [f"x{i}" for i in range(n)],
                       "cat": pd.Categorical(rng.choice(list("abc"), n)),
                       "when": pd.date_range("2020-01-01", periods=n, tz="Europe/Paris"),
                       "per": pd.period_range("2020-01", periods=n, freq="M"),
                       "I": pd.array(list(range(n - 1)) + [None], dtype="Int64"),
                       "arrow": pd.array(rng.normal(size=n), dtype="double[pyarrow]")}).set_index(["s", "i"])
    for proto in (2, 4, 5):
        (tmp_path / f"t{proto}.pkl").write_bytes(pickle.dumps(df, protocol=proto))
        got = load(f"{base}/t{proto}.pkl").df
        pd.testing.assert_frame_equal(got.reset_index(), df.reset_index(), check_dtype=False)
    _RAN.clear()
    (tmp_path / "evil.pkl").write_bytes(pickle.dumps(_Runs()))
    with pytest.raises(LoadError, match="download it"):
        load(f"{base}/evil.pkl")
    assert _RAN == []  # refused before running anything
    (tmp_path / "gadget.pkl").write_bytes(pickle.dumps(_Gadget()))
    with pytest.raises(LoadError, match=r"getattr\(DataFrame, '__init__'\)"):
        load(f"{base}/gadget.pkl")
    assert len(load(tmp_path / "t4.pkl").df) == n  # a local pickle named by its path is read as before
