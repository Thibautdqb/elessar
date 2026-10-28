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
