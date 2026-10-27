"""Command line: ``elessar data.csv --target y`` (also ``python -m elessar``)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from .config import StoneConfig
from . import recap
from .core import see

TEXT_SEPARATORS = {".csv": ",", ".tsv": "\t", ".txt": None}
READERS = {
    ".json": pd.read_json,
    ".jsonl": lambda p: pd.read_json(p, lines=True),
    ".parquet": pd.read_parquet,
    ".feather": pd.read_feather,
    ".pkl": pd.read_pickle,
    ".pickle": pd.read_pickle,
    ".xlsx": pd.read_excel,
    ".xls": pd.read_excel,
}


def read_table(path: str | Path, sep: str | None = None, nrows: int | None = None) -> pd.DataFrame:
    """Read a table from disk, choosing the reader from the file extension (compression allowed)."""
    path = Path(path)
    suffixes = [s.lower() for s in path.suffixes if s.lower() not in (".gz", ".bz2", ".zip", ".xz", ".zst")]
    ext = suffixes[-1] if suffixes else ".csv"
    if ext in TEXT_SEPARATORS:
        sep = sep or TEXT_SEPARATORS[ext]
        return pd.read_csv(path, sep=sep, nrows=nrows, engine="python" if sep is None else "c")
    if ext not in READERS:
        supported = ", ".join(sorted([*TEXT_SEPARATORS, *READERS]))
        raise SystemExit(f"elessar: unsupported file type {ext!r} (supported: {supported})")
    df = READERS[ext](path)
    return df.head(nrows) if nrows is not None else df


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="elessar",
        description="Know absolutely everything about a table: types, stats, distributions, "
        "associations with significance, data quality, plots.",
    )
    p.add_argument("path", help="CSV / TSV / JSON / Parquet / Feather / Pickle / Excel file")
    p.add_argument("-t", "--target", help="target column to analyse features against")
    p.add_argument("-o", "--out", default="elessar_report", help="output directory for plots and exports")
    p.add_argument("--no-plots", action="store_true", help="skip the PNG plots")
    p.add_argument("--sep", help="CSV separator (default: ',')")
    p.add_argument("--nrows", type=int, help="only read the first N rows")
    p.add_argument("--sample", type=int, help="max rows for expensive steps (default 100000)")
    p.add_argument("--alpha", type=float, help="significance level (default 0.05)")
    p.add_argument("--width", type=int, help="report width in characters (default: terminal width)")
    p.add_argument("--json", action="store_true", help="also write <out>/report.json")
    p.add_argument("--markdown", action="store_true", help="also write <out>/report.md")
    p.add_argument("--csv", action="store_true", help="also write <out>/summary.csv")
    p.add_argument("-q", "--quiet", action="store_true", help="do not print the report")
    p.add_argument("--open", choices=["auto", "html", "md", "none"], default="auto",
                   help="open the recap when done: auto = HTML recap when run from a terminal (default)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    df = read_table(args.path, sep=args.sep, nrows=args.nrows)
    cfg = StoneConfig()
    if args.sample:
        cfg.sample_size = args.sample
    if args.alpha:
        cfg.alpha = args.alpha
    if args.target is not None and args.target not in map(str, df.columns):
        print(f"elessar: target {args.target!r} not in columns: {', '.join(map(str, df.columns))}", file=sys.stderr)
        return 2
    open_report = {"auto": None, "html": "html", "md": "md", "none": False}[args.open]
    report = see(df, target=args.target, config=cfg, plots=not args.no_plots, out_dir=args.out, verbose=False,
                 open_report=False, name=Path(args.path).name)
    if not args.quiet:
        report.print(width=args.width)
        sys.stdout.flush()  # keep the report before the "wrote ..." messages when both streams share a file
    recap.maybe_open(report, open_report, sys.stderr)
    out = Path(args.out)
    if args.json or args.markdown or args.csv:
        out.mkdir(parents=True, exist_ok=True)
    written = [
        export(out / name)
        for flag, name, export in (
            (args.json, "report.json", report.to_json),
            (args.markdown, "report.md", report.to_markdown),
            (args.csv, "summary.csv", report.to_csv),
        )
        if flag
    ]
    for path in written:
        print(f"wrote {path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
