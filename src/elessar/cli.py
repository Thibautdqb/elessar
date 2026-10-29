"""Command line: ``elessar data.csv --target y`` (also ``python -m elessar``)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__, recap
from .config import StoneConfig
from .core import see
from .picker import pick_target
from .sources import LoadError, load


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="elessar",
        description="Automated exploratory data analysis and data-quality profiling: semantic types, statistics, "
        "distributions, associations with significance, target analysis, data quality, charts and an HTML report.",
    )
    p.add_argument("source", help="file, URL, glob pattern ('logs/*.csv') or folder: CSV / TSV / TXT, JSON / JSONL, "
                                  "Parquet, Feather, ORC, Excel / ODS, pickle, HDF5, SAS, Stata, SPSS, SQLite, XML "
                                  "(compressed or not)")
    p.add_argument("-V", "--version", action="version", version=f"elessar {__version__}")
    target = p.add_mutually_exclusive_group()
    target.add_argument("-t", "--target", help="target column to analyse features against ('ask' opens a picker)")
    target.add_argument("--no-target", action="store_true", help="do not ask for a target")
    p.add_argument("-o", "--out", help="output folder (default: elessar_report/<file>_<target-X or no-target>_<YYYY-MM-DD>/)")
    p.add_argument("--no-plots", action="store_true", help="skip the charts (and the HTML recap)")
    p.add_argument("--single-file", action="store_true", help="also write report_single.html with the charts inlined")
    read = p.add_argument_group("reading")
    read.add_argument("--sheet", help="Excel sheet name or number (default: the first sheet with data)")
    read.add_argument("--table", help="SQLite table / HDF5 key (default: the biggest table)")
    read.add_argument("--sep", help="CSV delimiter (default: sniffed)")
    read.add_argument("--encoding", help="text encoding (default: sniffed)")
    read.add_argument("--columns", help="comma-separated columns to read")
    read.add_argument("--nrows", type=int, help="only read the first N rows")
    read.add_argument("--max-rows", type=int, help="profile at most N rows; bigger tables are streamed and sampled, "
                                                   "with exact counts over every row (default 1000000)")
    read.add_argument("--engine", choices=["auto", "dask", "pandas"], default="auto",
                      help="streaming engine for big files (auto: dask when installed)")
    p.add_argument("-j", "--jobs", type=int, help="processes for the heavy steps (default: automatic; 1 = none)")
    p.add_argument("--sample", type=int, help="max rows for the expensive tests (default 100000)")
    p.add_argument("--alpha", type=float, help="significance level (default 0.05)")
    p.add_argument("--width", type=int, help="report width in characters (default: terminal width)")
    p.add_argument("--json", action="store_true", help="also write <out>/report.json")
    p.add_argument("--markdown", action="store_true", help="also write <out>/report.md")
    p.add_argument("--csv", action="store_true", help="also write <out>/summary.csv")
    p.add_argument("-q", "--quiet", action="store_true", help="do not print the report")
    p.add_argument("--open", choices=["auto", "html", "md", "none"], default="auto",
                   help="open the recap when done: auto = HTML recap when run from a terminal (default)")
    return p


def _interactive() -> bool:
    return bool(sys.stdin and sys.stdin.isatty() and sys.stdout.isatty())


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = StoneConfig()
    for flag, attr in (("sample", "sample_size"), ("alpha", "alpha"), ("max_rows", "max_rows"), ("jobs", "n_jobs")):
        if getattr(args, flag) is not None:
            setattr(cfg, attr, getattr(args, flag))
    cfg.single_file_report = args.single_file
    sheet = int(args.sheet) if args.sheet is not None and args.sheet.isdigit() else args.sheet
    columns = [c.strip() for c in args.columns.split(",")] if args.columns else None
    try:
        src = load(args.source, cfg, sheet=sheet, table=args.table, sep=args.sep, encoding=args.encoding,
                   columns=columns, nrows=args.nrows, engine=args.engine)
    except (LoadError, OSError, ValueError, ImportError) as exc:
        print(f"elessar: {exc}", file=sys.stderr)
        return 2
    target = args.target
    if target == "ask" or (target is None and not args.no_target and _interactive()):
        try:
            target = pick_target(src.df)
        except RuntimeError as exc:
            print(f"elessar: {exc}", file=sys.stderr)
            return 2
    if target is not None and str(target) not in map(str, src.df.columns):
        print(f"elessar: target {target!r} not in columns: {', '.join(map(str, src.df.columns))}", file=sys.stderr)
        return 2
    open_report = {"auto": None, "html": "html", "md": "md", "none": False}[args.open]
    report = see(src, target=target, config=cfg, plots=not args.no_plots, out_dir=args.out, verbose=False,
                 open_report=False)
    if not args.quiet:
        report.print(width=args.width)
        sys.stdout.flush()  # keep the report before the "wrote ..." messages when both streams share a file
    if recap.maybe_open(report, open_report, sys.stderr) is None and report.recap:
        print(f"\nReport: {report.recap['html'].resolve()}", file=sys.stderr)
    out = report.out_dir or Path(args.out or ".")
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
