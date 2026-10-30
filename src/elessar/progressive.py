"""Progressive typing of a table too big to load: decide every column's type on as few rows as
the evidence allows, before the full pass reads every row.

1. **First look**: the head of the table plus a few blocks from its middle and its end (the
   head alone is often unrepresentative: empty columns, formats that change half-way).
2. Every column goes down the typing decision tree (:mod:`elessar.types`), which says whether
   its decision is certain on the rows seen, or uncertain (a threshold test whose 95% interval
   crosses its threshold, labels probably still unseen, an ambiguous date order...).
3. While some columns are uncertain, more rows are drawn - twice as many each round - and only
   those columns are typed again, until every decision is settled, stops changing, or the row
   budget is spent.
4. **Time order is never destroyed**: when the first look finds a time axis, the extra rows are
   contiguous blocks at random positions instead of scattered rows, so series stay intact.

Rows are read through a :class:`Probe`: random positions are reached directly (byte offsets in
an uncompressed text file, row groups in Parquet, slices of a DataFrame); other sources only
offer their head.
"""

from __future__ import annotations

import io
import os
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

from ._utils import dedupe_names, quiet
from .config import StoneConfig
from .roles import birth_columns
from .timeseries import detect as detect_time
from .types import infer_type

FIRST_LOOK_ROWS = 10_000
FIRST_LOOK_BLOCKS = (0.25, 0.5, 0.75, 0.995)


class Coverage:
    """Intervals already read, so that no row is ever drawn twice (blocks would otherwise overlap)."""

    def __init__(self) -> None:
        self.spans: list[tuple[int, int]] = []

    def free_from(self, start: int) -> int:
        """The first position at or after ``start`` that has not been read."""
        moved = True
        while moved:
            moved = False
            for a, b in self.spans:
                if a <= start < b:
                    start, moved = b, True
        return start

    def room(self, start: int) -> int | None:
        """How far one can read from ``start`` before reaching a span already read (None = no limit)."""
        ahead = [a for a, _ in self.spans if a > start]
        return min(ahead) - start if ahead else None

    def add(self, start: int, end: int) -> None:
        if end > start:
            self.spans.append((start, end))


class Probe:
    """Read parts of a table without reading all of it, never the same row twice.

    ``take_head`` / ``take_block`` add rows to what has been taken; ``rows`` returns all of it as
    one frame, parsed the way a full read of those rows would parse them.
    """

    random_access = False

    def __init__(self) -> None:
        self.parts: list[pd.DataFrame] = []

    def take_head(self, n: int) -> int:
        raise NotImplementedError

    def take_block(self, position: float, n: int) -> int:
        """Take up to ``n`` contiguous unread rows from relative position ``position`` (0 = start, 1 = end)."""
        raise NotImplementedError

    def rows(self) -> pd.DataFrame:
        return _concat(self.parts)


class FrameProbe(Probe):
    random_access = True

    def __init__(self, df: pd.DataFrame) -> None:
        super().__init__()
        self.df, self.seen = df, Coverage()

    def take_head(self, n: int) -> int:
        return self._take(0, n)

    def take_block(self, position: float, n: int) -> int:
        return self._take(int(position * max(len(self.df) - n, 0)), n)

    def _take(self, start: int, n: int) -> int:
        start = self.seen.free_from(start)
        room = self.seen.room(start)
        end = min(start + n, len(self.df), start + room if room is not None else len(self.df))
        self.seen.add(start, end)
        if end > start:
            self.parts.append(self.df.iloc[start:end])
        return max(end - start, 0)


class CsvProbe(Probe):
    """Byte-offset access to an uncompressed delimited file: seek, skip to the next line, collect raw
    lines; all the lines taken are parsed together, so column types come out as in a full read."""

    random_access = True

    def __init__(self, path: str, kw: dict[str, Any]) -> None:
        super().__init__()
        self.path, self.kw = path, {k: v for k, v in kw.items() if k != "low_memory"}
        self.size = os.path.getsize(path)
        with open(path, "rb") as fh:
            self.header = fh.readline() if kw.get("header", 0) is not None else b""
            raw = fh.read(1 << 20)
        self.avg_line = len(raw) / max(raw.count(b"\n"), 1)
        self.lines: list[bytes] = []
        self.seen = Coverage()  # byte ranges already read

    def take_head(self, n: int) -> int:
        return self._take(len(self.header), n, at_line_start=True)

    def take_block(self, position: float, n: int) -> int:
        offset = int(position * max(self.size - n * self.avg_line, 0))
        return self._take(max(offset, len(self.header)), n, at_line_start=offset <= len(self.header))

    def _take(self, offset: int, n: int, at_line_start: bool) -> int:
        got = 0
        with open(self.path, "rb") as fh:
            fh.seek(offset)
            if not at_line_start:
                fh.readline()  # finish the line we landed in
            start = fh.tell()
            free = self.seen.free_from(start)
            if free != start:  # landed in a part already read: continue after it, from a line start
                fh.seek(free)
                start = free
            room = self.seen.room(start)
            while got < n and (room is None or fh.tell() < start + room):
                line = fh.readline()
                if not line:
                    break
                self.lines.append(line if line.endswith(b"\n") else line + b"\n")
                got += 1
            self.seen.add(start, fh.tell())
        return got

    def rows(self) -> pd.DataFrame:
        kw = dict(self.kw)
        if self.header:
            kw["encoding"] = self.kw.get("encoding")
        with quiet():
            return pd.read_csv(io.BytesIO(self.header + b"".join(self.lines)), **kw, on_bad_lines="skip")  # counted by the full pass


class ParquetProbe(Probe):
    """Row-group access to Parquet files: a block is a contiguous slice of one row group."""

    random_access = True

    def __init__(self, files: list[str], columns: list[str] | None = None) -> None:
        import pyarrow.parquet as pq

        super().__init__()
        self.columns = columns
        self.groups = [(f, i, pq.ParquetFile(f).metadata.row_group(i).num_rows)
                       for f in files for i in range(pq.ParquetFile(f).num_row_groups)]
        self.total = sum(g[2] for g in self.groups)
        self.seen = Coverage()  # global row positions already read

    def take_head(self, n: int) -> int:
        return self._take(0, n)

    def take_block(self, position: float, n: int) -> int:
        return self._take(int(position * max(self.total - n, 0)), n)

    def _take(self, start: int, n: int) -> int:
        """Up to ``n`` unread rows from global position ``start``, within one row group."""
        import pyarrow.parquet as pq

        start = self.seen.free_from(start)
        offset = 0
        for f, i, rows in self.groups:
            if offset + rows > start:
                room = self.seen.room(start)
                end = min(start + n, offset + rows, start + room if room is not None else offset + rows)
                self.seen.add(start, end)
                group = pq.ParquetFile(f).read_row_group(i, columns=self.columns).to_pandas()
                self.parts.append(group.iloc[start - offset:end - offset])
                return max(end - start, 0)
            offset += rows
        return 0


class HeadProbe(Probe):
    """Sources read front to back only (compressed text, JSON lines, SAS, Stata, SQLite)."""

    def __init__(self, first_chunk: Callable[[int], pd.DataFrame]) -> None:
        super().__init__()
        self.first_chunk = first_chunk

    def take_head(self, n: int) -> int:
        self.parts.append(self.first_chunk(n))
        return len(self.parts[-1])


@dataclass
class TypePlan:
    """The types decided before the full pass, and how they were reached."""

    types: dict[str, dict[str, Any]]
    rows_used: int
    rounds: list[dict[str, Any]] = field(default_factory=list)
    timeseries: dict[str, Any] | None = None
    sampling: str = "rows"  # "rows" (scattered) or "blocks" (contiguous, time order kept)
    stopped: str = ""
    rows: pd.DataFrame | None = field(default=None, repr=False)  # the rows the decisions were made on

    @property
    def uncertain(self) -> list[str]:
        return [c for c, d in self.types.items() if d.get("certainty") == "uncertain"]


def _normalise(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.set_axis(dedupe_names(list(frame.columns)), axis=1)


def _concat(frames: list[pd.DataFrame]) -> pd.DataFrame:
    frames = [_normalise(f) for f in frames if len(f)]
    if not frames:
        return pd.DataFrame()
    with quiet():
        return pd.concat(frames, ignore_index=True, sort=False)


def progressive_types(probe: Probe, cfg: StoneConfig) -> TypePlan:
    """Type every column on as few rows as the evidence allows (see the module docstring)."""
    rng = np.random.default_rng(cfg.random_state)
    probe.take_head(FIRST_LOOK_ROWS)
    if probe.random_access:
        for p in FIRST_LOOK_BLOCKS:
            probe.take_block(p, FIRST_LOOK_ROWS // 4)
    rows = _normalise(probe.rows())
    types = {c: infer_type(rows[c], cfg, sampled=True) for c in rows.columns}
    semantic = {c: d["semantic_type"] for c, d in types.items()}
    with quiet():
        births = {c for c, e in birth_columns(rows, semantic).items() if "rejected" not in e}
        ts = detect_time(rows, {c: "birth date" if c in births else t for c, t in semantic.items()})
    keep_order = ts is not None and ts["kind"] in ("single series", "panel")
    plan = TypePlan(types=types, rows_used=len(rows), timeseries=ts, sampling="blocks" if keep_order else "rows")
    plan.rounds.append({"rows": len(rows), "uncertain": plan.uncertain, "changed": []})
    unchanged = 0
    while plan.uncertain:
        if not probe.random_access:
            plan.stopped = "the source can only be read from its start"
            break
        if len(rows) >= cfg.probe_max_rows:
            plan.stopped = f"row budget ({cfg.probe_max_rows:,}) reached"
            break
        want = min(len(rows), cfg.probe_max_rows - len(rows))
        n_blocks = 8 if keep_order else 64  # few long blocks keep series whole; many short ones approach random rows
        size = max(want // n_blocks, 50)
        if sum(probe.take_block(float(p), size) for p in rng.random(n_blocks)) == 0:
            plan.stopped = "no more rows to draw"
            break
        rows = _normalise(probe.rows())
        changed = []
        for col in plan.uncertain:
            if col not in rows:
                continue
            new = infer_type(rows[col], cfg, sampled=True)
            if new["semantic_type"] != types[col]["semantic_type"]:
                changed.append(col)
            types[col] = new
        plan.rows_used = len(rows)
        plan.rounds.append({"rows": len(rows), "uncertain": plan.uncertain, "changed": changed})
        unchanged = 0 if changed else unchanged + 1
        if plan.uncertain and unchanged >= 2:
            plan.stopped = "decisions stable for two rounds but still uncertain"
            break
    if not plan.stopped:
        plan.stopped = "every decision settled"
    plan.rows = rows
    return plan


__all__ = ["Coverage", "CsvProbe", "FrameProbe", "HeadProbe", "ParquetProbe", "Probe", "TypePlan", "progressive_types"]
