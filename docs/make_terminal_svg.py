"""Build docs/terminal.svg: an animated terminal session running elessar on a real CSV.

The output shown is the real output of the ``elessar`` command line (an excerpt of it), so the
animation never drifts from what the tool prints. Pure SVG + CSS: no script, no external font,
rendered by GitHub; ``prefers-reduced-motion`` gets a static frame.

Usage:  python docs/make_terminal_svg.py titanic.csv Survived docs/terminal.svg
"""

from __future__ import annotations

import html
import re
import subprocess
import sys
import tempfile
from pathlib import Path

WIDTH_CHARS = 92
FONT_SIZE = 13
LINE_H = 18
CHAR_W = 7.85  # generous monospace advance at 13px, used only to size the window
PAD_X, TOP = 22, 58
VISIBLE = 30
HOLD = 5.0  # seconds the final screen stays before the loop restarts

BG, BAR, FG, DIM = "#0d1117", "#161b22", "#d1d7e0", "#7d8590"
COLORS = {"prompt": "#3fb950", "cmd": "#e6edf3", "title": "#79c0ff", "high": "#ff7b72", "medium": "#e3b341",
          "info": "#79c0ff", "good": "#3fb950", "warn": "#ffa657"}


def run_elessar(csv: str, target: str) -> list[str]:
    """Real run (plots included) from a scratch directory, so paths print as ``elessar_report/``."""
    with tempfile.TemporaryDirectory() as tmp:
        out = subprocess.run(
            [sys.executable, "-m", "elessar", str(Path(csv).resolve()), "--target", target, "--width", str(WIDTH_CHARS),
             "-o", "elessar_report"],
            check=True, capture_output=True, text=True, cwd=tmp,
        ).stdout
    return out.rstrip("\n").splitlines()


def excerpt(lines: list[str]) -> list[str]:
    """Overview, the first alerts, the column summary, one column's details, correlations, target."""
    def index(prefix: str) -> int:
        return next(i for i, line in enumerate(lines) if line.startswith(prefix))

    alerts, summary, details = index("ALERTS"), index("COLUMN SUMMARY"), index("COLUMN DETAILS")
    corr, target, plots = index("CORRELATIONS"), index("TARGET ANALYSIS"), index("PLOTS")
    column = next(i for i in range(details, corr) if lines[i].startswith("▸ Age"))
    block_end = next(i for i in range(column + 1, corr) if not lines[i].strip())
    return (lines[:alerts + 12] + ["  …", ""] + lines[summary:details] + lines[details:details + 2]
            + lines[column:block_end] + ["…", ""] + lines[corr:target] + lines[target:plots] + lines[plots:plots + 3])


def color_line(line: str) -> list[tuple[str, str | None, bool]]:
    """Split a line into (text, color, bold) segments."""
    if re.fullmatch(r"[═─ ]+", line):
        return [(line, DIM, False)]
    if re.match(r"^[A-Z][A-Z ]+(\(|$)", line) or line.startswith(" elessar report"):
        return [(line, COLORS["title"], True)]
    m = re.match(r"^(  )(HIGH|MEDIUM|info)( .*)$", line)
    if m:
        return [(m.group(1), None, False), (m.group(2), COLORS[m.group(2).lower()], True), (m.group(3), None, False)]
    if line.startswith("▸"):
        return [(line, "#e6edf3", True)]
    segments: list[tuple[str, str | None, bool]] = []
    for part in re.split(r"(LEAKAGE\?|(?<=\s)\*(?=\s|$)|not normal|\byes\b)", line):
        if not part:
            continue
        color = {"LEAKAGE?": COLORS["high"], "*": COLORS["good"], "not normal": COLORS["warn"], "yes": COLORS["good"]}
        segments.append((part, color.get(part), part == "LEAKAGE?"))
    return segments


def tspans(segments: list[tuple[str, str | None, bool]]) -> str:
    out = []
    for text, color, bold in segments:
        attrs = (f' fill="{color}"' if color else "") + (' font-weight="700"' if bold else "")
        out.append(f"<tspan{attrs}>{html.escape(text)}</tspan>")
    return "".join(out)


def build(output: list[str], csv_name: str, target: str) -> str:
    reveals: list[tuple[str, float, str]] = []  # (css class, appear time, animated property)
    rows: list[tuple[float, str]] = []  # (appear time, svg text content)
    t = 0.6

    def keyframes(name: str, appear: float, prop: str = "opacity") -> None:
        """Hidden until ``appear``, then visible until the loop fades out (``fill-opacity`` for tspans)."""
        reveals.append((name, appear, prop))

    def type_command(command: str, start: float, speed: float = 0.045) -> float:
        """A prompt line whose characters appear one by one; returns the time typing ends."""
        n = len(rows)
        keyframes(f"l{n}", start)
        chars = []
        for i, ch in enumerate(command):
            name = f"c{n}_{i}"
            keyframes(name, start + 0.35 + i * speed, "fill-opacity")
            chars.append(f'<tspan class="{name}">{html.escape(ch)}</tspan>')
        rows.append((start, f'<tspan fill="{COLORS["prompt"]}">❯ </tspan><tspan fill="{COLORS["cmd"]}">{"".join(chars)}</tspan>'))
        return start + 0.35 + len(command) * speed

    def print_line(segments: list[tuple[str, str | None, bool]], at: float) -> None:
        keyframes(f"l{len(rows)}", at)
        rows.append((at, tspans(segments)))

    t = type_command("pip install git+https://github.com/thibautdqb/elessar", t) + 0.4
    print_line([("Successfully installed elessar", DIM, False)], t)
    t += 0.6
    t = type_command(f"elessar {csv_name} --target {target}", t) + 0.5
    for line in output:
        pause = 0.35 if re.match(r"^[A-Z]", line) else 0.075
        t += pause
        print_line(color_line(line), t)
    t += 0.4
    final_prompt = len(rows)
    keyframes(f"l{final_prompt}", t)
    rows.append((t, f'<tspan fill="{COLORS["prompt"]}">❯ </tspan><tspan class="cursor">▋</tspan>'))
    total = t + HOLD

    def pct(at: float) -> str:
        return f"{100 * min(max(at / total, 0), 1):.3f}%"

    fade = pct(total - 0.6)
    css = [f"@keyframes {name}{{0%,{pct(at - 0.001)}{{{prop}:0}}{pct(at)},{fade}{{{prop}:1}}100%{{{prop}:0}}}}"
           f".{name}{{animation:{name} {total:.2f}s infinite}}" for name, at, prop in reveals]

    # scrolling: once the window is full, every new line moves the content up by one line
    scroll = ["0%{transform:translateY(0)}"]
    for i, (appear, _) in enumerate(rows):
        if i >= VISIBLE:
            shift = -(i - VISIBLE + 1) * LINE_H
            scroll.append(f"{pct(appear - 0.04)}{{transform:translateY({shift + LINE_H}px)}}"
                          f"{pct(appear)}{{transform:translateY({shift}px)}}")
    last_shift = -(len(rows) - VISIBLE) * LINE_H if len(rows) > VISIBLE else 0
    scroll.append(f"99.9%{{transform:translateY({last_shift}px)}}100%{{transform:translateY(0)}}")
    css.append(f"@keyframes scroll{{{''.join(scroll)}}}.content{{animation:scroll {total:.2f}s infinite}}")
    css.append("@keyframes blink{0%,49%{fill-opacity:1}50%,100%{fill-opacity:0}}.cursor{animation:blink 1s step-end infinite}")
    css.append("@media (prefers-reduced-motion: reduce){*{animation:none!important;opacity:1!important}"
               ".content{transform:none!important}}")

    body = []
    for i, (_, content) in enumerate(rows):
        body.append(f'<text class="l{i}" x="{PAD_X}" y="{TOP + i * LINE_H}" xml:space="preserve">{content}</text>')

    width = round(PAD_X * 2 + WIDTH_CHARS * CHAR_W)
    height = TOP + VISIBLE * LINE_H + 8
    font = "ui-monospace,SFMono-Regular,'SF Mono',Menlo,Consolas,'Liberation Mono','DejaVu Sans Mono',monospace"
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="elessar analysing {html.escape(csv_name)} in a terminal">
<title>elessar analysing {html.escape(csv_name)} in a terminal</title>
<style>
text{{font-family:{font};font-size:{FONT_SIZE}px;fill:{FG};white-space:pre;opacity:0}}
{chr(10).join(css)}
</style>
<defs><clipPath id="screen"><rect x="0" y="40" width="{width}" height="{height - 40}"/></clipPath></defs>
<rect width="{width}" height="{height}" rx="10" fill="{BG}"/>
<path d="M0 10a10 10 0 0 1 10-10h{width - 20}a10 10 0 0 1 10 10v26h-{width}z" fill="{BAR}"/>
<circle cx="20" cy="18" r="6" fill="#ff5f57"/><circle cx="40" cy="18" r="6" fill="#febc2e"/><circle cx="60" cy="18" r="6" fill="#28c840"/>
<text x="{width / 2}" y="22" text-anchor="middle" style="opacity:1;fill:{DIM};font-size:12px">elessar — zsh</text>
<g clip-path="url(#screen)"><g class="content">
{chr(10).join(body)}
</g></g>
</svg>
"""


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    csv, target, out = sys.argv[1:]
    svg = build(excerpt(run_elessar(csv, target)), "train.csv", target)
    Path(out).write_text(svg, encoding="utf-8")
    print(f"wrote {out} ({len(svg) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
