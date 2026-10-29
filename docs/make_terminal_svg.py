"""Build docs/terminal.svg: elessar installed and run in a terminal, then its report opening in a browser.

Everything shown is real: the terminal lines are the output of the ``elessar`` command line on the
given CSV, and the browser shows a screenshot of the ``report.html`` that run wrote (taken with
headless Chrome or Chromium, which must be installed). Pure SVG + CSS: no script, rendered by
GitHub; ``prefers-reduced-motion`` gets a static frame.

Usage:  python docs/make_terminal_svg.py titanic.csv Survived docs/terminal.svg
"""

from __future__ import annotations

import base64
import html
import io
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image

from elessar import __version__

WIDTH, HEIGHT = 900, 560
WIDTH_CHARS = 100
FONT_SIZE = 13
LINE_H = 18
PAD_X, TOP = 22, 58
VISIBLE = 27
SHOT_WIDTH = 1180  # CSS width of the report page when photographed
SHOT_HEIGHT = 7_400  # how much of the report the browser scrolls through
BROWSER_BAR = 74  # tab strip + address bar
SCROLL_SECONDS = 26.0
HOLD = 3.0

BG, BAR, FG, DIM = "#0d1117", "#161b22", "#d1d7e0", "#7d8590"
COLORS = {"prompt": "#3fb950", "cmd": "#e6edf3", "title": "#79c0ff", "high": "#ff7b72", "medium": "#e3b341",
          "info": "#79c0ff", "good": "#3fb950", "warn": "#ffa657"}


def run_elessar(csv: str, target: str, workdir: Path) -> tuple[list[str], Path]:
    """Real run (charts and report included) of ``elessar train.csv`` in ``workdir``: the report lands in its
    own dated folder, ``elessar_report/train_target-<target>_<date>/``, as for anyone running the command."""
    shutil.copy(csv, workdir / "train.csv")
    out = subprocess.run(
        [sys.executable, "-m", "elessar", "train.csv", "--target", target, "--width", str(WIDTH_CHARS),
         "--open", "none"],
        check=True, capture_output=True, text=True, cwd=workdir,
    ).stdout
    report = next((workdir / "elessar_report").glob("*/report.html"))
    return out.rstrip("\n").splitlines(), report


def screenshot(page: Path) -> Image.Image:
    """The top of the report, as the browser shows it."""
    chrome = next((c for c in ("google-chrome", "chromium", "chromium-browser") if shutil.which(c)), None)
    if chrome is None:
        raise SystemExit("needs google-chrome or chromium for the report screenshot")
    shot = page.parent / "shot.png"
    subprocess.run([chrome, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--force-device-scale-factor=1",
                    f"--window-size={SHOT_WIDTH},{SHOT_HEIGHT}", f"--screenshot={shot}", page.resolve().as_uri()],
                   check=True, capture_output=True)
    return Image.open(shot).convert("RGB")


def excerpt(lines: list[str]) -> list[str]:
    """The overview and the first alerts: what a person sees before the browser opens."""
    alerts = next(i for i, line in enumerate(lines) if line.startswith("ALERTS"))
    return lines[:alerts + 11] + ["  …"]


def color_line(line: str) -> list[tuple[str, str | None, bool]]:
    """Split a line into (text, color, bold) segments."""
    if re.fullmatch(r"[═─ ]+", line):
        return [(line, DIM, False)]
    if re.match(r"^[A-Z][A-Z ]+(\(|$)", line) or line.startswith(" elessar report"):
        return [(line, COLORS["title"], True)]
    m = re.match(r"^(  )(HIGH|MEDIUM|info)( .*)$", line)
    if m:
        return [(m.group(1), None, False), (m.group(2), COLORS[m.group(2).lower()], True), (m.group(3), None, False)]
    return [(line, None, False)]


def tspans(segments: list[tuple[str, str | None, bool]]) -> str:
    out = []
    for text, color, bold in segments:
        attrs = (f' fill="{color}"' if color else "") + (' font-weight="700"' if bold else "")
        out.append(f"<tspan{attrs}>{html.escape(text)}</tspan>")
    return "".join(out)


def build(output: list[str], shot: Image.Image, csv_name: str, target: str, report: str) -> str:
    reveals: list[tuple[str, float, str]] = []  # (css class, appear time, animated property)
    rows: list[tuple[float, str]] = []  # (appear time, svg text content)

    def type_command(command: str, start: float, speed: float = 0.04) -> float:
        """A prompt line whose characters appear one by one; returns the time typing ends."""
        n = len(rows)
        reveals.append((f"l{n}", start, "opacity"))
        chars = []
        for i, ch in enumerate(command):
            name = f"c{n}_{i}"
            reveals.append((name, start + 0.35 + i * speed, "fill-opacity"))
            chars.append(f'<tspan class="{name}">{html.escape(ch)}</tspan>')
        rows.append((start, f'<tspan fill="{COLORS["prompt"]}">❯ </tspan><tspan fill="{COLORS["cmd"]}">{"".join(chars)}</tspan>'))
        return start + 0.35 + len(command) * speed

    def print_line(segments: list[tuple[str, str | None, bool]], at: float) -> None:
        reveals.append((f"l{len(rows)}", at, "opacity"))
        rows.append((at, tspans(segments)))

    t = type_command("pip install elessar", 0.6) + 0.5
    print_line([(f"Successfully installed elessar-{__version__}", DIM, False)], t)
    t += 0.7
    t = type_command(f"elessar {csv_name} --target {target}", t) + 0.6
    for line in output:
        t += 0.3 if re.match(r"^[A-Z]", line) else 0.06
        print_line(color_line(line), t)
    t += 0.5
    print_line([("Report: ", DIM, False), (report, COLORS["title"], False)], t)
    t_open = t + 0.9
    t_scroll = t_open + 1.8
    total = t_scroll + SCROLL_SECONDS + HOLD

    def pct(at: float) -> str:
        return f"{100 * min(max(at / total, 0), 1):.3f}%"

    fade = pct(total - 0.6)
    css = [f"@keyframes {name}{{0%,{pct(at - 0.001)}{{{prop}:0}}{pct(at)},{fade}{{{prop}:1}}100%{{{prop}:0}}}}"
           f".{name}{{animation:{name} {total:.2f}s infinite}}" for name, at, prop in reveals]

    # terminal scrolling: once the window is full, each new line moves the content up by one line
    scroll = ["0%{transform:translateY(0)}"]
    for i, (appear, _) in enumerate(rows):
        if i >= VISIBLE:
            shift = -(i - VISIBLE + 1) * LINE_H
            scroll.append(f"{pct(appear - 0.04)}{{transform:translateY({shift + LINE_H}px)}}"
                          f"{pct(appear)}{{transform:translateY({shift}px)}}")
    last = -(len(rows) - VISIBLE) * LINE_H if len(rows) > VISIBLE else 0
    scroll.append(f"99.9%{{transform:translateY({last}px)}}100%{{transform:translateY(0)}}")
    css.append(f"@keyframes scroll{{{''.join(scroll)}}}.content{{animation:scroll {total:.2f}s infinite}}")

    # the browser window: pops in over the terminal, then the page scrolls slowly
    page_w = WIDTH - 2
    scale = page_w / shot.width
    page_h = round(shot.height * scale)
    view_h = HEIGHT - BROWSER_BAR - 1
    travel = page_h - view_h
    css.append(f"@keyframes browser{{0%,{pct(t_open - 0.01)}{{opacity:0;transform:translateY(18px) scale(.97)}}"
               f"{pct(t_open + 0.5)},{fade}{{opacity:1;transform:translateY(0) scale(1)}}100%{{opacity:0}}}}"
               f".browser{{transform-box:fill-box;transform-origin:50% 60%;animation:browser {total:.2f}s infinite}}")
    css.append(f"@keyframes page{{0%,{pct(t_scroll)}{{transform:translateY(0)}}"
               f"{pct(t_scroll + SCROLL_SECONDS)},100%{{transform:translateY(-{travel}px)}}}}"
               f".page{{animation:page {total:.2f}s infinite ease-in-out}}")
    css.append("@keyframes blink{0%,49%{fill-opacity:1}50%,100%{fill-opacity:0}}.cursor{animation:blink 1s step-end infinite}")
    css.append("@media (prefers-reduced-motion: reduce){*{animation:none!important;opacity:1!important}"
               ".content,.page{transform:none!important}.browser{opacity:1!important;transform:none!important}}")

    buf = io.BytesIO()
    shot.resize((page_w, page_h), Image.LANCZOS).save(buf, "JPEG", quality=70, optimize=True, progressive=True)
    data = base64.b64encode(buf.getvalue()).decode("ascii")

    body = [f'<text class="l{i}" x="{PAD_X}" y="{TOP + i * LINE_H}" xml:space="preserve">{content}</text>'
            for i, (_, content) in enumerate(rows)]
    font = "ui-monospace,SFMono-Regular,'SF Mono',Menlo,Consolas,'Liberation Mono','DejaVu Sans Mono',monospace"
    ui = "-apple-system,'Segoe UI',Inter,Lato,Helvetica,Arial,sans-serif"
    label = f"elessar installed and run on {csv_name} in a terminal, then its HTML report opening in a browser"
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}" role="img" aria-label="{html.escape(label)}">
<title>{html.escape(label)}</title>
<style>
text{{font-family:{font};font-size:{FONT_SIZE}px;fill:{FG};white-space:pre;opacity:0}}
.ui{{font-family:{ui};opacity:1}}
{chr(10).join(css)}
</style>
<defs>
<clipPath id="screen"><rect x="0" y="40" width="{WIDTH}" height="{HEIGHT - 40}"/></clipPath>
<clipPath id="viewport"><rect x="1" y="{BROWSER_BAR}" width="{page_w}" height="{view_h}"/></clipPath>
<clipPath id="window"><rect width="{WIDTH}" height="{HEIGHT}" rx="10"/></clipPath>
</defs>
<rect width="{WIDTH}" height="{HEIGHT}" rx="10" fill="{BG}"/>
<path d="M0 10a10 10 0 0 1 10-10h{WIDTH - 20}a10 10 0 0 1 10 10v26h-{WIDTH}z" fill="{BAR}"/>
<circle cx="20" cy="18" r="6" fill="#ff5f57"/><circle cx="40" cy="18" r="6" fill="#febc2e"/><circle cx="60" cy="18" r="6" fill="#28c840"/>
<text x="{WIDTH / 2}" y="22" text-anchor="middle" style="opacity:1;fill:{DIM};font-size:12px">elessar — zsh</text>
<g clip-path="url(#screen)"><g class="content">
{chr(10).join(body)}
</g></g>
<g class="browser" clip-path="url(#window)">
<rect width="{WIDTH}" height="{HEIGHT}" rx="10" fill="#f4f3ef"/>
<rect width="{WIDTH}" height="{BROWSER_BAR}" fill="#dfe1e5"/>
<circle cx="20" cy="18" r="6" fill="#ff5f57"/><circle cx="40" cy="18" r="6" fill="#febc2e"/><circle cx="60" cy="18" r="6" fill="#28c840"/>
<path d="M84 36v-20a8 8 0 0 1 8-8h196a8 8 0 0 1 8 8v20z" fill="#ffffff"/>
<text class="ui" x="102" y="27" style="font-size:12px;fill:#1f1f1f">{html.escape(csv_name)} · elessar</text>
<text class="ui" x="276" y="27" style="font-size:12px;fill:#5f6368">×</text>
<rect y="36" width="{WIDTH}" height="{BROWSER_BAR - 36}" fill="#ffffff"/>
<text class="ui" x="22" y="60" style="font-size:14px;fill:#5f6368">‹  ›  ↻</text>
<rect x="96" y="44" width="{WIDTH - 120}" height="24" rx="12" fill="#f1f3f4"/>
<text class="ui" x="114" y="60" style="font-size:12.5px;fill:#3c4043">file:///home/you/{html.escape(report)}</text>
<line x1="0" y1="{BROWSER_BAR - 0.5}" x2="{WIDTH}" y2="{BROWSER_BAR - 0.5}" stroke="#dadce0"/>
<g clip-path="url(#viewport)"><g class="page">
<image x="1" y="{BROWSER_BAR}" width="{page_w}" height="{page_h}" href="data:image/jpeg;base64,{data}"/>
</g></g>
</g>
</svg>
"""


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    csv, target, out = sys.argv[1:]
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        lines, report = run_elessar(csv, target, work)
        shot = screenshot(report)
        shown = str(report.relative_to(work))
    svg = build(excerpt(lines), shot, "train.csv", target, shown)
    Path(out).write_text(svg, encoding="utf-8")
    print(f"wrote {out} ({len(svg) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
