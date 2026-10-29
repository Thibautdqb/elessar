"""Render the decision trees of the README (docs/diagrams/*.mmd, mermaid) to PNG images.

GitHub draws mermaid itself, PyPI does not: the README shows these images on both. Rendered with
mermaid in headless Chrome or Chromium (which must be installed), in the dark style of the report.

Usage:  python docs/make_diagrams.py
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

from PIL import Image

HERE = Path(__file__).parent / "diagrams"
BG = "#0b111b"
THEME = {
    "darkMode": True, "background": BG, "fontFamily": "Inter, Lato, Helvetica, Arial, sans-serif", "fontSize": "15px",
    "primaryColor": "#132031", "primaryTextColor": "#eaf6fa", "primaryBorderColor": "#2fa9c6",
    "secondaryColor": "#132031", "tertiaryColor": "#132031", "lineColor": "#7489a0", "textColor": "#a7bac8",
    "edgeLabelBackground": BG, "clusterBkg": "#0e1624",
}
PAGE = """<!doctype html><html><head><meta charset="utf-8"><style>
body {{ margin:0; background:{bg}; }} #out {{ display:inline-block; padding:28px; background:{bg}; }}
</style></head><body><div id="out"></div><script type="module">
import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@11.4.1/dist/mermaid.esm.min.mjs";
mermaid.initialize({{ startOnLoad:false, theme:"base", themeVariables:{theme}, flowchart:{{ curve:"basis", padding:14, useMaxWidth:false }} }});
const {{ svg }} = await mermaid.render("d", {source});
document.getElementById("out").innerHTML = svg;
</script></body></html>"""


def render(mmd: Path, chrome: str, work: Path) -> Path:
    page = work / f"{mmd.stem}.html"
    page.write_text(PAGE.format(bg=BG, theme=json.dumps(THEME), source=json.dumps(mmd.read_text())), encoding="utf-8")
    shot = work / f"{mmd.stem}.png"
    subprocess.run([chrome, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--force-device-scale-factor=2",
                    "--virtual-time-budget=15000", "--window-size=2600,3400", f"--screenshot={shot}", page.as_uri()],
                   check=True, capture_output=True)
    img = Image.open(shot).convert("RGB")
    bg = Image.new("RGB", img.size, BG)
    from PIL import ImageChops

    box = ImageChops.difference(img, bg).getbbox()  # crop to the drawing plus a margin
    pad = 40
    if box:
        img = img.crop((max(box[0] - pad, 0), max(box[1] - pad, 0), min(box[2] + pad, img.width), min(box[3] + pad, img.height)))
    out = mmd.with_suffix(".png")
    img.save(out, optimize=True)
    return out


def main() -> None:
    chrome = next((c for c in ("google-chrome", "chromium", "chromium-browser") if shutil.which(c)), None)
    if chrome is None:
        raise SystemExit("needs google-chrome or chromium")
    with tempfile.TemporaryDirectory() as tmp:
        for mmd in sorted(HERE.glob("*.mmd")):
            out = render(mmd, chrome, Path(tmp))
            print(f"wrote {out} {Image.open(out).size}")


if __name__ == "__main__":
    main()
