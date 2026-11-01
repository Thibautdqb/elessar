"""Build the public Elessar documentation with the Python standard library."""

from __future__ import annotations

import argparse
import html
import json
import shutil
from pathlib import Path
from urllib.parse import urlsplit
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parent
PAGES = [
    ("", "home", "Elessar — automated exploratory data analysis in Python",
     "Elessar is an MIT-licensed Python library for automated EDA, semantic column types, data-quality profiling, statistical analysis and HTML reports.", "en"),
    ("quickstart/", "quickstart", "Install Elessar and create your first Python EDA report",
     "Install elessar from PyPI, profile a pandas DataFrame or CSV file, choose a target and export an HTML data-quality report.", "en"),
    ("api/", "api", "Elessar Python API: see, load, StoneConfig and Vision",
     "Reference for elessar.see(), load(), semantic types, statistics, distributions, associations, data quality, StoneConfig and Vision exports.", "en"),
    ("guides/data-quality/", "data-quality", "Data-quality profiling and semantic types with Elessar",
     "Inspect missing values, duplicates, semantic column types and quality alerts in pandas with elessar.see_quality() and elessar.see_types().", "en"),
    ("guides/large-datasets/", "large-datasets", "Profile CSV, Parquet and large datasets with Elessar",
     "Use Elessar with CSV, Parquet, Excel, pandas, Polars, Arrow and Dask. Understand streaming, sampling and exact dataset counts.", "en"),
    ("fr/", "fr", "Elessar — analyse exploratoire et qualité des données en Python",
     "Elessar est une bibliothèque Python pour automatiser l’analyse exploratoire, détecter les types sémantiques et produire un rapport de qualité des données.", "fr"),
]


def build(base_url: str, output: Path) -> None:
    parts = urlsplit(base_url)
    if parts.scheme != "https" or not parts.netloc or parts.query or parts.fragment:
        raise ValueError("base_url must be an absolute HTTPS URL without a query or fragment")
    base_url = base_url.rstrip("/") + "/"
    metadata = json.loads((ROOT / "site.json").read_text(encoding="utf-8"))
    repo = metadata["repository"]
    version = metadata["version"]
    output.mkdir(parents=True, exist_ok=True)
    shutil.copytree(ROOT / "assets", output / "assets", dirs_exist_ok=True)
    esc = html.escape
    nav = [("", "Overview"), ("quickstart/", "Quickstart"), ("api/", "API"),
           ("guides/data-quality/", "Data quality"), ("guides/large-datasets/", "Large datasets")]
    for path, key, title, description, lang in PAGES:
        canonical = base_url + path
        navigation = "".join(
            f'<a href="{esc(base_url + dest)}"' + (' aria-current="page"' if dest == path else '') + f'>{label}</a>'
            for dest, label in nav
        )
        alternates = ""
        if key in {"home", "fr"}:
            alternates = (f'<link rel="alternate" hreflang="en" href="{esc(base_url)}">'
                          f'<link rel="alternate" hreflang="fr" href="{esc(base_url + "fr/")}">'
                          f'<link rel="alternate" hreflang="x-default" href="{esc(base_url)}">')
        schema = {
            "@context": "https://schema.org", "@type": "SoftwareSourceCode" if key == "home" else "TechArticle",
            "@id": canonical + "#document", "name": title, "description": description,
            "url": canonical, "inLanguage": lang,
        }
        if key == "home":
            schema.update({"name": "Elessar", "codeRepository": repo, "programmingLanguage": "Python",
                           "runtimePlatform": "Python 3.10 or newer", "version": version,
                           "license": repo + "/blob/main/LICENSE", "downloadUrl": "https://pypi.org/project/elessar/",
                           "author": {"@type": "Person", "name": "thibautdqb", "url": "https://github.com/Thibautdqb"},
                           "keywords": "Python, exploratory data analysis, EDA, data profiling, data quality, pandas"})
        else:
            schema["about"] = {"@type": "SoftwareSourceCode", "name": "Elessar", "codeRepository": repo}
        body = (ROOT / "content" / f"{key}.html").read_text(encoding="utf-8")
        for token, value in {"{{BASE}}": base_url, "{{REPO}}": repo, "{{VERSION}}": version}.items():
            body = body.replace(token, esc(value))
        schema_json = json.dumps(schema, ensure_ascii=False).replace("<", "\\u003c")
        page = f'''<!doctype html>
<html lang="{lang}">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(title)}</title>
<meta name="description" content="{esc(description)}">
<meta name="robots" content="index,follow,max-image-preview:large">
<link rel="canonical" href="{esc(canonical)}">
{alternates}
<meta property="og:type" content="website">
<meta property="og:site_name" content="Elessar">
<meta property="og:title" content="{esc(title)}">
<meta property="og:description" content="{esc(description)}">
<meta property="og:url" content="{esc(canonical)}">
<meta property="og:locale" content="{'fr_FR' if lang == 'fr' else 'en_US'}">
<meta name="twitter:card" content="summary">
<meta name="theme-color" content="#123d32">
<link rel="icon" href="{esc(base_url)}assets/elfstone.svg" type="image/svg+xml">
<link rel="stylesheet" href="{esc(base_url)}assets/style.css">
<link rel="alternate" type="text/markdown" href="{esc(base_url)}reference.md" title="Elessar reference as Markdown">
<script type="application/ld+json">{schema_json}</script>
<script src="{esc(base_url)}assets/site.js" defer></script>
</head>
<body>
<a class="skip" href="#main">{'Aller au contenu' if lang == 'fr' else 'Skip to content'}</a>
<header class="header"><div class="header-inner">
<a class="wordmark" href="{esc(base_url)}" aria-label="Elessar home"><img src="{esc(base_url)}assets/elfstone.svg" width="27" height="31" alt="">elessar<span> / python</span></a>
<div class="header-links"><a href="{esc(repo)}">GitHub ↗</a><a href="https://pypi.org/project/elessar/">PyPI ↗</a><a href="{esc(base_url + ('fr/' if lang == 'en' else ''))}" lang="{'fr' if lang == 'en' else 'en'}">{'Français' if lang == 'en' else 'English'}</a></div>
</div></header>
<nav class="navigation" aria-label="Documentation">{navigation}</nav>
<main id="main" class="{'home' if key == 'home' else 'document'}">{body}</main>
<footer><div><a class="footer-brand" href="{esc(base_url)}">elessar</a><p>Automated exploratory data analysis for Python.</p></div><div><p>Documented version {esc(version)} · Python ≥ 3.10 · MIT</p><p><a href="{esc(repo)}/tree/{esc(metadata['source_commit'])}">Versioned source</a> · <a href="{esc(repo)}/blob/main/VALIDATION.md">Statistical validation</a> · <a href="{esc(base_url)}reference.md">Markdown reference</a> · <a href="{esc(base_url)}llms.txt">llms.txt</a></p></div></footer>
<div id="copy-status" class="sr-only" role="status" aria-live="polite"></div>
</body></html>
'''
        destination = output / path / "index.html"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(page, encoding="utf-8")
    reference = (ROOT / "content/reference.md").read_text(encoding="utf-8").replace("{{BASE}}", base_url)
    for filename in ("reference.md", "llms-full.txt"):
        (output / filename).write_text(reference, encoding="utf-8")
    llms = f'''# Elessar

> Elessar (`elessar` on PyPI) is an MIT-licensed Python library for automated exploratory data analysis (EDA) and data-quality profiling. Requires Python >=3.10. Install with `pip install elessar`; start with `import elessar` and `elessar.see(data)`.

Documented version: {version}. Package source: {repo}/tree/{metadata['source_commit']}.

## Documentation

- [Overview]({base_url}): capabilities, installation and scope.
- [Quickstart]({base_url}quickstart/): a complete pandas example and CLI usage.
- [Python API]({base_url}api/): see, load, focused analysis functions, StoneConfig and Vision.
- [Data quality]({base_url}guides/data-quality/): missing values, duplicates and semantic types.
- [Large datasets and inputs]({base_url}guides/large-datasets/): formats, streaming and sampling limits.
- [Full Markdown reference]({base_url}reference.md): compact package facts and examples in plain text.
- [Documentation en français]({base_url}fr/): présentation et démarrage rapide.

## Primary package sources

- [PyPI](https://pypi.org/project/elessar/): release files and installation metadata.
- [GitHub]({repo}): code, README, changelog and issue tracker.
- [Statistical validation]({repo}/blob/main/VALIDATION.md): methods and limitations.
'''
    (output / "llms.txt").write_text(llms, encoding="utf-8")
    (output / "robots.txt").write_text(f"User-agent: *\nAllow: /\n\nSitemap: {base_url}sitemap.xml\n", encoding="utf-8")
    (output / ".nojekyll").touch()
    ET.register_namespace("", "http://www.sitemaps.org/schemas/sitemap/0.9")
    ns = "{http://www.sitemaps.org/schemas/sitemap/0.9}"
    sitemap = ET.Element(ns + "urlset")
    for path, *_ in PAGES:
        ET.SubElement(ET.SubElement(sitemap, ns + "url"), ns + "loc").text = base_url + path
    ET.ElementTree(sitemap).write(output / "sitemap.xml", encoding="utf-8", xml_declaration=True)
    key_file = ROOT / "indexnow.key"
    if key_file.exists():
        key = key_file.read_text(encoding="utf-8").strip()
        if not (len(key) == 32 and all(ch in "0123456789abcdef" for ch in key)):
            raise ValueError("Invalid IndexNow key")
        (output / f"{key}.txt").write_text(key, encoding="utf-8")
    (output / "404.html").write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="robots" content="noindex">'
        f'<title>Page not found — Elessar</title><h1>Page not found</h1><a href="{esc(base_url)}">Elessar documentation</a>',
        encoding="utf-8")
    print(f"Built {len(PAGES)} HTML pages, sitemap, robots.txt and Markdown references in {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", help="Public HTTPS origin; overrides site.json")
    parser.add_argument("--output", type=Path, default=ROOT / "dist")
    args = parser.parse_args()
    config = json.loads((ROOT / "site.json").read_text(encoding="utf-8"))
    build(args.base_url or config["base_url"], args.output.resolve())
