"""Check the generated HTML, canonical URLs and internal links before deployment."""

from __future__ import annotations

import argparse
import json
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urljoin, urlsplit
from xml.etree import ElementTree as ET


class Page(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.links: list[str] = []
        self.ids: set[str] = set()
        self.duplicate_ids: set[str] = set()
        self.h1_count = 0
        self.title = ""
        self.description = ""
        self.canonical = ""
        self.noindex = False
        self.schemas: list[str] = []
        self.in_title = False
        self.in_schema = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        identifier = attributes.get("id")
        if identifier:
            if identifier in self.ids:
                self.duplicate_ids.add(identifier)
            self.ids.add(identifier)
        if tag == "h1":
            self.h1_count += 1
        if tag == "title":
            self.in_title = True
        if tag == "link" and attributes.get("rel") == "canonical":
            self.canonical = attributes.get("href") or ""
        if tag == "meta" and attributes.get("name") == "description":
            self.description = attributes.get("content") or ""
        if tag == "meta" and attributes.get("name") == "robots":
            self.noindex = "noindex" in (attributes.get("content") or "").lower()
        if tag == "script" and attributes.get("type") == "application/ld+json":
            self.in_schema = True
            self.schemas.append("")
        for attribute in ("href", "src"):
            value = attributes.get(attribute)
            if value:
                self.links.append(value)

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self.in_title = False
        if tag == "script":
            self.in_schema = False

    def handle_data(self, data: str) -> None:
        if self.in_title:
            self.title += data
        if self.in_schema:
            self.schemas[-1] += data


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def check(root: Path, base_url: str) -> int:
    base_url = base_url.rstrip("/") + "/"
    base = urlsplit(base_url)
    require(base.scheme == "https" and bool(base.netloc), "An absolute HTTPS base URL is required")
    root = root.resolve()
    sitemap = ET.parse(root / "sitemap.xml")
    locations = [node.text or "" for node in sitemap.findall(".//{http://www.sitemaps.org/schemas/sitemap/0.9}loc")]
    require(bool(locations) and len(set(locations)) == len(locations), "The sitemap must contain distinct URLs")
    pages: dict[str, Page] = {}
    for canonical in locations:
        require(canonical.startswith(base_url), f"Sitemap URL outside the project prefix: {canonical}")
        relative = canonical[len(base_url):]
        path = root / relative / "index.html"
        require(path.is_file(), f"Missing HTML for {canonical}")
        source = path.read_text(encoding="utf-8")
        require("{{" not in source, f"Unresolved template in {path}")
        require(".chatgpt.site" not in source, f"Old hosting URL in {path}")
        page = Page()
        page.feed(source)
        require(page.h1_count == 1, f"Expected one h1 in {path}")
        require(not page.duplicate_ids, f"Repeated anchors in {path}")
        require(bool(page.title and page.description), f"Missing page title or description in {path}")
        require(page.canonical == canonical, f"Canonical URL mismatch in {path}")
        require(not page.noindex, f"Unexpected noindex in {path}")
        require(bool(page.schemas), f"Missing structured data in {path}")
        for schema in page.schemas:
            require(json.loads(schema).get("url") == canonical, f"Structured-data URL mismatch in {path}")
        pages[canonical] = page
    for field in ("title", "description"):
        require(len({getattr(page, field) for page in pages.values()}) == len(pages), f"Repeated page {field}")
    for canonical, page in pages.items():
        for link in page.links:
            target = urlsplit(urljoin(canonical, link))
            if target.netloc != base.netloc or target.scheme != base.scheme:
                continue
            require(target.path.startswith(base.path), f"Link escapes the Pages project prefix: {link}")
            relative = unquote(target.path[len(base.path):])
            local = (root / relative).resolve()
            require(local.is_relative_to(root), f"Link escapes the output directory: {link}")
            if target.path.endswith("/"):
                local /= "index.html"
            require(local.is_file(), f"Broken internal link from {canonical}: {link}")
            target_page = pages.get(target._replace(query="", fragment="").geturl())
            if target.fragment and target_page is not None:
                require(target.fragment in target_page.ids, f"Missing anchor in {link}")
    html_files = {path.resolve() for path in root.rglob("index.html")}
    indexed_files = {(root / url[len(base_url):] / "index.html").resolve() for url in pages}
    require(html_files == indexed_files, "The sitemap must cover all generated index pages")
    require((root / ".nojekyll").is_file(), "Missing .nojekyll marker")
    require((root / "llms-full.txt").read_bytes() == (root / "reference.md").read_bytes(), "Markdown references disagree")
    for name in ("llms.txt", "llms-full.txt", "reference.md", "robots.txt"):
        text = (root / name).read_text(encoding="utf-8")
        require("{{" not in text and ".chatgpt.site" not in text, f"Stale URL or unresolved template in {name}")
    require(f"Sitemap: {base_url}sitemap.xml" in (root / "robots.txt").read_text(), "Wrong sitemap location")
    return len(pages)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--base-url", required=True)
    arguments = parser.parse_args()
    count = check(arguments.directory, arguments.base_url)
    print(f"Checked {count} pages: canonical URLs, project-prefix links, anchors, metadata and sitemap are valid.")
