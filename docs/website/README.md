# Elessar public documentation

Static documentation for [elessar on PyPI](https://pypi.org/project/elessar/).
Content documents version 1.1.0 and is checked against its versioned source.

Build using Python 3.10 or newer, with no third-party dependencies:

```bash
python build.py
```

`site.json` records the production origin, documented package version and source commit.
`build.py --base-url https://docs.example.org/` overrides the origin for a different host.
The generated `dist/` contains six HTML pages, accessible shared assets, a sitemap,
robots.txt, a Markdown reference and optional llms.txt/llms-full.txt convenience files.
The HTML contains all documentation text before JavaScript runs. JavaScript only adds
copy buttons. The site does not collect or analyse datasets.

Edit `content/*.html` for visible pages and `content/reference.md` for the compact
reference. Keep both consistent with the documented API. Review `site.json` whenever
the package version changes. Site sources are mirrored under `docs/website/` in
the Elessar GitHub repository; the hosted source repository retains hosting identity.

`indexnow.key` is a site-ownership verification value, published as a text file at
the host root. Its presence prepares IndexNow submission; building does not submit
URLs or demonstrate that any search engine indexed them.

After publication, `python notify_indexnow.py --submit` checks that ownership file
and notifies IndexNow for the sitemap URLs. Without `--submit`, it only reports a dry run.

## Discovery

- `robots.txt` allows crawling and advertises the absolute sitemap URL.
- Each HTML page has a unique title, description and canonical URL.
- The English overview and French page have reciprocal hreflang links.
- Homepage SoftwareSourceCode structured data matches visible package facts.
- The llms.txt files are optional reading aids, not an indexing or training standard.
- Search engines and AI providers decide whether and when to crawl, index, cite or
  train on content. Publication and a successful URL submission do not guarantee these outcomes.

To request Google indexing, verify ownership of the public origin in Search Console
and submit `sitemap.xml`; use URL inspection for a few key pages. Do not use Google's
Indexing API for these ordinary documentation pages. Bing Webmaster Tools and
IndexNow can notify participating engines after the ownership key is publicly available.
