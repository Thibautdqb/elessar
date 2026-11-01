# Elessar discoverability

Public documentation: https://elessar-python.fizzyknoll28.chatgpt.site/

The public site documents Elessar 1.1.0, with examples checked against commit
`2262600bc335e535297bfc64b05b646e6172590f`. Its source is mirrored in `docs/website/`.
Rebuild it with `python docs/website/build.py`; no third-party build dependency is required.
Production hosting identity is retained in the registered Site's source repository.
The mirror in this repository does not itself trigger a hosted deployment.

## Implemented

- Public overview, quickstart, API, data-quality and large-input guides, plus a French introduction.
- Complete HTML text that does not depend on JavaScript rendering.
- Unique page titles, descriptions, canonical URLs and relevant structured data.
- Reciprocal English/French overview language links.
- A sitemap of the six canonical HTML URLs and a robots.txt allowing crawlers.
- Compact Markdown API facts, llms.txt and llms-full.txt as optional reading aids.
- README links and a descriptive heading identifying this Python package.
- Homepage/Documentation package URLs prepared for the next PyPI release.
- An IndexNow ownership file and a submission helper.

These changes improve access to the documentation. They do not prove indexation,
rankings, AI citations or inclusion in any model's training data.

Verification on 2026-11-01: all six production HTML pages returned HTTP 200 with
distinct titles, with no meta or HTTP-header noindex directive. The sitemap,
robots.txt and llms.txt were public. The documented English/French quickstarts,
quality checks, CSV example, charts and HTML/Markdown/JSON/CSV exports executed
successfully against the 1.1.0 source. Local browser preview was unavailable;
visual browser QA was not completed.

IndexNow returned HTTP 202 for the six-URL submission on 2026-11-01: URLs received,
ownership-key validation pending. This is not evidence of indexing. Do not
resubmit repeatedly while waiting for validation or crawl diagnostics.

## After publication

1. Visit https://elessar-python.fizzyknoll28.chatgpt.site/sitemap.xml and confirm it is public.
2. To notify participating engines, run `python docs/website/notify_indexnow.py --submit`
   after building. Without `--submit`, the helper performs a dry run. An HTTP 200
   acknowledges receipt; it does not establish that a URL is indexed.
3. In Google Search Console, verify the URL-prefix property for the public origin
   and submit `sitemap.xml`. If Google gives you an HTML verification file, add that
   exact file to the hosted output and deploy it before completing verification.
   Use URL inspection to request indexing for the overview and quickstart.
4. Use Bing Webmaster Tools for crawl/index diagnostics and the sitemap.
5. Inspect real search impressions and referrals over time. A `site:` search is not
   a complete index-status report.

Google's Indexing API is not intended for ordinary library documentation. Sitemap
submission and Search Console URL inspection are the appropriate supported routes.
Search Console and Webmaster Tools actions require the maintainer's own account
and verified ownership; none are represented here as already completed.

## PyPI metadata

PyPI release descriptions and uploaded artifacts are versioned. Changes to this
repository's README or pyproject.toml do not rewrite the existing 1.1.0 release page.
The public website is available immediately; ship the updated package metadata in
the next normal release using the existing release workflow. No package version
or release tag was changed for this documentation work.

## Keeping the facts current

When releasing a new package version, review `docs/website/site.json`, the versioned
source links, visible examples and `content/reference.md` together. Rebuild and
check internal links, the six canonical URLs, the sitemap and the example outputs.
Do not advertise benchmarks, compatibility or features absent from the documented release.

Useful, independently written tutorials, reproducible examples and legitimate
downstream usage can help people discover the package. Prefer specific, accurate
material to bulk posts or artificial mentions. Publish announcements under the
maintainer's own editorial control.

## Primary documentation

- [Google: AI features and your website](https://developers.google.com/search/docs/appearance/ai-features)
- [Google: build and submit a sitemap](https://developers.google.com/search/docs/crawling-indexing/sitemaps/build-sitemap)
- [Google: Indexing API scope](https://developers.google.com/search/apis/indexing-api/v3/using-api)
- [OpenAI: search and training crawlers](https://developers.openai.com/api/docs/bots)
- [IndexNow: protocol and ownership verification](https://www.indexnow.org/documentation)

Google states that its AI search features use existing SEO requirements; there is
no required special AI text file. OpenAI distinguishes its search crawler from its
training crawler. Allowing crawlers and publishing llms.txt do not force a provider
to read, recommend, cite or train on Elessar.
