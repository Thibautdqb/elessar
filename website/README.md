# Elessar promotion website on GitHub Pages

This directory preserves the HTML, CSS and JavaScript of the Elessar presentation
site and prepares it for https://thibautdqb.github.io/elessar/.
It contains no hosting dependency on ChatGPT Sites. The Python package source,
root README, pyproject.toml and PyPI publishing workflow are outside this change.

## Build and check

Only Python 3.10 or newer is needed; no package installation is required.

```bash
python website/build.py
python website/check_site.py website/dist --base-url https://thibautdqb.github.io/elessar/
```

`website/dist/` contains the six HTML pages and their assets, the sitemap and the
Markdown references. It is the complete static deployment artifact. The build
keeps `/elessar/` in asset URLs, navigation, canonical links and sitemap entries.
`--base-url` overrides `site.json` for another GitHub Pages URL or a custom domain.
The workflow uses the URL supplied by GitHub's `configure-pages` action.

## First activation, after this PR is reviewed and merged

An administrator enables Pages once. In repository **Settings > Pages**, select
**GitHub Actions** as the build and deployment source. Then run **Elessar website**
from the Actions tab. The workflow can fail its initial push run until this
one-time Pages setting has been enabled; rerun it after activation.

Alternatively, with an authenticated GitHub CLI, use one command to enable Pages
and start deployment:

```bash
python website/enable_pages.py
```

This helper first checks that the workflow exists on `main`. If Pages is absent,
it creates a workflow-based Pages site through GitHub's API, then dispatches
`pages.yml`. If Pages already uses Actions, it preserves that setting. If an
existing Pages site uses another source, it stops for review. It does not edit
files, merge the PR, push commits or create a PyPI release.

Creating the Pages configuration requires repository administration and Pages
write permissions. Dispatching the workflow also requires access to run Actions.
The standard workflow `GITHUB_TOKEN` cannot perform initial Pages enablement by
itself; use the repository Settings UI or your authenticated CLI session. No
personal token is stored in this repository or needed by routine deployments.

The GitHub integration used to prepare this PR does not expose Pages settings,
so this PR does not represent Pages activation or a successful deployment.

## Routine automation

- Pull requests changing `website/` or `pages.yml` build and check the site and
  upload the HTML artifact. They do not deploy to production.
- After merge, pushes changing those paths on `main` build, check and publish
  automatically with GitHub's Pages actions.
- Manual workflow runs support redeployment and an optional IndexNow notification.
- The deployment summary reports GitHub's actual published URL.

IndexNow notifications are off by default. When selected, the helper verifies
the public ownership file before submitting only URLs within this project's
path. HTTP 202 means ownership verification is pending, not that pages are indexed.
Google Search Console ownership and sitemap submission are separate operations.

On a project site, `https://thibautdqb.github.io/robots.txt` is the host-level
robots policy. The file generated under `/elessar/robots.txt` cannot change that
host-level policy. Submit `/elessar/sitemap.xml` through a verified Search Console
property to advertise this project's sitemap directly.

## Updating the content

Edit `content/*.html` and `content/reference.md`, keeping package facts and examples
consistent. `site.json` records the documented package version and source commit;
it does not claim to track future releases automatically. The present content
documents Elessar 1.1.0. Preserve the factual review when updating those values.

Primary references:

- https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages
- https://github.com/actions/configure-pages/blob/v5/action.yml
- https://docs.github.com/en/rest/pages/pages#create-a-github-pages-site
- https://www.indexnow.org/documentation
