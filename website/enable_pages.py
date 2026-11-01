"""Enable GitHub Pages and start the existing workflow using your authenticated gh CLI."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess


def gh(arguments: list[str], payload: dict[str, str] | None = None) -> str:
    command = ["gh", *arguments]
    if payload is not None:
        command += ["--input", "-"]
    result = subprocess.run(command, input=json.dumps(payload) if payload is not None else None,
                            text=True, capture_output=True, check=False)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "GitHub CLI request failed")
    return result.stdout


def enable(repository: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
        raise ValueError("Use a repository in owner/name form")
    if shutil.which("gh") is None:
        raise RuntimeError("Install GitHub CLI and run gh auth login first")
    repo = json.loads(gh(["api", f"repos/{repository}"]))
    if repo.get("default_branch") != "main":
        raise RuntimeError("This workflow targets main; review its configuration before using another default branch")
    try:
        gh(["api", f"repos/{repository}/contents/.github/workflows/pages.yml?ref=main"])
    except RuntimeError as error:
        raise RuntimeError("The Pages workflow must be merged into main first. " + str(error)) from error
    if repo.get("has_pages"):
        pages = json.loads(gh(["api", f"repos/{repository}/pages"]))
        if pages.get("build_type") != "workflow":
            raise RuntimeError("Pages already uses a different publishing source; review Settings > Pages before changing it")
        print("GitHub Pages already uses GitHub Actions; configuration preserved.")
    else:
        gh(["api", "--method", "POST", f"repos/{repository}/pages"], {"build_type": "workflow"})
        print("GitHub Pages enabled with GitHub Actions as the publishing source.")
    gh(["workflow", "run", "pages.yml", "--ref", "main", "--repo", repository])
    print(f"Deployment workflow requested: https://github.com/{repository}/actions/workflows/pages.yml")
    print("Check that workflow for the published URL; requesting it does not prove deployment success.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="Thibautdqb/elessar")
    arguments = parser.parse_args()
    try:
        enable(arguments.repo)
    except (RuntimeError, ValueError) as error:
        raise SystemExit(str(error)) from None
