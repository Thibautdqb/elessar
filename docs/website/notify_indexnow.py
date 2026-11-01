"""Notify IndexNow once for the built sitemap. Dry run unless --submit is given."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submit", action="store_true", help="Verify the public key and send the six URLs")
    args = parser.parse_args()
    config = json.loads((ROOT / "site.json").read_text(encoding="utf-8"))
    origin = config["base_url"].rstrip("/")
    host = urlsplit(origin).netloc
    key = (ROOT / "indexnow.key").read_text(encoding="utf-8").strip()
    urls = [n.text for n in ET.parse(ROOT / "dist/sitemap.xml").findall(
        ".//{http://www.sitemaps.org/schemas/sitemap/0.9}loc")]
    if not urls or any(urlsplit(url).netloc != host for url in urls):
        raise SystemExit("The sitemap must contain only this site's URLs")
    print(f"IndexNow: {len(urls)} documentation URLs on {host}")
    if not args.submit:
        print("Dry run. Publish the site, then pass --submit to notify participating engines.")
        return
    key_url = f"{origin}/{key}.txt"
    headers = {"User-Agent": f"Elessar-Documentation/1.0 (+{origin}/)"}
    stage = "ownership verification"
    try:
        with urlopen(Request(key_url, headers=headers), timeout=20) as response:
            if response.status != 200 or response.read().decode("utf-8").strip() != key:
                raise SystemExit("The public ownership file is unavailable or does not match")
        payload = json.dumps({"host": host, "key": key, "keyLocation": key_url, "urlList": urls}).encode("utf-8")
        stage = "URL submission"
        request = Request("https://api.indexnow.org/indexnow", data=payload,
                          headers={**headers, "Content-Type": "application/json; charset=utf-8"}, method="POST")
        with urlopen(request, timeout=20) as response:
            status = "submission received; ownership verification pending" if response.status == 202 else "submission received"
            print(f"HTTP {response.status}: {status}; indexing is not guaranteed.")
    except HTTPError as error:
        raise SystemExit(f"IndexNow {stage} failed with HTTP {error.code}; no indexing result is implied") from None
    except (URLError, TimeoutError) as error:
        raise SystemExit(f"Network request unavailable: {error.reason if isinstance(error, URLError) else 'timeout'}") from None


if __name__ == "__main__":
    main()
