"""Download 500 recent Apache Jira issues with issue histories."""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen


API_URL = "https://issues.apache.org/jira/rest/api/2/search"
SAMPLE_SIZE = 500
PAGE_SIZE = 100
OUTPUT_PATH = Path("data/samples/apache_recent_500.ndjson")


def get_page(start_at: int, max_results: int) -> dict:
    query = urlencode(
        {
            "jql": "ORDER BY created DESC",
            "startAt": start_at,
            "maxResults": max_results,
            "expand": "changelog",
        }
    )
    request = Request(
        f"{API_URL}?{query}",
        headers={"User-Agent": "SprintIntelligenceDatasetSampler/1.0"},
    )
    with urlopen(request, timeout=60) as response:
        return json.load(response)


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    downloaded: list[dict] = []
    while len(downloaded) < SAMPLE_SIZE:
        payload = get_page(len(downloaded), min(PAGE_SIZE, SAMPLE_SIZE - len(downloaded)))
        issues = payload.get("issues", [])
        if not issues:
            break
        downloaded.extend(issues)
        print(f"Downloaded {len(downloaded)} of {SAMPLE_SIZE} issues")

    with OUTPUT_PATH.open("w", encoding="utf-8") as output:
        for issue in downloaded:
            output.write(json.dumps(issue, ensure_ascii=False) + "\n")
    print(f"Saved {len(downloaded)} issues to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
