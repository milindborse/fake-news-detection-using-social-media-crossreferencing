"""
Hacker News collector: fetches stories related to a claim using the free
Algolia HN Search API (no authentication required).

API docs: https://hn.algolia.com/api
"""

import json
import time
import urllib.parse
from datetime import datetime, timezone
from typing import Any
from urllib.request import Request, urlopen

from src import config
from src.nlp.claim_normalizer import build_search_query

# ---------------------------------------------------------------------------
# Type alias for a normalised Hacker News record
# ---------------------------------------------------------------------------
HNRecord = dict[str, Any]

_HN_SEARCH_URL = (
    "https://hn.algolia.com/api/v1/search"
    "?query={query}&tags=story&hitsPerPage={limit}"
)

# Minimum HN points for a story to be considered a quality signal
HN_MIN_POINTS: int = int(getattr(config, "HN_MIN_POINTS", 10))


def collect(
    claim: str,
    max_results: int | None = None,
) -> tuple[list[HNRecord], dict[str, Any]]:
    """Search Hacker News for stories related to *claim*.

    Args:
        claim: The news claim text to search for.
        max_results: Override the default max results from config.

    Returns:
        A tuple of ``(records, meta)`` where *records* is a (possibly empty)
        list of normalised dicts and *meta* contains status/error info.
    """
    if not config.ENABLE_HACKERNEWS:
        return [], {"source": "hackernews", "skipped": True, "reason": "disabled in config"}

    limit = max_results if max_results is not None else config.HN_MAX_RESULTS
    query = build_search_query(claim, max_keywords=6)
    meta: dict[str, Any] = {"source": "hackernews", "query": query}

    encoded_query = urllib.parse.quote(query)
    url = _HN_SEARCH_URL.format(query=encoded_query, limit=limit)

    headers = {"User-Agent": "news-credibility-checker/0.1"}

    records: list[HNRecord] = []

    for attempt in range(config.MAX_RETRIES + 1):
        try:
            req = Request(url, headers=headers)  # noqa: S310
            with urlopen(req, timeout=config.REQUEST_TIMEOUT) as resp:  # noqa: S310
                raw = resp.read()

            data = json.loads(raw.decode("utf-8", errors="replace"))
            hits = data.get("hits") or []

            for hit in hits[:limit]:
                story_url = hit.get("url") or f"https://news.ycombinator.com/item?id={hit.get('objectID', '')}"
                created_at = hit.get("created_at", "")
                points = hit.get("points") or 0
                num_comments = hit.get("num_comments") or 0

                records.append(
                    {
                        "source": "hackernews",
                        "title": hit.get("title", ""),
                        "text": (hit.get("story_text") or "")[:500],
                        "author": hit.get("author", ""),
                        "points": points,
                        "num_comments": num_comments,
                        "created_utc": created_at,
                        "url": story_url,
                        "domain": _domain_from_url(story_url),
                        "is_quality": points >= HN_MIN_POINTS,
                    }
                )

            meta["count"] = len(records)
            meta["error"] = False
            return records, meta

        except Exception as exc:  # noqa: BLE001
            if attempt < config.MAX_RETRIES:
                time.sleep(2 ** attempt)
                continue
            meta.update({"error": True, "message": str(exc)})
            return [], meta

    return records, meta


def _domain_from_url(url: str) -> str:
    """Extract domain from a URL string."""
    try:
        parsed = urllib.parse.urlparse(url)
        return parsed.netloc.lstrip("www.")
    except Exception:  # noqa: BLE001
        return ""
