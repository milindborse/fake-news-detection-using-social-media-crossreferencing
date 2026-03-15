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

_HN_ITEM_BASE_URL = "https://news.ycombinator.com/item?id="


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
    query = build_search_query(claim, max_keywords=4)
    meta: dict[str, Any] = {"source": "hackernews", "query": query}

    records = _search_hn(query, limit, meta)

    # Fallback: retry with fewer keywords for a broader search
    if not records and not meta.get("error"):
        broader_query = build_search_query(claim, max_keywords=2)
        if broader_query != query:
            meta["fallback_query"] = broader_query
            records = _search_hn(broader_query, limit, meta)

    meta["count"] = len(records)
    if "error" not in meta:
        meta["error"] = False
    return records, meta


def _search_hn(query: str, limit: int, meta: dict[str, Any]) -> list[HNRecord]:
    """Execute a single HN Algolia search and return records.

    Results are post-filtered to require at least 2 query keywords
    in the title or story text, preventing loosely related stories
    from inflating the credibility score.
    """
    encoded_query = urllib.parse.quote(query)
    url = _HN_SEARCH_URL.format(query=encoded_query, limit=limit)

    headers = {"User-Agent": "news-credibility-checker/0.1"}
    keywords = set(query.lower().split())

    records: list[HNRecord] = []

    for attempt in range(config.MAX_RETRIES + 1):
        try:
            req = Request(url, headers=headers)  # noqa: S310
            with urlopen(req, timeout=config.REQUEST_TIMEOUT) as resp:  # noqa: S310
                raw = resp.read()

            data = json.loads(raw.decode("utf-8", errors="replace"))
            hits = data.get("hits") or []

            for hit in hits:
                if len(records) >= limit:
                    break
                title = hit.get("title", "")
                story_text = hit.get("story_text") or ""
                combined = (title + " " + story_text).lower()

                # Require at least 2 keyword matches to filter loosely related stories
                overlap = sum(1 for kw in keywords if kw in combined and len(kw) >= 3)
                if overlap < 2 and keywords:
                    continue

                story_url = hit.get("url") or f"{_HN_ITEM_BASE_URL}{hit.get('objectID', '')}"
                created_at = hit.get("created_at", "")
                points = hit.get("points") or 0
                num_comments = hit.get("num_comments") or 0

                records.append(
                    {
                        "source": "hackernews",
                        "title": title,
                        "text": story_text[:500],
                        "author": hit.get("author", ""),
                        "points": points,
                        "num_comments": num_comments,
                        "created_utc": created_at,
                        "url": story_url,
                        "domain": _domain_from_url(story_url),
                        "is_quality": points >= config.HN_MIN_POINTS,
                    }
                )

            return records

        except Exception as exc:  # noqa: BLE001
            if attempt < config.MAX_RETRIES:
                time.sleep(2 ** attempt)
                continue
            meta.update({"error": True, "message": str(exc)})
            return []

    return records


def _domain_from_url(url: str) -> str:
    """Extract domain from a URL string."""
    try:
        parsed = urllib.parse.urlparse(url)
        return parsed.netloc.lstrip("www.")
    except Exception:  # noqa: BLE001
        return ""
