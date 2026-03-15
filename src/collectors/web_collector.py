"""
Web / news collector: fetches corroborating mentions from free web sources.

Strategy (in priority order, all free):
1. GDELT GKG API  – real-time news event data (no key required)
2. RSS feeds from Reuters, BBC, AP News, NPR  – no key required
3. DuckDuckGo instant-answer (HTML scrape)  – last resort, no key required

Returns normalised records with title, snippet, domain, published_date, url.
"""

import re
import time
import urllib.parse
from datetime import datetime
from typing import Any
from urllib.request import Request, urlopen
from xml.etree import ElementTree as ET

from src import config
from src.nlp.claim_normalizer import build_search_query

WebRecord = dict[str, Any]

# ---------------------------------------------------------------------------
# Free RSS feeds from reputable sources (sampled on startup)
# ---------------------------------------------------------------------------
_RSS_FEEDS: list[str] = [
    "https://feeds.reuters.com/reuters/topNews",
    "https://feeds.bbci.co.uk/news/rss.xml",
    "https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml",
    "https://www.theguardian.com/world/rss",
    "https://feeds.npr.org/1001/rss.xml",
    "https://apnews.com/apf-topnews?format=rss",
    "https://feeds.bbci.co.uk/news/world/rss.xml",
    "https://feeds.bbci.co.uk/news/science_and_environment/rss.xml",
    "https://rss.nytimes.com/services/xml/rss/nyt/World.xml",
    "https://www.theguardian.com/science/rss",
]

_GDELT_BASE = (
    "https://api.gdeltproject.org/api/v2/doc/doc"
    "?mode=artlist&maxrecords={limit}&format=json"
    "&query={query}&sort=hybridrel"
)


def _domain_from_url(url: str) -> str:
    """Extract domain from a URL string."""
    try:
        parsed = urllib.parse.urlparse(url)
        return parsed.netloc.lstrip("www.")
    except Exception:  # noqa: BLE001
        return ""


def _fetch_url(url: str, timeout: int = 10) -> bytes | None:
    """Fetch a URL with a simple retry loop, returning raw bytes or None."""
    headers = {"User-Agent": config.REDDIT_USER_AGENT or "news-credibility-checker/0.1"}
    for attempt in range(config.MAX_RETRIES + 1):
        try:
            req = Request(url, headers=headers)  # noqa: S310
            with urlopen(req, timeout=timeout) as resp:  # noqa: S310
                return resp.read()
        except Exception:  # noqa: BLE001
            if attempt < config.MAX_RETRIES:
                time.sleep(1)
    return None


def _collect_gdelt(query: str, limit: int) -> list[WebRecord]:
    """Fetch news articles from GDELT API.

    Uses a shorter keyword query (top 4) to improve recall.
    """
    # Use fewer keywords for a broader GDELT search
    short_query = " ".join(query.split()[:4])
    encoded_query = urllib.parse.quote(short_query)
    url = _GDELT_BASE.format(limit=limit, query=encoded_query)
    raw = _fetch_url(url, timeout=config.REQUEST_TIMEOUT)
    if not raw:
        return []

    import json

    try:
        data = json.loads(raw.decode("utf-8", errors="replace"))
    except json.JSONDecodeError:
        return []

    articles = data.get("articles") or []
    records: list[WebRecord] = []
    for art in articles[:limit]:
        article_url = art.get("url", "")
        records.append(
            {
                "source": "web",
                "title": art.get("title", ""),
                "snippet": art.get("seendate", ""),
                "domain": _domain_from_url(article_url),
                "published_date": art.get("seendate", ""),
                "url": article_url,
                "provider": "gdelt",
            }
        )
    return records


def _collect_rss(query: str, limit: int) -> list[WebRecord]:
    """Scan RSS feeds for items whose title/description matches *query* keywords."""
    keywords = set(query.lower().split())
    records: list[WebRecord] = []

    for feed_url in _RSS_FEEDS:
        if len(records) >= limit:
            break
        raw = _fetch_url(feed_url, timeout=config.REQUEST_TIMEOUT)
        if not raw:
            continue
        try:
            root = ET.fromstring(raw.decode("utf-8", errors="replace"))
        except ET.ParseError:
            continue

        # Handle both RSS 2.0 and Atom
        items = root.findall(".//item")
        if not items:
            items = root.findall(".//{http://www.w3.org/2005/Atom}entry")

        for item in items:
            title_el = item.find("title")
            if title_el is None:
                title_el = item.find("{http://www.w3.org/2005/Atom}title")
            link_el = item.find("link")
            if link_el is None:
                link_el = item.find("{http://www.w3.org/2005/Atom}link")
            desc_el = item.find("description")
            if desc_el is None:
                desc_el = item.find("{http://www.w3.org/2005/Atom}summary")
            pub_el = item.find("pubDate")
            if pub_el is None:
                pub_el = item.find("{http://www.w3.org/2005/Atom}updated")

            title = (title_el.text or "") if title_el is not None else ""
            link_text = (link_el.text or link_el.get("href", "")) if link_el is not None else ""
            desc = re.sub(r"<[^>]+>", "", (desc_el.text or "") if desc_el is not None else "")[:300]
            pub = (pub_el.text or "") if pub_el is not None else ""

            # Simple keyword relevance check (relaxed: 1 keyword match is enough)
            combined = (title + " " + desc).lower()
            overlap = sum(1 for kw in keywords if kw in combined and len(kw) >= 3)
            if overlap < 1 and keywords:
                continue

            item_url = link_text.strip()
            records.append(
                {
                    "source": "web",
                    "title": title.strip(),
                    "snippet": desc.strip(),
                    "domain": _domain_from_url(item_url),
                    "published_date": pub.strip(),
                    "url": item_url,
                    "provider": "rss",
                }
            )
            if len(records) >= limit:
                break

    return records


def _collect_duckduckgo(query: str, limit: int) -> list[WebRecord]:
    """Scrape DuckDuckGo HTML search results as a last-resort fallback."""
    encoded_query = urllib.parse.quote(query)
    url = f"https://html.duckduckgo.com/html/?q={encoded_query}"
    raw = _fetch_url(url, timeout=config.REQUEST_TIMEOUT)
    if not raw:
        return []

    html = raw.decode("utf-8", errors="replace")
    records: list[WebRecord] = []

    # Parse result blocks: each result has class="result__a" for title/link
    # and class="result__snippet" for the snippet
    title_pattern = re.compile(
        r'<a[^>]+class="result__a"[^>]+href="([^"]*)"[^>]*>(.*?)</a>',
        re.DOTALL,
    )
    snippet_pattern = re.compile(
        r'<a[^>]+class="result__snippet"[^>]*>(.*?)</a>',
        re.DOTALL,
    )

    titles = title_pattern.findall(html)
    snippets = snippet_pattern.findall(html)

    for i, (href, raw_title) in enumerate(titles[:limit]):
        title = re.sub(r"<[^>]+>", "", raw_title).strip()
        snippet = ""
        if i < len(snippets):
            snippet = re.sub(r"<[^>]+>", "", snippets[i]).strip()

        # DuckDuckGo wraps URLs via a redirect; extract the actual URL
        actual_url = href
        if "uddg=" in href:
            match = re.search(r"uddg=([^&]+)", href)
            if match:
                actual_url = urllib.parse.unquote(match.group(1))

        if not title:
            continue

        records.append(
            {
                "source": "web",
                "title": title,
                "snippet": snippet[:300],
                "domain": _domain_from_url(actual_url),
                "published_date": "",
                "url": actual_url,
                "provider": "duckduckgo",
            }
        )

    return records


def collect(
    claim: str,
    max_results: int | None = None,
) -> tuple[list[WebRecord], dict[str, Any]]:
    """Collect web/news mentions related to *claim*.

    Tries GDELT first, then RSS feeds.

    Args:
        claim: The news claim text.
        max_results: Override default from config.

    Returns:
        Tuple of ``(records, meta)``.
    """
    if not config.ENABLE_WEB:
        return [], {"source": "web", "skipped": True, "reason": "disabled in config"}

    limit = max_results if max_results is not None else config.WEB_MAX_RESULTS
    query = build_search_query(claim, max_keywords=6)
    meta: dict[str, Any] = {"source": "web", "query": query}

    records: list[WebRecord] = []

    # 1. GDELT
    try:
        gdelt_records = _collect_gdelt(query, limit)
        records.extend(gdelt_records)
        meta["gdelt_count"] = len(gdelt_records)
    except Exception as exc:  # noqa: BLE001
        meta["gdelt_error"] = str(exc)

    # 2. RSS (fill remaining slots)
    remaining = limit - len(records)
    if remaining > 0:
        try:
            rss_records = _collect_rss(query, remaining)
            records.extend(rss_records)
            meta["rss_count"] = len(rss_records)
        except Exception as exc:  # noqa: BLE001
            meta["rss_error"] = str(exc)

    # 3. DuckDuckGo HTML scrape (fill remaining slots as last resort)
    remaining = limit - len(records)
    if remaining > 0:
        try:
            ddg_records = _collect_duckduckgo(query, remaining)
            records.extend(ddg_records)
            meta["duckduckgo_count"] = len(ddg_records)
        except Exception as exc:  # noqa: BLE001
            meta["duckduckgo_error"] = str(exc)

    # Deduplicate by URL
    seen_urls: set[str] = set()
    unique: list[WebRecord] = []
    for r in records:
        url = r.get("url", "")
        if url and url not in seen_urls:
            seen_urls.add(url)
            unique.append(r)

    meta["count"] = len(unique)
    meta["error"] = False
    return unique[:limit], meta
