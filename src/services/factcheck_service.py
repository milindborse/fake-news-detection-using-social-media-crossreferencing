"""
Google Fact Check Tools API integration.

Queries Google's Fact Check API to retrieve professional fact-check
verdicts for a given claim.  When no API key is configured, falls back
to searching fact-check RSS feeds and DuckDuckGo site-scoped searches
of known fact-check publishers.

Usage:
    from src.services.factcheck_service import search_fact_checks
    results = search_fact_checks("COVID vaccines cause infertility")
"""

import logging
import re
import urllib.parse
from typing import Any
from urllib.request import Request, urlopen
from xml.etree import ElementTree as ET

import requests

from src import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Fact-check RSS feeds (free, no key required)
# ---------------------------------------------------------------------------
_FACTCHECK_RSS_FEEDS: dict[str, str] = {
    "https://www.snopes.com/feed/": "Snopes",
    "https://www.politifact.com/rss/all/": "PolitiFact",
    "https://www.factcheck.org/feed/": "FactCheck.org",
    "https://fullfact.org/feed/rss.xml": "Full Fact",
}

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

FactCheckResult = dict[str, Any]
"""Structured fact-check result with keys:
   claim, rating, publisher, url, confidence.
"""


def search_fact_checks(claim_text: str) -> list[FactCheckResult]:
    """Query the Google Fact Check Tools API for *claim_text*.

    When no API key is configured, falls back to searching fact-check
    RSS feeds and DuckDuckGo site-scoped searches.

    Args:
        claim_text: The normalised claim to look up.

    Returns:
        A list of :data:`FactCheckResult` dicts.  Empty list when no
        results are found or the API is unavailable.
    """
    api_key = config.GOOGLE_FACTCHECK_API_KEY
    if not api_key:
        logger.info("GOOGLE_FACTCHECK_API_KEY not set – using free fact-check fallback.")
        return _search_factcheck_free(claim_text)

    url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
    params = {
        "query": claim_text,
        "key": api_key,
        "languageCode": "en",
        "pageSize": config.FACTCHECK_MAX_RESULTS,
    }

    try:
        response = requests.get(url, params=params, timeout=config.REQUEST_TIMEOUT)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as exc:
        logger.error("Fact Check API request failed: %s", exc)
        return _search_factcheck_free(claim_text)

    results = _parse_response(data)
    # If API returned nothing, try free fallback too
    if not results:
        results = _search_factcheck_free(claim_text)
    return results


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Mapping from common textual ratings to a normalised confidence float.
_RATING_CONFIDENCE: dict[str, float] = {
    "pants on fire": 0.05,
    "mostly false": 0.25,
    "mostly true": 0.80,
    "half true": 0.50,
    "misleading": 0.30,
    "unproven": 0.40,
    "mixture": 0.50,
    "true": 0.95,
    "false": 0.10,
}


def _normalise_rating(raw_rating: str) -> float:
    """Map a textual fact-check rating to a 0-1 confidence float."""
    lower = raw_rating.strip().lower()
    # Check for exact match first
    if lower in _RATING_CONFIDENCE:
        return _RATING_CONFIDENCE[lower]
    # Then check for substring match (longer keywords checked first)
    for keyword, score in _RATING_CONFIDENCE.items():
        if keyword in lower:
            return score
    # Default: neutral if rating text is unrecognised
    return 0.50


def _parse_response(data: dict[str, Any]) -> list[FactCheckResult]:
    """Parse the raw Google Fact Check API JSON into structured results."""
    results: list[FactCheckResult] = []
    claims = data.get("claims", [])

    for item in claims:
        claim_text = item.get("text", "")
        for review in item.get("claimReview", []):
            rating = review.get("textualRating", "Unknown")
            publisher_info = review.get("publisher", {})
            result: FactCheckResult = {
                "claim": claim_text,
                "rating": rating,
                "publisher": publisher_info.get("name", "Unknown"),
                "url": review.get("url", ""),
                "confidence": _normalise_rating(rating),
            }
            results.append(result)

    return results


# ---------------------------------------------------------------------------
# Free fact-check fallback (no API key required)
# ---------------------------------------------------------------------------

def _fetch_url(url: str, timeout: int = 10) -> bytes | None:
    """Fetch a URL with retry, returning raw bytes or None.

    Only allows HTTPS URLs to prevent SSRF attacks.
    """
    if not url.startswith("https://"):
        return None
    headers = {"User-Agent": "news-credibility-checker/0.1"}
    for attempt in range(config.MAX_RETRIES + 1):
        try:
            req = Request(url, headers=headers)  # noqa: S310
            with urlopen(req, timeout=timeout) as resp:  # noqa: S310
                return resp.read()
        except Exception:  # noqa: BLE001
            if attempt < config.MAX_RETRIES:
                import time
                time.sleep(1)
    return None


def _extract_keywords(text: str) -> list[str]:
    """Extract simple keywords from text for matching."""
    words = re.sub(r"[^a-z0-9\s]", " ", text.lower()).split()
    stopwords = {"a", "an", "the", "is", "are", "was", "were", "be", "to", "of",
                 "and", "or", "in", "on", "at", "for", "by", "with", "from", "that",
                 "this", "it", "not", "no", "has", "have", "had", "do", "does", "did"}
    return [w for w in words if w not in stopwords and len(w) >= 3]


def _search_factcheck_rss(claim_text: str) -> list[FactCheckResult]:
    """Search fact-check RSS feeds for items matching the claim keywords."""
    keywords = _extract_keywords(claim_text)
    if not keywords:
        return []

    results: list[FactCheckResult] = []
    limit = config.FACTCHECK_MAX_RESULTS

    for feed_url, publisher in _FACTCHECK_RSS_FEEDS.items():
        if len(results) >= limit:
            break
        raw = _fetch_url(feed_url, timeout=config.REQUEST_TIMEOUT)
        if not raw:
            continue
        try:
            root = ET.fromstring(raw.decode("utf-8", errors="replace"))
        except ET.ParseError:
            continue

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

            title = (title_el.text or "") if title_el is not None else ""
            link_text = ""
            if link_el is not None:
                link_text = link_el.text or link_el.get("href", "") or ""
            desc = re.sub(r"<[^>]+>", "", (desc_el.text or "") if desc_el is not None else "")[:300]

            combined = (title + " " + desc).lower()
            overlap = sum(1 for kw in keywords if kw in combined)
            if overlap < 1:
                continue

            # Try to infer a rating from the title/description
            rating = _infer_rating_from_text(title + " " + desc)

            results.append({
                "claim": title.strip(),
                "rating": rating,
                "publisher": publisher,
                "url": link_text.strip(),
                "confidence": _normalise_rating(rating),
            })
            if len(results) >= limit:
                break

    return results


def _search_factcheck_ddg(claim_text: str) -> list[FactCheckResult]:
    """Search DuckDuckGo scoped to fact-check sites for the claim."""
    keywords = _extract_keywords(claim_text)
    if not keywords:
        return []

    query_terms = " ".join(keywords[:5])
    site_scope = "site:snopes.com OR site:politifact.com OR site:factcheck.org"
    full_query = f"{query_terms} {site_scope}"
    encoded = urllib.parse.quote(full_query)
    url = f"https://html.duckduckgo.com/html/?q={encoded}"

    raw = _fetch_url(url, timeout=config.REQUEST_TIMEOUT)
    if not raw:
        return []

    html = raw.decode("utf-8", errors="replace")
    results: list[FactCheckResult] = []
    limit = config.FACTCHECK_MAX_RESULTS

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

        actual_url = href
        if "uddg=" in href:
            match = re.search(r"uddg=([^&]+)", href)
            if match:
                actual_url = urllib.parse.unquote(match.group(1))

        if not title:
            continue

        # Determine publisher from URL domain
        publisher = "Unknown"
        try:
            domain = urllib.parse.urlparse(actual_url).netloc.lower().lstrip("www.")
        except Exception:  # noqa: BLE001
            domain = ""
        if domain == "snopes.com" or domain.endswith(".snopes.com"):
            publisher = "Snopes"
        elif domain == "politifact.com" or domain.endswith(".politifact.com"):
            publisher = "PolitiFact"
        elif domain == "factcheck.org" or domain.endswith(".factcheck.org"):
            publisher = "FactCheck.org"

        rating = _infer_rating_from_text(title + " " + snippet)

        results.append({
            "claim": title,
            "rating": rating,
            "publisher": publisher,
            "url": actual_url,
            "confidence": _normalise_rating(rating),
        })

    return results


def _search_factcheck_free(claim_text: str) -> list[FactCheckResult]:
    """Combine RSS and DuckDuckGo searches for fact-check results."""
    results: list[FactCheckResult] = []

    # Try RSS feeds first
    try:
        rss_results = _search_factcheck_rss(claim_text)
        results.extend(rss_results)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Fact-check RSS search failed: %s", exc)

    # Fill remaining with DuckDuckGo site-scoped search
    remaining = config.FACTCHECK_MAX_RESULTS - len(results)
    if remaining > 0:
        try:
            ddg_results = _search_factcheck_ddg(claim_text)
            # Deduplicate by URL
            existing_urls = {r["url"] for r in results}
            for r in ddg_results:
                if len(results) >= config.FACTCHECK_MAX_RESULTS:
                    break
                if r["url"] not in existing_urls:
                    results.append(r)
                    existing_urls.add(r["url"])
        except Exception as exc:  # noqa: BLE001
            logger.debug("Fact-check DuckDuckGo search failed: %s", exc)

    return results


def _infer_rating_from_text(text: str) -> str:
    """Infer a fact-check rating from title/description text."""
    lower = text.lower()
    if any(kw in lower for kw in ("pants on fire", "four pinocchios")):
        return "Pants on Fire"
    if any(kw in lower for kw in ("mostly false", "mostly wrong")):
        return "Mostly False"
    if any(kw in lower for kw in ("false", "fake", "hoax", "debunked", "fabricated")):
        return "False"
    if any(kw in lower for kw in ("misleading", "out of context", "missing context")):
        return "Misleading"
    if any(kw in lower for kw in ("mixture", "mixed", "half true", "half-true")):
        return "Mixture"
    if any(kw in lower for kw in ("mostly true", "mostly correct")):
        return "Mostly True"
    if any(kw in lower for kw in ("true", "correct", "accurate", "confirmed")):
        return "True"
    if any(kw in lower for kw in ("unproven", "unverified", "legend")):
        return "Unproven"
    return "Under Review"
