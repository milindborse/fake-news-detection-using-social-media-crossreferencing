"""
Google Fact Check Tools API integration.

Queries Google's Fact Check API to retrieve professional fact-check
verdicts for a given claim.  Returns structured results that can be
incorporated into the credibility scoring pipeline.

Usage:
    from src.services.factcheck_service import search_fact_checks
    results = search_fact_checks("COVID vaccines cause infertility")
"""

import logging
from typing import Any

import requests

from src import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

FactCheckResult = dict[str, Any]
"""Structured fact-check result with keys:
   claim, rating, publisher, url, confidence.
"""


def search_fact_checks(claim_text: str) -> list[FactCheckResult]:
    """Query the Google Fact Check Tools API for *claim_text*.

    Args:
        claim_text: The normalised claim to look up.

    Returns:
        A list of :data:`FactCheckResult` dicts.  Empty list when no
        results are found or the API is unavailable.
    """
    api_key = config.GOOGLE_FACTCHECK_API_KEY
    if not api_key:
        logger.warning("GOOGLE_FACTCHECK_API_KEY not set – skipping fact-check lookup.")
        return []

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
        return []

    return _parse_response(data)


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
