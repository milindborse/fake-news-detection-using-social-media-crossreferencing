"""
Source reputation scoring module.

Assigns a 0–1 reputation weight to evidence sources based on their domain.
Higher scores indicate more trustworthy / authoritative sources.

Usage:
    from src.scoring.source_reputation import get_domain_score
    score = get_domain_score("https://www.reuters.com/article/123")
"""

from urllib.parse import urlparse

# ---------------------------------------------------------------------------
# Domain reputation database (0.0 = untrusted, 1.0 = highly trusted)
# ---------------------------------------------------------------------------

# Trusted global news organisations
_TRUSTED_NEWS: dict[str, float] = {
    "bbc.com": 0.95,
    "bbc.co.uk": 0.95,
    "reuters.com": 0.95,
    "apnews.com": 0.90,
    "nytimes.com": 0.90,
    "theguardian.com": 0.85,
    "washingtonpost.com": 0.85,
    "npr.org": 0.90,
    "pbs.org": 0.85,
    "economist.com": 0.85,
    "aljazeera.com": 0.80,
    "france24.com": 0.80,
    "dw.com": 0.80,
}

# Fact-checking organisations
_FACTCHECKERS: dict[str, float] = {
    "snopes.com": 0.95,
    "factcheck.org": 0.95,
    "politifact.com": 0.90,
    "fullfact.org": 0.90,
    "checkyourfact.com": 0.80,
}

# Scientific and official sources
_SCIENTIFIC_OFFICIAL: dict[str, float] = {
    "who.int": 0.95,
    "nature.com": 0.95,
    "science.org": 0.95,
    "scientificamerican.com": 0.90,
    "cdc.gov": 0.95,
    "nih.gov": 0.95,
    "gov.uk": 0.85,
    "europa.eu": 0.85,
    "un.org": 0.90,
    "nasa.gov": 0.95,
    "arxiv.org": 0.80,
    "pubmed.ncbi.nlm.nih.gov": 0.90,
}

# Reference platforms
_REFERENCE: dict[str, float] = {
    "wikipedia.org": 0.80,
    "en.wikipedia.org": 0.80,
    "britannica.com": 0.85,
}

# Social / community platforms
_SOCIAL_PLATFORMS: dict[str, float] = {
    "reddit.com": 0.50,
    "news.ycombinator.com": 0.55,
    "twitter.com": 0.40,
    "x.com": 0.40,
    "facebook.com": 0.35,
    "medium.com": 0.45,
    "quora.com": 0.45,
}

# Default score for unknown domains
_DEFAULT_SCORE: float = 0.30

# Merged lookup table
_DOMAIN_SCORES: dict[str, float] = {
    **_TRUSTED_NEWS,
    **_FACTCHECKERS,
    **_SCIENTIFIC_OFFICIAL,
    **_REFERENCE,
    **_SOCIAL_PLATFORMS,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_domain_score(url_or_domain: str) -> float:
    """Return a 0–1 reputation score for the given URL or bare domain.

    The function extracts the domain from a full URL and looks it up in
    the reputation database.  If the domain is not found, a default score
    of ``0.3`` is returned.

    Args:
        url_or_domain: A full URL (e.g. ``https://www.bbc.com/news/...``)
                       or a bare domain (e.g. ``bbc.com``).

    Returns:
        Float reputation score between 0.0 and 1.0.
    """
    domain = _extract_domain(url_or_domain)

    # Direct lookup
    if domain in _DOMAIN_SCORES:
        return _DOMAIN_SCORES[domain]

    # Try matching against known domains as suffixes
    # (e.g. "world.bbc.com" should match "bbc.com")
    for known_domain, score in _DOMAIN_SCORES.items():
        if domain.endswith("." + known_domain):
            return score

    return _DEFAULT_SCORE


def get_source_type(url_or_domain: str) -> str:
    """Classify the source into a human-readable category.

    Args:
        url_or_domain: A full URL or bare domain.

    Returns:
        One of: ``"trusted_news"``, ``"factchecker"``,
        ``"scientific_official"``, ``"reference"``, ``"social_platform"``,
        or ``"unknown"``.
    """
    domain = _extract_domain(url_or_domain)

    for known, _tables_label in [
        (_TRUSTED_NEWS, "trusted_news"),
        (_FACTCHECKERS, "factchecker"),
        (_SCIENTIFIC_OFFICIAL, "scientific_official"),
        (_REFERENCE, "reference"),
        (_SOCIAL_PLATFORMS, "social_platform"),
    ]:
        if domain in known:
            return _tables_label
        for kd in known:
            if domain.endswith("." + kd):
                return _tables_label

    return "unknown"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_domain(url_or_domain: str) -> str:
    """Extract a clean domain from a URL or return the domain as-is."""
    # If it looks like a URL, parse it
    if "://" in url_or_domain:
        parsed = urlparse(url_or_domain)
        domain = parsed.hostname or url_or_domain
    else:
        domain = url_or_domain

    # Strip www. prefix for consistent matching
    domain = domain.removeprefix("www.").lower().strip()
    return domain
