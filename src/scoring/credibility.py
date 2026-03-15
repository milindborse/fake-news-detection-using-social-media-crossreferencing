"""
Credibility scoring engine: computes an explainable 0–100 credibility score
from collected evidence.

Signals (weighted):
    1. Corroboration  – independent sources/domains confirm the claim.
    2. Contradiction  – evidence contains debunking/hoax keywords.
    3. Source quality – domain reputation scoring (granular 0–1 weights).
    4. Recency        – how recent the evidence is (newer = more relevant).
    5. Fact-check     – professional fact-check verdicts (when available).
"""

import math
import re
from datetime import datetime, timezone
from typing import Any

from src import config
from src.scoring.source_reputation import get_domain_score

# ---------------------------------------------------------------------------
# Type definitions
# ---------------------------------------------------------------------------
Signal = dict[str, Any]
"""A single scoring signal with keys: name, value, weight, contribution, rationale."""

ScoreResult = dict[str, Any]
"""Return type of :func:`compute_score`."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _contains_contradiction(text: str) -> bool:
    """Return True if *text* contains any debunking/hoax keywords."""
    lower = text.lower()
    return any(kw in lower for kw in config.CONTRADICTION_KEYWORDS)


def _is_reliable_domain(domain: str) -> bool:
    """Return True if *domain* is in the reliable domains whitelist."""
    domain = domain.removeprefix("www.").lower()
    return any(domain == d or domain.endswith("." + d) for d in config.RELIABLE_DOMAINS)


def _parse_date(date_str: str) -> datetime | None:
    """Try to parse various date formats into a UTC datetime."""
    if not date_str:
        return None
    formats = [
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y%m%dT%H%M%SZ",   # GDELT
        "%a, %d %b %Y %H:%M:%S %z",  # RSS
        "%a, %d %b %Y %H:%M:%S GMT",
        "%Y-%m-%d",
    ]
    for fmt in formats:
        try:
            dt = datetime.strptime(date_str[:25], fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            continue
    return None


def _recency_score(date_str: str) -> float:
    """Return a 0–1 recency score (1 = very recent, 0 = very old / unknown)."""
    dt = _parse_date(date_str)
    if dt is None:
        return 0.3  # unknown date → neutral
    now = datetime.now(tz=timezone.utc)
    age_days = max((now - dt).total_seconds() / 86400, 0)
    # Decay: score = e^(-age/30) capped at 1
    return round(min(1.0, math.exp(-age_days / 30)), 4)


# ---------------------------------------------------------------------------
# Core scoring function
# ---------------------------------------------------------------------------

def compute_score(
    claim: str,
    reddit_records: list[dict[str, Any]],
    wiki_records: list[dict[str, Any]],
    web_records: list[dict[str, Any]],
    hn_records: list[dict[str, Any]] | None = None,
    factcheck_results: list[dict[str, Any]] | None = None,
    semantic_matches: list[dict[str, Any]] | None = None,
) -> ScoreResult:
    """Compute a credibility score and generate signal breakdown.

    Args:
        claim: The normalised claim text.
        reddit_records: Records from Reddit collector.
        wiki_records: Records from Wikipedia collector.
        web_records: Records from web collector.
        hn_records: Records from Hacker News collector (optional).
        factcheck_results: Results from the Google Fact Check API (optional).
        semantic_matches: Semantically matched evidence articles (optional).
            When provided, only evidence that appears in this list is
            counted as strong corroboration.

    Returns:
        A :data:`ScoreResult` dict with keys:
        ``score``, ``label``, ``signal_breakdown``, ``summary``.
    """
    if hn_records is None:
        hn_records = []
    if factcheck_results is None:
        factcheck_results = []
    if semantic_matches is None:
        semantic_matches = []

    all_records = reddit_records + wiki_records + web_records + hn_records
    total_evidence = len(all_records)

    # Build a set of URLs that are semantically verified
    _semantic_urls: set[str] = {m.get("url", "") for m in semantic_matches if m.get("url")}
    has_semantic_data = len(semantic_matches) > 0

    signals: list[Signal] = []

    # ------------------------------------------------------------------
    # Signal 1: Independent corroboration
    # ------------------------------------------------------------------
    # Count distinct domains/sources that mention the claim.
    # When semantic matching data is available, only semantically
    # verified evidence counts at full weight; keyword-only matches
    # are heavily discounted.
    semantic_domains: set[str] = set()
    keyword_only_domains: set[str] = set()

    for r in web_records:
        d = r.get("domain", "")
        if not d:
            continue
        d = d.lstrip("www.")
        url = r.get("url", "")
        if has_semantic_data and url and url in _semantic_urls:
            semantic_domains.add(d)
        else:
            keyword_only_domains.add(d)

    for r in reddit_records:
        url = r.get("url", "")
        if has_semantic_data and url and url in _semantic_urls:
            semantic_domains.add("reddit.com")
        else:
            keyword_only_domains.add("reddit.com")

    for r in hn_records:
        url = r.get("url", "")
        if has_semantic_data and url and url in _semantic_urls:
            semantic_domains.add("news.ycombinator.com")
        else:
            keyword_only_domains.add("news.ycombinator.com")

    # Wikipedia only gets counted at 0.3x weight toward corroboration
    if wiki_records:
        keyword_only_domains.add("wikipedia.org")

    # Semantic matches count fully; keyword-only matches count at 0.3x
    effective_domains = len(semantic_domains) + len(keyword_only_domains) * 0.3
    corroboration_raw = len(semantic_domains) + len(keyword_only_domains)

    # Normalise: 0 domains → 0, 1 → 0.2, 3 → 0.6, 5+ → 1.0
    corroboration_value = min(1.0, effective_domains / 5.0)

    signals.append(
        {
            "name": "Independent Corroboration",
            "value": round(corroboration_value, 3),
            "weight": config.WEIGHT_CORROBORATION,
            "contribution": round(corroboration_value * config.WEIGHT_CORROBORATION * 100, 1),
            "rationale": (
                f"Found mentions across {corroboration_raw} source(s)/domain(s) "
                f"({len(semantic_domains)} semantically verified). "
                "Only semantically verified sources count at full weight."
            ),
        }
    )

    # ------------------------------------------------------------------
    # Signal 2: Contradiction / debunking indicators
    # ------------------------------------------------------------------
    contradiction_hits = 0
    for r in all_records:
        text = " ".join(
            str(r.get(k, "")) for k in ("title", "text", "snippet", "summary")
        )
        if _contains_contradiction(text):
            contradiction_hits += 1

    # Proportion of records containing contradiction keywords
    if total_evidence > 0:
        contradiction_rate = contradiction_hits / total_evidence
    else:
        contradiction_rate = 0.0

    # Higher contradiction rate → lower credibility
    # We invert: credibility contribution = 1 - contradiction_rate
    contradiction_value = round(1.0 - min(1.0, contradiction_rate * 2), 3)

    signals.append(
        {
            "name": "Contradiction / Debunking Indicators",
            "value": contradiction_value,
            "weight": config.WEIGHT_CONTRADICTION,
            "contribution": round(contradiction_value * config.WEIGHT_CONTRADICTION * 100, 1),
            "rationale": (
                f"{contradiction_hits} of {total_evidence} evidence item(s) contain "
                "debunking/hoax keywords. More debunking signals lower the score."
            ),
        }
    )

    # ------------------------------------------------------------------
    # Signal 3: Source quality (domain reputation scoring)
    # ------------------------------------------------------------------
    # Use granular domain reputation scores instead of binary whitelist
    quality_scores: list[float] = []

    for r in web_records:
        domain = r.get("domain", "")
        url = r.get("url", domain)
        quality_scores.append(get_domain_score(url if url else domain))

    # Reddit quality heuristic combined with domain reputation
    reddit_base_score = get_domain_score("reddit.com")
    for r in reddit_records:
        is_quality_sub = r.get("is_quality_subreddit", False)
        has_min_score = r.get("score", 0) >= config.REDDIT_MIN_SCORE
        if is_quality_sub and has_min_score:
            quality_scores.append(min(1.0, reddit_base_score + 0.2))
        else:
            quality_scores.append(reddit_base_score)

    # Hacker News quality heuristic combined with domain reputation
    hn_base_score = get_domain_score("news.ycombinator.com")
    for r in hn_records:
        if r.get("is_quality", False):
            quality_scores.append(min(1.0, hn_base_score + 0.15))
        else:
            quality_scores.append(hn_base_score)

    # Wikipedia – discounted quality contribution
    wiki_score = get_domain_score("wikipedia.org")
    wiki_weight = config.WIKIPEDIA_QUALITY_WEIGHT
    for _r in wiki_records:
        quality_scores.append(wiki_score * wiki_weight)

    source_quality_value = (
        (sum(quality_scores) / len(quality_scores)) if quality_scores else 0.3
    )
    quality_hits = sum(1 for s in quality_scores if s >= 0.7)
    total_checked = len(quality_scores)

    signals.append(
        {
            "name": "Source Quality",
            "value": round(source_quality_value, 3),
            "weight": config.WEIGHT_SOURCE_QUALITY,
            "contribution": round(source_quality_value * config.WEIGHT_SOURCE_QUALITY * 100, 1),
            "rationale": (
                f"{quality_hits} of {total_checked} evidence item(s) come from "
                "reliable or high-quality sources."
            ),
        }
    )

    # ------------------------------------------------------------------
    # Signal 4: Recency relevance
    # ------------------------------------------------------------------
    recency_scores: list[float] = []
    for r in all_records:
        date_str = r.get("created_utc") or r.get("published_date") or ""
        recency_scores.append(_recency_score(date_str))

    recency_value = (sum(recency_scores) / len(recency_scores)) if recency_scores else 0.3

    signals.append(
        {
            "name": "Recency Relevance",
            "value": round(recency_value, 3),
            "weight": config.WEIGHT_RECENCY,
            "contribution": round(recency_value * config.WEIGHT_RECENCY * 100, 1),
            "rationale": (
                f"Average recency score of {round(recency_value, 2)} across "
                f"{len(recency_scores)} evidence item(s). "
                "More recent evidence is weighted higher."
            ),
        }
    )

    # ------------------------------------------------------------------
    # Signal 5: Fact-check verification (when available)
    # ------------------------------------------------------------------
    factcheck_rating = None
    factcheck_confidence = 0.0
    factcheck_sources: list[str] = []

    if factcheck_results:
        # Average the confidence scores from all fact-check results
        confidences = [r.get("confidence", 0.5) for r in factcheck_results]
        factcheck_confidence = sum(confidences) / len(confidences)
        factcheck_sources = [
            f"{r.get('publisher', 'Unknown')}" for r in factcheck_results
        ]
        # Use the first (most relevant) rating as the headline rating
        factcheck_rating = factcheck_results[0].get("rating", "Unknown")

        signals.append(
            {
                "name": "Fact-Check Verification",
                "value": round(factcheck_confidence, 3),
                "weight": config.WEIGHT_FACTCHECK,
                "contribution": round(
                    factcheck_confidence * config.WEIGHT_FACTCHECK * 100, 1
                ),
                "rationale": (
                    f"Professional fact-check(s) found from "
                    f"{', '.join(factcheck_sources)}. "
                    f"Rating: '{factcheck_rating}' "
                    f"(confidence: {factcheck_confidence:.2f})."
                ),
            }
        )

    # ------------------------------------------------------------------
    # Aggregate score
    # ------------------------------------------------------------------
    raw_score = sum(s["contribution"] for s in signals)

    # When fact-check results exist, normalise so signals still fit 0-100
    if factcheck_results:
        total_weight = (
            config.WEIGHT_CORROBORATION
            + config.WEIGHT_CONTRADICTION
            + config.WEIGHT_SOURCE_QUALITY
            + config.WEIGHT_RECENCY
            + config.WEIGHT_FACTCHECK
        )
        raw_score = raw_score / total_weight
    score = max(0, min(100, round(raw_score)))

    # Penalise if zero evidence at all
    if total_evidence == 0:
        score = max(0, score - 20)

    # Determine label
    if score >= config.LABEL_TRUE_THRESHOLD:
        label = "Likely True"
        label_color = "green"
    elif score <= config.LABEL_FALSE_THRESHOLD:
        label = "Likely False"
        label_color = "red"
    else:
        label = "Unclear"
        label_color = "orange"

    # Positive / negative signals for UI
    positive_signals = [s for s in signals if s["contribution"] >= 10]
    negative_signals = [s for s in signals if s["contribution"] < 10]

    # Summary text
    summary_parts: list[str] = []
    if total_evidence == 0:
        summary_parts.append("No evidence could be retrieved for this claim.")
    else:
        summary_parts.append(
            f"Analysis based on {total_evidence} evidence item(s) from "
            f"{corroboration_raw} source(s)."
        )
    if contradiction_hits > 0:
        summary_parts.append(
            f"{contradiction_hits} item(s) contained debunking/hoax language."
        )
    if quality_hits > 0:
        summary_parts.append(
            f"{quality_hits} item(s) came from reliable sources."
        )
    if factcheck_results:
        summary_parts.append(
            f"Professional fact-check from {', '.join(factcheck_sources)}: "
            f"'{factcheck_rating}'."
        )

    # Classify sources as supporting or contradicting
    supporting_sources: list[str] = []
    contradicting_sources: list[str] = []
    for r in all_records:
        text = " ".join(
            str(r.get(k, "")) for k in ("title", "text", "snippet", "summary")
        )
        source_label = r.get("domain", r.get("source", "unknown"))
        if _contains_contradiction(text):
            contradicting_sources.append(source_label)
        else:
            supporting_sources.append(source_label)

    # Build explanation string
    explanation_parts: list[str] = []
    if contradicting_sources:
        explanation_parts.append(
            f"Claim contradicted by {', '.join(set(contradicting_sources[:3]))}"
        )
    if factcheck_results and factcheck_confidence < 0.4:
        if explanation_parts:
            explanation_parts.append(f"and {factcheck_sources[0]} fact-check")
        else:
            explanation_parts.append(
                f"Claim rated low-credibility by {factcheck_sources[0]} fact-check"
            )
    if not explanation_parts and supporting_sources:
        explanation_parts.append(
            f"Claim supported by {', '.join(set(supporting_sources[:3]))}"
        )
    explanation = " ".join(explanation_parts) if explanation_parts else "Insufficient evidence."

    return {
        "score": score,
        "label": label,
        "label_color": label_color,
        "signal_breakdown": signals,
        "positive_signals": positive_signals,
        "negative_signals": negative_signals,
        "summary": " ".join(summary_parts),
        "total_evidence": total_evidence,
        "fact_check_rating": factcheck_rating,
        "supporting_sources": list(set(supporting_sources)),
        "contradicting_sources": list(set(contradicting_sources)),
        "explanation": explanation,
        "source_counts": {
            "reddit": len(reddit_records),
            "wikipedia": len(wiki_records),
            "web": len(web_records),
            "hackernews": len(hn_records),
            "factcheck": len(factcheck_results),
        },
        "analyzed_at": datetime.now(tz=timezone.utc).isoformat(),
    }
