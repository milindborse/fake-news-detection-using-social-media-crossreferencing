"""
Credibility scoring engine: computes an explainable 0–100 credibility score
from collected evidence.

Signals (weighted):
    1. Corroboration  – independent sources/domains confirm the claim.
    2. Contradiction  – evidence contains debunking/hoax keywords.
    3. Source quality – reliable domain whitelist + Reddit quality heuristics.
    4. Recency        – how recent the evidence is (newer = more relevant).
"""

import math
import re
from datetime import datetime, timezone
from typing import Any

from src import config

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
) -> ScoreResult:
    """Compute a credibility score and generate signal breakdown.

    Args:
        claim: The normalised claim text.
        reddit_records: Records from Reddit collector.
        wiki_records: Records from Wikipedia collector.
        web_records: Records from web collector.

    Returns:
        A :data:`ScoreResult` dict with keys:
        ``score``, ``label``, ``signal_breakdown``, ``summary``.
    """
    all_records = reddit_records + wiki_records + web_records
    total_evidence = len(all_records)

    signals: list[Signal] = []

    # ------------------------------------------------------------------
    # Signal 1: Independent corroboration
    # ------------------------------------------------------------------
    # Count distinct domains/sources that mention the claim
    domains: set[str] = set()
    for r in web_records:
        d = r.get("domain", "")
        if d:
            domains.add(d.lstrip("www."))
    if reddit_records:
        domains.add("reddit.com")
    if wiki_records:
        domains.add("wikipedia.org")

    corroboration_raw = len(domains)
    # Normalise: 0 domains → 0, 1 → 0.2, 3 → 0.6, 5+ → 1.0
    corroboration_value = min(1.0, corroboration_raw / 5.0)

    signals.append(
        {
            "name": "Independent Corroboration",
            "value": round(corroboration_value, 3),
            "weight": config.WEIGHT_CORROBORATION,
            "contribution": round(corroboration_value * config.WEIGHT_CORROBORATION * 100, 1),
            "rationale": (
                f"Found mentions across {corroboration_raw} distinct source(s)/domain(s). "
                "More independent sources increase credibility."
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
    # Signal 3: Source quality
    # ------------------------------------------------------------------
    quality_hits = 0
    total_checked = 0

    for r in web_records:
        total_checked += 1
        if _is_reliable_domain(r.get("domain", "")):
            quality_hits += 1

    # Reddit quality heuristic
    quality_reddit_hits = 0
    for r in reddit_records:
        total_checked += 1
        is_quality_sub = r.get("is_quality_subreddit", False)
        has_min_score = r.get("score", 0) >= config.REDDIT_MIN_SCORE
        if is_quality_sub and has_min_score:
            quality_reddit_hits += 1
            quality_hits += 1

    # Wikipedia is always considered a quality source
    if wiki_records:
        quality_hits += len(wiki_records)
        total_checked += len(wiki_records)

    source_quality_value = (quality_hits / max(total_checked, 1)) if total_checked else 0.3

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
    # Aggregate score
    # ------------------------------------------------------------------
    raw_score = sum(s["contribution"] for s in signals)
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

    return {
        "score": score,
        "label": label,
        "label_color": label_color,
        "signal_breakdown": signals,
        "positive_signals": positive_signals,
        "negative_signals": negative_signals,
        "summary": " ".join(summary_parts),
        "total_evidence": total_evidence,
        "source_counts": {
            "reddit": len(reddit_records),
            "wikipedia": len(wiki_records),
            "web": len(web_records),
        },
        "analyzed_at": datetime.now(tz=timezone.utc).isoformat(),
    }
