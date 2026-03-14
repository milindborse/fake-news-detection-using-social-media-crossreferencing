"""
Claim normalizer: basic text cleaning, keyword extraction, and entity detection.

Keeps dependencies lightweight; uses only stdlib + optional spaCy/regex.
"""

import re
import string
from typing import Optional

# ---------------------------------------------------------------------------
# Stopwords (lightweight, no NLTK required)
# ---------------------------------------------------------------------------
_STOPWORDS: set[str] = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "need", "dare",
    "that", "this", "these", "those", "it", "its", "they", "their",
    "there", "here", "about", "which", "what", "who", "when", "where",
    "how", "why", "not", "no", "nor", "so", "yet", "both", "either",
    "neither", "as", "if", "than", "then", "because", "while", "although",
    "though", "since", "until", "unless", "however", "therefore", "thus",
    "also", "just", "very", "too", "more", "most", "much", "many", "some",
    "any", "all", "each", "every", "other", "such", "same", "new", "old",
    "said", "says", "say", "according", "us", "we", "our", "i", "my",
    "he", "she", "his", "her", "him", "you", "your", "up", "down",
    "out", "over", "into", "through", "after", "before", "between",
    "during", "without", "within", "against", "along", "among", "around",
    "per", "re", "etc", "via", "vs", "ie", "eg",
}

# Common English bigrams that add little signal
_SKIP_BIGRAMS: set[str] = {"united states", "new york", "last year", "last week"}


def normalize_text(text: str) -> str:
    """Return a lowercased, whitespace-normalised version of *text*.

    Args:
        text: Raw claim or article text.

    Returns:
        Cleaned string suitable for keyword extraction.
    """
    text = text.lower()
    # Replace URLs with a placeholder
    text = re.sub(r"https?://\S+", " ", text)
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Keep only alphanumeric and basic punctuation
    text = re.sub(r"[^a-z0-9\s'\-]", " ", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_keywords(text: str, max_keywords: int = 10) -> list[str]:
    """Extract the most informative keywords from *text*.

    Uses a simple TF-proxy approach (frequency * length bonus) without any
    external NLP library.

    Args:
        text: Raw claim text.
        max_keywords: Maximum number of keywords to return.

    Returns:
        Ordered list of keyword strings (most relevant first).
    """
    normalized = normalize_text(text)
    tokens = normalized.split()

    # Remove stopwords and very short tokens
    candidates = [
        t for t in tokens
        if t not in _STOPWORDS and len(t) > 2 and not t.isdigit()
    ]

    # Frequency count
    freq: dict[str, int] = {}
    for token in candidates:
        freq[token] = freq.get(token, 0) + 1

    # Score = freq * sqrt(len) to prefer meaningful terms
    scored = sorted(freq.items(), key=lambda x: x[1] * (len(x[0]) ** 0.5), reverse=True)
    keywords = [word for word, _ in scored[:max_keywords]]

    return keywords


def extract_entities(text: str) -> dict[str, list[str]]:
    """Extract named-entity-like tokens using regex heuristics.

    Attempts spaCy if installed, otherwise falls back to capitalisation
    patterns (good enough for headline-style claims).

    Args:
        text: Raw claim text.

    Returns:
        Dictionary with keys ``"persons"``, ``"organizations"``,
        ``"locations"``, ``"dates"``, and ``"misc"``.
    """
    entities: dict[str, list[str]] = {
        "persons": [],
        "organizations": [],
        "locations": [],
        "dates": [],
        "misc": [],
    }

    # Try spaCy first (optional heavy dependency)
    try:
        import spacy  # type: ignore

        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            nlp = None

        if nlp is not None:
            doc = nlp(text[:10_000])  # cap text length to avoid timeouts
            label_map = {
                "PERSON": "persons",
                "ORG": "organizations",
                "GPE": "locations",
                "LOC": "locations",
                "DATE": "dates",
                "TIME": "dates",
            }
            for ent in doc.ents:
                key = label_map.get(ent.label_, "misc")
                val = ent.text.strip()
                if val and val not in entities[key]:
                    entities[key].append(val)
            return entities
    except ImportError:
        pass

    # Regex fallback: extract capitalised phrases (Title Case words not at
    # sentence start) as generic "misc" entities.
    # Date patterns
    date_pattern = re.compile(
        r"\b(?:January|February|March|April|May|June|July|August|"
        r"September|October|November|December)\s+\d{1,2},?\s*\d{4}\b"
        r"|\b\d{4}\b",
        re.IGNORECASE,
    )
    for match in date_pattern.finditer(text):
        val = match.group().strip()
        if val not in entities["dates"]:
            entities["dates"].append(val)

    # Capitalised-phrase extraction (simple NER proxy)
    cap_pattern = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b")
    seen: set[str] = set()
    sentences = re.split(r"[.!?]", text)
    for sentence in sentences:
        stripped = sentence.strip()
        first_word = stripped.split()[0] if stripped else ""
        for match in cap_pattern.finditer(stripped):
            phrase = match.group().strip()
            # Skip first word of sentence (likely capitalised due to grammar)
            if phrase == first_word:
                continue
            if phrase in seen or phrase.lower() in _STOPWORDS:
                continue
            seen.add(phrase)
            entities["misc"].append(phrase)

    return entities


def build_search_query(claim: str, max_keywords: int = 6) -> str:
    """Build a compact search query string from *claim* text.

    Args:
        claim: The input claim/article text.
        max_keywords: How many keywords to include.

    Returns:
        A space-separated keyword query string.
    """
    keywords = extract_keywords(claim, max_keywords=max_keywords)
    return " ".join(keywords)


def get_claim_summary(claim: str, max_chars: int = 300) -> str:
    """Return a truncated version of *claim* for display purposes.

    Args:
        claim: Original claim text.
        max_chars: Character limit for the summary.

    Returns:
        Truncated claim string ending with ``…`` if trimmed.
    """
    claim = claim.strip()
    if len(claim) <= max_chars:
        return claim
    return claim[:max_chars].rsplit(" ", 1)[0] + "…"
