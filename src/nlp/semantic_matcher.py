"""
Semantic similarity matching for evidence retrieval.

Uses sentence-transformers (``all-MiniLM-L6-v2``) to compute cosine
similarity between a claim and candidate evidence articles.  Only
articles above a configurable similarity threshold are returned.

Usage:
    from src.nlp.semantic_matcher import find_semantic_matches
    matches = find_semantic_matches(claim, evidence_articles)
"""

import logging
from typing import Any

import numpy as np

from src import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy-loaded model singleton
# ---------------------------------------------------------------------------
_model = None


def _get_model():
    """Lazy-load the sentence-transformer model on first use."""
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading sentence-transformer model '%s'…", config.SEMANTIC_MODEL_NAME)
            _model = SentenceTransformer(config.SEMANTIC_MODEL_NAME)
            logger.info("Model loaded successfully.")
        except ImportError:
            logger.error(
                "sentence-transformers is not installed. "
                "Install with: pip install sentence-transformers"
            )
            raise
    return _model


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

SemanticMatch = dict[str, Any]
"""A matched evidence article with keys: source, url, similarity, text."""


def generate_embedding(text: str) -> np.ndarray:
    """Generate a dense vector embedding for *text*.

    Args:
        text: Input text (claim or article).

    Returns:
        A 1-D numpy array (embedding vector).
    """
    model = _get_model()
    # Truncate very long texts to keep inference fast
    truncated = text[:config.SEMANTIC_MAX_TEXT_LENGTH]
    embedding = model.encode(truncated, convert_to_numpy=True, show_progress_bar=False)
    return embedding


def find_semantic_matches(
    claim: str,
    evidence_articles: list[dict[str, Any]],
    threshold: float | None = None,
) -> list[SemanticMatch]:
    """Find evidence articles that are semantically similar to *claim*.

    Each article dict is expected to have at least one text field among:
    ``"text"``, ``"snippet"``, ``"summary"``, ``"title"``.

    Args:
        claim: The normalised claim text.
        evidence_articles: List of evidence article dicts from collectors.
        threshold: Minimum cosine similarity to include (default from config).

    Returns:
        List of :data:`SemanticMatch` dicts sorted by similarity descending.
    """
    if threshold is None:
        threshold = config.SEMANTIC_SIMILARITY_THRESHOLD

    if not evidence_articles:
        return []

    # Build text representations for each article
    article_texts: list[str] = []
    for article in evidence_articles:
        parts = [
            str(article.get("title", "")),
            str(article.get("text", "")),
            str(article.get("snippet", "")),
            str(article.get("summary", "")),
        ]
        combined = " ".join(p for p in parts if p).strip()
        article_texts.append(combined if combined else "")

    # Filter out empty texts
    valid_indices = [i for i, t in enumerate(article_texts) if t]
    if not valid_indices:
        return []

    valid_texts = [article_texts[i] for i in valid_indices]

    # Generate embeddings
    claim_embedding = generate_embedding(claim)

    model = _get_model()
    truncated_texts = [t[:config.SEMANTIC_MAX_TEXT_LENGTH] for t in valid_texts]
    article_embeddings = model.encode(
        truncated_texts, convert_to_numpy=True, show_progress_bar=False
    )

    # Compute cosine similarities
    similarities = _cosine_similarity_batch(claim_embedding, article_embeddings)

    # Build results above threshold
    matches: list[SemanticMatch] = []
    for idx, sim in zip(valid_indices, similarities):
        if sim >= threshold:
            article = evidence_articles[idx]
            matches.append({
                "source": article.get("source", "unknown"),
                "url": article.get("url", ""),
                "similarity": round(float(sim), 4),
                "text": article_texts[idx][:500],
                "title": article.get("title", ""),
            })

    # Sort by similarity descending
    matches.sort(key=lambda m: m["similarity"], reverse=True)
    return matches


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _cosine_similarity_batch(
    query: np.ndarray, candidates: np.ndarray
) -> np.ndarray:
    """Compute cosine similarity between a single query and multiple candidates."""
    # Normalise vectors
    query_norm = query / (np.linalg.norm(query) + 1e-10)
    candidate_norms = candidates / (
        np.linalg.norm(candidates, axis=1, keepdims=True) + 1e-10
    )
    similarities = candidate_norms @ query_norm
    return similarities
